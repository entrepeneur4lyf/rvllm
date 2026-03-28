//! GPU forward pass orchestrator (Agent 13).
//!
//! `GpuModelRunner` drives the full Llama-family forward pass on CUDA:
//! token embedding lookup -> N transformer layers -> final RMSNorm -> LM head -> logits.
//!
//! All CUDA code is gated behind `#[cfg(feature = "cuda")]`. Under `mock-gpu`
//! (the default), this module provides a compile-compatible stub that returns
//! an error at runtime so existing Mac-side tests keep working.

// =========================================================================
//  CUDA implementation
// =========================================================================
#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::sync::Arc;

    use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, LaunchAsync};
    use tracing::{debug, info, trace};

    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;

    use rvllm_gpu::kernel_loader::KernelLoader;
    use rvllm_model_loader::gpu_weights::GpuModelWeights;
    use rvllm_kv_cache::engine_cuda::CudaCacheEngine;
    use crate::layers::linear_cuda::CudaLinearLayer;
    use crate::layers::norm_cuda::CudaRMSNorm;
    use crate::gpu_layer::{GpuLayerConfig, GpuLayerInput, GpuLayerWeights, GpuTransformerLayer};
    use rvllm_gpu::prelude::CublasHandle;

    pub struct GpuModelRunner {
        weights: GpuModelWeights,
        cache: CudaCacheEngine,
        blas: CublasHandle,
        loader: KernelLoader,
        config: ModelRunnerConfig,
        device: Arc<CudaDevice>,
        stream: CudaStream,
        layers: Vec<GpuTransformerLayer>,
        embed_tokens: CudaSlice<f32>,
        final_norm_weight: CudaSlice<f32>,
        lm_head_weight: CudaSlice<f32>,
        rms_norm_eps: f32,
        /// Precomputed RoPE cos table on GPU: [max_position, head_dim/2]
        rope_cos: CudaSlice<f32>,
        /// Precomputed RoPE sin table on GPU: [max_position, head_dim/2]
        rope_sin: CudaSlice<f32>,
    }

    impl GpuModelRunner {
        pub fn new(
            weights: GpuModelWeights,
            cache: CudaCacheEngine,
            blas: CublasHandle,
            loader: KernelLoader,
            config: ModelRunnerConfig,
            device: Arc<CudaDevice>,
        ) -> Result<Self> {
            debug!(
                num_layers = config.num_layers,
                hidden = config.hidden_size,
                vocab = config.vocab_size,
                "GpuModelRunner::new"
            );

            // cudarc 0.12: streams are created via fork_default_stream()
            let stream = device.fork_default_stream()
                .map_err(|e| LLMError::GpuError(format!("stream creation failed: {e}")))?;

            let embed_tokens = weights
                .get("model.embed_tokens.weight")
                .ok_or_else(|| LLMError::GpuError("missing model.embed_tokens.weight".into()))?
                .clone();

            let final_norm_weight = weights
                .get("model.norm.weight")
                .ok_or_else(|| LLMError::GpuError("missing model.norm.weight".into()))?
                .clone();

            let lm_head_weight = weights
                .get("lm_head.weight")
                .or_else(|| weights.get("model.embed_tokens.weight"))
                .ok_or_else(|| LLMError::GpuError("missing lm_head.weight and model.embed_tokens.weight".into()))?
                .clone();

            let mut layers = Vec::with_capacity(config.num_layers);
            for i in 0..config.num_layers {
                let layer_cfg = GpuLayerConfig {
                    hidden_size: config.hidden_size,
                    num_heads: config.num_heads,
                    num_kv_heads: config.num_kv_heads,
                    head_dim: config.head_dim,
                    intermediate_size: config.intermediate_size,
                    rms_norm_eps: 1e-5_f32,
                    layer_idx: i,
                };
                layers.push(GpuTransformerLayer::new(layer_cfg, Arc::clone(&device)));
            }

            // Precompute RoPE cos/sin tables
            let head_dim = config.head_dim;
            let max_pos = config.max_position.min(8192);
            let half_dim = head_dim / 2;
            let rope_theta = config.rope_theta;
            let mut cos_table = vec![0.0f32; max_pos * half_dim];
            let mut sin_table = vec![0.0f32; max_pos * half_dim];
            for pos in 0..max_pos {
                for i in 0..half_dim {
                    let freq = 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32);
                    let theta = pos as f32 * freq;
                    cos_table[pos * half_dim + i] = theta.cos();
                    sin_table[pos * half_dim + i] = theta.sin();
                }
            }
            let rope_cos = device.htod_sync_copy(&cos_table)
                .map_err(|e| LLMError::GpuError(format!("rope cos HtoD: {e}")))?;
            let rope_sin = device.htod_sync_copy(&sin_table)
                .map_err(|e| LLMError::GpuError(format!("rope sin HtoD: {e}")))?;
            info!(max_pos, half_dim, "RoPE tables uploaded to GPU");

            Ok(Self {
                weights,
                cache,
                blas,
                loader,
                config,
                device,
                stream,
                layers,
                embed_tokens,
                final_norm_weight,
                lm_head_weight,
                rms_norm_eps: 1e-5_f32,
                rope_cos,
                rope_sin,
            })
        }

        pub fn forward(
            &self,
            token_ids: &[u32],
            positions: &[u32],
            attn_meta: &crate::bridge::AttentionMetadata,
            is_prefill: bool,
        ) -> Result<Vec<f32>> {
            let num_tokens = token_ids.len();
            let num_seqs = attn_meta.context_lens.len();
            let hidden_size = self.config.hidden_size;
            let vocab_size = self.config.vocab_size;
            let block_size = self.cache.block_size();

            if num_tokens == 0 {
                return Err(LLMError::ModelError("empty input".into()));
            }

            debug!(num_tokens, num_seqs, is_prefill, "GpuModelRunner::forward");

            // Upload positions to GPU as i32 (CUDA kernels expect int*)
            let pos_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
            let positions_gpu: CudaSlice<i32> = self.device
                .htod_sync_copy(&pos_i32)
                .map_err(|e| LLMError::GpuError(format!("positions HtoD: {e}")))?;

            // Upload context_lens as i32
            let cl_i32: Vec<i32> = attn_meta.context_lens.iter().map(|&c| c as i32).collect();
            let context_lens_gpu: CudaSlice<i32> = self.device
                .htod_sync_copy(&cl_i32)
                .map_err(|e| LLMError::GpuError(format!("context_lens HtoD: {e}")))?;

            // Flatten block_tables to [num_seqs, max_blocks_per_seq] row-major as i32
            let max_blocks = attn_meta.block_tables.iter().map(|r| r.len()).max().unwrap_or(1).max(1);
            let mut flat_bt = vec![0i32; num_seqs * max_blocks];
            for (s, row) in attn_meta.block_tables.iter().enumerate() {
                for (b, &blk) in row.iter().enumerate() {
                    flat_bt[s * max_blocks + b] = blk as i32;
                }
            }
            let block_tables_gpu: CudaSlice<i32> = self.device
                .htod_sync_copy(&flat_bt)
                .map_err(|e| LLMError::GpuError(format!("block_tables HtoD: {e}")))?;

            // Upload slot_mapping as i32
            let sm_i32: Vec<i32> = attn_meta.slot_mapping.iter().map(|&s| s as i32).collect();
            let slot_mapping_gpu: CudaSlice<i32> = self.device
                .htod_sync_copy(&sm_i32)
                .map_err(|e| LLMError::GpuError(format!("slot_mapping HtoD: {e}")))?;

            let max_context_len = attn_meta.max_context_len;

            // Metadata dump (first call only)
            static PROBED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            let probe = !PROBED.swap(true, std::sync::atomic::Ordering::Relaxed);

            // Decode-specific probe (first decode call only)
            static DECODE_PROBED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            let decode_probe = !is_prefill && !DECODE_PROBED.swap(true, std::sync::atomic::Ordering::Relaxed);
            // Reset if this was a prefill call (only arm on actual decode)
            if is_prefill { DECODE_PROBED.store(false, std::sync::atomic::Ordering::Relaxed); }
            if probe {
                info!(
                    ?token_ids,
                    ?positions,
                    slot_mapping = ?&attn_meta.slot_mapping[..8.min(attn_meta.slot_mapping.len())],
                    context_lens = ?&attn_meta.context_lens,
                    block_tables = ?&attn_meta.block_tables,
                    max_context_len,
                    max_blocks,
                    is_prefill,
                    "PROBE metadata"
                );
            }
            if decode_probe {
                info!(
                    ?token_ids,
                    ?positions,
                    slot_mapping = ?&attn_meta.slot_mapping,
                    context_lens = ?&attn_meta.context_lens,
                    block_tables = ?&attn_meta.block_tables,
                    max_context_len,
                    max_blocks,
                    is_prefill,
                    "DECODE_PROBE metadata"
                );
            }

            // Step 1: token embedding lookup
            info!("gpu_runner: embedding lookup");
            let mut hidden_states = self.embedding_lookup(token_ids)?;
            if probe {
                let h: Vec<f32> = self.device.dtoh_sync_copy(&hidden_states).map_err(|e| LLMError::GpuError(format!("probe DtoH: {e}")))?;
                info!(len = h.len(), first5 = ?&h[..5.min(h.len())], "PROBE embed");
            }
            if decode_probe {
                let h: Vec<f32> = self.device.dtoh_sync_copy(&hidden_states).map_err(|e| LLMError::GpuError(format!("decode probe DtoH: {e}")))?;
                info!(len = h.len(), first5 = ?&h[..5.min(h.len())], "DECODE_PROBE embed");
            }

            // Step 2: transformer layers
            let gpu_cache = self.cache.gpu_cache();
            let num_layers = self.layers.len();
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if layer_idx == 0 || layer_idx == num_layers - 1 {
                    info!(layer = layer_idx, "gpu_runner: layer start");
                }
                let (key_cache, value_cache) = &gpu_cache[layer_idx];
                let weights = self.layer_weights(layer_idx)?;
                let input = GpuLayerInput {
                    hidden_states: &hidden_states,
                    positions: &positions_gpu,
                    key_cache,
                    value_cache,
                    block_tables: &block_tables_gpu,
                    context_lens: &context_lens_gpu,
                    slot_mapping: &slot_mapping_gpu,
                    num_tokens,
                    num_seqs,
                    max_context_len,
                    block_size,
                    is_prefill,
                    rope_cos: &self.rope_cos,
                    rope_sin: &self.rope_sin,
                };
                hidden_states = layer.forward(&input, &weights, &self.blas)?;
                if probe && layer_idx == 0 {
                    let h: Vec<f32> = self.device.dtoh_sync_copy(&hidden_states).map_err(|e| LLMError::GpuError(format!("probe DtoH: {e}")))?;
                    let nan = h.iter().any(|v| v.is_nan());
                    let mx = h.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    info!(nan, max = mx, first5 = ?&h[..5.min(h.len())], "PROBE layer0 output");

                    // Read back cache block 0 for layer 0 to verify reshape_and_cache
                    let (kc, vc) = &gpu_cache[0];
                    let kc_host: Vec<f32> = self.device.dtoh_sync_copy(kc).map_err(|e| LLMError::GpuError(format!("probe DtoH: {e}")))?;
                    let vc_host: Vec<f32> = self.device.dtoh_sync_copy(vc).map_err(|e| LLMError::GpuError(format!("probe DtoH: {e}")))?;
                    // Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
                    // Block 0, offset 0, kv_head 0: first head_dim elements
                    let head_dim = self.config.head_dim;
                    let kv_heads = self.config.num_kv_heads;
                    // slot 0 -> offset 0 -> kc_host[0 .. kv_heads*head_dim]
                    info!(
                        k_cache_first5 = ?&kc_host[..5.min(kc_host.len())],
                        v_cache_first5 = ?&vc_host[..5.min(vc_host.len())],
                        cache_total_len = kc_host.len(),
                        "PROBE cache block0"
                    );
                }
                if decode_probe && layer_idx == 0 {
                    let h: Vec<f32> = self.device.dtoh_sync_copy(&hidden_states).map_err(|e| LLMError::GpuError(format!("decode probe DtoH: {e}")))?;
                    let nan = h.iter().any(|v| v.is_nan());
                    let mx = h.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    info!(nan, max = mx, first5 = ?&h[..5.min(h.len())], "DECODE_PROBE layer0 output");
                }
                if probe && layer_idx == num_layers - 1 {
                    // Dump last token's hidden state after final layer
                    let h: Vec<f32> = self.device.dtoh_sync_copy(&hidden_states).map_err(|e| LLMError::GpuError(format!("probe DtoH: {e}")))?;
                    let last_start = (num_tokens - 1) * hidden_size;
                    let last_h = &h[last_start..last_start + hidden_size];
                    let mx = last_h.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mn = last_h.iter().cloned().fold(f32::INFINITY, f32::min);
                    info!(layer = layer_idx, last_max = mx, last_min = mn, first5 = ?&last_h[..5.min(last_h.len())], "PROBE last-layer last-token hidden");
                }
                if layer_idx == 0 || layer_idx == num_layers - 1 {
                    info!(layer = layer_idx, "gpu_runner: layer done");
                }
            }
            if decode_probe {
                let h: Vec<f32> = self.device.dtoh_sync_copy(&hidden_states).map_err(|e| LLMError::GpuError(format!("decode probe DtoH: {e}")))?;
                info!(first5 = ?&h[..5.min(h.len())], "DECODE_PROBE final hidden state");
            }

            // Step 3: final RMSNorm
            let normed = CudaRMSNorm::forward(
                &hidden_states,
                &self.final_norm_weight,
                self.rms_norm_eps,
                hidden_size,
                &self.loader,
                &self.stream,
            )?;

            // Sync before LM head: RMSNorm runs on self.stream, cuBLAS on default stream
            self.device.synchronize()
                .map_err(|e| LLMError::GpuError(format!("pre-LM-head sync: {e}")))?;

            // Step 4: LM head  normed [num_tokens, hidden] @ lm_head^T [hidden, vocab]
            let logits_gpu = CudaLinearLayer::forward_once(
                &normed,
                &self.lm_head_weight,
                None,
                num_tokens,
                vocab_size,
                hidden_size,
                &self.blas,
            )?;

            // Step 5: DtoH
            let logits_cpu = self.device
                .dtoh_sync_copy(&logits_gpu)
                .map_err(|e| LLMError::GpuError(format!("logits DtoH: {e}")))?;

            if decode_probe {
                let argmax = logits_cpu.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);
                let mut top5: Vec<(usize, f32)> = logits_cpu.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                top5.truncate(5);
                info!(
                    logits_len = logits_cpu.len(),
                    argmax_token = argmax,
                    top5 = ?top5,
                    "DECODE_PROBE logits"
                );
            }

            if probe {
                // Last token's logits (what sampler uses)
                let last_start = (num_tokens - 1) * vocab_size;
                let last_logits = &logits_cpu[last_start..last_start + vocab_size];
                let local_argmax = last_logits.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                let mut top5: Vec<(usize, f32)> = last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                top5.truncate(5);
                let last_max = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let last_min = last_logits.iter().cloned().fold(f32::INFINITY, f32::min);
                let nan = last_logits.iter().any(|v| v.is_nan());
                info!(num_tokens, vocab_size, local_argmax, nan, last_min, last_max, top5 = ?top5, "PROBE prefill last-token logits");
            }
            debug!(logits_len = logits_cpu.len(), expected = num_tokens * vocab_size, "forward complete");
            Ok(logits_cpu)
        }

        /// Per-layer weight references into the GPU weight map.
        fn layer_weights(&self, i: usize) -> Result<GpuLayerWeights<'_>> {
            let g = |name: &str| -> Result<&CudaSlice<f32>> {
                self.weights.get(name).ok_or_else(|| LLMError::GpuError(format!("missing weight: {name}")))
            };
            Ok(GpuLayerWeights {
                input_layernorm:           g(&format!("model.layers.{i}.input_layernorm.weight"))?,
                q_proj:                    g(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                k_proj:                    g(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                v_proj:                    g(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                o_proj:                    g(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                post_attention_layernorm:  g(&format!("model.layers.{i}.post_attention_layernorm.weight"))?,
                gate_proj:                 g(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                up_proj:                   g(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                down_proj:                 g(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
            })
        }

        fn embedding_lookup(&self, token_ids: &[u32]) -> Result<CudaSlice<f32>> {
            let num_tokens = token_ids.len();
            let hidden_size = self.config.hidden_size;

            let kernel = self.device
                .get_func("embedding_gather", "embedding_gather_kernel")
                .ok_or_else(|| LLMError::GpuError("embedding_gather_kernel not loaded".into()))?;

            let output = self.device
                .alloc_zeros::<f32>(num_tokens * hidden_size)
                .map_err(|e| LLMError::GpuError(format!("embed alloc: {e}")))?;

            let ids_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
            let ids_gpu = self.device
                .htod_sync_copy(&ids_i32)
                .map_err(|e| LLMError::GpuError(format!("token_ids HtoD: {e}")))?;

            let block_dim = hidden_size.min(1024) as u32;
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                kernel.launch(cfg, (
                    &output,
                    &self.embed_tokens,
                    &ids_gpu,
                    hidden_size as i32,
                    self.config.vocab_size as i32,
                )).map_err(|e| LLMError::GpuError(format!("embedding_gather launch: {e}")))?;
            }

            Ok(output)
        }

        pub fn config(&self) -> &ModelRunnerConfig {
            &self.config
        }

        pub fn cache(&self) -> &CudaCacheEngine {
            &self.cache
        }

        pub fn cache_mut(&mut self) -> &mut CudaCacheEngine {
            &mut self.cache
        }
    }
}

// Re-export under cuda feature gate.
#[cfg(feature = "cuda")]
pub use cuda_impl::GpuModelRunner;

// =========================================================================
//  Mock-GPU stub (default feature)
// =========================================================================
#[cfg(not(feature = "cuda"))]
mod mock_impl {
    use crate::bridge::{LLMError, Result};
    use crate::runner::ModelRunnerConfig;

    /// Stub GpuModelRunner for non-CUDA builds.
    ///
    /// Allows downstream code to reference the type without conditional
    /// compilation everywhere. All methods return an error at runtime.
    pub struct GpuModelRunner {
        config: ModelRunnerConfig,
    }

    impl GpuModelRunner {
        /// Returns an error -- real CUDA is required.
        pub fn forward(
            &self,
            _token_ids: &[u32],
            _positions: &[u32],
            _attn_meta: &crate::bridge::AttentionMetadata,
            _is_prefill: bool,
        ) -> Result<Vec<f32>> {
            Err(LLMError::GpuError(
                "GpuModelRunner requires the `cuda` feature".into(),
            ))
        }

        pub fn config(&self) -> &ModelRunnerConfig {
            &self.config
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use mock_impl::GpuModelRunner;

// =========================================================================
//  Tests (run under mock-gpu / default features)
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_runner_returns_error() {
        #[cfg(not(feature = "cuda"))]
        {
            let config = ModelRunnerConfig {
                num_layers: 2,
                hidden_size: 64,
                num_heads: 4,
                num_kv_heads: 4,
                head_dim: 16,
                intermediate_size: 128,
                vocab_size: 100,
                max_position: 512,
                rope_theta: 10000.0,
                dtype: "float32".to_string(),
                architecture: "LlamaForCausalLM".to_string(),
            };
            let runner = GpuModelRunner { config };
            let result = runner.forward(&[1, 2, 3], &[0, 1, 2], &[], &[]);
            assert!(result.is_err());
            let err_msg = format!("{}", result.unwrap_err());
            assert!(err_msg.contains("cuda"));
        }
    }

    #[test]
    fn config_accessible() {
        #[cfg(not(feature = "cuda"))]
        {
            let config = ModelRunnerConfig {
                num_layers: 4,
                hidden_size: 256,
                num_heads: 8,
                num_kv_heads: 8,
                head_dim: 32,
                intermediate_size: 512,
                vocab_size: 32000,
                max_position: 2048,
                rope_theta: 10000.0,
                dtype: "float16".to_string(),
                architecture: "LlamaForCausalLM".to_string(),
            };
            let runner = GpuModelRunner { config };
            assert_eq!(runner.config().num_layers, 4);
            assert_eq!(runner.config().vocab_size, 32000);
        }
    }
}
