//! GPU Transformer Layer -- one complete transformer block on CUDA.
//!
//! Combines all CUDA dispatch ops (Agents 2-7) into the standard
//! decoder-only transformer sequence:
//!
//! 1. RMSNorm(input)
//! 2. QKV projection (cuBLAS sgemm)
//! 3. RoPE on Q, K
//! 4. PagedAttention(Q, K_cache, V_cache)
//! 5. Output projection (cuBLAS sgemm)
//! 6. RMSNorm(residual + attn_out)
//! 7. MLP: gate+up (cuBLAS) -> fused_silu_mul -> down (cuBLAS)
//! 8. residual + mlp_out
//!
//! All code is gated behind `#[cfg(feature = "cuda")]`.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;

    use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtrMut, DeviceSlice, LaunchConfig, PushKernelArg};
    use half::f16;
    use tracing::{info, trace};

    use rvllm_core::error::{LLMError, Result};
    use rvllm_gpu::cublas::CublasHandle;
    use rvllm_gpu::kernel_loader::KernelLoader;

    /// Configuration for a single transformer layer.
    #[derive(Debug, Clone)]
    pub struct GpuLayerConfig {
        pub hidden_size: usize,
        pub num_heads: usize,
        pub num_kv_heads: usize,
        pub head_dim: usize,
        pub intermediate_size: usize,
        pub rms_norm_eps: f32,
        pub layer_idx: usize,
    }

    /// Weight references for a single transformer layer.
    ///
    /// All slices live on GPU and are owned by the GpuModelWeights container.
    /// This struct borrows them for the duration of a forward pass.
    pub struct GpuLayerWeights<'a> {
        // Pre-attention norm
        pub input_layernorm: &'a CudaSlice<f32>,
        // Attention projections
        pub q_proj: &'a CudaSlice<f32>,
        pub k_proj: &'a CudaSlice<f32>,
        pub v_proj: &'a CudaSlice<f32>,
        pub o_proj: &'a CudaSlice<f32>,
        // Optional QKV biases (Qwen2.5 has these)
        pub q_proj_bias: Option<&'a CudaSlice<f32>>,
        pub k_proj_bias: Option<&'a CudaSlice<f32>>,
        pub v_proj_bias: Option<&'a CudaSlice<f32>>,
        // Post-attention norm
        pub post_attention_layernorm: &'a CudaSlice<f32>,
        // MLP weights
        pub gate_proj: &'a CudaSlice<f32>,
        pub up_proj: &'a CudaSlice<f32>,
        pub down_proj: &'a CudaSlice<f32>,
    }

    /// FP16 weight references for a single transformer layer (f16 GEMM path).
    ///
    /// Projection weights are f16 (used with hgemm). Norm weights and biases
    /// remain f32 since RMSNorm and bias-add operate in f32.
    pub struct GpuLayerWeightsF16<'a> {
        pub input_layernorm: &'a CudaSlice<f32>,
        pub q_proj: &'a CudaSlice<f16>,
        pub k_proj: &'a CudaSlice<f16>,
        pub v_proj: &'a CudaSlice<f16>,
        pub o_proj: &'a CudaSlice<f16>,
        pub q_proj_bias: Option<&'a CudaSlice<f32>>,
        pub k_proj_bias: Option<&'a CudaSlice<f32>>,
        pub v_proj_bias: Option<&'a CudaSlice<f32>>,
        pub post_attention_layernorm: &'a CudaSlice<f32>,
        pub gate_proj: &'a CudaSlice<f16>,
        pub up_proj: &'a CudaSlice<f16>,
        pub down_proj: &'a CudaSlice<f16>,
        /// Fused QKV weight: [q_dim + kv_dim + kv_dim, hidden]. None if not fused.
        pub fused_qkv: Option<&'a CudaSlice<f16>>,
        /// Fused gate+up weight: [intermediate*2, hidden]. None if not fused.
        pub fused_gate_up: Option<&'a CudaSlice<f16>>,
        /// Pre-converted f16 norm weights and biases for zero-cast f16 forward.
        pub input_layernorm_f16: Option<&'a CudaSlice<f16>>,
        pub post_attention_layernorm_f16: Option<&'a CudaSlice<f16>>,
        pub fused_qkv_bias: Option<&'a CudaSlice<f16>>,
    }

    /// Metadata needed for a single layer forward pass.
    pub struct GpuLayerInput<'a> {
        /// Hidden states entering this layer, shape [num_tokens, hidden_size].
        pub hidden_states: &'a CudaSlice<f32>,
        /// Optional f16 hidden states for the zero-cast f16 forward path.
        pub hidden_states_f16: Option<&'a CudaSlice<f16>>,
        /// Position ids for RoPE, shape [num_tokens]. Kernels expect int*.
        pub positions: CudaView<'a, i32>,
        /// KV cache key block for this layer (f16), shape [num_blocks, block_size, num_kv_heads, head_dim].
        pub key_cache: &'a CudaSlice<f16>,
        /// KV cache value block for this layer (f16), shape [num_blocks, block_size, num_kv_heads, head_dim].
        pub value_cache: &'a CudaSlice<f16>,
        /// Block table mapping sequence positions to cache blocks, shape [num_seqs, max_blocks_per_seq].
        pub block_tables: CudaView<'a, i32>,
        /// Context length for each sequence, shape [num_seqs].
        pub context_lens: CudaView<'a, i32>,
        /// Slot mapping for cache writes during prefill, shape [num_tokens].
        pub slot_mapping: CudaView<'a, i32>,
        /// Number of tokens in the batch.
        pub num_tokens: usize,
        /// Number of sequences in the batch.
        pub num_seqs: usize,
        /// Maximum context length across sequences.
        pub max_context_len: u32,
        /// Block size for paged attention.
        pub block_size: usize,
        /// True during prefill (prompt processing), false during decode.
        pub is_prefill: bool,
        /// Per-sequence query token start positions: [num_seqs + 1] with sentinel.
        /// Built from actual query token counts, NOT context_lens.
        pub seq_start_pos: CudaView<'a, i32>,
        /// RoPE cos table on GPU: [max_position, head_dim/2].
        pub rope_cos: &'a CudaSlice<f32>,
        /// RoPE sin table on GPU: [max_position, head_dim/2].
        pub rope_sin: &'a CudaSlice<f32>,
    }

    /// One complete GPU transformer layer.
    ///
    /// Holds references to the kernel loader and cuBLAS handle;
    /// weights are passed in per-call via `GpuLayerWeights`.
    pub struct GpuTransformerLayer {
        config: GpuLayerConfig,
        stream: Arc<CudaStream>,
        loader: Arc<KernelLoader>,
    }

    impl GpuTransformerLayer {
        pub fn new(config: GpuLayerConfig, stream: Arc<CudaStream>, loader: Arc<KernelLoader>) -> Self {
            Self { config, stream, loader }
        }

        /// Execute a full transformer layer forward pass.
        ///
        /// Returns the output hidden states as a new CudaSlice<f32> of shape
        /// [num_tokens, hidden_size]. The caller is responsible for using this
        /// as input to the next layer.
        /// Fully f16 forward pass -- ZERO type casts within the layer.
        /// All inputs, intermediates, and outputs are f16. Uses f16 kernels
        /// throughout: rms_norm_f16, add_bias_f16, rotary_embedding_f16,
        /// reshape_and_cache_f16io, flash_attention_2_decode_f16io, hgemm,
        /// fused_residual_rmsnorm_f16, fused_silu_mul_f16, add_f16.
        // ==================================================================
        // Allocation audit (per layer, decode path with fused weights):
        //
        //  #  Buffer        Size (elems)         Overwritten by     Last read at
        //  -- ------------- -------------------- ------------------ ------------------
        //  1  normed        T*H                  rms_norm_f16       step 2 (QKV hgemm)
        //  2  qkv           T*(Q+2K)             hgemm (fused)      step 5 (KV cache write, attn Q slice)
        //  3  attn_out      T*Q                  decode_attn_f16io  step 7 (O proj hgemm)
        //  4  attn_proj     T*H                  hgemm (O proj)     step 8 (fused resid+norm)
        //  5  normed2       T*H                  fused_resid_rn     step 9 (gate_up hgemm)
        //  6  residual      T*H                  fused_resid_rn     step 11 (final add)
        //  7  gate_up       T*2I                 hgemm (gate+up)    step 9 (silu_mul reads both halves)
        //  8  fused(silu)   T*I                  silu_mul_f16_split step 10 (down proj hgemm)
        //  9  mlp_out       T*H                  hgemm (down proj)  step 11 (final add)
        // 10  final_out     T*H                  add_tensors_f16    returned to caller
        //
        // Prefill-only extras (instead of #3 above):
        //  3a q_f32         T*Q  (f32!)          cast_f16_to_f32    prefill_attention input
        //  3b attn_f32      T*Q  (f32!)          prefill_attention  cast_f32_to_f16 input
        //  3c attn_out      T*Q                  cast_f32_to_f16    step 7 (O proj hgemm)
        //  Prefill total: 12 allocs (3 extra for f32 round-trip)
        //
        // Unfused fallback extras:
        //  - separate Q/K/V instead of fused qkv: 3 allocs via hgemm_into (still 1 buf)
        //  - separate gate + up instead of fused: 2 hgemm_alloc + 1 silu = same count
        //
        // Total decode: 10 allocs
        // Total prefill: 12 allocs
        //
        // === Scratch buffer aliasing candidates ===
        //
        // Lifetime analysis (decode path, sequential execution on one stream):
        //
        //   normed     [1]: live during step 2, dead after QKV hgemm completes
        //   qkv        [2]: live steps 2-5, dead after KV cache write + attn Q read
        //   attn_out   [3]: live steps 6-7, dead after O proj hgemm
        //   attn_proj  [4]: live steps 7-8, dead after fused_resid_rmsnorm
        //   normed2    [5]: live steps 8-9, dead after gate_up hgemm
        //   residual   [6]: live steps 8-11, LONG-LIVED (spans entire MLP)
        //   gate_up    [7]: live step 9, dead after silu_mul
        //   fused      [8]: live steps 9-10, dead after down proj hgemm
        //   mlp_out    [9]: live steps 10-11, dead after final add
        //   final_out [10]: returned, cannot alias
        //
        // Reuse opportunities (non-overlapping lifetimes):
        //   A) normed [T*H] dead at step 2 -> reuse for attn_out [T*Q] at step 6
        //      (Q = num_heads*head_dim = H for most models, exact same size)
        //   B) qkv [T*(Q+2K)] dead at step 5 -> reuse for gate_up [T*2I] at step 9
        //      (qkv_dim ~ H+2*H/GQA_ratio, gate_up = 2*I; for llama 2I > qkv, but
        //       qkv buf could be a prefix if gate_up scratch is sized to max(qkv,2I))
        //   C) attn_out [T*Q] dead at step 7 -> reuse for fused/silu [T*I] at step 9
        //      (I > Q typically, so need max(Q,I) sizing)
        //   D) attn_proj [T*H] dead at step 8 -> reuse for mlp_out [T*H] at step 10
        //      (exact same size)
        //   E) normed2 [T*H] dead at step 9 -> reuse for mlp_out [T*H] at step 10
        //      (exact same size, alternative to D)
        //
        // LOW-HANGING FRUIT (same size, zero waste):
        //   1. attn_proj -> mlp_out  [both T*H, saves 1 alloc]
        //   2. normed -> attn_out    [T*H vs T*Q, same when Q=H, saves 1 alloc]
        //   3. normed2 is an alternative to attn_proj for mlp_out reuse
        //
        // With a single scratch buffer sized to max(T*qkv_dim, T*2I, T*H):
        //   Could eliminate 4-5 allocs, reducing 10 -> 5-6 per layer.
        // ==================================================================
        pub fn forward_f16(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeightsF16<'_>,
            blas: &CublasHandle,
        ) -> Result<CudaSlice<f16>> {
            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;

            let hidden_f16 = input.hidden_states_f16
                .ok_or_else(|| LLMError::GpuError("f16 hidden states required for f16 forward".into()))?;
            let norm_w = weights.input_layernorm_f16
                .ok_or_else(|| LLMError::GpuError("f16 input_layernorm required".into()))?;
            let post_norm_w = weights.post_attention_layernorm_f16
                .ok_or_else(|| LLMError::GpuError("f16 post_attention_layernorm required".into()))?;

            // 1. Pre-attention RMSNorm f16
            let normed = Self::rms_norm_f16(
                &self.stream, &self.loader,
                hidden_f16, norm_w,
                cfg.rms_norm_eps, num_tokens, hidden,
            )?;

            // 2. QKV projections: hgemm f16 x f16 -> f16
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            let qkv_dim = q_dim + kv_dim + kv_dim;

            let mut qkv = if let Some(fused_qkv) = weights.fused_qkv {
                Self::hgemm_alloc(&self.stream, blas, &normed, fused_qkv, num_tokens, qkv_dim, hidden)?
            } else {
                let mut qkv_buf = unsafe { self.stream.alloc::<f16>(num_tokens * qkv_dim) }
                    .map_err(|e| LLMError::GpuError(format!("qkv f16 alloc: {e}")))?;
                let q_end = num_tokens * q_dim;
                let k_end = q_end + num_tokens * kv_dim;
                {
                    let mut q_dst = qkv_buf.slice_mut(..q_end);
                    blas.hgemm_into(num_tokens, q_dim, hidden, 1.0, &normed, weights.q_proj, 0.0, &mut q_dst)?;
                }
                {
                    let mut k_dst = qkv_buf.slice_mut(q_end..k_end);
                    blas.hgemm_into(num_tokens, kv_dim, hidden, 1.0, &normed, weights.k_proj, 0.0, &mut k_dst)?;
                }
                {
                    let mut v_dst = qkv_buf.slice_mut(k_end..);
                    blas.hgemm_into(num_tokens, kv_dim, hidden, 1.0, &normed, weights.v_proj, 0.0, &mut v_dst)?;
                }
                qkv_buf
            };

            // 3. Apply fused QKV bias (f16) in-place
            let q_end = num_tokens * q_dim;
            let k_end = q_end + num_tokens * kv_dim;
            if let Some(bias) = weights.fused_qkv_bias {
                let mut qkv_view = qkv.slice_mut(..num_tokens * qkv_dim);
                Self::add_bias_f16_view(&self.stream, &self.loader, &mut qkv_view, bias, num_tokens, qkv_dim)?;
            }

            // 4. RoPE f16 in-place on Q and K regions (split_at_mut avoids double borrow)
            {
                let (mut q_part, mut kv_part) = qkv.split_at_mut(q_end);
                let mut k_view = kv_part.slice_mut(..num_tokens * kv_dim);
                Self::apply_rotary_embedding_f16_views(
                    &self.stream, &self.loader,
                    &mut q_part, &mut k_view,
                    &input.positions, input.rope_cos, input.rope_sin,
                    num_tokens, num_heads, num_kv_heads, head_dim,
                )?;
            }

            // 5. KV cache write: f16 K/V -> f16 cache (pure copy, no conversion)
            {
                let k_view = qkv.slice(q_end..k_end);
                let v_view = qkv.slice(k_end..);
                Self::cache_write_f16_views(
                    &self.stream, &self.loader,
                    &k_view, &v_view,
                    input.key_cache, input.value_cache, &input.slot_mapping,
                    num_tokens, num_kv_heads, head_dim,
                )?;
            }

            // 6. Attention: f16 Q, f16 KV cache, f16 output
            let attn_out = if input.is_prefill {
                // Prefill uses f32 Q kernel (prefill is one-shot, not perf-critical)
                // Fall back to f32 path for prefill: cast Q f16->f32, run f32 prefill, cast back
                let cast_f16_f32 = self.loader.get_func("cast_fp", "cast_f16_to_f32_kernel")
                    .map_err(|e| LLMError::GpuError(format!("load cast f16->f32: {e}")))?;
                let cast_f32_f16 = self.loader.get_func("cast_fp", "cast_f32_to_f16_kernel")
                    .map_err(|e| LLMError::GpuError(format!("load cast f32->f16: {e}")))?;
                let q_f16 = qkv.slice(..q_end);
                let q_f32 = Self::cast_f16_to_f32(&self.stream, &q_f16, q_end, &cast_f16_f32)?;
                let attn_f32 = Self::prefill_attention(
                    &self.stream, &self.loader,
                    &q_f32.as_view(),
                    input.key_cache, input.value_cache,
                    &input.block_tables, &input.context_lens, &input.seq_start_pos,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?;
                Self::cast_f32_to_f16(&self.stream, &attn_f32, num_tokens * num_heads * head_dim, &cast_f32_f16)?
            } else {
                Self::decode_attention_f16io(
                    &self.stream, &self.loader,
                    &qkv.slice(..q_end),
                    input.key_cache, input.value_cache,
                    &input.block_tables, &input.context_lens,
                    num_tokens, input.num_seqs, num_heads, num_kv_heads, head_dim,
                    input.max_context_len, input.block_size,
                )?
            };

            // 7. Output projection: hgemm f16
            let attn_proj = Self::hgemm_alloc(&self.stream, blas, &attn_out, weights.o_proj, num_tokens, hidden, q_dim)?;

            // 8. Fused residual + post-attention RMSNorm f16
            let (normed2, residual) = Self::fused_residual_rmsnorm_f16(
                &self.stream, &self.loader,
                hidden_f16, &attn_proj, post_norm_w,
                cfg.rms_norm_eps, num_tokens, hidden,
            )?;

            // 9. MLP: gate+up -> fused_silu_mul_f16 -> down, all f16
            let fused = if let Some(fused_gate_up) = weights.fused_gate_up {
                let gate_up = Self::hgemm_alloc(&self.stream, blas, &normed2, fused_gate_up, num_tokens, intermediate * 2, hidden)?;
                let n = num_tokens * intermediate;
                Self::fused_silu_mul_f16_split(&self.stream, &self.loader, &gate_up, n)?
            } else {
                let gate = Self::hgemm_alloc(&self.stream, blas, &normed2, weights.gate_proj, num_tokens, intermediate, hidden)?;
                let up = Self::hgemm_alloc(&self.stream, blas, &normed2, weights.up_proj, num_tokens, intermediate, hidden)?;
                Self::fused_silu_mul_f16(&self.stream, &self.loader, &gate, &up, num_tokens * intermediate)?
            };

            // 10. Down projection: hgemm f16
            let mlp_out = Self::hgemm_alloc(&self.stream, blas, &fused, weights.down_proj, num_tokens, hidden, intermediate)?;

            // 11. Residual add f16
            Self::add_tensors_f16(&self.stream, &self.loader, &residual, &mlp_out, num_tokens * hidden)
        }

        pub fn forward(
            &self,
            input: &GpuLayerInput<'_>,
            weights: &GpuLayerWeights<'_>,
            blas: &CublasHandle,
        ) -> Result<CudaSlice<f32>> {
            let cfg = &self.config;
            let num_tokens = input.num_tokens;
            let hidden = cfg.hidden_size;
            let num_heads = cfg.num_heads;
            let num_kv_heads = cfg.num_kv_heads;
            let head_dim = cfg.head_dim;
            let intermediate = cfg.intermediate_size;

            trace!(
                layer = cfg.layer_idx,
                num_tokens,
                "gpu transformer layer forward"
            );

            // ---------------------------------------------------------------
            // 1. Pre-attention RMSNorm
            // ---------------------------------------------------------------
            let normed = Self::rms_norm(
                &self.stream,
                &self.loader,
                input.hidden_states,
                weights.input_layernorm,
                cfg.rms_norm_eps,
                num_tokens,
                hidden,
            )?;

            // All ops on this stream -- no cross-stream sync needed

            // ---------------------------------------------------------------
            // 2. QKV projections via cuBLAS sgemm
            //    input [num_tokens, hidden] x weight^T [hidden, proj_dim]
            // ---------------------------------------------------------------
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            let mut q = Self::linear(
                &self.stream,
                blas,
                &normed,
                weights.q_proj,
                num_tokens,
                q_dim,
                hidden,
            )?;
            let mut k = Self::linear(
                &self.stream,
                blas,
                &normed,
                weights.k_proj,
                num_tokens,
                kv_dim,
                hidden,
            )?;
            let mut v = Self::linear(
                &self.stream,
                blas,
                &normed,
                weights.v_proj,
                num_tokens,
                kv_dim,
                hidden,
            )?;

            // Apply QKV biases if present (e.g. Qwen2.5)
            if let Some(bias) = weights.q_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut q, bias, num_tokens, q_dim)?;
            }
            if let Some(bias) = weights.k_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut k, bias, num_tokens, kv_dim)?;
            }
            if let Some(bias) = weights.v_proj_bias {
                Self::add_bias(&self.stream, &self.loader, &mut v, bias, num_tokens, kv_dim)?;
            }

            // ---------------------------------------------------------------
            // 3. RoPE on Q and K (in-place)
            // ---------------------------------------------------------------
            Self::apply_rotary_embedding_inplace(
                &self.stream,
                &self.loader,
                &mut q,
                &mut k,
                &input.positions,
                input.rope_cos,
                input.rope_sin,
                num_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            )?;

            // ---------------------------------------------------------------
            // 4. KV cache write + Attention (prefill vs decode)
            // ---------------------------------------------------------------
            // Always write K/V into paged cache via slot_mapping
            info!(layer = cfg.layer_idx, "gpu_layer: cache_write start");
            Self::cache_write(
                &self.stream,
                &self.loader,
                &k,
                &v,
                input.key_cache,
                input.value_cache,
                &input.slot_mapping,
                num_tokens,
                num_kv_heads,
                head_dim,
            )?;

            info!(layer = cfg.layer_idx, "gpu_layer: cache_write done");

            let attn_out = if input.is_prefill {
                // Prefill: use FA2 prefill kernel reading from paged cache
                info!(layer = cfg.layer_idx, "gpu_layer: prefill_attention start");
                Self::prefill_attention(
                    &self.stream,
                    &self.loader,
                    &q.as_view(),
                    input.key_cache,
                    input.value_cache,
                    &input.block_tables,
                    &input.context_lens,
                    &input.seq_start_pos,
                    num_tokens,
                    input.num_seqs,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    input.max_context_len,
                    input.block_size,
                )?
            } else {
                // Decode: read from paged cache
                info!(layer = cfg.layer_idx, "gpu_layer: decode_attention start");
                Self::decode_attention(
                    &self.stream,
                    &self.loader,
                    &q.as_view(),
                    input.key_cache,
                    input.value_cache,
                    &input.block_tables,
                    &input.context_lens,
                    num_tokens,
                    input.num_seqs,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    input.max_context_len,
                    input.block_size,
                )?
            };

            // ---------------------------------------------------------------
            // 5. Output projection
            // ---------------------------------------------------------------
            let attn_proj = Self::linear(
                &self.stream,
                blas,
                &attn_out,
                weights.o_proj,
                num_tokens,
                hidden,
                q_dim,
            )?;

            // ---------------------------------------------------------------
            // Fused residual + post-attention RMSNorm (1 kernel instead of 2)
            // ---------------------------------------------------------------
            let fused_rn_kernel = self.loader.get_func("fused_residual_rmsnorm", "fused_residual_rmsnorm_kernel")?;
            let (normed2, residual) = crate::layers::fused_ops::fused_residual_rmsnorm(
                &self.stream, &fused_rn_kernel,
                input.hidden_states, &attn_proj,
                weights.post_attention_layernorm, cfg.rms_norm_eps,
                num_tokens, hidden,
            )?;

            // ---------------------------------------------------------------
            // 7. MLP: gate_proj + up_proj -> fused_silu_mul -> down_proj
            // ---------------------------------------------------------------
            let gate = Self::linear(
                &self.stream,
                blas,
                &normed2,
                weights.gate_proj,
                num_tokens,
                intermediate,
                hidden,
            )?;
            let up = Self::linear(
                &self.stream,
                blas,
                &normed2,
                weights.up_proj,
                num_tokens,
                intermediate,
                hidden,
            )?;

            let fused = Self::fused_silu_mul(&self.stream, &self.loader, &gate, &up, num_tokens * intermediate)?;

            let mlp_out = Self::linear(
                &self.stream,
                blas,
                &fused,
                weights.down_proj,
                num_tokens,
                hidden,
                intermediate,
            )?;

            // ---------------------------------------------------------------
            // 8. Residual: residual + mlp_out
            // ---------------------------------------------------------------
            let output = Self::add_tensors(&self.stream, &self.loader, &residual, &mlp_out, num_tokens * hidden)?;

            Ok(output)
        }

        // ===================================================================
        // Private dispatch helpers
        //
        // Each wraps the corresponding CUDA kernel or cuBLAS call.
        // These are the seams where Agent 2-7 implementations plug in.
        // ===================================================================

        /// RMSNorm: out[i] = (x[i] / rms) * weight[i % hidden]
        /// where rms = sqrt(mean(x^2) + eps).
        ///
        /// Dispatches to the rms_norm CUDA kernel.
        fn rms_norm(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<f32>> {
            let n = num_tokens * hidden_size;
            let mut output = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("rms_norm alloc failed: {e}")))?;

            // Launch rms_norm kernel: one block per token, hidden_size threads per block.
            // The kernel reads `input`, `weight`, writes `output`.
            let module_name = "rms_norm";
            let func_name = "rms_norm_kernel";
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                // kernel uses extern __shared__ float sdata[blockDim.x]
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };

            // SAFETY: All CudaSlice pointers are valid device memory allocated on
            // the same device. Grid/block dims are checked above. The kernel reads
            // `input` [num_tokens * hidden_size], `weight` [hidden_size], and writes
            // `output` [num_tokens * hidden_size].
            let kernel = loader.get_func(module_name, func_name)?;
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(input)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm launch failed: {e}")))?;
            }

            Ok(output)
        }

        /// Linear projection via cuBLAS sgemm.
        /// Computes output = input @ weight^T where:
        ///   input: [m, k], weight: [n, k] (row-major), output: [m, n].
        /// Add bias in-place: tensor[i*dim + j] += bias[j]
        fn add_bias(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            tensor: &mut CudaSlice<f32>,
            bias: &CudaSlice<f32>,
            num_tokens: usize,
            dim: usize,
        ) -> Result<()> {
            let kernel = loader.get_func("add_bias", "add_bias_kernel")?;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let dim_i32 = dim as i32;
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(tensor)
                    .arg(bias)
                    .arg(&dim_i32)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("add_bias launch: {e}")))?;
            }
            Ok(())
        }

        /// add_bias on a CudaViewMut (for QKV buffer slices).
        fn add_bias_view(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            tensor: &mut CudaViewMut<'_, f32>,
            bias: &CudaSlice<f32>,
            num_tokens: usize,
            dim: usize,
        ) -> Result<()> {
            let kernel = loader.get_func("add_bias", "add_bias_kernel")?;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let dim_i32 = dim as i32;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(tensor)
                    .arg(bias)
                    .arg(&dim_i32)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("add_bias_view launch: {e}")))?;
            }
            Ok(())
        }

        /// RoPE on CudaViewMut Q/K slices from the contiguous QKV buffer.
        fn apply_rotary_embedding_inplace_views(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &mut CudaViewMut<'_, f32>,
            k: &mut CudaViewMut<'_, f32>,
            positions: &CudaView<'_, i32>,
            rope_cos: &CudaSlice<f32>,
            rope_sin: &CudaSlice<f32>,
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            if num_tokens == 0 { return Ok(()); }
            let kernel = loader.get_func("rotary_embedding", "rotary_embedding_kernel")?;
            let half_dim = head_dim / 2;
            let grid_y = num_heads.max(num_kv_heads) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, grid_y, 1),
                block_dim: (half_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(q).arg(k)
                    .arg(rope_cos).arg(rope_sin).arg(positions)
                    .arg(&(num_tokens as i32)).arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32)).arg(&(head_dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rope views launch: {e}")))?;
            }
            Ok(())
        }

        /// cache_write using CudaView K/V from contiguous QKV buffer.
        fn cache_write_views(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            k: &CudaView<'_, f32>,
            v: &CudaView<'_, f32>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            slot_mapping: &CudaView<'_, i32>,
            num_tokens: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            let kv_dim = num_kv_heads * head_dim;
            let kernel = loader.get_func("reshape_and_cache", "reshape_and_cache_f16_kernel")?;
            let threads = kv_dim.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(k).arg(v)
                    .arg(key_cache).arg(value_cache)
                    .arg(slot_mapping)
                    .arg(&(num_kv_heads as i32)).arg(&(head_dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("cache_write_views launch: {e}")))?;
            }
            Ok(())
        }

        fn linear(
            stream: &Arc<CudaStream>,
            blas: &CublasHandle,
            input: &CudaSlice<f32>,
            weight: &CudaSlice<f32>,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = stream
                .alloc_zeros::<f32>(m * n)
                .map_err(|e| LLMError::GpuError(format!("linear alloc failed: {e}")))?;

            blas.sgemm(m, n, k, 1.0, input, weight, 0.0, &mut output)?;

            Ok(output)
        }

        /// Apply RoPE in-place on Q and K -- zero allocs, zero copies.
        #[allow(clippy::too_many_arguments)]
        fn apply_rotary_embedding_inplace(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &mut CudaSlice<f32>,
            k: &mut CudaSlice<f32>,
            positions: &CudaView<'_, i32>,
            rope_cos: &CudaSlice<f32>,
            rope_sin: &CudaSlice<f32>,
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            let kernel = loader.get_func("rotary_embedding", "rotary_embedding_kernel")?;
            let half_dim = head_dim / 2;
            let grid_y = num_heads.max(num_kv_heads) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, grid_y, 1),
                block_dim: (half_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let num_tokens_i32 = num_tokens as i32;
            let num_heads_i32 = num_heads as i32;
            let num_kv_heads_i32 = num_kv_heads as i32;
            let head_dim_i32 = head_dim as i32;
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(q)
                    .arg(k)
                    .arg(rope_cos)
                    .arg(rope_sin)
                    .arg(positions)
                    .arg(&num_tokens_i32)
                    .arg(&num_heads_i32)
                    .arg(&num_kv_heads_i32)
                    .arg(&head_dim_i32)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rope inplace launch: {e}")))?;
            }
            Ok(())
        }

        /// DEPRECATED: use apply_rotary_embedding_inplace instead.
        /// Kept for backward compatibility -- allocates q_out/k_out copies.
        #[allow(dead_code)]
        fn apply_rotary_embedding(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &CudaSlice<f32>,
            k: &CudaSlice<f32>,
            positions: &CudaView<'_, i32>,
            rope_cos: &CudaSlice<f32>,
            rope_sin: &CudaSlice<f32>,
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
            let q_len = num_tokens * num_heads * head_dim;
            let k_len = num_tokens * num_kv_heads * head_dim;

            // Clone Q and K so we can apply rotation in-place.
            let mut q_out = stream
                .alloc_zeros::<f32>(q_len)
                .map_err(|e| LLMError::GpuError(format!("rope q alloc: {e}")))?;
            let mut k_out = stream
                .alloc_zeros::<f32>(k_len)
                .map_err(|e| LLMError::GpuError(format!("rope k alloc: {e}")))?;

            stream
                .memcpy_dtod(q, &mut q_out)
                .map_err(|e| LLMError::GpuError(format!("rope q copy: {e}")))?;
            stream
                .memcpy_dtod(k, &mut k_out)
                .map_err(|e| LLMError::GpuError(format!("rope k copy: {e}")))?;

            let kernel = loader.get_func("rotary_embedding", "rotary_embedding_kernel")?;

            // Single launch: grid (num_tokens, max(num_heads, num_kv_heads), 1)
            // The kernel internally guards `if (head_idx < num_kv_heads)` for K.
            let half_dim = head_dim / 2;
            let grid_y = num_heads.max(num_kv_heads) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, grid_y, 1),
                block_dim: (half_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            let num_tokens_i32 = num_tokens as i32;
            let num_heads_i32 = num_heads as i32;
            let num_kv_heads_i32 = num_kv_heads as i32;
            let head_dim_i32 = head_dim as i32;

            // positions is u32 on GPU but kernel expects int* (i32).
            // They're the same size; cast is safe.
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&mut q_out)
                    .arg(&mut k_out)
                    .arg(rope_cos)
                    .arg(rope_sin)
                    .arg(positions)
                    .arg(&num_tokens_i32)
                    .arg(&num_heads_i32)
                    .arg(&num_kv_heads_i32)
                    .arg(&head_dim_i32)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rope k launch failed: {e}")))?;
            }

            Ok((q_out, k_out))
        }

        /// Paged attention forward pass.
        ///
        /// Writes new K,V into the f16 cache at slot_mapping positions.
        /// Input k/v are f32 (from projection); the kernel converts to f16 on write.
        /// Uses reshape_and_cache_f16_kernel: 1 launch per layer.
        #[allow(clippy::too_many_arguments)]
        fn cache_write(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            k: &CudaSlice<f32>,
            v: &CudaSlice<f32>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            slot_mapping: &CudaView<'_, i32>,
            num_tokens: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            let kernel = loader.get_func("reshape_and_cache", "reshape_and_cache_f16_kernel")?;

            let kv_dim = num_kv_heads * head_dim;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (kv_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            let num_tokens_i32 = num_tokens as i32;
            let num_kv_heads_i32 = num_kv_heads as i32;
            let head_dim_i32 = head_dim as i32;

            // Kernel: (f16* key_cache, f16* value_cache, f32* key, f32* value, int* slot_mapping, int, int, int)
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(key_cache)
                    .arg(value_cache)
                    .arg(k)
                    .arg(v)
                    .arg(slot_mapping)
                    .arg(&num_tokens_i32)
                    .arg(&num_kv_heads_i32)
                    .arg(&head_dim_i32)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("reshape_and_cache_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// Prefill attention: write K/V to cache, then launch flash_attention_2_kernel
        /// reading from the paged cache with real block_tables.
        /// Naive prefill attention: per-head Q@K^T -> softmax -> @V via cuBLAS.
        /// Bypasses FA2 kernel for correctness. Used only during prefill (once per request).
        #[allow(dead_code)]
        fn naive_prefill_attention(
            stream: &Arc<CudaStream>,
            blas: &CublasHandle,
            q: &CudaSlice<f32>, // [num_tokens, num_heads * head_dim]
            k: &CudaSlice<f32>, // [num_tokens, num_kv_heads * head_dim]
            v: &CudaSlice<f32>, // [num_tokens, num_kv_heads * head_dim]
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<CudaSlice<f32>> {
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let heads_per_kv = num_heads / num_kv_heads;
            let q_stride = num_heads * head_dim;

            // Output: [num_tokens, num_heads * head_dim]
            let mut output = stream
                .alloc_zeros::<f32>(num_tokens * q_stride)
                .map_err(|e| LLMError::GpuError(format!("naive attn output alloc: {e}")))?;

            // Per-head attention via cuBLAS
            for h in 0..num_heads {
                let kv_h = h / heads_per_kv;

                // Extract Q_head [num_tokens, head_dim] from Q [num_tokens, num_heads * head_dim]
                // Extract K_head [num_tokens, head_dim] from K [num_tokens, num_kv_heads * head_dim]
                // Extract V_head [num_tokens, head_dim] from V [num_tokens, num_kv_heads * head_dim]
                // Use CPU gather for correctness (not perf-critical for prefill)
                let q_all: Vec<f32> = stream
                    .clone_dtoh(q)
                    .map_err(|e| LLMError::GpuError(format!("naive attn q DtoH: {e}")))?;
                let k_all: Vec<f32> = stream
                    .clone_dtoh(k)
                    .map_err(|e| LLMError::GpuError(format!("naive attn k DtoH: {e}")))?;
                let v_all: Vec<f32> = stream
                    .clone_dtoh(v)
                    .map_err(|e| LLMError::GpuError(format!("naive attn v DtoH: {e}")))?;

                let kv_stride = num_kv_heads * head_dim;
                let mut qh = vec![0.0f32; num_tokens * head_dim];
                let mut kh = vec![0.0f32; num_tokens * head_dim];
                let mut vh = vec![0.0f32; num_tokens * head_dim];

                for t in 0..num_tokens {
                    for d in 0..head_dim {
                        qh[t * head_dim + d] = q_all[t * q_stride + h * head_dim + d];
                        kh[t * head_dim + d] = k_all[t * kv_stride + kv_h * head_dim + d];
                        vh[t * head_dim + d] = v_all[t * kv_stride + kv_h * head_dim + d];
                    }
                }

                // scores[i][j] = sum_d qh[i][d] * kh[j][d] * scale (with causal mask)
                let mut scores = vec![0.0f32; num_tokens * num_tokens];
                for qi in 0..num_tokens {
                    for ki in 0..num_tokens {
                        if ki > qi {
                            scores[qi * num_tokens + ki] = f32::NEG_INFINITY;
                        } else {
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                dot += qh[qi * head_dim + d] * kh[ki * head_dim + d];
                            }
                            scores[qi * num_tokens + ki] = dot * scale;
                        }
                    }
                }

                // Softmax per row
                for qi in 0..num_tokens {
                    let row = &mut scores[qi * num_tokens..(qi + 1) * num_tokens];
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for v in row.iter_mut() {
                        *v = (*v - max).exp();
                        sum += *v;
                    }
                    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                    for v in row.iter_mut() {
                        *v *= inv;
                    }
                }

                // out_head = scores @ vh
                let mut out_head = vec![0.0f32; num_tokens * head_dim];
                for qi in 0..num_tokens {
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for ki in 0..num_tokens {
                            acc += scores[qi * num_tokens + ki] * vh[ki * head_dim + d];
                        }
                        out_head[qi * head_dim + d] = acc;
                    }
                }

                // Scatter back into output
                let mut out_all: Vec<f32> = stream
                    .clone_dtoh(&output)
                    .map_err(|e| LLMError::GpuError(format!("naive attn out DtoH: {e}")))?;
                for t in 0..num_tokens {
                    for d in 0..head_dim {
                        out_all[t * q_stride + h * head_dim + d] = out_head[t * head_dim + d];
                    }
                }
                output = stream
                    .clone_htod(&out_all)
                    .map_err(|e| LLMError::GpuError(format!("naive attn out HtoD: {e}")))?;
            }

            Ok(output)
        }

        /// FA2 prefill attention reading from f16 paged cache with real block_tables.
        /// Q is f32, cache is f16; the kernel loads f16 and promotes to f32 internally.
        fn prefill_attention(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &CudaView<'_, f32>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            block_tables: &CudaView<'_, i32>,
            context_lens: &CudaView<'_, i32>,
            seq_start_pos: &CudaView<'_, i32>,
            num_tokens: usize,
            num_seqs: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<CudaSlice<f32>> {
            let out_len = num_tokens * num_heads * head_dim;
            let output = stream
                .alloc_zeros::<f32>(out_len)
                .map_err(|e| LLMError::GpuError(format!("prefill_attn alloc: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            const FA2_BC: usize = 64;
            const FA2_THREADS: u32 = 128;
            let shared_mem_bytes = ((2 * FA2_BC * head_dim + FA2_BC + (FA2_THREADS as usize / 32))
                * std::mem::size_of::<f32>()) as u32;

            let kernel = loader.get_func("flash_attention", "flash_attention_2_f16kv_kernel")?;

            let bt_len = DeviceSlice::len(block_tables);
            info!(
                num_tokens,
                num_seqs,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_context_len,
                bt_len,
                shared_mem_bytes,
                "prefill_attention: dimensions"
            );

            if num_seqs == 0 {
                return Err(LLMError::GpuError(
                    "prefill_attention: num_seqs == 0".into(),
                ));
            }

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (FA2_THREADS, 1, 1),
                shared_mem_bytes,
            };

            // seq_start_pos is pre-computed by the caller (gpu_runner.rs) from actual
            // query token positions, not context_lens. This correctly handles mixed
            // prefill+decode batches where context_lens != query token counts.

            let max_blocks_per_seq = if num_seqs > 0 {
                (DeviceSlice::len(block_tables) / num_seqs) as i32
            } else {
                1
            };

            // Opt into extended shared memory if needed (A100 supports up to 100KB)
            if shared_mem_bytes > 49152 {
                kernel.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                ).map_err(|e| LLMError::GpuError(format!("prefill FA2 set max shared mem: {e}")))?;
            }

            let p_num_heads = num_heads as i32;
            let p_num_kv = num_kv_heads as i32;
            let p_head_dim = head_dim as i32;
            let p_block_size = block_size as i32;
            let p_max_ctx = max_context_len as i32;
            let p_num_tokens = num_tokens as i32;
            let p_causal = 1i32;

            // FA2 prefill kernel: 16 args via builder pattern
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&output)
                    .arg(q)
                    .arg(key_cache)
                    .arg(value_cache)
                    .arg(block_tables)
                    .arg(context_lens)
                    .arg(seq_start_pos)
                    .arg(&scale)
                    .arg(&p_num_heads)
                    .arg(&p_num_kv)
                    .arg(&p_head_dim)
                    .arg(&p_block_size)
                    .arg(&p_max_ctx)
                    .arg(&max_blocks_per_seq)
                    .arg(&p_num_tokens)
                    .arg(&p_causal)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("prefill FA2 launch: {e}")))?;
            }

            Ok(output)
        }

        /// Decode attention: read f16 K/V from paged cache, one FA2 decode kernel per layer.
        /// Q is f32, cache is f16; kernel promotes f16 to f32 on load.
        fn decode_attention(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &CudaView<'_, f32>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            block_tables: &CudaView<'_, i32>,
            context_lens: &CudaView<'_, i32>,
            num_tokens: usize,
            num_seqs: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<CudaSlice<f32>> {
            let out_len = num_tokens * num_heads * head_dim;
            let mut output = stream
                .alloc_zeros::<f32>(out_len)
                .map_err(|e| LLMError::GpuError(format!("paged_attn alloc failed: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            // Use FA2 decode kernel: correct block-level reductions, GQA support.
            // FA2_BC=64, FA2_THREADS=128 (compile-time constants in flash_attention.cu)
            const FA2_BC: usize = 64;
            const FA2_THREADS: u32 = 128;
            // smem: s_key[FA2_BC*head_dim] + s_val[FA2_BC*head_dim] + s_score[FA2_BC] + s_reduce[FA2_THREADS/32]
            let shared_mem_bytes = ((2 * FA2_BC * head_dim + FA2_BC + (FA2_THREADS as usize / 32))
                * std::mem::size_of::<f32>()) as u32;

            let module_name = "flash_attention";
            let func_name = "flash_attention_2_decode_f16kv_kernel";

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (FA2_THREADS, 1, 1),
                shared_mem_bytes,
            };

            let kernel = loader.get_func(module_name, func_name)?;

            // Opt into extended shared memory if needed
            if shared_mem_bytes > 49152 {
                kernel.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                ).map_err(|e| LLMError::GpuError(format!("decode FA2 set max shared mem: {e}")))?;
            }

            let p_num_heads = num_heads as i32;
            let p_num_kv_heads = num_kv_heads as i32;
            let p_head_dim = head_dim as i32;
            let p_block_size = block_size as i32;
            let p_max_blocks = (block_tables.len() / num_seqs.max(1)) as i32;

            // SAFETY: All slices are valid GPU memory on this device.
            // output: [num_seqs, num_heads, head_dim]
            // q:      [num_seqs, num_heads, head_dim]  (decode: num_seqs == num_tokens)
            // key_cache, value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            // block_tables: [num_seqs, max_blocks_per_seq]
            // context_lens: [num_seqs]
            // Scalar int args cast from usize; all values fit in i32 range.
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(q)
                    .arg(key_cache)
                    .arg(value_cache)
                    .arg(block_tables)
                    .arg(context_lens)
                    .arg(&scale)
                    .arg(&p_num_heads)
                    .arg(&p_num_kv_heads)
                    .arg(&p_head_dim)
                    .arg(&p_block_size)
                    .arg(&p_max_blocks)
                    .launch(cfg)
                    .map_err(|e| {
                        LLMError::GpuError(format!("flash_attention_2_decode launch failed: {e}"))
                    })?;
            }

            Ok(output)
        }

        /// Fused SiLU activation with element-wise multiply: out = silu(gate) * up.
        ///
        /// Dispatches to the activation CUDA kernel.
        fn fused_silu_mul(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            gate: &CudaSlice<f32>,
            up: &CudaSlice<f32>,
            n: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul alloc failed: {e}")))?;

            let module_name = "activation";
            let func_name = "fused_silu_mul_kernel";

            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };

            let kernel = loader.get_func(module_name, func_name)?;

            let n_i32 = n as i32;

            // SAFETY: gate, up, and output all have exactly n elements.
            // Grid covers all elements with ceil division.
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(gate)
                    .arg(up)
                    .arg(&n_i32)
                    .launch(cfg)
                    .map_err(|e| {
                        LLMError::GpuError(format!("fused_silu_mul launch failed: {e}"))
                    })?;
            }

            Ok(output)
        }

        /// Fused SiLU*mul on a contiguous [gate || up] buffer.
        /// gate = gate_up[0..n], up = gate_up[n..2n].
        fn fused_silu_mul_split(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            gate_up: &CudaSlice<f32>,
            n: usize,
        ) -> Result<CudaSlice<f32>> {
            let gate_view = gate_up.slice(..n);
            let up_view = gate_up.slice(n..n * 2);

            let mut output = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_split alloc: {e}")))?;

            let kernel = loader.get_func("activation", "fused_silu_mul_kernel")?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_i32 = n as i32;
            unsafe {
                stream
                    .launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(&gate_view)
                    .arg(&up_view)
                    .arg(&n_i32)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_split launch: {e}")))?;
            }
            Ok(output)
        }

        /// Element-wise tensor addition: out = a + b.
        ///
        /// Tries "add_bias" module first (Agent 20's dedicated kernel), then
        /// "activation" module, then falls back to CPU.
        fn add_tensors(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            a: &CudaSlice<f32>,
            b: &CudaSlice<f32>,
            n: usize,
        ) -> Result<CudaSlice<f32>> {
            let mut output = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("add_tensors alloc failed: {e}")))?;

            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };

            let n_i32 = n as i32;

            // Try dedicated add_bias module first, then activation module
            let kernel = loader
                .get_func("add_bias", "add_kernel")
                .or_else(|_| loader.get_func("activation", "add_kernel"));

            match kernel {
                Ok(k) => {
                    // SAFETY: a, b, output all have exactly n elements.
                    unsafe {
                        stream
                            .launch_builder(&k)
                            .arg(&mut output)
                            .arg(a)
                            .arg(b)
                            .arg(&n_i32)
                            .launch(cfg)
                            .map_err(|e| {
                                LLMError::GpuError(format!("add_kernel launch failed: {e}"))
                            })?;
                    }
                }
                Err(_) => {
                    // Fallback: CPU add (only until kernels are compiled).
                    let a_host = stream
                        .clone_dtoh(a)
                        .map_err(|e| LLMError::GpuError(format!("add dtoh a failed: {e}")))?;
                    let b_host = stream
                        .clone_dtoh(b)
                        .map_err(|e| LLMError::GpuError(format!("add dtoh b failed: {e}")))?;
                    let sum: Vec<f32> = a_host
                        .iter()
                        .zip(b_host.iter())
                        .map(|(x, y)| x + y)
                        .collect();
                    output = stream
                        .clone_htod(&sum)
                        .map_err(|e| LLMError::GpuError(format!("add htod failed: {e}")))?;
                }
            }

            Ok(output)
        }

        // ===================================================================
        // f16 dispatch helpers -- zero-cast f16 forward path
        // ===================================================================

        /// RMSNorm f16: f16 input, f16 weight, f16 output.
        fn rms_norm_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<CudaSlice<f16>> {
            let n = num_tokens * hidden_size;
            // Safety: kernel writes all num_tokens * hidden_size elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("rms_norm_f16 alloc: {e}")))?;
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = loader.get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(input)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm_f16 launch: {e}")))?;
            }
            Ok(output)
        }

        /// In-place RMSNorm f16: normalizes `input` directly, no output allocation.
        /// Safe because the kernel uses __syncthreads() between the read pass and
        /// write pass within each block (single-token-per-block launch).
        fn rms_norm_f16_inplace(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &mut CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<()> {
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = loader.get_func("rms_norm_f16", "rms_norm_f16_kernel")?;
            unsafe {
                let (raw_ptr, _guard) = DevicePtrMut::device_ptr_mut(input, stream);
                stream.launch_builder(&kernel)
                    .arg(&raw_ptr)  // output (same ptr)
                    .arg(&raw_ptr)  // input  (same ptr)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rms_norm_f16_inplace launch: {e}")))?;
            }
            Ok(())
        }

        /// hgemm with output allocation: f16 x f16 -> f16.
        fn hgemm_alloc(
            stream: &Arc<CudaStream>,
            blas: &CublasHandle,
            input: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<CudaSlice<f16>> {
            // Safety: hgemm with beta=0 writes all m*n elements
            let mut output = unsafe { stream.alloc::<f16>(m * n) }
                .map_err(|e| LLMError::GpuError(format!("hgemm_alloc: {e}")))?;
            blas.hgemm(m, n, k, f16::ONE, input, weight, f16::ZERO, &mut output)?;
            Ok(output)
        }

        /// Add bias f16 in-place on a CudaViewMut<f16>.
        fn add_bias_f16_view(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            tensor: &mut CudaViewMut<'_, f16>,
            bias: &CudaSlice<f16>,
            num_tokens: usize,
            dim: usize,
        ) -> Result<()> {
            let kernel = loader.get_func("add_bias_f16", "add_bias_f16_kernel")?;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(tensor)
                    .arg(bias)
                    .arg(&(dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("add_bias_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// RoPE f16 in-place on CudaViewMut<f16> Q/K slices.
        fn apply_rotary_embedding_f16_views(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &mut CudaViewMut<'_, f16>,
            k: &mut CudaViewMut<'_, f16>,
            positions: &CudaView<'_, i32>,
            rope_cos: &CudaSlice<f32>,
            rope_sin: &CudaSlice<f32>,
            num_tokens: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            if num_tokens == 0 { return Ok(()); }
            let kernel = loader.get_func("rotary_embedding_f16", "rotary_embedding_f16_kernel")?;
            let half_dim = head_dim / 2;
            let grid_y = num_heads.max(num_kv_heads) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, grid_y, 1),
                block_dim: (half_dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(q).arg(k)
                    .arg(rope_cos).arg(rope_sin).arg(positions)
                    .arg(&(num_tokens as i32)).arg(&(num_heads as i32))
                    .arg(&(num_kv_heads as i32)).arg(&(head_dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("rope_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// Cache write f16: f16 K/V input -> f16 paged cache (pure copy).
        fn cache_write_f16_views(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            k: &CudaView<'_, f16>,
            v: &CudaView<'_, f16>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            slot_mapping: &CudaView<'_, i32>,
            num_tokens: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            let kv_dim = num_kv_heads * head_dim;
            let kernel = loader.get_func("reshape_and_cache_f16", "reshape_and_cache_f16io_kernel")?;
            let threads = kv_dim.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(key_cache).arg(value_cache)
                    .arg(k).arg(v)
                    .arg(slot_mapping)
                    .arg(&(num_tokens as i32))
                    .arg(&(num_kv_heads as i32))
                    .arg(&(head_dim as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("cache_write_f16 launch: {e}")))?;
            }
            Ok(())
        }

        /// FA2 decode attention with f16 I/O: f16 Q, f16 KV cache, f16 output.
        #[allow(clippy::too_many_arguments)]
        fn decode_attention_f16io(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            q: &CudaView<'_, f16>,
            key_cache: &CudaSlice<f16>,
            value_cache: &CudaSlice<f16>,
            block_tables: &CudaView<'_, i32>,
            context_lens: &CudaView<'_, i32>,
            num_tokens: usize,
            num_seqs: usize,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            max_context_len: u32,
            block_size: usize,
        ) -> Result<CudaSlice<f16>> {
            let out_len = num_tokens * num_heads * head_dim;
            // Safety: attention kernel writes all out_len elements
            let mut output = unsafe { stream.alloc::<f16>(out_len) }
                .map_err(|e| LLMError::GpuError(format!("decode_attn_f16io alloc: {e}")))?;

            let scale = 1.0f32 / (head_dim as f32).sqrt();

            const FA2_BC: usize = 64;
            const FA2_THREADS: u32 = 128;
            let shared_mem_bytes = ((2 * FA2_BC * head_dim + FA2_BC + (FA2_THREADS as usize / 32))
                * std::mem::size_of::<f32>()) as u32;

            let kernel = loader.get_func("flash_attention", "flash_attention_2_decode_f16io_kernel")?;

            let cfg = LaunchConfig {
                grid_dim: (num_seqs as u32, num_heads as u32, 1),
                block_dim: (FA2_THREADS, 1, 1),
                shared_mem_bytes,
            };

            if shared_mem_bytes > 49152 {
                kernel.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                ).map_err(|e| LLMError::GpuError(format!("decode FA2 f16io set max shared mem: {e}")))?;
            }

            let p_num_heads = num_heads as i32;
            let p_num_kv_heads = num_kv_heads as i32;
            let p_head_dim = head_dim as i32;
            let p_block_size = block_size as i32;
            let p_max_blocks = (block_tables.len() / num_seqs.max(1)) as i32;

            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(q)
                    .arg(key_cache).arg(value_cache)
                    .arg(block_tables).arg(context_lens)
                    .arg(&scale)
                    .arg(&p_num_heads).arg(&p_num_kv_heads)
                    .arg(&p_head_dim).arg(&p_block_size)
                    .arg(&p_max_blocks)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("FA2 decode f16io launch: {e}")))?;
            }
            Ok(output)
        }

        /// Fused residual add + RMSNorm, all f16.
        /// Returns (normed_output, residual) both f16.
        fn fused_residual_rmsnorm_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            input: &CudaSlice<f16>,
            add: &CudaSlice<f16>,
            weight: &CudaSlice<f16>,
            eps: f32,
            num_tokens: usize,
            hidden_size: usize,
        ) -> Result<(CudaSlice<f16>, CudaSlice<f16>)> {
            let n = num_tokens * hidden_size;
            // Safety: kernel writes all n elements to both output and residual
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_rn_f16 output alloc: {e}")))?;
            let mut residual = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_rn_f16 residual alloc: {e}")))?;
            let block_threads = hidden_size.min(1024) as u32;
            let cfg = LaunchConfig {
                grid_dim: (num_tokens as u32, 1, 1),
                block_dim: (block_threads, 1, 1),
                shared_mem_bytes: block_threads * std::mem::size_of::<f32>() as u32,
            };
            let kernel = loader.get_func("fused_residual_rmsnorm_f16", "fused_residual_rmsnorm_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(&mut residual)
                    .arg(input)
                    .arg(add)
                    .arg(weight)
                    .arg(&eps)
                    .arg(&(hidden_size as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("fused_rn_f16 launch: {e}")))?;
            }
            Ok((output, residual))
        }

        /// Fused SiLU * mul, f16 variant.
        fn fused_silu_mul_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            gate: &CudaSlice<f16>,
            up: &CudaSlice<f16>,
            n: usize,
        ) -> Result<CudaSlice<f16>> {
            // Safety: element-wise kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel = loader.get_func("activation_f16", "fused_silu_mul_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(gate).arg(up)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16 launch: {e}")))?;
            }
            Ok(output)
        }

        /// Fused SiLU*mul on a contiguous [gate || up] buffer, f16 variant.
        fn fused_silu_mul_f16_split(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            gate_up: &CudaSlice<f16>,
            n: usize,
        ) -> Result<CudaSlice<f16>> {
            let gate_view = gate_up.slice(..n);
            let up_view = gate_up.slice(n..n * 2);
            // Safety: element-wise kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16_split alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel = loader.get_func("activation_f16", "fused_silu_mul_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(&gate_view).arg(&up_view)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("fused_silu_mul_f16_split launch: {e}")))?;
            }
            Ok(output)
        }

        /// Element-wise tensor addition f16: out = a + b.
        fn add_tensors_f16(
            stream: &Arc<CudaStream>,
            loader: &KernelLoader,
            a: &CudaSlice<f16>,
            b: &CudaSlice<f16>,
            n: usize,
        ) -> Result<CudaSlice<f16>> {
            // Safety: element-wise kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("add_tensors_f16 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let kernel = loader.get_func("add_bias_f16", "add_f16_kernel")?;
            unsafe {
                stream.launch_builder(&kernel)
                    .arg(&mut output)
                    .arg(a).arg(b)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("add_f16 launch: {e}")))?;
            }
            Ok(output)
        }

        /// Cast f16 -> f32 (used only for prefill fallback).
        fn cast_f16_to_f32(
            stream: &Arc<CudaStream>,
            input: &CudaView<'_, f16>,
            n: usize,
            kernel: &CudaFunction,
        ) -> Result<CudaSlice<f32>> {
            // Safety: cast kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f32>(n) }
                .map_err(|e| LLMError::GpuError(format!("cast_f16_f32 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(kernel)
                    .arg(&mut output)
                    .arg(input)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("cast_f16_f32 launch: {e}")))?;
            }
            Ok(output)
        }

        /// Cast f32 -> f16 (used only for prefill fallback).
        fn cast_f32_to_f16(
            stream: &Arc<CudaStream>,
            input: &CudaSlice<f32>,
            n: usize,
            kernel: &CudaFunction,
        ) -> Result<CudaSlice<f16>> {
            // Safety: cast kernel writes all n elements
            let mut output = unsafe { stream.alloc::<f16>(n) }
                .map_err(|e| LLMError::GpuError(format!("cast_f32_f16 alloc: {e}")))?;
            let threads = 256u32;
            let blocks = ((n as u32) + threads - 1) / threads;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                stream.launch_builder(kernel)
                    .arg(&mut output)
                    .arg(input)
                    .arg(&(n as i32))
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("cast_f32_f16 launch: {e}")))?;
            }
            Ok(output)
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::*;

#[cfg(test)]
mod tests {
    // Tests run under default features (mock-gpu), so we verify the module
    // compiles but the CUDA types are not exposed.
    #[test]
    fn module_compiles_without_cuda() {
        // Under mock-gpu the `inner` module is not compiled.
        // This test confirms that the crate still builds cleanly.
        assert!(true);
    }
}
