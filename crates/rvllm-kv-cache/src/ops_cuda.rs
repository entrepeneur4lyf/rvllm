//! CUDA-accelerated cache operations for paged KV buffers.
//!
//! Provides GPU-native implementations of copy_blocks, swap_in, swap_out,
//! and reshape_and_cache that avoid round-tripping data through the host
//! where possible.
//!
//! All code is behind `#[cfg(feature = "cuda")]` and requires cudarc.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;

    use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use half::f16;
    use rvllm_core::prelude::{LLMError, Result};
    use tracing::debug;

    const COPY_BLOCKS_FN: &str = "copy_blocks_kernel";

    fn map_cuda<E: std::fmt::Display>(context: &str) -> impl FnOnce(E) -> LLMError + '_ {
        move |e| LLMError::GpuError(format!("{context}: {e}"))
    }

    /// Load the copy_blocks PTX module and return the kernel function.
    ///
    /// Reads pre-compiled PTX from the path specified by RVLLM_KERNEL_DIR
    /// env var (defaults to `kernels/` relative to the workspace root).
    /// Requires that `copy_blocks.ptx` has been compiled from `copy_blocks.cu`
    /// via `nvcc --ptx` or the build system (Agent 1 / Agent 20).
    fn load_copy_blocks_func(context: &Arc<CudaContext>) -> Result<CudaFunction> {
        let kernel_dir =
            std::env::var("RVLLM_KERNEL_DIR").unwrap_or_else(|_| "kernels".to_string());
        let ptx_path = std::path::Path::new(&kernel_dir).join("copy_blocks.ptx");

        let ptx_src = std::fs::read_to_string(&ptx_path).map_err(|e| {
            LLMError::GpuError(format!(
                "failed to read PTX from {}: {e}. Compile with: nvcc --ptx kernels/copy_blocks.cu -o kernels/copy_blocks.ptx",
                ptx_path.display()
            ))
        })?;

        let ptx = cudarc::nvrtc::compile_ptx(ptx_src)
            .map_err(|e| LLMError::GpuError(format!("failed to compile copy_blocks PTX: {e}")))?;

        let module: Arc<CudaModule> = context
            .load_module(ptx)
            .map_err(map_cuda("failed to load copy_blocks module"))?;

        let func = module
            .load_function(COPY_BLOCKS_FN)
            .map_err(map_cuda("failed to load copy_blocks_kernel function"))?;

        Ok(func)
    }

    /// Copy cache blocks on the GPU using the copy_blocks CUDA kernel.
    ///
    /// `key_cache` and `value_cache` are flat f16 GPU buffers shaped
    /// [num_blocks, block_size, num_heads, head_dim].
    /// `block_mapping` contains (src_block, dst_block) pairs.
    ///
    /// Launches the copy_blocks_kernel with one CUDA block per mapping pair.
    /// Note: the copy_blocks_kernel operates on raw bytes via element count,
    /// so it works with f16 buffers -- each "element" is 2 bytes.
    pub fn copy_blocks_cuda(
        key_cache: &mut CudaSlice<f16>,
        value_cache: &mut CudaSlice<f16>,
        block_mapping: &[(i64, i64)],
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        if block_mapping.is_empty() {
            return Ok(());
        }

        let func = load_copy_blocks_func(context)?;

        let num_pairs = block_mapping.len();

        // Flatten mapping to contiguous [src0, dst0, src1, dst1, ...] for the kernel
        let mut flat_mapping: Vec<i64> = Vec::with_capacity(num_pairs * 2);
        for &(src, dst) in block_mapping {
            flat_mapping.push(src);
            flat_mapping.push(dst);
        }

        let d_mapping = stream
            .clone_htod(&flat_mapping)
            .map_err(map_cuda("copy_blocks: upload mapping"))?;

        let elems_per_block = block_size * num_heads * head_dim;
        let threads = elems_per_block.min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_pairs as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: kernel signature matches copy_blocks_kernel(half*, half*, long*, int, int, int, int).
        // All device pointers come from valid CudaSlice allocations on the same device.
        unsafe {
            stream
                .launch_builder(&func)
                .arg(key_cache)
                .arg(value_cache)
                .arg(&d_mapping)
                .arg(&(num_pairs as i32))
                .arg(&(block_size as i32))
                .arg(&(num_heads as i32))
                .arg(&(head_dim as i32))
                .launch(cfg)
                .map_err(map_cuda("copy_blocks: kernel launch"))?;
        }

        debug!(num_pairs, "copy_blocks_cuda complete");
        Ok(())
    }

    /// Swap cache blocks from CPU to GPU (swap-in).
    ///
    /// Each (cpu_block_idx, gpu_block_idx) pair copies one block of key and value
    /// data from host memory into device memory.
    ///
    /// `gpu_key`/`gpu_value`: device-side cache buffers.
    /// `cpu_key`/`cpu_value`: host-side cache data (flat, same layout as GPU).
    pub fn swap_in_cuda(
        gpu_key: &mut CudaSlice<f16>,
        gpu_value: &mut CudaSlice<f16>,
        cpu_key: &[f16],
        cpu_value: &[f16],
        mapping: &[(usize, usize)],
        elements_per_block: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        if mapping.is_empty() {
            return Ok(());
        }

        // Download current GPU state, patch mapped blocks from CPU, re-upload.
        // cudarc doesn't expose sub-slice async memcpy, so this is the safe path.
        let mut key_host = stream
            .clone_dtoh(gpu_key)
            .map_err(map_cuda("swap_in: dtoh key"))?;
        let mut val_host = stream
            .clone_dtoh(gpu_value)
            .map_err(map_cuda("swap_in: dtoh value"))?;

        for &(cpu_idx, gpu_idx) in mapping {
            let cpu_off = cpu_idx * elements_per_block;
            let gpu_off = gpu_idx * elements_per_block;
            let cpu_end = cpu_off + elements_per_block;
            let gpu_end = gpu_off + elements_per_block;

            if cpu_end > cpu_key.len() {
                return Err(LLMError::MemoryError(format!(
                    "swap_in: CPU block {} offset {cpu_end} exceeds buffer len {}",
                    cpu_idx,
                    cpu_key.len()
                )));
            }
            if gpu_end > key_host.len() {
                return Err(LLMError::MemoryError(format!(
                    "swap_in: GPU block {} offset {gpu_end} exceeds buffer len {}",
                    gpu_idx,
                    key_host.len()
                )));
            }

            key_host[gpu_off..gpu_end].copy_from_slice(&cpu_key[cpu_off..cpu_end]);
            val_host[gpu_off..gpu_end].copy_from_slice(&cpu_value[cpu_off..cpu_end]);
        }

        stream
            .memcpy_htod(&key_host, gpu_key)
            .map_err(map_cuda("swap_in: htod key"))?;
        stream
            .memcpy_htod(&val_host, gpu_value)
            .map_err(map_cuda("swap_in: htod value"))?;

        debug!(pairs = mapping.len(), "swap_in_cuda complete");
        Ok(())
    }

    /// Swap cache blocks from GPU to CPU (swap-out).
    ///
    /// Each (gpu_block_idx, cpu_block_idx) pair copies one block of key and value
    /// data from device memory into host memory.
    pub fn swap_out_cuda(
        gpu_key: &CudaSlice<f16>,
        gpu_value: &CudaSlice<f16>,
        cpu_key: &mut [f16],
        cpu_value: &mut [f16],
        mapping: &[(usize, usize)],
        elements_per_block: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        if mapping.is_empty() {
            return Ok(());
        }

        let key_host = stream
            .clone_dtoh(gpu_key)
            .map_err(map_cuda("swap_out: dtoh key"))?;
        let val_host = stream
            .clone_dtoh(gpu_value)
            .map_err(map_cuda("swap_out: dtoh value"))?;

        for &(gpu_idx, cpu_idx) in mapping {
            let gpu_off = gpu_idx * elements_per_block;
            let cpu_off = cpu_idx * elements_per_block;
            let gpu_end = gpu_off + elements_per_block;
            let cpu_end = cpu_off + elements_per_block;

            if gpu_end > key_host.len() {
                return Err(LLMError::MemoryError(format!(
                    "swap_out: GPU block {} offset {gpu_end} exceeds buffer len {}",
                    gpu_idx,
                    key_host.len()
                )));
            }
            if cpu_end > cpu_key.len() {
                return Err(LLMError::MemoryError(format!(
                    "swap_out: CPU block {} offset {cpu_end} exceeds buffer len {}",
                    cpu_idx,
                    cpu_key.len()
                )));
            }

            cpu_key[cpu_off..cpu_end].copy_from_slice(&key_host[gpu_off..gpu_end]);
            cpu_value[cpu_off..cpu_end].copy_from_slice(&val_host[gpu_off..gpu_end]);
        }

        debug!(pairs = mapping.len(), "swap_out_cuda complete");
        Ok(())
    }

    /// Reshape and cache key/value tensors into paged GPU buffers.
    ///
    /// `key` and `value` are host-side flat token-level tensors of shape
    /// [num_tokens, num_heads * head_dim].
    /// `slot_mapping` maps each token to a flat slot in the cache buffer.
    /// Negative slot values are skipped (padding tokens).
    ///
    /// Cache buffers are shaped [num_blocks, block_size, num_heads, head_dim] (flat).
    ///
    /// Stages the scatter on the host then uploads once, avoiding per-token
    /// kernel launches. A dedicated reshape_and_cache CUDA kernel can be
    /// integrated via KernelLoader (Agent 1) for zero-copy on-device scatter.
    pub fn reshape_and_cache_cuda(
        key: &[f16],
        value: &[f16],
        key_cache: &mut CudaSlice<f16>,
        value_cache: &mut CudaSlice<f16>,
        slot_mapping: &[i32],
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
        stream: &Arc<CudaStream>,
    ) -> Result<()> {
        let head_stride = num_heads * head_dim;
        let num_tokens = slot_mapping.len();

        if key.len() != num_tokens * head_stride {
            return Err(LLMError::MemoryError(format!(
                "reshape_and_cache_cuda: key len {} != num_tokens({}) * head_stride({})",
                key.len(),
                num_tokens,
                head_stride
            )));
        }
        if value.len() != num_tokens * head_stride {
            return Err(LLMError::MemoryError(format!(
                "reshape_and_cache_cuda: value len {} != num_tokens({}) * head_stride({})",
                value.len(),
                num_tokens,
                head_stride
            )));
        }

        let block_stride = block_size * head_stride;

        let mut key_data = stream
            .clone_dtoh(key_cache)
            .map_err(map_cuda("reshape_and_cache: dtoh key"))?;
        let mut val_data = stream
            .clone_dtoh(value_cache)
            .map_err(map_cuda("reshape_and_cache: dtoh value"))?;

        for (token_idx, &slot) in slot_mapping.iter().enumerate() {
            if slot < 0 {
                continue;
            }
            let slot = slot as usize;
            let block_idx = slot / block_size;
            let block_offset = slot % block_size;

            let cache_offset = block_idx * block_stride + block_offset * head_stride;
            let src_offset = token_idx * head_stride;

            if cache_offset + head_stride > key_data.len() {
                return Err(LLMError::MemoryError(format!(
                    "reshape_and_cache_cuda: cache offset {} + {} exceeds buffer len {}",
                    cache_offset,
                    head_stride,
                    key_data.len()
                )));
            }

            key_data[cache_offset..cache_offset + head_stride]
                .copy_from_slice(&key[src_offset..src_offset + head_stride]);
            val_data[cache_offset..cache_offset + head_stride]
                .copy_from_slice(&value[src_offset..src_offset + head_stride]);
        }

        stream
            .memcpy_htod(&key_data, key_cache)
            .map_err(map_cuda("reshape_and_cache: htod key"))?;
        stream
            .memcpy_htod(&val_data, value_cache)
            .map_err(map_cuda("reshape_and_cache: htod value"))?;

        debug!(num_tokens, "reshape_and_cache_cuda complete");
        Ok(())
    }

    /// High-level cache ops handle bundling device reference with cache geometry.
    ///
    /// Provides a convenient method-based interface for the worker (Agent 15)
    /// and engine layers to call cache operations without threading geometry
    /// parameters through every call.
    pub struct CudaCacheOps {
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        num_heads: usize,
        head_dim: usize,
        block_size: usize,
    }

    impl CudaCacheOps {
        pub fn new(
            context: Arc<CudaContext>,
            stream: Arc<CudaStream>,
            num_heads: usize,
            head_dim: usize,
            block_size: usize,
        ) -> Self {
            Self {
                context,
                stream,
                num_heads,
                head_dim,
                block_size,
            }
        }

        pub fn elements_per_block(&self) -> usize {
            self.block_size * self.num_heads * self.head_dim
        }

        pub fn context(&self) -> &Arc<CudaContext> {
            &self.context
        }

        pub fn stream(&self) -> &Arc<CudaStream> {
            &self.stream
        }

        /// Copy blocks within GPU cache using the CUDA copy_blocks kernel.
        pub fn copy_blocks(
            &self,
            key_cache: &mut CudaSlice<f16>,
            value_cache: &mut CudaSlice<f16>,
            block_mapping: &[(i64, i64)],
        ) -> Result<()> {
            copy_blocks_cuda(
                key_cache,
                value_cache,
                block_mapping,
                self.block_size,
                self.num_heads,
                self.head_dim,
                &self.context,
                &self.stream,
            )
        }

        /// Swap blocks from CPU into GPU cache.
        pub fn swap_in(
            &self,
            gpu_key: &mut CudaSlice<f16>,
            gpu_value: &mut CudaSlice<f16>,
            cpu_key: &[f16],
            cpu_value: &[f16],
            mapping: &[(usize, usize)],
        ) -> Result<()> {
            swap_in_cuda(
                gpu_key,
                gpu_value,
                cpu_key,
                cpu_value,
                mapping,
                self.elements_per_block(),
                &self.stream,
            )
        }

        /// Swap blocks from GPU cache out to CPU.
        pub fn swap_out(
            &self,
            gpu_key: &CudaSlice<f16>,
            gpu_value: &CudaSlice<f16>,
            cpu_key: &mut [f16],
            cpu_value: &mut [f16],
            mapping: &[(usize, usize)],
        ) -> Result<()> {
            swap_out_cuda(
                gpu_key,
                gpu_value,
                cpu_key,
                cpu_value,
                mapping,
                self.elements_per_block(),
                &self.stream,
            )
        }

        /// Reshape and scatter key/value tokens into paged cache buffers.
        pub fn reshape_and_cache(
            &self,
            key: &[f16],
            value: &[f16],
            key_cache: &mut CudaSlice<f16>,
            value_cache: &mut CudaSlice<f16>,
            slot_mapping: &[i32],
        ) -> Result<()> {
            reshape_and_cache_cuda(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.num_heads,
                self.head_dim,
                self.block_size,
                &self.stream,
            )
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::*;
