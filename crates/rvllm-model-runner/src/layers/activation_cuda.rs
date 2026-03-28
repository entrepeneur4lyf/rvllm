//! CUDA dispatch for activation kernels: SiLU, GELU, fused SiLU*mul.
//!
//! Launches the element-wise kernels defined in `kernels/activation.cu` via cudarc.
//! All three kernels use the same launch config:
//!   Grid:  (ceil(n / 256), 1, 1)
//!   Block: (256, 1, 1)
//!   Shared memory: 0
//!
//! Gated behind `#[cfg(feature = "cuda")]` -- mock-gpu builds never see the inner types.

#[cfg(feature = "cuda")]
mod inner {
    use std::sync::Arc;

    use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
        LaunchConfig, PushKernelArg,
    };
    use tracing::trace;

    use rvllm_core::prelude::{LLMError, Result};

    const BLOCK_SIZE: u32 = 256;

    fn launch_cfg(n: u32) -> LaunchConfig {
        let grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Load the activation PTX module into the context, returning the module handle.
    fn load_activation_module(
        context: &Arc<CudaContext>,
        ptx_bytes: &[u8],
    ) -> Result<Arc<CudaModule>> {
        let ptx_str = std::str::from_utf8(ptx_bytes)
            .map_err(|e| LLMError::GpuError(format!("activation PTX is not valid UTF-8: {e}")))?;
        let module = context
            .load_module(cudarc::nvrtc::Ptx::from_src(ptx_str))
            .map_err(|e| LLMError::GpuError(format!("failed to load activation PTX: {e}")))?;
        trace!("activation module loaded");
        Ok(module)
    }

    /// CUDA SiLU (Swish) activation: x / (1 + exp(-x)).
    ///
    /// Holds a preloaded kernel handle to avoid repeated PTX lookups.
    pub struct CudaSiLU {
        context: Arc<CudaContext>,
        func: CudaFunction,
    }

    impl CudaSiLU {
        /// Load the activation PTX and extract `silu_kernel`.
        ///
        /// `ptx_bytes` should be the compiled PTX of `kernels/activation.cu`.
        pub fn new(context: Arc<CudaContext>, ptx_bytes: &[u8]) -> Result<Self> {
            let module = load_activation_module(&context, ptx_bytes)?;
            let func = module.load_function("silu_kernel").map_err(|e| {
                LLMError::GpuError(format!("silu_kernel not found in activation module: {e}"))
            })?;
            trace!("CudaSiLU: loaded silu_kernel");
            Ok(Self { context, func })
        }

        /// Apply SiLU element-wise, returning a new device buffer.
        pub fn forward(
            &self,
            input: &CudaSlice<f32>,
            stream: &Arc<CudaStream>,
        ) -> Result<CudaSlice<f32>> {
            let n = input.len();
            let mut output = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("SiLU alloc failed: {e}")))?;
            let cfg = launch_cfg(n as u32);
            let n_i32 = n as i32;
            // SAFETY: kernel reads `n` f32 from `input`, writes `n` f32 to `output`.
            // Both device slices have length >= n. The i32 `n` matches the kernel
            // signature `int n`.
            unsafe {
                stream
                    .launch_builder(&self.func)
                    .arg(&mut output)
                    .arg(input)
                    .arg(&n_i32)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("silu_kernel launch failed: {e}")))?;
            }
            trace!(n, "silu_kernel launched");
            Ok(output)
        }

        /// Apply SiLU in-place (output overwrites input).
        pub fn forward_inplace(
            &self,
            data: &mut CudaSlice<f32>,
            stream: &Arc<CudaStream>,
        ) -> Result<()> {
            let n = data.len();
            let cfg = launch_cfg(n as u32);
            let n_i32 = n as i32;
            // SAFETY: In-place: output==input is safe for element-wise SiLU.
            // Extract raw device pointer to pass aliased output/input to the kernel.
            unsafe {
                let (ptr, _sync_guard) = DevicePtrMut::device_ptr_mut(data, stream);
                stream
                    .launch_builder(&self.func)
                    .arg(&ptr) // output
                    .arg(&ptr) // input (aliased)
                    .arg(&n_i32)
                    .launch(cfg)
                    .map_err(|e| {
                        LLMError::GpuError(format!("silu_kernel inplace launch failed: {e}"))
                    })?;
            }
            trace!(n, "silu_kernel launched (inplace)");
            Ok(())
        }

        pub fn context(&self) -> &Arc<CudaContext> {
            &self.context
        }
    }

    /// CUDA GELU activation (tanh approximation).
    pub struct CudaGELU {
        context: Arc<CudaContext>,
        func: CudaFunction,
    }

    impl CudaGELU {
        /// Load the activation PTX and extract `gelu_kernel`.
        pub fn new(context: Arc<CudaContext>, ptx_bytes: &[u8]) -> Result<Self> {
            let module = load_activation_module(&context, ptx_bytes)?;
            let func = module.load_function("gelu_kernel").map_err(|e| {
                LLMError::GpuError(format!("gelu_kernel not found in activation module: {e}"))
            })?;
            trace!("CudaGELU: loaded gelu_kernel");
            Ok(Self { context, func })
        }

        /// Apply GELU element-wise, returning a new device buffer.
        pub fn forward(
            &self,
            input: &CudaSlice<f32>,
            stream: &Arc<CudaStream>,
        ) -> Result<CudaSlice<f32>> {
            let n = input.len();
            let mut output = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("GELU alloc failed: {e}")))?;
            let cfg = launch_cfg(n as u32);
            let n_i32 = n as i32;
            // SAFETY: kernel reads `n` f32 from `input`, writes `n` f32 to `output`.
            // Both slices are device-allocated with length >= n.
            unsafe {
                stream
                    .launch_builder(&self.func)
                    .arg(&mut output)
                    .arg(input)
                    .arg(&n_i32)
                    .launch(cfg)
                    .map_err(|e| LLMError::GpuError(format!("gelu_kernel launch failed: {e}")))?;
            }
            trace!(n, "gelu_kernel launched");
            Ok(output)
        }

        /// Apply GELU in-place.
        pub fn forward_inplace(
            &self,
            data: &mut CudaSlice<f32>,
            stream: &Arc<CudaStream>,
        ) -> Result<()> {
            let n = data.len();
            let cfg = launch_cfg(n as u32);
            let n_i32 = n as i32;
            // SAFETY: pure element-wise -- same aliasing rationale as CudaSiLU::forward_inplace.
            unsafe {
                let (ptr, _sync_guard) = DevicePtrMut::device_ptr_mut(data, stream);
                stream
                    .launch_builder(&self.func)
                    .arg(&ptr)
                    .arg(&ptr)
                    .arg(&n_i32)
                    .launch(cfg)
                    .map_err(|e| {
                        LLMError::GpuError(format!("gelu_kernel inplace launch failed: {e}"))
                    })?;
            }
            trace!(n, "gelu_kernel launched (inplace)");
            Ok(())
        }

        pub fn context(&self) -> &Arc<CudaContext> {
            &self.context
        }
    }

    /// Fused SiLU(gate) * up on GPU -- single kernel, saves a full memory traversal
    /// and a temporary buffer compared to separate SiLU + element-wise multiply.
    pub struct CudaFusedSiLUMul {
        context: Arc<CudaContext>,
        func: CudaFunction,
    }

    impl CudaFusedSiLUMul {
        /// Load the activation PTX and extract `fused_silu_mul_kernel`.
        pub fn new(context: Arc<CudaContext>, ptx_bytes: &[u8]) -> Result<Self> {
            let module = load_activation_module(&context, ptx_bytes)?;
            let func = module.load_function("fused_silu_mul_kernel").map_err(|e| {
                LLMError::GpuError(format!(
                    "fused_silu_mul_kernel not found in activation module: {e}"
                ))
            })?;
            trace!("CudaFusedSiLUMul: loaded fused_silu_mul_kernel");
            Ok(Self { context, func })
        }

        /// Compute silu(gate) * up element-wise, returning a new device buffer.
        ///
        /// `gate` and `up` must have the same length.
        pub fn forward(
            &self,
            gate: &CudaSlice<f32>,
            up: &CudaSlice<f32>,
            stream: &Arc<CudaStream>,
        ) -> Result<CudaSlice<f32>> {
            let n = gate.len();
            if up.len() != n {
                return Err(LLMError::GpuError(format!(
                    "fused_silu_mul: gate len {} != up len {}",
                    n,
                    up.len()
                )));
            }
            let mut output = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| LLMError::GpuError(format!("fused_silu_mul alloc failed: {e}")))?;
            let cfg = launch_cfg(n as u32);
            let n_i32 = n as i32;
            // SAFETY: kernel reads `n` elements each from `gate` and `up`, writes `n`
            // elements to `output`. All three slices are device-allocated with length >= n.
            unsafe {
                stream
                    .launch_builder(&self.func)
                    .arg(&mut output)
                    .arg(gate)
                    .arg(up)
                    .arg(&n_i32)
                    .launch(cfg)
                    .map_err(|e| {
                        LLMError::GpuError(format!("fused_silu_mul_kernel launch failed: {e}"))
                    })?;
            }
            trace!(n, "fused_silu_mul_kernel launched");
            Ok(output)
        }

        /// Compute silu(gate) * up, writing the result into `gate` in-place.
        ///
        /// `gate` and `up` must have the same length.
        pub fn forward_inplace(
            &self,
            gate: &mut CudaSlice<f32>,
            up: &CudaSlice<f32>,
            stream: &Arc<CudaStream>,
        ) -> Result<()> {
            let n = gate.len();
            if up.len() != n {
                return Err(LLMError::GpuError(format!(
                    "fused_silu_mul inplace: gate len {} != up len {}",
                    n,
                    up.len()
                )));
            }
            let cfg = launch_cfg(n as u32);
            let n_i32 = n as i32;
            // SAFETY: output aliases gate for in-place element-wise op.
            // Extract raw pointers to pass aliased gate as both output and input.
            unsafe {
                let (gate_ptr, _gate_guard) = DevicePtrMut::device_ptr_mut(gate, stream);
                let (up_ptr, _up_guard) = DevicePtr::device_ptr(up, stream);
                stream
                    .launch_builder(&self.func)
                    .arg(&gate_ptr) // output (aliases gate)
                    .arg(&gate_ptr) // gate input
                    .arg(&up_ptr) // up input
                    .arg(&n_i32)
                    .launch(cfg)
                    .map_err(|e| {
                        LLMError::GpuError(format!(
                            "fused_silu_mul_kernel inplace launch failed: {e}"
                        ))
                    })?;
            }
            trace!(n, "fused_silu_mul_kernel launched (inplace)");
            Ok(())
        }

        pub fn context(&self) -> &Arc<CudaContext> {
            &self.context
        }
    }
}

#[cfg(feature = "cuda")]
pub use inner::{CudaFusedSiLUMul, CudaGELU, CudaSiLU};

#[cfg(test)]
mod tests {
    #[test]
    fn module_compiles() {
        // Compile-only sanity check under default (mock-gpu) features.
    }
}
