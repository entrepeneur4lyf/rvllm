//! CUDA kernel loader: loads PTX files and launches kernels via cudarc.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::nvrtc::Ptx;
use tracing::{debug, info, trace};

/// Extract the raw CUfunction handle from a CudaFunction.
/// CudaFunction's first field is `cu_function: sys::CUfunction` (a *mut CUfunc_st).
/// SAFETY: CudaFunction is a non-repr(C) struct but the first field is always at offset 0 for
/// a struct whose first field is a pointer (no padding before it). This is an implementation
/// detail of cudarc 0.19 and is only used in unsafe raw-launch paths.
unsafe fn extract_cu_function(func: &CudaFunction) -> cudarc::driver::sys::CUfunction {
    std::ptr::read((func as *const CudaFunction).cast::<cudarc::driver::sys::CUfunction>())
}

use crate::Result;

/// Known kernel -> function-name mappings for the vllm-rs kernel set.
/// These are the `extern "C" __global__` entry points in each .cu file.
static KERNEL_FUNCTIONS: &[(&str, &[&str])] = &[
    (
        "activation",
        &["silu_kernel", "fused_silu_mul_kernel", "gelu_kernel"],
    ),
    ("argmax", &["argmax_kernel"]),
    (
        "activation_f16",
        &[
            "silu_f16_kernel",
            "fused_silu_mul_f16_kernel",
            "gelu_f16_kernel",
        ],
    ),
    ("add_bias", &["add_bias_kernel", "add_kernel"]),
    (
        "add_bias_f16",
        &["add_bias_f16_kernel", "add_f16_kernel", "add_inplace_f16_kernel"],
    ),
    ("copy_blocks", &["copy_blocks_kernel"]),
    ("embedding_gather", &["embedding_gather_kernel"]),
    ("embedding_gather_f16", &["embedding_gather_f16_kernel"]),
    (
        "flash_attention",
        &[
            "flash_attention_2_kernel",
            "flash_attention_2_decode_kernel",
            "flash_attention_2_f16kv_kernel",
            "flash_attention_2_decode_f16kv_kernel",
            "flash_attention_2_decode_f16io_kernel",
        ],
    ),
    (
        "fp8_kv",
        &[
            "quantize_kv_kernel",
            "dequantize_kv_kernel",
            "quantize_paged_kv_kernel",
            "dequantize_paged_kv_kernel",
        ],
    ),
    ("fused_residual_rmsnorm", &["fused_residual_rmsnorm_kernel"]),
    ("fused_residual_rmsnorm_f16", &["fused_residual_rmsnorm_f16_kernel"]),
    (
        "paged_attention",
        &[
            "paged_attention_v2_kernel",
            "paged_attention_v2_f16kv_kernel",
        ],
    ),
    (
        "split_kv_attention",
        &[
            "split_kv_decode_f16kv_kernel",
            "split_kv_combine_kernel",
            "split_kv_decode_fp8kv_kernel",
            "split_kv_decode_single_f16kv_kernel",
        ],
    ),
    (
        "reshape_and_cache",
        &["reshape_and_cache_kernel", "reshape_and_cache_f16_kernel"],
    ),
    (
        "cast_fp",
        &["cast_f32_to_f16_kernel", "cast_f16_to_f32_kernel"],
    ),
    (
        "reshape_and_cache_f16",
        &["reshape_and_cache_f16io_kernel"],
    ),
    ("gemv_f16", &["gemv_f16_kernel", "gemv_batched_f16_kernel"]),
    ("rms_norm", &["rms_norm_kernel"]),
    ("rms_norm_f16", &["rms_norm_f16_kernel"]),
    ("rotary_embedding", &["rotary_embedding_kernel"]),
    ("rotary_embedding_f16", &["rotary_embedding_f16_kernel"]),
    ("softmax", &["softmax_kernel"]),
    (
        "fused_lm_head_argmax",
        &[
            "fused_lm_head_argmax_kernel",
            "fused_lm_head_argmax_reduce_kernel",
        ],
    ),
    (
        "fused_lm_head_argmax_f16",
        &["fused_lm_head_argmax_f16_kernel"],
    ),
];

/// Loads and manages CUDA PTX modules, providing kernel launch capabilities.
///
/// Wraps `cudarc::driver::CudaContext` module management with a higher-level
/// API that understands the vllm-rs kernel naming conventions.
pub struct KernelLoader {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    modules: HashMap<String, Arc<CudaModule>>,
    loaded_func_names: HashMap<String, Vec<&'static str>>,
}

impl KernelLoader {
    /// Create a new KernelLoader and load all .ptx files from `kernel_dir`.
    ///
    /// Falls back to the `RVLLM_KERNEL_DIR` environment variable if `kernel_dir`
    /// does not contain any .ptx files.
    pub fn new(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        kernel_dir: &Path,
    ) -> Result<Self> {
        let mut loader = Self {
            context,
            stream,
            modules: HashMap::new(),
            loaded_func_names: HashMap::new(),
        };

        let dir = if kernel_dir.exists() && kernel_dir.is_dir() {
            kernel_dir.to_path_buf()
        } else if let Ok(env_dir) = std::env::var("RVLLM_KERNEL_DIR") {
            let p = Path::new(&env_dir).to_path_buf();
            if p.exists() && p.is_dir() {
                p
            } else {
                info!(
                    dir = %kernel_dir.display(),
                    env_dir = %p.display(),
                    "no PTX directory found, kernel loader created empty"
                );
                return Ok(loader);
            }
        } else {
            info!(
                dir = %kernel_dir.display(),
                "no PTX directory found, kernel loader created empty"
            );
            return Ok(loader);
        };

        loader.load_directory(&dir)?;
        Ok(loader)
    }

    /// Create a KernelLoader with no pre-loaded kernels.
    /// Kernels can be loaded later via `load_ptx`.
    pub fn empty(context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
        Self {
            context,
            stream,
            modules: HashMap::new(),
            loaded_func_names: HashMap::new(),
        }
    }

    /// Load a single PTX module from raw bytes (UTF-8 PTX source).
    ///
    /// The `name` is used as the module name for later `get_func` / `launch` calls.
    /// Function names are resolved from the known kernel mapping if available,
    /// otherwise the caller should use `load_ptx_with_functions` to specify them.
    pub fn load_ptx(&mut self, name: &str, ptx_bytes: &[u8]) -> Result<()> {
        let func_names = self.resolve_function_names(name);
        self.load_ptx_with_functions(name, ptx_bytes, &func_names)
    }

    /// Load a PTX module from raw bytes, explicitly specifying function names to register.
    pub fn load_ptx_with_functions(
        &mut self,
        name: &str,
        ptx_bytes: &[u8],
        func_names: &[&'static str],
    ) -> Result<()> {
        let ptx_src = std::str::from_utf8(ptx_bytes).map_err(|e| {
            crate::LLMError::GpuError(format!("PTX bytes for '{name}' are not valid UTF-8: {e}"))
        })?;

        let ptx = Ptx::from_src(ptx_src);

        let module = self.context.load_module(ptx).map_err(|e| {
            crate::LLMError::GpuError(format!("failed to load PTX module '{name}': {e}"))
        })?;

        debug!(module = name, functions = ?func_names, "loaded PTX module");
        self.modules.insert(name.to_string(), module);
        self.loaded_func_names
            .insert(name.to_string(), func_names.to_vec());
        Ok(())
    }

    /// Load a PTX file from disk by path.
    pub fn load_ptx_file(&mut self, name: &str, path: &Path) -> Result<()> {
        let func_names = self.resolve_function_names(name);
        let ptx = Ptx::from_file(path);

        let module = self.context.load_module(ptx).map_err(|e| {
            crate::LLMError::GpuError(format!("failed to load PTX file '{}': {e}", path.display()))
        })?;

        debug!(module = name, path = %path.display(), "loaded PTX file");
        self.modules.insert(name.to_string(), module);
        self.loaded_func_names
            .insert(name.to_string(), func_names.to_vec());
        Ok(())
    }

    /// Retrieve a loaded CUDA function by module and function name.
    pub fn get_func(&self, module: &str, function: &str) -> Result<CudaFunction> {
        let m = self
            .modules
            .get(module)
            .ok_or_else(|| crate::LLMError::GpuError(format!("module '{module}' not loaded")))?;
        m.load_function(function).map_err(|e| {
            crate::LLMError::GpuError(format!(
                "function '{function}' not found in module '{module}': {e}"
            ))
        })
    }

    /// Check if a module has been loaded.
    pub fn has_module(&self, module: &str) -> bool {
        self.modules.contains_key(module)
    }

    /// Check if a specific function is available in a loaded module.
    pub fn has_func(&self, module: &str, function: &str) -> bool {
        self.modules
            .get(module)
            .and_then(|m| m.load_function(function).ok())
            .is_some()
    }

    /// Launch a kernel on the loader's stream.
    ///
    /// # Safety
    /// The caller must ensure that the kernel arguments match the kernel signature
    /// exactly: correct types, correct count, correct mutability for output buffers.
    pub unsafe fn launch_raw(
        &self,
        module: &str,
        function: &str,
        cfg: LaunchConfig,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        let func = self.get_func(module, function)?;
        self.context
            .bind_to_thread()
            .map_err(|e| crate::LLMError::GpuError(format!("CUDA bind failed: {e}")))?;
        let cu_func = extract_cu_function(&func);
        cudarc::driver::result::launch_kernel(
            cu_func,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.stream.cu_stream(),
            args,
        )
        .map_err(|e| {
            crate::LLMError::GpuError(format!("kernel launch {module}::{function} failed: {e}"))
        })
    }

    /// Launch a kernel on a specific stream.
    ///
    /// # Safety
    /// Same requirements as `launch_raw`, plus the caller must ensure no
    /// data races between this stream and other concurrent streams.
    pub unsafe fn launch_on_stream_raw(
        &self,
        module: &str,
        function: &str,
        cfg: LaunchConfig,
        stream: &CudaStream,
        args: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        let func = self.get_func(module, function)?;
        self.context
            .bind_to_thread()
            .map_err(|e| crate::LLMError::GpuError(format!("CUDA bind failed: {e}")))?;
        let cu_func = extract_cu_function(&func);
        cudarc::driver::result::launch_kernel(
            cu_func,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            stream.cu_stream(),
            args,
        )
        .map_err(|e| {
            crate::LLMError::GpuError(format!(
                "kernel launch {module}::{function} on stream failed: {e}"
            ))
        })
    }

    /// Returns a reference to the underlying CUDA context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Returns a reference to the underlying CUDA stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Returns a reference to the underlying CUDA device (context alias).
    /// Kept for backward compatibility during migration.
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// List all loaded module names.
    pub fn loaded_modules(&self) -> Vec<&str> {
        self.modules.keys().map(|s| s.as_str()).collect()
    }

    // --- private helpers ---

    /// Scan a directory for .ptx files and load each one.
    fn load_directory(&mut self, dir: &Path) -> Result<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            crate::LLMError::GpuError(format!("cannot read kernel dir '{}': {e}", dir.display()))
        })?;

        let mut count = 0u32;
        for entry in entries {
            let entry =
                entry.map_err(|e| crate::LLMError::GpuError(format!("readdir error: {e}")))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("ptx") {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                self.load_ptx_file(stem, &path)?;
                count += 1;
            }
        }

        info!(dir = %dir.display(), count, "loaded PTX files from directory");
        Ok(())
    }

    /// Resolve function names for a known kernel module, or fall back to
    /// a convention-based guess (`{name}_kernel`).
    fn resolve_function_names(&self, name: &str) -> Vec<&'static str> {
        for &(module_name, funcs) in KERNEL_FUNCTIONS {
            if module_name == name {
                return funcs.to_vec();
            }
        }
        // Convention fallback: leak a string so we get &'static str.
        // This is intentional -- kernel names live for the process lifetime.
        let fallback: &'static str = Box::leak(format!("{name}_kernel").into_boxed_str());
        trace!(
            module = name,
            fallback,
            "using convention-based function name"
        );
        vec![fallback]
    }
}

/// Helper to build a `LaunchConfig` from grid/block tuples and shared memory size.
pub fn launch_config(
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
) -> LaunchConfig {
    LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: shared_mem,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn resolve_known_kernel_names() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let loader = KernelLoader::empty(context, stream);

        let names = loader.resolve_function_names("activation");
        assert_eq!(
            names,
            &["silu_kernel", "fused_silu_mul_kernel", "gelu_kernel"]
        );

        let names = loader.resolve_function_names("rms_norm");
        assert_eq!(names, &["rms_norm_kernel"]);
    }

    #[test]
    fn resolve_unknown_uses_convention() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let loader = KernelLoader::empty(context, stream);

        let names = loader.resolve_function_names("my_custom");
        assert_eq!(names, &["my_custom_kernel"]);
    }

    #[test]
    fn empty_loader_has_no_modules() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let loader = KernelLoader::empty(context, stream);
        assert!(loader.loaded_modules().is_empty());
        assert!(!loader.has_module("anything"));
    }

    #[test]
    fn new_with_nonexistent_dir() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        let loader =
            KernelLoader::new(context, stream, &PathBuf::from("/nonexistent/path")).unwrap();
        assert!(loader.loaded_modules().is_empty());
    }
}
