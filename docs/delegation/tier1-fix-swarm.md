# Tier 1 Fix Swarm -- WILL PRODUCE WRONG OUTPUT

## Bugs to fix (from review)

### Bug #1: Paged attention race condition
- File: kernels/paged_attention.cu:71-80
- Warp reduction stores partial via dim_idx == 0, but multiple warps write to same logits[t]
- Fix: use shared memory reduction instead of racy warp-level writes

### Bug #6: Scheduler re-adds scheduled groups
- File: crates/rvllm-engine/src/gpu_engine.rs (FifoScheduler)
- After scheduling, groups cloned back into self.running -> infinite re-scheduling
- Fix: scheduled groups should be removed from pending, not re-added

### Bug #8: Non-power-of-2 RMSNorm reduction
- Files: kernels/rms_norm.cu:37, kernels/fused_residual_rmsnorm.cu:42
- Tree reduction `for (s = stride/2; s > 0; s >>= 1)` drops elements beyond last power-of-2
- Fix: add odd-element fold-in like softmax.cu already has

### Bug #2: RoPE base hardcoded
- File: crates/rvllm-model-runner/src/gpu_runner.rs:104
- Hardcoded rope_theta = 1_000_000.0
- Fix: read from ModelRunnerConfig (which already has it from config.json)

## Agent assignments

### Agent A: Fix paged_attention.cu race (#1)
- EDIT: kernels/paged_attention.cu
- DO NOT TOUCH: any .rs file, any other .cu file
- Fix the warp reduction race: use shared memory atomicAdd or proper tree reduction
- Keep kernel signature unchanged

### Agent B: Fix scheduler re-add (#6)
- EDIT: crates/rvllm-engine/src/gpu_engine.rs (FifoScheduler only)
- DO NOT TOUCH: async_gpu_engine.rs, worker code, kernel files
- Fix: scheduled groups should not be cloned back into self.running/self.groups

### Agent C: Fix RMSNorm non-power-of-2 (#8)
- EDIT: kernels/rms_norm.cu, kernels/fused_residual_rmsnorm.cu
- DO NOT TOUCH: any .rs file, softmax.cu (already fixed), other .cu files
- Fix: add odd-element handling in tree reduction (same pattern as softmax.cu)

### Agent D: Fix RoPE base from config (#2)
- EDIT: crates/rvllm-model-runner/src/gpu_runner.rs (rope_theta line only)
- DO NOT TOUCH: kernels, engine, worker
- Fix: read rope_theta from self.config instead of hardcoding 1_000_000.0
- Check ModelRunnerConfig for the field

## Rules
1. Each agent touches ONLY its assigned files
2. No benchmarking
3. No architecture changes
4. Keep kernel signatures unchanged
5. Test: after all fixes, one smoke test on A100
