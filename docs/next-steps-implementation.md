# Next Steps: Implementation Details

Each optimization is designed to be implemented as a single change, coherency-checked, and benchmarked before moving to the next. No swarms. One at a time.

---

## Step 1: CUDA Graph Capture/Replay (In Progress)

### What
Capture the entire decode forward pass into a CUDA graph after warmup. On subsequent decode steps with the same padded batch size, replay the graph instead of re-launching 252+ individual kernels (28 layers x 9 kernels each).

### Why it matters
Kernel launch overhead is ~10us per launch. At 252 launches per step, that's ~2.5ms of pure CPU-GPU launch overhead. A CUDA graph replays all 252 launches as a single driver call (~5us total). At N=128 producing 128 tokens per step, eliminating 2.5ms of overhead on a ~8ms step is a 30% improvement.

### Why it broke last time
The previous implementation allocated new GPU buffers for metadata (token_ids, positions, block_tables) every step via `clone_htod()`. CUDA graphs bake buffer POINTERS into the recorded kernel launches. On replay, kernels read from the CAPTURE-TIME pointers (now pointing to freed memory or old data), not the new buffers. Result: stale positions -> wrong RoPE -> wrong attention -> degenerate output.

### Correct implementation
1. Pre-allocate persistent GPU buffers at init time, sized for `max_num_seqs`:
   - `persistent_token_ids: CudaSlice<i32>` [max_num_seqs]
   - `persistent_positions: CudaSlice<i32>` [max_num_seqs]
   - `persistent_context_lens: CudaSlice<i32>` [max_num_seqs]
   - `persistent_block_tables: CudaSlice<i32>` [max_num_seqs * max_blocks_per_seq]
   - `persistent_slot_mapping: CudaSlice<i32>` [max_num_seqs]
   - `persistent_seq_start_pos: CudaSlice<i32>` [max_num_seqs + 1]

2. In `forward_ex()`, write metadata into persistent buffers via `memcpy_htod()` (updates data at existing pointers, doesn't change pointers).

3. Capture: after GRAPH_WARMUP_CALLS=3, call `cuStreamBeginCapture`, run `forward_ex()` (which uses persistent buffers), call `cuStreamEndCapture`. Store the graph keyed by padded batch size.

4. Replay: on subsequent decode steps with matching batch size, `memcpy_htod()` new metadata into persistent buffers, then `cuGraphLaunch`. The replayed kernels read updated data from the same pointers.

5. Guards:
   - Only capture decode steps (all query_lens == 1)
   - Never capture prefill (variable query lengths)
   - If batch size changes to a new padded size, capture a new graph
   - Padded sizes: {1, 2, 4, 8, 16, 32, 64, 128, 256}

### Files
- `crates/rvllm-model-runner/src/gpu_runner.rs` -- persistent buffers, forward_ex changes
- `crates/rvllm-worker/src/gpu_worker.rs` -- capture/replay orchestration
- `crates/rvllm-worker/src/graph_runner.rs` -- graph pool management

### Verification
- Coherency: 5 diverse prompts at N=1, same outputs as non-graph path
- Coherency: N=128 batch, all 128 responses coherent
- Benchmark: expect 15-25% gain across all N

### Expected result
8,578 -> ~10,000-10,700 tok/s

---

## Step 2: GQA-Optimized Attention Kernel

### What
Qwen2.5-1.5B uses Grouped Query Attention with 12 query heads and 2 KV heads (ratio 6:1). The current FA2 decode kernel launches one thread block per (seq, head) pair with grid (num_seqs, 12, 1). Each of the 12 blocks independently loads the KV cache for its assigned KV head. But 6 query heads share the same KV head -- so we load the same KV data 6 times.

A GQA-optimized kernel launches one block per (seq, kv_head) pair with grid (num_seqs, 2, 1). Each block loads KV once and processes all 6 query heads that share it.

### Why it matters
For the decode kernel, KV cache loading is the primary bottleneck. With 32-token context and 128 head_dim, each KV tile is 32 * 128 * 2 bytes (f16) = 8KB for K and 8KB for V = 16KB per tile. At 12 heads, that's 12 * 16KB = 192KB per sequence per step. With GQA optimization: 2 * 16KB = 32KB. That's 6x less KV bandwidth.

At longer contexts (512-4096 tokens), the savings are proportionally larger and attention becomes a bigger fraction of total time.

### Implementation
Create `kernels/flash_attention_gqa.cu` with `flash_attention_2_decode_gqa_f16kv_kernel`:

```c
// Grid: (num_seqs, num_kv_heads, 1)  -- NOT num_heads
// Block: (256, 1, 1)  -- 8 warps to handle 6 query heads
// Shared memory: K tile + V tile + 6 Q vectors + 6 score arrays

__global__ void flash_attention_2_decode_gqa_f16kv_kernel(
    float* output,              // [num_seqs, num_heads, head_dim]
    const float* query,         // [num_seqs, num_heads, head_dim]
    const half* key_cache,      // [num_blocks, block_size, num_kv_heads, head_dim]
    const half* value_cache,
    const int* block_tables,
    const int* context_lens,
    float scale,
    int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_blocks_per_seq
) {
    int seq_idx = blockIdx.x;
    int kv_head_idx = blockIdx.y;
    int heads_per_kv = num_heads / num_kv_heads;  // 6

    // Load Q vectors for all 6 query heads into registers/shared mem
    // For each KV tile:
    //   Load K tile ONCE from paged cache
    //   For each of 6 query heads:
    //     Compute QK dot product
    //     Online softmax update
    //   Load V tile ONCE from paged cache
    //   For each of 6 query heads:
    //     Accumulate P*V
    // Write 6 output vectors
}
```

Key design choices:
- 256 threads (8 warps). Assign warps to query heads: warp 0-1 handle Q head 0, warp 2-3 handle Q head 1, etc. (but 8 warps for 6 heads doesn't divide evenly -- alternative: 192 threads with 6 warps, one per query head)
- K and V tiles loaded cooperatively by ALL threads (coalesced), then each warp subgroup computes its own query head's dot product
- Shared memory: K[64 * 128] + V[64 * 128] (shared) + Q[6 * 128] + S[6 * 64] = ~66KB (fits in extended smem)

### Files
- New: `kernels/flash_attention_gqa.cu`
- `crates/rvllm-gpu/src/kernel_loader.rs` -- register new kernel
- `crates/rvllm-model-runner/src/gpu_layer.rs` -- dispatch to GQA kernel when num_heads != num_kv_heads

### Verification
- Coherency: output must match non-GQA kernel exactly (same QK scores, same softmax, same output)
- Test with diverse prompts at N=1 and N=128
- Benchmark: expect 10-25% gain depending on context length

### Expected result
~10,500 -> ~11,500-12,000 tok/s (after CUDA graph fix)

---

## Step 3: Fused QKV Projection

### What
Replace 3 separate cuBLAS hgemm calls for Q, K, V projections with 1 fused call.

Currently:
```
Q = hidden @ q_proj.T   // hgemm: [N, 1536] x [1536, 1536] -> [N, 1536]
K = hidden @ k_proj.T   // hgemm: [N, 1536] x [256, 1536]  -> [N, 256]
V = hidden @ v_proj.T   // hgemm: [N, 1536] x [256, 1536]  -> [N, 256]
```

Fused:
```
QKV_weight = concat(q_proj, k_proj, v_proj)  // [2048, 1536]
QKV = hidden @ QKV_weight.T                   // [N, 2048]
Q = QKV[:, :1536]
K = QKV[:, 1536:1792]
V = QKV[:, 1792:2048]
```

### Why it matters
3 kernel launches -> 1. Larger GEMM has better GPU utilization (more output tiles). The input matrix (hidden) is read from HBM once instead of 3 times. At FP16, hidden is [N, 1536] = N * 3KB. Reading it 3x wastes 2 * N * 3KB of bandwidth per layer.

### Implementation
In `gpu_layer.rs`:
1. Add `qkv_fused_weight: RefCell<Option<CudaSlice<f16>>>` to GpuTransformerLayer
2. On first forward call, concatenate q/k/v projection weights into one buffer via `memcpy_dtod` (3 copies, done once, cached)
3. In forward_f16(), do one hgemm with the fused weight, then split output into Q, K, V via CudaView slices (zero-copy, just pointer+offset)

The split is zero-cost because Q, K, V occupy contiguous memory in the fused output. CudaView/CudaSlice slicing gives sub-views without copying.

### Files
- `crates/rvllm-model-runner/src/gpu_layer.rs`

### Verification
- Coherency: Q, K, V values must be identical to unfused path
- Test at N=1 and N=128
- Benchmark: expect 3-8% gain

---

## Step 4: Fused Gate+Up Projection

### What
Same pattern as Step 3 but for the MLP gate and up projections.

Currently:
```
gate = hidden @ gate_proj.T  // [N, 8960]
up   = hidden @ up_proj.T    // [N, 8960]
```

Fused:
```
gate_up_weight = concat(gate_proj, up_proj)  // [17920, 1536]
gate_up = hidden @ gate_up_weight.T          // [N, 17920]
gate = gate_up[:, :8960]
up   = gate_up[:, 8960:]
```

### Why it matters
Gate+up GEMMs are the largest per-layer operations (8960 output dim vs 1536 for QKV). Fusing them saves 1 kernel launch and 1 redundant read of the hidden state.

### Implementation
Same pattern as Step 3. Add fused weight buffer, concatenate once, single hgemm, split output.

### Files
- `crates/rvllm-model-runner/src/gpu_layer.rs`

### Verification
- Coherency check after
- Benchmark: expect 2-5% gain

---

## Step 5: cublasLt for Small-Batch Decode

### What
Switch from cuBLAS `cublasGemmEx` to `cublasLtMatmul` for decode GEMMs where M (batch size) is small. cublasLt's heuristic automatically selects split-K algorithms that create more parallelism for tall-skinny shapes.

### Why it matters
At N=1, Q projection [1, 1536] x [1536, 1536] produces 12 output tiles. 108 SMs, 12 tiles = 11% utilization. cublasLt with split-K=8 creates 96 tiles = 89% utilization.

The gain is largest at N=1-32 where SM underutilization is severe. At N=128+, standard cuBLAS is already well-utilized and cublasLt adds no benefit (slight overhead from algorithm selection).

### Implementation
Code already exists in `crates/rvllm-gpu/src/cublaslt_ops.rs` behind the `cublaslt` feature flag. What's needed:
1. Wire `CublasLtOps` into `GpuWorker` (create alongside `CublasHandle`)
2. Pass it through to `GpuModelRunner` and `GpuTransformerLayer`
3. In `forward_f16()`, check `num_tokens <= CUBLASLT_M_THRESHOLD` (32). If yes, use `forward_once_f16_lt()`. If no, use standard `forward_once_f16()`.

### Files
- `crates/rvllm-worker/src/gpu_worker.rs` -- create CublasLtOps
- `crates/rvllm-model-runner/src/gpu_runner.rs` -- store and pass through
- `crates/rvllm-model-runner/src/gpu_layer.rs` -- dispatch

### Verification
- Coherency at N=1 (most affected)
- Benchmark at N=1, N=4, N=16, N=32, N=128 (should see gains only at low N)
- Expected: 20-40% at N=1, 0% at N=128+

---

## Step 6: Fix Split-KV Wiring

### What
Wire the existing `split_kv_attention.cu` kernels into the decode attention path for long-context scenarios.

### Why it failed last time
The kernel argument order in the Rust dispatch didn't match the CUDA kernel signature. Need to carefully verify every arg matches.

### Implementation
1. Read `kernels/split_kv_attention.cu` and extract the EXACT signature of `split_kv_decode_f16kv_kernel`
2. Read `kernels/split_kv_attention.cu` for `split_kv_combine_kernel` signature
3. In `gpu_layer.rs` decode_attention, add split-KV path gated by `choose_num_splits(max_context_len) > 1`
4. Match every kernel argument BY NAME AND TYPE against the CUDA signature
5. Test with a LONGER prompt (512+ tokens) where split-KV actually fires

### Verification
- Coherency with 5-token prompt (num_splits=1, uses single kernel)
- Coherency with 512-token prompt (num_splits=2+, uses split+combine)
- Benchmark with longer prompts to see actual gain

---

## Step 7: CuTE/CUTLASS Tensor Core Attention

### What
Rewrite the FA2 decode kernel using CuTE layout algebra and MMA atoms to route QK dot products and PV accumulation through tensor cores instead of scalar FMA.

### Why it matters
A100 tensor cores: 312 TFLOPS FP16. CUDA cores: 19.5 TFLOPS FP32. Current kernel uses CUDA cores for all attention math. Switching to tensor cores is a 16x compute throughput improvement for the attention phase. Even though attention is memory-bound for decode, the faster compute allows processing more KV positions per clock cycle, reducing overall latency.

### Implementation
1. Vendor CUTLASS 3.x headers into `kernels/third_party/cutlass/include/`
2. Add `--std=c++17 -I third_party/cutlass/include` to nvcc flags in `build.rs` and `build.sh`
3. Create `kernels/cute_flash_decode.cu` using CuTE abstractions:
   - `SM80_16x8x16_F32F16F16F32` MMA atom for QK and PV
   - `SM80_CP_ASYNC_CACHEALWAYS` copy atom for KV tile loads
   - Online softmax in registers between MMA tiles
   - `extern "C"` wrapper for PTX-compatible function names
4. Register in kernel_loader.rs, dispatch from gpu_layer.rs

### Prerequisites
Steps 1-2 (CUDA graphs + GQA) should be done first. CuTE is high complexity.

### Verification
- Compare output against CPU reference attention (existing test infra)
- Coherency across all batch sizes
- Expected: 20-40% overall throughput gain

---

## Step 8: FP8 Inference

### What
Load model weights as FP8 (E4M3), use cuBLAS FP8 GEMM. Halves weight memory bandwidth vs FP16.

### Why it matters
At N=128, weight loading (2.5GB FP16) takes ~1.25ms on A100. With FP8 (1.25GB), it takes ~0.6ms. This directly reduces the per-step time for the bandwidth-bound GEMM operations.

### Implementation
1. Add FP8 weight loading path in `gpu_loader.rs` (quantize FP16->FP8 with per-tensor scale)
2. Use `cublasLtMatmul` with `CUDA_R_8F_E4M3` data type
3. Wire existing `fp8_kv.cu` for FP8 KV cache end-to-end
4. Add `--dtype fp8` CLI flag

### Prerequisites
Steps 1-5 should be done first. FP8 needs careful quality validation.

### Verification
- Output quality comparison: FP8 vs FP16 (perplexity, coherency)
- May need calibration for per-tensor vs per-channel scaling
- Expected: 30-60% throughput gain with minimal quality loss

---

## Sequencing Summary

```
Step 1: CUDA graphs (in progress) -> coherency check -> benchmark
Step 2: GQA attention kernel      -> coherency check -> benchmark
Step 3: Fused QKV                  -> coherency check -> benchmark
Step 4: Fused gate+up              -> coherency check -> benchmark
Step 5: cublasLt wiring            -> coherency check -> benchmark
Step 6: Split-KV fix               -> coherency check -> benchmark (long context)
Step 7: CuTE tensor core attention -> coherency check -> benchmark
Step 8: FP8 inference              -> coherency check -> benchmark + quality eval
```

Each step is independent. If any step breaks coherency, revert it and move to the next. Cumulative expected gain: 8,578 -> 14,000-16,000 tok/s.
