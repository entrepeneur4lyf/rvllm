# Decode Correctness Swarm Spec

## Known State (HEAD=ef87794+)

### What works
- Embedding gather: GPU kernel, correct args (hidden_size, vocab_size)
- RMSNorm: correct arg order (output, input, weight, eps, hidden_size)
- QKV projections: cuBLAS sgemm with OP_T/OP_N, correct for PyTorch [out,in] layout
- RoPE: single kernel launch, 9 correct args, GPU cos/sin tables
- reshape_and_cache: new kernel, writes K/V to paged cache via slot_mapping
- Prefill FA2: flash_attention_2_kernel with extended shared memory opt-in
- Decode FA2: flash_attention_2_decode_kernel
- Block tables: now persistent across step() calls in gpu_engine.rs
- Softmax kernel: shared-memory tree reduction (PR #1)

### What's broken
- **Prefill forward pass produces correct tensors** (verified via probes)
- **Decode steps produce token 0 (`!`) every time**
- The engine step loop works: step() -> schedule() -> build_metadata() -> worker.execute() -> gpu_forward()
- The issue is specifically in decode-step data flow, not prefill

### Architecture
- `crates/rvllm-engine/src/gpu_engine.rs` - engine loop, metadata builder, block allocation
- `crates/rvllm-engine/src/async_gpu_engine.rs` - async wrapper, background step loop
- `crates/rvllm-worker/src/gpu_worker.rs` - worker, delegates to GpuModelRunner
- `crates/rvllm-worker/src/input.rs` - prepares ModelInput from SequenceGroupMetadata
- `crates/rvllm-model-runner/src/gpu_runner.rs` - GpuModelRunner::forward()
- `crates/rvllm-model-runner/src/gpu_layer.rs` - GpuTransformerLayer::forward()
- `kernels/flash_attention.cu` - both FA2 kernels
- `kernels/reshape_and_cache.cu` - KV cache scatter kernel

### Decode step data flow
1. `gpu_engine.rs::step()` calls `build_metadata(&scheduled_groups)`
2. For decode: `is_prompt=false`, output_token_ids has generated tokens
3. `input.rs::prepare_decode()` builds ModelInput with:
   - token_ids: just the new token
   - position_ids: [prompt_len + output_len - 1]
   - slot_mapping: from block_tables for the new position
   - context_lens: [total_tokens_so_far]
   - block_tables: persistent from engine
4. `GpuModelRunner::forward()` runs with is_prefill=false
5. `GpuTransformerLayer::forward()` calls:
   - cache_write (reshape_and_cache for the new token)
   - decode_attention (flash_attention_2_decode_kernel)

### Kernel signatures (exact, from .cu files)
```
flash_attention_2_decode_kernel(output, query, key_cache, value_cache, block_tables, context_lens, scale, num_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq)
```
- block_tables: int* [num_seqs, max_blocks_per_seq]
- context_lens: int* [num_seqs]
- key_cache/value_cache: float* [num_blocks, block_size, num_kv_heads, head_dim]

```
reshape_and_cache_kernel(key_cache, value_cache, key, value, slot_mapping, num_tokens, num_kv_heads, head_dim)
```
- slot_mapping: int* [num_tokens] -- each = block_idx * block_size + block_offset

### Suspects for decode failure
1. `input.rs::prepare_decode()` may compute wrong position_ids or slot_mapping for decode
2. `context_lens` may not include the new token being generated
3. `block_tables` flattening may have wrong max_blocks_per_seq for decode
4. The cache written during prefill may not be readable during decode (layout mismatch)
5. `flash_attention_2_decode_kernel` grid/block config may be wrong for num_seqs=1
6. `u32` vs `i32` type mismatch for metadata passed to kernels expecting `int*`

### Remote A100 access
- ssh -p 15882 root@ssh6.vast.ai
- Binary: /root/vllm-rs/target/release/rvllm
- Model: Qwen/Qwen2.5-1.5B (cached at /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/)

## Agent Assignments

### Agent 1: Decode metadata verification
**Goal**: Verify that `prepare_decode()` produces correct metadata for the decode step
**Files to READ**: `crates/rvllm-worker/src/input.rs`, `crates/rvllm-engine/src/gpu_engine.rs`
**Files to EDIT**: NONE (read-only investigation)
**Output**: Report whether position_ids, slot_mapping, context_lens, block_tables are correct for a decode step following a 5-token prefill

### Agent 2: Cache layout verification
**Goal**: Verify reshape_and_cache writes to the correct locations and FA2 decode reads from the correct locations
**Files to READ**: `kernels/reshape_and_cache.cu`, `kernels/flash_attention.cu`, `crates/rvllm-model-runner/src/gpu_layer.rs`
**Files to EDIT**: NONE
**Output**: Report whether the cache layout contract matches between reshape_and_cache (writer) and flash_attention_2_decode_kernel (reader)

### Agent 3: Type mismatch audit
**Goal**: Check every kernel launch in gpu_layer.rs for arg type mismatches (u32 vs i32, wrong pointer types)
**Files to READ**: `crates/rvllm-model-runner/src/gpu_layer.rs`, `kernels/*.cu`
**Files to EDIT**: NONE
**Output**: List of every type mismatch between Rust launch args and CUDA kernel signatures

### Agent 4: Decode attention probe
**Goal**: Add targeted probes for the FIRST decode step only (not prefill)
**Files to EDIT**: `crates/rvllm-model-runner/src/gpu_runner.rs` (probes only, no logic changes)
**Output**: Values at each stage of the first decode step:
  - token_id and position_id for the decode token
  - post-embedding values
  - post-QKV projection values
  - post-cache-write: readback of the slot just written
  - post-decode-attention output
  - final logits argmax

### Agent 5-20: Reserved for fixes based on findings from agents 1-4

## Rules
1. Agents 1-3 are READ-ONLY investigations
2. Agent 4 adds probes only, no logic changes
3. No agent touches: engine loop, async wrapper, scheduler, API routes
4. No agent does benchmarking
5. No agent changes kernel .cu files
6. Every agent reports exact file:line references
7. Findings go to stdout, not committed
