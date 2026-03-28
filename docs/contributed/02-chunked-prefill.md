# Spec 2: Chunked Prefill

## Summary

Long prompts (thousands of tokens) block the entire GPU for a single prefill step, starving decode requests of forward passes and spiking latency. Chunked prefill splits long prompts into fixed-size chunks and interleaves them with decode steps, keeping latency bounded while maintaining throughput.

rvLLM's scheduler already has chunked prefill scaffolding (`max_prefill_chunk`, `num_prompt_tokens_processed`, `token_chunk_size`). The scheduler correctly computes chunk sizes and advances progress across steps. What's missing is the downstream wiring: the worker/model runner doesn't use `token_chunk_size` to build partial prefill inputs, and attention doesn't handle mixed prefill+decode batches correctly when a request is mid-prefill.

## vLLM Reference Behavior

### Unified Token Budget

vLLM's v1 scheduler doesn't distinguish "prefill phase" from "decode phase." Each step has a token budget (`max_num_batched_tokens`). Every request contributes `num_new_tokens = num_tokens - num_computed_tokens`, clamped by the remaining budget. This naturally handles:
- Full prefills (new request, all tokens are new)
- Chunked prefills (long request, only chunk_size tokens fit in budget)
- Decode (1 new token per request)
- Mixed batches (some requests prefilling, others decoding, in the same step)

### Chunk Size Control

- `long_prefill_token_threshold`: max tokens per prefill chunk (default: ~4% of `max_model_len`)
- `max_num_partial_prefills`: limit on concurrent partially-prefilled requests (default: 1)
- If chunked prefill is disabled and a prefill exceeds budget, the scheduler stops (doesn't schedule it partially)

### Mixed Batch Attention

The attention backend splits the batch into decode tokens (query_len=1) and prefill tokens (query_len>1). Decode tokens come first. FlashAttention uses different kernel paths for each group. The key insight: `seq_lens` for a chunked prefill request equals `num_computed_tokens + chunk_size` (the full context seen so far), while `query_lens` equals only the new chunk size.

### Output Suppression

Partially-prefilled requests don't produce output tokens. vLLM tracks `is_prefill_chunk = (num_computed_tokens < num_tokens)` and discards sampled tokens for incomplete prefills.

### Key vLLM Files

- `vllm/v1/core/sched/scheduler.py` -- token budget logic, `_update_after_schedule()`
- `vllm/v1/worker/gpu_model_runner.py` -- position computation, `query_start_loc`, `seq_lens`
- `vllm/v1/attention/backends/utils.py` -- `split_decodes_and_prefills()`

## Current rvLLM State

### What Works

The scheduler in `rvllm-scheduler` correctly handles chunked prefill:

```rust
// SchedulerConfig
pub max_prefill_chunk: usize,      // analogous to long_prefill_token_threshold

// SequenceGroup
pub num_prompt_tokens_processed: usize,  // tracks chunk progress

// tokens_for_group() correctly computes chunk sizes:
fn tokens_for_group(&self, group: &SequenceGroup) -> usize {
    let remaining = group.remaining_prefill();
    if remaining > 0 {
        if self.config.max_prefill_chunk > 0 {
            remaining.min(self.config.max_prefill_chunk)
        } else {
            remaining
        }
    } else {
        group.num_active()
    }
}
```

`SchedulerOutputs` includes `token_chunk_size` per scheduled group. Tests exist: `chunked_prefill_splits_long_prompt`, `chunked_prefill_progresses_over_steps`.

### What's Missing

1. **Worker input preparation doesn't use `token_chunk_size`**: The worker always builds input for the full prompt or 1 decode token. It doesn't handle "tokens 512..1024 of a 4096-token prompt."
2. **Positions are wrong for chunks**: A second chunk should have positions starting at `num_computed_tokens`, not 0.
3. **Attention metadata for mid-prefill**: `seq_lens` should be `num_computed_tokens + chunk_size`, not just `chunk_size`. The KV cache already has the earlier chunks' data.
4. **Slot mapping for chunks**: New tokens in a chunk need slots in the KV cache starting at position `num_computed_tokens`, not position 0.
5. **Output suppression**: No mechanism to discard sampled tokens for incomplete prefills.
6. **Mixed batch splitting**: Attention backend doesn't split decode vs prefill tokens within a single batch.

## Implementation Plan

### Phase 1: Worker Input Preparation for Chunks

**Files**: `crates/rvllm-worker/src/input.rs`, `crates/rvllm-engine/src/gpu_engine.rs`

When `SequenceGroupMetadata` indicates a chunked prefill (`token_chunk_size < prompt_len`):

1. **Token IDs**: Use `prompt_tokens[num_computed..num_computed+chunk_size]` instead of the full prompt.
2. **Positions**: Generate `[num_computed, num_computed+1, ..., num_computed+chunk_size-1]`.
3. **Context length**: Set to `num_computed + chunk_size` (total context, including earlier chunks in KV cache).
4. **Slot mapping**: Compute slots for positions `num_computed..num_computed+chunk_size`.

The engine must pass `num_computed_tokens` and `token_chunk_size` through `SequenceGroupMetadata` to the worker.

### Phase 2: Attention Metadata for Partial Prefills

**Files**: `crates/rvllm-attention/src/metadata.rs`, `crates/rvllm-worker/src/input.rs`

`AttentionMetadata` needs to distinguish `query_lens` from `seq_lens`:

```rust
pub struct AttentionMetadata {
    pub query_lens: Vec<usize>,    // tokens being computed THIS step (chunk_size for prefill, 1 for decode)
    pub seq_lens: Vec<usize>,      // total context length (num_computed + chunk_size)
    pub seq_start_pos: Vec<usize>, // prefix sum of query_lens (for FlashAttention)
    // ... existing fields
}
```

For a mixed batch with 3 decode requests and 1 chunked prefill (chunk_size=256, total context=768):
- `query_lens = [1, 1, 1, 256]`
- `seq_lens = [100, 50, 75, 768]`
- `seq_start_pos = [0, 1, 2, 3, 259]`

### Phase 3: Output Suppression for Incomplete Prefills

**Files**: `crates/rvllm-engine/src/gpu_engine.rs`, `crates/rvllm-engine/src/output.rs`

After the worker returns sampled tokens, the engine must check whether each request's prefill is complete:

```rust
// In step(), after sampling:
for output in &worker_outputs {
    let group = &scheduled_groups[&output.request_id];
    if group.is_prefilling() {
        // Don't emit this output to the client -- prefill is incomplete
        // Advance num_prompt_tokens_processed instead
        continue;
    }
    // Normal output handling
}
```

### Phase 4: Block Allocation for Chunks

**Files**: `crates/rvllm-block-manager/src/manager.rs`, `crates/rvllm-engine/src/gpu_engine.rs`

Blocks must be allocated incrementally:
- On the first chunk, allocate blocks for `ceil(chunk_size / block_size)` blocks.
- On subsequent chunks, allocate additional blocks only if needed (new tokens cross a block boundary).
- The block table grows incrementally across chunks.

The engine tracks the block table per sequence and extends it as chunks are processed.

## Testing Strategy

1. **Basic chunked prefill**: Submit a 2048-token prompt with `max_prefill_chunk=512`. Verify it takes 4 steps to complete prefill, then decode starts.
2. **Position correctness**: Verify that chunk 2 gets positions [512, 513, ..., 1023], not [0, 1, ..., 511].
3. **Mixed batch**: While a long prompt is chunked-prefilling, submit short requests. Verify the short requests decode normally alongside the chunks.
4. **Output suppression**: Verify no output tokens are emitted during prefill chunks, only after the last chunk.
5. **Coherency**: Output from chunked prefill must be identical to non-chunked prefill for the same prompt.
6. **KV cache correctness**: After chunked prefill, the KV cache must contain the same data as a full prefill. Test by comparing attention outputs.

## Files Changed

| File | Change |
|------|--------|
| `crates/rvllm-worker/src/input.rs` | Handle `num_computed_tokens` and `token_chunk_size` in input preparation |
| `crates/rvllm-attention/src/metadata.rs` | Ensure `query_lens` vs `seq_lens` distinction |
| `crates/rvllm-engine/src/gpu_engine.rs` | Pass chunk metadata to worker, suppress partial prefill outputs |
| `crates/rvllm-engine/src/output.rs` | Filter outputs for incomplete prefills |
| `crates/rvllm-block-manager/src/manager.rs` | Incremental block allocation across chunks |
| `crates/rvllm-sequence/src/sequence.rs` | Ensure `num_computed_tokens` is advanced per chunk |

## Dependencies

- None (can be implemented independently of prefix caching)
- Pairs well with prefix caching: cached prefix blocks reduce the first chunk

## Open Questions

- Should rvLLM default to chunked prefill enabled or disabled? vLLM defaults to enabled in v1. Recommendation: disabled by default for now (`max_prefill_chunk = 0` means no chunking).
- Should we limit concurrent partial prefills (`max_num_partial_prefills`)? Recommendation: yes, default to 1, matching vLLM.
