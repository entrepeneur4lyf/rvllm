# Spec 1: Automatic Prefix Caching (APC)

## Summary

When multiple requests share a prompt prefix (e.g., system prompt, few-shot examples), the KV cache blocks for that prefix can be reused instead of recomputed. This is called Automatic Prefix Caching (APC). vLLM implements this via chain hashing of block-aligned token sequences, an LRU eviction policy on freed blocks, and scheduler integration that skips already-computed tokens.

rvLLM has a `PrefixCache` struct and `BlockManager` integration, but the implementation has correctness issues and is not wired end-to-end. This spec brings it to parity with vLLM 0.18.0.

## vLLM Reference Behavior

### Chain Hashing

vLLM hashes each block independently using a chain: each block's hash depends on its parent block's hash plus the current block's token IDs. This ensures two sequences share a block only if their ENTIRE prefix up to that block boundary is identical, while keeping hash computation O(block_size) per block rather than O(prefix_length).

```
block_hash[0] = hash(SEED, tokens[0..block_size])
block_hash[1] = hash(block_hash[0], tokens[block_size..2*block_size])
block_hash[N] = hash(block_hash[N-1], tokens[N*block_size..(N+1)*block_size])
```

Only full blocks are hashed and cached. Partial blocks (the last block of a prompt that isn't full) are never cached.

### Block Lookup

On a new request, the scheduler computes block hashes incrementally, then walks the hash chain looking for cache hits. The walk stops at the first miss -- you can't skip a block in the middle because later blocks depend on earlier ones via the chain hash.

The hit count determines `num_computed_tokens`, which tells the model runner to skip those tokens during prefill.

### Eviction

Freed blocks go onto an LRU free list (doubly-linked list for O(1) operations). When a block is freed, it stays in the hash table -- it's only removed when its slot is reallocated for a new block. Cache hits call `touch()` to move the block out of the free list and increment its reference count.

### Key vLLM Files

- `vllm/v1/core/kv_cache_utils.py` -- `hash_block_tokens()`, `FreeKVCacheBlockQueue`, `KVCacheBlock`
- `vllm/v1/core/block_pool.py` -- `BlockPool`, `cache_full_blocks()`, `get_cached_block()`
- `vllm/v1/core/single_type_kv_cache_manager.py` -- `find_longest_cache_hit()`
- `vllm/v1/core/sched/scheduler.py` -- `get_computed_blocks()` call during scheduling

## Current rvLLM State

### What Exists

- `PrefixCache` in `crates/rvllm-block-manager/src/prefix_cache.rs` with `lookup()`, `insert()`, `release()`, `evict_one()`
- `BlockManager` integration: `allocate()` checks prefix cache, `register_prefix()` caches blocks after prefill
- Config flag: `CacheConfig.enable_prefix_caching` in `crates/rvllm-config/src/cache.rs`
- Engine creates `PrefixCache` if enabled, calls `count_hits()` during step

### What's Broken

1. **Wrong hash algorithm**: Hashes `tokens[0..(block_idx+1)*block_size]` (entire prefix, grows linearly) instead of chain-hashing each block independently. O(n^2) for long sequences.
2. **Cache hits are decoration**: `count_hits()` is logged but never used to skip computation. The model runner always recomputes the full prefill.
3. **Wrong block IDs on registration**: Engine calls `register_prefix_blocks()` with placeholder `BlockId(i as u32)` instead of the actual allocated block IDs from the block table.
4. **O(n) eviction**: Scans `HashMap` for minimum `last_access` instead of O(1) LRU list.
5. **No scheduler integration**: The scheduler has zero awareness of prefix caching.

## Implementation Plan

### Phase 1: Fix the Hash Algorithm

**File**: `crates/rvllm-block-manager/src/prefix_cache.rs`

Replace the current `hash_prefix()` with chain hashing:

```rust
pub fn hash_block(parent_hash: Option<PrefixHash>, block_tokens: &[TokenId]) -> PrefixHash {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    if let Some(ph) = parent_hash {
        ph.0.hash(&mut hasher);
    }
    block_tokens.hash(&mut hasher);
    PrefixHash(hasher.finish())
}
```

Add a method to compute all block hashes for a token sequence:

```rust
pub fn compute_block_hashes(&self, tokens: &[TokenId]) -> Vec<PrefixHash> {
    let mut hashes = Vec::new();
    let mut parent = None;
    for chunk in tokens.chunks_exact(self.block_size) {
        let h = Self::hash_block(parent, chunk);
        hashes.push(h);
        parent = Some(h);
    }
    hashes
}
```

Update `lookup()` to walk the hash chain and stop at the first miss, returning the count of consecutive cache hits.

### Phase 2: O(1) LRU Eviction

**File**: `crates/rvllm-block-manager/src/prefix_cache.rs`

Replace the `HashMap`-based eviction with a doubly-linked list. Use an intrusive list or a `VecDeque<BlockId>` with a position map for O(1) removal:

```rust
struct LruEntry {
    block_id: BlockId,
    hash: PrefixHash,
    prev: Option<usize>,
    next: Option<usize>,
}

struct LruList {
    entries: Vec<LruEntry>,      // arena-allocated nodes
    head: Option<usize>,         // LRU end (evict from here)
    tail: Option<usize>,         // MRU end (insert here)
    block_to_slot: HashMap<BlockId, usize>,
}
```

Operations:
- `push_back(block_id, hash)` -- add to MRU end (O(1))
- `pop_front()` -- evict from LRU end (O(1))
- `remove(block_id)` -- remove on cache hit / touch (O(1) via block_to_slot)

### Phase 3: Wire Cache Hits to Skip Computation

**Files**:
- `crates/rvllm-scheduler/src/scheduler.rs` -- add prefix cache awareness
- `crates/rvllm-sequence/src/sequence.rs` -- add `num_computed_tokens` field
- `crates/rvllm-engine/src/gpu_engine.rs` -- pass correct block IDs, use computed token count
- `crates/rvllm-worker/src/input.rs` -- adjust input preparation for partial prefill

This is the critical integration:

1. **Scheduler**: Before scheduling a new request, call `prefix_cache.lookup(block_hashes)` to get the number of cached blocks. Set `sequence.num_computed_tokens = cached_blocks * block_size`.

2. **Engine**: When building `SequenceGroupMetadata` for the worker, include `num_computed_tokens`. Pass the actual block IDs from `seq_block_tables` (not placeholders) when registering prefix blocks.

3. **Worker input preparation**: When `num_computed_tokens > 0`, the prefill input should only include tokens from `num_computed_tokens` onwards. Positions start at `num_computed_tokens`. The block table includes the cached blocks (already allocated) plus new blocks for remaining tokens.

4. **Block allocation**: `BlockManager::allocate()` should "adopt" cached blocks (increment ref count, add to the sequence's block table) rather than allocating new physical blocks for the cached portion.

### Phase 4: Correct Block Registration

**File**: `crates/rvllm-engine/src/gpu_engine.rs`

After a prefill completes, register the actual allocated block IDs:

```rust
// After worker returns, register newly-filled blocks in prefix cache
if let Some(ref mut pc) = self.prefix_cache {
    if let Some(block_table) = self.seq_block_tables.get(&seq_id) {
        let block_hashes = pc.compute_block_hashes(&prompt_tokens);
        for (hash, &block_id) in block_hashes.iter().zip(block_table.iter()) {
            pc.insert(*hash, block_id);
        }
    }
}
```

## Testing Strategy

1. **Hash correctness**: Two sequences with identical prefixes produce identical block hashes. Different prefixes produce different hashes. Chain property: changing token 0 changes ALL subsequent block hashes.
2. **Cache hit**: Submit request A, then request B with the same system prompt. Verify B's `num_computed_tokens` equals the shared prefix length (rounded down to block boundary).
3. **Eviction**: Fill the cache beyond `max_cached_blocks`, verify LRU blocks are evicted first, MRU blocks survive.
4. **Coherency**: Output from a cache-hit prefill must be identical to a full prefill. Test with diverse prompts.
5. **Ref counting**: Verify blocks in use by active sequences are never evicted.

## Files Changed

| File | Change |
|------|--------|
| `crates/rvllm-block-manager/src/prefix_cache.rs` | Rewrite hash algorithm, add LRU list, fix lookup/insert |
| `crates/rvllm-block-manager/src/manager.rs` | Update allocate() to adopt cached blocks |
| `crates/rvllm-scheduler/src/scheduler.rs` | Add prefix cache lookup before scheduling |
| `crates/rvllm-sequence/src/sequence.rs` | Ensure `num_computed_tokens` is set from cache hits |
| `crates/rvllm-engine/src/gpu_engine.rs` | Fix block ID registration, wire cache hit count |
| `crates/rvllm-worker/src/input.rs` | Adjust input prep for partial prefill (skip computed tokens) |

## Open Questions

- Should we support extra hash keys (LoRA adapter name, multimodal inputs) from the start, or add them later? Recommendation: add an `extra_keys: Option<&[u8]>` parameter to `hash_block()` now but don't use it yet.
- Should the prefix cache be per-engine or shared across engines (for tensor parallelism)? vLLM keeps it per-scheduler. Recommendation: per-engine for now.
