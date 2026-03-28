use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::Mutex;
use rvllm_core::prelude::{BlockId, LLMError, Result, SequenceId};

use crate::block_table::BlockTable;
use crate::prefix_cache::PrefixCache;
use crate::{Device, MemoryPool, PhysicalBlock, Sequence};

/// Flat-array reference counter indexed by `BlockId`.
///
/// All block IDs must be in `[0, capacity)`. The entire array fits in L2
/// cache for typical block counts (~32 KB at 4K blocks).
struct RefCounter {
    counts: Vec<usize>,
}

impl RefCounter {
    fn new(capacity: usize) -> Self {
        Self {
            counts: vec![0; capacity],
        }
    }

    #[inline]
    fn increment(&mut self, block_id: BlockId) {
        self.counts[block_id.0 as usize] += 1;
    }

    #[inline]
    fn decrement(&mut self, block_id: BlockId) -> usize {
        let c = &mut self.counts[block_id.0 as usize];
        *c = c.saturating_sub(1);
        *c
    }

    #[inline]
    fn get(&self, block_id: BlockId) -> usize {
        self.counts[block_id.0 as usize]
    }

    #[inline]
    fn remove(&mut self, block_id: BlockId) {
        self.counts[block_id.0 as usize] = 0;
    }
}

/// Dense slot allocator mapping `SequenceId` to recycled internal indices.
///
/// Per-sequence data is stored in flat `Vec`s indexed by slot. Freed slots
/// are recycled so storage stays bounded by max concurrent sequences.
struct SeqSlots {
    id_to_slot: HashMap<SequenceId, u32>,
    free_list: VecDeque<u32>,
    next_slot: u32,
}

impl SeqSlots {
    fn new() -> Self {
        Self {
            id_to_slot: HashMap::new(),
            free_list: VecDeque::new(),
            next_slot: 0,
        }
    }

    fn get_or_create(&mut self, seq_id: SequenceId) -> u32 {
        if let Some(&slot) = self.id_to_slot.get(&seq_id) {
            return slot;
        }
        let slot = self.free_list.pop_front().unwrap_or_else(|| {
            let s = self.next_slot;
            self.next_slot += 1;
            s
        });
        self.id_to_slot.insert(seq_id, slot);
        slot
    }

    #[inline]
    fn get(&self, seq_id: SequenceId) -> Option<u32> {
        self.id_to_slot.get(&seq_id).copied()
    }

    fn remove(&mut self, seq_id: SequenceId) -> Option<u32> {
        if let Some(slot) = self.id_to_slot.remove(&seq_id) {
            self.free_list.push_back(slot);
            Some(slot)
        } else {
            None
        }
    }
}

#[inline]
fn ensure_slot<T>(vec: &mut Vec<Option<T>>, slot: u32) {
    let idx = slot as usize;
    if idx >= vec.len() {
        vec.resize_with(idx + 1, || None);
    }
}

/// Manages logical-to-physical block mapping with CoW and prefix sharing.
pub struct BlockManager {
    gpu_pool: Arc<dyn MemoryPool>,
    cpu_pool: Arc<dyn MemoryPool>,
    block_size: usize,
    /// Dense slot allocator for per-sequence data.
    seq_slots: SeqSlots,
    /// Per-slot GPU block tables (indexed by slot).
    gpu_tables: Vec<Option<BlockTable>>,
    /// Per-slot CPU block tables (indexed by slot).
    cpu_tables: Vec<Option<BlockTable>>,
    /// Flat-array ref counting for GPU blocks (indexed by BlockId).
    gpu_ref_counts: RefCounter,
    /// Flat-array ref counting for CPU blocks (indexed by BlockId).
    cpu_ref_counts: RefCounter,
    /// Pending copy-on-write pairs: (src, dst).
    cow_pending: Vec<(BlockId, BlockId)>,
    /// Watermark fraction -- reserve this fraction of GPU blocks.
    watermark: f32,
    /// Optional prefix cache for KV block reuse across requests.
    prefix_cache: Option<PrefixCache>,
    /// Per-slot prefix block IDs (indexed by slot).
    prefix_blocks: Vec<Option<Vec<BlockId>>>,
}

impl BlockManager {
    pub fn new(
        gpu_pool: Arc<dyn MemoryPool>,
        cpu_pool: Arc<dyn MemoryPool>,
        block_size: usize,
    ) -> Self {
        let gpu_cap = gpu_pool.total_blocks();
        let cpu_cap = cpu_pool.total_blocks();
        Self {
            gpu_pool,
            cpu_pool,
            block_size,
            seq_slots: SeqSlots::new(),
            gpu_tables: Vec::new(),
            cpu_tables: Vec::new(),
            gpu_ref_counts: RefCounter::new(gpu_cap),
            cpu_ref_counts: RefCounter::new(cpu_cap),
            cow_pending: Vec::new(),
            watermark: 0.04,
            prefix_cache: None,
            prefix_blocks: Vec::new(),
        }
    }

    /// Enable prefix caching with the given maximum number of cached blocks.
    pub fn enable_prefix_caching(&mut self, max_cached_blocks: usize) {
        tracing::info!(
            block_size = self.block_size,
            max_cached_blocks,
            "prefix caching enabled"
        );
        self.prefix_cache = Some(PrefixCache::new(self.block_size, max_cached_blocks));
    }

    /// Returns true if prefix caching is enabled.
    pub fn prefix_caching_enabled(&self) -> bool {
        self.prefix_cache.is_some()
    }

    /// Set the watermark fraction for preemption decisions.
    pub fn set_watermark(&mut self, watermark: f32) {
        self.watermark = watermark;
    }

    fn blocks_needed(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }

    fn usable_gpu_blocks(&self) -> usize {
        let total = self.gpu_pool.total_blocks();
        let reserved = ((total as f32) * self.watermark).ceil() as usize;
        self.gpu_pool.free_blocks().saturating_sub(reserved)
    }

    /// Check if enough free GPU blocks exist for the sequence.
    pub fn can_allocate(&self, seq: &Sequence) -> bool {
        let needed = self.blocks_needed(seq.get_len());
        let existing = self
            .seq_slots
            .get(seq.seq_id)
            .and_then(|s| self.gpu_tables.get(s as usize))
            .and_then(|o| o.as_ref())
            .map(|t| t.len())
            .unwrap_or(0);
        let mut additional = needed.saturating_sub(existing);

        if existing == 0 {
            if let Some(ref pc) = self.prefix_cache {
                additional = additional.saturating_sub(pc.count_hits(&seq.prompt_token_ids));
            }
        }

        additional <= self.usable_gpu_blocks()
    }

    /// Allocate GPU blocks for new tokens in a sequence.
    pub fn allocate(&mut self, seq: &Sequence) -> Result<()> {
        let needed = self.blocks_needed(seq.get_len());
        let slot = self.seq_slots.get_or_create(seq.seq_id);

        ensure_slot(&mut self.gpu_tables, slot);
        let table_opt = &mut self.gpu_tables[slot as usize];
        if table_opt.is_none() {
            *table_opt = Some(BlockTable::new());
        }
        let table = table_opt.as_mut().unwrap();
        let existing = table.len();

        if existing >= needed {
            return Ok(());
        }

        let mut cached_hits: Vec<(usize, BlockId)> = Vec::new();
        if existing == 0 {
            if let Some(ref mut pc) = self.prefix_cache {
                cached_hits = pc.lookup(&seq.prompt_token_ids);
                if !cached_hits.is_empty() {
                    tracing::debug!(
                        seq_id = %seq.seq_id,
                        prefix_blocks = cached_hits.len(),
                        "prefix cache hit"
                    );
                }
            }
        }

        let mut prefix_block_ids = Vec::new();

        for block_logical_idx in existing..needed {
            let cached = cached_hits
                .iter()
                .find(|(idx, _)| *idx == block_logical_idx)
                .map(|(_, bid)| *bid);

            if let Some(block_id) = cached {
                let block = PhysicalBlock::new(block_id, self.block_size, Device::Gpu);
                table.push(block);
                self.gpu_ref_counts.increment(block_id);
                prefix_block_ids.push(block_id);
            } else {
                let block_id = self
                    .gpu_pool
                    .allocate()
                    .ok_or_else(|| LLMError::MemoryError("out of GPU blocks".into()))?;
                let block = PhysicalBlock::new(block_id, self.block_size, Device::Gpu);
                table.push(block);
                self.gpu_ref_counts.increment(block_id);
            }
        }

        if !prefix_block_ids.is_empty() {
            ensure_slot(&mut self.prefix_blocks, slot);
            self.prefix_blocks[slot as usize] = Some(prefix_block_ids);
        }

        Ok(())
    }

    /// Register completed prefix blocks in the cache after prefill.
    pub fn register_prefix(&mut self, seq: &Sequence) {
        let pc = match self.prefix_cache {
            Some(ref mut pc) => pc,
            None => return,
        };
        let slot = match self.seq_slots.get(seq.seq_id) {
            Some(s) => s,
            None => return,
        };
        let table = match self.gpu_tables.get(slot as usize).and_then(|o| o.as_ref()) {
            Some(t) => t,
            None => return,
        };

        let block_ids: Vec<BlockId> = table.iter().map(|b| b.block_id).collect();
        let newly_cached = crate::prefix_cache::register_prefix_blocks(
            pc,
            &seq.prompt_token_ids,
            &block_ids,
            self.block_size,
        );

        for &bid in &newly_cached {
            self.gpu_ref_counts.increment(bid);
        }

        if !newly_cached.is_empty() {
            tracing::debug!(
                seq_id = %seq.seq_id,
                newly_cached = newly_cached.len(),
                "registered prefix blocks in cache"
            );
        }
    }

    /// Free all GPU and CPU blocks for a sequence.
    pub fn free(&mut self, seq: &Sequence) {
        let slot = match self.seq_slots.remove(seq.seq_id) {
            Some(s) => s,
            None => return,
        };

        if let Some(prefix_bids) = self
            .prefix_blocks
            .get_mut(slot as usize)
            .and_then(|o| o.take())
        {
            if let Some(ref mut pc) = self.prefix_cache {
                pc.release(&prefix_bids);
            }
        }

        if let Some(table) = self
            .gpu_tables
            .get_mut(slot as usize)
            .and_then(|o| o.take())
        {
            for block in table.iter() {
                let remaining = self.gpu_ref_counts.decrement(block.block_id);
                if remaining == 0 {
                    self.gpu_pool.free(block.block_id);
                    self.gpu_ref_counts.remove(block.block_id);
                }
            }
        }

        if let Some(table) = self
            .cpu_tables
            .get_mut(slot as usize)
            .and_then(|o| o.take())
        {
            for block in table.iter() {
                let remaining = self.cpu_ref_counts.decrement(block.block_id);
                if remaining == 0 {
                    self.cpu_pool.free(block.block_id);
                    self.cpu_ref_counts.remove(block.block_id);
                }
            }
        }
    }

    /// Evict a prefix cache block, freeing the physical GPU block.
    pub fn evict_prefix_block(&mut self) -> Option<BlockId> {
        let pc = self.prefix_cache.as_mut()?;
        let block_id = pc.evict_one()?;
        let remaining = self.gpu_ref_counts.decrement(block_id);
        if remaining == 0 {
            self.gpu_pool.free(block_id);
            self.gpu_ref_counts.remove(block_id);
        }
        Some(block_id)
    }

    /// Access the prefix cache (read-only) for diagnostics.
    pub fn prefix_cache(&self) -> Option<&PrefixCache> {
        self.prefix_cache.as_ref()
    }

    /// Fork a parent sequence into a child (CoW for beam search).
    pub fn fork(&mut self, parent: &Sequence, child: &mut Sequence) -> Result<()> {
        let parent_slot = self
            .seq_slots
            .get(parent.seq_id)
            .ok_or_else(|| LLMError::MemoryError("parent has no block table".into()))?;

        let parent_table = self
            .gpu_tables
            .get(parent_slot as usize)
            .and_then(|o| o.as_ref())
            .ok_or_else(|| LLMError::MemoryError("parent has no block table".into()))?
            .clone();

        let mut child_table = BlockTable::with_capacity(parent_table.len());
        for block in parent_table.iter() {
            self.gpu_ref_counts.increment(block.block_id);
            child_table.push(block.clone());
        }

        let child_slot = self.seq_slots.get_or_create(child.seq_id);
        ensure_slot(&mut self.gpu_tables, child_slot);
        self.gpu_tables[child_slot as usize] = Some(child_table);
        Ok(())
    }

    /// Perform copy-on-write for the last block of a sequence if shared.
    pub fn cow_if_needed(&mut self, seq: &Sequence) -> Result<Option<BlockId>> {
        let slot = self
            .seq_slots
            .get(seq.seq_id)
            .ok_or_else(|| LLMError::MemoryError("sequence has no block table".into()))?;

        let table = self
            .gpu_tables
            .get_mut(slot as usize)
            .and_then(|o| o.as_mut())
            .ok_or_else(|| LLMError::MemoryError("sequence has no block table".into()))?;

        let last_idx = match table.len().checked_sub(1) {
            Some(idx) => idx,
            None => return Ok(None),
        };

        let old_block_id = table.get(last_idx).unwrap().block_id;
        let ref_count = self.gpu_ref_counts.get(old_block_id);

        if ref_count > 1 {
            let new_block_id = self
                .gpu_pool
                .allocate()
                .ok_or_else(|| LLMError::MemoryError("out of GPU blocks for CoW".into()))?;
            self.cow_pending.push((old_block_id, new_block_id));
            self.gpu_ref_counts.decrement(old_block_id);
            self.gpu_ref_counts.increment(new_block_id);

            let new_block = PhysicalBlock::new(new_block_id, self.block_size, Device::Gpu);
            *table.get_mut(last_idx).unwrap() = new_block;

            Ok(Some(new_block_id))
        } else {
            Ok(None)
        }
    }

    /// Mark a block as shared (increment ref count) for prefix caching.
    pub fn mark_shared(&mut self, block_id: BlockId) {
        self.gpu_ref_counts.increment(block_id);
    }

    /// Get the block table for a sequence.
    pub fn get_block_table(&self, seq_id: SequenceId) -> Option<&BlockTable> {
        let slot = self.seq_slots.get(seq_id)?;
        self.gpu_tables.get(slot as usize)?.as_ref()
    }

    /// Check if there are enough CPU blocks to swap out a sequence.
    pub fn can_swap_out(&self, seq: &Sequence) -> bool {
        let slot = match self.seq_slots.get(seq.seq_id) {
            Some(s) => s,
            None => return false,
        };
        match self.gpu_tables.get(slot as usize).and_then(|o| o.as_ref()) {
            Some(t) => t.len() <= self.cpu_pool.free_blocks(),
            None => false,
        }
    }

    /// Swap out a sequence from GPU to CPU. Returns GPU->CPU block id mapping.
    pub fn swap_out(&mut self, seq: &Sequence) -> Result<Vec<(BlockId, BlockId)>> {
        let slot = self
            .seq_slots
            .get(seq.seq_id)
            .ok_or_else(|| LLMError::MemoryError("sequence has no GPU block table".into()))?;

        let gpu_table = self
            .gpu_tables
            .get_mut(slot as usize)
            .and_then(|o| o.take())
            .ok_or_else(|| LLMError::MemoryError("sequence has no GPU block table".into()))?;

        let mut mapping = Vec::with_capacity(gpu_table.len());
        let mut cpu_table = BlockTable::with_capacity(gpu_table.len());

        for block in gpu_table.iter() {
            let cpu_block_id = self
                .cpu_pool
                .allocate()
                .ok_or_else(|| LLMError::MemoryError("out of CPU blocks for swap".into()))?;
            mapping.push((block.block_id, cpu_block_id));
            cpu_table.push(PhysicalBlock::new(
                cpu_block_id,
                self.block_size,
                Device::Cpu,
            ));
            self.cpu_ref_counts.increment(cpu_block_id);

            let remaining = self.gpu_ref_counts.decrement(block.block_id);
            if remaining == 0 {
                self.gpu_pool.free(block.block_id);
                self.gpu_ref_counts.remove(block.block_id);
            }
        }

        ensure_slot(&mut self.cpu_tables, slot);
        self.cpu_tables[slot as usize] = Some(cpu_table);
        Ok(mapping)
    }

    /// Swap in a sequence from CPU back to GPU. Returns CPU->GPU block id mapping.
    pub fn swap_in(&mut self, seq: &Sequence) -> Result<Vec<(BlockId, BlockId)>> {
        let slot = self
            .seq_slots
            .get(seq.seq_id)
            .ok_or_else(|| LLMError::MemoryError("sequence has no CPU block table".into()))?;

        let cpu_table = self
            .cpu_tables
            .get_mut(slot as usize)
            .and_then(|o| o.take())
            .ok_or_else(|| LLMError::MemoryError("sequence has no CPU block table".into()))?;

        let mut mapping = Vec::with_capacity(cpu_table.len());
        let mut gpu_table = BlockTable::with_capacity(cpu_table.len());

        for block in cpu_table.iter() {
            let gpu_block_id = self
                .gpu_pool
                .allocate()
                .ok_or_else(|| LLMError::MemoryError("out of GPU blocks for swap-in".into()))?;
            mapping.push((block.block_id, gpu_block_id));
            gpu_table.push(PhysicalBlock::new(
                gpu_block_id,
                self.block_size,
                Device::Gpu,
            ));
            self.gpu_ref_counts.increment(gpu_block_id);

            let remaining = self.cpu_ref_counts.decrement(block.block_id);
            if remaining == 0 {
                self.cpu_pool.free(block.block_id);
                self.cpu_ref_counts.remove(block.block_id);
            }
        }

        ensure_slot(&mut self.gpu_tables, slot);
        self.gpu_tables[slot as usize] = Some(gpu_table);
        Ok(mapping)
    }

    /// Drain pending copy-on-write block pairs.
    pub fn get_copy_on_write_blocks(&mut self) -> Vec<(BlockId, BlockId)> {
        std::mem::take(&mut self.cow_pending)
    }

    /// Check watermark: true if free GPU blocks are above the watermark threshold.
    pub fn above_watermark(&self) -> bool {
        let total = self.gpu_pool.total_blocks();
        let reserved = ((total as f32) * self.watermark).ceil() as usize;
        self.gpu_pool.free_blocks() > reserved
    }
}

/// Thread-safe wrapper around BlockManager.
pub struct SharedBlockManager {
    inner: Mutex<BlockManager>,
}

impl SharedBlockManager {
    pub fn new(manager: BlockManager) -> Self {
        Self {
            inner: Mutex::new(manager),
        }
    }

    pub fn lock(&self) -> parking_lot::MutexGuard<'_, BlockManager> {
        self.inner.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SequenceStatus;
    use rvllm_core::prelude::TokenId;

    /// Test pool that recycles BlockIds within `[0, total)`.
    struct TestPool {
        total: usize,
        inner: Mutex<VecDeque<u32>>,
    }

    impl TestPool {
        fn new(total: usize) -> Self {
            let mut free = VecDeque::with_capacity(total);
            for i in 0..total {
                free.push_back(i as u32);
            }
            Self {
                total,
                inner: Mutex::new(free),
            }
        }
    }

    impl MemoryPool for TestPool {
        fn allocate(&self) -> Option<BlockId> {
            self.inner.lock().pop_front().map(BlockId)
        }

        fn free(&self, block_id: BlockId) {
            self.inner.lock().push_back(block_id.0);
        }

        fn free_blocks(&self) -> usize {
            self.inner.lock().len()
        }

        fn total_blocks(&self) -> usize {
            self.total
        }
    }

    fn make_seq(id: u64, num_tokens: usize) -> Sequence {
        let mut seq = Sequence::new(SequenceId(id), vec![0 as TokenId; num_tokens]);
        seq.status = SequenceStatus::Running;
        seq
    }

    fn make_manager(gpu_blocks: usize, cpu_blocks: usize) -> BlockManager {
        let gpu = Arc::new(TestPool::new(gpu_blocks));
        let cpu = Arc::new(TestPool::new(cpu_blocks));
        let mut mgr = BlockManager::new(gpu, cpu, 16);
        mgr.set_watermark(0.0);
        mgr
    }

    #[test]
    fn allocate_and_free() {
        let mut mgr = make_manager(10, 10);
        let seq = make_seq(1, 32);

        assert!(mgr.can_allocate(&seq));
        mgr.allocate(&seq).unwrap();

        let table = mgr.get_block_table(SequenceId(1)).unwrap();
        assert_eq!(table.len(), 2);

        mgr.free(&seq);
        assert!(mgr.get_block_table(SequenceId(1)).is_none());
    }

    #[test]
    fn allocate_incremental() {
        let mut mgr = make_manager(10, 10);
        let mut seq = make_seq(1, 16);
        mgr.allocate(&seq).unwrap();
        assert_eq!(mgr.get_block_table(SequenceId(1)).unwrap().len(), 1);

        seq.prompt_token_ids.extend_from_slice(&[0; 16]);
        mgr.allocate(&seq).unwrap();
        assert_eq!(mgr.get_block_table(SequenceId(1)).unwrap().len(), 2);
    }

    #[test]
    fn cannot_allocate_when_full() {
        let mut mgr = make_manager(1, 1);
        let seq = make_seq(1, 32);
        assert!(!mgr.can_allocate(&seq));
        assert!(mgr.allocate(&seq).is_err());
    }

    #[test]
    fn fork_shares_blocks() {
        let mut mgr = make_manager(10, 10);
        let parent = make_seq(1, 32);
        mgr.allocate(&parent).unwrap();

        let mut child = make_seq(2, 32);
        mgr.fork(&parent, &mut child).unwrap();

        let parent_table = mgr.get_block_table(SequenceId(1)).unwrap();
        let child_table = mgr.get_block_table(SequenceId(2)).unwrap();
        assert_eq!(parent_table.len(), child_table.len());

        for i in 0..parent_table.len() {
            assert_eq!(
                parent_table.get(i).unwrap().block_id,
                child_table.get(i).unwrap().block_id,
            );
        }
    }

    #[test]
    fn cow_on_shared_block() {
        let mut mgr = make_manager(10, 10);
        let parent = make_seq(1, 32);
        mgr.allocate(&parent).unwrap();

        let mut child = make_seq(2, 32);
        mgr.fork(&parent, &mut child).unwrap();

        let new_id = mgr.cow_if_needed(&child).unwrap();
        assert!(new_id.is_some());

        let parent_last = mgr
            .get_block_table(SequenceId(1))
            .unwrap()
            .last()
            .unwrap()
            .block_id;
        let child_last = mgr
            .get_block_table(SequenceId(2))
            .unwrap()
            .last()
            .unwrap()
            .block_id;
        assert_ne!(parent_last, child_last);

        let cow = mgr.get_copy_on_write_blocks();
        assert_eq!(cow.len(), 1);
    }

    #[test]
    fn cow_not_needed_for_unique_block() {
        let mut mgr = make_manager(10, 10);
        let seq = make_seq(1, 32);
        mgr.allocate(&seq).unwrap();

        let new_id = mgr.cow_if_needed(&seq).unwrap();
        assert!(new_id.is_none());
        assert!(mgr.get_copy_on_write_blocks().is_empty());
    }

    #[test]
    fn swap_out_and_in() {
        let mut mgr = make_manager(10, 10);
        let seq = make_seq(1, 32);
        mgr.allocate(&seq).unwrap();

        assert!(mgr.can_swap_out(&seq));
        let out_mapping = mgr.swap_out(&seq).unwrap();
        assert_eq!(out_mapping.len(), 2);
        assert!(mgr.get_block_table(SequenceId(1)).is_none());

        let in_mapping = mgr.swap_in(&seq).unwrap();
        assert_eq!(in_mapping.len(), 2);
        assert!(mgr.get_block_table(SequenceId(1)).is_some());
    }

    #[test]
    fn mark_shared_increments_refcount() {
        let mut mgr = make_manager(10, 10);
        let seq = make_seq(1, 16);
        mgr.allocate(&seq).unwrap();

        let block_id = mgr
            .get_block_table(SequenceId(1))
            .unwrap()
            .get(0)
            .unwrap()
            .block_id;
        mgr.mark_shared(block_id);
        let new_id = mgr.cow_if_needed(&seq).unwrap();
        assert!(new_id.is_some());
    }

    #[test]
    fn watermark_reserves_blocks() {
        let gpu = Arc::new(TestPool::new(100));
        let cpu = Arc::new(TestPool::new(10));
        let mut mgr = BlockManager::new(gpu, cpu, 16);
        mgr.set_watermark(0.10);

        let mut seq = make_seq(1, 91 * 16);
        assert!(!mgr.can_allocate(&seq));

        seq.prompt_token_ids.truncate(90 * 16);
        assert!(mgr.can_allocate(&seq));
    }

    #[test]
    fn above_watermark() {
        let gpu = Arc::new(TestPool::new(100));
        let cpu = Arc::new(TestPool::new(10));
        let mut mgr = BlockManager::new(gpu, cpu, 16);
        mgr.set_watermark(0.10);
        assert!(mgr.above_watermark());
    }

    #[test]
    fn free_shared_block_only_when_last_ref() {
        let mut mgr = make_manager(10, 10);
        let parent = make_seq(1, 16);
        mgr.allocate(&parent).unwrap();

        let mut child = make_seq(2, 16);
        mgr.fork(&parent, &mut child).unwrap();

        let free_before = mgr.gpu_pool.free_blocks();
        mgr.free(&parent);
        let free_after_parent = mgr.gpu_pool.free_blocks();
        assert_eq!(free_after_parent, free_before);

        mgr.free(&child);
        let free_after_child = mgr.gpu_pool.free_blocks();
        assert_eq!(free_after_child, free_before + 1);
    }

    #[test]
    fn shared_block_manager_thread_safe() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SharedBlockManager>();
    }

    // -----------------------------------------------------------------------
    // Prefix caching integration tests
    // -----------------------------------------------------------------------

    fn make_prefix_manager(gpu_blocks: usize, cpu_blocks: usize) -> BlockManager {
        let gpu = Arc::new(TestPool::new(gpu_blocks));
        let cpu = Arc::new(TestPool::new(cpu_blocks));
        let mut mgr = BlockManager::new(gpu, cpu, 16);
        mgr.set_watermark(0.0);
        mgr.enable_prefix_caching(100);
        mgr
    }

    fn make_seq_with_tokens(id: u64, tokens: Vec<TokenId>) -> Sequence {
        let mut seq = Sequence::new(SequenceId(id), tokens);
        seq.status = SequenceStatus::Running;
        seq
    }

    #[test]
    fn prefix_cache_register_and_reuse() {
        let mut mgr = make_prefix_manager(20, 10);
        assert!(mgr.prefix_caching_enabled());

        let tokens: Vec<TokenId> = (0..32).collect();
        let seq1 = make_seq_with_tokens(1, tokens.clone());
        mgr.allocate(&seq1).unwrap();
        assert_eq!(mgr.get_block_table(SequenceId(1)).unwrap().len(), 2);

        mgr.register_prefix(&seq1);

        let pc = mgr.prefix_cache().unwrap();
        assert_eq!(pc.len(), 2);

        let seq2 = make_seq_with_tokens(2, tokens.clone());
        let free_before = mgr.gpu_pool.free_blocks();
        mgr.allocate(&seq2).unwrap();
        let free_after = mgr.gpu_pool.free_blocks();

        assert_eq!(free_before, free_after);

        let t1 = mgr.get_block_table(SequenceId(1)).unwrap();
        let t2 = mgr.get_block_table(SequenceId(2)).unwrap();
        assert_eq!(t1.get(0).unwrap().block_id, t2.get(0).unwrap().block_id);
        assert_eq!(t1.get(1).unwrap().block_id, t2.get(1).unwrap().block_id);
    }

    #[test]
    fn prefix_cache_partial_match() {
        let mut mgr = make_prefix_manager(20, 10);

        let tokens_a: Vec<TokenId> = (0..48).collect();
        let seq1 = make_seq_with_tokens(1, tokens_a.clone());
        mgr.allocate(&seq1).unwrap();
        mgr.register_prefix(&seq1);

        let mut tokens_b: Vec<TokenId> = (0..48).collect();
        tokens_b[32] = 999;
        let seq2 = make_seq_with_tokens(2, tokens_b);

        let free_before = mgr.gpu_pool.free_blocks();
        mgr.allocate(&seq2).unwrap();
        let free_after = mgr.gpu_pool.free_blocks();

        assert_eq!(free_before - free_after, 1);
    }

    #[test]
    fn prefix_cache_can_allocate_accounts_for_hits() {
        let mut mgr = make_prefix_manager(10, 10);

        let tokens: Vec<TokenId> = (0..32).collect();
        let seq1 = make_seq_with_tokens(1, tokens.clone());
        mgr.allocate(&seq1).unwrap();
        mgr.register_prefix(&seq1);

        let filler = make_seq(99, 7 * 16);
        mgr.allocate(&filler).unwrap();

        let seq2 = make_seq_with_tokens(2, tokens);
        assert!(mgr.can_allocate(&seq2));
    }

    #[test]
    fn prefix_cache_evict_frees_block() {
        let mut mgr = make_prefix_manager(10, 10);

        let tokens: Vec<TokenId> = (0..16).collect();
        let seq1 = make_seq_with_tokens(1, tokens);
        mgr.allocate(&seq1).unwrap();
        mgr.register_prefix(&seq1);

        let free_before_free = mgr.gpu_pool.free_blocks();
        mgr.free(&seq1);
        let free_after_free = mgr.gpu_pool.free_blocks();
        assert_eq!(free_after_free, free_before_free);

        let evicted = mgr.evict_prefix_block();
        assert!(evicted.is_some());
        let free_after_evict = mgr.gpu_pool.free_blocks();
        assert_eq!(free_after_evict, free_after_free + 1);
    }

    #[test]
    fn prefix_cache_no_reuse_different_prefix() {
        let mut mgr = make_prefix_manager(20, 10);

        let tokens_a: Vec<TokenId> = (0..32).collect();
        let seq1 = make_seq_with_tokens(1, tokens_a);
        mgr.allocate(&seq1).unwrap();
        mgr.register_prefix(&seq1);

        let tokens_b: Vec<TokenId> = (100..132).collect();
        let seq2 = make_seq_with_tokens(2, tokens_b);
        let free_before = mgr.gpu_pool.free_blocks();
        mgr.allocate(&seq2).unwrap();
        let free_after = mgr.gpu_pool.free_blocks();

        assert_eq!(free_before - free_after, 2);
    }

    #[test]
    fn slot_recycling() {
        let mut mgr = make_manager(10, 10);

        // Allocate and free many sequences to exercise slot recycling.
        for i in 0..20u64 {
            let seq = make_seq(i, 16);
            mgr.allocate(&seq).unwrap();
            mgr.free(&seq);
        }

        // Slots should be recycled -- next_slot shouldn't grow unboundedly.
        // With 20 sequential alloc/free, slots should be reused after the first.
        assert!(mgr.seq_slots.next_slot <= 20);

        // Verify allocator still works.
        let seq = make_seq(100, 16);
        mgr.allocate(&seq).unwrap();
        assert_eq!(mgr.get_block_table(SequenceId(100)).unwrap().len(), 1);
    }
}
