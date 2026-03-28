use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use parking_lot::Mutex;
use rvllm_block_manager::{BlockManager, MemoryPool};
use rvllm_core::prelude::{BlockId, SequenceId, TokenId};
use rvllm_sequence::{Sequence, SequenceStatus};

// ---------------------------------------------------------------------------
// Pool that recycles BlockIds in [0, total)
// ---------------------------------------------------------------------------

struct BenchPool {
    total: usize,
    inner: Mutex<VecDeque<u32>>,
}

impl BenchPool {
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

impl MemoryPool for BenchPool {
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

// ---------------------------------------------------------------------------
// Old HashMap-based RefCounter for comparison
// ---------------------------------------------------------------------------

struct OldRefCounter {
    counts: HashMap<BlockId, usize>,
}

impl OldRefCounter {
    fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }
    #[inline]
    fn increment(&mut self, block_id: BlockId) {
        *self.counts.entry(block_id).or_insert(0) += 1;
    }
    #[inline]
    fn decrement(&mut self, block_id: BlockId) -> usize {
        let count = self.counts.entry(block_id).or_insert(1);
        *count = count.saturating_sub(1);
        *count
    }
    #[inline]
    fn get(&self, block_id: BlockId) -> usize {
        self.counts.get(&block_id).copied().unwrap_or(0)
    }
}

struct NewRefCounter {
    counts: Vec<usize>,
}

impl NewRefCounter {
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
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_ref_counter(c: &mut Criterion) {
    let mut group = c.benchmark_group("ref_counter");

    for num_blocks in [256, 1024, 4096] {
        // Setup: populate both counters with the same data
        let block_ids: Vec<BlockId> = (0..num_blocks).map(|i| BlockId(i as u32)).collect();

        group.bench_with_input(
            BenchmarkId::new("hashmap/increment+get", num_blocks),
            &num_blocks,
            |b, &n| {
                b.iter(|| {
                    let mut rc = OldRefCounter::new();
                    for &bid in &block_ids {
                        rc.increment(bid);
                    }
                    let mut sum = 0usize;
                    for &bid in &block_ids {
                        sum += rc.get(bid);
                    }
                    black_box(sum)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("flat_vec/increment+get", num_blocks),
            &num_blocks,
            |b, &n| {
                b.iter(|| {
                    let mut rc = NewRefCounter::new(n);
                    for &bid in &block_ids {
                        rc.increment(bid);
                    }
                    let mut sum = 0usize;
                    for &bid in &block_ids {
                        sum += rc.get(bid);
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

fn bench_cow_hot_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("cow_hot_path");

    // Simulate the decode-step hot path: N sequences, each calling
    // cow_if_needed which does 1 seq table lookup + 1 ref count check.
    for num_seqs in [64, 256, 512] {
        let block_size = 16;
        let tokens_per_seq = 64 * block_size; // 64 blocks per seq
        let blocks_per_seq = 64;
        let total_blocks = num_seqs * blocks_per_seq + 512; // headroom

        group.bench_with_input(
            BenchmarkId::new("allocate_cow_free", num_seqs),
            &num_seqs,
            |b, &n| {
                b.iter(|| {
                    let gpu = Arc::new(BenchPool::new(total_blocks));
                    let cpu = Arc::new(BenchPool::new(64));
                    let mut mgr = BlockManager::new(gpu, cpu, block_size);
                    mgr.set_watermark(0.0);

                    let seqs: Vec<Sequence> =
                        (0..n as u64).map(|i| make_seq(i, tokens_per_seq)).collect();

                    // Allocate all
                    for seq in &seqs {
                        mgr.allocate(seq).unwrap();
                    }

                    // Simulate decode step: cow_if_needed for each
                    for seq in &seqs {
                        black_box(mgr.cow_if_needed(seq).unwrap());
                    }

                    // Free all
                    for seq in &seqs {
                        mgr.free(seq);
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_fork_cow_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("fork_cow_cycle");

    // Beam search pattern: allocate parent, fork N children, cow each child
    for num_beams in [4, 8, 16] {
        let block_size = 16;
        let tokens = 32 * block_size; // 32 blocks
        let total_blocks = (num_beams + 1) * 32 + 128;

        group.bench_with_input(BenchmarkId::new("beams", num_beams), &num_beams, |b, &n| {
            b.iter(|| {
                let gpu = Arc::new(BenchPool::new(total_blocks));
                let cpu = Arc::new(BenchPool::new(16));
                let mut mgr = BlockManager::new(gpu, cpu, block_size);
                mgr.set_watermark(0.0);

                let parent = make_seq(0, tokens);
                mgr.allocate(&parent).unwrap();

                let mut children: Vec<Sequence> =
                    (1..=n as u64).map(|i| make_seq(i, tokens)).collect();

                for child in &mut children {
                    mgr.fork(&parent, child).unwrap();
                }

                for child in &children {
                    black_box(mgr.cow_if_needed(child).unwrap());
                }

                for child in &children {
                    mgr.free(child);
                }
                mgr.free(&parent);
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_ref_counter,
    bench_cow_hot_path,
    bench_fork_cow_cycle
);
criterion_main!(benches);
