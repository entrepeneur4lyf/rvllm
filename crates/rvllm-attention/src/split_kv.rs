//! Split-KV paged attention backend (inspired by b12x).
//!
//! Distributes KV blocks across multiple CUDA thread blocks per (seq, head),
//! then combines partial results. Significantly improves throughput at high
//! concurrency by enabling parallel KV processing.
//!
//! Split heuristic (from b12x):
//!   context_len <= 1024  -> 1 split (direct, no workspace)
//!   context_len <= 4096  -> 2 splits
//!   context_len <= 8192  -> 4 splits
//!   context_len <= 16384 -> 8 splits
//!   context_len > 16384  -> 16 splits

use half::f16;
use tracing::trace;

use crate::backend::AttentionBackend;
use crate::buffer::GpuBuffer;
use rvllm_core::prelude::Result;

/// Choose number of splits based on max context length.
/// Matches b12x's split bucket heuristic.
pub fn choose_num_splits(max_context_len: usize) -> usize {
    if max_context_len <= 1024 {
        1
    } else if max_context_len <= 4096 {
        2
    } else if max_context_len <= 8192 {
        4
    } else if max_context_len <= 16384 {
        8
    } else {
        16
    }
}

/// Split-KV attention backend (CPU reference implementation).
///
/// Mirrors the CUDA split-KV kernel logic for correctness testing.
/// For GPU execution, use `CudaSplitKvAttention` (requires `cuda` feature).
pub struct SplitKvAttention {
    /// Number of KV heads (for GQA).
    num_kv_heads: Option<usize>,
}

impl Default for SplitKvAttention {
    fn default() -> Self {
        Self::new()
    }
}

impl SplitKvAttention {
    pub fn new() -> Self {
        Self { num_kv_heads: None }
    }

    pub fn with_kv_heads(num_kv_heads: usize) -> Self {
        Self {
            num_kv_heads: Some(num_kv_heads),
        }
    }
}

impl AttentionBackend for SplitKvAttention {
    fn forward(
        &self,
        query: &GpuBuffer<f16>,
        key_cache: &GpuBuffer<f16>,
        value_cache: &GpuBuffer<f16>,
        block_tables: &GpuBuffer<i32>,
        context_lens: &GpuBuffer<i32>,
        max_context_len: usize,
        scale: f32,
    ) -> Result<GpuBuffer<f16>> {
        let num_seqs = context_lens.data.len();
        let num_heads = query.shape[1];
        let head_dim = query.shape[2];
        let block_size = key_cache.shape[1];
        let num_kv_heads = self.num_kv_heads.unwrap_or(num_heads);
        let max_blocks = block_tables.shape.get(1).copied().unwrap_or(0);

        if num_seqs == 0 {
            return Ok(GpuBuffer {
                data: vec![],
                shape: vec![0, num_heads, head_dim],
            });
        }

        let num_splits = choose_num_splits(max_context_len);
        trace!(
            num_seqs,
            num_heads,
            head_dim,
            num_splits,
            max_context_len,
            "split_kv forward"
        );

        let mut output = vec![f16::ZERO; num_seqs * num_heads * head_dim];

        for seq in 0..num_seqs {
            let ctx_len = context_lens.data[seq] as usize;
            if ctx_len == 0 {
                continue;
            }

            let actual_splits = num_splits.min(ctx_len.div_ceil(64).max(1));

            for head in 0..num_heads {
                let kv_head = if num_kv_heads == num_heads {
                    head
                } else {
                    head / (num_heads / num_kv_heads)
                };

                // Compute partials per split
                let total_tiles = ctx_len.div_ceil(64);
                let tiles_per_split = total_tiles.div_ceil(actual_splits);

                let mut split_outs: Vec<Vec<f32>> = Vec::new();
                let mut split_maxes: Vec<f32> = Vec::new();
                let mut split_sums: Vec<f32> = Vec::new();

                for s in 0..actual_splits {
                    let start_tile = s * tiles_per_split;
                    let end_tile = total_tiles.min(start_tile + tiles_per_split);

                    if start_tile >= total_tiles {
                        split_outs.push(vec![0.0; head_dim]);
                        split_maxes.push(f32::NEG_INFINITY);
                        split_sums.push(0.0);
                        continue;
                    }

                    let mut row_max = f32::NEG_INFINITY;
                    let mut row_sum = 0.0f32;
                    let mut acc = vec![0.0f32; head_dim];

                    for tile in start_tile..end_tile {
                        let tile_start = tile * 64;
                        let tile_len = 64.min(ctx_len - tile_start);

                        for t in 0..tile_len {
                            let kv_pos = tile_start + t;
                            let page_idx = kv_pos / block_size;
                            let page_off = kv_pos % block_size;
                            let phys_block =
                                block_tables.data[seq * max_blocks + page_idx] as usize;

                            // Q * K dot product
                            let mut dot = 0.0f32;
                            for d in 0..head_dim {
                                let q_val =
                                    query.data[(seq * num_heads + head) * head_dim + d].to_f32();
                                let k_idx = ((phys_block * block_size + page_off) * num_kv_heads
                                    + kv_head)
                                    * head_dim
                                    + d;
                                let k_val = key_cache.data[k_idx].to_f32();
                                dot += q_val * k_val;
                            }
                            dot *= scale;

                            // Online softmax
                            if dot > row_max {
                                let correction = (row_max - dot).exp();
                                for a in acc.iter_mut() {
                                    *a *= correction;
                                }
                                row_sum *= correction;
                                row_max = dot;
                            }
                            let weight = (dot - row_max).exp();
                            row_sum += weight;

                            // Accumulate V
                            for (d, acc_val) in acc.iter_mut().enumerate().take(head_dim) {
                                let v_idx = ((phys_block * block_size + page_off) * num_kv_heads
                                    + kv_head)
                                    * head_dim
                                    + d;
                                let v_val = value_cache.data[v_idx].to_f32();
                                *acc_val += weight * v_val;
                            }
                        }
                    }

                    split_outs.push(acc);
                    split_maxes.push(row_max);
                    split_sums.push(row_sum);
                }

                // Combine splits
                let global_max = split_maxes
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                let out_base = (seq * num_heads + head) * head_dim;
                let mut combined_sum = 0.0f32;
                let mut combined_out = vec![0.0f32; head_dim];

                for s in 0..actual_splits {
                    if split_sums[s] <= 0.0 {
                        continue;
                    }
                    let correction = (split_maxes[s] - global_max).exp();
                    combined_sum += correction * split_sums[s];
                    for (d, comb_val) in combined_out.iter_mut().enumerate().take(head_dim) {
                        *comb_val += correction * split_outs[s][d];
                    }
                }

                let inv_sum = if combined_sum > 0.0 {
                    1.0 / combined_sum
                } else {
                    0.0
                };
                for (d, comb_val) in combined_out.iter().enumerate().take(head_dim) {
                    output[out_base + d] = f16::from_f32(comb_val * inv_sum);
                }
            }
        }

        Ok(GpuBuffer {
            data: output,
            shape: vec![num_seqs, num_heads, head_dim],
        })
    }

    fn name(&self) -> &str {
        "SplitKvAttention-CPU"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_splits_heuristic() {
        assert_eq!(choose_num_splits(512), 1);
        assert_eq!(choose_num_splits(1024), 1);
        assert_eq!(choose_num_splits(1025), 2);
        assert_eq!(choose_num_splits(4096), 2);
        assert_eq!(choose_num_splits(4097), 4);
        assert_eq!(choose_num_splits(8192), 4);
        assert_eq!(choose_num_splits(8193), 8);
        assert_eq!(choose_num_splits(16384), 8);
        assert_eq!(choose_num_splits(16385), 16);
    }

    fn buf_f16(vals: &[f32], shape: Vec<usize>) -> GpuBuffer<f16> {
        GpuBuffer {
            data: vals.iter().map(|&v| f16::from_f32(v)).collect(),
            shape,
        }
    }

    fn buf_i32(vals: &[i32], shape: Vec<usize>) -> GpuBuffer<i32> {
        GpuBuffer {
            data: vals.to_vec(),
            shape,
        }
    }

    #[test]
    fn split_kv_single_token_single_block() {
        // 1 seq, 1 head, head_dim=4, block_size=4, 1 block with 2 tokens
        let query = buf_f16(&[1.0, 0.0, 0.0, 0.0], vec![1, 1, 4]);
        let key_cache = buf_f16(
            &[
                1.0, 0.0, 0.0, 0.0, // token 0, head 0
                0.0, 1.0, 0.0, 0.0, // token 1, head 0
                0.0, 0.0, 0.0, 0.0, // token 2 (unused)
                0.0, 0.0, 0.0, 0.0, // token 3 (unused)
            ],
            vec![1, 4, 1, 4],
        );
        let value_cache = buf_f16(
            &[
                1.0, 2.0, 3.0, 4.0, // token 0
                5.0, 6.0, 7.0, 8.0, // token 1
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![1, 4, 1, 4],
        );
        let block_tables = buf_i32(&[0], vec![1, 1]);
        let context_lens = buf_i32(&[2], vec![1]);

        let backend = SplitKvAttention::new();
        let scale = 1.0f32 / (4.0f32).sqrt();
        let out = backend
            .forward(
                &query,
                &key_cache,
                &value_cache,
                &block_tables,
                &context_lens,
                2,
                scale,
            )
            .unwrap();

        assert_eq!(out.shape, vec![1, 1, 4]);
        // With query=[1,0,0,0], K0=[1,0,0,0] dot=1*scale, K1=[0,1,0,0] dot=0
        // So softmax heavily weights token 0 -> output ~= V0 = [1,2,3,4]
        let v0 = out.data[0].to_f32();
        assert!(v0 > 0.5, "expected output[0] > 0.5, got {}", v0);
    }

    #[test]
    fn split_kv_empty_sequence() {
        let query = buf_f16(&[1.0, 0.0], vec![1, 1, 2]);
        let key_cache = buf_f16(&[0.0; 8], vec![1, 4, 1, 2]);
        let value_cache = buf_f16(&[0.0; 8], vec![1, 4, 1, 2]);
        let block_tables = buf_i32(&[0], vec![1, 1]);
        let context_lens = buf_i32(&[0], vec![1]);

        let backend = SplitKvAttention::new();
        let out = backend
            .forward(
                &query,
                &key_cache,
                &value_cache,
                &block_tables,
                &context_lens,
                0,
                1.0,
            )
            .unwrap();

        assert_eq!(out.shape, vec![1, 1, 2]);
    }

    #[test]
    fn split_kv_backend_name() {
        let backend = SplitKvAttention::new();
        assert_eq!(backend.name(), "SplitKvAttention-CPU");
    }
}
