use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rvllm_core::prelude::SamplingParams;
use rvllm_sampling::sampler::Sampler;

fn bench_greedy(c: &mut Criterion) {
    let mut group = c.benchmark_group("greedy_sample");
    let sampler = Sampler::new();
    let params = SamplingParams {
        temperature: 0.0,
        ..Default::default()
    };

    for vocab_size in [32000, 65536, 128256] {
        let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                b.iter(|| {
                    black_box(
                        sampler
                            .sample(&logits, vocab_size, &params, &[], &mut rng)
                            .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_top_p(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_p_sample");
    let sampler = Sampler::new();
    let params = SamplingParams {
        temperature: 0.8,
        top_p: 0.9,
        ..Default::default()
    };

    for vocab_size in [32000, 128256] {
        let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                let mut rng = ChaCha8Rng::seed_from_u64(42);
                b.iter(|| {
                    black_box(
                        sampler
                            .sample(&logits, vocab_size, &params, &[], &mut rng)
                            .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k_sample");
    let sampler = Sampler::new();
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        ..Default::default()
    };

    for vocab_size in [32000, 128256] {
        let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                let mut rng = ChaCha8Rng::seed_from_u64(42);
                b.iter(|| {
                    black_box(
                        sampler
                            .sample(&logits, vocab_size, &params, &[], &mut rng)
                            .unwrap(),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_repetition_penalty(c: &mut Criterion) {
    let mut group = c.benchmark_group("repetition_penalty");
    let sampler = Sampler::new();

    let vocab_size = 128256;
    let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();
    let past_tokens: Vec<u32> = (0..512).collect();

    let params = SamplingParams {
        temperature: 0.0,
        repetition_penalty: 1.1,
        ..Default::default()
    };

    group.bench_function("512_past_tokens", |b| {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        b.iter(|| {
            black_box(
                sampler
                    .sample(&logits, vocab_size, &params, &past_tokens, &mut rng)
                    .unwrap(),
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_greedy,
    bench_top_p,
    bench_top_k,
    bench_repetition_penalty
);
criterion_main!(benches);
