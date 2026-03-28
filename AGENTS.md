# AGENTS.md — rvLLM

> Agent-facing reference for autonomous development on this repository.

## Language & Stack

- **Primary:** Rust (edition 2021), 23 workspace crates under `crates/`
- **Secondary:** Python (PyO3 bindings via maturin), CUDA C++ kernels
- **Build:** Cargo workspace, Makefile targets
- **API:** OpenAI-compatible REST (Axum), Prometheus metrics

## Setup

```bash
# Clone and verify (no GPU required — uses mock-gpu backend)
git clone https://github.com/entrepeneur4lyf/rvllm.git
cd rvllm
cargo test --workspace

# Install pre-commit hooks (fmt + clippy)
make setup

# Environment variables
cp .env.example .env
```

## Build Commands

```bash
# Local development (mock-gpu, no CUDA)
cargo build --release -p rvllm-server   # or: make build

# CUDA build (Linux + NVIDIA GPU)
cargo build --release --features cuda -p rvllm-server   # or: make build-cuda

# Check workspace compiles
cargo check --workspace   # or: make check

# Compile CUDA kernels to PTX (requires nvcc)
bash kernels/build.sh   # or: make kernels
```

## Test Commands

```bash
# Run all tests (790 tests, mock-gpu, no GPU required)
cargo test --workspace   # or: make test

# Run a specific crate's tests
cargo test -p rvllm-sampling

# Run tests with CUDA features
cargo test --workspace --features rvllm-server/cuda   # or: make test-cuda

# API compatibility tests (requires running server)
VLLM_RS_URL=http://localhost:8000 python3 -m pytest tests/api_compat/ -v

# Benchmarks
cargo bench --package rvllm-bench --bench sampling_bench   # or: make bench
```

## Lint & Format

```bash
# Format code
cargo fmt --all

# Check formatting (CI-safe)
cargo fmt --all -- --check

# Lint with clippy
cargo clippy --workspace

# Pre-commit: run both before committing
cargo fmt --all -- --check && cargo clippy --workspace
```

## Run the Server

```bash
# Start serving (downloads model from HuggingFace)
./target/release/rvllm serve --model Qwen/Qwen2.5-1.5B

# With options
./target/release/rvllm serve \
  --model Qwen/Qwen2.5-1.5B \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90

# Docker
make docker
docker run --gpus all -p 8000:8000 rvllm:latest serve --model Qwen/Qwen2.5-1.5B
```

## Architecture

```
crates/
  rvllm-core/          # Shared types, errors, traits
  rvllm-config/        # Configuration loading and validation
  rvllm-gpu/           # GPU abstraction (CUDA via cudarc, mock-gpu for dev)
  rvllm-memory/        # GPU/CPU memory pool management
  rvllm-block-manager/ # Paged KV-cache block allocation
  rvllm-kv-cache/      # KV-cache engine and operations
  rvllm-attention/     # Attention backends (FlashAttention-2, PagedAttention, SplitKV)
  rvllm-tokenizer/     # Tokenizer wrapper (HuggingFace tokenizers)
  rvllm-model-loader/  # Model weight loading (safetensors, GGUF)
  rvllm-model-runner/  # Model architectures and forward pass (Llama, Qwen, Mistral, etc.)
  rvllm-sampling/      # Token sampling (greedy, top-k, top-p, penalties)
  rvllm-scheduler/     # Request scheduling and batching
  rvllm-sequence/      # Sequence and sequence group management
  rvllm-worker/        # GPU worker (forward pass orchestration)
  rvllm-executor/      # Executor abstraction (single-GPU, tensor-parallel)
  rvllm-engine/        # LLM engine (ties scheduler + worker + sampling)
  rvllm-api/           # OpenAI-compatible REST API (Axum)
  rvllm-server/        # HTTP server binary entry point
  rvllm-quant/         # Quantization (FP8, GPTQ, AWQ, Q4)
  rvllm-speculative/   # Speculative decoding
  rvllm-telemetry/     # Structured logging, Prometheus metrics
  rvllm-bench/         # Benchmarks
  rvllm-python/        # Python bindings (PyO3)
```

## Code Conventions

- `cargo fmt` before every commit
- `cargo clippy --workspace` should be warning-free
- All public items need `///` doc comments
- Use `tracing::{info, debug, warn, error}` for logging, never `println!`
- Error handling: return `Result<T>` with `LLMError`, never `unwrap()` in library code
- CUDA-specific code behind `#[cfg(feature = "cuda")]`
- Most crates use `#![forbid(unsafe_code)]` — respect this

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Text completion (streaming + non-streaming) |
| `/v1/chat/completions` | POST | Chat completion (streaming + non-streaming) |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List available models |
| `/health` | GET | Liveness check |
| `/metrics` | GET | Prometheus metrics |

## Environment Variables

See `.env.example` for the full list. Key variables:

- `RVLLM_HOST` / `RVLLM_PORT` — Server bind address (default: 0.0.0.0:8000)
- `RVLLM_MODEL` — HuggingFace model name or local path
- `RUST_LOG` — Log level (info, debug, trace)
- `CUDA_VISIBLE_DEVICES` — GPU selection
- `HF_TOKEN` — HuggingFace access token (for gated models)
