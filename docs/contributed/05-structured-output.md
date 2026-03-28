# Spec 5: Structured Output Backends

## Summary

Structured output constrains model generation to match a format (JSON schema, regex, grammar). vLLM uses pluggable backends (xgrammar, outlines, guidance, lm-format-enforcer) that compile constraints into finite state machines operating on the tokenizer's vocabulary, producing per-step bitmasks of allowed tokens. rvLLM has a custom recursive-descent guided decoding engine, but it operates at the character level (first-byte heuristic) rather than the token level, making it unsound for multi-byte tokens.

This spec replaces the character-level approach with a token-level FSM backend, maintaining the existing API surface.

## vLLM Reference Behavior

### Architecture

vLLM's structured output system has three layers:

1. **API layer**: `response_format` field in chat/completion requests with types: `json_object`, `json_schema`, `regex`, `grammar` (EBNF), `choice`, `structural_tag`.
2. **Backend layer**: Pluggable backends compile constraints into `StructuredOutputGrammar` objects. Each backend handles compilation differently but exposes the same interface.
3. **Mask layer**: Per decoding step, `grammar.fill_bitmask(bitmask, batch_index)` writes a compact int32 bitmask (1 bit per vocab token). The bitmask is applied to logits before sampling.

### Token-Level FSM

The critical difference from rvLLM: backends operate on **token IDs**, not characters. The grammar compilation step pre-computes, for every FSM state, which **complete tokens** are valid transitions. This correctly handles multi-byte tokens, BPE merge artifacts, and special tokens.

Example: If the grammar allows `"name"` at the current position, the FSM knows that token ID 1234 (representing `"name"`) is valid, but token ID 5678 (representing `"nam"`) is only valid if there exist valid continuations from state after `"nam"`.

### Bitmask Representation

Compact bitfield: `ceil(vocab_size / 32)` int32 values per request. For a 128K vocabulary, that's 4096 int32 values = 16KB per request per step. Much more efficient than iterating the full vocab with f32 masks.

### Per-Step Flow

1. For each request with structured output:
   - `grammar.fill_bitmask(bitmask_tensor, batch_index)` -- backend writes allowed token bits
2. `apply_token_bitmask_inplace(logits, bitmask)` -- masked tokens get logit = -inf
3. Normal sampling proceeds on masked logits
4. After sampling: `grammar.accept_tokens(sampled_token)` -- advance FSM state

### Backend Comparison

| Backend | Compilation | Per-token | Rust-native | Notes |
|---|---|---|---|---|
| xgrammar | Fast (C++) | Bitmask fill via `GrammarMatcher` | No (C++ with Python bindings) | Default in vLLM, best performance |
| outlines | Moderate (Rust core) | Index lookup via `Guide` | **Yes** (`outlines-core` crate) | Rust-native FSM |
| guidance | Fast (C++/Rust) | Bitmask via `llguidance` | Partial | Most feature-complete |
| lm-format-enforcer | Slow (Python) | Token enumeration | No | Simplest, slowest |

### Key vLLM Files

- `vllm/v1/structured_output/__init__.py` -- `StructuredOutputManager`, backend selection
- `vllm/v1/structured_output/backend_xgrammar.py` -- xgrammar backend
- `vllm/v1/structured_output/backend_outlines.py` -- outlines backend
- `vllm/v1/structured_output/backend_types.py` -- `StructuredOutputGrammar` trait

## Current rvLLM State

### What Exists

`crates/rvllm-sampling/src/guided.rs` and `json_schema.rs`:

- **Constraint types**: `ResponseFormat::JsonObject`, `JsonSchema`, `Regex`, `Text`
- **Custom JSON schema parser**: Recursive descent that compiles JSON Schema into `SchemaNode` tree
- **Token masking**: Character-level -- computes valid next bytes, then checks first byte of each token
- **Integration**: Applied per-sequence in the worker before sampling

### Soundness Issues

The first-byte heuristic is fundamentally wrong for multi-byte tokens:

```rust
// Current approach (incorrect):
fn apply_char_mask(&self, logits: &mut [f32], vocab: &VocabTable, allowed_bytes: &[u8]) {
    for entry in &vocab.entries {
        let first_byte = entry.text.as_bytes()[0];
        if !allowed_bytes.contains(&first_byte) {
            logits[id] = f32::NEG_INFINITY;
        }
    }
}
```

Problem: Token `"name"` (4 bytes) is allowed if `"` is valid, even when the full string `"name"` violates the grammar. Similarly, token `"null"` is allowed if `n` is valid, even in a context where only a number is expected.

The regex engine uses recursive backtracking (exponential worst case) and is limited to ASCII.

## Implementation Plan

### Recommended Backend: `outlines-core`

`outlines-core` is the Rust-native FSM engine used by the outlines Python library. It provides:
- Regex → DFA compilation
- JSON Schema → regex → DFA compilation
- Token-level FSM that maps `(state, token_id) → next_state`
- Vocabulary indexing for efficient allowed-token lookup

**IMPORTANT CAVEAT**: The `outlines-core` Rust crate API needs to be validated before implementation begins. The research identified it as the best candidate for Rust integration, but the exact API surface (struct names, method signatures, version compatibility) must be verified against the published crate on crates.io. If the crate API has changed significantly or doesn't expose the needed functionality, `llguidance` (which also has a Rust component) would be the fallback.

### Phase 1: Backend Trait and Bitmask Infrastructure

**Files**: `crates/rvllm-sampling/src/structured.rs` (new)

Define a backend-agnostic trait:

```rust
pub trait StructuredOutputBackend: Send + Sync {
    /// Compile a constraint into a grammar instance for a single request.
    fn compile(&self, constraint: &ResponseFormat, vocab: &VocabTable) -> Result<Box<dyn Grammar>>;
}

pub trait Grammar: Send {
    /// Write allowed token bitmask for the current state.
    fn fill_bitmask(&self, bitmask: &mut [u32]);
    /// Advance the FSM by accepting a token.
    fn accept_token(&mut self, token_id: TokenId);
    /// Check if the grammar has reached an accepting state.
    fn is_terminated(&self) -> bool;
    /// Reset to initial state.
    fn reset(&mut self);
    /// Rollback N tokens (for speculative decoding).
    fn rollback(&mut self, num_tokens: usize);
}
```

Bitmask utilities:

```rust
pub fn apply_bitmask_to_logits(logits: &mut [f32], bitmask: &[u32]) {
    for (token_id, logit) in logits.iter_mut().enumerate() {
        let word_idx = token_id / 32;
        let bit_idx = token_id % 32;
        if bitmask[word_idx] & (1 << bit_idx) == 0 {
            *logit = f32::NEG_INFINITY;
        }
    }
}
```

### Phase 2: Outlines-Core Integration

**Files**: `crates/rvllm-sampling/src/outlines_backend.rs` (new), `crates/rvllm-sampling/Cargo.toml`

Integrate with the `outlines-core` crate (pending API validation):

```rust
// Pseudocode -- exact API TBD after crate validation
use outlines_core::{Guide, Index, Vocabulary};

pub struct OutlinesBackend {
    vocabulary: Vocabulary,  // Built from VocabTable at init
}

impl StructuredOutputBackend for OutlinesBackend {
    fn compile(&self, constraint: &ResponseFormat, vocab: &VocabTable) -> Result<Box<dyn Grammar>> {
        let pattern = match constraint {
            ResponseFormat::JsonSchema { json_schema } => {
                // Convert JSON schema to regex pattern
                outlines_core::json_schema_to_regex(json_schema)?
            }
            ResponseFormat::Regex { pattern } => pattern.clone(),
            ResponseFormat::JsonObject => {
                // Standard JSON object regex
                outlines_core::json_schema_to_regex(&serde_json::json!({"type": "object"}))?
            }
            _ => return Ok(Box::new(UnconstrainedGrammar)),
        };
        let index = Index::new(&pattern, &self.vocabulary)?;
        let guide = Guide::new(index);
        Ok(Box::new(OutlinesGrammar { guide }))
    }
}
```

### Phase 3: Replace Guided Decoding in Worker

**Files**: `crates/rvllm-worker/src/gpu_worker.rs`, `crates/rvllm-sampling/src/lib.rs`

Replace the character-level `GuidedDecodingState::apply_mask()` with:

```rust
// Per sequence with structured output:
let bitmask = grammar.fill_bitmask();
apply_bitmask_to_logits(&mut logits, &bitmask);
// ... sample ...
grammar.accept_token(sampled_token_id);
```

The existing `GuidedDecodingState` can be kept as a fallback (behind a feature flag) for cases where the outlines backend doesn't support a constraint type.

### Phase 4: Grammar Compilation Caching

**Files**: `crates/rvllm-sampling/src/structured.rs`

Cache compiled grammars by constraint key to avoid recompilation for repeated schemas:

```rust
struct GrammarCache {
    cache: HashMap<String, Arc<CompiledGrammar>>,
    max_size: usize,
}
```

The cache key is the canonical string representation of the constraint (schema JSON or regex pattern). Each request gets a cloned grammar instance from the cached compilation.

## API Surface

The OpenAI-compatible API surface stays the same:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": { "type": "object", "properties": { "name": { "type": "string" } } }
  }
}
```

No new API endpoints. The `response_format` field already maps to `ResponseFormat` in `rvllm-core/src/types.rs`.

## Testing Strategy

1. **Soundness test**: Generate JSON constrained to a schema. Parse the output with a JSON schema validator. Must pass 100% of the time (the current character-level approach fails this).
2. **Multi-byte token test**: Use a constraint that requires `"hello"` and verify the model outputs exactly that token sequence, not partial tokens.
3. **Regex test**: Constrain to `[0-9]{3}-[0-9]{4}` and verify output matches.
4. **Performance**: Benchmark bitmask fill time per step. Target: <1ms for 128K vocab.
5. **Parity**: Compare constrained output token sequences between rvLLM and Python vLLM for the same model, prompt, and schema with greedy decoding.

## Files Changed

| File | Change |
|------|--------|
| `crates/rvllm-sampling/src/structured.rs` | New: Backend trait, bitmask utilities |
| `crates/rvllm-sampling/src/outlines_backend.rs` | New: outlines-core integration |
| `crates/rvllm-sampling/src/lib.rs` | Export new modules |
| `crates/rvllm-sampling/Cargo.toml` | Add outlines-core dependency |
| `crates/rvllm-worker/src/gpu_worker.rs` | Replace character-level masking with bitmask |
| `crates/rvllm-sampling/src/guided.rs` | Keep as fallback, mark as deprecated |

## Open Questions

- **outlines-core API stability**: The crate is actively developed. Pin to a specific version and validate the API before starting. If the Rust crate doesn't expose `Guide`/`Index` directly, may need to use the underlying `regex-automata` DFA approach directly.
- **EBNF grammar support**: outlines-core may not support EBNF/Lark grammars. If needed, `llguidance` (Rust bindings) handles EBNF. Recommendation: defer EBNF to a follow-up.
- **GPU bitmask application**: vLLM uses a Triton kernel to apply bitmasks on GPU. For rvLLM, CPU bitmask application is fine initially. GPU kernel can be added later if this becomes a bottleneck.
- **Async compilation**: vLLM compiles grammars asynchronously to avoid blocking the engine. For rvLLM, synchronous compilation is fine initially since Rust compilation is faster than Python. Add async if compilation latency becomes measurable.
