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

`crates/rvllm-sampling/src/guided.rs` (867 lines) and `json_schema.rs` (633 lines):

- **API types**: `ResponseFormat` enum in `rvllm-core/src/types.rs` with variants: `Text`, `JsonObject`, `JsonSchema { json_schema }`, `Regex { pattern }`
- **Internal types**: `Constraint` enum in `guided.rs` with variants: `None`, `Json`, `JsonSchema(SchemaNode)`, `Regex(String)` — compiled from `ResponseFormat`
- **Custom JSON schema parser**: Compiles JSON Schema into `SchemaNode` tree (11 variants including Object, Array, String with constraints, Number, Boolean, Null, AnyOf, Enum, Const). Uses `ParseContext` to track partial output.
- **Token masking**: Character-level via `apply_char_mask()` -- computes valid next bytes, then checks first byte of each token. Includes EOS handling (allowed only when `is_complete()`) and bounds checking.
- **Regex engine**: Hand-written recursive backtracking (no external `regex` dependency). Supports literal chars, `.`, `*`, `+`, `?`, `\d`, `\D`, `\w`, `\W`, `\s`, `\S`. Does NOT support `[...]` character classes or `{n,m}` quantifiers despite doc comments claiming otherwise. Limited to ASCII (`0x20..=0x7E` + `\n\t\r`).
- **Integration**: Applied per-sequence in the worker before sampling via `state.apply_mask()`

### Critical Runtime Issues

**The guided decoding system is currently non-functional at runtime**, worse than just "unsound":

1. **`vocab_table` is never populated**: `GpuWorker.vocab_table` is initialized as `None` (gpu_worker.rs line 427) and no code anywhere sets it to `Some(...)`. The `if let Some(ref vocab) = self.vocab_table` guard at line 1634 never matches, making `apply_mask()` a no-op.

2. **`state.advance()` is never called**: The worker calls `state.apply_mask()` but never calls `state.advance(sampled_token_text)`. The `GuidedDecodingState.generated` field stays empty, so even if `vocab_table` were populated, the constraint would only ever enforce from the initial state.

3. **Silent error swallowing**: If schema compilation fails, the code silently falls back to unconstrained text generation with no logging.

4. **Memory leak**: `guided_states: HashMap<u64, GuidedDecodingState>` entries are never removed when sequences complete.

### Soundness Issues (if the above were fixed)

The first-byte heuristic is fundamentally wrong for multi-byte tokens:

```rust
// Actual code (guided.rs lines 168-190, simplified):
fn apply_char_mask(&self, logits: &mut [f32], vocab: &VocabTable, allowed_bytes: &[u8]) {
    for entry in &vocab.entries {
        let id = entry.id as usize;
        if id >= logits.len() { continue; }
        if entry.id == vocab.eos_token_id {
            if !self.is_complete() { logits[id] = f32::NEG_INFINITY; }
            continue;
        }
        if entry.text.is_empty() { continue; }
        let first_byte = entry.text.as_bytes()[0];
        if !allowed_bytes.contains(&first_byte) {
            logits[id] = f32::NEG_INFINITY;
        }
    }
}
```

Problem: Token `"name"` (4 bytes) is allowed if `"` is valid, even when the full string `"name"` violates the grammar. Similarly, token `"null"` is allowed if `n` is valid, even in a context where only a number is expected.

Additional soundness gaps:
- `StringConstraints` (`min_length`, `max_length`, `pattern`, `enum_values`) are parsed but never enforced during guided decoding
- Array/object validation is structural only (tracks brackets/braces, doesn't validate items against schema)
- The regex engine uses recursive backtracking (exponential worst case) and is limited to ASCII

## Implementation Plan

### Recommended Backend: `outlines-core`

`outlines-core` is the Rust-native FSM engine used by the outlines Python library. It provides:
- Regex → DFA compilation
- JSON Schema → regex → DFA compilation
- Token-level FSM that maps `(state, token_id) → next_state`
- Vocabulary indexing for efficient allowed-token lookup

**IMPORTANT CAVEAT**: The `outlines-core` Rust crate API needs to be validated before implementation begins. The research identified it as the best candidate for Rust integration, but the exact API surface (struct names, method signatures, version compatibility) must be verified against the published crate on crates.io. If the crate API has changed significantly or doesn't expose the needed functionality, `llguidance` (which also has a Rust component) would be the fallback.

### Phase 0: Fix Runtime Prerequisites

**Files**: `crates/rvllm-worker/src/gpu_worker.rs`

Before the backend can be replaced, the runtime integration needs basic fixes:
1. **Populate `vocab_table`**: Build `VocabTable` from the tokenizer during worker initialization.
2. **Call `state.advance()`**: After sampling, advance the guided state with the sampled token text.
3. **Add `warn!()` logging**: When schema compilation fails and falls back to unconstrained.
4. **Clean up `guided_states`**: Remove entries when sequences complete.

These fixes make the existing system at least functional for testing the new backend against.

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
    /// Rollback N tokens (for speculative decoding integration with Spec 04).
    fn rollback(&mut self, num_tokens: usize);
}

/// Trivial no-constraint grammar (used for ResponseFormat::Text)
pub struct UnconstrainedGrammar;
impl Grammar for UnconstrainedGrammar {
    fn fill_bitmask(&self, bitmask: &mut [u32]) { bitmask.iter_mut().for_each(|w| *w = u32::MAX); }
    fn accept_token(&mut self, _token_id: TokenId) {}
    fn is_terminated(&self) -> bool { false }
    fn reset(&mut self) {}
    fn rollback(&mut self, _num_tokens: usize) {}
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
let mut bitmask = vec![0u32; (vocab_size + 31) / 32];
grammar.fill_bitmask(&mut bitmask);
apply_bitmask_to_logits(&mut logits, &bitmask);
// ... sample ...
grammar.accept_token(sampled_token_id);
```

The bitmask should be applied BEFORE other logit processors (temperature, top-k, top-p, repetition penalty) to ensure invalid tokens are eliminated before sampling.

The existing `GuidedDecodingState` can be kept as a fallback (behind a feature flag) for cases where the outlines backend doesn't support a constraint type. Also update the public `apply_guided_mask()` free function exported from `lib.rs`.

### Phase 4: Grammar Compilation Caching

**Files**: `crates/rvllm-sampling/src/structured.rs`

Cache compiled grammars by constraint key to avoid recompilation for repeated schemas:

```rust
struct GrammarCache {
    cache: HashMap<String, Arc<CompiledGrammar>>,
    max_size: usize,
}
```

The cache key is the canonical string representation of the constraint (schema JSON or regex pattern). **Note**: `serde_json::to_string()` produces non-canonical JSON (keys in insertion order). Use a sorted-keys serialization or hash the schema deterministically for consistent cache keys. Each request gets a cloned grammar instance from the cached compilation.

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
| `crates/rvllm-worker/src/gpu_worker.rs` | Phase 0: populate `vocab_table`, call `advance()`, clean up states. Phase 3: replace masking with bitmask |
| `crates/rvllm-sampling/src/structured.rs` | New: Backend trait, bitmask utilities, `UnconstrainedGrammar` |
| `crates/rvllm-sampling/src/outlines_backend.rs` | New: outlines-core integration |
| `crates/rvllm-sampling/src/lib.rs` | Export new modules, update `apply_guided_mask()` |
| `crates/rvllm-sampling/Cargo.toml` | Add outlines-core dependency (first external regex/FSM dep) |
| `crates/rvllm-sampling/src/guided.rs` | Keep as fallback, mark as deprecated, fix doc comment (claims `[...]` and `{n,m}` support that doesn't exist) |

## Open Questions

- **outlines-core API stability**: The crate is actively developed. Pin to a specific version and validate the API before starting. All Phase 2+ code examples in this spec are illustrative — the exact API surface (`Guide`, `Index`, `Vocabulary`) must be verified against the published crate on crates.io. If the Rust crate doesn't expose these types directly, may need to use the underlying `regex-automata` DFA approach directly.
- **EBNF grammar support**: outlines-core may not support EBNF/Lark grammars. vLLM supports `grammar` (EBNF), `choice`, and `structural_tag` types that rvLLM doesn't currently have. If needed, `llguidance` (Rust bindings) handles EBNF. Recommendation: defer EBNF and vLLM-specific types to a follow-up.
- **GPU bitmask application**: vLLM uses a Triton kernel to apply bitmasks on GPU. For rvLLM, CPU bitmask application is fine initially. GPU kernel can be added later if this becomes a bottleneck.
- **Async compilation**: vLLM compiles grammars asynchronously to avoid blocking the engine. For rvLLM, synchronous compilation is fine initially since Rust compilation is faster than Python. Add async if compilation latency becomes measurable.
- **Testing note**: The spec's test case `[0-9]{3}-[0-9]{4}` requires `[...]` character class and `{n,m}` quantifier support — the current regex engine cannot handle this, confirming the need for the outlines-core backend. The parity test (comparing against Python vLLM) requires a running vLLM instance and is an optional integration test, not a unit test.
