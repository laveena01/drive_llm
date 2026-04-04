# Phase 1: Vector Encoder Module - Context

**Gathered:** 2026-04-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Complete the VectorPrefixEncoder with proper `__init__`, learned query cross-attention, type embedding, and unit test. The stub's `forward()` is written but references undefined `self.obj_in`, `self.encoder`, `self.to_prefix`. This phase fills in `__init__` and adjusts `forward()` to use cross-attention instead of mean pooling.

</domain>

<decisions>
## Implementation Decisions

### Encoder Dimensions
- **D-01:** Use paper-scale dimensions: `hidden_dim=256`, `prefix_len=64`. User has 48GB GPU — no need to compromise.
- **D-02:** Keep existing config defaults: `n_layers=2`, `n_heads=4`, `dropout=0.1`.

### Pooling Strategy
- **D-03:** Use **learned query tokens with cross-attention** (paper-faithful), NOT mean pooling + linear reshape.
  - `nn.Parameter(prefix_len, hidden_dim)` query tokens attend to object embeddings via `nn.MultiheadAttention`.
  - Each of the 64 prefix tokens independently attends to all objects — preserves per-object identity.
  - `to_prefix` becomes a simple `(hidden_dim → t5_d_model)` per-token projection instead of a massive bottleneck linear.
  - Replaces the mean-pool section in the existing `forward()` stub.

### Type Embedding
- **D-04:** Add **learned type embedding** (`nn.Embedding(num_types, type_embed_dim)`) for the `type_id` categorical feature.
  - Extract `type_id` (last dim) from vectors, cast to long, look up embedding.
  - Concatenate with the remaining 7 continuous dims → project from `(7 + type_embed_dim)` to `hidden_dim`.
  - Avoids treating categorical type_id as continuous float.
  - `num_types=4` (car, pedestrian, traffic_light, object), `type_embed_dim=16` (negligible overhead).

### Agent's Discretion
- Exact layer norm placement (pre-norm vs post-norm in transformer layers)
- Weight initialization strategy (Xavier, Kaiming, or default)
- Whether to add positional encoding to object tokens (objects are unordered, so likely skip)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Encoder Stub
- `llm_driving/vector_encoder.py` — Contains VectorEncoderConfig dataclass and VectorPrefixEncoder forward() stub that needs __init__

### Paper Reference Architecture
- Official VectorEncoder: `github.com/wayveai/Driving-with-LLMs/blob/main/models/vector_encoder.py` — Per-type MLPs + Perceiver with learned queries (our adaptation uses unified encoder but same query-attention concept)
- Official VectorObservation: `github.com/wayveai/Driving-with-LLMs/blob/main/utils/vector_utils.py` — Data structures for vector observations

### Current Config
- `llm_driving/config.py` — Contains USE_VECTOR_PREFIX, VECTOR_ENCODER_CONFIG dict, T5_D_MODEL constant

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `VectorEncoderConfig` dataclass: already defines all hyperparameters needed
- `forward()` stub: debug checks, padding mask logic can be kept (adjust pooling section)

### Established Patterns
- `config.py` stores all hyperparameters as module-level constants/dicts
- Logger pattern: `logger = logging.getLogger("llm_driving")` used throughout

### Integration Points
- `VectorPrefixEncoder` imported by Phase 2's `VectorPrefixT5` wrapper
- Config values from `config.py`: `MAX_OBJECTS=10`, `VECTOR_DIM=8`, `T5_D_MODEL=768`

</code_context>

<specifics>
## Specific Ideas

- Paper uses `num_queries=64` and `model_dim=256` for the Perceiver — we match these as `prefix_len=64` and `hidden_dim=256`
- Paper's `llm_proj` (Linear from encoder output to LLM hidden size) maps to our `to_prefix` (Linear from hidden_dim to t5_d_model)
- The paper's Perceiver uses `num_blocks=7` — we use `n_layers=2` since our input is much simpler (10 objects × 8D vs 80 objects × 33D)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-vector-encoder-module*
*Context gathered: 2026-04-04*
