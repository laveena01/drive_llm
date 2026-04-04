# Phase 1: Vector Encoder Module - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-04
**Phase:** 01-vector-encoder-module
**Areas discussed:** Encoder dimensions, Pooling strategy, Type embedding

---

## Encoder Dimensions

| Option | Description | Selected |
|--------|-------------|----------|
| Paper-scale (256 hidden, 64 prefix) | Match official implementation dimensions | ✓ |
| Smaller (128 hidden, 16 prefix) | More conservative for smaller input | |
| Larger (512 hidden, 64 prefix) | More capacity | |

**User's choice:** Paper-scale dimensions (hidden_dim=256, prefix_len=64)
**Notes:** User has 48GB GPU — no need to compromise on dimensions.

---

## Pooling Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Masked mean pool + linear reshape | Current stub approach. Average objects → huge linear expansion. | |
| Learned query tokens + cross-attention | Paper approach. 64 query tokens attend independently to objects. | ✓ |
| Direct projection | Skip pooling. Only works if prefix_len == max_objects. | |

**User's choice:** Learned query tokens with cross-attention (paper-faithful)
**Notes:** User asked about pros/cons. Key insight: mean pooling creates bottleneck (256 → 49,152 linear); cross-attention preserves per-object identity and uses a simpler (256 → 768) per-token projection.

---

## Type Embedding

| Option | Description | Selected |
|--------|-------------|----------|
| Raw float | type_id as continuous number in 8D vector | |
| Learned embedding | nn.Embedding(4, 16) for categorical type_id | ✓ |

**User's choice:** Learned embedding (recommended)
**Notes:** Treating categorical as continuous is a known ML pitfall. 4×16=64 params, negligible overhead.

## Agent's Discretion

- Layer norm placement
- Weight initialization strategy
- Whether to add positional encoding to object tokens

## Deferred Ideas

None
