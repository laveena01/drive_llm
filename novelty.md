# Novelty Plan — MTech Thesis

## Current Contributions (Implemented)

### Novelty 1: Multi-Dimensional Risk Scoring
**Status**: Complete

The base paper approach uses simple distance-based risk. This work replaces it with
a principled multi-component risk model (`risk_calculator.py`) that better reflects
real driving hazards:

| Component | Weight | What it captures |
|-----------|--------|-----------------|
| Collision Risk | 40% | TTC (closing speed), distance baseline, lateral conflict (cut-in) |
| Pedestrian Risk | 30% | Vulnerable road users weighted higher; activates off-axis |
| Uncertainty Risk | 20% | Distant + small + fast objects (less predictable) |
| Regulatory Risk | 10% | Red traffic lights |

**Aggregation**: Log-sum-exp over objects (soft-max, not hard-max — more stable gradients).

**Integration**: Risk drives oracle action labels in Stage 2 (`policy_from_risk()`).
Stage 1 is intentionally kept risk-free — the model learns pure object description.

**Ablation row**: Baseline (distance-only) vs. +Multi-Risk

---

## Planned Novelties

### Novelty 2: Temporal Context (Primary — Next to Implement)

**Motivation**: Every frame is currently an isolated snapshot. Object velocity
(rel_vx, rel_vy) is an instantaneous within-frame feature. A car decelerating over
4 consecutive frames and now 8m away is far more dangerous than a car with the same
instantaneous velocity in isolation — the current model cannot distinguish these.

**What changes**: Feed a sliding window of K=4 past frames (2 seconds at nuScenes 2Hz)
to the model so it sees how the scene evolves.

#### Architecture

```
Current:
  (B, 10, 8) → VectorPrefixEncoder → (B, 64, 768) → T5

Proposed:
  (B, K, 10, 8)
       │
       ├── Per-frame encoding (shared VectorPrefixEncoder weights)
       │   Applied independently to each of K frames
       │
       ▼
  (B, K, d_slot)   ← one embedding per frame
       │
       TemporalTransformer (2 layers, 4 heads)
       (attends across K frames; frame position embeddings 0..K-1)
       │
       ▼
  Take last frame output: (B, d_slot)
       │
       Linear → reshape
       │
       ▼
  (B, 64, 768)   ← same output shape as before; T5 and Stage 2 unchanged
```

**Key design choices**:
- Shared encoder weights across frames (parameter efficient, forces generalisation)
- Frame position embeddings (model knows oldest vs newest frame)
- Padding mask for scene-start frames (window < K at beginning of scene)
- Past frames' object positions transformed into current ego frame (coordinate consistency)
- Output shape unchanged → Stage 2 and inference code need zero modifications

#### Files to Change
| File | Change |
|------|--------|
| `config.py` | `USE_TEMPORAL=True`, `TEMPORAL_WINDOW=4` |
| `nuscenes_data.py` | `get_frame_window()` — K-frame window with ego transform |
| `datasets_builder.py` | Add `vectors_window` and `window_len` to captioning samples |
| `vector_encoder.py` | Add `TemporalVectorEncoder` class |
| `vector_prefix_t5.py` | Branch on `USE_TEMPORAL`, accept `vectors_window` + `window_len` |
| `data_collator.py` | Collate `vectors_window` → `(B, K, 10, 8)` tensor |
| `training.py` | Pass `vectors_window`, `window_len` from batch to model |

#### nuScenes Infrastructure Available
- `sample["next"]` / `sample["prev"]` — linked list for traversal
- `ego_pose_token` on LIDAR_TOP sample data — real ego position + rotation per frame
- Instance tokens — objects tracked across frames (available if object-level alignment needed)

#### Ablation
- K=2 (1 second), K=4 (2 seconds), K=8 (4 seconds)
- With/without coordinate transform (ablate transform quality)

---

### Novelty 3: Real Ego-Motion Integration (Quick Fix, Foundational)

**Motivation**: `DEFAULT_EGO_SPEED = 10.0 m/s` is hardcoded always.
nuScenes has real ego velocity from pose transforms. All TTC and risk calculations
are wrong when ego is stopped at a light or at highway speed.

**What to do**:
1. Compute ego velocity from consecutive pose transforms in `nuscenes_data.py`
2. Pass real `ego_speed` to `risk_calculator.py`
3. Add ego velocity + acceleration + heading change to the vector input or scene context

**Thesis framing**: "Improved base system" — makes risk scores accurate and results credible.
Not a primary novelty, but strengthens the foundation before Temporal Context experiments.

---

### Novelty 4 (Optional): Spatial Relation Graph Encoder

**Motivation**: Current encoder treats each object independently. Object interactions
matter for driving: a pedestrian stepping between two parked cars is different from a
pedestrian in open space.

**What it is**: Replace/augment the Transformer in `VectorPrefixEncoder` with a
Graph Attention Network (GAT) where edges connect object pairs within a proximity
threshold. Edge features: relative position, relative velocity.

**Thesis fit**: Complementary to temporal — temporal models "what was", spatial graph
models "how objects relate right now". Can be ablated independently.

**Implementation**: Use `torch_geometric` or hand-roll attention with adjacency mask.

---

### Novelty 5 (Optional): Contrastive Risk-Aligned Encoding

**Motivation**: Stage 1 trains by caption reconstruction loss only. The encoder may
learn scene description quality without learning risk-discriminative representations.

**What it is**: Add a contrastive objective to Stage 1 — scenes with similar risk
levels should have nearby encoder representations; scenes with very different risk
levels should be far apart (InfoNCE / triplet loss style).

**Value**: Creates interpretable embeddings. Show with t-SNE that embeddings cluster
by risk level → strong visualisation for thesis analysis chapter.

---

## Recommended Thesis Novelty Stack

```
System                              Metrics to Report
─────────────────────────────────   ─────────────────────────────────
Baseline (snapshot, distance risk)  Action accuracy, BLEU-1, ROUGE-L
+ Novelty 1: Multi-Risk             ΔAction acc, ΔParse OK rate
+ Novelty 3: Real Ego-Motion        Δ Risk accuracy (system quality)
+ Novelty 2: Temporal (K=4)         ΔAction acc, ΔBLEU-1, ΔROUGE-L
+ Novelty 2: Temporal ablation K    Best K for performance vs cost
```

This gives a clean 4-5 row ablation table which is the backbone of an MTech thesis results chapter.

---

## Temporal Context: Implementation Order

1. `config.py` — add flags (10 min)
2. `nuscenes_data.py` — add `get_frame_window()` with ego transform (2-3 hours)
3. `datasets_builder.py` — add window fields to samples (1 hour)
4. `vector_encoder.py` — add `TemporalVectorEncoder` class (2-3 hours)
5. `vector_prefix_t5.py` — branch on `USE_TEMPORAL` (1 hour)
6. `data_collator.py` — collate window tensors (30 min)
7. `training.py` — pass window through batch → model (30 min)
8. Sanity checks: window_len distribution, coordinate transform validation, loss curve

Total estimated implementation: 1-2 days of focused work.
