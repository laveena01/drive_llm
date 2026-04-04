# Drive-LLM: Vector Prefix Pipeline

## What This Is

A multimodal autonomous driving LLM that fuses object-level vector data from nuScenes with FLAN-T5 through a learned vector prefix encoder, enabling the model to reason about driving scenes and produce explainable control decisions. Implements the core contribution of "Driving with LLMs" (Chen et al., ICRA 2024) — learned vector-to-language fusion via prefix injection — adapted to a FLAN-T5 + nuScenes + risk-aware stack.

## Core Value

The vector prefix encoder must successfully learn to align numeric object vectors to T5's representation space during Stage-1 pretraining, such that Stage-2 driving QA produces measurably better or comparable action predictions versus the current text-serialization baseline.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. These capabilities exist in the current codebase. -->

- ✓ Object-level 8D vector extraction from nuScenes (nuscenes_data.py) — existing
- ✓ lanGen caption generation from vectors (langen.py) — existing
- ✓ Risk calculation with TTC, collision, pedestrian weighting (risk_calculator.py) — existing
- ✓ Stage-1 text pipeline: vector string → caption (training.py) — existing
- ✓ Stage-2 text pipeline: caption + risk + question → action/risk answer (training.py) — existing
- ✓ Dataset builder with 5 action + 3 risk questions per frame (datasets_builder.py) — existing
- ✓ Inference with format enforcement and action parsing (inference.py) — existing
- ✓ Run logging and reproducibility (logging_utils.py) — existing

### Active

<!-- Current scope. Building toward these. -->

- [ ] Complete VectorPrefixEncoder: linear projection + transformer encoder + pooling + prefix projection → (B, PREFIX_LEN, t5_d_model)
- [ ] Prefix injection into T5 encoder: prepend vector embeddings to text token embeddings with proper attention masking
- [ ] LoRA integration: freeze FLAN-T5 base weights, add LoRA adapters to q/v projections
- [ ] Modified Stage-1 training: vector tensors → VectorPrefixEncoder → prefix + text prompt → T5 generates caption
- [ ] Modified Stage-2 training: inherit Stage-1 encoder weights, finetune for driving QA with risk
- [ ] Dataset builder update: emit raw vector tensors alongside text data for prefix training
- [ ] Modified inference: load vector encoder + LoRA weights, run prefix-conditioned generation
- [ ] Evaluation comparison: text-baseline vs vector-prefix pipeline on same data split
- [ ] Config unification: USE_VECTOR_PREFIX / USE_LORA flags actually control training path

### Out of Scope

- LLaMA backbone — using FLAN-T5 (simpler, smaller, matches existing stack)
- Perceiver architecture for encoder — using simpler transformer encoder (sufficient for 10-object, 8D vectors)
- Wayve simulator data / .pkl files — using nuScenes as data source
- Per-type descriptor MLPs (route, vehicle, pedestrian, ego separate encoders) — using unified encoder for flat vector array
- Image/camera modality — paper and project are vector-only
- Real-time deployment — research/training focus
- Multi-GPU DDP training — single 48GB GPU target

## Context

**Paper reference:** "Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving" (Long Chen et al., Wayve, ICRA 2024). arXiv:2310.01957. Official code: github.com/wayveai/Driving-with-LLMs.

**Current codebase status:** Risk-aware two-stage text pipeline. Vectors are serialized as comma-separated strings and fed to T5 as text tokens. The VectorPrefixEncoder exists only as an incomplete stub — forward() references undefined members (obj_in, encoder, to_prefix). Config declares USE_VECTOR_PREFIX=True and USE_LORA=True but no code path consumes these flags.

**Key architectural difference from paper:** The paper uses 4 separate typed descriptors (route_30x17, vehicle_30x33, pedestrian_20x9, ego_31) processed by per-type MLPs into a Perceiver. Our adaptation uses a single flat tensor (10x8) through a unified encoder. This is simpler but loses the semantic grouping advantage. The 8D vector [rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id] captures the essential per-object information.

**Risk module (our extension):** Not in the original paper. We keep it as a separate text signal injected into Stage-2 prompts. This means Stage-1 trains the vector encoder purely on vector→caption alignment, while Stage-2 benefits from both learned vector representations (via the encoder inherited from Stage-1) and explicit risk summaries.

**Compute:** 48GB VRAM GPU available. FLAN-T5-base has ~250M params; with LoRA (r=8) and vector encoder overhead, well within budget.

## Constraints

- **Model backbone**: FLAN-T5-base (google/flan-t5-base) — existing choice, keep for comparability
- **Data source**: nuScenes-mini initially, nuScenes-trainval for full experiments — existing pipeline
- **GPU**: Single 48GB VRAM — sufficient for FLAN-T5 + encoder + LoRA
- **Dependencies**: PyTorch, transformers, peft (for LoRA), datasets — standard ML stack
- **Backward compatibility**: Text-baseline pipeline must remain runnable (USE_VECTOR_PREFIX=False should fall back to current behavior)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep FLAN-T5 instead of migrating to LLaMA | Simpler, smaller, matches existing training/eval code; concept transfer is valid | — Pending |
| Keep nuScenes instead of Wayve .pkl data | User has existing data pipeline; Wayve data uses different descriptor format | — Pending |
| Unified vector encoder instead of per-type MLPs | Only 10 objects × 8D; Perceiver is overkill; simpler encoder is adequate | — Pending |
| Keep risk module alongside vector prefix | Risk is an additive signal not in original paper; provides richer Stage-2 context | — Pending |
| LoRA on q/v projections with r=8, alpha=16 | Config already declares these values; standard efficient finetuning | — Pending |
| Freeze T5 base, train encoder + LoRA only | Paper approach; prevents catastrophic forgetting of language capabilities | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-04 after initialization*
