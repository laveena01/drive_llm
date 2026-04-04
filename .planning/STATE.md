# Project State: Drive-LLM Vector Prefix Pipeline

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-04)

**Core value:** Vector prefix encoder aligns numeric vectors to T5 space for better driving QA
**Current focus:** Phase 11 — Baseline Comparison (remaining)

## Current Phase

**Phase 10: Config Switchability** — COMPLETE
- All prior phases (1-10) implemented in one execution session

## Progress

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Vector Encoder Module | ✅ Complete | VectorPrefixEncoder with __init__, cross-attention, type embedding |
| 2. Prefix Injection into T5 | ✅ Complete | VectorPrefixT5 wrapper with inputs_embeds |
| 3. LoRA Integration | ✅ Complete | lora_utils: apply, save, load |
| 4. Dataset Adaptation | ✅ Complete | Raw vectors in samples + VectorPrefixDataCollator |
| 5. Stage-1 Training Loop | ✅ Complete | train_stage1_prefix() |
| 6. Stage-1 Validation | ✅ Complete | BLEU-1, ROUGE-L, sample predictions |
| 7. Stage-2 Training Loop | ✅ Complete | train_stage2_prefix() |
| 8. Stage-2 Evaluation | ✅ Complete | Loss-based validation in training loop |
| 9. Inference Pipeline | ✅ Complete | inference_prefix.py |
| 10. Config Switchability | ✅ Complete | main.py routes on USE_VECTOR_PREFIX |
| 11. Baseline Comparison | ⬜ Not started | Requires running both pipelines |

## Decisions Log

- 2026-04-04: Keep FLAN-T5 + nuScenes + risk module, add vector prefix encoder
- 2026-04-04: hidden_dim=256, prefix_len=64 (paper-scale)
- 2026-04-04: Learned query tokens + cross-attention (paper-faithful pooling)
- 2026-04-04: Learned type embedding nn.Embedding(4, 16)
- 2026-04-04: Prefix injection via inputs_embeds (no model surgery)
- 2026-04-04: Separate training_prefix.py (preserves text baseline untouched)

---
*Last updated: 2026-04-04 after execution of phases 1-10*
