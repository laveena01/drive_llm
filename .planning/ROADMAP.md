# Roadmap: Drive-LLM Vector Prefix Pipeline

**Created:** 2026-04-04
**Phases:** 11
**Granularity:** Fine
**Mode:** Sequential

## Phase Overview

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|-----------------|
| 1 | Vector Encoder Module | Complete VectorPrefixEncoder with full __init__ and forward | VENC-01..05 | 5 |
| 2 | Prefix Injection into T5 | Wire encoder output into T5 encoder as prefix embeddings | PINJ-01..05 | 5 |
| 3 | LoRA Integration | Add LoRA adapters to T5 via peft library | LORA-01..05 | 5 |
| 4 | Dataset Adaptation | Emit vector tensors from dataset builder, custom collator | DATA-01..05 | 5 |
| 5 | Stage-1 Training Loop | Train vector encoder + LoRA for caption generation | STG1-01..03,05 | 4 |
| 6 | Stage-1 Validation | Generate captions, compute metrics, log results | STG1-04,06 | 2 |
| 7 | Stage-2 Training Loop | Fine-tune QA from Stage-1 checkpoint with risk | STG2-01..03 | 3 |
| 8 | Stage-2 Evaluation | Full eval with action/risk metrics in both modes | STG2-04..06 | 3 |
| 9 | Inference Pipeline | Load and run prefix-conditioned inference | INFR-01..04 | 4 |
| 10 | Config Switchability | USE_VECTOR_PREFIX/USE_LORA flags control full pipeline | CONF-01..05 | 5 |
| 11 | Baseline Comparison | Side-by-side text vs prefix evaluation report | EVAL-01..03 | 3 |

---

## Phase 1: Vector Encoder Module

**Goal:** Complete the VectorPrefixEncoder with proper __init__, forward, shape validation, and unit test.

**Requirements:** VENC-01, VENC-02, VENC-03, VENC-04, VENC-05

**Success criteria:**
1. VectorPrefixEncoder can be instantiated with VectorEncoderConfig
2. forward() accepts (B, 10, 8) input and returns (B, 16, 768) output
3. Padding mask correctly zeros out attention for padded object slots
4. Gradient flows from output back through all encoder parameters
5. Unit test passes with multiple batch sizes and object counts

**Files:** `llm_driving/vector_encoder.py`

**UI hint**: no

---

## Phase 2: Prefix Injection into T5

**Goal:** Create VectorPrefixT5 wrapper that prepends encoder output to T5 encoder input embeddings.

**Requirements:** PINJ-01, PINJ-02, PINJ-03, PINJ-04, PINJ-05

**Success criteria:**
1. VectorPrefixT5 wraps AutoModelForSeq2SeqLM and injects prefix before T5 encoder
2. Attention mask is extended with ones for prefix positions
3. Label indices are not shifted by prefix (decoder-side labels unaffected in encoder-decoder model)
4. model.generate() works with prefix injection via inputs_embeds override
5. Smoke test: random vectors + text prompt → T5 produces output tokens (untrained, just verifying forward pass)

**Files:** `llm_driving/vector_prefix_t5.py` (new)

**UI hint**: no

---

## Phase 3: LoRA Integration

**Goal:** Wire peft LoRA adapters into T5 and verify only encoder + LoRA params are trainable.

**Requirements:** LORA-01, LORA-02, LORA-03, LORA-04, LORA-05

**Success criteria:**
1. get_peft_model() applies LoRA to T5 encoder+decoder q/v projections
2. model.print_trainable_parameters() shows <5% trainable (LoRA + encoder only)
3. Base T5 parameters have requires_grad=False
4. Vector encoder parameters have requires_grad=True
5. save/load roundtrip: save LoRA adapter + encoder weights, reload, verify identical outputs

**Files:** `llm_driving/vector_prefix_t5.py` (extend), `llm_driving/config.py` (verify flags)

**UI hint**: no

---

## Phase 4: Dataset Adaptation

**Goal:** Modify dataset builder to emit vector tensors alongside text, create custom data collator.

**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04, DATA-05

**Success criteria:**
1. captioning JSON samples include "vectors" (list of floats) and "num_objects" (int) fields
2. QA JSON samples include same vector fields plus all existing text fields
3. Custom VectorDataCollator batches vectors into (B, MAX_OBJECTS, VECTOR_DIM) tensors
4. When USE_VECTOR_PREFIX=False, existing text-only code path works unchanged
5. Round-trip test: build mini dataset → load → collate → verify tensor shapes

**Files:** `llm_driving/datasets_builder.py` (modify), `llm_driving/data_collator.py` (new)

**UI hint**: no

---

## Phase 5: Stage-1 Training Loop

**Goal:** Implement the Stage-1 training function that trains vector encoder + LoRA for caption generation.

**Requirements:** STG1-01, STG1-02, STG1-03, STG1-05

**Success criteria:**
1. train_stage1_prefix() creates VectorPrefixT5 with LoRA, loads data, trains
2. Vector tensors flow through encoder → prefix → T5 → caption loss
3. Only encoder + LoRA params receive gradients
4. Checkpoint saves both encoder state_dict and LoRA adapter weights
5. Training loop runs without error on a small data sample (smoke test with 2-3 samples)

**Files:** `llm_driving/training.py` (extend with train_stage1_prefix function)

**UI hint**: no

---

## Phase 6: Stage-1 Validation

**Goal:** Add validation loop that generates captions from prefix-conditioned model and computes metrics.

**Requirements:** STG1-04, STG1-06

**Success criteria:**
1. After training, model generates captions from validation vectors using prefix pipeline
2. BLEU-1 and ROUGE-L computed between generated and ground-truth captions
3. Sample predictions logged (input vectors → predicted caption vs ground truth)
4. Metrics and predictions saved to JSON files under stage1 output directory
5. Loss curves visible in training logs

**Files:** `llm_driving/training.py` (extend validation in train_stage1_prefix)

**UI hint**: no

---

## Phase 7: Stage-2 Training Loop

**Goal:** Fine-tune the model for driving QA starting from Stage-1 checkpoint, with risk injection.

**Requirements:** STG2-01, STG2-02, STG2-03

**Success criteria:**
1. Stage-2 loads Stage-1 checkpoint (encoder + LoRA weights)
2. Training continues on QA data: vector prefix + caption + risk + question → answer
3. Risk text is injected into Stage-2 prompts (same as current pipeline)
4. Encoder + LoRA params continue receiving gradients
5. Training runs without error on small QA data sample

**Files:** `llm_driving/training.py` (extend with train_stage2_prefix function)

**UI hint**: no

---

## Phase 8: Stage-2 Evaluation

**Goal:** Full evaluation with action accuracy, risk metrics, safety metrics in both oracle and stage1 modes.

**Requirements:** STG2-04, STG2-05, STG2-06

**Success criteria:**
1. oracle_caption mode: Stage-2 evaluates with ground-truth captions + vector prefix
2. stage1_caption mode: Stage-1 generates caption, then Stage-2 evaluates with it
3. Action accuracy, missed_brake_rate, brake MAE, unsafe_continue metrics computed
4. Risk level accuracy computed for risk questions
5. All metrics saved to eval_metrics.json with mode labels

**Files:** `llm_driving/training.py` (extend eval in train_stage2_prefix)

**UI hint**: no

---

## Phase 9: Inference Pipeline

**Goal:** Standalone inference that loads checkpoint and runs prefix-conditioned generation.

**Requirements:** INFR-01, INFR-02, INFR-03, INFR-04

**Success criteria:**
1. inference.py loads base T5 + LoRA adapter + vector encoder from checkpoint dir
2. Accepts raw vector arrays as input, runs full encode → prefix → generate pipeline
3. 5-line action format enforcement (enforce_5_lines) works on prefix-pipeline output
4. Both action and risk question types supported
5. Can run on a single sample from command line

**Files:** `llm_driving/inference.py` (modify to support prefix pipeline)

**UI hint**: no

---

## Phase 10: Config Switchability

**Goal:** USE_VECTOR_PREFIX and USE_LORA flags control which pipeline runs end-to-end.

**Requirements:** CONF-01, CONF-02, CONF-03, CONF-04, CONF-05

**Success criteria:**
1. USE_VECTOR_PREFIX=False → current text pipeline runs exactly as before (no regression)
2. USE_VECTOR_PREFIX=True → vector prefix pipeline runs
3. USE_LORA=False with USE_VECTOR_PREFIX=True → full model training (no LoRA, just encoder + full T5)
4. FREEZE_BASE_MODEL=False → all T5 params trainable (ablation mode)
5. main.py handles both paths cleanly with config-driven branching

**Files:** `llm_driving/config.py`, `main.py`, `llm_driving/training.py`

**UI hint**: no

---

## Phase 11: Baseline Comparison

**Goal:** Run both pipelines on same data split, produce side-by-side comparison report.

**Requirements:** EVAL-01, EVAL-02, EVAL-03

**Success criteria:**
1. Text-baseline eval metrics saved with pipeline_type="text_baseline"
2. Vector-prefix eval metrics saved with pipeline_type="vector_prefix"
3. Comparison report shows delta for: action_accuracy, missed_brake_rate, BLEU-1, ROUGE-L
4. Report in markdown format under run directory

**Files:** `llm_driving/evaluation.py` (new), comparison report output

**UI hint**: no

---

## Dependencies

```
Phase 1 (Encoder) → Phase 2 (Injection) → Phase 3 (LoRA)
                                                ↓
Phase 4 (Data) ─────────────────────────→ Phase 5 (Stage-1 Train)
                                                ↓
                                          Phase 6 (Stage-1 Val)
                                                ↓
                                          Phase 7 (Stage-2 Train)
                                                ↓
                                          Phase 8 (Stage-2 Eval)
                                                ↓
                                          Phase 9 (Inference)
                                                ↓
                                          Phase 10 (Switchability)
                                                ↓
                                          Phase 11 (Comparison)
```

Note: Phase 4 (Data) can be developed in parallel with Phases 1-3 but is listed as Phase 4 for sequential execution.

---
*Roadmap created: 2026-04-04*
*Last updated: 2026-04-04 after initial creation*
