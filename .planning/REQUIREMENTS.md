# Requirements: Drive-LLM Vector Prefix Pipeline

**Defined:** 2026-04-04
**Core Value:** The vector prefix encoder must learn to align numeric object vectors to T5's representation space, producing measurably better or comparable driving QA results versus the text-serialization baseline.

## v1 Requirements

### Vector Encoder

- [ ] **VENC-01**: VectorPrefixEncoder __init__ constructs obj_in (Linear), transformer encoder layers, and to_prefix (Linear) matching config dimensions
- [ ] **VENC-02**: VectorPrefixEncoder forward() takes (B, MAX_OBJECTS, VECTOR_DIM) tensor + num_objects mask and returns (B, PREFIX_LEN, t5_d_model) embeddings
- [ ] **VENC-03**: Padding mask correctly excludes zero-padded object slots from transformer attention
- [ ] **VENC-04**: VectorEncoderConfig dataclass fully parameterizes all dimensions (max_objects, vector_dim, hidden_dim, prefix_len, t5_d_model, n_layers, n_heads, dropout)
- [ ] **VENC-05**: Unit test validates output shapes and gradient flow through encoder

### Prefix Injection

- [ ] **PINJ-01**: T5 encoder input embeddings are prepended with vector prefix embeddings before forward pass
- [ ] **PINJ-02**: Attention mask is extended to cover prefix tokens (all ones for prefix, original mask for text)
- [ ] **PINJ-03**: Labels are not affected by prefix (prefix tokens get -100 label, text labels unchanged)
- [ ] **PINJ-04**: Custom model wrapper (VectorPrefixT5) encapsulates prefix injection logic
- [ ] **PINJ-05**: Generation (model.generate()) correctly handles prefix injection via prepare_inputs_for_generation or inputs_embeds

### LoRA Integration

- [ ] **LORA-01**: LoRA adapters applied to T5 encoder and decoder q/v projection layers using peft library
- [ ] **LORA-02**: Base T5 weights are frozen; only LoRA adapters + vector encoder are trainable
- [ ] **LORA-03**: LoRA config uses r=8, alpha=16, dropout=0.05 as declared in config.py
- [ ] **LORA-04**: Trainable parameter count is logged at training start
- [ ] **LORA-05**: LoRA + encoder weights can be saved and loaded independently from base model

### Dataset Adaptation

- [ ] **DATA-01**: Dataset builder emits raw vector tensors (numpy arrays) in addition to text fields for each sample
- [ ] **DATA-02**: Stage-1 samples include: vector tensor, num_objects count, text prompt, caption target
- [ ] **DATA-03**: Stage-2 samples include: vector tensor, num_objects, caption, risk_text, question, target
- [ ] **DATA-04**: Custom data collator creates batched vector tensors with proper padding
- [ ] **DATA-05**: Existing text-only JSON format backward compatible (USE_VECTOR_PREFIX=False path unchanged)

### Stage-1 Training (Vector → Caption)

- [ ] **STG1-01**: Stage-1 trains VectorPrefixEncoder + LoRA, with T5 base frozen
- [ ] **STG1-02**: Input: vector prefix embeddings + "Describe the driving scene" text prompt → Target: lanGen caption
- [ ] **STG1-03**: Training uses AdaFactor optimizer, gradient accumulation, gradient clipping
- [ ] **STG1-04**: Validation generates captions using prefix-conditioned model and computes BLEU-1/ROUGE-L
- [ ] **STG1-05**: Checkpoint saves include vector encoder state_dict + LoRA adapter weights
- [ ] **STG1-06**: Training logs loss curves, validation metrics, sample predictions per epoch

### Stage-2 Training (Driving QA)

- [ ] **STG2-01**: Stage-2 initializes from Stage-1 checkpoint (encoder weights + LoRA weights carry over)
- [ ] **STG2-02**: Stage-2 continues training encoder + LoRA on QA task (caption + risk + question → action/risk answer)
- [ ] **STG2-03**: Risk text injection into Stage-2 prompts preserved (same as current pipeline)
- [ ] **STG2-04**: Evaluation runs both oracle_caption and stage1_caption modes
- [ ] **STG2-05**: Action accuracy, missed brake rate, risk level accuracy metrics computed
- [ ] **STG2-06**: Evaluation includes brake MAE and unsafe_continue_high_rate safety metrics

### Inference

- [ ] **INFR-01**: Inference loads base T5 + LoRA adapters + vector encoder from checkpoint directory
- [ ] **INFR-02**: Inference accepts raw vector arrays and runs full prefix pipeline (encode → inject → generate)
- [ ] **INFR-03**: 5-line action format enforcement preserved (enforce_5_lines)
- [ ] **INFR-04**: Inference supports both action and risk question types

### Config & Switchability

- [ ] **CONF-01**: USE_VECTOR_PREFIX=True activates prefix pipeline; False falls back to text-only baseline
- [ ] **CONF-02**: USE_LORA=True activates LoRA; False trains full model (for ablation)
- [ ] **CONF-03**: FREEZE_BASE_MODEL flag controls whether T5 base weights are frozen
- [ ] **CONF-04**: All vector encoder hyperparameters controlled from config.py
- [ ] **CONF-05**: Single main.py entry point handles both pipelines based on config flags

### Evaluation & Comparison

- [ ] **EVAL-01**: Side-by-side eval metrics: text-baseline vs vector-prefix on same validation split
- [ ] **EVAL-02**: Metrics JSON includes pipeline type identifier (text_baseline / vector_prefix)
- [ ] **EVAL-03**: Comparison report shows delta in action_accuracy, missed_brake_rate, BLEU-1, ROUGE-L

## v2 Requirements

### Enhanced Encoder

- **VENC2-01**: Per-type sub-encoders (vehicle MLP, pedestrian MLP) before transformer, closer to paper
- **VENC2-02**: Ego state vector as separate input to encoder (not part of object array)
- **VENC2-03**: Route information encoding (if available from nuScenes map data)

### nuScenes Full Scale

- **DATA2-01**: Support nuScenes v1.0-trainval (full dataset, not just mini)
- **DATA2-02**: Multi-GPU DDP training support for larger dataset

### Advanced Training

- **STG2-01**: Curriculum learning: increase question difficulty over epochs
- **STG2-02**: Weighted loss on number tokens (like paper's weighted_mask)

## Out of Scope

| Feature | Reason |
|---------|--------|
| LLaMA backbone migration | Using FLAN-T5 — concept transfer is valid, simpler stack |
| Perceiver architecture | Overkill for 10-object 8D vectors; transformer encoder sufficient |
| Wayve .pkl data format | Using nuScenes; different descriptor format |
| Image/camera modality | Paper and project are vector-only |
| Real-time deployment | Research/training focus |
| Multi-GPU DDP | Single 48GB GPU target for v1 |
| WandB integration | Using local logging for now |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| VENC-01 | Phase 1 | Pending |
| VENC-02 | Phase 1 | Pending |
| VENC-03 | Phase 1 | Pending |
| VENC-04 | Phase 1 | Pending |
| VENC-05 | Phase 1 | Pending |
| PINJ-01 | Phase 2 | Pending |
| PINJ-02 | Phase 2 | Pending |
| PINJ-03 | Phase 2 | Pending |
| PINJ-04 | Phase 2 | Pending |
| PINJ-05 | Phase 2 | Pending |
| LORA-01 | Phase 3 | Pending |
| LORA-02 | Phase 3 | Pending |
| LORA-03 | Phase 3 | Pending |
| LORA-04 | Phase 3 | Pending |
| LORA-05 | Phase 3 | Pending |
| DATA-01 | Phase 4 | Pending |
| DATA-02 | Phase 4 | Pending |
| DATA-03 | Phase 4 | Pending |
| DATA-04 | Phase 4 | Pending |
| DATA-05 | Phase 4 | Pending |
| STG1-01 | Phase 5 | Pending |
| STG1-02 | Phase 5 | Pending |
| STG1-03 | Phase 5 | Pending |
| STG1-04 | Phase 6 | Pending |
| STG1-05 | Phase 5 | Pending |
| STG1-06 | Phase 6 | Pending |
| STG2-01 | Phase 7 | Pending |
| STG2-02 | Phase 7 | Pending |
| STG2-03 | Phase 7 | Pending |
| STG2-04 | Phase 8 | Pending |
| STG2-05 | Phase 8 | Pending |
| STG2-06 | Phase 8 | Pending |
| INFR-01 | Phase 9 | Pending |
| INFR-02 | Phase 9 | Pending |
| INFR-03 | Phase 9 | Pending |
| INFR-04 | Phase 9 | Pending |
| CONF-01 | Phase 10 | Pending |
| CONF-02 | Phase 10 | Pending |
| CONF-03 | Phase 10 | Pending |
| CONF-04 | Phase 10 | Pending |
| CONF-05 | Phase 10 | Pending |
| EVAL-01 | Phase 11 | Pending |
| EVAL-02 | Phase 11 | Pending |
| EVAL-03 | Phase 11 | Pending |

**Coverage:**
- v1 requirements: 42 total
- Mapped to phases: 42
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-04*
*Last updated: 2026-04-04 after initial definition*
