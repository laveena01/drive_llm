# Architecture Documentation

## System Overview

An end-to-end LLM-based autonomous driving decision system built on FLAN-T5-base.
Raw nuScenes sensor data is distilled into structured 8D object vectors, which are
encoded into learned prefix embeddings and injected into T5's encoder for two tasks:
1. **Stage 1**: Scene description (vectors → natural language caption)
2. **Stage 2**: Driving action (caption + risk → structured control output)

---

## Full Pipeline Diagram

```
nuScenes Dataset (v1.0-trainval)
         │
         ▼
┌─────────────────────────────────────────────┐
│           nuscenes_data.py                  │
│                                             │
│  get_object_vectors_for_sample()            │
│  ─────────────────────────────              │
│  • Extract ego pose, rotation matrix        │
│  • Per annotation: transform to ego frame   │
│  • Compute: rel_x, rel_y, dist,             │
│             rel_vx, rel_vy, heading,        │
│             size, type_id                   │
│  • Sort by distance, pad to MAX_OBJECTS=10  │
│                                             │
│  Output: (10, 8) float32 array              │
└─────────────────────┬───────────────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
┌────────────────┐      ┌───────────────────────────┐
│  langen.py     │      │  risk_calculator.py        │
│                │      │                             │
│ Oracle caption │      │ calculate_risk_from_vectors │
│ (rule-based)   │      │                             │
│                │      │ Per object:                 │
│ Describes each │      │  • TTC (closing speed)      │
│ object using   │      │  • Collision risk           │
│ distance/speed │      │  • Pedestrian weight        │
│ heuristics     │      │  • Uncertainty proxy        │
│                │      │  • Regulatory (red light)   │
│ Output: str    │      │                             │
└───────┬────────┘      │ Aggregation: log-sum-exp    │
        │               │                             │
        │               │ Output: FrameRiskData        │
        │               │  risk_level: CRITICAL/HIGH/ │
        │               │    MODERATE/LOW/MINIMAL     │
        │               │  min_ttc, max_collision_%   │
        └───────┬───────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│         datasets_builder.py                 │
│                                             │
│  Captioning samples (Stage 1):              │
│   input:  "Describe:"                       │
│   target: oracle caption                    │
│   vectors: (10, 8) float list               │
│   num_objects: int                          │
│                                             │
│  QA samples (Stage 2):                      │
│   input:  ### OBSERVATION / RISK /          │
│           QUESTION / OUTPUT FORMAT          │
│   target: structured action or risk text    │
│   question_type: "action" | "risk"          │
│   5 action variants + 3 risk variants       │
│   = 8 QA samples per frame                  │
└─────────────────┬───────────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
STAGE 1 TRAINING           STAGE 2 TRAINING
```

---

## Stage 1: Vector → Caption

### Input Format
```
Text: "Describe:"
Vectors: (10, 8) tensor — 10 objects × 8 features
         [rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id]
```

### Vector Format (8D per object)
| Index | Feature | Unit | Description |
|-------|---------|------|-------------|
| 0 | rel_x | m | Forward distance in ego frame (positive = ahead) |
| 1 | rel_y | m | Lateral distance in ego frame (positive = left) |
| 2 | dist | m | Euclidean distance sqrt(x²+y²) |
| 3 | rel_vx | m/s | Forward velocity component in ego frame |
| 4 | rel_vy | m/s | Lateral velocity component in ego frame |
| 5 | heading | rad | Object yaw angle in ego frame |
| 6 | size | m | Average of (length + width) / 2 |
| 7 | type_id | int | 0=car, 1=pedestrian, 2=traffic_light, 3=other |

Objects sorted by distance. Padded with zeros up to MAX_OBJECTS=10.

### VectorPrefixEncoder Architecture

```
Input: (B, 10, 8)  +  num_objects: (B,)
         │
         ├── continuous: (B, 10, 7)   ← features 0-6
         └── type_id:    (B, 10)      ← feature 7
                              │
                    nn.Embedding(4, 16)  ← learned type embedding
                              │
                    type_embeds: (B, 10, 16)
                              │
         ┌────────────────────┘
         │ concat [continuous ‖ type_embeds]
         ▼
    (B, 10, 23)  →  Linear(23, 256)  →  (B, 10, 256)
         │
         + object_position_embed(0..9)   ← nn.Embedding(10, 256)
         │
         ▼
    TransformerEncoder(d=256, heads=4, layers=2, FFN=1024, GELU, pre-norm)
         │   (self-attention across 10 objects)
         │   (padding mask applied for empty slots)
         ▼
    (B, 10, 256)
         │
    token_expansion: Linear(256, 6*256)  → reshape → (B, 10, 6, 256)
         │   (each object expands to 6 dedicated prefix slots)
         │
    reshape → (B, 60, 256)     ← object prefix tokens
         │
    + 4 global learnable tokens (nn.Parameter)
         │
    → (B, 64, 256)
         │
    + slot_position_embed(0..63)   ← nn.Embedding(64, 256)
         │
    LayerNorm + zero-out padded slots
         │
    Linear(256, 768)   ← project to T5 embedding space
         │
         ▼
    (B, 64, 768)  ← prefix embeddings
```

**Key design**: Per-object slot allocation (6 tokens/object) instead of pooling.
Prevents mean-field collapse that caused repetitive captions in previous versions.

### VectorPrefixT5 — Prefix Injection

```python
# _build_prefix_inputs():
prefix = vector_encoder(vectors, num_objects)          # (B, 64, 768)
text_embeds = t5.shared(input_ids)                     # (B, seq_len, 768)
inputs_embeds = cat([prefix, text_embeds], dim=1)      # (B, 64+seq_len, 768)
attention_mask = cat([ones(B,64), attention_mask], 1)  # extend mask for prefix

# Forward call:
t5(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
```

T5 sees prefix tokens as if they were the first 64 tokens of the input sequence.
No changes to T5 internals — standard encoder-decoder attention applies.

### Stage 1 Output Format (Oracle Target)
```
There are 2 cars and 1 pedestrian nearby.
A medium-sized car is 12.5 meters straight ahead, moving fast.
A small car is 18.2 meters ahead-left, moving slowly.
A small pedestrian is 6.1 meters slightly to the right, moving steadily.
My current speed is 10.0 m/s.
The route continues straight ahead.
```

### Stage 1 Training
- **Loop**: Custom PyTorch (not HF Trainer)
- **Optimizer**: AdamW, LR=2e-5, weight_decay=0
- **Scheduler**: Linear warmup (100 steps) → linear decay
- **Multi-GPU**: HuggingFace Accelerate (`accelerator.prepare()`)
- **Validation**: Forward-pass loss only in multi-GPU (beam search skipped to avoid NCCL timeout)
- **Checkpointing**: Best by val loss

---

## Stage 2: Caption + Risk → Action

### Input Format (ACTION question)
```
### OBSERVATION
[Stage 1 caption — describes scene in natural language]

### RISK
Risk level: MODERATE. Minimum time-to-collision: 4.5 seconds.
Maximum collision risk: 35%.

### QUESTION
How should the car drive in this situation and why?

### OUTPUT FORMAT
You are an AI Driver.
Return EXACTLY 5 lines (each on its own line), and nothing else:
Here are my actions:
- Accelerator pedal: <0-100>%
- Brake pedal: <0-100>%
- Steering: <left/straight/right>
Reason: <one short sentence>
```

### Target Output (ACTION)
```
Here are my actions:
- Accelerator pedal: 10%
- Brake pedal: 20%
- Steering: straight
Reason: Moderate risk with low TTC=4.5s, braking to increase safety margin.
```

### Target Output (RISK question)
```
Risk level: MODERATE.
Reason: TTC=4.5s, collision=35%.
```

### Risk Score Components
```
FrameRiskData per frame:
├── Collision Risk  (weight 0.40)
│   ├── TTC risk:         55% — time-to-collision from closing speed
│   ├── Distance baseline: 25% — static nearby objects still risky
│   └── Lateral conflict:  20% — near-path / cut-in detection
│   └── Front cone gating: soft sigmoid over ±45° arc
│
├── Pedestrian Risk (weight 0.30)
│   └── Activates for pedestrians/cyclists/motorcycles
│   └── sqrt(front_weight) — allows risk even slightly off-path
│
├── Uncertainty Risk (weight 0.20)
│   └── Combines distance (farther = uncertain) + small size + high speed
│
└── Regulatory Risk (weight 0.10)
    └── Red traffic light detection
```

Aggregation: log-sum-exp over all objects (soft-max over risks, not hard max).

Thresholds:
- CRITICAL: avg_total_risk ≥ 0.7
- HIGH: ≥ 0.5
- MODERATE: ≥ 0.3
- LOW: ≥ 0.15
- MINIMAL: < 0.15

### Oracle Action Labels (`policy_from_risk`)
```
CRITICAL  → Accelerator: 0%,  Brake: 80%, Steering: straight
HIGH      → Accelerator: 5%,  Brake: 50%, Steering: straight
MODERATE  → Accelerator: 30%, Brake: 20%, Steering: straight
LOW       → Accelerator: 60%, Brake: 5%,  Steering: straight
MINIMAL   → Accelerator: 80%, Brake: 0%,  Steering: straight
```

### Stage 2 Training
- **Framework**: HuggingFace Trainer
- **Model**: FLAN-T5-base (optionally + LoRA, currently disabled)
- **Dataset**: 8 QA samples per frame × all frames
- **Multi-GPU**: HF Trainer handles DDP natively

---

## Data Flow Summary

```
nuScenes sample_token
    │
    ▼ nuscenes_data.py
(10, 8) vectors  +  num_objects
    │
    ├──────────────────────────────┐
    │                              │
    ▼ langen.py                    ▼ risk_calculator.py
oracle_caption (str)          FrameRiskData
    │                              │
    └──────────┬───────────────────┘
               │
               ▼ datasets_builder.py
    Captioning sample:             QA sample:
    ┌──────────────────┐           ┌────────────────────────────┐
    │ input: "Describe:"│           │ input: OBS+RISK+Q+FORMAT   │
    │ target: caption  │           │ target: action or risk text │
    │ vectors: (10,8)  │           │ vectors: (10,8)             │
    │ num_objects: int │           │ num_objects: int            │
    └────────┬─────────┘           └────────────┬───────────────┘
             │                                  │
             ▼ Stage 1                          ▼ Stage 2
    VectorPrefixT5                     FLAN-T5 (HF Trainer)
    (VectorPrefixEncoder + T5)
             │                                  │
             ▼                                  ▼
    Generated caption               Structured action output
```

---

## File Responsibilities

| File | Responsibility |
|------|---------------|
| `config.py` | Single source of truth for all hyperparameters |
| `nuscenes_data.py` | Raw sensor → 8D vectors per frame |
| `langen.py` | Rule-based oracle captions for Stage 1 supervision |
| `risk_calculator.py` | Multi-dimensional risk score → oracle action labels |
| `datasets_builder.py` | Builds JSON datasets consumed by training |
| `vector_encoder.py` | `VectorPrefixEncoder` — Transformer → 64 prefix tokens |
| `vector_prefix_t5.py` | `VectorPrefixT5` — wraps T5 with prefix injection |
| `data_collator.py` | Batches samples into tensors for DataLoader |
| `lora_utils.py` | Checkpoint save/load, optional LoRA wrapping |
| `training.py` | Stage 1 custom loop + Stage 2 HF Trainer |
| `main.py` | Orchestrates dataset build → Stage 1 → Stage 2 |

---

## Model Parameter Count (Approximate)

| Component | Parameters |
|-----------|-----------|
| FLAN-T5-base | ~250M |
| VectorPrefixEncoder | ~2M |
| **Total (full fine-tune)** | **~252M** |
| Total (with LoRA r=8) | ~251M (T5 frozen) + ~0.5M LoRA |
