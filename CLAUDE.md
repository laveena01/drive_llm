# CLAUDE.md — Project Context for Claude Code

## Project Overview

LLM-based autonomous driving decision system using nuScenes dataset.
Converts raw sensor-derived object vectors into natural language scene descriptions,
then generates structured driving actions. MTech thesis project.

**Branch**: `claude/nostalgic-bardeen`
**Server path**: `/u/student/2021/cs21resch15003/drive_llm/`
**nuScenes data**: `/u/student/2021/cs21resch15003/data/nuscenes` (v1.0-trainval)

---

## Repository Structure

```
drive_llm/
├── main.py                        # Entry point — builds data, runs Stage 1 + Stage 2
├── inference_main.py              # Inference-only script
├── env.yml                        # Conda environment (Python 3.10, PyTorch, HF, accelerate)
└── llm_driving/
    ├── config.py                  # All hyperparameters and paths
    ├── nuscenes_data.py           # nuScenes vector extraction
    ├── datasets_builder.py        # Builds captioning + QA datasets
    ├── langen.py                  # Oracle caption generator (rule-based)
    ├── risk_calculator.py         # Multi-dimensional risk scoring (thesis novelty)
    ├── vector_encoder.py          # VectorPrefixEncoder — per-object slot Transformer
    ├── vector_prefix_t5.py        # VectorPrefixT5 — wraps T5 with prefix injection
    ├── data_collator.py           # VectorPrefixDataCollator for DataLoader
    ├── lora_utils.py              # save/load checkpoint, apply_lora
    └── training.py                # Stage 1 (custom loop) + Stage 2 (HF Trainer)
```

---

## Running the Code

### Single GPU
```bash
python main.py
```

### Multi-GPU (3 GPUs on DGX A100)
```bash
accelerate launch --num_processes=3 main.py
```

### Conda environment
```bash
conda activate drive_llm
```

---

## Two-Stage Pipeline

### Stage 1: Vectors → Caption
- Input: object vectors `(MAX_OBJECTS=10, VECTOR_DIM=8)` + minimal text prompt `"Describe:"`
- Model: `VectorPrefixT5` — FLAN-T5-base with 64 learned prefix tokens from `VectorPrefixEncoder`
- Output: natural language scene description
- Training: custom AdamW loop with linear warmup, HuggingFace Accelerate for multi-GPU
- Epochs: 10, LR: 2e-5, Batch: 4

### Stage 2: Caption + Risk + Question → Action
- Input: Stage 1 caption + risk summary + driving question
- Model: FLAN-T5 (optionally with LoRA, currently off)
- Output: structured 5-line action (Accelerator%, Brake%, Steering, Reason)
- Training: HuggingFace Trainer
- Epochs: 8, LR: 2e-5, Batch: 4

---

## Key Design Decisions

### VectorPrefixEncoder — Per-Object Slot Design
- Each object gets `TOKENS_PER_OBJECT=6` dedicated prefix slots (not shared pooling)
- 10 objects × 6 tokens = 60 object prefix slots + 4 global tokens = 64 total
- Prevents mean-field collapse that caused repetitive output in earlier versions
- Type ID (car/pedestrian/traffic_light/other) has its own `nn.Embedding` (16D)

### Risk Score (Thesis Novelty 1)
- Computed per-frame from object vectors via `risk_calculator.py`
- Components: collision risk (40%), pedestrian risk (30%), uncertainty (20%), regulatory (10%)
- Drives oracle action labels in Stage 2 via `policy_from_risk()`
- NOT fed to Stage 1 — captioning is pure object description

### Multi-GPU (Accelerate)
- Stage 1 uses `accelerator.prepare(model, optimizer, train_loader)`
- Validation runs on main process only (val_loader NOT prepared)
- In multi-GPU mode: beam search generation is skipped during validation (only forward-pass loss)
  to avoid NCCL watchdog timeout. BLEU/ROUGE = None during training.
- `InitProcessGroupKwargs(timeout=timedelta(hours=2))` is set but PyTorch 2.9 does not
  propagate it to per-op watchdog — the architectural fix (no beam search in val) is the real solution.

---

## Known Issues / Fixes Applied

| Issue | Fix | File |
|-------|-----|------|
| NCCL timeout during validation | Skip beam search in multi-GPU val (`generate=False`) | `training.py` |
| `InitProcessGroupKwargs` not propagating in PyTorch 2.9 | Architectural fix above | `training.py` |
| Silent crashes with no log output | Added try/except at step, epoch, val, checkpoint, final save | `training.py` |
| NaN loss crashing training | Detected and skipped with log message | `training.py` |
| CUDA OOM crashing training | Caught, logged, `empty_cache()`, batch skipped | `training.py` |

---

## Config Quick Reference (`llm_driving/config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_OBJECTS` | 10 | Objects per frame |
| `VECTOR_DIM` | 8 | `[rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id]` |
| `MODEL_NAME` | `google/flan-t5-base` | Base LLM |
| `PREFIX_LEN` | 64 | Prefix tokens injected to T5 encoder |
| `TOKENS_PER_OBJECT` | 6 | Prefix slots per object |
| `STAGE1_EPOCHS` | 10 | |
| `STAGE2_EPOCHS` | 8 | |
| `USE_LORA` | False | LoRA disabled, full fine-tuning |
| `FREEZE_BASE_MODEL` | False | T5 weights unfrozen |
| `GEN_NUM_BEAMS` | 4 | Beam search width |
| `NUSC_VERSION` | `v1.0-trainval` | Full dataset |

---

## Git Workflow

- Main branch: `main`
- Working branch: `claude/nostalgic-bardeen`
- Remote: `https://github.com/laveena01/drive_llm.git`
- Always pull before making changes on server: `git pull origin claude/nostalgic-bardeen`

---

## Pending Work

- Temporal context novelty (K-frame sliding window) — plan written, not yet implemented
- Real ego-motion (replace hardcoded `DEFAULT_EGO_SPEED=10.0` with actual nuScenes ego velocity)
- Create PR to main when Stage 1 training is confirmed stable
