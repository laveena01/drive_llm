# llm_driving/training.py

import os
import json
import re
import copy
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from .config import (
    MODEL_NAME,
    STAGE1_OUTPUT_DIR,
    STAGE2_OUTPUT_DIR,
)
from . import config as cfg

logger = logging.getLogger("llm_driving")

# ---------------------------
# Helpers
# ---------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


PAPER_FORMAT_INSTRUCTION = (
    "\n\nYou are an AI Driver.\n"
    "Return EXACTLY 5 lines (with newlines), and nothing else:\n"
    "Here are my actions:\n"
    "- Accelerator pedal: <0-100>%\n"
    "- Brake pedal: <0-100>%\n"
    "- Steering: <left/straight/right>\n"
    "Reason: <one short sentence>\n"
    "Do NOT ask questions. Do NOT add extra text.\n"
)

RISK_FORMAT_INSTRUCTION = (
    "\n\nAnswer in 1-2 short lines using this template ONLY:\n"
    "Risk level: <CRITICAL|HIGH|MODERATE|LOW|MINIMAL>.\n"
    "Reason: <brief; mention TTC/collision/pedestrian if relevant>.\n"
    "Do NOT include driving controls.\n"
)

def _ensure_paper_format(prompt: str) -> str:
    p = prompt or ""
    if "Here are my actions:" in p and "Brake pedal" in p and "Steering" in p:
        return p
    return p + PAPER_FORMAT_INSTRUCTION

def _build_stage1_prompt(vec_str: str) -> str:
    return f"Describe the driving scene from object vectors:\n{vec_str}"

def _build_stage2_prompt_from_caption(
    caption: str,
    risk_text: str,
    qa_question: str,
    question_type: str,
) -> str:
    """
    Rebuilds Stage-2 prompt for stage1_caption evaluation.
    Must match the dataset_builder templates.
    """
    caption = caption or ""
    risk_text = risk_text or ""
    qa_question = qa_question or "How should the car drive in this situation and why?"
    question_type = (question_type or "action").strip().lower()

    if question_type == "risk":
        out_fmt = "### OUTPUT FORMAT\n" + RISK_FORMAT_INSTRUCTION.strip()
    else:
        out_fmt = "### OUTPUT FORMAT\n" + _ensure_paper_format("").strip()

    prompt = (
        "### OBSERVATION\n"
        f"{caption}\n\n"
        "### RISK\n"
        f"{risk_text}\n\n"
        "### QUESTION\n"
        f"{qa_question}\n\n"
        f"{out_fmt}\n"
    )
    return prompt


# ---------------------------
# Metrics: BLEU-1 + ROUGE-L
# ---------------------------

def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def _lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def rouge_l_f1(pred: str, ref: str) -> float:
    pred_toks = _normalize_text(pred).split()
    ref_toks = _normalize_text(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = _lcs_len(pred_toks, ref_toks)
    prec = lcs / max(1, len(pred_toks))
    rec = lcs / max(1, len(ref_toks))
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))

def bleu1(pred: str, ref: str) -> float:
    pred_toks = _normalize_text(pred).split()
    ref_toks = _normalize_text(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0

    ref_counts: Dict[str, int] = {}
    for t in ref_toks:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    used: Dict[str, int] = {}
    match = 0
    for t in pred_toks:
        used[t] = used.get(t, 0) + 1
        if t in ref_counts and used[t] <= ref_counts[t]:
            match += 1

    precision = match / max(1, len(pred_toks))

    bp = 1.0
    if len(pred_toks) < len(ref_toks):
        bp = float(np.exp(1 - (len(ref_toks) / max(1, len(pred_toks)))))

    return float(bp * precision)


# ---------------------------
# Parsing + format enforcement
# ---------------------------

def _extract_pct_after(text: str, key: str) -> Optional[int]:
    t = (text or "").lower()
    key = key.lower()
    if key not in t:
        return None
    after = t.split(key, 1)[1].strip()
    num = ""
    for ch in after:
        if ch.isdigit():
            num += ch
        else:
            break
    if not num:
        return None
    v = int(num)
    return max(0, min(100, v))

def _extract_brake_percent(text: str) -> Optional[int]:
    return _extract_pct_after(text, "brake pedal:")

def _extract_accel_percent(text: str) -> Optional[int]:
    return _extract_pct_after(text, "accelerator pedal:")

def _extract_steer(text: str) -> Optional[str]:
    t = (text or "").lower()
    m = re.search(r"steering:\s*(left|straight|right)", t)
    return m.group(1) if m else None

def _extract_reason(text: str) -> Optional[str]:
    t = (text or "").strip()
    m = re.search(r"reason:\s*(.+)$", t, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None

def enforce_5_lines(text: str) -> Tuple[str, int]:
    accel = _extract_accel_percent(text)
    brake = _extract_brake_percent(text)
    steer = _extract_steer(text)
    reason = _extract_reason(text)

    ok = 1 if (accel is not None and brake is not None and steer is not None and reason is not None) else 0

    if accel is None: accel = 0
    if brake is None: brake = 0
    if steer not in ("left", "straight", "right"): steer = "straight"
    if not reason: reason = "N/A"

    fixed = (
        "Here are my actions:\n"
        f"- Accelerator pedal: {accel}%\n"
        f"- Brake pedal: {brake}%\n"
        f"- Steering: {steer}\n"
        f"Reason: {reason}\n"
    )
    return fixed, ok

def _format_compliance_5line(text: str) -> int:
    if not text:
        return 0
    lines = [ln.rstrip("\n") for ln in (text or "").splitlines()]
    lines = [ln.strip() for ln in lines if ln.strip()]
    if len(lines) != 5:
        return 0
    if not lines[0].lower().startswith("here are my actions"):
        return 0
    if "accelerator pedal:" not in lines[1].lower():
        return 0
    if "brake pedal:" not in lines[2].lower():
        return 0
    if "steering:" not in lines[3].lower():
        return 0
    if not lines[4].lower().startswith("reason:"):
        return 0
    if _extract_brake_percent(text) is None or _extract_accel_percent(text) is None:
        return 0
    if _extract_steer(text) is None:
        return 0
    return 1

def _map_text_to_action_label(text: str) -> str:
    b = _extract_brake_percent(text)
    if b is None:
        return "OTHER"
    if b >= 30:
        return "BRAKE"
    if b >= 5:
        return "CAUTION"
    return "CONTINUE"

def _extract_risk_level(text: str) -> Optional[str]:
    """
    Extracts Risk level: <...> from risk answers.
    """
    m = re.search(r"risk level:\s*(CRITICAL|HIGH|MODERATE|LOW|MINIMAL)", text or "", flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def _print_eval_risk_summary(outputs: List[Dict], mode: str):
    """
    Keeps your old summary but only for ACTION questions.
    """
    action_outputs = [o for o in outputs if o.get("question_type", "action") == "action"]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"[RISK OUTCOMES] Evaluation Mode: {mode} (ACTION ONLY)")
    logger.info(f"{'=' * 60}")

    gt_actions = [o["gt_action"] for o in action_outputs]
    pred_actions = [o["pred_action"] for o in action_outputs]

    gt_counts = Counter(gt_actions)
    pred_counts = Counter(pred_actions)

    logger.info(f"\nTotal action samples: {len(action_outputs)}")

    logger.info("\n--- Ground Truth Action Distribution ---")
    for action in ["BRAKE", "CAUTION", "CONTINUE", "OTHER"]:
        count = gt_counts.get(action, 0)
        pct = (count / len(action_outputs)) * 100 if action_outputs else 0
        bar = "=" * int(pct / 2)
        logger.info(f"  {action:10s}: {count:4d} ({pct:5.1f}%) {bar}")

    logger.info("\n--- Predicted Action Distribution ---")
    for action in ["BRAKE", "CAUTION", "CONTINUE", "OTHER"]:
        count = pred_counts.get(action, 0)
        pct = (count / len(action_outputs)) * 100 if action_outputs else 0
        bar = "=" * int(pct / 2)
        logger.info(f"  {action:10s}: {count:4d} ({pct:5.1f}%) {bar}")

    logger.info("\n--- Action Prediction Confusion ---")
    correct = sum(1 for o in action_outputs if o["gt_action"] == o["pred_action"] and o["gt_action"] != "OTHER")
    total_valid = sum(1 for o in action_outputs if o["gt_action"] != "OTHER")
    logger.info(f"  Correct: {correct}/{total_valid} = {correct/max(1,total_valid)*100:.1f}%")

    for action in ["BRAKE", "CAUTION", "CONTINUE"]:
        gt_this = [o for o in action_outputs if o["gt_action"] == action]
        if gt_this:
            correct_this = sum(1 for o in gt_this if o["pred_action"] == action)
            logger.info(f"  {action}: {correct_this}/{len(gt_this)} = {correct_this/len(gt_this)*100:.1f}% recall")

    missed_brakes = [o for o in action_outputs if o["gt_action"] == "BRAKE" and o["pred_action"] != "BRAKE"]
    if missed_brakes:
        logger.info(f"\n--- Missed BRAKE Decisions (showing up to 3) ---")
        for o in missed_brakes[:3]:
            logger.info(f"  GT: BRAKE, Pred: {o['pred_action']}")
            logger.info(f"    Raw output: {o.get('prediction_raw', 'N/A')[:100]}...")

    logger.info(f"{'=' * 60}\n")


# ---------------------------
# Tokenization (text path)
# ---------------------------

def _tokenize_captioning(batch, tokenizer):
    model_inputs = tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=320,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"],
            truncation=True,
            padding="max_length",
            max_length=320,
        )["input_ids"]

    pad_id = tokenizer.pad_token_id
    labels = [[(tok if tok != pad_id else -100) for tok in seq] for seq in labels]
    model_inputs["labels"] = labels
    return model_inputs

def _tokenize_qa(batch, tokenizer):
    model_inputs = tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=384,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"],
            truncation=True,
            padding="max_length",
            max_length=192,
        )["input_ids"]

    pad_id = tokenizer.pad_token_id
    labels = [[(tok if tok != pad_id else -100) for tok in seq] for seq in labels]
    model_inputs["labels"] = labels
    return model_inputs


# ===================================================================
# Vector Prefix: Dataset wrapper for custom training loop
# ===================================================================

class VectorPrefixDataset(TorchDataset):
    """Wraps a list of sample dicts (from datasets_builder JSON) for DataLoader."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------
# Stage 1 training (text path - original)
# ---------------------------

def _train_stage1_text(captioning_path: str):
    """Original text-only Stage 1: vector string -> caption."""
    logger.info("\n" + "=" * 80)
    logger.info("[STAGE 1 - TEXT] Vector -> Caption training started.")
    logger.info(f"[STAGE 1] Loading captioning data from: {captioning_path}")

    with open(captioning_path, "r") as f:
        data = json.load(f)

    full_ds = Dataset.from_list(data)
    logger.info(f"[STAGE 1] Total samples: {len(full_ds)}")

    split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    logger.info(f"[STAGE 1] Train samples: {len(train_ds)}  |  Val samples: {len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return _tokenize_captioning(batch, tokenizer)

    logger.info("[STAGE 1] Tokenizing datasets...")
    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

    logger.info(f"[STAGE 1] Loading model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    _ensure_dir(STAGE1_OUTPUT_DIR)
    logger.info(f"[STAGE 1] Output directory: {STAGE1_OUTPUT_DIR}")

    training_args = TrainingArguments(
        output_dir=STAGE1_OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        fp16=False,
        optim="adafactor",
        learning_rate=5e-4,
        max_grad_norm=1.0,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )

    logger.info("[STAGE 1] Starting training...")
    trainer.train()
    trainer.save_model(STAGE1_OUTPUT_DIR)
    tokenizer.save_pretrained(STAGE1_OUTPUT_DIR)

    logger.info("[STAGE 1] Training finished. Running evaluation...")
    eval_metrics = trainer.evaluate()
    logger.info(f"[STAGE 1] Eval metrics: {eval_metrics}")

    metrics_path = os.path.join(STAGE1_OUTPUT_DIR, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"[STAGE 1] Saved eval metrics to {metrics_path}")

    logger.info("[STAGE 1] Generating predictions on validation set...")
    model.eval()
    val_preds = []
    for i, sample in enumerate(eval_ds):
        input_text = sample["input"]
        gt_text = sample["target"]
        try:
            inputs = tokenizer(input_text, return_tensors="pt", max_length=320, truncation=True).to(model.device)
            pred_ids = model.generate(
                **inputs,
                max_new_tokens=260,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=4,
                repetition_penalty=1.2,
            )
            pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        except Exception:
            logger.exception(f"[STAGE 1] Generation failed on val sample idx={i}.")
            pred_text = ""

        val_preds.append({"input": input_text, "ground_truth": gt_text, "prediction": pred_text})

    preds_path = os.path.join(STAGE1_OUTPUT_DIR, "val_predictions.json")
    with open(preds_path, "w") as f:
        json.dump(val_preds, f, indent=2)
    logger.info(f"[STAGE 1] Saved {len(val_preds)} validation predictions to {preds_path}")

    logger.info("[STAGE 1] Done.\n" + "=" * 80)
    return model, tokenizer


# ---------------------------
# Stage 1 training (vector prefix path — custom PyTorch loop)
# ---------------------------

def _train_stage1_prefix(captioning_path: str):
    """
    Vector Prefix Stage 1: vectors -> learned prefix embeddings -> T5 -> caption.
    Uses custom PyTorch training loop with AdamW + linear warmup.
    Supports multi-GPU via HuggingFace Accelerate.
    """
    import datetime
    from accelerate import Accelerator, InitProcessGroupKwargs
    from .vector_encoder import VectorEncoderConfig, parse_vec_str
    from .vector_prefix_t5 import VectorPrefixT5
    from .data_collator import VectorPrefixDataCollator
    from .lora_utils import apply_lora, save_checkpoint

    # accelerator = Accelerator()
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(hours=2))]
    )

    logger.info("\n" + "=" * 80)
    logger.info("[STAGE 1 - VECTOR PREFIX] Vector -> Caption training started.")
    logger.info(f"[STAGE 1] Loading captioning data from: {captioning_path}")
    logger.info(f"[STAGE 1] Accelerate: {accelerator.num_processes} process(es), device={accelerator.device}")

    with open(captioning_path, "r") as f:
        data = json.load(f)

    # Ensure vectors field exists (fallback: parse from input text)
    for sample in data:
        if "vectors" not in sample:
            input_text = sample["input"]
            vec_str = input_text.split("\n", 1)[1] if "\n" in input_text else ""
            vectors, n = parse_vec_str(vec_str, cfg.MAX_OBJECTS, cfg.VECTOR_DIM)
            sample["vectors"] = vectors.tolist()
            sample["num_objects"] = n

    logger.info(f"[STAGE 1] Total samples: {len(data)}")

    # Train/val split (deterministic across all processes)
    torch.manual_seed(cfg.SEED)
    n_val = max(1, int(len(data) * 0.2))
    n_train = len(data) - n_val
    indices = torch.randperm(len(data)).tolist()
    train_samples = [data[i] for i in indices[:n_train]]
    val_samples = [data[i] for i in indices[n_train:]]
    logger.info(f"[STAGE 1] Train: {len(train_samples)} | Val: {len(val_samples)}")

    # Build model
    encoder_config = VectorEncoderConfig(**cfg.VECTOR_ENCODER_CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = VectorPrefixT5(MODEL_NAME, encoder_config, tokenizer)

    if cfg.FREEZE_BASE_MODEL:
        model.freeze_t5_base()

    if cfg.USE_LORA:
        model = apply_lora(
            model, r=cfg.LORA_R, alpha=cfg.LORA_ALPHA,
            dropout=cfg.LORA_DROPOUT, target_modules=cfg.LORA_TARGET_MODULES,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"[STAGE 1] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    logger.info(f"[STAGE 1] use_lora: {cfg.USE_LORA}")
    logger.info(f"[STAGE 1] freeze_base: {cfg.FREEZE_BASE_MODEL}")

    # Override text input for Stage-1 (minimal prompt)
    for s in train_samples:
        s["input"] = cfg.STAGE1_TEXT_PROMPT
    for s in val_samples:
        s["input"] = cfg.STAGE1_TEXT_PROMPT

    # Data loaders
    collator = VectorPrefixDataCollator(
        tokenizer=tokenizer,
        max_input_length=cfg.STAGE1_MAX_INPUT_LEN,
        max_target_length=cfg.STAGE1_MAX_TARGET_LEN,
        max_objects=cfg.MAX_OBJECTS,
        vector_dim=cfg.VECTOR_DIM,
    )

    train_loader = DataLoader(
        VectorPrefixDataset(train_samples),
        batch_size=cfg.STAGE1_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    # val_loader is NOT prepared by Accelerate — runs on main process only
    val_loader = DataLoader(
        VectorPrefixDataset(val_samples),
        batch_size=cfg.STAGE1_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    # Optimizer (created before accelerator.prepare)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.STAGE1_LR, weight_decay=0.0)

    # Accelerate: wrap model, optimizer, train_loader (handles DDP + device placement)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Scheduler (created after prepare — len(train_loader) is now per-process correct)
    total_steps = len(train_loader) * cfg.STAGE1_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    _ensure_dir(STAGE1_OUTPUT_DIR)
    logger.info(f"[STAGE 1] Output directory: {STAGE1_OUTPUT_DIR}")

    # Training loop
    best_val_loss = float("inf")
    training_log = []

    for epoch in range(cfg.STAGE1_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            # Accelerate's prepared loader places tensors on the correct device
            vectors = batch["vectors"]
            num_objects = batch["num_objects"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(
                vectors=vectors,
                num_objects=num_objects,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

            if accelerator.is_main_process and ((step + 1) % cfg.LOGGING_STEPS == 0 or step == 0):
                logger.info(
                    f"[Stage-1] Epoch {epoch+1}/{cfg.STAGE1_EPOCHS}, "
                    f"Step {step+1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation — main process only, using unwrapped model
        val_loss = 0.0
        val_preds_text = []
        val_refs_text = []
        bleu1_val = 0.0
        rouge_l_val = 0.0

        # In multi-GPU mode: always validate (val loss is fast), but skip beam search
        # generation (slow — causes NCCL watchdog timeout). BLEU/ROUGE = None in logs.
        # In single-GPU mode: full validation including generation every epoch.
        run_generation = (accelerator.num_processes == 1)

        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            val_loss, val_preds_text, val_refs_text = _validate_stage1_prefix(
                unwrapped, val_loader, tokenizer, accelerator.device,
                generate=run_generation,
            )

            # Compute caption metrics (only meaningful when generation ran)
            bleu1_val = _compute_bleu1_list(val_preds_text, val_refs_text) if run_generation else None
            rouge_l_val = _compute_rouge_l_list(val_preds_text, val_refs_text) if run_generation else None

            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "bleu1": bleu1_val,
                "rouge_l": rouge_l_val,
            }
            training_log.append(epoch_log)

            if run_generation:
                logger.info(
                    f"[Stage-1] Epoch {epoch+1}/{cfg.STAGE1_EPOCHS} -- "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"BLEU-1: {bleu1_val:.4f}, "
                    f"ROUGE-L: {rouge_l_val:.4f}"
                )
            else:
                logger.info(
                    f"[Stage-1] Epoch {epoch+1}/{cfg.STAGE1_EPOCHS} -- "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}  "
                    f"(BLEU/ROUGE skipped in multi-GPU mode)"
                )

            # Log sample predictions
            if val_preds_text:
                n_show = min(3, len(val_preds_text))
                logger.info(f"[Stage-1] Sample predictions (epoch {epoch+1}):")
                for i in range(n_show):
                    logger.info(f"  [pred] {val_preds_text[i][:200]}")
                    logger.info(f"  [ref]  {val_refs_text[i][:200]}")
                    logger.info("")

            # Save best checkpoint by val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_dir = os.path.join(STAGE1_OUTPUT_DIR, "best_checkpoint")
                save_checkpoint(unwrapped, ckpt_dir, epoch=epoch + 1)
                logger.info(f"[Stage-1] Saved best checkpoint (val_loss={val_loss:.4f})")

        # Sync all processes before next epoch
        accelerator.wait_for_everyone()

    # Save final checkpoint + logs (main process only)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)

        final_dir = os.path.join(STAGE1_OUTPUT_DIR, "final_checkpoint")
        save_checkpoint(unwrapped, final_dir, epoch=cfg.STAGE1_EPOCHS)

        # Save training log
        log_path = os.path.join(STAGE1_OUTPUT_DIR, "stage1_training_log.json")
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        logger.info(f"[Stage-1] Training log saved to {log_path}")

        # Save eval metrics
        eval_metrics = {
            "eval_loss": best_val_loss,
            "bleu1": bleu1_val,
            "rouge_l": rouge_l_val,
        }
        metrics_path = os.path.join(STAGE1_OUTPUT_DIR, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)

        # Save sample predictions
        pred_path = os.path.join(STAGE1_OUTPUT_DIR, "val_predictions.json")
        preds_data = [
            {"prediction": p, "ground_truth": r}
            for p, r in zip(val_preds_text, val_refs_text)
        ]
        with open(pred_path, "w") as f:
            json.dump(preds_data, f, indent=2)
        logger.info(f"[STAGE 1] Saved {len(preds_data)} validation predictions to {pred_path}")

    accelerator.wait_for_everyone()
    logger.info("[STAGE 1] Done.\n" + "=" * 80)
    return accelerator.unwrap_model(model), tokenizer


def _validate_stage1_prefix(model, val_loader, tokenizer, device, generate=True):
    """Run validation: compute loss, and optionally generate captions for BLEU/ROUGE.

    Pass generate=False in multi-GPU mode — the forward-pass-only path takes
    ~30s and will not trigger the NCCL watchdog timeout.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for batch in val_loader:
            vectors = batch["vectors"].to(device)
            num_objects = batch["num_objects"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Loss (always computed — fast forward pass)
            outputs = model(
                vectors=vectors,
                num_objects=num_objects,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            n_batches += 1

            if generate:
                # Beam search generation — slow, skipped in multi-GPU mode
                gen_ids = model.generate(
                    vectors=vectors,
                    num_objects=num_objects,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=cfg.GEN_MAX_NEW_TOKENS_STAGE1,
                    num_beams=cfg.GEN_NUM_BEAMS,
                )
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                all_preds.extend(preds)

                ref_ids = labels.clone()
                ref_ids[ref_ids == -100] = tokenizer.pad_token_id
                refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
                all_refs.extend(refs)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, all_preds, all_refs


def _compute_bleu1_list(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    scores = [bleu1(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


def _compute_rouge_l_list(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    scores = [rouge_l_f1(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


# ---------------------------
# Stage 1 entry point (dispatch)
# ---------------------------

def train_stage1(captioning_path: str):
    if cfg.USE_VECTOR_PREFIX:
        return _train_stage1_prefix(captioning_path)
    else:
        return _train_stage1_text(captioning_path)


# ---------------------------
# Stage 2 training (text path - original)
# ---------------------------

def _train_stage2_text(model_stage1, tokenizer, qa_path: str):
    """Original text-only Stage 2: caption + risk + question -> answer."""
    logger.info("\n" + "=" * 80)
    logger.info("[STAGE 2 - TEXT] Driving QA finetuning started.")
    logger.info(f"[STAGE 2] Loading QA data from: {qa_path}")
    logger.info(f"[STAGE 2] Initializing Stage-2 model from Stage-1 checkpoint dir: {STAGE1_OUTPUT_DIR}")

    model_stage2 = copy.deepcopy(model_stage1)
    model_stage2.train()

    with open(qa_path, "r") as f:
        data = json.load(f)

    full_ds = Dataset.from_list(data)
    logger.info(f"[STAGE 2] Total samples: {len(full_ds)}")

    split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    logger.info(f"[STAGE 2] Train samples: {len(train_ds)}  |  Val samples: {len(eval_ds)}")

    def tokenize_fn(batch):
        return _tokenize_qa(batch, tokenizer)

    logger.info("[STAGE 2] Tokenizing datasets...")
    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

    _ensure_dir(STAGE2_OUTPUT_DIR)
    logger.info(f"[STAGE 2] Output directory: {STAGE2_OUTPUT_DIR}")

    training_args = TrainingArguments(
        output_dir=STAGE2_OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        fp16=False,
        optim="adafactor",
        learning_rate=5e-4,
        max_grad_norm=1.0,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model_stage2,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )

    logger.info("[STAGE 2] Starting training...")
    trainer.train()
    logger.info("[STAGE 2] Training finished. Running evaluation (loss only)...")

    raw_eval = trainer.evaluate()
    logger.info(f"[STAGE 2] Raw eval output: {raw_eval}")

    model_stage1.eval()
    model_stage2.eval()

    def _gen_text(model, prompt: str, max_new_tokens: int, ensure_paper: bool) -> str:
        if ensure_paper:
            prompt = _ensure_paper_format(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=384, truncation=True).to(model.device)
        pred_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
        return tokenizer.decode(pred_ids[0], skip_special_tokens=True)

    def run_eval(mode: str) -> Tuple[Dict, List[Dict]]:
        correct = 0
        total = 0
        missed_brake = 0
        brake_total = 0
        unsafe_continue_high = 0
        highcrit_total = 0
        brake_mae_sum_on_brake_gt = 0.0
        brake_mae_count_on_brake_gt = 0
        risk_correct = 0
        risk_total = 0
        bleu_sum = 0.0
        rouge_sum = 0.0
        fmt_sum = 0.0
        parse_ok_sum = 0.0
        outputs: List[Dict] = []

        for si, sample in enumerate(eval_ds):
            raw_input_text = sample["input"]
            gt_text = sample["target"]
            qtype = (sample.get("question_type") or "action").strip().lower()
            question = sample.get("question", "How should the car drive in this situation and why?")
            risk_text = sample.get("risk_text", "")
            risk_level = sample.get("risk_level", None)

            try:
                if mode == "oracle_caption":
                    stage2_prompt = raw_input_text
                    caption_used = None
                else:
                    vec_str = sample.get("vec_str", "")
                    s1_prompt = _build_stage1_prompt(vec_str)
                    caption_pred = _gen_text(model_stage1, s1_prompt, max_new_tokens=260, ensure_paper=False)
                    caption_used = caption_pred
                    stage2_prompt = _build_stage2_prompt_from_caption(
                        caption_pred, risk_text=risk_text, qa_question=question, question_type=qtype
                    )

                pred_raw = _gen_text(
                    model_stage2, stage2_prompt,
                    max_new_tokens=90, ensure_paper=(qtype == "action"),
                )

                if qtype == "action":
                    pred_fixed, parse_ok = enforce_5_lines(pred_raw)
                    gt_action = _map_text_to_action_label(gt_text)
                    pred_action = _map_text_to_action_label(pred_fixed)

                    rl = (sample.get("risk_level") or "").strip().upper()
                    if rl in ("HIGH", "CRITICAL"):
                        highcrit_total += 1
                        if pred_action == "CONTINUE":
                            unsafe_continue_high += 1

                    if gt_action == "BRAKE":
                        gt_brk = _extract_brake_percent(gt_text)
                        pr_brk = _extract_brake_percent(pred_fixed)
                        if gt_brk is not None and pr_brk is not None:
                            brake_mae_sum_on_brake_gt += abs(float(pr_brk) - float(gt_brk))
                            brake_mae_count_on_brake_gt += 1

                    if gt_action != "OTHER":
                        total += 1
                        if gt_action == pred_action:
                            correct += 1
                        if gt_action == "BRAKE":
                            brake_total += 1
                            if pred_action != "BRAKE":
                                missed_brake += 1

                    bleu_sum += bleu1(pred_fixed, gt_text)
                    rouge_sum += rouge_l_f1(pred_fixed, gt_text)
                    fmt_sum += float(_format_compliance_5line(pred_fixed))
                    parse_ok_sum += float(parse_ok)
                else:
                    gt_rl = (risk_level or _extract_risk_level(gt_text) or "")
                    pr_rl = (_extract_risk_level(pred_raw) or "")
                    if gt_rl:
                        risk_total += 1
                        if pr_rl == gt_rl.upper():
                            risk_correct += 1
                    pred_fixed = pred_raw
                    gt_action = "OTHER"
                    pred_action = "OTHER"
                    parse_ok = 0

            except Exception:
                logger.exception(f"[STAGE 2][EVAL] Failed on eval sample idx={si} (mode={mode}).")
                pred_raw = ""
                pred_fixed = ""
                caption_used = None
                gt_action = "OTHER"
                pred_action = "OTHER"
                parse_ok = 0

            outputs.append({
                "mode": mode,
                "question_type": qtype,
                "question": question,
                "input": stage2_prompt if mode != "oracle_caption" else raw_input_text,
                "ground_truth": gt_text,
                "prediction_raw": pred_raw,
                "prediction_fixed": pred_fixed,
                "gt_action": gt_action,
                "pred_action": pred_action,
                "parse_ok": int(parse_ok),
                "caption_used": caption_used,
                "risk_level": risk_level,
            })

        metrics = {
            "action_accuracy": float(correct / total) if total > 0 else 0.0,
            "n_action_samples": int(total),
            "missed_brake_rate": float(missed_brake / brake_total) if brake_total > 0 else 0.0,
            "n_brake_gt": int(brake_total),
            "unsafe_continue_high_rate": float(unsafe_continue_high / highcrit_total) if highcrit_total > 0 else 0.0,
            "n_highcrit_action_samples": int(highcrit_total),
            "brake_mae_on_brake_gt": float(brake_mae_sum_on_brake_gt / brake_mae_count_on_brake_gt) if brake_mae_count_on_brake_gt > 0 else 0.0,
            "n_brake_mae_samples": int(brake_mae_count_on_brake_gt),
            "risk_level_accuracy": float(risk_correct / risk_total) if risk_total > 0 else 0.0,
            "n_risk_samples": int(risk_total),
            "bleu1_action": float(bleu_sum / max(1, total)) if total > 0 else 0.0,
            "rougeL_f1_action": float(rouge_sum / max(1, total)) if total > 0 else 0.0,
            "format_compliance_action": float(fmt_sum / max(1, total)) if total > 0 else 0.0,
            "parse_ok_rate_action": float(parse_ok_sum / max(1, total)) if total > 0 else 0.0,
        }
        return metrics, outputs

    logger.info("[STAGE 2] Computing metrics: oracle_caption...")
    oracle_metrics, oracle_outputs = run_eval("oracle_caption")
    logger.info(f"[STAGE 2] oracle_caption metrics: {oracle_metrics}")
    _print_eval_risk_summary(oracle_outputs, "oracle_caption")

    logger.info("[STAGE 2] Computing metrics: stage1_caption...")
    stage1_metrics, stage1_outputs = run_eval("stage1_caption")
    logger.info(f"[STAGE 2] stage1_caption metrics: {stage1_metrics}")
    _print_eval_risk_summary(stage1_outputs, "stage1_caption")

    eval_metrics: Dict = {}
    if isinstance(raw_eval, dict):
        for k, v in raw_eval.items():
            try:
                eval_metrics[k] = float(v)
            except Exception:
                eval_metrics[k] = v

    eval_metrics["oracle_caption"] = oracle_metrics
    eval_metrics["stage1_caption"] = stage1_metrics

    metrics_path = os.path.join(STAGE2_OUTPUT_DIR, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"[STAGE 2] Saved eval metrics to {metrics_path}")

    preds_path1 = os.path.join(STAGE2_OUTPUT_DIR, "val_predictions_oracle_caption.json")
    with open(preds_path1, "w") as f:
        json.dump(oracle_outputs, f, indent=2)

    preds_path2 = os.path.join(STAGE2_OUTPUT_DIR, "val_predictions_stage1_caption.json")
    with open(preds_path2, "w") as f:
        json.dump(stage1_outputs, f, indent=2)

    logger.info("[STAGE 2] Done.\n" + "=" * 80)
    return model_stage2


# ---------------------------
# Stage 2 training (vector prefix path)
# ---------------------------

def _train_stage2_with_lora(model_stage1, tokenizer, qa_path: str):
    """
    Stage 2: text-only QA training (caption + risk + question -> answer).
    model_stage1 is VectorPrefixT5; extract T5 for Stage 2, keep full model for caption gen.
    """
    from .vector_prefix_t5 import VectorPrefixT5
    from .vector_encoder import parse_vec_str

    logger.info("\n" + "=" * 80)
    logger.info("[STAGE 2 - PREFIX] Driving QA finetuning started.")
    logger.info(f"[STAGE 2] Loading QA data from: {qa_path}")

    # Keep full VectorPrefixT5 for caption generation in eval
    caption_model = model_stage1
    caption_model.eval()

    # Extract T5 from VectorPrefixT5, deep-copy for Stage 2
    if isinstance(model_stage1, VectorPrefixT5):
        t5_base = copy.deepcopy(model_stage1.t5)
    else:
        t5_base = copy.deepcopy(model_stage1)

    # Unfreeze T5 for Stage 2
    for param in t5_base.parameters():
        param.requires_grad = False

    # Apply LoRA if enabled
    if cfg.USE_LORA:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.LORA_TARGET_MODULES,
        )
        model_stage2 = get_peft_model(t5_base, lora_config)
        model_stage2.print_trainable_parameters()
        logger.info("[STAGE 2] LoRA applied to T5 model.")
    else:
        # No LoRA: unfreeze all and finetune fully
        model_stage2 = t5_base
        for param in model_stage2.parameters():
            param.requires_grad = True
        logger.info("[STAGE 2] No LoRA; full finetuning enabled.")

    model_stage2.train()

    with open(qa_path, "r") as f:
        data = json.load(f)

    full_ds = Dataset.from_list(data)
    logger.info(f"[STAGE 2] Total samples: {len(full_ds)}")

    split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    logger.info(f"[STAGE 2] Train: {len(train_ds)} | Val: {len(eval_ds)}")

    def tokenize_fn(batch):
        return _tokenize_qa(batch, tokenizer)

    logger.info("[STAGE 2] Tokenizing datasets...")
    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)

    _ensure_dir(STAGE2_OUTPUT_DIR)
    logger.info(f"[STAGE 2] Output directory: {STAGE2_OUTPUT_DIR}")

    training_args = TrainingArguments(
        output_dir=STAGE2_OUTPUT_DIR,
        per_device_train_batch_size=cfg.STAGE2_BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=cfg.STAGE2_EPOCHS,
        fp16=False,
        optim="adafactor",
        learning_rate=cfg.STAGE2_LR,
        max_grad_norm=1.0,
        logging_steps=cfg.LOGGING_STEPS,
        eval_strategy=cfg.EVAL_STRATEGY,
        save_strategy=cfg.SAVE_STRATEGY,
        save_total_limit=cfg.SAVE_TOTAL_LIMIT,
        disable_tqdm=cfg.DISABLE_TQDM,
    )

    trainer = Trainer(
        model=model_stage2,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )

    logger.info("[STAGE 2] Starting training...")
    trainer.train()
    logger.info("[STAGE 2] Training finished.")

    # Save model (LoRA adapter + base)
    trainer.save_model(STAGE2_OUTPUT_DIR)
    tokenizer.save_pretrained(STAGE2_OUTPUT_DIR)

    raw_eval = trainer.evaluate()
    logger.info(f"[STAGE 2] Raw eval output: {raw_eval}")

    # --- Evaluation ---
    model_stage2.eval()
    device = next(model_stage2.parameters()).device

    def _gen_text_s2(prompt: str, max_new_tokens: int, ensure_paper: bool) -> str:
        if ensure_paper:
            prompt = _ensure_paper_format(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=384, truncation=True).to(device)
        pred_ids = model_stage2.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=cfg.GEN_NUM_BEAMS,
            early_stopping=cfg.GEN_EARLY_STOPPING,
            no_repeat_ngram_size=cfg.GEN_NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=cfg.GEN_REPETITION_PENALTY,
        )
        return tokenizer.decode(pred_ids[0], skip_special_tokens=True)

    def _gen_caption_from_vectors(sample: Dict) -> str:
        """Generate caption using VectorPrefixT5 model from vector tensors."""
        # Try raw vectors first, fallback to parsing vec_str
        if "vectors" in sample and sample["vectors"]:
            vectors = sample["vectors"]
        else:
            vec_str = sample.get("vec_str", "")
            vectors, n = parse_vec_str(vec_str, cfg.MAX_OBJECTS, cfg.VECTOR_DIM)
            vectors = vectors.tolist()
            sample["num_objects"] = n

        num_obj = sample.get("num_objects", sample.get("use_n", 0))

        cap_device = caption_model.device
        vectors_t = torch.tensor([vectors], dtype=torch.float32).to(cap_device)
        num_obj_t = torch.tensor([num_obj], dtype=torch.long).to(cap_device)

        text_inputs = tokenizer(
            cfg.STAGE1_TEXT_PROMPT,
            return_tensors="pt",
            max_length=cfg.STAGE1_MAX_INPUT_LEN,
            truncation=True,
        ).to(cap_device)

        pred_ids = caption_model.generate(
            vectors=vectors_t,
            num_objects=num_obj_t,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            max_new_tokens=cfg.GEN_MAX_NEW_TOKENS_STAGE1,
            num_beams=cfg.GEN_NUM_BEAMS,
            early_stopping=cfg.GEN_EARLY_STOPPING,
            no_repeat_ngram_size=cfg.GEN_NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=cfg.GEN_REPETITION_PENALTY,
        )
        return tokenizer.decode(pred_ids[0], skip_special_tokens=True)

    def run_eval(mode: str) -> Tuple[Dict, List[Dict]]:
        correct = 0
        total = 0
        missed_brake = 0
        brake_total = 0
        unsafe_continue_high = 0
        highcrit_total = 0
        brake_mae_sum_on_brake_gt = 0.0
        brake_mae_count_on_brake_gt = 0
        risk_correct = 0
        risk_total = 0
        bleu_sum = 0.0
        rouge_sum = 0.0
        fmt_sum = 0.0
        parse_ok_sum = 0.0
        outputs: List[Dict] = []

        for si, sample in enumerate(eval_ds):
            raw_input_text = sample["input"]
            gt_text = sample["target"]
            qtype = (sample.get("question_type") or "action").strip().lower()
            question = sample.get("question", "How should the car drive in this situation and why?")
            risk_text = sample.get("risk_text", "")
            risk_level = sample.get("risk_level", None)

            try:
                if mode == "oracle_caption":
                    stage2_prompt = raw_input_text
                    caption_used = None
                else:
                    # Generate caption using VectorPrefixT5
                    caption_pred = _gen_caption_from_vectors(sample)
                    caption_used = caption_pred
                    stage2_prompt = _build_stage2_prompt_from_caption(
                        caption_pred, risk_text=risk_text, qa_question=question, question_type=qtype
                    )

                pred_raw = _gen_text_s2(
                    stage2_prompt,
                    max_new_tokens=cfg.GEN_MAX_NEW_TOKENS_STAGE2,
                    ensure_paper=(qtype == "action"),
                )

                if qtype == "action":
                    pred_fixed, parse_ok = enforce_5_lines(pred_raw)
                    gt_action = _map_text_to_action_label(gt_text)
                    pred_action = _map_text_to_action_label(pred_fixed)

                    rl = (sample.get("risk_level") or "").strip().upper()
                    if rl in ("HIGH", "CRITICAL"):
                        highcrit_total += 1
                        if pred_action == "CONTINUE":
                            unsafe_continue_high += 1

                    if gt_action == "BRAKE":
                        gt_brk = _extract_brake_percent(gt_text)
                        pr_brk = _extract_brake_percent(pred_fixed)
                        if gt_brk is not None and pr_brk is not None:
                            brake_mae_sum_on_brake_gt += abs(float(pr_brk) - float(gt_brk))
                            brake_mae_count_on_brake_gt += 1

                    if gt_action != "OTHER":
                        total += 1
                        if gt_action == pred_action:
                            correct += 1
                        if gt_action == "BRAKE":
                            brake_total += 1
                            if pred_action != "BRAKE":
                                missed_brake += 1

                    bleu_sum += bleu1(pred_fixed, gt_text)
                    rouge_sum += rouge_l_f1(pred_fixed, gt_text)
                    fmt_sum += float(_format_compliance_5line(pred_fixed))
                    parse_ok_sum += float(parse_ok)
                else:
                    gt_rl = (risk_level or _extract_risk_level(gt_text) or "")
                    pr_rl = (_extract_risk_level(pred_raw) or "")
                    if gt_rl:
                        risk_total += 1
                        if pr_rl == gt_rl.upper():
                            risk_correct += 1
                    pred_fixed = pred_raw
                    gt_action = "OTHER"
                    pred_action = "OTHER"
                    parse_ok = 0

            except Exception:
                logger.exception(f"[STAGE 2][EVAL] Failed on eval sample idx={si} (mode={mode}).")
                pred_raw = ""
                pred_fixed = ""
                caption_used = None
                gt_action = "OTHER"
                pred_action = "OTHER"
                parse_ok = 0

            outputs.append({
                "mode": mode,
                "question_type": qtype,
                "question": question,
                "input": stage2_prompt if mode != "oracle_caption" else raw_input_text,
                "ground_truth": gt_text,
                "prediction_raw": pred_raw,
                "prediction_fixed": pred_fixed,
                "gt_action": gt_action,
                "pred_action": pred_action,
                "parse_ok": int(parse_ok),
                "caption_used": caption_used,
                "risk_level": risk_level,
            })

        metrics = {
            "action_accuracy": float(correct / total) if total > 0 else 0.0,
            "n_action_samples": int(total),
            "missed_brake_rate": float(missed_brake / brake_total) if brake_total > 0 else 0.0,
            "n_brake_gt": int(brake_total),
            "unsafe_continue_high_rate": float(unsafe_continue_high / highcrit_total) if highcrit_total > 0 else 0.0,
            "n_highcrit_action_samples": int(highcrit_total),
            "brake_mae_on_brake_gt": float(brake_mae_sum_on_brake_gt / brake_mae_count_on_brake_gt) if brake_mae_count_on_brake_gt > 0 else 0.0,
            "n_brake_mae_samples": int(brake_mae_count_on_brake_gt),
            "risk_level_accuracy": float(risk_correct / risk_total) if risk_total > 0 else 0.0,
            "n_risk_samples": int(risk_total),
            "bleu1_action": float(bleu_sum / max(1, total)) if total > 0 else 0.0,
            "rougeL_f1_action": float(rouge_sum / max(1, total)) if total > 0 else 0.0,
            "format_compliance_action": float(fmt_sum / max(1, total)) if total > 0 else 0.0,
            "parse_ok_rate_action": float(parse_ok_sum / max(1, total)) if total > 0 else 0.0,
        }
        return metrics, outputs

    # Run evaluation only on main process (avoids duplicate work in multi-GPU)
    if not hasattr(trainer, 'is_world_process_zero') or trainer.is_world_process_zero():
        logger.info("[STAGE 2] Computing metrics: oracle_caption...")
        oracle_metrics, oracle_outputs = run_eval("oracle_caption")
        logger.info(f"[STAGE 2] oracle_caption metrics: {oracle_metrics}")
        _print_eval_risk_summary(oracle_outputs, "oracle_caption")

        logger.info("[STAGE 2] Computing metrics: stage1_caption...")
        stage1_metrics, stage1_outputs = run_eval("stage1_caption")
        logger.info(f"[STAGE 2] stage1_caption metrics: {stage1_metrics}")
        _print_eval_risk_summary(stage1_outputs, "stage1_caption")

        eval_metrics: Dict = {}
        if isinstance(raw_eval, dict):
            for k, v in raw_eval.items():
                try:
                    eval_metrics[k] = float(v)
                except Exception:
                    eval_metrics[k] = v

        eval_metrics["oracle_caption"] = oracle_metrics
        eval_metrics["stage1_caption"] = stage1_metrics

        metrics_path = os.path.join(STAGE2_OUTPUT_DIR, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        logger.info(f"[STAGE 2] Saved eval metrics to {metrics_path}")

        preds_path1 = os.path.join(STAGE2_OUTPUT_DIR, "val_predictions_oracle_caption.json")
        with open(preds_path1, "w") as f:
            json.dump(oracle_outputs, f, indent=2)

        preds_path2 = os.path.join(STAGE2_OUTPUT_DIR, "val_predictions_stage1_caption.json")
        with open(preds_path2, "w") as f:
            json.dump(stage1_outputs, f, indent=2)

    logger.info("[STAGE 2] Done.\n" + "=" * 80)
    return model_stage2


# ---------------------------
# Stage 2 entry point (dispatch)
# ---------------------------

def train_stage2(model_stage1, tokenizer, qa_path: str):
    if cfg.USE_VECTOR_PREFIX:
        return _train_stage2_with_lora(model_stage1, tokenizer, qa_path)
    else:
        return _train_stage2_text(model_stage1, tokenizer, qa_path)
