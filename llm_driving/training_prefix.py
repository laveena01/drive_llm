# llm_driving/training_prefix.py
"""
Vector Prefix Training Pipeline.

Stage-1: Vector tensors → VectorPrefixEncoder → prefix + text prompt → T5 generates caption
Stage-2: Inherit Stage-1, fine-tune for driving QA with risk injection

Keeps the existing text pipeline (training.py) untouched.
Controlled by USE_VECTOR_PREFIX flag in config.py.
"""

from __future__ import annotations
import os
import json
import logging
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import (
    MODEL_NAME, T5_D_MODEL, SEED,
    MAX_OBJECTS, VECTOR_DIM,
    VECTOR_ENCODER_CONFIG,
    USE_LORA, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    FREEZE_BASE_MODEL,
    STAGE1_EPOCHS, STAGE1_BATCH_SIZE, STAGE1_LR,
    STAGE1_MAX_INPUT_LEN, STAGE1_MAX_TARGET_LEN,
    STAGE1_TEXT_PROMPT,
    STAGE2_EPOCHS, STAGE2_BATCH_SIZE, STAGE2_LR,
    STAGE2_MAX_INPUT_LEN, STAGE2_MAX_TARGET_LEN,
    STAGE1_OUTPUT_DIR, STAGE2_OUTPUT_DIR,
    GEN_NUM_BEAMS, GEN_MAX_NEW_TOKENS_STAGE1, GEN_MAX_NEW_TOKENS_STAGE2,
    LOGGING_STEPS,
)
from .vector_encoder import VectorEncoderConfig
from .vector_prefix_t5 import VectorPrefixT5
from .lora_utils import apply_lora, save_checkpoint, load_checkpoint
from .data_collator import VectorPrefixDataCollator

logger = logging.getLogger("llm_driving")


# ──────────────────────────────────────────────────────────
# Dataset wrapper
# ──────────────────────────────────────────────────────────

class VectorPrefixDataset(TorchDataset):
    """Wraps a list of sample dicts (from datasets_builder JSON) for DataLoader."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ──────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────

def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


def compute_bleu1(predictions: List[str], references: List[str]) -> float:
    """Simple unigram BLEU (BLEU-1)."""
    if not predictions:
        return 0.0
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = _normalize_text(pred).split()
        ref_tokens = set(_normalize_text(ref).split())
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        matches = sum(1 for t in pred_tokens if t in ref_tokens)
        scores.append(matches / len(pred_tokens))
    return sum(scores) / len(scores)


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """Simple ROUGE-L (longest common subsequence F1)."""
    if not predictions:
        return 0.0

    def _lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = _normalize_text(pred).split()
        ref_tokens = _normalize_text(ref).split()
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        lcs = _lcs_length(pred_tokens, ref_tokens)
        prec = lcs / len(pred_tokens) if pred_tokens else 0
        rec = lcs / len(ref_tokens) if ref_tokens else 0
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append(2 * prec * rec / (prec + rec))
    return sum(scores) / len(scores)


# ──────────────────────────────────────────────────────────
# Stage-1: Vector → Caption
# ──────────────────────────────────────────────────────────

def train_stage1_prefix(
    captioning_data_path: str,
    output_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    device: Optional[str] = None,
    val_split: float = 0.1,
) -> str:
    """
    Train Stage-1: vector prefix → caption generation.
    
    Args:
        captioning_data_path: Path to JSON with captioning samples
        output_dir: Where to save checkpoints
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        device: torch device string
        val_split: Fraction of data for validation
        
    Returns:
        Path to saved checkpoint directory
    """
    output_dir = output_dir or STAGE1_OUTPUT_DIR
    epochs = epochs or STAGE1_EPOCHS
    batch_size = batch_size or STAGE1_BATCH_SIZE
    lr = lr or STAGE1_LR
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("[Stage-1 Prefix] Starting vector prefix caption training")
    logger.info(f"  output_dir:  {output_dir}")
    logger.info(f"  epochs:      {epochs}")
    logger.info(f"  batch_size:  {batch_size}")
    logger.info(f"  lr:          {lr}")
    logger.info(f"  device:      {device}")
    logger.info(f"  use_lora:    {USE_LORA}")
    logger.info("=" * 60)

    # --- Load data ---
    logger.info(f"[Stage-1] Loading captioning data from {captioning_data_path}")
    with open(captioning_data_path, "r") as f:
        all_samples = json.load(f)
    logger.info(f"[Stage-1] Loaded {len(all_samples)} captioning samples")

    # --- Train/val split ---
    torch.manual_seed(SEED)
    n_val = max(1, int(len(all_samples) * val_split))
    n_train = len(all_samples) - n_val
    indices = torch.randperm(len(all_samples)).tolist()
    train_samples = [all_samples[i] for i in indices[:n_train]]
    val_samples = [all_samples[i] for i in indices[n_train:]]
    logger.info(f"[Stage-1] Train: {len(train_samples)}, Val: {len(val_samples)}")

    # --- Build model ---
    encoder_config = VectorEncoderConfig(**VECTOR_ENCODER_CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = VectorPrefixT5(MODEL_NAME, encoder_config, tokenizer)

    if FREEZE_BASE_MODEL:
        model.freeze_t5_base()

    if USE_LORA:
        model = apply_lora(
            model, r=LORA_R, alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES,
        )

    model.to(device)

    # --- Override text input for Stage-1 ---
    # In prefix mode, the text input is just a simple prompt
    # The actual scene information comes from the vector prefix
    for s in train_samples:
        s["input"] = STAGE1_TEXT_PROMPT
    for s in val_samples:
        s["input"] = STAGE1_TEXT_PROMPT

    # --- Data loaders ---
    collator = VectorPrefixDataCollator(
        tokenizer=tokenizer,
        max_input_length=STAGE1_MAX_INPUT_LEN,
        max_target_length=STAGE1_MAX_TARGET_LEN,
        max_objects=MAX_OBJECTS,
        vector_dim=VECTOR_DIM,
    )

    train_loader = DataLoader(
        VectorPrefixDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        VectorPrefixDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # --- Optimizer + scheduler ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    # --- Training loop ---
    best_val_loss = float("inf")
    training_log = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            # Move to device
            vectors = batch["vectors"].to(device)
            num_objects = batch["num_objects"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            outputs = model(
                vectors=vectors,
                num_objects=num_objects,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

            if (step + 1) % LOGGING_STEPS == 0 or step == 0:
                logger.info(
                    f"[Stage-1] Epoch {epoch+1}/{epochs}, "
                    f"Step {step+1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # --- Validation ---
        val_loss, val_preds, val_refs = _validate_stage1(
            model, val_loader, tokenizer, device
        )

        # Compute metrics
        bleu1 = compute_bleu1(val_preds, val_refs)
        rouge_l = compute_rouge_l(val_preds, val_refs)

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "bleu1": bleu1,
            "rouge_l": rouge_l,
        }
        training_log.append(epoch_log)

        logger.info(
            f"[Stage-1] Epoch {epoch+1}/{epochs} — "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"BLEU-1: {bleu1:.4f}, "
            f"ROUGE-L: {rouge_l:.4f}"
        )

        # Log sample predictions
        if val_preds:
            n_show = min(3, len(val_preds))
            logger.info(f"[Stage-1] Sample predictions (epoch {epoch+1}):")
            for i in range(n_show):
                logger.info(f"  [pred] {val_preds[i][:200]}")
                logger.info(f"  [ref]  {val_refs[i][:200]}")
                logger.info("")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir = os.path.join(output_dir, "best_checkpoint")
            save_checkpoint(model, ckpt_dir, epoch=epoch + 1)
            logger.info(f"[Stage-1] Saved best checkpoint (val_loss={val_loss:.4f})")

    # Save final checkpoint
    final_dir = os.path.join(output_dir, "final_checkpoint")
    save_checkpoint(model, final_dir, epoch=epochs)

    # Save training log
    log_path = os.path.join(output_dir, "stage1_training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"[Stage-1] Training log saved to {log_path}")

    # Save sample predictions
    pred_path = os.path.join(output_dir, "stage1_predictions.json")
    preds_data = [
        {"prediction": p, "reference": r}
        for p, r in zip(val_preds, val_refs)
    ]
    with open(pred_path, "w") as f:
        json.dump(preds_data, f, indent=2)

    logger.info("[Stage-1] Training complete!")
    return final_dir


def _validate_stage1(model, val_loader, tokenizer, device):
    """Run validation: compute loss + generate captions."""
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

            # Loss
            outputs = model(
                vectors=vectors,
                num_objects=num_objects,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            n_batches += 1

            # Generate
            gen_ids = model.generate(
                vectors=vectors,
                num_objects=num_objects,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=GEN_MAX_NEW_TOKENS_STAGE1,
                num_beams=GEN_NUM_BEAMS,
            )
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_preds.extend(preds)

            # Decode references (replace -100 with pad_token_id)
            ref_ids = labels.clone()
            ref_ids[ref_ids == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)
            all_refs.extend(refs)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, all_preds, all_refs


# ──────────────────────────────────────────────────────────
# Stage-2: Driving QA (caption + risk + question → action)
# ──────────────────────────────────────────────────────────

def train_stage2_prefix(
    qa_data_path: str,
    stage1_checkpoint_dir: str,
    output_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    device: Optional[str] = None,
    val_split: float = 0.1,
) -> str:
    """
    Train Stage-2: driving QA from Stage-1 checkpoint.
    
    Loads encoder + LoRA weights from Stage-1, continues training on QA data.
    Risk text is injected via text prompts (same as current pipeline).
    
    Args:
        qa_data_path: Path to JSON with QA samples
        stage1_checkpoint_dir: Stage-1 checkpoint to initialize from
        output_dir: Where to save Stage-2 checkpoints
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        device: torch device string
        val_split: Fraction of data for validation
        
    Returns:
        Path to saved checkpoint directory
    """
    output_dir = output_dir or STAGE2_OUTPUT_DIR
    epochs = epochs or STAGE2_EPOCHS
    batch_size = batch_size or STAGE2_BATCH_SIZE
    lr = lr or STAGE2_LR
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("[Stage-2 Prefix] Starting driving QA training")
    logger.info(f"  stage1_ckpt: {stage1_checkpoint_dir}")
    logger.info(f"  output_dir:  {output_dir}")
    logger.info(f"  epochs:      {epochs}")
    logger.info(f"  batch_size:  {batch_size}")
    logger.info(f"  lr:          {lr}")
    logger.info(f"  device:      {device}")
    logger.info("=" * 60)

    # --- Load data ---
    logger.info(f"[Stage-2] Loading QA data from {qa_data_path}")
    with open(qa_data_path, "r") as f:
        all_samples = json.load(f)
    logger.info(f"[Stage-2] Loaded {len(all_samples)} QA samples")

    # --- Train/val split ---
    torch.manual_seed(SEED)
    n_val = max(1, int(len(all_samples) * val_split))
    n_train = len(all_samples) - n_val
    indices = torch.randperm(len(all_samples)).tolist()
    train_samples = [all_samples[i] for i in indices[:n_train]]
    val_samples = [all_samples[i] for i in indices[n_train:]]
    logger.info(f"[Stage-2] Train: {len(train_samples)}, Val: {len(val_samples)}")

    # --- Load model from Stage-1 checkpoint ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_checkpoint(
        model_name=MODEL_NAME,
        checkpoint_dir=stage1_checkpoint_dir,
        device=device,
        apply_lora_config=USE_LORA,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=LORA_TARGET_MODULES,
    )
    model.to(device)

    # Ensure encoder + LoRA are trainable
    for param in model.vector_encoder.parameters():
        param.requires_grad = True

    model.print_trainable_parameters()

    # --- Data loaders ---
    collator = VectorPrefixDataCollator(
        tokenizer=tokenizer,
        max_input_length=STAGE2_MAX_INPUT_LEN,
        max_target_length=STAGE2_MAX_TARGET_LEN,
        max_objects=MAX_OBJECTS,
        vector_dim=VECTOR_DIM,
    )

    train_loader = DataLoader(
        VectorPrefixDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        VectorPrefixDataset(val_samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # --- Optimizer + scheduler ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    # --- Training loop ---
    best_val_loss = float("inf")
    training_log = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            vectors = batch["vectors"].to(device)
            num_objects = batch["num_objects"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                vectors=vectors,
                num_objects=num_objects,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

            if (step + 1) % LOGGING_STEPS == 0 or step == 0:
                logger.info(
                    f"[Stage-2] Epoch {epoch+1}/{epochs}, "
                    f"Step {step+1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # --- Validation ---
        val_loss = _validate_stage2_loss(model, val_loader, device)

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
        }
        training_log.append(epoch_log)

        logger.info(
            f"[Stage-2] Epoch {epoch+1}/{epochs} — "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir = os.path.join(output_dir, "best_checkpoint")
            save_checkpoint(model, ckpt_dir, epoch=epoch + 1)
            logger.info(f"[Stage-2] Saved best checkpoint (val_loss={val_loss:.4f})")

    # Save final checkpoint
    final_dir = os.path.join(output_dir, "final_checkpoint")
    save_checkpoint(model, final_dir, epoch=epochs)

    # Save training log
    log_path = os.path.join(output_dir, "stage2_training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"[Stage-2] Training log saved to {log_path}")

    logger.info("[Stage-2] Training complete!")
    return final_dir


def _validate_stage2_loss(model, val_loader, device):
    """Compute validation loss for Stage-2."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            vectors = batch["vectors"].to(device)
            num_objects = batch["num_objects"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                vectors=vectors,
                num_objects=num_objects,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)
