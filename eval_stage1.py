"""
Standalone single-GPU evaluation for Stage 1 (vectors -> caption).

Loads the Stage 1 checkpoint from a finished run and runs full beam-search
generation over the validation split, producing:
  - runs/<run_id>/stage1/val_predictions.json
  - runs/<run_id>/stage1/eval_metrics.json  (adds/updates bleu1, rouge_l, val_loss)

This script is the multi-GPU-safe replacement for the in-training eval block
that was skipped when `accelerate launch --num_processes>1 main.py` is used.

Usage:
    python eval_stage1.py --run 20260421_120000
    python eval_stage1.py --run <RUN_ID> --device cuda:0 --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llm_driving import config as cfg
from llm_driving.data_collator import VectorPrefixDataCollator
from llm_driving.lora_utils import load_checkpoint
from llm_driving.training import (
    VectorPrefixDataset,
    _compute_bleu1_list,
    _compute_rouge_l_list,
    _validate_stage1_prefix,
)
from llm_driving.vector_encoder import parse_vec_str

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("eval_stage1")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-val evaluation for Stage 1")
    p.add_argument("--run", required=True, help="Run ID (folder name under runs/)")
    p.add_argument(
        "--checkpoint",
        default="best_checkpoint",
        help="Checkpoint subdir under runs/<run>/stage1/ (default: best_checkpoint)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g. cuda:0, cpu). Defaults to cuda if available.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=cfg.STAGE1_BATCH_SIZE,
        help=f"Eval batch size (default: {cfg.STAGE1_BATCH_SIZE})",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    run_dir = os.path.join(cfg.RUNS_DIR, args.run)
    stage1_dir = os.path.join(run_dir, "stage1")
    checkpoint_dir = os.path.join(stage1_dir, args.checkpoint)
    captioning_path = os.path.join(run_dir, "data", "vector_captioning_data.json")

    if not os.path.isdir(checkpoint_dir):
        logger.error(f"Checkpoint dir not found: {checkpoint_dir}")
        sys.exit(1)
    if not os.path.isfile(captioning_path):
        logger.error(f"Captioning data not found: {captioning_path}")
        sys.exit(1)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"[EVAL1] Run          : {args.run}")
    logger.info(f"[EVAL1] Checkpoint   : {checkpoint_dir}")
    logger.info(f"[EVAL1] Captioning   : {captioning_path}")
    logger.info(f"[EVAL1] Device       : {device}")
    logger.info(f"[EVAL1] Batch size   : {args.batch_size}")

    # --- Load data and reproduce the same train/val split used in training ---
    with open(captioning_path, "r") as f:
        data = json.load(f)

    for sample in data:
        if "vectors" not in sample:
            input_text = sample.get("input", "")
            vec_str = input_text.split("\n", 1)[1] if "\n" in input_text else ""
            vectors, n = parse_vec_str(vec_str, cfg.MAX_OBJECTS, cfg.VECTOR_DIM)
            sample["vectors"] = vectors.tolist()
            sample["num_objects"] = n

    torch.manual_seed(cfg.SEED)
    n_val = max(1, int(len(data) * 0.2))
    n_train = len(data) - n_val
    indices = torch.randperm(len(data)).tolist()
    val_samples = [data[i] for i in indices[n_train:]]
    logger.info(f"[EVAL1] Total samples: {len(data)} | Val: {len(val_samples)}")

    # Stage 1 uses the minimal prompt
    for s in val_samples:
        s["input"] = cfg.STAGE1_TEXT_PROMPT

    # --- Load model + tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    model = load_checkpoint(
        model_name=cfg.MODEL_NAME,
        checkpoint_dir=checkpoint_dir,
        device=device,
        apply_lora_config=cfg.USE_LORA,
    )
    model.eval()

    # --- DataLoader ---
    collator = VectorPrefixDataCollator(
        tokenizer=tokenizer,
        max_input_length=cfg.STAGE1_MAX_INPUT_LEN,
        max_target_length=cfg.STAGE1_MAX_TARGET_LEN,
        max_objects=cfg.MAX_OBJECTS,
        vector_dim=cfg.VECTOR_DIM,
    )
    val_loader = DataLoader(
        VectorPrefixDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # --- Run full beam-search generation ---
    logger.info(f"[EVAL1] Running beam-search generation on {len(val_samples)} val samples...")
    val_loss, preds, refs = _validate_stage1_prefix(
        model, val_loader, tokenizer, device, generate=True
    )
    bleu1_score = _compute_bleu1_list(preds, refs)
    rouge_l_score = _compute_rouge_l_list(preds, refs)
    logger.info(
        f"[EVAL1] val_loss={val_loss:.4f} | BLEU-1={bleu1_score:.4f} | "
        f"ROUGE-L={rouge_l_score:.4f} | n_preds={len(preds)}"
    )

    # --- Write predictions ---
    os.makedirs(stage1_dir, exist_ok=True)
    pred_path = os.path.join(stage1_dir, "val_predictions.json")
    preds_data = [{"prediction": p, "ground_truth": r} for p, r in zip(preds, refs)]
    with open(pred_path, "w") as f:
        json.dump(preds_data, f, indent=2)
    logger.info(f"[EVAL1] Wrote {len(preds_data)} predictions to {pred_path}")

    # --- Update eval_metrics.json (merge with anything training wrote) ---
    metrics_path = os.path.join(stage1_dir, "eval_metrics.json")
    metrics: dict = {}
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f) or {}
        except Exception:
            metrics = {}
    metrics.update(
        {
            "eval_loss": float(val_loss),
            "bleu1": float(bleu1_score),
            "rouge_l": float(rouge_l_score),
            "n_val_samples": len(val_samples),
        }
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[EVAL1] Wrote metrics to {metrics_path}")
    logger.info("[EVAL1] Done.")


if __name__ == "__main__":
    main()
