"""
Standalone single-GPU evaluation for Stage 1 (vectors -> caption).

Loads the Stage 1 checkpoint from a finished run and runs full beam-search
generation over the validation split, producing:
  - runs/<run_id>/stage1/val_predictions.json
  - runs/<run_id>/stage1/eval_metrics.json  (adds/updates bleu1, rouge_l, val_loss)

This script is the multi-GPU-safe replacement for the in-training eval block
that was skipped when `accelerate launch --num_processes>1 main.py` is used.

Single-GPU usage:
    python eval_stage1.py --run 20260421_120000
    python eval_stage1.py --run <RUN_ID> --device cuda:0 --batch_size 4

Multi-GPU (embarrassingly-parallel sharding — no NCCL, no hangs):
    # Launch one process per GPU, each handling a disjoint slice of the val set.
    CUDA_VISIBLE_DEVICES=0 python eval_stage1.py --run <RUN_ID> --shard_idx 0 --num_shards 3 &
    CUDA_VISIBLE_DEVICES=1 python eval_stage1.py --run <RUN_ID> --shard_idx 1 --num_shards 3 &
    CUDA_VISIBLE_DEVICES=2 python eval_stage1.py --run <RUN_ID> --shard_idx 2 --num_shards 3 &
    wait
    python merge_shards.py --run <RUN_ID> --stage 1 --num_shards 3
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
from llm_driving.logging_utils import setup_logging
from llm_driving.lora_utils import load_checkpoint
from llm_driving.training import (
    VectorPrefixDataset,
    _compute_bleu1_list,
    _compute_rouge_l_list,
    _validate_stage1_prefix,
)
from llm_driving.vector_encoder import parse_vec_str

logger = logging.getLogger("llm_driving")


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
    p.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Shard index (0-based). Used with --num_shards for multi-GPU.",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards. Each process handles samples[shard_idx::num_shards].",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.num_shards < 1 or args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        logging.basicConfig(level=logging.INFO)
        logger.error(
            f"Invalid sharding args: shard_idx={args.shard_idx} num_shards={args.num_shards}"
        )
        sys.exit(1)
    is_sharded = args.num_shards > 1

    run_dir = os.path.join(cfg.RUNS_DIR, args.run)
    stage1_dir = os.path.join(run_dir, "stage1")
    checkpoint_dir = os.path.join(stage1_dir, args.checkpoint)
    captioning_path = os.path.join(run_dir, "data", "vector_captioning_data.json")

    # Validate inputs BEFORE creating any directories, to avoid leaking empty
    # runs/<bogus_id>/stage1/ folders if the user passes a wrong --run.
    if not os.path.isdir(stage1_dir):
        logging.basicConfig(level=logging.INFO)
        logger.error(f"Stage 1 dir not found: {stage1_dir}")
        sys.exit(1)
    if not os.path.isdir(checkpoint_dir):
        logging.basicConfig(level=logging.INFO)
        logger.error(f"Checkpoint dir not found: {checkpoint_dir}")
        sys.exit(1)
    if not os.path.isfile(captioning_path):
        logging.basicConfig(level=logging.INFO)
        logger.error(f"Captioning data not found: {captioning_path}")
        sys.exit(1)

    # All inputs valid — wire up combined file + stdout logging
    # (writes to runs/<run>/stage1/eval_stage1[_shardN].log).
    log_filename = (
        f"eval_stage1_shard{args.shard_idx}.log" if is_sharded else "eval_stage1.log"
    )
    setup_logging(stage1_dir, log_filename)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"[EVAL1] Run          : {args.run}")
    logger.info(f"[EVAL1] Checkpoint   : {checkpoint_dir}")
    logger.info(f"[EVAL1] Captioning   : {captioning_path}")
    logger.info(f"[EVAL1] Device       : {device}")
    logger.info(f"[EVAL1] Batch size   : {args.batch_size}")
    if is_sharded:
        logger.info(f"[EVAL1] Sharding     : shard {args.shard_idx} of {args.num_shards}")

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

    # Reproduce the EXACT same train/val split used by training.py.
    # When USE_SCENE_LEVEL_SPLIT=True, this groups by scene_idx; otherwise
    # falls back to the random sample-level split.
    if getattr(cfg, "USE_SCENE_LEVEL_SPLIT", False):
        from llm_driving.training import _scene_aware_split
        _, val_samples = _scene_aware_split(
            data,
            test_size=getattr(cfg, "SCENE_LEVEL_SPLIT_TEST_SIZE", 0.2),
            seed=getattr(cfg, "SCENE_LEVEL_SPLIT_SEED", 42),
        )
    else:
        torch.manual_seed(cfg.SEED)
        n_val = max(1, int(len(data) * 0.2))
        n_train = len(data) - n_val
        indices = torch.randperm(len(data)).tolist()
        val_samples = [data[i] for i in indices[n_train:]]
    full_val_size = len(val_samples)

    # Shard the val samples: shard i handles samples[i::N]. Stride-slicing keeps
    # each shard roughly equal in size and preserves sample identity.
    if is_sharded:
        val_samples = val_samples[args.shard_idx :: args.num_shards]
        logger.info(
            f"[EVAL1] Total samples: {len(data)} | Full val: {full_val_size} | "
            f"This shard: {len(val_samples)}"
        )
    else:
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
    # Step 2 / Part B: when USE_TEMPORAL is on, the model expects
    # vectors_window / num_objects_window / window_len in every batch.
    # Mirror the training-time collator config so the eval batch shape
    # matches what TemporalVectorEncoder consumes.
    _collator_temporal_window = (
        int(getattr(cfg, "TEMPORAL_WINDOW", 4))
        if bool(getattr(cfg, "USE_TEMPORAL", False))
        else 0
    )
    collator = VectorPrefixDataCollator(
        tokenizer=tokenizer,
        max_input_length=cfg.STAGE1_MAX_INPUT_LEN,
        max_target_length=cfg.STAGE1_MAX_TARGET_LEN,
        max_objects=cfg.MAX_OBJECTS,
        vector_dim=cfg.VECTOR_DIM,
        temporal_window=_collator_temporal_window,
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
    if is_sharded:
        pred_filename = f"val_predictions_shard{args.shard_idx}.json"
    else:
        pred_filename = "val_predictions.json"
    pred_path = os.path.join(stage1_dir, pred_filename)
    preds_data = [{"prediction": p, "ground_truth": r} for p, r in zip(preds, refs)]
    with open(pred_path, "w") as f:
        json.dump(preds_data, f, indent=2)
    logger.info(f"[EVAL1] Wrote {len(preds_data)} predictions to {pred_path}")

    # --- Update eval_metrics.json (merge with anything training wrote) ---
    # In sharded mode, write a per-shard metrics file so merge_shards.py has
    # the shard's val_loss (loss is computed directly on local forward passes;
    # bleu/rouge will be recomputed by merge_shards.py from concatenated preds).
    if is_sharded:
        metrics_path = os.path.join(
            stage1_dir, f"eval_metrics_shard{args.shard_idx}.json"
        )
        metrics: dict = {
            "eval_loss": float(val_loss),
            "bleu1": float(bleu1_score),
            "rouge_l": float(rouge_l_score),
            "n_val_samples": len(val_samples),
            "shard_idx": int(args.shard_idx),
            "num_shards": int(args.num_shards),
        }
    else:
        metrics_path = os.path.join(stage1_dir, "eval_metrics.json")
        metrics = {}
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
    if is_sharded:
        logger.info(
            f"[EVAL1] Shard {args.shard_idx} done. Run "
            f"'python merge_shards.py --run {args.run} --stage 1 "
            f"--num_shards {args.num_shards}' after all shards finish."
        )
    logger.info("[EVAL1] Done.")


if __name__ == "__main__":
    main()
