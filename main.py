# main.py

"""
Entry point for the nuScenes-mini + LLM driving pipeline.

Supports two modes controlled by USE_VECTOR_PREFIX in config.py:
  - False: Text-only baseline (existing pipeline via training.py)
  - True:  Vector prefix pipeline (via training_prefix.py)

Pipeline (both modes):
  1) Build datasets (Stage 1 captioning + Stage 2 QA) from nuScenes-mini.
  2) Train Stage 1: vector -> caption.
  3) Train Stage 2: caption + question -> driving answer.

All run artifacts (datasets + outputs) are saved under runs/<RUN_ID>/...
"""

import json
import os
import sys
import logging

from llm_driving.config import (
    RUN_ID,
    RUN_DIR,
    CAPTIONING_DATA_PATH,
    QA_DATA_PATH,
    USE_VECTOR_PREFIX,
    RUN_STAGE1,
    RUN_STAGE2,
    STAGE1_OUTPUT_DIR,
)
from llm_driving.datasets_builder import build_datasets_full_mini
from llm_driving import config as cfg

logger = logging.getLogger("llm_driving")


def _fallback_setup_logging(run_dir: str, log_filename: str = "train.log") -> None:
    """Fallback if logging_utils import fails (keeps pipeline runnable)."""
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, log_filename)

    log = logging.getLogger("llm_driving")
    log.setLevel(logging.INFO)
    log.propagate = False

    if not log.handlers:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        log.addHandler(fh)

        ch = logging.StreamHandler(sys.__stdout__)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        log.addHandler(ch)

    log.info(f"[LOG] Fallback logging initialized. File: {log_path}")


def _save_config_snapshot():
    """
    Save a snapshot of current config values for reproducibility.
    (Only uppercase fields are saved.)
    """
    snapshot = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    out_path = os.path.join(RUN_DIR, "config_snapshot.json")
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    logger.info(f"[MAIN] Saved config snapshot to: {out_path}")


def main():
    # Initialize logging ASAP
    try:
        from llm_driving.logging_utils import setup_logging
        setup_logging(RUN_DIR, "train.log")
    except Exception:
        _fallback_setup_logging(RUN_DIR, "train.log")

    logger.info("=" * 90)
    logger.info("[MAIN] LLM driving pipeline started.")
    logger.info(f"[MAIN] RUN_ID : {RUN_ID}")
    logger.info(f"[MAIN] RUN_DIR: {RUN_DIR}")
    logger.info(f"[MAIN] Pipeline mode: {'VECTOR PREFIX' if USE_VECTOR_PREFIX else 'TEXT BASELINE'}")
    logger.info("-" * 90)
    logger.info(f"[MAIN] Captioning dataset path : {CAPTIONING_DATA_PATH}")
    logger.info(f"[MAIN] QA dataset path         : {QA_DATA_PATH}")
    logger.info("=" * 90)

    _save_config_snapshot()

    # ─── Step 1: Build datasets ───────────────────────────────────────────
    logger.info("\n[MAIN] Step 1/3: Building datasets from nuScenes-mini...")

    if os.path.exists(CAPTIONING_DATA_PATH) and os.path.exists(QA_DATA_PATH):
        logger.info("[MAIN] Found existing dataset JSONs. Skipping dataset building.")
    else:
        captioning_samples, qa_samples = build_datasets_full_mini(
            max_frames_per_scene=None,
            captioning_path=CAPTIONING_DATA_PATH,
            qa_path=QA_DATA_PATH,
        )
        logger.info(
            f"[MAIN] Step 1/3 DONE: "
            f"{len(captioning_samples)} captioning samples, "
            f"{len(qa_samples)} QA samples."
        )

    # ─── Route based on USE_VECTOR_PREFIX ─────────────────────────────────
    if USE_VECTOR_PREFIX:
        _run_vector_prefix_pipeline()
    else:
        _run_text_baseline_pipeline()

    logger.info("\n[MAIN] Pipeline finished successfully.")
    logger.info("=" * 90)
    logger.info(f"[MAIN] All outputs saved under: {RUN_DIR}")
    logger.info("=" * 90)


def _run_vector_prefix_pipeline():
    """Run the vector prefix pipeline (Phases 1-11 implementation)."""
    from llm_driving.training_prefix import train_stage1_prefix, train_stage2_prefix

    # Stage 1: Vector prefix → Caption
    if RUN_STAGE1:
        logger.info("\n[MAIN] Step 2/3: Training Stage 1 — Vector Prefix → Caption...")
        stage1_dir = train_stage1_prefix(CAPTIONING_DATA_PATH)
        logger.info(f"[MAIN] Step 2/3 DONE: Stage 1 checkpoint at {stage1_dir}")
    else:
        # Use existing checkpoint
        stage1_dir = os.path.join(STAGE1_OUTPUT_DIR, "final_checkpoint")
        logger.info(f"[MAIN] Step 2/3 SKIPPED: Using existing Stage 1 at {stage1_dir}")

    # Stage 2: Driving QA (from Stage 1 checkpoint)
    if RUN_STAGE2:
        logger.info("\n[MAIN] Step 3/3: Training Stage 2 — Driving QA with risk...")
        stage2_dir = train_stage2_prefix(QA_DATA_PATH, stage1_dir)
        logger.info(f"[MAIN] Step 3/3 DONE: Stage 2 checkpoint at {stage2_dir}")
    else:
        logger.info("[MAIN] Step 3/3 SKIPPED: RUN_STAGE2 = False")


def _run_text_baseline_pipeline():
    """Run the existing text-only baseline pipeline (unchanged)."""
    from llm_driving.training import train_stage1, train_stage2

    # Stage 1
    if RUN_STAGE1:
        logger.info("\n[MAIN] Step 2/3: Training Stage 1 (text baseline: vector → caption)...")
        model_stage1, tokenizer = train_stage1(CAPTIONING_DATA_PATH)
        logger.info("[MAIN] Step 2/3 DONE: Stage 1 training finished.")
    else:
        logger.info("[MAIN] Step 2/3 SKIPPED: RUN_STAGE1 = False")
        model_stage1, tokenizer = None, None

    # Stage 2
    if RUN_STAGE2:
        logger.info("\n[MAIN] Step 3/3: Training Stage 2 (text baseline: caption + Q → action)...")
        if model_stage1 is None:
            logger.error("[MAIN] Cannot run Stage 2 without Stage 1 model in text mode!")
            return
        _ = train_stage2(model_stage1, tokenizer, QA_DATA_PATH)
        logger.info("[MAIN] Step 3/3 DONE: Stage 2 training finished.")
    else:
        logger.info("[MAIN] Step 3/3 SKIPPED: RUN_STAGE2 = False")


if __name__ == "__main__":
    main()
