# main.py

"""
Entry point for the nuScenes-mini + LLM driving toy pipeline.

Pipeline:
1) Build datasets (Stage 1 captioning + Stage 2 QA) from nuScenes-mini.
2) Train Stage 1 model: vector -> caption.
3) Train Stage 2 model: caption + question -> driving answer.

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
)
from llm_driving.datasets_builder import build_datasets_full_mini
from llm_driving.training import train_stage1, train_stage2
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
        from llm_driving.logging_utils import setup_logging  # your file defines this
        setup_logging(RUN_DIR, "train.log")
    except Exception:
        _fallback_setup_logging(RUN_DIR, "train.log")

    logger.info("=" * 90)
    logger.info("[MAIN] LLM driving pipeline started.")
    logger.info(f"[MAIN] RUN_ID : {RUN_ID}")
    logger.info(f"[MAIN] RUN_DIR: {RUN_DIR}")
    logger.info("-" * 90)
    logger.info(f"[MAIN] USE_VECTOR_PREFIX : {cfg.USE_VECTOR_PREFIX}")
    logger.info(f"[MAIN] USE_LORA          : {cfg.USE_LORA}")
    logger.info(f"[MAIN] FREEZE_BASE_MODEL : {cfg.FREEZE_BASE_MODEL}")
    logger.info(f"[MAIN] PREFIX_LEN        : {cfg.PREFIX_LEN}")
    if cfg.USE_VECTOR_PREFIX:
        logger.info(f"[MAIN] TOKENS_PER_OBJECT : {cfg.TOKENS_PER_OBJECT}")
        logger.info(f"[MAIN] NUM_OBJECT_TYPES  : {cfg.NUM_OBJECT_TYPES}")
        logger.info(f"[MAIN] STAGE1_EPOCHS     : {cfg.STAGE1_EPOCHS}")
        logger.info(f"[MAIN] STAGE1_TEXT_PROMPT : {cfg.STAGE1_TEXT_PROMPT!r}")
    logger.info(f"[MAIN] Captioning dataset path : {CAPTIONING_DATA_PATH}")
    logger.info(f"[MAIN] QA dataset path         : {QA_DATA_PATH}")
    logger.info("=" * 90)

    # Check peft is available if LoRA is enabled
    if cfg.USE_LORA:
        try:
            import peft
            logger.info(f"[MAIN] peft version: {peft.__version__}")
        except ImportError:
            logger.error("[MAIN] peft is required for LoRA but not installed! Run: pip install peft")
            raise

    _save_config_snapshot()

    # 1) Build datasets using all scenes in nuScenes-mini
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

    # 2) Stage 1: Vector -> Caption pretraining
    logger.info("\n[MAIN] Step 2/3: Training Stage 1 (vector → caption)...")
    model_stage1, tokenizer = train_stage1(CAPTIONING_DATA_PATH)
    logger.info("[MAIN] Step 2/3 DONE: Stage 1 training finished.")

    # 3) Stage 2: Driving QA finetuning
    logger.info("\n[MAIN] Step 3/3: Training Stage 2 (caption + question → action/answer)...")
    _ = train_stage2(model_stage1, tokenizer, QA_DATA_PATH)
    logger.info("[MAIN] Step 3/3 DONE: Stage 2 training finished.")

    logger.info("\n[MAIN] Pipeline finished successfully.")
    logger.info("=" * 90)
    logger.info(f"[MAIN] All outputs saved under: {RUN_DIR}")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
