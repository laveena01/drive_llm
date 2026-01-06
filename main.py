# main.py
"""
Entry point for the nuScenes + LLM driving toy pipeline.

Pipeline:
1) Build datasets (Stage 1 captioning + Stage 2 QA)
2) Train Stage 1: vector -> caption
3) Train Stage 2: caption + question -> driving answer

All run artifacts are saved under runs/<RUN_ID>/...
"""

import json
import os
import time

from llm_driving.config import (
    RUN_ID,
    RUN_DIR,
    CAPTIONING_DATA_PATH,
    QA_DATA_PATH,
    RUN_STAGE1,
    RUN_STAGE2,
)
from llm_driving.datasets_builder import build_datasets_full_mini
from llm_driving.training import train_stage1, train_stage2
from llm_driving import config as cfg
from llm_driving.logger import setup_logger


def _dist_info() -> tuple[int, int, int]:
    """
    Works for torchrun:
      RANK, WORLD_SIZE, LOCAL_RANK are exported.
    If not launched with torchrun, defaults to single process.
    """
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local_rank


def _rank0_log(logger, rank: int, msg: str) -> None:
    if rank == 0:
        logger.info(msg)


def _save_config_snapshot(logger, rank: int) -> None:
    """Save config snapshot once (rank0 only)."""
    if rank != 0:
        return
    snapshot = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    os.makedirs(RUN_DIR, exist_ok=True)
    out_path = os.path.join(RUN_DIR, "config_snapshot.json")
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    logger.info(f"[MAIN] Saved config snapshot to: {out_path}")


def _wait_for_paths(paths: list[str], rank: int, poll_sec: float = 2.0, timeout_sec: int = 3600) -> None:
    """
    Non-rank0 workers wait until dataset JSONs exist (rank0 builds them).
    Timeout avoids hanging forever if rank0 crashed.
    """
    t0 = time.time()
    while True:
        if all(os.path.exists(p) for p in paths):
            return
        if time.time() - t0 > timeout_sec:
            missing = [p for p in paths if not os.path.exists(p)]
            raise TimeoutError(f"[MAIN][rank={rank}] Timed out waiting for: {missing}")
        time.sleep(poll_sec)


def main():
    rank, world, local_rank = _dist_info()

    # (Optional) reduces tokenizer thread spam / contention
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Setup logger with single consolidated log file in RUN_DIR
    log_file = os.path.join(RUN_DIR, "training.log")
    logger = setup_logger(name="llm_driving", log_file=log_file, rank=rank)

    if rank == 0:
        logger.info("=" * 90)
        logger.info("[MAIN] LLM driving pipeline started.")
        logger.info(f"[MAIN] RUN_ID : {RUN_ID}")
        logger.info(f"[MAIN] RUN_DIR: {RUN_DIR}")
        logger.info("-" * 90)
        logger.info(f"[MAIN] Captioning dataset path : {CAPTIONING_DATA_PATH}")
        logger.info(f"[MAIN] QA dataset path         : {QA_DATA_PATH}")
        logger.info(f"[MAIN] Distributed: rank={rank}/{world} (local_rank={local_rank})")
        logger.info("=" * 90)

    _save_config_snapshot(logger, rank)

    # 1) Build datasets (rank0 only)
    if rank == 0:
        logger.info("\n[MAIN] Step 1/3: Building datasets ...")

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

    # Everyone waits until JSONs exist
    _wait_for_paths([CAPTIONING_DATA_PATH, QA_DATA_PATH], rank)

    # 2) Stage 1
    if RUN_STAGE1:
        _rank0_log(logger, rank, "\n[MAIN] Step 2/3: Training Stage 1 (vector → caption)...")
        model_stage1, tokenizer = train_stage1(CAPTIONING_DATA_PATH)
        _rank0_log(logger, rank, "[MAIN] Step 2/3 DONE: Stage 1 training finished.")
    else:
        model_stage1, tokenizer = None, None
        _rank0_log(logger, rank, "\n[MAIN] Step 2/3: Skipping Stage 1 (RUN_STAGE1=False).")

    # 3) Stage 2
    if RUN_STAGE2:
        _rank0_log(logger, rank, "\n[MAIN] Step 3/3: Training Stage 2 (caption + question → action/answer)...")
        _ = train_stage2(model_stage1, tokenizer, QA_DATA_PATH)
        _rank0_log(logger, rank, "[MAIN] Step 3/3 DONE: Stage 2 training finished.")
    else:
        _rank0_log(logger, rank, "\n[MAIN] Step 3/3: Skipping Stage 2 (RUN_STAGE2=False).")

    _rank0_log(logger, rank, "\n[MAIN] Pipeline finished successfully.")
    _rank0_log(logger, rank, "=" * 90)
    _rank0_log(logger, rank, f"[MAIN] All outputs saved under: {RUN_DIR}")
    _rank0_log(logger, rank, "=" * 90)


if __name__ == "__main__":
    main()
