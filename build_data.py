"""
Build datasets only (Stage 1 captioning + Stage 2 QA) — no training.

Use this BEFORE `accelerate launch --num_processes=N main.py` for full
v1.0-trainval runs to avoid the NCCL distributed-barrier timeout that
fires when rank 0 spends >10 min building data alone while ranks 1..N-1
sit at `state.wait_for_everyone()`.

Workflow:
    # 1. Pin a RUN_ID so the pre-build and the training share runs/<id>/.
    export RUN_ID=20260509_step2_full

    # 2. Build the dataset on a single process (no torch.distributed).
    #    Takes ~hours for v1.0-trainval; safe to run inside tmux.
    python build_data.py

    # 3. Launch multi-GPU training. main.py sees the existing JSONs and
    #    skips the build, so the barrier is hit immediately after init —
    #    well within the 10-min timeout window.
    accelerate launch --num_processes=3 main.py

Usage:
    python build_data.py
"""

from __future__ import annotations

import logging
import os
import sys

from llm_driving.config import (
    RUN_ID,
    RUN_DIR,
    CAPTIONING_DATA_PATH,
    QA_DATA_PATH,
)
from llm_driving.datasets_builder import build_datasets_full_mini

logger = logging.getLogger("llm_driving")


def _setup_logging(run_dir: str, log_filename: str = "build_data.log") -> None:
    """Lightweight logging setup — mirrors main.py's fallback behaviour but
    writes to its own log file so a follow-up `accelerate launch main.py`
    doesn't clobber the build log."""
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, log_filename)

    log = logging.getLogger("llm_driving")
    log.setLevel(logging.INFO)
    log.propagate = False

    if log.handlers:
        # Avoid duplicate handlers if this script is re-imported.
        return

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

    log.info(f"[BUILD_DATA] Logging initialised. File: {log_path}")


def main() -> None:
    _setup_logging(RUN_DIR, "build_data.log")

    logger.info("=" * 90)
    logger.info("[BUILD_DATA] Dataset-only build (no training, no torch.distributed).")
    logger.info(f"[BUILD_DATA] RUN_ID            : {RUN_ID}")
    logger.info(f"[BUILD_DATA] RUN_DIR           : {RUN_DIR}")
    logger.info(f"[BUILD_DATA] Captioning output : {CAPTIONING_DATA_PATH}")
    logger.info(f"[BUILD_DATA] QA output         : {QA_DATA_PATH}")
    logger.info("=" * 90)

    if os.path.exists(CAPTIONING_DATA_PATH) and os.path.exists(QA_DATA_PATH):
        logger.info(
            "[BUILD_DATA] Existing dataset JSONs already present under this RUN_ID. "
            "Nothing to do — delete them if you want a fresh build."
        )
        return

    captioning_samples, qa_samples = build_datasets_full_mini(
        max_frames_per_scene=None,
        captioning_path=CAPTIONING_DATA_PATH,
        qa_path=QA_DATA_PATH,
    )

    # Drop a pointer to this RUN_ID so the next `accelerate launch main.py`
    # auto-resolves to the same run dir without the user having to export
    # RUN_ID manually. config._resolve_run_id() reads it. Explicit RUN_ID
    # env var still wins over this pointer.
    pointer_path = os.path.join("runs", ".latest_build_run_id")
    try:
        os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
        with open(pointer_path, "w") as f:
            f.write(RUN_ID + "\n")
        logger.info(f"[BUILD_DATA] Wrote latest-build pointer: {pointer_path} -> {RUN_ID}")
    except Exception as e:
        logger.warning(
            f"[BUILD_DATA] Could not write pointer {pointer_path}: {e}. "
            f"Falling back: pass RUN_ID={RUN_ID} explicitly to the next launch."
        )

    logger.info(
        f"[BUILD_DATA] DONE. "
        f"{len(captioning_samples)} captioning samples, "
        f"{len(qa_samples)} QA samples."
    )
    logger.info(
        "[BUILD_DATA] Next step: `accelerate launch --num_processes=3 main.py` "
        "(RUN_ID auto-detected from the pointer file)."
    )


if __name__ == "__main__":
    main()
