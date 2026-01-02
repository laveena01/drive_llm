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

from llm_driving.config import (
    RUN_ID,
    RUN_DIR,
    CAPTIONING_DATA_PATH,
    QA_DATA_PATH,
)
from llm_driving.datasets_builder import build_datasets_full_mini
from llm_driving.training import train_stage1, train_stage2
from llm_driving import config as cfg


def _save_config_snapshot():
    """
    Save a snapshot of current config values for reproducibility.
    (Only uppercase fields are saved.)
    """
    snapshot = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
    out_path = f"{RUN_DIR}/config_snapshot.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[MAIN] Saved config snapshot to: {out_path}")


def main():
    print("=" * 90)
    print("[MAIN] LLM driving pipeline started.")
    print(f"[MAIN] RUN_ID : {RUN_ID}")
    print(f"[MAIN] RUN_DIR: {RUN_DIR}")
    print("-" * 90)
    print(f"[MAIN] Captioning dataset path : {CAPTIONING_DATA_PATH}")
    print(f"[MAIN] QA dataset path         : {QA_DATA_PATH}")
    print("=" * 90)

    # Save config snapshot early
    _save_config_snapshot()

    # 1) Build datasets using all scenes in nuScenes-mini
   # 1) Build datasets using all scenes in nuScenes-mini
    print("\n[MAIN] Step 1/3: Building datasets from nuScenes-mini...")

    if os.path.exists(CAPTIONING_DATA_PATH) and os.path.exists(QA_DATA_PATH):
        print("[MAIN] Found existing dataset JSONs. Skipping dataset building.")
    else:
        captioning_samples, qa_samples = build_datasets_full_mini(
            max_frames_per_scene=None,   # use all frames in each scene
            captioning_path=CAPTIONING_DATA_PATH,
            qa_path=QA_DATA_PATH,
        )
        print(
            f"[MAIN] Step 1/3 DONE: "
            f"{len(captioning_samples)} captioning samples, "
            f"{len(qa_samples)} QA samples."
        )


    # 2) Stage 1: Vector -> Caption pretraining
    print("\n[MAIN] Step 2/3: Training Stage 1 (vector → caption)...")
    model_stage1, tokenizer = train_stage1(CAPTIONING_DATA_PATH)
    print("[MAIN] Step 2/3 DONE: Stage 1 training finished.")

    # 3) Stage 2: Driving QA finetuning
    print("\n[MAIN] Step 3/3: Training Stage 2 (caption + question → action/answer)...")
    _ = train_stage2(model_stage1, tokenizer, QA_DATA_PATH)
    print("[MAIN] Step 3/3 DONE: Stage 2 training finished.")

    print("\n[MAIN] Pipeline finished successfully.")
    print("=" * 90)
    print(f"[MAIN] All outputs saved under: {RUN_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()
