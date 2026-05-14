"""
Step 4 / M1 — go/no-go checkpoint for Stage 1 caption generation.

Goal: after Stage 1 training finishes, before kicking off Stage 2 training,
verify that the Stage 1 model has actually learned to emit temporal
descriptors (Part C lines: "closing", "decel", "receding", "steady").

If the model is NOT generating these terms, Stage 2 training will see
snapshot-only captions and the whole pipeline collapses back to a null
Step 2 result. STOP and diagnose Stage 1 first.

Decision rule (configurable below):
  >= 6/10 captions contain temporal terms → continue to Stage 2
  <= 3/10 captions contain temporal terms → STOP; debug Stage 1
  4-5/10 ambiguous: continue but flag for closer look post-eval

Usage:
    python check_stage1_captions.py --run <RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from datasets import Dataset

from llm_driving import config as cfg
from llm_driving.lora_utils import load_checkpoint
from llm_driving.vector_encoder import parse_vec_str


TEMPORAL_TERMS = ("closing", "decel", "receding", "steady")
DECISION_PASS = 6   # >= this many out of N → green light
DECISION_FAIL = 3   # <= this many out of N → red light


def _has_temporal(text: str) -> bool:
    low = text.lower()
    return any(term in low for term in TEMPORAL_TERMS)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run ID under runs/")
    p.add_argument(
        "--n_samples", type=int, default=10,
        help="Number of mid-scene val samples to inspect (default 10).",
    )
    p.add_argument(
        "--device", default=None, help="cuda:0 etc; defaults to auto.",
    )
    args = p.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    run_dir = os.path.join(cfg.RUNS_DIR, args.run)
    stage1_ckpt = os.path.join(run_dir, "stage1", "best_checkpoint")
    qa_path = os.path.join(run_dir, "data", "driving_qa_data.json")

    if not os.path.isdir(stage1_ckpt):
        print(f"[M1] Stage 1 checkpoint not found at {stage1_ckpt}")
        sys.exit(2)
    if not os.path.isfile(qa_path):
        print(f"[M1] QA data not found at {qa_path}")
        sys.exit(2)

    print(f"[M1] Loading Stage 1 from {stage1_ckpt}")
    model = load_checkpoint(
        model_name=cfg.MODEL_NAME,
        checkpoint_dir=stage1_ckpt,
        device=device,
        apply_lora_config=cfg.USE_LORA,
    )
    model.eval()
    print(f"[M1] Loaded. use_temporal={getattr(model, 'use_temporal', False)}")

    # Tokenizer for the text prompt. Stage 1 saves raw .pt state dicts,
    # not HF save_pretrained format, so the checkpoint dir doesn't have a
    # config.json / tokenizer.json. Load from the base model name instead
    # (same arch ⇒ same tokenizer).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)

    print(f"[M1] Loading QA data from {qa_path}")
    with open(qa_path, "r") as f:
        data = json.load(f)

    # Pick mid-scene action samples with a fully-populated window.
    # Deduplicate by (scene_idx, frame_in_scene) so each unique frame
    # appears at most once — the 5 action questions per frame would
    # otherwise inflate the sample count without adding diversity.
    K = int(cfg.TEMPORAL_WINDOW)
    seen_frames = set()
    candidates = []
    for sample in data:
        if (
            sample.get("question_type") == "action"
            and int(sample.get("frame_in_scene", 0)) > K
            and int(sample.get("window_len", 1)) == K
        ):
            key = (sample.get("scene_idx"), sample.get("frame_in_scene"))
            if key in seen_frames:
                continue
            seen_frames.add(key)
            candidates.append(sample)
        if len(candidates) >= args.n_samples:
            break

    if not candidates:
        print("[M1] No suitable mid-scene samples found.")
        sys.exit(2)

    # Already deduplicated and trimmed to n_samples above.
    selected = candidates

    n_passed = 0
    print(f"\n[M1] Inspecting {len(selected)} captions (mid-scene, fully-populated window):\n")
    for i, sample in enumerate(selected):
        if "vectors" in sample and sample["vectors"]:
            vectors = sample["vectors"]
        else:
            vectors_arr, _n = parse_vec_str(
                sample.get("vec_str", ""), cfg.MAX_OBJECTS, cfg.VECTOR_DIM,
            )
            vectors = vectors_arr.tolist()

        num_obj = int(sample.get("num_objects", sample.get("use_n", 0)))
        vt = torch.tensor([vectors], dtype=torch.float32, device=device)
        nt = torch.tensor([num_obj], dtype=torch.long, device=device)

        # Temporal kwargs.
        vw_t = now_t = wl_t = opm_t = None
        if getattr(model, "use_temporal", False):
            vw_sample = sample.get("vectors_window")
            now_sample = sample.get("num_objects_window")
            wl_sample = sample.get("window_len")
            opm_sample = sample.get("object_present_mask")
            vw_t = torch.tensor([vw_sample], dtype=torch.float32, device=device)
            now_t = torch.tensor([now_sample], dtype=torch.long, device=device)
            wl_t = torch.tensor([int(wl_sample)], dtype=torch.long, device=device)
            if opm_sample is not None:
                opm_t = torch.tensor([opm_sample], dtype=torch.bool, device=device)

        text_in = tokenizer(
            cfg.STAGE1_TEXT_PROMPT,
            return_tensors="pt",
            max_length=cfg.STAGE1_MAX_INPUT_LEN,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            pred_ids = model.generate(
                vectors=vt, num_objects=nt,
                vectors_window=vw_t, num_objects_window=now_t,
                window_len=wl_t, object_present_mask=opm_t,
                input_ids=text_in["input_ids"], attention_mask=text_in["attention_mask"],
                max_new_tokens=cfg.GEN_MAX_NEW_TOKENS_STAGE1,
                num_beams=cfg.GEN_NUM_BEAMS,
                early_stopping=cfg.GEN_EARLY_STOPPING,
                no_repeat_ngram_size=cfg.GEN_NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=cfg.GEN_REPETITION_PENALTY,
            )
        gen_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        has_temp = _has_temporal(gen_text)
        if has_temp:
            n_passed += 1

        gt_text = sample.get("oracle_caption_debug", "<no oracle caption stored>")
        print(f"--- sample {i+1}/{len(selected)} (frame_in_scene={sample.get('frame_in_scene')}, has_temp={has_temp}) ---")
        print("GT  :", gt_text[:300].replace("\n", " | "))
        print("GEN :", gen_text[:300].replace("\n", " | "))
        print()

    print(f"[M1] {n_passed}/{len(selected)} captions contain temporal terms.")

    if n_passed >= DECISION_PASS:
        print(f"[M1] PASS — Stage 1 has learned to emit temporal info. Proceed to Stage 2.")
        sys.exit(0)
    elif n_passed <= DECISION_FAIL:
        print(
            f"[M1] FAIL — Stage 1 is NOT emitting temporal info "
            f"({n_passed}/{len(selected)} below {DECISION_FAIL + 1}). "
            f"STOP before Stage 2 training. Diagnose: "
            f"(a) is the tracked window being passed to the model? "
            f"(b) is lanGen producing temporal lines in the targets? "
            f"(c) is the prefix encoder collapsing?"
        )
        sys.exit(1)
    else:
        print(
            f"[M1] AMBIGUOUS — {n_passed}/{len(selected)} below ideal but above failure. "
            f"Proceed with caution; double-check Stage 2 eval results."
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
