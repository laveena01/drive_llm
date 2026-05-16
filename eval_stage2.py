"""
Standalone single-GPU evaluation for Stage 2 (caption + risk -> action).

Loads both the Stage 1 checkpoint (for `stage1_caption` mode) and the Stage 2
checkpoint, then runs full beam-search generation over the validation split,
producing:
  - runs/<run_id>/stage2/val_predictions_oracle_caption.json
  - runs/<run_id>/stage2/val_predictions_stage1_caption.json
  - runs/<run_id>/stage2/eval_metrics.json  (merges with anything training wrote)

This script is the multi-GPU-safe replacement for the in-training post-Stage-2
eval block that was skipped when `accelerate launch --num_processes>1 main.py`
is used.

Single-GPU usage:
    python eval_stage2.py --run 20260421_120000
    python eval_stage2.py --run <RUN_ID> --device cuda:0

Multi-GPU (embarrassingly-parallel sharding — no NCCL, no hangs):
    # Launch one process per GPU, each handling a disjoint slice of the val set.
    CUDA_VISIBLE_DEVICES=0 python eval_stage2.py --run <RUN_ID> --shard_idx 0 --num_shards 3 &
    CUDA_VISIBLE_DEVICES=1 python eval_stage2.py --run <RUN_ID> --shard_idx 1 --num_shards 3 &
    CUDA_VISIBLE_DEVICES=2 python eval_stage2.py --run <RUN_ID> --shard_idx 2 --num_shards 3 &
    wait
    python merge_shards.py --run <RUN_ID> --stage 2 --num_shards 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_driving import config as cfg
from llm_driving.eval_extras import (
    compute_future_summary,
    compute_slice_metrics,
    enrich_qa_samples,
    extract_hard_cases,
)
from llm_driving.logging_utils import setup_logging
from llm_driving.lora_utils import load_checkpoint as load_stage1_checkpoint
from llm_driving.training import (
    _build_stage2_prompt_from_caption,
    _ensure_paper_format,
    _extract_brake_percent,
    _extract_risk_level,
    _format_compliance_5line,
    _map_text_to_action_label,
    _print_eval_risk_summary,
    bleu1,
    enforce_5_lines,
    rouge_l_f1,
)
from llm_driving.vector_encoder import parse_vec_str

logger = logging.getLogger("llm_driving")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-val evaluation for Stage 2")
    p.add_argument("--run", required=True, help="Run ID (folder name under runs/)")
    p.add_argument(
        "--stage1_checkpoint",
        default="best_checkpoint",
        help="Checkpoint subdir under runs/<run>/stage1/ (default: best_checkpoint)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device (e.g. cuda:0, cpu). Defaults to cuda if available.",
    )
    p.add_argument(
        "--skip_oracle",
        action="store_true",
        help="Skip the oracle_caption evaluation pass.",
    )
    p.add_argument(
        "--skip_stage1",
        action="store_true",
        help="Skip the stage1_caption evaluation pass (no Stage 1 model needed).",
    )
    p.add_argument(
        "--skip_risk_masked",
        action="store_true",
        help="Skip the risk_masked_caption evaluation pass (Step 5-lite ablation column).",
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
    stage1_ckpt_dir = os.path.join(run_dir, "stage1", args.stage1_checkpoint)
    stage2_dir = os.path.join(run_dir, "stage2")
    qa_path = os.path.join(run_dir, "data", "driving_qa_data.json")

    if not os.path.isdir(stage2_dir):
        # stage2_dir missing -> can't set up file logging under it. Log to terminal
        # via a minimal basicConfig and exit.
        logging.basicConfig(level=logging.INFO)
        logger.error(f"Stage 2 checkpoint dir not found: {stage2_dir}")
        sys.exit(1)

    # stage2_dir exists (Stage 2 training produced it). Wire up combined
    # file + stdout logging (writes to runs/<run>/stage2/eval_stage2[_shardN].log).
    log_filename = (
        f"eval_stage2_shard{args.shard_idx}.log" if is_sharded else "eval_stage2.log"
    )
    setup_logging(stage2_dir, log_filename)
    if not os.path.isfile(qa_path):
        logger.error(f"QA data not found: {qa_path}")
        sys.exit(1)
    if not args.skip_stage1 and not os.path.isdir(stage1_ckpt_dir):
        logger.error(
            f"Stage 1 checkpoint not found: {stage1_ckpt_dir}. "
            f"Pass --skip_stage1 to evaluate oracle_caption only."
        )
        sys.exit(1)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"[EVAL2] Run          : {args.run}")
    logger.info(f"[EVAL2] Stage1 ckpt  : {stage1_ckpt_dir}")
    logger.info(f"[EVAL2] Stage2 dir   : {stage2_dir}")
    logger.info(f"[EVAL2] QA data      : {qa_path}")
    logger.info(f"[EVAL2] Device       : {device}")
    if is_sharded:
        logger.info(f"[EVAL2] Sharding     : shard {args.shard_idx} of {args.num_shards}")

    # --- Load Stage 2 (action model) + tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(stage2_dir)
    model_stage2 = AutoModelForSeq2SeqLM.from_pretrained(stage2_dir).to(device)
    model_stage2.eval()
    logger.info("[EVAL2] Stage 2 model loaded.")

    # --- Load Stage 1 (caption model) if needed ---
    caption_model = None
    if not args.skip_stage1:
        caption_model = load_stage1_checkpoint(
            model_name=cfg.MODEL_NAME,
            checkpoint_dir=stage1_ckpt_dir,
            device=device,
            apply_lora_config=cfg.USE_LORA,
        )
        caption_model.eval()
        logger.info("[EVAL2] Stage 1 caption model loaded.")

    # --- Dataset + reproduce same 80/20 split as training ---
    with open(qa_path, "r") as f:
        data = json.load(f)
    # E1+E2: attach future-aware oracle labels, risk transitions, and density
    # buckets to every QA sample. Idempotent: if a future build already wrote
    # these fields, the values will be overwritten with the same content.
    enrich_qa_samples(data, horizon=4)
    full_ds = Dataset.from_list(data)
    split = full_ds.train_test_split(test_size=0.2, seed=42)
    eval_ds = split["test"]
    full_val_size = len(eval_ds)

    # Shard: stride-slice with samples[i::N]. Dataset.select preserves order.
    if is_sharded:
        shard_indices = list(range(args.shard_idx, full_val_size, args.num_shards))
        eval_ds = eval_ds.select(shard_indices)
        logger.info(
            f"[EVAL2] Total samples: {len(full_ds)} | Full val: {full_val_size} | "
            f"This shard: {len(eval_ds)}"
        )
    else:
        logger.info(f"[EVAL2] Total samples: {len(full_ds)} | Val: {len(eval_ds)}")

    # --- Helpers ---
    def _gen_text_s2(prompt: str, max_new_tokens: int, ensure_paper: bool) -> str:
        if ensure_paper:
            prompt = _ensure_paper_format(prompt)
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=384, truncation=True
        ).to(device)
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
        assert caption_model is not None
        if "vectors" in sample and sample["vectors"]:
            vectors = sample["vectors"]
        else:
            vec_str = sample.get("vec_str", "")
            vectors_arr, n = parse_vec_str(vec_str, cfg.MAX_OBJECTS, cfg.VECTOR_DIM)
            vectors = vectors_arr.tolist()
            sample["num_objects"] = n
        num_obj = sample.get("num_objects", sample.get("use_n", 0))

        cap_device = caption_model.device
        vectors_t = torch.tensor([vectors], dtype=torch.float32).to(cap_device)
        num_obj_t = torch.tensor([num_obj], dtype=torch.long).to(cap_device)

        # Step 2 / Part B + Step 4 / Part A — temporal kwargs.
        vw_t = now_t = wl_t = opm_t = None
        if getattr(caption_model, "use_temporal", False):
            K = int(getattr(cfg, "TEMPORAL_WINDOW", 4))
            vw_sample = sample.get("vectors_window", None)
            now_sample = sample.get("num_objects_window", None)
            wl_sample = sample.get("window_len", None)
            opm_sample = sample.get("object_present_mask", None)
            if vw_sample is not None and now_sample is not None and wl_sample is not None:
                vw_t = torch.tensor([vw_sample], dtype=torch.float32).to(cap_device)
                now_t = torch.tensor([now_sample], dtype=torch.long).to(cap_device)
                wl_t = torch.tensor([int(wl_sample)], dtype=torch.long).to(cap_device)
                if opm_sample is not None:
                    opm_t = torch.tensor([opm_sample], dtype=torch.bool).to(cap_device)
                else:
                    opm_t = torch.ones(
                        1, K, cfg.MAX_OBJECTS, dtype=torch.bool, device=cap_device,
                    )
            else:
                vw_t = torch.zeros(
                    1, K, cfg.MAX_OBJECTS, cfg.VECTOR_DIM,
                    dtype=torch.float32, device=cap_device,
                )
                vw_t[0, K - 1] = vectors_t[0]
                now_t = torch.zeros(1, K, dtype=torch.long, device=cap_device)
                now_t[0, K - 1] = num_obj_t[0]
                wl_t = torch.tensor([1], dtype=torch.long, device=cap_device)
                opm_t = torch.zeros(1, K, cfg.MAX_OBJECTS, dtype=torch.bool, device=cap_device)
                for j in range(int(num_obj_t[0].item())):
                    opm_t[0, K - 1, j] = True

        text_inputs = tokenizer(
            cfg.STAGE1_TEXT_PROMPT,
            return_tensors="pt",
            max_length=cfg.STAGE1_MAX_INPUT_LEN,
            truncation=True,
        ).to(cap_device)

        pred_ids = caption_model.generate(
            vectors=vectors_t,
            num_objects=num_obj_t,
            vectors_window=vw_t,
            num_objects_window=now_t,
            window_len=wl_t,
            object_present_mask=opm_t,
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
            question = sample.get(
                "question", "How should the car drive in this situation and why?"
            )
            risk_text = sample.get("risk_text", "")
            risk_level = sample.get("risk_level", None)
            gt_brake_pct: float | None = None
            pred_brake_pct: float | None = None

            try:
                if mode == "oracle_caption":
                    stage2_prompt = raw_input_text
                    caption_used = None
                elif mode == "risk_masked_caption":
                    # Step 5-lite: use the oracle caption (stored on sample)
                    # but rebuild the prompt with empty risk_text so the
                    # `### RISK` block is stripped (per-call override in the
                    # strengthened gate at training.py:_build_stage2_prompt_from_caption).
                    # Exposes what the model has learned from caption + question
                    # alone, separating real capability from prompt-leak.
                    oracle_cap = sample.get("oracle_caption_debug", "") or ""
                    caption_used = oracle_cap
                    stage2_prompt = _build_stage2_prompt_from_caption(
                        oracle_cap,
                        risk_text="",
                        qa_question=question,
                        question_type=qtype,
                    )
                else:
                    caption_pred = _gen_caption_from_vectors(sample)
                    caption_used = caption_pred
                    stage2_prompt = _build_stage2_prompt_from_caption(
                        caption_pred,
                        risk_text=risk_text,
                        qa_question=question,
                        question_type=qtype,
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
                        gt_brake_pct = _extract_brake_percent(gt_text)
                        pred_brake_pct = _extract_brake_percent(pred_fixed)
                        if gt_brake_pct is not None and pred_brake_pct is not None:
                            brake_mae_sum_on_brake_gt += abs(
                                float(pred_brake_pct) - float(gt_brake_pct)
                            )
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
                    gt_rl = risk_level or _extract_risk_level(gt_text) or ""
                    pr_rl = _extract_risk_level(pred_raw) or ""
                    if gt_rl:
                        risk_total += 1
                        if pr_rl == gt_rl.upper():
                            risk_correct += 1
                    pred_fixed = pred_raw
                    gt_action = "OTHER"
                    pred_action = "OTHER"
                    parse_ok = 0

            except Exception:
                logger.exception(f"[EVAL2] Failed on sample idx={si} (mode={mode})")
                pred_raw = ""
                pred_fixed = ""
                caption_used = None
                gt_action = "OTHER"
                pred_action = "OTHER"
                parse_ok = 0
                stage2_prompt = raw_input_text

            outputs.append(
                {
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
                    # E2/E3 metadata: needed both for slice metrics here and
                    # for the merge_shards.py recomputation pass.
                    "scene_idx": sample.get("scene_idx"),
                    "frame_in_scene": sample.get("frame_in_scene"),
                    "frame_idx": sample.get("frame_idx"),
                    "num_objects": sample.get("num_objects"),
                    "density_bucket": sample.get("density_bucket"),
                    "risk_transition": sample.get("risk_transition"),
                    "risk_level_future": sample.get("risk_level_future"),
                    "brake_required_future": sample.get("brake_required_future"),
                    "gt_brake_pct": gt_brake_pct,
                    "pred_brake_pct": pred_brake_pct,
                    # Vectors carried only to support the hard-case dump (E3).
                    # Adds bytes to the per-shard predictions JSON but keeps
                    # everything self-contained for offline analysis.
                    "vectors": sample.get("vectors"),
                }
            )

            if (si + 1) % 200 == 0:
                logger.info(f"[EVAL2][{mode}] {si + 1}/{len(eval_ds)} samples done")

        metrics = {
            "action_accuracy": float(correct / total) if total > 0 else 0.0,
            "n_action_samples": int(total),
            "missed_brake_rate": float(missed_brake / brake_total) if brake_total > 0 else 0.0,
            "n_brake_gt": int(brake_total),
            "unsafe_continue_high_rate": (
                float(unsafe_continue_high / highcrit_total) if highcrit_total > 0 else 0.0
            ),
            "n_highcrit_action_samples": int(highcrit_total),
            "brake_mae_on_brake_gt": (
                float(brake_mae_sum_on_brake_gt / brake_mae_count_on_brake_gt)
                if brake_mae_count_on_brake_gt > 0
                else 0.0
            ),
            "n_brake_mae_samples": int(brake_mae_count_on_brake_gt),
            "risk_level_accuracy": float(risk_correct / risk_total) if risk_total > 0 else 0.0,
            "n_risk_samples": int(risk_total),
            "bleu1_action": float(bleu_sum / max(1, total)) if total > 0 else 0.0,
            "rougeL_f1_action": float(rouge_sum / max(1, total)) if total > 0 else 0.0,
            "format_compliance_action": float(fmt_sum / max(1, total)) if total > 0 else 0.0,
            "parse_ok_rate_action": float(parse_ok_sum / max(1, total)) if total > 0 else 0.0,
        }
        return metrics, outputs

    # --- Run eval passes ---
    combined_metrics: Dict = {}

    def _pred_filename(mode: str) -> str:
        base = f"val_predictions_{mode}"
        if is_sharded:
            return f"{base}_shard{args.shard_idx}.json"
        return f"{base}.json"

    def _hard_cases_filename(mode: str) -> str:
        base = f"hard_cases_{mode}"
        if is_sharded:
            return f"{base}_shard{args.shard_idx}.json"
        return f"{base}.json"

    def _strip_vectors_for_preds(outs: List[Dict]) -> List[Dict]:
        """Drop the per-sample `vectors` field before writing predictions JSON.
        Vectors are large (~80 floats per row); we only need them in the
        hard-cases dump for offline analysis."""
        slim: List[Dict] = []
        for o in outs:
            o2 = {k: v for k, v in o.items() if k != "vectors"}
            slim.append(o2)
        return slim

    def _finalize_mode(mode: str, metrics: Dict, outputs: List[Dict]) -> Dict:
        """Augment per-mode metrics with slice metrics + future-aware summary,
        and dump per-mode hard cases. Returns the augmented metrics dict."""
        slice_metrics = compute_slice_metrics(outputs)
        future_summary = compute_future_summary(outputs)
        augmented = {
            **metrics,
            **future_summary,
            "slices": slice_metrics,
        }
        hard_cases = extract_hard_cases(outputs, max_per_kind=200)
        hc_path = os.path.join(stage2_dir, _hard_cases_filename(mode))
        with open(hc_path, "w") as f:
            json.dump(hard_cases, f, indent=2)
        logger.info(
            f"[EVAL2] {mode}: wrote {len(hard_cases)} hard cases to {hc_path}"
        )
        return augmented

    if not args.skip_oracle:
        logger.info("[EVAL2] Running oracle_caption pass...")
        oracle_metrics, oracle_outputs = run_eval("oracle_caption")
        logger.info(f"[EVAL2] oracle_caption metrics: {oracle_metrics}")
        _print_eval_risk_summary(oracle_outputs, "oracle_caption")

        preds_path1 = os.path.join(stage2_dir, _pred_filename("oracle_caption"))
        with open(preds_path1, "w") as f:
            json.dump(_strip_vectors_for_preds(oracle_outputs), f, indent=2)
        logger.info(f"[EVAL2] Wrote {len(oracle_outputs)} oracle preds to {preds_path1}")
        combined_metrics["oracle_caption"] = _finalize_mode(
            "oracle_caption", oracle_metrics, oracle_outputs
        )

    if not args.skip_stage1:
        logger.info("[EVAL2] Running stage1_caption pass...")
        stage1_metrics, stage1_outputs = run_eval("stage1_caption")
        logger.info(f"[EVAL2] stage1_caption metrics: {stage1_metrics}")
        _print_eval_risk_summary(stage1_outputs, "stage1_caption")

        preds_path2 = os.path.join(stage2_dir, _pred_filename("stage1_caption"))
        with open(preds_path2, "w") as f:
            json.dump(_strip_vectors_for_preds(stage1_outputs), f, indent=2)
        logger.info(f"[EVAL2] Wrote {len(stage1_outputs)} stage1 preds to {preds_path2}")
        combined_metrics["stage1_caption"] = _finalize_mode(
            "stage1_caption", stage1_metrics, stage1_outputs
        )

    if not args.skip_risk_masked:
        # Step 5-lite: third eval pass — rebuild oracle-caption prompt with
        # empty risk_text. Same trained model, only the input is lesioned.
        # Exposes prompt-leak: any saturation that drops here was the model
        # cribbing from `Risk level: HIGH` in the input rather than learning.
        logger.info("[EVAL2] Running risk_masked_caption pass...")
        rm_metrics, rm_outputs = run_eval("risk_masked_caption")
        logger.info(f"[EVAL2] risk_masked_caption metrics: {rm_metrics}")
        _print_eval_risk_summary(rm_outputs, "risk_masked_caption")

        preds_path3 = os.path.join(stage2_dir, _pred_filename("risk_masked_caption"))
        with open(preds_path3, "w") as f:
            json.dump(_strip_vectors_for_preds(rm_outputs), f, indent=2)
        logger.info(f"[EVAL2] Wrote {len(rm_outputs)} risk_masked preds to {preds_path3}")
        combined_metrics["risk_masked_caption"] = _finalize_mode(
            "risk_masked_caption", rm_metrics, rm_outputs
        )

    # --- Merge + write eval_metrics.json ---
    # Sharded mode: write shard-local metrics to eval_metrics_shard{i}.json.
    # merge_shards.py will recompute the canonical eval_metrics.json from the
    # concatenated predictions (more accurate than averaging shard metrics).
    if is_sharded:
        metrics_path = os.path.join(
            stage2_dir, f"eval_metrics_shard{args.shard_idx}.json"
        )
        shard_metrics = {
            **combined_metrics,
            "shard_idx": int(args.shard_idx),
            "num_shards": int(args.num_shards),
            "n_shard_samples": len(eval_ds),
        }
        with open(metrics_path, "w") as f:
            json.dump(shard_metrics, f, indent=2)
        logger.info(f"[EVAL2] Wrote shard metrics to {metrics_path}")
        logger.info(
            f"[EVAL2] Shard {args.shard_idx} done. Run "
            f"'python merge_shards.py --run {args.run} --stage 2 "
            f"--num_shards {args.num_shards}' after all shards finish."
        )
    else:
        metrics_path = os.path.join(stage2_dir, "eval_metrics.json")
        existing: Dict = {}
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}
        existing.update(combined_metrics)
        with open(metrics_path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info(f"[EVAL2] Wrote metrics to {metrics_path}")
    logger.info("[EVAL2] Done.")


if __name__ == "__main__":
    main()
