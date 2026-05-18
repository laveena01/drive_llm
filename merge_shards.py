"""
Merge per-shard eval outputs produced by `eval_stage1.py` / `eval_stage2.py`
when run in sharded multi-GPU mode into the canonical single-file outputs.

Stage 1 produces per-shard:
  - runs/<run>/stage1/val_predictions_shard{i}.json
  - runs/<run>/stage1/eval_metrics_shard{i}.json
This script writes:
  - runs/<run>/stage1/val_predictions.json        (concatenated)
  - runs/<run>/stage1/eval_metrics.json           (bleu1/rouge_l recomputed
    from concatenated preds; eval_loss = sample-weighted average)

Stage 2 produces per-shard (for each pass that was run):
  - runs/<run>/stage2/val_predictions_oracle_caption_shard{i}.json
  - runs/<run>/stage2/val_predictions_stage1_caption_shard{i}.json
  - runs/<run>/stage2/eval_metrics_shard{i}.json
This script writes:
  - runs/<run>/stage2/val_predictions_oracle_caption.json   (concatenated)
  - runs/<run>/stage2/val_predictions_stage1_caption.json   (concatenated)
  - runs/<run>/stage2/eval_metrics.json                     (recomputed from
    concatenated outputs using the same aggregators as in-training eval)

Usage:
    python merge_shards.py --run <RUN_ID> --stage 1 --num_shards 3
    python merge_shards.py --run <RUN_ID> --stage 2 --num_shards 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

from llm_driving import config as cfg
from llm_driving.eval_extras import (
    compute_future_summary,
    compute_slice_metrics,
)
from llm_driving.logging_utils import setup_logging
from llm_driving.training import (
    _compute_bleu1_list,
    _compute_rouge_l_list,
    _extract_brake_percent,
    _format_compliance_5line,
    bleu1,
    rouge_l_f1,
)

logger = logging.getLogger("llm_driving")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge per-shard eval outputs")
    p.add_argument("--run", required=True, help="Run ID (folder name under runs/)")
    p.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2],
        help="Which stage to merge (1 or 2).",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        required=True,
        help="Total number of shards used during eval (must match --num_shards "
        "passed to eval_stage*.py).",
    )
    return p.parse_args()


def _read_shard_json(path: str) -> Any:
    if not os.path.isfile(path):
        logger.error(f"Missing shard file: {path}")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


def _interleave(shards: List[List[Dict]]) -> List[Dict]:
    """Reverse of stride-sharding: shards[0] was samples[0::N], shards[1]
    was samples[1::N], etc. Interleaving shards[0][0], shards[1][0], ...,
    shards[0][1], shards[1][1], ... reconstructs the original order."""
    merged: List[Dict] = []
    max_len = max((len(s) for s in shards), default=0)
    for pos in range(max_len):
        for s in shards:
            if pos < len(s):
                merged.append(s[pos])
    return merged


def _merge_stage1(run_dir: str, num_shards: int) -> None:
    stage1_dir = os.path.join(run_dir, "stage1")
    if not os.path.isdir(stage1_dir):
        logger.error(f"Stage 1 dir not found: {stage1_dir}")
        sys.exit(1)

    setup_logging(stage1_dir, "merge_shards.log")
    logger.info(f"[MERGE1] Merging {num_shards} shards under {stage1_dir}")

    # Concatenate predictions
    shard_preds: List[List[Dict]] = []
    for i in range(num_shards):
        path = os.path.join(stage1_dir, f"val_predictions_shard{i}.json")
        shard_preds.append(_read_shard_json(path))
        logger.info(f"[MERGE1] Shard {i}: {len(shard_preds[-1])} preds")
    merged_preds = _interleave(shard_preds)
    logger.info(f"[MERGE1] Merged: {len(merged_preds)} preds total")

    out_preds_path = os.path.join(stage1_dir, "val_predictions.json")
    with open(out_preds_path, "w") as f:
        json.dump(merged_preds, f, indent=2)
    logger.info(f"[MERGE1] Wrote {out_preds_path}")

    # Recompute BLEU-1 and ROUGE-L on the full concatenated set
    preds = [p["prediction"] for p in merged_preds]
    refs = [p["ground_truth"] for p in merged_preds]
    bleu1_score = _compute_bleu1_list(preds, refs)
    rouge_l_score = _compute_rouge_l_list(preds, refs)

    # Sample-weighted average of per-shard eval_loss (falls back gracefully
    # if a shard's metrics file is missing).
    weighted_loss_sum = 0.0
    total_n = 0
    for i in range(num_shards):
        m_path = os.path.join(stage1_dir, f"eval_metrics_shard{i}.json")
        if not os.path.isfile(m_path):
            logger.warning(f"[MERGE1] Missing shard metrics: {m_path}; skipping")
            continue
        with open(m_path, "r") as f:
            sm = json.load(f)
        n = int(sm.get("n_val_samples", 0))
        loss = float(sm.get("eval_loss", 0.0))
        weighted_loss_sum += loss * n
        total_n += n
    avg_loss = (weighted_loss_sum / total_n) if total_n > 0 else 0.0

    metrics_path = os.path.join(stage1_dir, "eval_metrics.json")
    existing: Dict = {}
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                existing = json.load(f) or {}
        except Exception:
            existing = {}
    existing.update(
        {
            "eval_loss": float(avg_loss),
            "bleu1": float(bleu1_score),
            "rouge_l": float(rouge_l_score),
            "n_val_samples": len(merged_preds),
            "merged_from_shards": int(num_shards),
        }
    )
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(
        f"[MERGE1] Wrote metrics: bleu1={bleu1_score:.4f} rouge_l={rouge_l_score:.4f} "
        f"eval_loss={avg_loss:.4f} n={len(merged_preds)} -> {metrics_path}"
    )
    logger.info("[MERGE1] Done.")


def _recompute_stage2_metrics(outputs: List[Dict]) -> Dict:
    """Re-run the aggregator logic from eval_stage2.run_eval over already-
    generated per-sample outputs. Works because every field we need is
    persisted in each output dict."""
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

    # Row 4 v2: per-component risk-decomp MAE counters. Pull straight from
    # the per-sample gt_<comp>_pct / pred_<comp>_pct fields written by
    # eval_stage2.py — same pattern as brake_mae above.
    _RC_COMPS = ("collision", "pedestrian", "uncertainty", "regulatory")
    rc_mae_sum = {c: 0.0 for c in _RC_COMPS}
    rc_n = {c: 0 for c in _RC_COMPS}

    for out in outputs:
        qtype = (out.get("question_type") or "action").strip().lower()
        gt_text = out.get("ground_truth", "")
        pred_fixed = out.get("prediction_fixed", "")
        pred_raw = out.get("prediction_raw", "")
        gt_action = out.get("gt_action", "OTHER")
        pred_action = out.get("pred_action", "OTHER")
        parse_ok = int(out.get("parse_ok", 0))
        risk_level = (out.get("risk_level") or "").strip().upper()

        if qtype == "action":
            if risk_level in ("HIGH", "CRITICAL"):
                highcrit_total += 1
                if pred_action == "CONTINUE":
                    unsafe_continue_high += 1

            if gt_action == "BRAKE":
                gt_brk = _extract_brake_percent(gt_text)
                pr_brk = _extract_brake_percent(pred_fixed)
                if gt_brk is not None and pr_brk is not None:
                    brake_mae_sum_on_brake_gt += abs(float(pr_brk) - float(gt_brk))
                    brake_mae_count_on_brake_gt += 1

            # Row 4 v2: per-component risk-decomp MAE. Pulled from the
            # per-sample fields already on `out` (written by eval_stage2.py).
            for _comp in _RC_COMPS:
                gt_v = out.get(f"gt_{_comp}_pct")
                pr_v = out.get(f"pred_{_comp}_pct")
                if gt_v is None or pr_v is None:
                    continue
                try:
                    rc_mae_sum[_comp] += abs(float(pr_v) - float(gt_v))
                    rc_n[_comp] += 1
                except Exception:
                    pass

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
            # risk_level question — scored separately
            from llm_driving.training import _extract_risk_level

            gt_rl = risk_level or (_extract_risk_level(gt_text) or "")
            pr_rl = _extract_risk_level(pred_raw) or ""
            if gt_rl:
                risk_total += 1
                if pr_rl == gt_rl.upper():
                    risk_correct += 1

    metrics_dict = {
        "action_accuracy": float(correct / total) if total > 0 else 0.0,
        "n_action_samples": int(total),
        "missed_brake_rate": (
            float(missed_brake / brake_total) if brake_total > 0 else 0.0
        ),
        "n_brake_gt": int(brake_total),
        "unsafe_continue_high_rate": (
            float(unsafe_continue_high / highcrit_total)
            if highcrit_total > 0
            else 0.0
        ),
        "n_highcrit_action_samples": int(highcrit_total),
        "brake_mae_on_brake_gt": (
            float(brake_mae_sum_on_brake_gt / brake_mae_count_on_brake_gt)
            if brake_mae_count_on_brake_gt > 0
            else 0.0
        ),
        "n_brake_mae_samples": int(brake_mae_count_on_brake_gt),
        "risk_level_accuracy": (
            float(risk_correct / risk_total) if risk_total > 0 else 0.0
        ),
        "n_risk_samples": int(risk_total),
        "bleu1_action": float(bleu_sum / max(1, total)) if total > 0 else 0.0,
        "rougeL_f1_action": float(rouge_sum / max(1, total)) if total > 0 else 0.0,
        "format_compliance_action": (
            float(fmt_sum / max(1, total)) if total > 0 else 0.0
        ),
        "parse_ok_rate_action": (
            float(parse_ok_sum / max(1, total)) if total > 0 else 0.0
        ),
    }
    # Row 4 v2: append per-component MAE + sample count to top-level metrics.
    for _comp in _RC_COMPS:
        n = rc_n[_comp]
        metrics_dict[f"{_comp}_risk_mae"] = (
            float(rc_mae_sum[_comp] / n) if n > 0 else 0.0
        )
        metrics_dict[f"n_{_comp}_risk_samples"] = int(n)
    return metrics_dict


def _merge_stage2(run_dir: str, num_shards: int) -> None:
    stage2_dir = os.path.join(run_dir, "stage2")
    if not os.path.isdir(stage2_dir):
        logger.error(f"Stage 2 dir not found: {stage2_dir}")
        sys.exit(1)

    setup_logging(stage2_dir, "merge_shards.log")
    logger.info(f"[MERGE2] Merging {num_shards} shards under {stage2_dir}")

    combined_metrics: Dict = {}
    for mode in ("oracle_caption", "stage1_caption", "risk_masked_caption"):
        # Detect which modes actually ran — if shard 0 doesn't have the file,
        # assume the whole mode was skipped.
        shard0 = os.path.join(stage2_dir, f"val_predictions_{mode}_shard0.json")
        if not os.path.isfile(shard0):
            logger.info(f"[MERGE2] No shards found for '{mode}' — skipping")
            continue

        shard_outputs: List[List[Dict]] = []
        for i in range(num_shards):
            path = os.path.join(stage2_dir, f"val_predictions_{mode}_shard{i}.json")
            shard_outputs.append(_read_shard_json(path))
            logger.info(f"[MERGE2] {mode} shard {i}: {len(shard_outputs[-1])} outputs")
        merged_outputs = _interleave(shard_outputs)
        logger.info(f"[MERGE2] {mode} merged: {len(merged_outputs)} outputs total")

        out_path = os.path.join(stage2_dir, f"val_predictions_{mode}.json")
        with open(out_path, "w") as f:
            json.dump(merged_outputs, f, indent=2)
        logger.info(f"[MERGE2] Wrote {out_path}")

        metrics = _recompute_stage2_metrics(merged_outputs)
        # E1+E2: recompute future-aware summary and stratified slice metrics
        # over the concatenated predictions. The per-sample fields needed
        # (risk_level_future, brake_required_future, risk_transition,
        # density_bucket, gt_brake_pct, pred_brake_pct) are persisted per
        # shard by eval_stage2.py.
        future_summary = compute_future_summary(merged_outputs)
        slice_metrics = compute_slice_metrics(merged_outputs)
        metrics = {**metrics, **future_summary, "slices": slice_metrics}
        logger.info(
            f"[MERGE2] {mode} metrics (with future + slices): "
            f"action_acc={metrics.get('action_accuracy'):.4f} "
            f"missed_brake={metrics.get('missed_brake_rate'):.4f} "
            f"future_brake_recall={metrics.get('future_brake_recall'):.4f}"
        )
        combined_metrics[mode] = metrics

        # E3: concat per-shard hard_cases_<mode>_shard{i}.json into a
        # canonical hard_cases_<mode>.json. Per-shard files were emitted
        # by eval_stage2._finalize_mode. If any are missing (e.g. an older
        # eval run), skip silently.
        hc_records: List[Dict] = []
        any_missing = False
        for i in range(num_shards):
            hc_path_i = os.path.join(
                stage2_dir, f"hard_cases_{mode}_shard{i}.json"
            )
            if not os.path.isfile(hc_path_i):
                any_missing = True
                logger.warning(f"[MERGE2] Missing hard cases shard: {hc_path_i}")
                continue
            with open(hc_path_i, "r") as f:
                shard_hc = json.load(f) or []
            hc_records.extend(shard_hc)
        if hc_records or not any_missing:
            hc_out = os.path.join(stage2_dir, f"hard_cases_{mode}.json")
            with open(hc_out, "w") as f:
                json.dump(hc_records, f, indent=2)
            logger.info(
                f"[MERGE2] {mode}: wrote {len(hc_records)} merged hard cases "
                f"to {hc_out}"
            )

    metrics_path = os.path.join(stage2_dir, "eval_metrics.json")
    existing: Dict = {}
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                existing = json.load(f) or {}
        except Exception:
            existing = {}
    existing.update(combined_metrics)
    existing["merged_from_shards"] = int(num_shards)
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"[MERGE2] Wrote merged metrics to {metrics_path}")
    logger.info("[MERGE2] Done.")


def main() -> None:
    args = _parse_args()
    if args.num_shards < 2:
        logging.basicConfig(level=logging.INFO)
        logger.error(
            f"--num_shards must be >= 2 for merging; got {args.num_shards}"
        )
        sys.exit(1)

    run_dir = os.path.join(cfg.RUNS_DIR, args.run)
    if not os.path.isdir(run_dir):
        logging.basicConfig(level=logging.INFO)
        logger.error(f"Run dir not found: {run_dir}")
        sys.exit(1)

    if args.stage == 1:
        _merge_stage1(run_dir, args.num_shards)
    else:
        _merge_stage2(run_dir, args.num_shards)


if __name__ == "__main__":
    main()
