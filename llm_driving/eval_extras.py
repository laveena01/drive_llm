# llm_driving/eval_extras.py

"""
Eval-time helpers for stratified slice metrics, future-aware oracle labels,
and hard-case dumps.

All operations here are pure functions over QA-sample / per-prediction dicts.
No model inference, no nuScenes access, no I/O — safe to call from training,
eval scripts, and merge_shards alike.

Three concepts:

1. Future-aware labels (E1)
   For each frame `t` in scene `s`, look ahead to frames `t .. t+H` (same scene)
   and aggregate:
     - risk_level_future:    max severity over the window
     - brake_required_future: True if any frame in window has policy_label == "BRAKE"
   These let the eval ask "did the model brake when it *was about to be needed*?",
   which the per-frame oracle alone cannot ask.

2. Stratified slices (E2)
   Slice action samples by:
     - risk_level (current frame)        -- exposes ceiling effects
     - risk_transition (rising/falling)  -- highlights temporal-interesting cases
     - density_bucket                    -- highlights multi-object configurations
     - risk_level_future                 -- pairs with E1 above
   Per slice we report n / action_accuracy / missed_brake_rate / brake_mae.

3. Hard cases (E3)
   The cases that actually matter for safety: missed brakes and unsafe-continues
   on HIGH/CRITICAL frames. Extract a bounded number with full context (vectors,
   risk components, captions) for qualitative analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("llm_driving")

# Severity ordering; higher = worse.
RISK_LEVEL_ORDER: Dict[str, int] = {
    "MINIMAL": 0,
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def _risk_severity(level: Optional[str]) -> int:
    if not level:
        return -1
    return RISK_LEVEL_ORDER.get(str(level).strip().upper(), -1)


def _max_risk(levels: List[str]) -> str:
    best_level = ""
    best_sev = -1
    for lvl in levels:
        s = _risk_severity(lvl)
        if s > best_sev:
            best_sev = s
            best_level = (lvl or "").strip().upper()
    return best_level or "MINIMAL"


# -----------------------------------------------------------------------------
# E1: future-aware labels
# -----------------------------------------------------------------------------

def compute_future_labels(
    qa_samples: List[Dict],
    horizon: int = 4,
) -> List[Dict]:
    """
    For each QA sample, attach:
      - risk_level_future:     max severity over [t .. t+H] in the same scene
      - brake_required_future: True if any frame in [t .. t+H] has
                                policy_label == "BRAKE"

    Operates in-place; also returns the same list for chaining.

    Frames are looked up by (scene_idx, frame_in_scene). Multiple QA samples
    share a frame (5 action + 3 risk per frame in the current builder); we
    deduplicate to one entry per frame before computing the lookahead.
    """
    frame_map: Dict[Tuple[int, int], Dict] = {}
    for s in qa_samples:
        try:
            scene = int(s.get("scene_idx", -1))
            frame = int(s.get("frame_in_scene", -1))
        except Exception:
            continue
        if scene < 0 or frame < 0:
            continue
        key = (scene, frame)
        if key not in frame_map:
            frame_map[key] = {
                "risk_level": str(s.get("risk_level") or "").strip().upper(),
                "policy_label": str(s.get("policy_label") or "").strip().upper(),
            }

    n_with_future = 0
    for s in qa_samples:
        try:
            scene = int(s.get("scene_idx", -1))
            frame = int(s.get("frame_in_scene", -1))
        except Exception:
            scene, frame = -1, -1

        if scene < 0 or frame < 0:
            # No scene/frame metadata — fall back to current-frame labels.
            s["risk_level_future"] = (
                str(s.get("risk_level") or "").strip().upper() or "MINIMAL"
            )
            s["brake_required_future"] = (
                str(s.get("policy_label") or "").strip().upper() == "BRAKE"
            )
            continue

        levels: List[str] = []
        any_brake = False
        for k in range(0, horizon + 1):
            entry = frame_map.get((scene, frame + k))
            if entry is None:
                continue
            levels.append(entry["risk_level"])
            if entry["policy_label"] == "BRAKE":
                any_brake = True

        s["risk_level_future"] = (
            _max_risk(levels)
            if levels
            else (str(s.get("risk_level") or "").strip().upper() or "MINIMAL")
        )
        s["brake_required_future"] = bool(any_brake)
        n_with_future += 1

    logger.info(
        "[eval_extras] compute_future_labels: horizon=%d, qa_samples=%d, "
        "frames=%d, labelled=%d",
        horizon, len(qa_samples), len(frame_map), n_with_future,
    )
    return qa_samples


# -----------------------------------------------------------------------------
# E2: risk transitions and density bucketing
# -----------------------------------------------------------------------------

def compute_risk_transitions(qa_samples: List[Dict]) -> List[Dict]:
    """
    Annotate each QA sample with `risk_transition` relative to the previous
    frame in the same scene:
      "rising"  : current severity > previous
      "falling" : current severity < previous
      "stable"  : equal severities
      "unknown" : no previous frame (scene start, or missing metadata)
    """
    frame_map: Dict[Tuple[int, int], str] = {}
    for s in qa_samples:
        try:
            scene = int(s.get("scene_idx", -1))
            frame = int(s.get("frame_in_scene", -1))
        except Exception:
            continue
        if scene < 0 or frame < 0:
            continue
        key = (scene, frame)
        if key not in frame_map:
            frame_map[key] = str(s.get("risk_level") or "").strip().upper()

    for s in qa_samples:
        try:
            scene = int(s.get("scene_idx", -1))
            frame = int(s.get("frame_in_scene", -1))
        except Exception:
            scene, frame = -1, -1

        if scene < 0 or frame < 0 or frame == 0:
            s["risk_transition"] = "unknown"
            continue

        prev = frame_map.get((scene, frame - 1))
        cur = str(s.get("risk_level") or "").strip().upper()
        ps, cs = _risk_severity(prev), _risk_severity(cur)
        if prev is None or ps < 0 or cs < 0:
            s["risk_transition"] = "unknown"
        elif cs > ps:
            s["risk_transition"] = "rising"
        elif cs < ps:
            s["risk_transition"] = "falling"
        else:
            s["risk_transition"] = "stable"
    return qa_samples


def density_bucket(num_objects: int) -> str:
    try:
        n = int(num_objects or 0)
    except Exception:
        n = 0
    if n <= 0:
        return "empty"
    if n <= 2:
        return "sparse"
    if n <= 5:
        return "moderate"
    return "dense"


def annotate_density(qa_samples: List[Dict]) -> List[Dict]:
    """Attach `density_bucket` to each sample based on `num_objects`."""
    for s in qa_samples:
        s["density_bucket"] = density_bucket(s.get("num_objects", 0))
    return qa_samples


def enrich_qa_samples(qa_samples: List[Dict], horizon: int = 4) -> List[Dict]:
    """
    Convenience wrapper: future labels + risk transitions + density buckets,
    in the order they should be applied (future and transition both need
    deduped frame maps; density is independent).
    """
    compute_future_labels(qa_samples, horizon=horizon)
    compute_risk_transitions(qa_samples)
    annotate_density(qa_samples)
    return qa_samples


# -----------------------------------------------------------------------------
# E2: slice metrics
# -----------------------------------------------------------------------------

def _slice_metrics_for_outputs(outputs: List[Dict]) -> Dict:
    """
    Compute aggregate action metrics over a list of per-sample output dicts.
    Only `question_type == "action"` rows contribute. Outputs missing the
    question_type field are treated as actions for backward compatibility.
    """
    correct = 0
    total = 0
    missed_brake = 0
    brake_total = 0
    brake_mae_sum = 0.0
    brake_mae_n = 0
    future_brake_total = 0
    future_brake_correct = 0  # predicted BRAKE when future-required
    future_missed = 0

    # Step 5-lite follow-up: tolerance-based brake metrics. With continuous
    # brake targets, raw MAE understates real-world adequacy — an error of
    # ≤5pp is practically negligible, ≤10pp is acceptable, ≤20pp is
    # tolerable. Report all three alongside the strict MAE.
    brake_within_5 = 0
    brake_within_10 = 0
    brake_within_20 = 0

    for out in outputs:
        if str(out.get("question_type") or "action").strip().lower() != "action":
            continue
        gt_action = out.get("gt_action", "OTHER")
        pred_action = out.get("pred_action", "OTHER")

        if gt_action != "OTHER":
            total += 1
            if gt_action == pred_action:
                correct += 1
            if gt_action == "BRAKE":
                brake_total += 1
                if pred_action != "BRAKE":
                    missed_brake += 1

        gt_brk = out.get("gt_brake_pct")
        pr_brk = out.get("pred_brake_pct")
        if gt_action == "BRAKE" and gt_brk is not None and pr_brk is not None:
            try:
                diff = abs(float(pr_brk) - float(gt_brk))
                brake_mae_sum += diff
                brake_mae_n += 1
                if diff <= 5:
                    brake_within_5 += 1
                if diff <= 10:
                    brake_within_10 += 1
                if diff <= 20:
                    brake_within_20 += 1
            except Exception:
                pass

        if bool(out.get("brake_required_future", False)):
            future_brake_total += 1
            if pred_action == "BRAKE":
                future_brake_correct += 1
            else:
                future_missed += 1

    return {
        "n": int(total),
        "action_accuracy": float(correct / total) if total > 0 else 0.0,
        "missed_brake_rate": (
            float(missed_brake / brake_total) if brake_total > 0 else 0.0
        ),
        "n_brake_gt": int(brake_total),
        "brake_mae_on_brake_gt": (
            float(brake_mae_sum / brake_mae_n) if brake_mae_n > 0 else 0.0
        ),
        "n_brake_mae_samples": int(brake_mae_n),
        "brake_within_5pct": (
            float(brake_within_5 / brake_mae_n) if brake_mae_n > 0 else 0.0
        ),
        "brake_within_10pct": (
            float(brake_within_10 / brake_mae_n) if brake_mae_n > 0 else 0.0
        ),
        "brake_within_20pct": (
            float(brake_within_20 / brake_mae_n) if brake_mae_n > 0 else 0.0
        ),
        "future_brake_recall": (
            float(future_brake_correct / future_brake_total)
            if future_brake_total > 0
            else 0.0
        ),
        "n_future_brake_gt": int(future_brake_total),
        "future_missed_brake_rate": (
            float(future_missed / future_brake_total)
            if future_brake_total > 0
            else 0.0
        ),
    }


def compute_slice_metrics(outputs: List[Dict]) -> Dict:
    """
    Aggregate slice metrics across four slice families:
      - by_risk_level         (current frame)
      - by_risk_transition    (rising/falling/stable/unknown)
      - by_density            (empty/sparse/moderate/dense)
      - by_risk_level_future  (max severity over t..t+H)

    Returns a dict-of-dicts keyed by slice family then slice value.
    """
    by_risk: Dict[str, List[Dict]] = {}
    by_trans: Dict[str, List[Dict]] = {}
    by_density: Dict[str, List[Dict]] = {}
    by_future: Dict[str, List[Dict]] = {}

    for o in outputs:
        rl = (str(o.get("risk_level") or "")).strip().upper() or "UNKNOWN"
        tr = (str(o.get("risk_transition") or "unknown")).lower()
        db = (str(o.get("density_bucket") or "unknown")).lower()
        rf = (str(o.get("risk_level_future") or "")).strip().upper() or "UNKNOWN"
        by_risk.setdefault(rl, []).append(o)
        by_trans.setdefault(tr, []).append(o)
        by_density.setdefault(db, []).append(o)
        by_future.setdefault(rf, []).append(o)

    return {
        "by_risk_level": {k: _slice_metrics_for_outputs(v) for k, v in by_risk.items()},
        "by_risk_transition": {
            k: _slice_metrics_for_outputs(v) for k, v in by_trans.items()
        },
        "by_density": {
            k: _slice_metrics_for_outputs(v) for k, v in by_density.items()
        },
        "by_risk_level_future": {
            k: _slice_metrics_for_outputs(v) for k, v in by_future.items()
        },
    }


def compute_future_summary(outputs: List[Dict]) -> Dict:
    """
    Top-level future-aware metrics aggregated over all action outputs.
    Pulled out separately so it lands in `eval_metrics.json` alongside the
    existing aggregate keys (rather than nested under slices).
    """
    future_brake_total = 0
    future_brake_correct = 0
    future_missed = 0
    n_action = 0
    for o in outputs:
        if str(o.get("question_type") or "action").strip().lower() != "action":
            continue
        n_action += 1
        if bool(o.get("brake_required_future", False)):
            future_brake_total += 1
            if o.get("pred_action") == "BRAKE":
                future_brake_correct += 1
            else:
                future_missed += 1
    return {
        "future_brake_recall": (
            float(future_brake_correct / future_brake_total)
            if future_brake_total > 0
            else 0.0
        ),
        "n_future_brake_gt": int(future_brake_total),
        "future_missed_brake_rate": (
            float(future_missed / future_brake_total)
            if future_brake_total > 0
            else 0.0
        ),
        "n_action_samples_future_eval": int(n_action),
    }


# -----------------------------------------------------------------------------
# E3: hard cases
# -----------------------------------------------------------------------------

def _hard_case_record(o: Dict, kind: str) -> Dict:
    return {
        "kind": kind,
        "mode": o.get("mode"),
        "scene_idx": o.get("scene_idx"),
        "frame_in_scene": o.get("frame_in_scene"),
        "frame_idx": o.get("frame_idx"),
        "num_objects": o.get("num_objects"),
        "density_bucket": o.get("density_bucket"),
        "risk_level": o.get("risk_level"),
        "risk_level_future": o.get("risk_level_future"),
        "brake_required_future": o.get("brake_required_future"),
        "risk_transition": o.get("risk_transition"),
        "gt_action": o.get("gt_action"),
        "pred_action": o.get("pred_action"),
        "gt_brake_pct": o.get("gt_brake_pct"),
        "pred_brake_pct": o.get("pred_brake_pct"),
        "ground_truth": o.get("ground_truth"),
        "prediction_fixed": o.get("prediction_fixed"),
        "caption_used": o.get("caption_used"),
        "vectors": o.get("vectors"),
    }


def extract_hard_cases(
    outputs: List[Dict],
    max_per_kind: int = 200,
) -> List[Dict]:
    """
    Returns up to `max_per_kind` records of each kind:
      - missed_brake:    gt_action == BRAKE, pred_action != BRAKE
      - unsafe_continue: risk_level in HIGH/CRITICAL, pred_action == CONTINUE

    A single output may qualify for both kinds; in that case both records are
    emitted (they describe distinct failure modes).
    """
    missed: List[Dict] = []
    unsafe: List[Dict] = []
    for o in outputs:
        if str(o.get("question_type") or "action").strip().lower() != "action":
            continue
        gt_action = o.get("gt_action", "OTHER")
        pred_action = o.get("pred_action", "OTHER")
        risk_level = (str(o.get("risk_level") or "")).strip().upper()

        if (
            gt_action == "BRAKE"
            and pred_action != "BRAKE"
            and len(missed) < max_per_kind
        ):
            missed.append(_hard_case_record(o, kind="missed_brake"))
        if (
            risk_level in ("HIGH", "CRITICAL")
            and pred_action == "CONTINUE"
            and len(unsafe) < max_per_kind
        ):
            unsafe.append(_hard_case_record(o, kind="unsafe_continue"))
    return missed + unsafe
