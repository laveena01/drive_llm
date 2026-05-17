# llm_driving/risk_calculator.py

"""
Phase-2 Risk Module (thesis-friendly + driving-realistic)

This file keeps the SAME public API used by your pipeline:
  - calculate_risk_from_vectors(...)
  - get_risk_summary_text(...)
  - policy_from_risk(...)

But upgrades the internals to be more realistic and more explainable:

1) Soft front/path weighting (front gating, not hard discard)
2) Correct closing-speed TTC (TTC only if approaching)
3) Distance baseline risk (static close objects still risky)
4) Lateral conflict risk (near-path / cut-in style)
5) Lightweight uncertainty proxy risk (distance + small size + speed)
6) Stable scene aggregation using log-sum-exp ("softmax" over objects)

Outputs remain compatible with Phase-1:
FrameRiskData has: risk_level, max_collision_risk, max_pedestrian_risk, min_ttc, avg_total_risk, num_risk_objects, object_risks
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
import logging
import math

import numpy as np

from . import config as cfg

from .config import (
    DEFAULT_EGO_SPEED,
    RISK_FRONT_CONE_DEG,
    RISK_LATERAL_BAND_M,
    RISK_REQUIRE_IN_FRONT,
)

logger = logging.getLogger("llm_driving")


# =============================================================================
# TYPE ID MAPPING (matches nuscenes_data.py)
# =============================================================================

TYPE_ID_TO_CATEGORY = {
    0: "vehicle.car",
    1: "human.pedestrian.adult",
    2: "traffic_light",
    3: "object.other",
}

# More vulnerable road users get higher weight.
TYPE_WEIGHT = {
    "human.pedestrian.adult": 2.5,
    "human.pedestrian.child": 3.0,
    "vehicle.bicycle": 2.0,
    "vehicle.motorcycle": 1.8,
    "vehicle.car": 1.0,
    "vehicle.truck": 1.3,
    "vehicle.bus": 1.3,
    "vehicle.emergency": 1.5,
    "traffic_light": 0.2,
    "object.other": 0.8,
}


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class FrameRiskData:
    """Risk data for an entire frame."""
    risk_level: str
    max_collision_risk: float
    max_pedestrian_risk: float
    min_ttc: Optional[float]
    avg_total_risk: float
    num_risk_objects: int
    object_risks: List[Dict]

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# HELPERS
# =============================================================================

def _sigmoid(x: float) -> float:
    # safe sigmoid
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _logsumexp(xs: List[float], lam: float) -> float:
    """
    Stable log-sum-exp aggregator. Returns:
        (1/lam) * log(sum(exp(lam * x_i)))
    When lam is large, approaches max(x_i) but smoother.
    """
    if not xs:
        return 0.0
    m = max(xs)
    # if all -inf, return 0
    if not math.isfinite(m):
        return 0.0
    s = 0.0
    for x in xs:
        s += math.exp(lam * (x - m))
    return (m + (math.log(max(s, 1e-12)) / lam))


def _type_weight(obj_type: str) -> float:
    if obj_type in TYPE_WEIGHT:
        return float(TYPE_WEIGHT[obj_type])
    # fuzzy match
    lo = (obj_type or "").lower()
    if "pedestrian" in lo:
        return 2.5
    if "bicycle" in lo:
        return 2.0
    if "motorcycle" in lo:
        return 1.8
    if "truck" in lo or "bus" in lo:
        return 1.3
    return 1.0


def _front_weight(rel_x: float, rel_y: float, dist: float) -> float:
    """
    Soft front/path weight in [0,1].

    - Uses angle vs RISK_FRONT_CONE_DEG as a soft gate (not hard).
    - Optionally suppresses behind objects when RISK_REQUIRE_IN_FRONT is True.
    """
    if dist <= 1e-6:
        return 1.0

    # cos(theta) = rel_x / dist  (1.0 is directly ahead)
    cos_th = max(-1.0, min(1.0, rel_x / dist))

    # convert "front cone degrees" to a cosine threshold
    # if cone=60°, cos=0.5. If cone=30°, cos=0.866.
    cone = float(RISK_FRONT_CONE_DEG)
    cone = max(5.0, min(85.0, cone))
    cos_thr = math.cos(math.radians(cone))

    # soft transition around cos_thr
    # scale decides softness; 0.08~0.15 is reasonable
    w = _sigmoid((cos_th - cos_thr) / 0.10)

    if RISK_REQUIRE_IN_FRONT and rel_x <= 0.0:
        w *= 0.1  # still keep tiny weight (rear object rarely relevant)

    return float(max(0.0, min(1.0, w)))


def _lateral_conflict(rel_x: float, rel_y: float) -> float:
    """
    Lateral conflict: higher when |y| is near the lane/path band and x is ahead.
    Produces a smooth risk in [0,1].
    """
    y = abs(float(rel_y))
    # if within band -> high; beyond band -> decays
    band = float(RISK_LATERAL_BAND_M)
    band = max(0.5, band)

    # exp decay outside the band
    lat = math.exp(-max(0.0, (y - band)) / (band + 1e-6))

    # suppress if far behind
    if rel_x < -2.0:
        lat *= 0.2
    return float(max(0.0, min(1.0, lat)))


def _uncertainty_proxy(dist: float, size: float, speed: float) -> float:
    """
    Lightweight uncertainty proxy (no sensor covariance available):
    - farther objects -> more uncertain
    - very small size -> more uncertain
    - higher speed magnitude -> more uncertain
    """
    d_term = min(1.0, (dist / 50.0)) * 0.5
    # size proxy: small objects are harder (size is coarse in your vectors)
    s_term = 0.0
    if size <= 0.6:
        s_term = 0.35
    elif size <= 1.0:
        s_term = 0.20
    elif size <= 1.5:
        s_term = 0.10

    v_term = min(0.3, speed / 30.0)
    return float(max(0.0, min(1.0, d_term + s_term + v_term)))


def _closing_speed_and_ttc(rel_x: float, rel_y: float, dist: float, rel_vx: float, rel_vy: float) -> Tuple[float, Optional[float]]:
    """
    Compute closing speed along line-of-sight and TTC.

    In ego frame, rel_v is object's velocity relative to ego.
    closing_speed = - dot(rel_v, unit_pos)
      >0  -> approaching
      <=0 -> not approaching (TTC = inf)
    """
    if dist <= 1e-6:
        return 0.0, 0.01
    ux = rel_x / dist
    uy = rel_y / dist
    closing = - (rel_vx * ux + rel_vy * uy)
    if closing <= 1e-3:
        return float(closing), None
    ttc = dist / (closing + 1e-6)
    return float(closing), float(max(0.01, ttc))


# =============================================================================
# CORE: RISK CALCULATION FOR FRAME VECTORS
# =============================================================================

def calculate_risk_from_vectors(
    vectors: np.ndarray,
    use_n: Optional[int] = None,
    ego_speed: Optional[float] = None,
    traffic_light: Optional[str] = None,
) -> FrameRiskData:
    """
    Vector format (Option A, VECTOR_DIM=8):
      [rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id]
    """
    try:
        arr = np.asarray(vectors)
    except Exception:
        logger.exception("[risk_calculator] vectors could not be converted to ndarray.")
        arr = vectors

    if arr is None or len(arr) == 0:
        return FrameRiskData(
            risk_level="MINIMAL",
            max_collision_risk=0.0,
            max_pedestrian_risk=0.0,
            min_ttc=None,
            avg_total_risk=0.0,
            num_risk_objects=0,
            object_risks=[],
        )

    if ego_speed is None:
        ego_speed = float(DEFAULT_EGO_SPEED)

    if use_n is None:
        use_n = int(len(arr))
    use_n = max(0, min(int(use_n), int(len(arr))))

    if use_n == 0:
        return FrameRiskData(
            risk_level="MINIMAL",
            max_collision_risk=0.0,
            max_pedestrian_risk=0.0,
            min_ttc=None,
            avg_total_risk=0.0,
            num_risk_objects=0,
            object_risks=[],
        )

    # Hyperparameters (tunable, but keep fixed for Phase-2 baseline)
    TTC_T = 3.0      # seconds (risk decays with TTC)
    DIST_D = 18.0    # meters (risk decays with distance)
    LAM = 10.0       # logsumexp sharpness (10 ~ fairly max-like)

    object_risks: List[Dict] = []
    per_obj_total: List[float] = []

    max_collision = 0.0
    max_pedestrian = 0.0
    min_ttc = float("inf")

    for i in range(use_n):
        vec = arr[i]
        try:
            rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id = vec
        except Exception:
            continue

        rel_x = float(rel_x)
        rel_y = float(rel_y)

        # dist in vectors can be slightly inconsistent; recompute for safety
        try:
            dist_f = float(dist)
            if not math.isfinite(dist_f):
                dist_f = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        except Exception:
            dist_f = math.sqrt(rel_x * rel_x + rel_y * rel_y)

        # velocity magnitude
        try:
            rvx = float(rel_vx)
            rvy = float(rel_vy)
        except Exception:
            rvx, rvy = 0.0, 0.0
        speed = math.sqrt(rvx * rvx + rvy * rvy)

        try:
            size_f = float(size)
            if not math.isfinite(size_f):
                size_f = 1.0
        except Exception:
            size_f = 1.0

        try:
            obj_type = TYPE_ID_TO_CATEGORY.get(int(type_id), "object.other")
        except Exception:
            obj_type = "object.other"

        tw = _type_weight(obj_type)
        fw = _front_weight(rel_x, rel_y, dist_f)
        lat = _lateral_conflict(rel_x, rel_y)

        closing_speed, ttc = _closing_speed_and_ttc(rel_x, rel_y, dist_f, rvx, rvy)

        # TTC risk: only if approaching
        if ttc is None:
            ttc_risk = 0.0
        else:
            ttc_risk = math.exp(-ttc / TTC_T)
            ttc_risk = float(max(0.0, min(1.0, ttc_risk)))

        # Distance baseline risk (always applicable)
        dist_risk = math.exp(-dist_f / DIST_D)
        dist_risk = float(max(0.0, min(1.0, dist_risk)))

        # Uncertainty proxy
        unc_risk = _uncertainty_proxy(dist_f, size_f, speed)

        # Collision risk combines TTC + distance + lateral, and is weighted by front/path
        # weights sum to 1.0 inside parentheses
        collision_raw = (0.55 * ttc_risk) + (0.25 * dist_risk) + (0.20 * lat)
        collision = fw * collision_raw

        # Pedestrian risk: emphasize vulnerable road users; still front weighted,
        # but keep some risk even when slightly off-path (use sqrt(fw)).
        ped = 0.0
        if "pedestrian" in obj_type or "bicycle" in obj_type or "motorcycle" in obj_type:
            ped = (math.sqrt(max(fw, 0.0)) * (0.60 * collision_raw + 0.40 * dist_risk))
            ped = min(1.0, ped * min(1.5, tw / 1.5))

        # Type-weighted collision (cap to 1.0)
        collision = min(1.0, collision * min(1.6, (tw / 1.2)))
        ped = float(max(0.0, min(1.0, ped)))

        # Regulatory risk (kept minimal; you can expand later)
        reg = 0.0
        if traffic_light == "red":
            # only meaningful if we are moving forward and object is ahead (rough proxy)
            reg = 0.6 if float(ego_speed) > 1.0 else 0.3
        elif traffic_light == "yellow":
            reg = 0.25

        # Total per-object risk (keep weights similar to Phase-1 expectations)
        total = (0.40 * collision) + (0.30 * ped) + (0.20 * unc_risk) + (0.10 * reg)
        total = float(max(0.0, min(1.0, total)))

        # Risk level per object (for debugging/analysis)
        if total >= 0.8:
            obj_level = "CRITICAL"
        elif total >= 0.6:
            obj_level = "HIGH"
        elif total >= 0.4:
            obj_level = "MODERATE"
        elif total >= 0.2:
            obj_level = "LOW"
        else:
            obj_level = "MINIMAL"

        # Track global stats
        max_collision = max(max_collision, float(collision))
        max_pedestrian = max(max_pedestrian, float(ped))
        per_obj_total.append(total)

        if ttc is not None and fw >= 0.3:
            min_ttc = min(min_ttc, float(ttc))

        object_risks.append({
            "idx": i,
            "type": obj_type,
            "distance": round(dist_f, 3),
            "closing_speed": round(float(closing_speed), 3),
            "ttc": round(float(ttc), 3) if ttc is not None else None,
            "front_weight": round(float(fw), 3),
            "lateral_conflict": round(float(lat), 3),
            "risk": {
                "collision_risk": round(float(collision), 3),
                "pedestrian_risk": round(float(ped), 3),
                "dist_risk": round(float(dist_risk), 3),
                "ttc_risk": round(float(ttc_risk), 3),
                "uncertainty_risk": round(float(unc_risk), 3),
                "regulatory_risk": round(float(reg), 3),
                "total_risk": round(float(total), 3),
                "risk_level": obj_level,
            }
        })

    effective_n = len(object_risks)
    if effective_n == 0:
        return FrameRiskData(
            risk_level="MINIMAL",
            max_collision_risk=0.0,
            max_pedestrian_risk=0.0,
            min_ttc=None,
            avg_total_risk=0.0,
            num_risk_objects=0,
            object_risks=[],
        )

    # Scene aggregation (soft max): lets the most dangerous object dominate but remains stable.
    scene_total = float(_logsumexp(per_obj_total, lam=LAM))
    avg_risk = float(sum(per_obj_total) / max(1, len(per_obj_total)))

    # Frame-level risk level from aggregated score, with TTC override
    if (min_ttc < float("inf") and min_ttc < 2.0) or scene_total >= 0.80:
        risk_level = "CRITICAL"
    elif (min_ttc < float("inf") and min_ttc < 3.0) or scene_total >= 0.60:
        risk_level = "HIGH"
    elif (min_ttc < float("inf") and min_ttc < 5.0) or scene_total >= 0.40:
        risk_level = "MODERATE"
    elif scene_total >= 0.20:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"

    return FrameRiskData(
        risk_level=risk_level,
        max_collision_risk=round(float(max_collision), 3),
        max_pedestrian_risk=round(float(max_pedestrian), 3),
        min_ttc=round(float(min_ttc), 2) if min_ttc < float("inf") else None,
        avg_total_risk=round(float(avg_risk), 3),
        num_risk_objects=int(effective_n),
        object_risks=object_risks,
    )


# =============================================================================
# RISK-BASED POLICY DETERMINATION
# =============================================================================

def policy_from_risk(risk_data: FrameRiskData) -> Tuple[int, int, str, str, str]:
    """
    Determine driving policy based on risk assessment.
    Returns: (accel, brake, steer, reason, policy_label)

    When cfg.USE_CONTINUOUS_ACTIONS is True, the discrete bucketed
    (accel, brake) values produced below are OVERRIDDEN at the end with
    smooth integers derived from a composite risk score. policy_label
    and steer remain discrete (used by classification metrics).
    """
    steer = "straight"
    rl = risk_data.risk_level
    min_ttc = risk_data.min_ttc
    max_collision = float(risk_data.max_collision_risk)
    max_ped = float(risk_data.max_pedestrian_risk)

    if risk_data.num_risk_objects == 0:
        accel, brake, reason, label = 20, 0, "No nearby obstacles detected.", "CONTINUE"
    elif rl == "CRITICAL":
        if min_ttc is not None and min_ttc < 2.0:
            accel, brake, reason, label = 0, 90, f"Critical risk: TTC={min_ttc:.1f}s, emergency braking required.", "BRAKE"
        else:
            accel, brake, reason, label = 0, 80, f"Critical risk (collision={max_collision:.0%}), braking hard.", "BRAKE"
    elif rl == "HIGH":
        if max_ped >= 0.5:
            accel, brake, reason, label = 0, 60, f"High pedestrian risk ({max_ped:.0%}), reducing speed.", "BRAKE"
        else:
            accel, brake, reason, label = 0, 50, f"High risk (collision={max_collision:.0%}), slowing down.", "BRAKE"
    elif rl == "MODERATE":
        if min_ttc is not None and min_ttc < 4.0:
            accel, brake, reason, label = 0, 40, f"Moderate risk with low TTC={min_ttc:.1f}s, braking to increase safety margin.", "BRAKE"
        elif max_ped >= 0.3:
            accel, brake, reason, label = 5, 20, "Moderate pedestrian risk, proceed with caution.", "CAUTION"
        else:
            accel, brake, reason, label = 10, 20, f"Moderate risk (collision={max_collision:.0%}), proceed carefully.", "CAUTION"
    elif rl == "LOW":
        accel, brake, reason, label = 15, 0, "Low risk detected, maintaining awareness.", "CONTINUE"
    else:
        accel, brake, reason, label = 20, 0, "Minimal risk, safe to continue.", "CONTINUE"

    # Step 5-lite follow-up: continuous-action override.
    # Replaces bucketed (accel, brake) with smooth integers from a
    # composite risk score. policy_label and steer stay discrete.
    if getattr(cfg, "USE_CONTINUOUS_ACTIONS", False):
        composite_risk = max(max_collision, max_ped)
        max_brake = int(getattr(cfg, "CONTINUOUS_BRAKE_MAX", 90))
        max_accel = int(getattr(cfg, "CONTINUOUS_ACCEL_MAX", 20))
        brake = max(0, min(max_brake, int(round(max_brake * composite_risk))))
        accel = max(0, min(max_accel, int(round(max_accel * (1.0 - composite_risk)))))

    return accel, brake, steer, reason, label


def compute_future_aware_action(
    risk_data_window: List[FrameRiskData],
    horizon_seconds_per_step: float = 0.5,
) -> Tuple[int, int, str, str, str]:
    """
    Step 4 / Part B: produce a "future-aware" action for the CURRENT frame
    given the risk_data sequence [risk_t, risk_{t+1}, ..., risk_{t+H}].

    Semantics: "what should I do NOW knowing what's coming?"

    Rule: if any frame within the lookahead window has a BRAKE policy_label,
    return that earliest-brake action with a reason explaining that it
    anticipates future risk. The brake percentage / steer come from the
    *earliest* future BRAKE so the model learns to react in advance, not
    wait until current frame is already CRITICAL.

    Falls back to the current frame's per-frame action if no future BRAKE is
    required. Always returns a 5-tuple of the same shape as policy_from_risk:
    (accel, brake, steer, reason, policy_label).

    Handles scene-end edge case naturally: if `risk_data_window` is shorter
    than expected (truncated near scene end), this still works — just
    examines whatever frames are available.
    """
    if not risk_data_window:
        # Defensive: no frames at all -> CONTINUE.
        return 20, 0, "straight", "No data available.", "CONTINUE"

    current = risk_data_window[0]
    current_action = policy_from_risk(current)

    # If current frame ALREADY requires BRAKE, the future-aware action is
    # the same — keep label consistent.
    if current_action[4] == "BRAKE":
        return current_action

    # Look for the earliest future BRAKE.
    for k in range(1, len(risk_data_window)):
        future_action = policy_from_risk(risk_data_window[k])
        if future_action[4] == "BRAKE":
            accel, brake, steer, reason, label = future_action
            lookahead_s = k * horizon_seconds_per_step
            anticipation_reason = (
                f"Anticipating risk in {lookahead_s:.1f}s: {reason}"
            )
            return accel, brake, steer, anticipation_reason, label

    # No future BRAKE in the window — return current frame's action.
    return current_action


def get_risk_summary_text(risk_data: FrameRiskData) -> str:
    """
    Compact, Stage-2 friendly summary (keep short to avoid token bloat).
    """
    if risk_data.num_risk_objects == 0:
        return "The driving situation is clear with no significant risks."

    parts = [f"Risk level: {risk_data.risk_level}."]

    if risk_data.min_ttc is not None:
        parts.append(f"Minimum time-to-collision: {risk_data.min_ttc:.1f} seconds.")

    if risk_data.max_collision_risk >= 0.3:
        parts.append(f"Maximum collision risk: {risk_data.max_collision_risk:.0%}.")

    if risk_data.max_pedestrian_risk >= 0.3:
        parts.append(f"Pedestrian risk: {risk_data.max_pedestrian_risk:.0%}.")

    return " ".join(parts)
