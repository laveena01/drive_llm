# llm_driving/risk_calculator.py

"""
Risk-aware driving calculations integrated from the risk module.

Option-A update:
- Uses ego-frame velocity components (rel_vx, rel_vy) from vectors.
- Removes the incorrect "heading -> velocity direction" assumption.
- Adds simple path gating (front cone + lateral band) to avoid side objects
  dominating TTC/collision risk.
"""

from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
import logging
import math

import numpy as np

from .nuscenes_risk_integration import SimpleRiskCalculator

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


# =============================================================================
# RISK CALCULATION FOR FRAME VECTORS
# =============================================================================

@dataclass
class FrameRiskData:
    """Risk data for an entire frame"""
    risk_level: str
    max_collision_risk: float
    max_pedestrian_risk: float
    min_ttc: Optional[float]
    avg_total_risk: float
    num_risk_objects: int
    object_risks: List[Dict]

    def to_dict(self) -> Dict:
        return asdict(self)


def _in_path_gate(rel_x: float, rel_y: float) -> bool:
    """
    Simple geometric gating to decide whether an object is likely in the ego path.
    """
    if RISK_REQUIRE_IN_FRONT and rel_x <= 0.0:
        return False

    # angle wrt ego forward (+x)
    angle_deg = abs(math.degrees(math.atan2(rel_y, rel_x + 1e-6)))
    if angle_deg > float(RISK_FRONT_CONE_DEG):
        return False

    if abs(rel_y) > float(RISK_LATERAL_BAND_M):
        return False

    return True


def calculate_risk_from_vectors(
    vectors: np.ndarray,
    use_n: Optional[int] = None,
    ego_speed: Optional[float] = None,
    traffic_light: Optional[str] = None,
) -> FrameRiskData:
    """
    Calculate comprehensive risk data from object vectors.

    Vector format (Option A, VECTOR_DIM=8):
      [rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id]

    Args:
        vectors: (MAX_OBJECTS, VECTOR_DIM) array
        use_n: number of objects to use (None -> infer)
        ego_speed: ego vehicle speed in m/s (None -> DEFAULT_EGO_SPEED)
        traffic_light: optional ('red','yellow','green')

    Returns:
        FrameRiskData
    """
    try:
        arr = np.asarray(vectors)
    except Exception:
        logger.exception("[risk_calculator] vectors could not be converted to ndarray.")
        arr = vectors

    if arr is None or len(arr) == 0:
        logger.warning("[risk_calculator] Empty vectors array; returning MINIMAL risk.")
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

    calculator = SimpleRiskCalculator(ego_speed=float(ego_speed))

    object_risks: List[Dict] = []
    total_risk_sum = 0.0
    max_collision = 0.0
    max_pedestrian = 0.0
    min_ttc = float("inf")

    for i in range(use_n):
        vec = arr[i]

        # Option-A unpack (8D)
        try:
            rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id = vec
        except Exception:
            logger.exception("[risk_calculator] Bad vector at index=%d; skipping.", i)
            continue

        rel_x = float(rel_x)
        rel_y = float(rel_y)

        position = np.array([rel_x, rel_y], dtype=np.float32)

        # Option-A: use ego-frame velocity components directly
        try:
            vx = float(rel_vx)
            vy = float(rel_vy)
        except Exception:
            logger.exception("[risk_calculator] rel_vx/rel_vy invalid at index=%d; using zero velocity.", i)
            vx, vy = 0.0, 0.0
        velocity = np.array([vx, vy], dtype=np.float32)

        # Size as [width, length] proxy duplicated
        try:
            s = float(size)
        except Exception:
            logger.exception("[risk_calculator] size invalid at index=%d; using 1.0.", i)
            s = 1.0
        size_arr = np.array([s, s], dtype=np.float32)

        try:
            obj_type = TYPE_ID_TO_CATEGORY.get(int(type_id), "object.other")
        except Exception:
            logger.exception("[risk_calculator] type_id invalid at index=%d; using object.other.", i)
            obj_type = "object.other"

        # Compute raw risk from calculator
        try:
            risk_components, metadata = calculator.calculate_risk(
                position=position,
                velocity=velocity,
                size=size_arr,
                obj_type=obj_type,
                traffic_light=traffic_light,
            )
        except Exception:
            logger.exception("[risk_calculator] calculate_risk failed at index=%d; skipping.", i)
            continue

        # ---------- Path gating (mandatory) ----------
        # If object is NOT in-path, down-weight collision/TTC contributions
        in_path = _in_path_gate(rel_x, rel_y)

        # risk_components has: collision_risk, pedestrian_risk, ttc_risk, regulatory_risk, total_risk, risk_level
        col = float(risk_components.collision_risk)
        ped = float(risk_components.pedestrian_risk)
        ttc_r = float(risk_components.ttc_risk)
        reg = float(risk_components.regulatory_risk)

        if not in_path:
            # Keep ped risk (vulnerable users can matter even if slightly off path),
            # but suppress collision/ttc so far-side objects don't force CRITICAL.
            col *= 0.2
            ttc_r *= 0.2

        # Recompute total risk using the same weights as your SimpleRiskCalculator
        total = col * 0.40 + ped * 0.30 + ttc_r * 0.20 + reg * 0.10

        # Recompute a derived risk_level (consistent with calculator thresholds)
        if total >= 0.8:
            derived_level = "CRITICAL"
        elif total >= 0.6:
            derived_level = "HIGH"
        elif total >= 0.4:
            derived_level = "MODERATE"
        elif total >= 0.2:
            derived_level = "LOW"
        else:
            derived_level = "MINIMAL"

        # TTC tracking
        # TTC tracking (IMPORTANT: only consider in-path objects)
        ttc_val = metadata.get("ttc", None)
        if in_path and ttc_val is not None:
            try:
                ttc_f = float(ttc_val)
                if ttc_f < min_ttc:
                    min_ttc = ttc_f
            except Exception:
                pass

        # Aggregate stats (use gated collision/ped values!)
        total_risk_sum += float(total)
        max_collision = max(max_collision, float(col))
        max_pedestrian = max(max_pedestrian, float(ped))

        object_risks.append({
            "idx": i,
            "type": obj_type,
            "distance": metadata.get("distance"),
            "ttc": metadata.get("ttc"),
            "closing_speed": metadata.get("closing_speed"),
            "in_path": bool(in_path),
            "risk": {
                "collision_risk": round(col, 3),
                "pedestrian_risk": round(ped, 3),
                "ttc_risk": round(ttc_r, 3),
                "regulatory_risk": round(reg, 3),
                "total_risk": round(total, 3),
                "risk_level": derived_level,
            }
        })

    effective_n = len(object_risks)
    if effective_n == 0:
        logger.warning("[risk_calculator] All objects skipped; returning MINIMAL risk.")
        return FrameRiskData(
            risk_level="MINIMAL",
            max_collision_risk=0.0,
            max_pedestrian_risk=0.0,
            min_ttc=None,
            avg_total_risk=0.0,
            num_risk_objects=0,
            object_risks=[],
        )

    avg_risk = total_risk_sum / effective_n

    # Frame-level risk: based on max collision OR small TTC (same style as before, but now collision is gated)
    score = 0.7 * max_collision + 0.3 * avg_risk

    if (min_ttc < float("inf") and min_ttc < 2.0) or score >= 0.75:
        risk_level = "CRITICAL"
    elif (min_ttc < float("inf") and min_ttc < 3.0) or score >= 0.55:
        risk_level = "HIGH"
    elif (min_ttc < float("inf") and min_ttc < 5.0) or score >= 0.35:
        risk_level = "MODERATE"
    elif score >= 0.18:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"

    return FrameRiskData(
        risk_level=risk_level,
        max_collision_risk=round(float(max_collision), 3),
        max_pedestrian_risk=round(float(max_pedestrian), 3),
        min_ttc=round(min_ttc, 2) if min_ttc < float("inf") else None,
        avg_total_risk=round(float(avg_risk), 3),
        num_risk_objects=int(effective_n),
        object_risks=object_risks,
    )


# =============================================================================
# RISK-BASED POLICY DETERMINATION
# =============================================================================

def policy_from_risk(risk_data: FrameRiskData) -> Tuple[int, int, str, str, str]:
    """
    Determine driving policy based on multi-dimensional risk assessment.
    Returns: (accel, brake, steer, reason, policy_label)
    """
    steer = "straight"
    risk_level = risk_data.risk_level
    min_ttc = risk_data.min_ttc
    max_collision = risk_data.max_collision_risk
    max_pedestrian = risk_data.max_pedestrian_risk

    if risk_data.num_risk_objects == 0:
        return 20, 0, steer, "No nearby obstacles detected.", "CONTINUE"

    if risk_level == "CRITICAL":
        if min_ttc is not None and min_ttc < 2:
            return 0, 90, steer, f"Critical risk: TTC={min_ttc:.1f}s, emergency braking required.", "BRAKE"
        return 0, 80, steer, f"Critical risk level (collision={max_collision:.0%}), braking hard.", "BRAKE"

    if risk_level == "HIGH":
        if max_pedestrian >= 0.5:
            return 0, 60, steer, f"High pedestrian risk ({max_pedestrian:.0%}), reducing speed.", "BRAKE"
        return 0, 50, steer, f"High risk (collision={max_collision:.0%}), slowing down.", "BRAKE"

    if risk_level == "MODERATE":
        if max_pedestrian >= 0.3:
            return 5, 30, steer, "Moderate pedestrian risk, proceeding with caution.", "CAUTION"
        return 10, 20, steer, f"Moderate risk (collision={max_collision:.0%}), proceed carefully.", "CAUTION"

    if risk_level == "LOW":
        # IMPORTANT: brake=0 to keep labels consistent with your bucket metric
        return 15, 0, steer, "Low risk detected, maintaining awareness.", "CONTINUE"

    return 20, 0, steer, "Minimal risk, safe to continue.", "CONTINUE"


def get_risk_summary_text(risk_data: FrameRiskData) -> str:
    """Generate a human-readable risk summary."""
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