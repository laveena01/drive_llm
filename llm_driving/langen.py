# llm_driving/langen.py

"""
lanGen-style utilities:
- vector -> structured language caption (Stage 1 output)
- vector string formatting (Stage 1 input)

Option-A update:
Vector now contains ego-frame velocity components (rel_vx, rel_vy).
"""

from typing import List
import math
import numpy as np
from collections import Counter
import logging

from .config import MAX_OBJECTS, VECTOR_DIM

logger = logging.getLogger("llm_driving")


def describe_object(obj_vec: np.ndarray) -> str:
    """
    Option-A vector:
    [rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id]
    Returns a stable, paper-style language description for one object.
    """
    try:
        rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id = obj_vec
    except Exception:
        logger.exception("[lanGen] describe_object received invalid obj_vec shape.")
        return "An object is nearby."

    # ---------- Object type ----------
    tid = int(type_id)
    if tid == 0:
        obj_type = "car"
    elif tid == 1:
        obj_type = "pedestrian"
    elif tid == 2:
        obj_type = "traffic light"
    else:
        obj_type = "object"

    # ---------- Size (coarse, stable bins) ----------
    size = float(size)
    if size >= 2.5:
        size_desc = "large"
    elif size <= 1.0:
        size_desc = "small"
    else:
        size_desc = "medium-sized"

    # ---------- Motion (speed magnitude from vx, vy) ----------
    try:
        speed = float(math.sqrt(float(rel_vx) ** 2 + float(rel_vy) ** 2))
    except Exception:
        speed = 0.0

    if speed >= 2.0:
        speed_desc = "moving fast"
    else:
        speed_desc = "moving steadily"

    # ---------- Direction ----------
    angle_deg = math.degrees(math.atan2(float(rel_y), float(rel_x) + 1e-6))
    if angle_deg > 45:
        direction = "far to the left"
    elif angle_deg > 10:
        direction = "slightly to the left"
    elif angle_deg < -45:
        direction = "far to the right"
    elif angle_deg < -10:
        direction = "slightly to the right"
    else:
        direction = "straight ahead"

    return (
        f"A {size_desc} {obj_type} is {float(dist):.1f} meters "
        f"{direction}, {speed_desc}."
    )


def lanGen(frame: dict) -> str:
    """
    Create a structured language caption for a frame (Stage 1 target).

    frame must contain:
        - "vectors": np.ndarray(MAX_OBJECTS, VECTOR_DIM)
        - "num_objects": int

    Optional risk enhancement when frame contains:
        - "risk_data": dict with risk assessment
    """
    if "num_objects" not in frame or "vectors" not in frame:
        logger.error("[lanGen] frame missing required keys: expected 'num_objects' and 'vectors'.")
        return "There are no relevant objects nearby.\nMy current speed is 10.0 m/s.\nThe route continues straight ahead."

    num_objects = int(frame["num_objects"])
    vectors = frame["vectors"]
    risk_data = frame.get("risk_data", None)

    try:
        arr = np.asarray(vectors)
        if arr.ndim != 2 or arr.shape[1] < VECTOR_DIM:
            logger.warning(f"[lanGen] vectors shape looks odd: got {arr.shape}, expected (*, {VECTOR_DIM}).")
        if not np.isfinite(arr[: min(num_objects, MAX_OBJECTS)]).all():
            logger.warning("[lanGen] vectors contain NaN/Inf (will still generate caption).")
    except Exception:
        logger.exception("[lanGen] Failed to validate vectors; proceeding anyway.")

    lines: List[str] = []

    if num_objects == 0:
        lines.append("There are no relevant objects nearby.")
    else:
        use_n = min(num_objects, MAX_OBJECTS)

        type_names = {0: "car", 1: "pedestrian", 2: "traffic light", 3: "object"}
        counts = Counter(int(v[-1]) for v in vectors[:use_n])

        summary_parts = []
        for tid in [0, 1, 2, 3]:
            c = counts.get(tid, 0)
            if c > 0:
                name = type_names[tid]
                if c > 1:
                    name = "traffic lights" if name == "traffic light" else name + "s"
                summary_parts.append(f"{c} {name}")

        lines.append("There are " + ", ".join(summary_parts) + " nearby." if summary_parts else "There are objects nearby.")

        for i in range(use_n):
            lines.append(describe_object(vectors[i]))

    lines.append("My current speed is 10.0 m/s.")
    lines.append("The route continues straight ahead.")

    if isinstance(risk_data, dict):
        risk_level = risk_data.get("risk_level", "UNKNOWN")
        min_ttc = risk_data.get("min_ttc")
        max_collision = risk_data.get("max_collision_risk", 0)
        max_pedestrian = risk_data.get("max_pedestrian_risk", 0)

        risk_parts = [f"Risk assessment: {risk_level}."]
        if min_ttc is not None:
            try:
                risk_parts.append(f"Time-to-collision: {float(min_ttc):.1f}s.")
            except Exception:
                logger.warning("[lanGen] min_ttc in risk_data is not numeric.")
        try:
            if float(max_collision) >= 0.3:
                risk_parts.append(f"Collision risk: {float(max_collision):.0%}.")
        except Exception:
            logger.warning("[lanGen] max_collision_risk in risk_data is not numeric.")
        try:
            if float(max_pedestrian) >= 0.3:
                risk_parts.append(f"Pedestrian risk: {float(max_pedestrian):.0%}.")
        except Exception:
            logger.warning("[lanGen] max_pedestrian_risk in risk_data is not numeric.")

        lines.append(" ".join(risk_parts))

    return "\n".join(lines)


def vector_to_string(vectors: np.ndarray, num_objects: int) -> str:
    """
    Convert numeric object vectors into a compact text string (Stage 1 input).
    """
    num_objects = int(num_objects)
    if num_objects == 0:
        return ""

    objs: List[str] = []
    use_n = min(num_objects, MAX_OBJECTS)

    try:
        arr = np.asarray(vectors)
        if arr.ndim != 2:
            logger.warning(f"[vector_to_string] vectors has unexpected ndim={arr.ndim}.")
    except Exception:
        logger.exception("[vector_to_string] Could not convert vectors to ndarray.")

    for i in range(use_n):
        obj = vectors[i]
        objs.append(",".join([f"{float(x):.2f}" for x in obj]))

    return "; ".join(objs)