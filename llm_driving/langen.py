# llm_driving/langen.py

"""
lanGen-style utilities:
- vector -> structured language caption (Stage 1 output)
- vector string formatting (Stage 1 input)
"""

from typing import List
import math
import numpy as np
from collections import Counter

from .config import MAX_OBJECTS, VECTOR_DIM


def describe_object(obj_vec: np.ndarray) -> str:
    """
    obj_vec: [rel_x, rel_y, dist, rel_speed, heading, size, type_id]
    Returns a stable, paper-style language description for one object.
    """

    rel_x, rel_y, dist, rel_speed, heading, size, type_id = obj_vec

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

    # ---------- Motion (avoid semantic risk) ----------
    rel_speed = float(rel_speed)
    if abs(rel_speed) >= 2.0:
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
    """

    num_objects = int(frame["num_objects"])
    vectors = frame["vectors"]

    lines: List[str] = []

    if num_objects == 0:
        lines.append("There are no relevant objects nearby.")
    else:
        use_n = min(num_objects, MAX_OBJECTS)

        # ---------- Stable object count summary ----------
        type_names = {
            0: "car",
            1: "pedestrian",
            2: "traffic light",
            3: "object",
        }

        counts = Counter(int(v[-1]) for v in vectors[:use_n])

        summary_parts = []
        for tid in [0, 1, 2, 3]:
            c = counts.get(tid, 0)
            if c > 0:
                name = type_names[tid]
                if c > 1:
                    if name == "traffic light":
                        name = "traffic lights"
                    else:
                        name = name + "s"
                summary_parts.append(f"{c} {name}")

        lines.append("There are " + ", ".join(summary_parts) + " nearby.")

        # ---------- Per-object descriptions ----------
        for i in range(use_n):
            lines.append(describe_object(vectors[i]))

    # ---------- Ego + route (fixed placeholders) ----------
    lines.append("My current speed is 10.0 m/s.")
    lines.append("The route continues straight ahead.")

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

    for i in range(use_n):
        obj = vectors[i]
        objs.append(",".join([f"{float(x):.2f}" for x in obj]))

    return "; ".join(objs)
