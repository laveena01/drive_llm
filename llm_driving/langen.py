# llm_driving/langen.py

"""
lanGen-style utilities:
- vector -> structured language caption (Stage 1 output)
- vector string formatting (Stage 1 input)

Option-A update:
Vector now contains ego-frame velocity components (rel_vx, rel_vy).
"""

from typing import List, Optional
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


def _temporal_object_lines(
    vectors_window: np.ndarray,
    object_present_mask: np.ndarray,
    window_len: int,
    num_objects: int,
    top_n: int = 3,
) -> List[str]:
    """
    Step 4 / Part C: build compact per-object motion lines from a K-frame
    tracked window. Picks the top-N nearest objects in the current frame
    that were present in at least 2 frames of the window, then describes
    each one's deceleration / closing rate / lateral drift.

    Returns at most `top_n` lines, each ~12 tokens. Empty list if no
    candidates qualify.
    """
    if vectors_window is None or window_len < 2:
        return []
    K = int(vectors_window.shape[0])
    M = int(vectors_window.shape[1])
    n_objs = min(num_objects, M)
    if n_objs == 0:
        return []

    # Current frame is slot K-1 in the right-aligned window.
    cur = vectors_window[K - 1]  # (M, VECTOR_DIM)
    # type_id: 0=car, 1=pedestrian, 2=traffic_light, 3=object.
    type_names = {0: "car", 1: "ped", 2: "light", 3: "obj"}

    # Pick candidate slots: present in current frame + at least one past frame.
    candidates = []
    for j in range(n_objs):
        if not bool(object_present_mask[K - 1, j]):
            continue
        # Find the *earliest* past frame where this slot was present.
        earliest_past = None
        for k in range(K - 1):
            if bool(object_present_mask[k, j]):
                earliest_past = k
                break
        if earliest_past is None:
            continue  # only in current frame, no temporal signal yet.
        candidates.append((j, earliest_past))

    if not candidates:
        return []

    # Sort by current-frame distance, take top N.
    candidates.sort(key=lambda x: float(cur[x[0]][2]))
    candidates = candidates[:top_n]

    lines: List[str] = []
    for j, earliest_past in candidates:
        cur_vec = cur[j]
        past_vec = vectors_window[earliest_past, j]

        # Distance / closing rate over the window.
        dist_now = float(cur_vec[2])
        dist_past = float(past_vec[2])
        n_steps = K - 1 - earliest_past  # how many keyframes ago
        # Assume 0.5 s per keyframe (nuScenes 2 Hz).
        dt = max(0.001, n_steps * 0.5)
        closing = (dist_past - dist_now) / dt  # +ve = approaching

        # Speed magnitudes (in ego-frame relative velocity).
        speed_now = float(math.sqrt(cur_vec[3] ** 2 + cur_vec[4] ** 2))
        speed_past = float(math.sqrt(past_vec[3] ** 2 + past_vec[4] ** 2))
        delta_speed_per_s = (speed_now - speed_past) / dt
        # decel > 0 means slowing.
        decel = max(0.0, -delta_speed_per_s)

        tid = int(cur_vec[7])
        type_name = type_names.get(tid, "obj")

        # Build the compact line. Choose phrasing based on dominant signal.
        if closing >= 0.5 and decel >= 0.5:
            lines.append(
                f"Obj{j} ({type_name}, {dist_now:.1f}m): closing {closing:.1f}m/s, decel {decel:.1f}m/s."
            )
        elif closing >= 0.5:
            lines.append(
                f"Obj{j} ({type_name}, {dist_now:.1f}m): closing {closing:.1f}m/s."
            )
        elif decel >= 0.5:
            lines.append(
                f"Obj{j} ({type_name}, {dist_now:.1f}m): decel {decel:.1f}m/s."
            )
        elif closing <= -0.5:
            lines.append(
                f"Obj{j} ({type_name}, {dist_now:.1f}m): receding {-closing:.1f}m/s."
            )
        else:
            # Steady-state — still emit so the model sees there's no concerning motion.
            lines.append(
                f"Obj{j} ({type_name}, {dist_now:.1f}m): steady."
            )
    return lines


def lanGen(
    frame: dict,
    vectors_window: Optional[np.ndarray] = None,
    object_present_mask: Optional[np.ndarray] = None,
    window_len: int = 1,
    temporal_top_n: int = 3,
) -> str:
    """
    Create a structured language caption for a frame (Stage 1 target).

    frame must contain:
        - "vectors": np.ndarray(MAX_OBJECTS, VECTOR_DIM)
        - "num_objects": int

    Optional risk enhancement when frame contains:
        - "risk_data": dict with risk assessment

    Step 4 / Part C: when `vectors_window`, `object_present_mask`, and
    `window_len >= 2` are provided, also emit motion descriptions for the
    top-N closest objects that have a temporal trajectory (present in at
    least 2 frames). These compact lines (~12 tokens each) give Stage 1 a
    training target that rewards encoding temporal information into text —
    Stage 2 (which only reads text) then has temporal content to consume.
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

        # Step 4 / Part C: append compact motion descriptors for the top-N
        # closest objects with a temporal trajectory. Stage 1's training
        # target now rewards encoding motion patterns into text — that
        # information then flows to Stage 2 through the caption channel.
        if (
            vectors_window is not None
            and object_present_mask is not None
            and int(window_len) >= 2
        ):
            try:
                vw_arr = np.asarray(vectors_window, dtype=np.float32)
                opm_arr = np.asarray(object_present_mask, dtype=bool)
                temporal_lines = _temporal_object_lines(
                    vw_arr,
                    opm_arr,
                    int(window_len),
                    use_n,
                    top_n=int(temporal_top_n),
                )
                lines.extend(temporal_lines)
            except Exception:
                logger.exception("[lanGen] temporal-line generation failed; continuing without them.")

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