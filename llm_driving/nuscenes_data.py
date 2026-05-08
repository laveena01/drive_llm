# llm_driving/nuscenes_data.py

"""
Utilities for loading nuScenes and extracting object-level vectors.

Vector format per object (UPDATED for Option A):
[rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id]

Where:
- rel_x, rel_y are in EGO frame (meters)
- dist is sqrt(rel_x^2 + rel_y^2)
- rel_vx, rel_vy are object's velocity components in EGO frame (m/s)
- heading is yaw angle (radians) of the object's box in ego frame (orientation, not velocity direction)
- size is avg(length, width) proxy (meters)
- type_id: 0 car, 1 pedestrian, 2 traffic_light, 3 other
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from .config import NUSC_ROOT, NUSC_VERSION, MAX_OBJECTS, VECTOR_DIM

logger = logging.getLogger("llm_driving")


def init_nuscenes() -> NuScenes:
    logger.info("[nuscenes_data] Initializing nuScenes with:")
    logger.info(f"  - dataroot = {NUSC_ROOT}")
    logger.info(f"  - version  = {NUSC_VERSION}")

    nusc = NuScenes(version=NUSC_VERSION, dataroot=NUSC_ROOT, verbose=True)

    logger.info("[nuscenes_data] nuScenes initialized successfully.")
    return nusc


def _yaw_from_quaternion(q: Quaternion) -> float:
    """
    Return yaw (rotation around z) from quaternion.
    nuScenes uses (w, x, y, z).
    """
    w, x, y, z = q.w, q.x, q.y, q.z
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def get_object_vectors_for_sample(
    nusc: NuScenes,
    sample_token: str,
    override_ego_t: Optional[np.ndarray] = None,
    override_ego_q: Optional[Quaternion] = None,
) -> Tuple[np.ndarray, int, List[str]]:
    """
    Extract object vectors for a given sample token.
    Uses ego pose (LIDAR_TOP) to convert global boxes -> ego frame.

    Step 2 / Part B: when `override_ego_t` and `override_ego_q` are both
    provided, object positions and velocities are re-expressed in *that*
    ego frame instead of this sample's own ego frame. This is what lets
    `get_frame_window` express past-frame objects in the *current* frame's
    coordinates so the temporal transformer sees a consistent reference
    (without this, a parked car would appear to drift across the window
    simply because ego moved between frames).

    Returns:
        Tuple of (vectors, count, category_names)
        - vectors: (MAX_OBJECTS, VECTOR_DIM) array
        - count: number of valid objects
        - category_names: list of category names for each object (optional use)
    """
    sample = nusc.get("sample", sample_token)

    lidar_sd_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_sd_token)
    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    if override_ego_t is not None and override_ego_q is not None:
        ego_t = np.asarray(override_ego_t, dtype=np.float32)
        ego_q = override_ego_q
    else:
        ego_t = np.array(ego_pose["translation"], dtype=np.float32)  # (x,y,z)
        ego_q = Quaternion(ego_pose["rotation"])  # (w,x,y,z)
    ego_q_inv = ego_q.inverse

    vectors: List[List[float]] = []
    categories: List[str] = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)

        obj_t = np.array(ann["translation"], dtype=np.float32)
        obj_q = Quaternion(ann["rotation"])

        # --- Transform position: global -> ego frame ---
        rel_global = obj_t - ego_t
        rel_ego = ego_q_inv.rotate(rel_global)

        rel_x = float(rel_ego[0])
        rel_y = float(rel_ego[1])
        dist = float(np.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-6))

        # --- Velocity: global -> ego frame components (UPDATED) ---
        rel_vx, rel_vy = 0.0, 0.0
        try:
            vx, vy, vz = nusc.box_velocity(ann_token)
            if not np.any(np.isnan([vx, vy, vz])):
                v_global = np.array([vx, vy, vz], dtype=np.float32)
                v_ego = ego_q_inv.rotate(v_global)
                rel_vx = float(v_ego[0])
                rel_vy = float(v_ego[1])
        except Exception:
            rel_vx, rel_vy = 0.0, 0.0

        # --- Heading: object yaw in ego frame (orientation) ---
        obj_q_ego = ego_q_inv * obj_q
        heading = _yaw_from_quaternion(obj_q_ego)

        # --- Size proxy ---
        w, l, h = ann["size"]
        size = float((float(w) + float(l)) / 2.0)

        # --- Type id ---
        category = ann["category_name"]
        if "vehicle" in category:
            type_id = 0
        elif "pedestrian" in category:
            type_id = 1
        elif "traffic_light" in category:
            type_id = 2
        else:
            type_id = 3

        # UPDATED vector: [x, y, dist, vx, vy, heading, size, type_id]
        vectors.append([rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, float(type_id)])
        categories.append(category)

    # Sort by distance so MAX_OBJECTS are nearest ones
    if vectors:
        sorted_pairs = sorted(zip(vectors, categories), key=lambda x: x[0][2])
        vectors = [p[0] for p in sorted_pairs]
        categories = [p[1] for p in sorted_pairs]

    padded = np.zeros((MAX_OBJECTS, VECTOR_DIM), dtype=np.float32)
    count = min(len(vectors), MAX_OBJECTS)
    if count > 0:
        padded[:count, :] = np.array(vectors[:count], dtype=np.float32)

    categories = categories[:count]
    return padded, count, categories


def _compute_ego_speeds(
    translations: List[np.ndarray],
    timestamps_us: List[int],
) -> List[float]:
    """
    Per-frame ego speed (m/s) from a sequence of global ego-pose translations
    and timestamps in microseconds.

    Central difference for interior frames; forward/backward difference at
    the boundaries. Speed is the 2-D planar magnitude (ignores z drift).
    Clamped to >= 0. Returns 0.0 for any frame with non-positive dt.

    A single-frame scene returns [0.0]; downstream code should treat 0.0 as
    "speed unknown" and fall back to `DEFAULT_EGO_SPEED` if needed.
    """
    n = len(translations)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    speeds: List[float] = []
    for i in range(n):
        if i == 0:
            j_a, j_b = 0, 1
        elif i == n - 1:
            j_a, j_b = n - 2, n - 1
        else:
            j_a, j_b = i - 1, i + 1
        dt_s = (timestamps_us[j_b] - timestamps_us[j_a]) / 1.0e6
        if dt_s <= 0:
            speeds.append(0.0)
            continue
        dx = float(translations[j_b][0] - translations[j_a][0])
        dy = float(translations[j_b][1] - translations[j_a][1])
        speed = float(np.sqrt(dx * dx + dy * dy) / dt_s)
        speeds.append(max(0.0, speed))
    return speeds


def get_scene_frames_vectors(
    nusc: NuScenes,
    scene_idx: int = 0,
    max_frames: Optional[int] = None,
    temporal_window: Optional[int] = None,
) -> List[Dict]:
    """
    Returns list of frames, each as:
      {
        "vectors": np.ndarray(MAX_OBJECTS, VECTOR_DIM),
        "num_objects": int,
        "sample_token": str,
        "categories": List[str],
        # Step 2 / Part A — real ego-motion (replaces the global
        # DEFAULT_EGO_SPEED constant for downstream risk computation):
        "ego_speed": float,                       # m/s, planar 2-D, >= 0
        "ego_pose_translation": List[float],      # [tx, ty, tz] global frame
        "ego_pose_rotation": List[float],         # [w, x, y, z] quaternion
        "lidar_timestamp_us": int                 # for completeness / debug
      }

    `ego_pose_translation` and `ego_pose_rotation` are persisted on each
    frame so the Part B (temporal-window) coordinate transform can re-express
    past frames' object positions in the *current* frame's ego coordinates
    without a second nuScenes pass.
    """
    scene = nusc.scene[scene_idx]
    token = scene["first_sample_token"]

    logger.info(
        f"[nuscenes_data] Collecting frames for scene {scene_idx} "
        f"(scene_token={scene['token']}, max_frames={max_frames})"
    )

    # First pass: collect per-frame data (vectors + ego pose) without
    # computing ego_speed yet — speed needs neighbouring frames.
    pending_frames: List[Dict] = []
    pending_translations: List[np.ndarray] = []
    pending_timestamps_us: List[int] = []
    frame_idx = 0

    while token:
        if max_frames is not None and frame_idx >= max_frames:
            break

        vecs, num_obj, categories = get_object_vectors_for_sample(nusc, token)
        sample = nusc.get("sample", token)

        # Fetch ego pose tied to the LIDAR_TOP keyframe — same pose that
        # `get_object_vectors_for_sample` uses internally for ego-frame
        # transforms, so positions and ego_speed are mutually consistent.
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ego_t = np.array(ego_pose["translation"], dtype=np.float64)
        ego_q = list(ego_pose["rotation"])  # [w, x, y, z]
        ts_us = int(lidar_sd["timestamp"])

        pending_frames.append(
            {
                "vectors": vecs,
                "num_objects": num_obj,
                "sample_token": token,
                "categories": categories,
                "ego_pose_translation": [float(ego_t[0]), float(ego_t[1]), float(ego_t[2])],
                "ego_pose_rotation": [float(c) for c in ego_q],
                "lidar_timestamp_us": ts_us,
            }
        )
        pending_translations.append(ego_t)
        pending_timestamps_us.append(ts_us)

        token = sample["next"]
        frame_idx += 1

        if frame_idx % 20 == 0:
            logger.info(f"[nuscenes_data]  Processed {frame_idx} frames in scene {scene_idx}...")

    # Second pass: attach ego_speed via central-difference on translations.
    speeds = _compute_ego_speeds(pending_translations, pending_timestamps_us)
    for f, s in zip(pending_frames, speeds):
        f["ego_speed"] = float(s)

    if pending_frames:
        speeds_arr = np.array(speeds, dtype=np.float32)
        logger.info(
            f"[nuscenes_data] Scene {scene_idx} ego_speed (m/s): "
            f"min={float(speeds_arr.min()):.2f} "
            f"mean={float(speeds_arr.mean()):.2f} "
            f"max={float(speeds_arr.max()):.2f}"
        )

    # Step 2 / Part B — opt-in temporal window. Run a third pass once
    # per-frame ego data is in place so each window can re-express past
    # frames in the *current* frame's ego coordinates.
    if temporal_window is not None and temporal_window > 0 and pending_frames:
        for i in range(len(pending_frames)):
            vw, nw, wl = get_frame_window(nusc, pending_frames, i, temporal_window)
            pending_frames[i]["vectors_window"] = vw
            pending_frames[i]["num_objects_window"] = nw
            pending_frames[i]["window_len"] = wl
        logger.info(
            f"[nuscenes_data] Scene {scene_idx}: built K={temporal_window} "
            f"temporal window for {len(pending_frames)} frames"
        )

    logger.info(f"[nuscenes_data] Extracted {len(pending_frames)} frames for scene {scene_idx}")
    return pending_frames


def get_frame_window(
    nusc: NuScenes,
    scene_frames: List[Dict],
    current_idx: int,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build a K-frame temporal window ending at `current_idx` in the same scene.

    Step 2 / Part B: each *past* frame's object positions and velocities are
    re-expressed in the **current** frame's ego coordinates (via the
    `override_ego_t` / `override_ego_q` path in
    `get_object_vectors_for_sample`). The current frame itself is reused
    directly from `scene_frames[current_idx]` since its vectors are already
    in its own ego frame.

    Layout (right-aligned: current frame at slot k-1):
        slot:        0          1     ...     k-2        k-1
        frame:    t-(K-1)   t-(K-2)   ...    t-1          t

    For scene-start frames where fewer than K real frames exist, the leading
    slots are zero-padded; `window_len` reports how many real frames the
    window contains.

    Args:
        nusc:          initialised NuScenes
        scene_frames:  list returned by `get_scene_frames_vectors`
        current_idx:   index of the *current* frame in `scene_frames`
        k:             window size (≥ 1)

    Returns:
        vectors_window:     (k, MAX_OBJECTS, VECTOR_DIM) float32, zero-padded
        num_objects_window: (k,) int64, zero where the slot is padding
        window_len:         int in [1, k], number of real frames present
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not 0 <= current_idx < len(scene_frames):
        raise IndexError(
            f"current_idx={current_idx} out of range for {len(scene_frames)} frames"
        )

    cur_frame = scene_frames[current_idx]
    cur_ego_t = np.asarray(cur_frame["ego_pose_translation"], dtype=np.float32)
    cur_ego_q = Quaternion(cur_frame["ego_pose_rotation"])

    start = max(0, current_idx - k + 1)
    real_indices = list(range(start, current_idx + 1))
    window_len = len(real_indices)

    vectors_window = np.zeros((k, MAX_OBJECTS, VECTOR_DIM), dtype=np.float32)
    num_objects_window = np.zeros((k,), dtype=np.int64)

    # Right-align: current frame at slot k-1, oldest real frame at slot
    # k - window_len, slots 0..(k-window_len-1) stay zero (scene-start pad).
    for slot, frame_idx in enumerate(real_indices, start=k - window_len):
        past_frame = scene_frames[frame_idx]
        if frame_idx == current_idx:
            # Current frame's vectors are already in current ego coords.
            vectors_window[slot] = np.asarray(past_frame["vectors"], dtype=np.float32)
            num_objects_window[slot] = int(past_frame["num_objects"])
        else:
            # Past frame: recompute object vectors using *current* ego pose
            # so positions/velocities sit in a consistent reference frame.
            vecs, n, _ = get_object_vectors_for_sample(
                nusc,
                past_frame["sample_token"],
                override_ego_t=cur_ego_t,
                override_ego_q=cur_ego_q,
            )
            vectors_window[slot] = vecs
            num_objects_window[slot] = int(n)

    return vectors_window, num_objects_window, window_len