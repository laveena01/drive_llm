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
    target_instance_tokens: Optional[List[str]] = None,
) -> Tuple[np.ndarray, int, List[str], List[str]]:
    """
    Extract object vectors for a given sample token.
    Uses ego pose (LIDAR_TOP) to convert global boxes -> ego frame.

    Step 2 / Part B: when `override_ego_t` and `override_ego_q` are both
    provided, object positions and velocities are re-expressed in *that*
    ego frame instead of this sample's own ego frame.

    Step 4 / Part A: when `target_instance_tokens` is provided (length =
    MAX_OBJECTS), the returned vectors are aligned to those identities
    rather than sorted by distance — slot j of the output carries the
    vector for `target_instance_tokens[j]` (or zeros if that instance is
    not present in this sample). The `count` returned in this mode is the
    number of slots that were successfully filled (i.e. how many of the
    target instances appeared here). `categories` is aligned to the
    target slots; empty string for missing slots.

    Returns:
        Tuple of (vectors, count, category_names, instance_tokens)
        - vectors: (MAX_OBJECTS, VECTOR_DIM) array
        - count: number of valid (filled) slots
        - category_names: list of category names per slot
        - instance_tokens: list of instance_tokens per slot (empty string
          for missing/zero slots in tracked mode)
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

    # Per-annotation parse → dict keyed by instance_token for fast lookup.
    parsed: Dict[str, Tuple[List[float], str]] = {}

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        instance_token = ann.get("instance_token", "")

        obj_t = np.array(ann["translation"], dtype=np.float32)
        obj_q = Quaternion(ann["rotation"])

        # --- Transform position: global -> ego frame ---
        rel_global = obj_t - ego_t
        rel_ego = ego_q_inv.rotate(rel_global)

        rel_x = float(rel_ego[0])
        rel_y = float(rel_ego[1])
        dist = float(np.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-6))

        # --- Velocity: global -> ego frame components ---
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

        vec = [rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, float(type_id)]
        parsed[instance_token] = (vec, category)

    padded = np.zeros((MAX_OBJECTS, VECTOR_DIM), dtype=np.float32)

    if target_instance_tokens is None:
        # --- Original sort-by-distance behaviour ---
        vectors_list = [v for v, _ in parsed.values()]
        categories_list = [c for _, c in parsed.values()]
        instance_tokens_list = list(parsed.keys())
        if vectors_list:
            sort_idx = sorted(
                range(len(vectors_list)), key=lambda i: vectors_list[i][2]
            )
            vectors_list = [vectors_list[i] for i in sort_idx]
            categories_list = [categories_list[i] for i in sort_idx]
            instance_tokens_list = [instance_tokens_list[i] for i in sort_idx]

        count = min(len(vectors_list), MAX_OBJECTS)
        if count > 0:
            padded[:count, :] = np.array(vectors_list[:count], dtype=np.float32)
        categories_out = categories_list[:count]
        instance_tokens_out = instance_tokens_list[:count]
        return padded, count, categories_out, instance_tokens_out

    # --- Step 4 / Part A: tracked mode ---
    # Output is aligned to target_instance_tokens; missing instances get
    # zero rows. Slots beyond len(target_instance_tokens) stay zero.
    categories_out = ["" for _ in range(MAX_OBJECTS)]
    instance_tokens_out = ["" for _ in range(MAX_OBJECTS)]
    count = 0
    n_targets = min(len(target_instance_tokens), MAX_OBJECTS)
    for slot in range(n_targets):
        tok = target_instance_tokens[slot]
        if tok and tok in parsed:
            vec, cat = parsed[tok]
            padded[slot, :] = np.array(vec, dtype=np.float32)
            categories_out[slot] = cat
            instance_tokens_out[slot] = tok
            count += 1
        # else: leave zero row, empty category/token (marks "object absent
        # in this frame" — caller composes object_present_mask).
    return padded, count, categories_out, instance_tokens_out


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

        vecs, num_obj, categories, instance_tokens = get_object_vectors_for_sample(
            nusc, token
        )
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
                # Step 4 / Part A: instance_tokens per slot. Slot j's token
                # follows the same physical object across frames when used
                # by `get_frame_window_tracked`.
                "instance_tokens": instance_tokens,
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

    # Step 2 / Part B + Step 4 / Part A — opt-in temporal window. Dispatch
    # between the buggy sort-by-distance variant (kept for ablation) and
    # the tracked-identity variant based on cfg.USE_TRACKED_TEMPORAL.
    if temporal_window is not None and temporal_window > 0 and pending_frames:
        # Import here to avoid a circular at module load time.
        from .config import USE_TRACKED_TEMPORAL as _USE_TRACKED

        if _USE_TRACKED:
            for i in range(len(pending_frames)):
                vw, nw, opm, wl = get_frame_window_tracked(
                    nusc, pending_frames, i, temporal_window
                )
                pending_frames[i]["vectors_window"] = vw
                pending_frames[i]["num_objects_window"] = nw
                pending_frames[i]["object_present_mask"] = opm
                pending_frames[i]["window_len"] = wl
            logger.info(
                f"[nuscenes_data] Scene {scene_idx}: built K={temporal_window} "
                f"TRACKED temporal window for {len(pending_frames)} frames"
            )
        else:
            for i in range(len(pending_frames)):
                vw, nw, wl = get_frame_window(nusc, pending_frames, i, temporal_window)
                pending_frames[i]["vectors_window"] = vw
                pending_frames[i]["num_objects_window"] = nw
                pending_frames[i]["window_len"] = wl
                # Legacy sort-by-distance path has no identity tracking —
                # synthesize an all-True mask so downstream collator code
                # still works (treats every slot as present in every frame).
                pending_frames[i]["object_present_mask"] = np.ones(
                    (temporal_window, MAX_OBJECTS), dtype=bool
                )
            logger.info(
                f"[nuscenes_data] Scene {scene_idx}: built K={temporal_window} "
                f"sort-by-distance temporal window for {len(pending_frames)} frames"
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
            vecs, n, _, _ = get_object_vectors_for_sample(
                nusc,
                past_frame["sample_token"],
                override_ego_t=cur_ego_t,
                override_ego_q=cur_ego_q,
            )
            vectors_window[slot] = vecs
            num_objects_window[slot] = int(n)

    return vectors_window, num_objects_window, window_len


def get_frame_window_tracked(
    nusc: NuScenes,
    scene_frames: List[Dict],
    current_idx: int,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Build a K-frame temporal window with **object identity preserved across
    frames** using nuScenes `instance_token`.

    Step 4 / Part A: this fixes the silent correctness bug in
    `get_frame_window` (the sort-by-distance variant), where "slot 0 at
    t-3" might be a completely different physical object than "slot 0 at
    t". Here, slot j across all K frames refers to the SAME instance_token
    (taken from the current frame's slot j), or is zero-padded with
    `object_present_mask[k, j] = False` when that instance wasn't visible
    in frame k.

    Layout (right-aligned: current frame at slot k-1):
        slot:        0          1     ...     k-2        k-1
        frame:    t-(K-1)   t-(K-2)   ...    t-1          t

    Identity anchor = current frame's instance_tokens (slot 0..M-1).
    Past frames are re-expressed in the *current* ego frame via
    `override_ego_t/q`.

    Args:
        nusc:          initialised NuScenes
        scene_frames:  list returned by `get_scene_frames_vectors`. Each
                       frame must already carry `instance_tokens`,
                       `ego_pose_translation`, `ego_pose_rotation`.
        current_idx:   index of the current frame in `scene_frames`
        k:             window size (>= 1)

    Returns:
        vectors_window:      (k, MAX_OBJECTS, VECTOR_DIM) float32
        num_objects_window:  (k,) int64 — valid slot count per frame
        object_present_mask: (k, MAX_OBJECTS) bool — True if slot j was
                             present in frame k. Slots beyond
                             num_anchor_objects are always False.
        window_len:          int in [1, k] — number of real frames (the
                             K - window_len leading slots are zero-padded
                             for scene-start cases)
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
    # The anchor identities are the current frame's slot 0..num_objects-1.
    # Padded slots beyond num_objects have empty instance_tokens — those
    # rows stay zero in every frame's window slot.
    anchor_tokens: List[str] = list(cur_frame.get("instance_tokens", []))
    # Pad / truncate to MAX_OBJECTS.
    anchor_tokens = (anchor_tokens + [""] * MAX_OBJECTS)[:MAX_OBJECTS]
    num_anchor = int(cur_frame["num_objects"])

    start = max(0, current_idx - k + 1)
    real_indices = list(range(start, current_idx + 1))
    window_len = len(real_indices)

    vectors_window = np.zeros((k, MAX_OBJECTS, VECTOR_DIM), dtype=np.float32)
    num_objects_window = np.zeros((k,), dtype=np.int64)
    object_present_mask = np.zeros((k, MAX_OBJECTS), dtype=bool)

    # Right-align: current frame at slot k-1.
    for slot_in_window, frame_idx in enumerate(real_indices, start=k - window_len):
        past_frame = scene_frames[frame_idx]

        if frame_idx == current_idx:
            # Current frame: the anchors ARE this frame's first
            # num_anchor slots — copy directly without re-walking nuScenes.
            vectors_window[slot_in_window] = np.asarray(
                past_frame["vectors"], dtype=np.float32
            )
            num_objects_window[slot_in_window] = int(past_frame["num_objects"])
            for j in range(num_anchor):
                if anchor_tokens[j]:
                    object_present_mask[slot_in_window, j] = True
        else:
            # Past frame: re-extract objects aligned to anchor identities
            # AND in the current ego frame (so positions are comparable
            # across the window).
            vecs, n_present, _, present_tokens = get_object_vectors_for_sample(
                nusc,
                past_frame["sample_token"],
                override_ego_t=cur_ego_t,
                override_ego_q=cur_ego_q,
                target_instance_tokens=anchor_tokens,
            )
            vectors_window[slot_in_window] = vecs
            num_objects_window[slot_in_window] = int(n_present)
            for j in range(num_anchor):
                if present_tokens[j]:
                    object_present_mask[slot_in_window, j] = True

    return vectors_window, num_objects_window, object_present_mask, window_len