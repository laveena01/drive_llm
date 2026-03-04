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
) -> Tuple[np.ndarray, int, List[str]]:
    """
    Extract object vectors for a given sample token.
    Uses ego pose (LIDAR_TOP) to convert global boxes -> ego frame.

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


def get_scene_frames_vectors(
    nusc: NuScenes,
    scene_idx: int = 0,
    max_frames: Optional[int] = None,
) -> List[Dict]:
    """
    Returns list of frames, each as:
      {
        "vectors": np.ndarray(MAX_OBJECTS, VECTOR_DIM),
        "num_objects": int,
        "sample_token": str,
        "categories": List[str]
      }
    """
    scene = nusc.scene[scene_idx]
    token = scene["first_sample_token"]

    logger.info(
        f"[nuscenes_data] Collecting frames for scene {scene_idx} "
        f"(scene_token={scene['token']}, max_frames={max_frames})"
    )

    frames: List[Dict] = []
    frame_idx = 0

    while token:
        if max_frames is not None and frame_idx >= max_frames:
            break

        vecs, num_obj, categories = get_object_vectors_for_sample(nusc, token)
        frames.append(
            {
                "vectors": vecs,
                "num_objects": num_obj,
                "sample_token": token,
                "categories": categories,
            }
        )

        sample = nusc.get("sample", token)
        token = sample["next"]
        frame_idx += 1

        if frame_idx % 20 == 0:
            logger.info(f"[nuscenes_data]  Processed {frame_idx} frames in scene {scene_idx}...")

    logger.info(f"[nuscenes_data] Extracted {len(frames)} frames for scene {scene_idx}")
    return frames