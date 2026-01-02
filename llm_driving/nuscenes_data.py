# llm_driving/nuscenes_data.py

"""
Utilities for loading nuScenes and extracting object-level vectors.

Vector format per object:
[rel_x, rel_y, dist, rel_speed, heading, size, type_id]

Where:
- rel_x, rel_y are in EGO frame (meters)
- dist is sqrt(rel_x^2 + rel_y^2)
- rel_speed is magnitude (m/s) (ego-frame magnitude)
- heading is yaw angle (radians) of the object's box in ego frame
- size is avg(length, width) proxy (meters)
- type_id: 0 car, 1 pedestrian, 2 traffic_light, 3 other
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from .config import NUSC_ROOT, NUSC_VERSION, MAX_OBJECTS, VECTOR_DIM


def init_nuscenes() -> NuScenes:
    print(f"[nuscenes_data] Initializing nuScenes with:")
    print(f"  - dataroot = {NUSC_ROOT}")
    print(f"  - version  = {NUSC_VERSION}")
    nusc = NuScenes(version=NUSC_VERSION, dataroot=NUSC_ROOT, verbose=True)
    print("[nuscenes_data] nuScenes initialized successfully.\n")
    return nusc


def _yaw_from_quaternion(q: Quaternion) -> float:
    """
    Return yaw (rotation around z) from quaternion.
    nuScenes uses (w, x, y, z).
    """
    # Quaternion yaw extraction:
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    w, x, y, z = q.w, q.x, q.y, q.z
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def get_object_vectors_for_sample(
    nusc: NuScenes,
    sample_token: str,
) -> Tuple[np.ndarray, int]:
    """
    Extract object vectors for a given sample token.
    Uses ego pose (LIDAR_TOP) to convert global boxes -> ego frame.
    """
    sample = nusc.get("sample", sample_token)

    # Use LIDAR_TOP sample_data for ego pose (standard in nuScenes)
    lidar_sd_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_sd_token)
    ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])

    ego_t = np.array(ego_pose["translation"], dtype=np.float32)  # (x,y,z)
    ego_q = Quaternion(ego_pose["rotation"])  # (w,x,y,z)
    ego_q_inv = ego_q.inverse

    vectors: List[List[float]] = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)

        # --- Global box translation/orientation ---
        obj_t = np.array(ann["translation"], dtype=np.float32)
        obj_q = Quaternion(ann["rotation"])

        # --- Transform position: global -> ego frame ---
        rel_global = obj_t - ego_t  # still global axes
        rel_ego = ego_q_inv.rotate(rel_global)  # now ego axes

        rel_x = float(rel_ego[0])
        rel_y = float(rel_ego[1])
        dist = float(np.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-6))

        # --- Velocity (global) -> ego frame magnitude ---
        rel_speed = 0.0
        try:
            vx, vy, vz = nusc.box_velocity(ann_token)
            if not np.any(np.isnan([vx, vy, vz])):
                v_global = np.array([vx, vy, vz], dtype=np.float32)
                v_ego = ego_q_inv.rotate(v_global)
                rel_speed = float(np.linalg.norm(v_ego[:2]))  # horizontal speed
        except Exception:
            rel_speed = 0.0

        # --- Heading: object yaw in ego frame ---
        # Convert object orientation into ego frame: q_ego_inv * q_obj
        obj_q_ego = ego_q_inv * obj_q
        heading = _yaw_from_quaternion(obj_q_ego)  # radians

        # --- Size proxy ---
        # ann["size"] = [w, l, h] in nuScenes (width, length, height)
        # We'll use avg of width & length (stable scalar)
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

        vectors.append([rel_x, rel_y, dist, rel_speed, heading, size, float(type_id)])

    # Sort by distance so MAX_OBJECTS are the nearest ones (much more stable)
    vectors.sort(key=lambda v: v[2])

    padded = np.zeros((MAX_OBJECTS, VECTOR_DIM), dtype=np.float32)
    count = min(len(vectors), MAX_OBJECTS)
    if count > 0:
        padded[:count, :] = np.array(vectors[:count], dtype=np.float32)

    return padded, count


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
        "sample_token": str
      }
    """
    scene = nusc.scene[scene_idx]
    token = scene["first_sample_token"]

    print(f"[nuscenes_data] Collecting frames for scene {scene_idx} (token={scene['token']})")

    frames: List[Dict] = []
    frame_idx = 0

    while token:
        if max_frames is not None and frame_idx >= max_frames:
            break

        vecs, num_obj = get_object_vectors_for_sample(nusc, token)
        frames.append(
            {
                "vectors": vecs,
                "num_objects": num_obj,
                "sample_token": token,
            }
        )

        sample = nusc.get("sample", token)
        token = sample["next"]
        frame_idx += 1

        if frame_idx % 20 == 0:
            print(f"[nuscenes_data]  Processed {frame_idx} frames in scene {scene_idx}...")

    print(f"[nuscenes_data] Extracted {len(frames)} frames for scene {scene_idx}")
    return frames
