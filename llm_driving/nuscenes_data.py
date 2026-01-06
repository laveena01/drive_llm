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
    """
    IMPORTANT:
    Some nuScenes-devkit versions try to load lidarseg/panoptic for v1.0-trainval
    and will crash if those folders are not present.

    We explicitly disable those optional tasks if the devkit supports the args.
    """
    print(f"[nuscenes_data] Initializing nuScenes with:")
    print(f"  - dataroot = {NUSC_ROOT}")
    print(f"  - version  = {NUSC_VERSION}")

    base_kwargs = dict(version=NUSC_VERSION, dataroot=NUSC_ROOT, verbose=True)

    # Try the most explicit signature first (newer devkit)
    try:
        nusc = NuScenes(**base_kwargs, lidarseg=False, panoptic=False)
        print("[nuscenes_data] nuScenes initialized (lidarseg=False, panoptic=False).\n")
        return nusc
    except TypeError:
        pass

    # Some devkit versions have lidarseg but not panoptic
    try:
        nusc = NuScenes(**base_kwargs, lidarseg=False)
        print("[nuscenes_data] nuScenes initialized (lidarseg=False).\n")
        return nusc
    except TypeError:
        pass

    # Fallback (old devkit) – may still crash if it *forces* lidarseg on trainval
    try:
        nusc = NuScenes(**base_kwargs)
        print("[nuscenes_data] nuScenes initialized.\n")
        return nusc
    except FileNotFoundError as e:
        # Give a very clear error if the devkit is forcing lidarseg without args support
        raise FileNotFoundError(
            f"{e}\n\n"
            "Your nuScenes devkit is trying to load lidarseg/panoptic labels, but they are missing.\n"
            "Fix options:\n"
            "  1) Install a newer nuscenes-devkit that supports lidarseg=False/panoptic=False, OR\n"
            "  2) Download the lidarseg (and/or panoptic) folders for your version.\n"
        )


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

        obj_t = np.array(ann["translation"], dtype=np.float32)
        obj_q = Quaternion(ann["rotation"])

        rel_global = obj_t - ego_t
        rel_ego = ego_q_inv.rotate(rel_global)

        rel_x = float(rel_ego[0])
        rel_y = float(rel_ego[1])
        dist = float(np.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-6))

        rel_speed = 0.0
        try:
            vx, vy, vz = nusc.box_velocity(ann_token)
            if not np.any(np.isnan([vx, vy, vz])):
                v_global = np.array([vx, vy, vz], dtype=np.float32)
                v_ego = ego_q_inv.rotate(v_global)
                rel_speed = float(np.linalg.norm(v_ego[:2]))
        except Exception:
            rel_speed = 0.0

        obj_q_ego = ego_q_inv * obj_q
        heading = _yaw_from_quaternion(obj_q_ego)

        w, l, h = ann["size"]
        size = float((float(w) + float(l)) / 2.0)

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
