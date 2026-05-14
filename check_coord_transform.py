"""
Step 4 / M2 — coordinate transform sanity test.

Goal: verify `get_frame_window_tracked` (and the `override_ego_t/q` path in
`get_object_vectors_for_sample`) correctly re-expresses past-frame objects
in the current frame's ego coordinates.

Expected behaviour:
  - Stationary world object + stationary ego → (rel_x, rel_y) ≈ constant
    across all K frames (within ~0.5 m noise from annotation jitter).
  - Stationary world object + moving ego → (rel_x, rel_y) evolves smoothly
    (monotonic in the dominant motion axis), NOT jumping around.

If you see chaotic jumps in (rel_x, rel_y) for a parked car across K
frames, the override_ego transform is broken — STOP and fix before any
training.

Usage:
    NUSC_VERSION=v1.0-mini python check_coord_transform.py
    # or rely on config.py's NUSC_VERSION
"""

from __future__ import annotations

import sys
import numpy as np

# Force mini for fast inspection. If you want trainval, comment this out.
import os
os.environ.setdefault("NUSC_VERSION_OVERRIDE", "v1.0-mini")

from llm_driving.config import MAX_OBJECTS, VECTOR_DIM, TEMPORAL_WINDOW
from llm_driving.nuscenes_data import (
    init_nuscenes,
    get_scene_frames_vectors,
    get_frame_window_tracked,
)


def _print_object_track(window, present_mask, slot_j: int, label: str) -> None:
    K = window.shape[0]
    print(f"  [{label}] slot {slot_j}:")
    for k in range(K):
        if not bool(present_mask[k, slot_j]):
            print(f"    frame k={k}: <absent>")
            continue
        v = window[k, slot_j]
        rel_x = float(v[0])
        rel_y = float(v[1])
        dist = float(v[2])
        speed = float(np.sqrt(v[3] ** 2 + v[4] ** 2))
        print(
            f"    frame k={k}: rel_x={rel_x:+7.2f}m  rel_y={rel_y:+7.2f}m  "
            f"dist={dist:6.2f}m  speed={speed:5.2f}m/s"
        )


def main() -> None:
    print("[M2] Initialising nuScenes (mini)...")
    nusc = init_nuscenes()

    K = int(TEMPORAL_WINDOW)
    print(f"[M2] K (TEMPORAL_WINDOW) = {K}")

    scene_idx = 0
    print(f"[M2] Walking scene {scene_idx}, building tracked windows...")
    scene_frames = get_scene_frames_vectors(
        nusc, scene_idx=scene_idx, max_frames=None, temporal_window=K
    )
    print(f"[M2] Got {len(scene_frames)} frames in scene {scene_idx}.")

    # Find a mid-scene frame with a fully-populated window.
    mid_idx = None
    for i, f in enumerate(scene_frames):
        if int(f.get("window_len", 1)) == K and int(f["num_objects"]) >= 1:
            mid_idx = i
            break
    if mid_idx is None:
        print("[M2] No mid-scene frame with a full K-window found — aborting.")
        sys.exit(1)

    frame = scene_frames[mid_idx]
    window = np.asarray(frame["vectors_window"], dtype=np.float32)        # (K, M, D)
    present = np.asarray(frame["object_present_mask"], dtype=bool)         # (K, M)
    num_anchor = int(frame["num_objects"])
    ego_speed = float(frame.get("ego_speed", 0.0))

    print(
        f"\n[M2] Inspecting frame_in_scene={mid_idx}, "
        f"num_objects={num_anchor}, ego_speed={ego_speed:.2f} m/s"
    )

    # Identify candidates with consistent presence across all K frames.
    consistent_slots = [
        j for j in range(num_anchor)
        if all(bool(present[k, j]) for k in range(K))
    ]
    if not consistent_slots:
        print("[M2] No object visible in all K frames — picking the one in most frames.")
        # Fallback: pick the slot with the most True flags.
        slot_counts = [int(present[:, j].sum()) for j in range(num_anchor)]
        consistent_slots = [int(np.argmax(slot_counts))]

    # Find the most "stationary" candidate (smallest speed in current frame).
    cur = window[K - 1]
    speeds = [(j, float(np.sqrt(cur[j][3] ** 2 + cur[j][4] ** 2))) for j in consistent_slots]
    speeds.sort(key=lambda x: x[1])
    stationary_slot = speeds[0][0]

    # And the most "moving" candidate.
    moving_slot = speeds[-1][0] if len(speeds) > 1 else stationary_slot

    print(
        f"[M2] Stationary candidate: slot {stationary_slot} "
        f"(current-frame speed {speeds[0][1]:.2f} m/s)"
    )
    _print_object_track(window, present, stationary_slot, "STATIONARY")

    if moving_slot != stationary_slot:
        print(
            f"[M2] Moving candidate:     slot {moving_slot} "
            f"(current-frame speed {speeds[-1][1]:.2f} m/s)"
        )
        _print_object_track(window, present, moving_slot, "MOVING")

    # Verdict guidance.
    print("\n[M2] Verdict:")
    print("  ✓ Stationary track's (rel_x, rel_y) should evolve SMOOTHLY across K frames.")
    print("    If ego is stationary too → near-constant within ~0.5 m.")
    print("    If ego is moving        → straight-line trajectory in ego frame.")
    print("  ✗ If you see chaotic jumps, the override_ego_t/q transform is broken.")
    print("  ✓ Moving track's velocity should be consistent with its position delta.")


if __name__ == "__main__":
    main()
