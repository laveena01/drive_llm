# llm_driving/datasets_builder.py

"""
Builds datasets for:
1) Vector Captioning: (input: vector string) -> (target: lanGen caption)
2) Driving QA (paper-style): (input: caption + question + format) -> (target: actions + reason)

PART-1 updates:
- Remove min_dist leakage from Stage-2 input (still stored as metadata)
- Keep vec_str for stage1_caption eval
- Keep oracle caption only as debug field
"""

from typing import List, Dict
from collections import Counter
import json

from .nuscenes_data import get_scene_frames_vectors, init_nuscenes
from .langen import lanGen, vector_to_string
from .config import CAPTIONING_DATA_PATH, QA_DATA_PATH, MAX_OBJECTS


# Track policy decisions for logging
_policy_log: List[Dict] = []


PAPER_FORMAT_INSTRUCTION = (
    "You are an AI Driver.\n"
    "Return EXACTLY 5 lines (each on its own line), and nothing else:\n"
    "Here are my actions:\n"
    "- Accelerator pedal: <0-100>%\n"
    "- Brake pedal: <0-100>%\n"
    "- Steering: <left/straight/right>\n"
    "Reason: <one short sentence>\n"
    "Do NOT ask questions. Do NOT add extra text.\n"
)


def _paper_target(accel: int, brake: int, steer: str, reason: str) -> str:
    accel = int(max(0, min(100, accel)))
    brake = int(max(0, min(100, brake)))
    if steer not in ("left", "straight", "right"):
        steer = "straight"
    return (
        "Here are my actions:\n"
        f"- Accelerator pedal: {accel}%\n"
        f"- Brake pedal: {brake}%\n"
        f"- Steering: {steer}\n"
        f"Reason: {reason}\n"
    )


def _policy_from_min_dist(num_objects: int, min_dist: float, frame_idx: int = -1) -> tuple[int, int, str, str, str]:
    """
    Rule-based policy to determine driving action based on minimum distance to objects.

    Risk levels:
    - BRAKE (HIGH RISK):    min_dist < 10m  -> brake hard (70%) or slow down (40%)
    - CAUTION (MED RISK):   10m <= min_dist < 15m -> light brake + accel
    - CONTINUE (LOW RISK):  min_dist >= 15m or no objects -> normal driving
    """
    steer = "straight"

    if num_objects == 0:
        risk_level = "LOW"
        decision = "CONTINUE"
        accel, brake = 20, 0
        reason = "No nearby obstacles detected."
    elif min_dist < 6.0:
        risk_level = "CRITICAL"
        decision = "BRAKE"
        accel, brake = 0, 70
        reason = f"An object is very close ({min_dist:.1f} m), so brake hard."
    elif min_dist < 10.0:
        risk_level = "HIGH"
        decision = "BRAKE"
        accel, brake = 0, 40
        reason = f"An object is close ({min_dist:.1f} m), so slow down."
    elif min_dist < 15.0:
        risk_level = "MEDIUM"
        decision = "CAUTION"
        accel, brake = 10, 10
        reason = f"Objects are within caution range (min {min_dist:.1f} m), proceed carefully."
    else:
        risk_level = "LOW"
        decision = "CONTINUE"
        accel, brake = 20, 0
        reason = f"All objects are far enough (min {min_dist:.1f} m), continue."

    # Log the policy decision
    _policy_log.append({
        "frame_idx": frame_idx,
        "num_objects": num_objects,
        "min_dist": min_dist,
        "risk_level": risk_level,
        "decision": decision,
        "accel": accel,
        "brake": brake,
    })

    return accel, brake, steer, reason, decision


def _make_samples_from_frames(
    frames: List[Dict],
    captioning_samples: List[Dict],
    qa_samples: List[Dict],
    scene_idx: int = 0,
):
    print(f"[datasets_builder]   Converting {len(frames)} frames into captioning + QA samples...")
    for idx, frame in enumerate(frames):
        num_objects = int(frame["num_objects"])
        caption = lanGen(frame)

        vec_str = vector_to_string(frame["vectors"], num_objects)

        # --- Stage 1: vector -> caption ---
        captioning_samples.append({
            "input": f"Describe the driving scene from object vectors:\n{vec_str}",
            "target": caption,
        })

        # --- Stage 2: paper-style actions ---
        use_n = min(num_objects, MAX_OBJECTS)
        if use_n == 0:
            min_dist = 999.0
        else:
            dists = [float(frame["vectors"][i][2]) for i in range(use_n)]
            min_dist = min(dists)

        qa_question = "How should the car drive in this situation and why?"

        global_frame_idx = len(qa_samples)  # unique frame index across all scenes
        accel, brake, steer, reason, policy_label = _policy_from_min_dist(use_n, min_dist, frame_idx=global_frame_idx)
        qa_target = _paper_target(accel, brake, steer, reason)

        # Log individual risk decisions (every 10th frame to avoid spam)
        if idx % 10 == 0:
            risk = _policy_log[-1]["risk_level"] if _policy_log else "?"
            print(f"    [RISK] Scene {scene_idx} Frame {idx}: objects={use_n}, min_dist={min_dist:.1f}m -> {risk} risk -> {policy_label}")

        # PART-1: remove min_dist from the Stage-2 input (keep only caption + question + format)
        qa_input = (
            "### OBSERVATION\n"
            f"{caption}\n\n"
            "### QUESTION\n"
            f"{qa_question}\n\n"
            "### OUTPUT FORMAT\n"
            f"{PAPER_FORMAT_INSTRUCTION}"
        )

        qa_samples.append({
            "input": qa_input,
            "target": qa_target,

            # Needed for stage1_caption eval: Stage1(vec_str)->caption_pred->Stage2
            "vec_str": vec_str,

            # Debug only
            "oracle_caption_debug": caption,

            # Metadata (not leaked into prompt)
            "min_dist": float(min_dist),
            "policy_label": policy_label,
            "use_n": int(use_n),
        })

        if (idx + 1) % 50 == 0:
            print(f"[datasets_builder]     Processed {idx + 1}/{len(frames)} frames in this scene...")


def _print_policy_summary():
    """Print a summary of all policy decisions made during dataset building."""
    if not _policy_log:
        print("[POLICY SUMMARY] No policy decisions logged.")
        return

    print("\n" + "=" * 80)
    print("[POLICY SUMMARY] Risk Assessment Outcomes")
    print("=" * 80)

    # Count by risk level
    risk_counts = Counter(p["risk_level"] for p in _policy_log)
    decision_counts = Counter(p["decision"] for p in _policy_log)

    total = len(_policy_log)
    print(f"\nTotal frames processed: {total}")

    print("\n--- Risk Level Distribution ---")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = risk_counts.get(level, 0)
        pct = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {level:10s}: {count:4d} ({pct:5.1f}%) {bar}")

    print("\n--- Decision Distribution ---")
    for decision in ["BRAKE", "CAUTION", "CONTINUE"]:
        count = decision_counts.get(decision, 0)
        pct = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {decision:10s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Distance statistics
    dists = [p["min_dist"] for p in _policy_log if p["min_dist"] < 900]
    if dists:
        import numpy as np
        print("\n--- Distance Statistics (objects present) ---")
        print(f"  Min distance:  {min(dists):.2f} m")
        print(f"  Max distance:  {max(dists):.2f} m")
        print(f"  Mean distance: {np.mean(dists):.2f} m")
        print(f"  Median distance: {np.median(dists):.2f} m")

    # Show some example critical/high risk frames
    critical_frames = [p for p in _policy_log if p["risk_level"] in ["CRITICAL", "HIGH"]]
    if critical_frames:
        print(f"\n--- Sample High-Risk Frames (showing up to 5) ---")
        for p in critical_frames[:5]:
            print(f"  Frame {p['frame_idx']}: {p['num_objects']} objects, min_dist={p['min_dist']:.1f}m "
                  f"-> {p['risk_level']} -> Brake={p['brake']}%")

    print("=" * 80 + "\n")


def build_datasets_full_mini(
    max_frames_per_scene: int | None = None,
    captioning_path: str = CAPTIONING_DATA_PATH,
    qa_path: str = QA_DATA_PATH,
) -> tuple[list, list]:
    global _policy_log
    _policy_log = []  # Reset for fresh run

    print("[datasets_builder] Initializing nuScenes for full-mini dataset creation...")
    nusc = init_nuscenes()

    captioning_samples: list[dict] = []
    qa_samples: list[dict] = []

    num_scenes = len(nusc.scene)
    print(f"[datasets_builder] Building data from all {num_scenes} scenes in nuScenes-mini.")
    print(f"[datasets_builder]   max_frames_per_scene = {max_frames_per_scene}")

    for scene_idx in range(num_scenes):
        print(f"\n[datasets_builder] Processing scene {scene_idx}/{num_scenes - 1}...")
        frames = get_scene_frames_vectors(
            nusc,
            scene_idx=scene_idx,
            max_frames=max_frames_per_scene,
        )
        print(f"[datasets_builder]   Retrieved {len(frames)} frames from scene {scene_idx}.")
        _make_samples_from_frames(frames, captioning_samples, qa_samples, scene_idx=scene_idx)
        print(f"[datasets_builder]   After scene {scene_idx}: "
              f"{len(captioning_samples)} captioning samples, {len(qa_samples)} QA samples.")

    # Print risk assessment summary
    _print_policy_summary()

    print(f"\n[datasets_builder] Saving captioning dataset to: {captioning_path}")
    with open(captioning_path, "w") as f:
        json.dump(captioning_samples, f, indent=2)

    print(f"[datasets_builder] Saving QA dataset to: {qa_path}")
    with open(qa_path, "w") as f:
        json.dump(qa_samples, f, indent=2)

    print(f"[datasets_builder] DONE. Captioning samples: {len(captioning_samples)} | QA samples: {len(qa_samples)}")
    return captioning_samples, qa_samples
