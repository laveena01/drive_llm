# llm_driving/datasets_builder.py

"""
Builds datasets for:
1) Vector Captioning: (input: vector string) -> (target: lanGen caption)
2) Driving QA (paper-style): (input: caption + question + format) -> (target: actions + reason)

Key changes in this version:
- Robust min_dist computation using use_n = min(num_objects, MAX_OBJECTS)
- PAPER_FORMAT_INSTRUCTION aligned with training.py (stricter format, discourages questions)
- Store vec_str (needed for stage1_caption eval in training.py)
- Keep oracle caption only as debug field (oracle_caption_debug) to avoid confusion
"""

from typing import List, Dict
import json

from .nuscenes_data import get_scene_frames_vectors, init_nuscenes
from .langen import lanGen, vector_to_string
from .config import CAPTIONING_DATA_PATH, QA_DATA_PATH, MAX_OBJECTS
from .logger import get_logger

logger = get_logger("datasets_builder")


# ---------------------------
# Paper-style output template
# ---------------------------

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


def _policy_from_min_dist(num_objects: int, min_dist: float) -> tuple[int, int, str, str, str]:
    """
    Convert min_dist into proxy low-level controls (paper-like).
    Returns: (accel, brake, steer, reason, policy_label)
    """
    steer = "straight"

    if num_objects == 0:
        return 20, 0, steer, "No nearby obstacles detected.", "CONTINUE"

    if min_dist < 6.0:
        return 0, 70, steer, f"An object is very close ({min_dist:.1f} m), so brake hard.", "BRAKE"
    elif min_dist < 10.0:
        return 0, 40, steer, f"An object is close ({min_dist:.1f} m), so slow down.", "BRAKE"
    elif min_dist < 15.0:
        return 10, 10, steer, f"Objects are within caution range (min {min_dist:.1f} m), proceed carefully.", "CAUTION"
    else:
        return 20, 0, steer, f"All objects are far enough (min {min_dist:.1f} m), continue.", "CONTINUE"


def _make_samples_from_frames(
    frames: List[Dict],
    captioning_samples: List[Dict],
    qa_samples: List[Dict],
):
    logger.info(f"[datasets_builder]   Converting {len(frames)} frames into captioning + QA samples...")
    for idx, frame in enumerate(frames):
        num_objects = int(frame["num_objects"])
        caption = lanGen(frame)

        # NOTE: vec_str should only include up to MAX_OBJECTS because vectors are padded/truncated
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
            dists = [float(frame["vectors"][i][2]) for i in range(use_n)]  # index 2 assumed distance
            min_dist = min(dists)

        qa_question = "How should the car drive in this situation and why?"

        accel, brake, steer, reason, policy_label = _policy_from_min_dist(use_n, min_dist)
        qa_target = _paper_target(accel, brake, steer, reason)

        # Strong prompt structure
        qa_input = (
            "### OBSERVATION\n"
            f"{caption}\n"
            f"Minimum object distance: {min_dist:.1f} m\n\n"
            "### QUESTION\n"
            f"{qa_question}\n\n"
            "### OUTPUT FORMAT\n"
            f"{PAPER_FORMAT_INSTRUCTION}"
        )

        qa_samples.append({
            "input": qa_input,
            "target": qa_target,

            # Needed for paper-faithful eval: Stage1->Stage2
            "vec_str": vec_str,

            # Keep oracle caption only as a debug field (optional)
            "oracle_caption_debug": caption,

            # Optional metadata
            "min_dist": float(min_dist),
            "policy_label": policy_label,
            "use_n": int(use_n),
        })

        if (idx + 1) % 50 == 0:
            logger.info(f"[datasets_builder]     Processed {idx + 1}/{len(frames)} frames in this scene...")


def build_datasets_full_mini(
    max_frames_per_scene: int | None = None,
    captioning_path: str = CAPTIONING_DATA_PATH,
    qa_path: str = QA_DATA_PATH,
) -> tuple[list, list]:
    logger.info("[datasets_builder] Initializing nuScenes for full-mini dataset creation...")
    nusc = init_nuscenes()

    captioning_samples: list[dict] = []
    qa_samples: list[dict] = []

    num_scenes = len(nusc.scene)
    logger.info(f"[datasets_builder] Building data from all {num_scenes} scenes in nuScenes-mini.")
    logger.info(f"[datasets_builder]   max_frames_per_scene = {max_frames_per_scene}")

    for scene_idx in range(num_scenes):
        logger.info(f"\n[datasets_builder] Processing scene {scene_idx}/{num_scenes - 1}...")
        frames = get_scene_frames_vectors(
            nusc,
            scene_idx=scene_idx,
            max_frames=max_frames_per_scene,
        )
        logger.info(f"[datasets_builder]   Retrieved {len(frames)} frames from scene {scene_idx}.")
        _make_samples_from_frames(frames, captioning_samples, qa_samples)
        logger.info(f"[datasets_builder]   After scene {scene_idx}: "
              f"{len(captioning_samples)} captioning samples, {len(qa_samples)} QA samples.")

    logger.info(f"\n[datasets_builder] Saving captioning dataset to: {captioning_path}")
    with open(captioning_path, "w") as f:
        json.dump(captioning_samples, f, indent=2)

    logger.info(f"[datasets_builder] Saving QA dataset to: {qa_path}")
    with open(qa_path, "w") as f:
        json.dump(qa_samples, f, indent=2)

    logger.info(f"[datasets_builder] DONE. Captioning samples: {len(captioning_samples)} | QA samples: {len(qa_samples)}")
    return captioning_samples, qa_samples
