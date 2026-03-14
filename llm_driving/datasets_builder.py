# llm_driving/datasets_builder.py

"""
Builds datasets for:
1) Vector Captioning: (input: vector string) -> (target: lanGen caption) [NO risk in Stage-1]
2) Driving QA (paper-style): (input: caption + risk + question + format) -> target depends on question type

Phase-3 update:
- Adds multiple questions per frame, split into:
  A) ACTION questions -> 5-line output (accelerator/brake/steer/reason)
  B) RISK questions   -> 1-2 line output: "Risk level: ... Reason: ..."
- Keeps Phase-2 cleanup: DO NOT leak risk into lanGen captions (Option-B)
"""

from typing import List, Dict
from collections import Counter
import json
import logging
import re

from .nuscenes_data import get_scene_frames_vectors, init_nuscenes
from .langen import lanGen, vector_to_string
from .config import CAPTIONING_DATA_PATH, QA_DATA_PATH, MAX_OBJECTS
from llm_driving.risk_calculator import calculate_risk_from_vectors, get_risk_summary_text, policy_from_risk

logger = logging.getLogger("llm_driving")

# Track policy decisions for logging (per-frame, not per-question)
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

RISK_FORMAT_INSTRUCTION = (
    "Answer in 1-2 short lines using this template ONLY:\n"
    "Risk level: <CRITICAL|HIGH|MODERATE|LOW|MINIMAL>.\n"
    "Reason: <brief; mention TTC/collision/pedestrian if relevant>.\n"
    "Do NOT include driving controls.\n"
)

# -----------------------------
# Phase-3 question sets
# -----------------------------
ACTION_QUESTIONS: List[str] = [
    "How should the car drive in this situation and why?",
    "Should the car brake now or continue?",
    "What brake percentage should be applied and why?",
    "Should the ego slow down, maintain speed, or speed up?",
    "What caution should the vehicle take in the next 2 seconds?",
]

RISK_QUESTIONS: List[str] = [
    "What is the risk level and the main reason?",
    "Is there an imminent collision risk (low TTC)? Explain briefly.",
    "Are pedestrians a significant risk here? Explain briefly.",
]


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


def _strip_any_risk_lines_from_caption(caption: str) -> str:
    """Extra safety guard: strip risk lines if they ever appear in caption."""
    if not caption:
        return caption
    kept = []
    for ln in caption.splitlines():
        low = ln.lower().strip()
        if low.startswith("risk assessment:") or low.startswith("risk level:") or low.startswith("time-to-collision"):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _risk_target_from_risk_data(risk_data) -> str:
    """
    Create a stable short answer for risk questions.
    Keep it simple so evaluation is easy and thesis-friendly.
    """
    rl = getattr(risk_data, "risk_level", "UNKNOWN")
    min_ttc = getattr(risk_data, "min_ttc", None)
    max_col = float(getattr(risk_data, "max_collision_risk", 0.0))
    max_ped = float(getattr(risk_data, "max_pedestrian_risk", 0.0))

    reasons = []
    if min_ttc is not None:
        try:
            if float(min_ttc) <= 3.0:
                reasons.append(f"TTC={float(min_ttc):.1f}s")
        except Exception:
            pass
    if max_col >= 0.30:
        reasons.append(f"collision={max_col:.0%}")
    if max_ped >= 0.30:
        reasons.append(f"ped={max_ped:.0%}")
    if not reasons:
        reasons.append("no strong risk factors")

    # 1-2 lines max
    return f"Risk level: {rl}.\nReason: " + ", ".join(reasons) + ".\n"


def _make_samples_from_frames(
    frames: List[Dict],
    captioning_samples: List[Dict],
    qa_samples: List[Dict],
    scene_idx: int = 0,
):
    logger.info(f"[datasets_builder]   Converting {len(frames)} frames into captioning + QA samples...")
    for idx, frame in enumerate(frames):
        num_objects = int(frame["num_objects"])
        use_n = min(num_objects, MAX_OBJECTS)

        # Risk (Stage-2 only)
        risk_data = calculate_risk_from_vectors(
            vectors=frame["vectors"],
            use_n=use_n,
            ego_speed=None,
            traffic_light=None,
        )
        risk_text = get_risk_summary_text(risk_data)

        # IMPORTANT: caption must NOT see risk_data (Option-B)
        frame_for_caption = dict(frame)
        frame_for_caption.pop("risk_data", None)
        caption = lanGen(frame_for_caption)
        caption = _strip_any_risk_lines_from_caption(caption)

        vec_str = vector_to_string(frame["vectors"], num_objects)

        # --- Stage 1: vector -> caption (caption-only target) ---
        captioning_samples.append({
            "input": f"Describe the driving scene from object vectors:\n{vec_str}",
            "target": caption,
        })

        # --- metadata: min_dist ---
        if use_n == 0:
            min_dist = 999.0
        else:
            dists = [float(frame["vectors"][i][2]) for i in range(use_n)]
            min_dist = float(min(dists))

        # --- action policy from risk ---
        accel, brake, steer, reason, policy_label = policy_from_risk(risk_data)
        qa_target_action = _paper_target(accel, brake, steer, reason)

        # frame-level index (stable across multiple questions)
        frame_idx_global = len(_policy_log)

        # log per-frame once
        _policy_log.append({
            "frame_idx": frame_idx_global,
            "num_objects": int(use_n),
            "min_dist": float(min_dist),
            "risk_level": str(getattr(risk_data, "risk_level", "UNKNOWN")),
            "decision": str(policy_label),
            "accel": int(accel),
            "brake": int(brake),
            "scene_idx": int(scene_idx),
            "frame_in_scene": int(idx),
        })

        if idx % 10 == 0:
            logger.info(
                f"    [RISK] Scene {scene_idx} Frame {idx}: objects={use_n}, "
                f"min_dist={min_dist:.1f}m -> {getattr(risk_data,'risk_level','?')} -> {policy_label}"
            )

        # --- Stage 2 samples: ACTION questions ---
        for qid, qa_question in enumerate(ACTION_QUESTIONS):
            qa_input = (
                "### OBSERVATION\n"
                f"{caption}\n\n"
                "### RISK\n"
                f"{risk_text}\n\n"
                "### QUESTION\n"
                f"{qa_question}\n\n"
                "### OUTPUT FORMAT\n"
                f"{PAPER_FORMAT_INSTRUCTION}"
            )

            qa_samples.append({
                "input": qa_input,
                "target": qa_target_action,

                "question_type": "action",
                "question_id": int(qid),
                "question": qa_question,

                "vec_str": vec_str,
                "oracle_caption_debug": caption,
                "risk_text": risk_text,
                "risk_level": str(getattr(risk_data, "risk_level", "UNKNOWN")),

                "min_dist": float(min_dist),
                "policy_label": str(policy_label),
                "use_n": int(use_n),

                "frame_idx": int(frame_idx_global),
                "scene_idx": int(scene_idx),
                "frame_in_scene": int(idx),
            })

        # --- Stage 2 samples: RISK questions ---
        risk_target = _risk_target_from_risk_data(risk_data)
        for rqid, qa_question in enumerate(RISK_QUESTIONS):
            qa_input = (
                "### OBSERVATION\n"
                f"{caption}\n\n"
                "### RISK\n"
                f"{risk_text}\n\n"
                "### QUESTION\n"
                f"{qa_question}\n\n"
                "### OUTPUT FORMAT\n"
                f"{RISK_FORMAT_INSTRUCTION}"
            )

            qa_samples.append({
                "input": qa_input,
                "target": risk_target,

                "question_type": "risk",
                "question_id": int(len(ACTION_QUESTIONS) + rqid),
                "question": qa_question,

                "vec_str": vec_str,
                "oracle_caption_debug": caption,
                "risk_text": risk_text,
                "risk_level": str(getattr(risk_data, "risk_level", "UNKNOWN")),

                "min_dist": float(min_dist),
                "policy_label": str(policy_label),
                "use_n": int(use_n),

                "frame_idx": int(frame_idx_global),
                "scene_idx": int(scene_idx),
                "frame_in_scene": int(idx),
            })

        if (idx + 1) % 50 == 0:
            logger.info(f"[datasets_builder]     Processed {idx + 1}/{len(frames)} frames in this scene...")


def _print_policy_summary():
    """Print a summary of all frame-level decisions (not per-question)."""
    if not _policy_log:
        logger.info("[POLICY SUMMARY] No policy decisions logged.")
        return

    logger.info("\n" + "=" * 80)
    logger.info("[POLICY SUMMARY] Risk Assessment Outcomes (per-frame)")
    logger.info("=" * 80)

    risk_counts = Counter(p["risk_level"] for p in _policy_log)
    decision_counts = Counter(p["decision"] for p in _policy_log)

    total = len(_policy_log)
    logger.info(f"\nTotal frames processed: {total}")

    logger.info("\n--- Risk Level Distribution ---")
    for level in ["CRITICAL", "HIGH", "MODERATE", "LOW", "MINIMAL"]:
        count = risk_counts.get(level, 0)
        pct = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        logger.info(f"  {level:10s}: {count:4d} ({pct:5.1f}%) {bar}")

    logger.info("\n--- Decision Distribution ---")
    for decision in ["BRAKE", "CAUTION", "CONTINUE"]:
        count = decision_counts.get(decision, 0)
        pct = (count / total) * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        logger.info(f"  {decision:10s}: {count:4d} ({pct:5.1f}%) {bar}")

    logger.info("=" * 80 + "\n")


def build_datasets_full_mini(
    max_frames_per_scene: int | None = None,
    captioning_path: str = CAPTIONING_DATA_PATH,
    qa_path: str = QA_DATA_PATH,
) -> tuple[list, list]:
    global _policy_log
    _policy_log = []  # Reset for fresh run

    logger.info("[datasets_builder] Initializing nuScenes for full-mini dataset creation...")
    nusc = init_nuscenes()

    captioning_samples: list[dict] = []
    qa_samples: list[dict] = []

    num_scenes = len(nusc.scene)
    logger.info(f"[datasets_builder] Building data from all {num_scenes} scenes in nuScenes-mini.")
    logger.info(f"[datasets_builder]   max_frames_per_scene = {max_frames_per_scene}")
    logger.info(f"[datasets_builder]   action_questions={len(ACTION_QUESTIONS)} risk_questions={len(RISK_QUESTIONS)}")

    for scene_idx in range(num_scenes):
        logger.info(f"\n[datasets_builder] Processing scene {scene_idx}/{num_scenes - 1}...")
        frames = get_scene_frames_vectors(
            nusc,
            scene_idx=scene_idx,
            max_frames=max_frames_per_scene,
        )
        logger.info(f"[datasets_builder]   Retrieved {len(frames)} frames from scene {scene_idx}.")
        _make_samples_from_frames(frames, captioning_samples, qa_samples, scene_idx=scene_idx)
        logger.info(
            f"[datasets_builder]   After scene {scene_idx}: "
            f"{len(captioning_samples)} captioning samples, {len(qa_samples)} QA samples."
        )

    _print_policy_summary()

    logger.info(f"\n[datasets_builder] Saving captioning dataset to: {captioning_path}")
    with open(captioning_path, "w") as f:
        json.dump(captioning_samples, f, indent=2)

    logger.info(f"[datasets_builder] Saving QA dataset to: {qa_path}")
    with open(qa_path, "w") as f:
        json.dump(qa_samples, f, indent=2)

    frames_n = len(_policy_log)
    qpf = len(ACTION_QUESTIONS) + len(RISK_QUESTIONS)
    logger.info(
        f"[datasets_builder] DONE. Captioning samples: {len(captioning_samples)} | "
        f"QA samples: {len(qa_samples)} (= {frames_n} frames × {qpf} questions)"
    )
    return captioning_samples, qa_samples