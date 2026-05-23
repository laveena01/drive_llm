"""
counterfactual_eval.py — perturbation-based robustness eval for v2 models.

For each val sample, apply N perturbations (object remove, threat inject,
TTC halve, risk-level swap, pedestrian mask, ego-speed double), re-run
inference, and record (base_brake, pert_brake, base_action, pert_action).

Sharded for multi-GPU. Output: runs/<RUN>/counterfactual/counterfactual_shard{i}.json
"""
import argparse
import json
import os
import sys
from typing import Optional, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_driving.training import (
    _build_stage2_prompt_from_caption,
    _extract_brake_percent,
    _extract_accel_percent,
    _map_text_to_action_label,
    enforce_5_lines,
)
from llm_driving.risk_calculator import (
    calculate_risk_from_vectors,
    get_risk_summary_text,
)
from llm_driving import config as cfg

PERTURBATIONS = [
    "P1_remove_closest",
    "P2_inject_threat",
    "P3_halve_ttc",
    "P4_swap_risk_level",
    "P5_mask_pedestrians",
    "P6_double_ego_speed",
]


def perturb(sample: dict, kind: str) -> dict:
    """Return a NEW sample with the perturbation applied."""
    s = dict(sample)
    vectors = np.array(s["vectors"], dtype=np.float32)
    use_n = int(s.get("num_objects", 0))
    ego_speed = float(s.get("ego_speed", 0.0) or 0.0)
    s["_perturbation_applied"] = True
    s["_p4_swap_direction"] = None

    if kind == "P1_remove_closest":
        if use_n == 0:
            s["_perturbation_applied"] = False
            return s
        dists = vectors[:use_n, 2]
        idx = int(np.argmin(dists))
        kept = np.delete(vectors[:use_n], idx, axis=0)
        new_vec = np.zeros_like(vectors)
        new_vec[: kept.shape[0]] = kept
        use_n -= 1
        s["vectors"] = new_vec.tolist()
        s["num_objects"] = use_n
        rd = calculate_risk_from_vectors(new_vec, use_n=use_n, ego_speed=ego_speed)
        s["risk_text"] = get_risk_summary_text(rd)

    elif kind == "P2_inject_threat":
        if use_n >= 10:
            s["_perturbation_applied"] = False
            return s
        # car at 4m straight ahead closing at 5 m/s; type_id=0 (car)
        threat = np.array([4.0, 0.0, 4.0, -5.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        new_vec = vectors.copy()
        new_vec[use_n] = threat
        use_n += 1
        s["vectors"] = new_vec.tolist()
        s["num_objects"] = use_n
        rd = calculate_risk_from_vectors(new_vec, use_n=use_n, ego_speed=ego_speed)
        s["risk_text"] = get_risk_summary_text(rd)

    elif kind == "P3_halve_ttc":
        # Pretend ego is twice as fast — apparent TTC halves
        rd = calculate_risk_from_vectors(vectors, use_n=use_n, ego_speed=ego_speed * 2)
        s["risk_text"] = get_risk_summary_text(rd)

    elif kind == "P4_swap_risk_level":
        rt = s.get("risk_text", "") or ""
        if "Risk level: LOW" in rt:
            s["risk_text"] = rt.replace("Risk level: LOW", "Risk level: CRITICAL", 1)
            s["_p4_swap_direction"] = "low_to_critical"
        elif "Risk level: CRITICAL" in rt:
            s["risk_text"] = rt.replace("Risk level: CRITICAL", "Risk level: LOW", 1)
            s["_p4_swap_direction"] = "critical_to_low"
        else:
            s["_perturbation_applied"] = False

    elif kind == "P5_mask_pedestrians":
        if use_n == 0:
            s["_perturbation_applied"] = False
            return s
        ped_mask = vectors[:use_n, 7] == 1
        if not ped_mask.any():
            s["_perturbation_applied"] = False
            return s
        kept = vectors[:use_n][~ped_mask]
        new_use_n = int(kept.shape[0])
        new_vec = np.zeros_like(vectors)
        new_vec[:new_use_n] = kept
        s["vectors"] = new_vec.tolist()
        s["num_objects"] = new_use_n
        rd = calculate_risk_from_vectors(new_vec, use_n=new_use_n, ego_speed=ego_speed)
        s["risk_text"] = get_risk_summary_text(rd)

    elif kind == "P6_double_ego_speed":
        new_speed = ego_speed * 2 if ego_speed > 0 else 20.0
        s["ego_speed"] = new_speed
        rd = calculate_risk_from_vectors(vectors, use_n=use_n, ego_speed=new_speed)
        s["risk_text"] = get_risk_summary_text(rd)

    else:
        raise ValueError(f"Unknown perturbation: {kind}")

    return s


def run_inference(model, tokenizer, sample: dict, device: str, max_new_tokens: int):
    caption = sample.get("oracle_caption_debug", "") or ""
    risk_text = sample.get("risk_text", "") or ""
    qa_question = sample.get("question", "") or ""
    question_type = sample.get("question_type", "action") or "action"
    prompt = _build_stage2_prompt_from_caption(
        caption=caption,
        risk_text=risk_text,
        qa_question=qa_question,
        question_type=question_type,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=getattr(cfg, "STAGE2_MAX_INPUT_LEN", 384),
    ).to(device)
    # R7 v2: when Stage 2 is VectorPrefixT5, also pass the sample's (possibly
    # perturbed) vectors so the vector channel reflects the perturbation.
    # Under R2 v2 baseline, vector perturbations only reach Stage 2 via the
    # (stale) caption + risk_text; under R7 they additionally flow through
    # the vector prefix encoder — see plan section "Counterfactual semantics
    # under R7".
    extra_kwargs = {}
    if getattr(cfg, "USE_STAGE2_VECTOR_PREFIX", False):
        vectors = sample.get("vectors")
        if vectors is None:
            vectors = np.zeros((cfg.MAX_OBJECTS, cfg.VECTOR_DIM), dtype=np.float32).tolist()
        n_obj = int(sample.get("num_objects", 0))
        vectors_t = torch.tensor([vectors], dtype=torch.float32, device=device)
        num_obj_t = torch.tensor([n_obj], dtype=torch.long, device=device)
        extra_kwargs["vectors"] = vectors_t
        extra_kwargs["num_objects"] = num_obj_t
    with torch.no_grad():
        out = model.generate(
            **inputs,
            **extra_kwargs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=8,
            repetition_penalty=1.5,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # enforce_5_lines returns (str, int) — unpack and keep only the cleaned text.
    if text:
        fixed_out = enforce_5_lines(text)
        fixed = fixed_out[0] if isinstance(fixed_out, tuple) else fixed_out
    else:
        fixed = text
    return {
        "raw": text,
        "fixed": fixed,
        "brake": _extract_brake_percent(fixed),
        "accel": _extract_accel_percent(fixed),
        "action_label": _map_text_to_action_label(fixed),
    }


def load_val_samples(qa_path: str) -> List[dict]:
    """Load val samples. Prefer 'split' field; else use scene-aware split."""
    data = json.load(open(qa_path))
    # Try the explicit split field first
    val = [s for s in data if s.get("split") == "val"]
    if not val:
        # Fall back: replicate scene-aware split using SCENE_LEVEL_SPLIT_SEED
        from sklearn.model_selection import train_test_split
        all_scene_idx = sorted({int(s["scene_idx"]) for s in data if "scene_idx" in s})
        if not all_scene_idx:
            raise RuntimeError("No 'split' field on samples and no 'scene_idx' to derive split")
        _, val_scenes = train_test_split(
            all_scene_idx,
            test_size=getattr(cfg, "SCENE_LEVEL_SPLIT_TEST_SIZE", 0.2),
            random_state=getattr(cfg, "SCENE_LEVEL_SPLIT_SEED", 42),
        )
        val_set = set(val_scenes)
        val = [s for s in data if int(s.get("scene_idx", -1)) in val_set]
    # Filter to action questions only
    val = [s for s in val if s.get("question_type") == "action"]
    return val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument(
        "--perturbations",
        default=",".join(PERTURBATIONS),
        help="Comma-separated subset of P1_remove_closest,P2_inject_threat,P3_halve_ttc,P4_swap_risk_level,P5_mask_pedestrians,P6_double_ego_speed",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Cap shard size for fast iteration. 0=no cap.")
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.run)
    stage2_dir = os.path.join(run_dir, "stage2", "best_checkpoint")
    if not os.path.isdir(stage2_dir):
        # Fall back to model files saved directly under stage2/
        stage2_dir = os.path.join(run_dir, "stage2")
    qa_path = os.path.join(run_dir, "data", "driving_qa_data.json")
    out_dir = os.path.join(run_dir, "counterfactual")
    os.makedirs(out_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[CF] loading tokenizer + model from {stage2_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(stage2_dir)
    # R7 v2 branch: under USE_STAGE2_VECTOR_PREFIX=True, Stage 2 is a
    # VectorPrefixT5 saved via lora_utils.save_checkpoint (vector_encoder.pt
    # + t5_model.pt + encoder_config.pt). HF's AutoModel cannot reconstruct
    # the custom vector_encoder submodule, so we must use load_checkpoint.
    if getattr(cfg, "USE_STAGE2_VECTOR_PREFIX", False):
        from llm_driving.lora_utils import load_checkpoint as load_vp_checkpoint
        print("[CF] USE_STAGE2_VECTOR_PREFIX=True — loading Stage 2 as VectorPrefixT5", flush=True)
        model = load_vp_checkpoint(
            model_name=cfg.MODEL_NAME,
            checkpoint_dir=stage2_dir,
            device=device,
            apply_lora_config=False,  # R7 v2 trains full FT, not LoRA
        )
        model.eval()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(stage2_dir).to(device).eval()

    print(f"[CF] loading val samples from {qa_path}", flush=True)
    val_data = load_val_samples(qa_path)
    print(f"[CF] total val action samples: {len(val_data)}", flush=True)

    # Shard (stride pattern, mirrors eval_stage2.py)
    indices = list(range(args.shard_idx, len(val_data), args.num_shards))
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    shard = [val_data[i] for i in indices]
    print(f"[CF] shard {args.shard_idx}/{args.num_shards} -> {len(shard)} samples", flush=True)

    perturbations = [p.strip() for p in args.perturbations.split(",") if p.strip()]
    results = {p: [] for p in perturbations}

    for i, sample in enumerate(shard):
        if i % 50 == 0:
            print(f"[CF] sample {i}/{len(shard)}", flush=True)
        base = run_inference(model, tokenizer, sample, device, args.max_new_tokens)
        for kind in perturbations:
            ps = perturb(sample, kind)
            pert = run_inference(model, tokenizer, ps, device, args.max_new_tokens)
            db = None
            if base["brake"] is not None and pert["brake"] is not None:
                db = pert["brake"] - base["brake"]
            results[kind].append({
                "question_id": sample.get("question_id"),
                "scene_idx": sample.get("scene_idx"),
                "frame_in_scene": sample.get("frame_in_scene"),
                "risk_level": sample.get("risk_level"),
                "base_brake": base["brake"],
                "pert_brake": pert["brake"],
                "delta_brake": db,
                "base_action": base["action_label"],
                "pert_action": pert["action_label"],
                "perturbation_applied": ps.get("_perturbation_applied", True),
                "p4_swap_direction": ps.get("_p4_swap_direction"),
            })

    out_path = os.path.join(out_dir, f"counterfactual_shard{args.shard_idx}.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[CF] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
