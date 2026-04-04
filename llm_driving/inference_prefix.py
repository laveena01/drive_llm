# llm_driving/inference_prefix.py
"""
Prefix-conditioned inference for the VectorPrefixT5 pipeline.

Loads base T5 + LoRA adapter + vector encoder from checkpoint,
accepts raw vector arrays, and runs full prefix-conditioned generation.
"""

from __future__ import annotations
import os
import json
import logging
from typing import Dict, List, Optional

import torch
import numpy as np
from transformers import AutoTokenizer

from .config import (
    MODEL_NAME, MAX_OBJECTS, VECTOR_DIM,
    USE_LORA, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    GEN_NUM_BEAMS, GEN_EARLY_STOPPING,
    GEN_NO_REPEAT_NGRAM_SIZE, GEN_REPETITION_PENALTY,
    GEN_MAX_NEW_TOKENS_STAGE1, GEN_MAX_NEW_TOKENS_STAGE2,
    STAGE1_TEXT_PROMPT,
)
from .lora_utils import load_checkpoint
from .inference import enforce_5_lines, _map_text_to_action_label

logger = logging.getLogger("llm_driving")


def load_prefix_model(checkpoint_dir: str, device: str = "cpu"):
    """
    Load a VectorPrefixT5 model from a checkpoint directory.

    Args:
        checkpoint_dir: Directory with vector_encoder.pt + lora_adapter/
        device: Target device

    Returns:
        (model, tokenizer) tuple
    """
    model = load_checkpoint(
        model_name=MODEL_NAME,
        checkpoint_dir=checkpoint_dir,
        device=device,
        apply_lora_config=USE_LORA,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        lora_target_modules=LORA_TARGET_MODULES,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def predict_caption(
    model,
    tokenizer,
    vectors: list,
    num_objects: int,
    device: str = "cpu",
) -> str:
    """
    Generate a scene caption from raw vectors (Stage-1 inference).

    Args:
        model: VectorPrefixT5 model
        tokenizer: T5 tokenizer
        vectors: List of object vectors (each a list of floats)
        num_objects: Number of valid objects
        device: torch device

    Returns:
        Generated caption string
    """
    # Prepare vector tensor
    vec_tensor = torch.zeros(1, MAX_OBJECTS, VECTOR_DIM)
    n = min(len(vectors), MAX_OBJECTS)
    for i in range(n):
        v = vectors[i]
        vec_len = min(len(v), VECTOR_DIM)
        vec_tensor[0, i, :vec_len] = torch.tensor(v[:vec_len], dtype=torch.float)

    num_obj_tensor = torch.tensor([min(num_objects, MAX_OBJECTS)], dtype=torch.long)

    # Tokenize text prompt
    inputs = tokenizer(
        STAGE1_TEXT_PROMPT,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    # Generate
    with torch.no_grad():
        gen_ids = model.generate(
            vectors=vec_tensor.to(device),
            num_objects=num_obj_tensor.to(device),
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=GEN_MAX_NEW_TOKENS_STAGE1,
            num_beams=GEN_NUM_BEAMS,
            early_stopping=GEN_EARLY_STOPPING,
            no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=GEN_REPETITION_PENALTY,
        )

    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def predict_action(
    model,
    tokenizer,
    vectors: list,
    num_objects: int,
    prompt: str,
    device: str = "cpu",
    question_type: str = "action",
) -> Dict:
    """
    Generate a driving action/risk answer from vectors + text prompt (Stage-2 inference).

    Args:
        model: VectorPrefixT5 model
        tokenizer: T5 tokenizer
        vectors: List of object vectors
        num_objects: Number of valid objects
        prompt: Full text prompt (caption + risk + question + format)
        device: torch device
        question_type: "action" or "risk"

    Returns:
        Dict with prediction, enforced output, and action label
    """
    # Prepare vectors
    vec_tensor = torch.zeros(1, MAX_OBJECTS, VECTOR_DIM)
    n = min(len(vectors), MAX_OBJECTS)
    for i in range(n):
        v = vectors[i]
        vec_len = min(len(v), VECTOR_DIM)
        vec_tensor[0, i, :vec_len] = torch.tensor(v[:vec_len], dtype=torch.float)

    num_obj_tensor = torch.tensor([min(num_objects, MAX_OBJECTS)], dtype=torch.long)

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=192,
    )

    # Generate
    with torch.no_grad():
        gen_ids = model.generate(
            vectors=vec_tensor.to(device),
            num_objects=num_obj_tensor.to(device),
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=GEN_MAX_NEW_TOKENS_STAGE2,
            num_beams=GEN_NUM_BEAMS,
            early_stopping=GEN_EARLY_STOPPING,
            no_repeat_ngram_size=GEN_NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=GEN_REPETITION_PENALTY,
        )

    pred_raw = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    result = {"prediction_raw": pred_raw}

    if question_type == "action":
        pred_fixed, parse_ok = enforce_5_lines(pred_raw)
        result["prediction_fixed"] = pred_fixed
        result["parse_ok"] = parse_ok
        result["action_label"] = _map_text_to_action_label(pred_fixed)
    else:
        result["prediction_fixed"] = pred_raw  # risk answers don't need 5-line enforcement

    return result


def run_prefix_inference(
    stage2_checkpoint_dir: str,
    qa_data_path: str,
    output_path: Optional[str] = None,
    limit: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Run full prefix-conditioned inference on QA dataset.

    Args:
        stage2_checkpoint_dir: Stage-2 checkpoint directory
        qa_data_path: Path to QA dataset JSON
        output_path: Where to save results
        limit: Max samples to evaluate
        device: torch device

    Returns:
        Results dict with metrics and predictions
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"[Prefix Inference] Loading model from {stage2_checkpoint_dir}")
    model, tokenizer = load_prefix_model(stage2_checkpoint_dir, device)

    logger.info(f"[Prefix Inference] Loading QA data from {qa_data_path}")
    with open(qa_data_path, "r") as f:
        qa_data = json.load(f)

    if limit:
        qa_data = qa_data[:limit]

    # Use last 20% for eval (matching existing pipeline)
    n_eval = max(1, int(len(qa_data) * 0.2))
    eval_data = qa_data[-n_eval:]

    preds = []
    correct = 0
    total = 0
    parse_ok_sum = 0

    for i, sample in enumerate(eval_data):
        vectors = sample.get("vectors", [])
        num_objects = sample.get("num_objects", 0)
        question_type = sample.get("question_type", "action")

        result = predict_action(
            model, tokenizer,
            vectors=vectors,
            num_objects=num_objects,
            prompt=sample["input"],
            device=device,
            question_type=question_type,
        )

        gt = sample["target"]
        gt_action = _map_text_to_action_label(gt)

        if question_type == "action":
            parse_ok_sum += result.get("parse_ok", 0)
            pred_action = result.get("action_label", "OTHER")

            if gt_action != "OTHER":
                total += 1
                if gt_action == pred_action:
                    correct += 1

        preds.append({
            "input": sample["input"][:200],
            "ground_truth": gt,
            "prediction_raw": result["prediction_raw"],
            "prediction_fixed": result["prediction_fixed"],
            "question_type": question_type,
            "gt_action": gt_action,
            "pred_action": result.get("action_label", "N/A"),
        })

        if (i + 1) % 10 == 0:
            logger.info(f"[Prefix Inference] Processed {i+1}/{len(eval_data)}")

    action_acc = correct / total if total > 0 else 0.0
    parse_ok_rate = parse_ok_sum / max(1, sum(1 for p in preds if p.get("question_type") == "action"))

    metrics = {
        "pipeline_type": "vector_prefix",
        "action_accuracy": float(action_acc),
        "parse_ok_rate": float(parse_ok_rate),
        "num_eval_samples": len(preds),
        "num_action_samples": total,
    }

    logger.info(f"[Prefix Inference] action_accuracy={action_acc:.4f}, parse_ok_rate={parse_ok_rate:.3f}")

    result = {
        "metrics": metrics,
        "predictions": preds,
        "model_dir": stage2_checkpoint_dir,
        "qa_path": qa_data_path,
    }

    if output_path is None:
        output_path = os.path.join(stage2_checkpoint_dir, "prefix_inference_outputs.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"[Prefix Inference] Saved to {output_path}")

    return result
