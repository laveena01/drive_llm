# llm_driving/inference.py

import os
import json
from typing import Dict, List, Tuple

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .config import (
    MODEL_NAME,
    CAPTIONING_DATA_PATH,
    QA_DATA_PATH,
    STAGE1_OUTPUT_DIR,
    STAGE2_OUTPUT_DIR,
)
from .logger import get_logger

logger = get_logger("inference")

# Same as datasets_builder paper instruction (keep consistent!)
PAPER_FORMAT_INSTRUCTION = (
    "\n\nYou are an AI Driver.\n"
    "Return EXACTLY 5 lines (with newlines), and nothing else:\n"
    "Here are my actions:\n"
    "- Accelerator pedal: <0-100>%\n"
    "- Brake pedal: <0-100>%\n"
    "- Steering: <left/straight/right>\n"
    "Reason: <one short sentence>\n"
    "Do NOT ask questions. Do NOT add extra text.\n"
)

def _load_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)

def _save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _paper_view_block(inp: str, out: str) -> str:
    return (
        "=== INPUT ===\n"
        f"{inp}\n\n"
        "=== OUTPUT ===\n"
        f"{out}\n"
    )

def _extract_brake_percent(text: str):
    t = text.lower()
    key = "brake pedal:"
    if key not in t:
        return None
    after = t.split(key, 1)[1].strip()
    num = ""
    for ch in after:
        if ch.isdigit():
            num += ch
        else:
            break
    return int(num) if num else None

def _map_text_to_action_label(text: str) -> str:
    b = _extract_brake_percent(text)
    if b is None:
        return "OTHER"
    if b >= 30:
        return "BRAKE"
    if b >= 5:
        return "CAUTION"
    return "CONTINUE"

def _gen(model, tokenizer, prompt: str, max_len: int = 192, max_new: int = 120) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(model.device)

    pred_ids = model.generate(
        **inputs,
        max_new_tokens=max_new,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(pred_ids[0], skip_special_tokens=True)

def run_inference_two_stage(
    run_dir_stage1: str = STAGE1_OUTPUT_DIR,
    run_dir_stage2: str = STAGE2_OUTPUT_DIR,
    qa_path: str = QA_DATA_PATH,
    out_path: str | None = None,
    limit: int | None = None,
) -> Dict:
    """
    Runs paper-style inference:
      stage1: vectors->caption  (optional for evaluation if your QA input already contains caption)
      stage2: caption+question->paper output

    Your current QA dataset already stores caption in `input`.
    So inference mostly means: stage2 generate on QA `input` + format instruction.
    """
    logger.info("[INF] Loading Stage-2 model from:", run_dir_stage2)
    tok2 = AutoTokenizer.from_pretrained(run_dir_stage2)
    m2 = AutoModelForSeq2SeqLM.from_pretrained(run_dir_stage2)

    # If GPU available in your env, HF will move with .to("cuda") outside;
    # but keep it simple here:
    if hasattr(m2, "cuda"):
        try:
            m2 = m2.cuda()
        except Exception:
            pass

    logger.info("[INF] Loading QA data:", qa_path)
    qa_data = _load_json(qa_path)
    if limit is not None:
        qa_data = qa_data[:limit]

    # Use a stable split for reporting (same seed as training)
    full_ds = Dataset.from_list(qa_data)
    split = full_ds.train_test_split(test_size=0.2, seed=42)
    eval_ds = split["test"]

    preds = []
    correct = 0
    total = 0

    for s in eval_ds:
        raw_inp = s["input"]
        gt = s["target"]

        prompt = raw_inp + PAPER_FORMAT_INSTRUCTION
        pred = _gen(m2, tok2, prompt, max_len=192, max_new=90)

        gt_action = _map_text_to_action_label(gt)
        pred_action = _map_text_to_action_label(pred)

        if gt_action != "OTHER":
            total += 1
            if gt_action == pred_action:
                correct += 1

        preds.append({
            "input": raw_inp,
            "ground_truth": gt,
            "prediction": pred,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "paper_view": _paper_view_block(raw_inp, pred),
        })

    action_acc = (correct / total) if total > 0 else 0.0
    logger.info(f"[INF] action_accuracy={action_acc:.4f} on {total} labeled samples")

    # ---- metrics (ROUGE + BLEU + 1 key simple metric) ----
    # ROUGE + BLEU are okay for LLM text overlap reporting (with caveat).
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")

        references = [p["ground_truth"] for p in preds]
        predictions = [p["prediction"] for p in preds]

        rouge_scores = rouge.compute(predictions=predictions, references=references)
        bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])

        metrics = {
            "action_accuracy": action_acc,
            "rougeL": float(rouge_scores.get("rougeL", 0.0)),
            "rouge1": float(rouge_scores.get("rouge1", 0.0)),
            "bleu": float(bleu_score.get("bleu", 0.0)),
        }
    except Exception as e:
        logger.warning("[INF] Could not compute ROUGE/BLEU (missing evaluate package?). Error:", e)
        metrics = {"action_accuracy": action_acc}

    result = {
        "metrics": metrics,
        "num_eval_samples": len(preds),
        "predictions": preds,
    }

    if out_path is None:
        out_path = os.path.join(run_dir_stage2, "inference_outputs.json")

    _save_json(out_path, result)
    logger.info("[INF] Saved inference outputs to:", out_path)
    return result
