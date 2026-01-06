# llm_driving/training.py

"""
Training utilities for:
- Stage 1: Vector -> Caption (captioning pretraining)
- Stage 2: Caption + Question -> Answer (Driving QA finetuning)

Paper-faithful flow + metrics:
1) Stage-2 model is trained from MODEL_NAME (base pretrained), NOT Stage-1 weights.
2) Stage-2 eval reports TWO modes:
   - oracle_caption: uses dataset prompt (lanGen caption embedded)
   - stage1_caption: uses Stage-1 caption generated from vec_str, then Stage-2 consumes it
3) Metrics:
   - action_accuracy (proxy via Brake pedal %)
   - BLEU-1
   - ROUGE-L F1
   - format_compliance (5-line output + parseable brake line)

CRITICAL stability changes:
- fp16 disabled (prevents NaNs)
- Adafactor optimizer for T5 + safer LR + grad clipping
"""

import os
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .config import (
    MODEL_NAME,
    STAGE1_OUTPUT_DIR,
    STAGE2_OUTPUT_DIR,
)
from .logger import get_logger

logger = get_logger("training")

# ---------------------------
# Helpers
# ---------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# Keep Stage-2 prompt format local to avoid circular imports
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

def _ensure_paper_format(prompt: str) -> str:
    """Append format instruction if it's not already present."""
    p = prompt or ""
    if "Here are my actions:" in p and "Brake pedal" in p and "Steering" in p:
        return p
    return p + PAPER_FORMAT_INSTRUCTION

def _build_stage1_prompt(vec_str: str) -> str:
    return f"Describe the driving scene from object vectors:\n{vec_str}"

def _build_stage2_prompt_from_caption(caption: str, min_dist: Optional[float]) -> str:
    qa_question = "How should the car drive in this situation and why?"
    min_dist_line = ""
    if min_dist is not None:
        min_dist_line = f"\nMinimum object distance: {float(min_dist):.1f} m\n\n"
    prompt = (
        caption
        + min_dist_line
        + f"Question: {qa_question}"
    )
    return _ensure_paper_format(prompt)


# ---------------------------
# Metrics: BLEU-1 + ROUGE-L
# ---------------------------

def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()

def _lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]

def rouge_l_f1(pred: str, ref: str) -> float:
    pred_toks = _normalize_text(pred).split()
    ref_toks = _normalize_text(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0
    lcs = _lcs_len(pred_toks, ref_toks)
    prec = lcs / max(1, len(pred_toks))
    rec = lcs / max(1, len(ref_toks))
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))

def bleu1(pred: str, ref: str) -> float:
    pred_toks = _normalize_text(pred).split()
    ref_toks = _normalize_text(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0

    ref_counts: Dict[str, int] = {}
    for t in ref_toks:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    used: Dict[str, int] = {}
    match = 0
    for t in pred_toks:
        used[t] = used.get(t, 0) + 1
        if t in ref_counts and used[t] <= ref_counts[t]:
            match += 1

    precision = match / max(1, len(pred_toks))

    bp = 1.0
    if len(pred_toks) < len(ref_toks):
        bp = float(np.exp(1 - (len(ref_toks) / max(1, len(pred_toks)))))

    return float(bp * precision)


# ---------------------------
# Stage 2 proxy action metric
# ---------------------------

def _extract_brake_percent(text: str):
    t = (text or "").lower()
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

def _format_compliance(text: str) -> int:
    """
    1 if output looks like the required 5 lines AND brake percent parseable, else 0.
    """
    if not text:
        return 0
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) != 5:
        return 0
    if not lines[0].lower().startswith("here are my actions"):
        return 0
    if "accelerator pedal:" not in lines[1].lower():
        return 0
    if "brake pedal:" not in lines[2].lower():
        return 0
    if "steering:" not in lines[3].lower():
        return 0
    if not lines[4].lower().startswith("reason:"):
        return 0
    if _extract_brake_percent(text) is None:
        return 0
    return 1


# ---------------------------
# Tokenization
# ---------------------------

def _tokenize_captioning(batch, tokenizer):
    model_inputs = tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=192,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"],
            truncation=True,
            padding="max_length",
            max_length=192,
        )["input_ids"]

    pad_id = tokenizer.pad_token_id
    labels = [[(tok if tok != pad_id else -100) for tok in seq] for seq in labels]
    model_inputs["labels"] = labels
    return model_inputs

def _tokenize_qa(batch, tokenizer):
    model_inputs = tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=192,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"],
            truncation=True,
            padding="max_length",
            max_length=192,   # 5-line output is long enough
        )["input_ids"]

    pad_id = tokenizer.pad_token_id
    labels = [[(tok if tok != pad_id else -100) for tok in seq] for seq in labels]
    model_inputs["labels"] = labels
    return model_inputs


# ---------------------------
# Stage 1 training
# ---------------------------

def train_stage1(captioning_path: str):
    logger.info("\n" + "=" * 80)
    logger.info("[STAGE 1] Vector → Caption training started.")
    logger.info(f"[STAGE 1] Loading captioning data from: {captioning_path}")

    with open(captioning_path, "r") as f:
        data = json.load(f)

    full_ds = Dataset.from_list(data)
    logger.info(f"[STAGE 1] Total samples: {len(full_ds)}")

    split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    logger.info(f"[STAGE 1] Train samples: {len(train_ds)}  |  Val samples: {len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return _tokenize_captioning(batch, tokenizer)

    logger.info("[STAGE 1] Tokenizing datasets...")
    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["input", "target"])
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True, remove_columns=["input", "target"])

    logger.info(f"[STAGE 1] Loading model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    _ensure_dir(STAGE1_OUTPUT_DIR)
    logger.info(f"[STAGE 1] Output directory: {STAGE1_OUTPUT_DIR}")

    training_args = TrainingArguments(
        output_dir=STAGE1_OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=10,

        # stability for T5
        fp16=False,
        optim="adafactor",
        learning_rate=5e-4,
        max_grad_norm=1.0,

        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )

    logger.info("[STAGE 1] Starting training...")
    trainer.train()
    logger.info("[STAGE 1] Training finished. Running evaluation...")

    eval_metrics = trainer.evaluate()
    logger.info("\n[STAGE 1] Eval metrics:", eval_metrics)

    metrics_path = os.path.join(STAGE1_OUTPUT_DIR, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"[STAGE 1] Saved eval metrics to {metrics_path}")

    logger.info("[STAGE 1] Generating predictions on validation set...")
    model.eval()
    val_preds = []
    for sample in eval_ds:
        input_text = sample["input"]
        gt_text = sample["target"]
        inputs = tokenizer(input_text, return_tensors="pt", max_length=192, truncation=True).to(model.device)

        pred_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=4,
            repetition_penalty=1.2,
        )
        pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)

        val_preds.append({"input": input_text, "ground_truth": gt_text, "prediction": pred_text})

    preds_path = os.path.join(STAGE1_OUTPUT_DIR, "val_predictions.json")
    with open(preds_path, "w") as f:
        json.dump(val_preds, f, indent=2)
    logger.info(f"[STAGE 1] Saved {len(val_preds)} validation predictions to {preds_path}")

    # quick sanity
    if len(eval_ds) > 0:
        s0 = eval_ds[0]
        inputs = tokenizer(s0["input"], return_tensors="pt", max_length=192, truncation=True).to(model.device)
        pred_ids = model.generate(**inputs, max_new_tokens=160, num_beams=4, early_stopping=True)
        pred_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)

        logger.info("\n[STAGE 1 TEST SAMPLE]")
        logger.info("INPUT:\n", s0["input"])
        logger.info("\nPRED CAPTION:\n", pred_text)
        logger.info("\nGT CAPTION:\n", s0["target"])

    logger.info("[STAGE 1] Done.\n" + "=" * 80)
    return model, tokenizer


# ---------------------------
# Stage 2 training + evaluation
# ---------------------------

def train_stage2(model_stage1, tokenizer, qa_path: str):
    """
    model_stage1: trained stage-1 model (vector->caption) used for stage1_caption evaluation.
    Stage-2 model is trained from base pretrained MODEL_NAME weights.
    """
    logger.info("\n" + "=" * 80)
    logger.info("[STAGE 2] Driving QA finetuning started.")
    logger.info(f"[STAGE 2] Loading QA data from: {qa_path}")

    logger.info(f"[STAGE 2] Loading Stage-2 model from base: {MODEL_NAME}")
    model_stage2 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(model_stage1.device)

    with open(qa_path, "r") as f:
        data = json.load(f)

    full_ds = Dataset.from_list(data)
    logger.info(f"[STAGE 2] Total samples: {len(full_ds)}")

    split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    logger.info(f"[STAGE 2] Train samples: {len(train_ds)}  |  Val samples: {len(eval_ds)}")

    def tokenize_fn(batch):
        # ensure paper-format instruction is present during training too
        batch_inp = batch["input"]
        batch = dict(batch)
        batch["input"] = [_ensure_paper_format(x) for x in batch_inp]
        return _tokenize_qa(batch, tokenizer)

    logger.info("[STAGE 2] Tokenizing datasets...")
    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["input", "target"])
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True, remove_columns=["input", "target"])

    _ensure_dir(STAGE2_OUTPUT_DIR)
    logger.info(f"[STAGE 2] Output directory: {STAGE2_OUTPUT_DIR}")

    training_args = TrainingArguments(
        output_dir=STAGE2_OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=5,

        # stability for T5
        fp16=False,
        optim="adafactor",
        learning_rate=5e-4,
        max_grad_norm=1.0,

        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model_stage2,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )

    logger.info("[STAGE 2] Starting training...")
    trainer.train()
    logger.info("[STAGE 2] Training finished. Running evaluation (loss only)...")

    raw_eval = trainer.evaluate()
    logger.info("\n[STAGE 2] Raw eval output:", raw_eval)

    model_stage1.eval()
    model_stage2.eval()

    def _gen_text(model, prompt: str, max_new_tokens: int, no_repeat_ngram_size: int = 3) -> str:
        prompt = _ensure_paper_format(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=192, truncation=True).to(model_stage2.device)
        pred_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=1.2,
        )
        return tokenizer.decode(pred_ids[0], skip_special_tokens=True)

    def run_eval(mode: str) -> Tuple[Dict, List[Dict]]:
        correct = 0
        total = 0
        bleu_sum = 0.0
        rouge_sum = 0.0
        fmt_sum = 0.0
        n = 0

        outputs: List[Dict] = []

        for sample in eval_ds:
            raw_input_text = sample["input"]
            gt_text = sample["target"]

            if mode == "oracle_caption":
                stage2_prompt = _ensure_paper_format(raw_input_text)
                caption_used = None
            else:
                vec_str = sample.get("vec_str", "")
                min_dist = sample.get("min_dist", None)

                s1_prompt = _build_stage1_prompt(vec_str)
                caption_pred = _gen_text(model_stage1, s1_prompt, max_new_tokens=160, no_repeat_ngram_size=4)
                caption_used = caption_pred

                stage2_prompt = _build_stage2_prompt_from_caption(caption_pred, min_dist)

            pred_text = _gen_text(model_stage2, stage2_prompt, max_new_tokens=90, no_repeat_ngram_size=3)

            gt_action = _map_text_to_action_label(gt_text)
            pred_action = _map_text_to_action_label(pred_text)

            if gt_action != "OTHER":
                total += 1
                if gt_action == pred_action:
                    correct += 1

            bleu_sum += bleu1(pred_text, gt_text)
            rouge_sum += rouge_l_f1(pred_text, gt_text)
            fmt_sum += float(_format_compliance(pred_text))
            n += 1

            outputs.append({
                "mode": mode,
                "input": stage2_prompt if mode != "oracle_caption" else raw_input_text,
                "ground_truth": gt_text,
                "prediction": pred_text,
                "gt_action": gt_action,
                "pred_action": pred_action,
                "format_ok": int(_format_compliance(pred_text)),
                "caption_used": caption_used,
            })

        metrics = {
            "action_accuracy": float(correct / total) if total > 0 else 0.0,
            "bleu1": float(bleu_sum / max(1, n)),
            "rougeL_f1": float(rouge_sum / max(1, n)),
            "format_compliance": float(fmt_sum / max(1, n)),
            "n_samples": int(n),
            "n_action_samples": int(total),
        }
        return metrics, outputs

    logger.info("[STAGE 2] Computing metrics: oracle_caption...")
    oracle_metrics, oracle_outputs = run_eval("oracle_caption")
    logger.info("[STAGE 2] oracle_caption metrics:", oracle_metrics)

    logger.info("[STAGE 2] Computing metrics: stage1_caption...")
    stage1_metrics, stage1_outputs = run_eval("stage1_caption")
    logger.info("[STAGE 2] stage1_caption metrics:", stage1_metrics)

    eval_metrics: Dict = {}
    if isinstance(raw_eval, dict):
        for k, v in raw_eval.items():
            try:
                eval_metrics[k] = float(v)
            except Exception:
                eval_metrics[k] = v

    eval_metrics["oracle_caption"] = oracle_metrics
    eval_metrics["stage1_caption"] = stage1_metrics

    metrics_path = os.path.join(STAGE2_OUTPUT_DIR, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"[STAGE 2] Saved eval metrics to {metrics_path}")

    preds_path1 = os.path.join(STAGE2_OUTPUT_DIR, "val_predictions_oracle_caption.json")
    with open(preds_path1, "w") as f:
        json.dump(oracle_outputs, f, indent=2)
    logger.info(f"[STAGE 2] Saved oracle-caption predictions to {preds_path1}")

    preds_path2 = os.path.join(STAGE2_OUTPUT_DIR, "val_predictions_stage1_caption.json")
    with open(preds_path2, "w") as f:
        json.dump(stage1_outputs, f, indent=2)
    logger.info(f"[STAGE 2] Saved stage1-caption predictions to {preds_path2}")

    if len(oracle_outputs) > 0:
        logger.info("\n[STAGE 2 TEST SAMPLE - oracle_caption]")
        logger.info("PROMPT:\n", oracle_outputs[0]["input"])
        logger.info("\nPRED:\n", oracle_outputs[0]["prediction"])
        logger.info("\nGT:\n", oracle_outputs[0]["ground_truth"])

    if len(stage1_outputs) > 0:
        logger.info("\n[STAGE 2 TEST SAMPLE - stage1_caption]")
        logger.info("PROMPT:\n", stage1_outputs[0]["input"])
        logger.info("\nCAPTION USED:\n", stage1_outputs[0]["caption_used"])
        logger.info("\nPRED:\n", stage1_outputs[0]["prediction"])
        logger.info("\nGT:\n", stage1_outputs[0]["ground_truth"])

    logger.info("[STAGE 2] Done.\n" + "=" * 80)
    return model_stage2
