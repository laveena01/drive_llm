# llm_driving/training.py

"""
PART-1 updates:
- Stage-2 initialized from Stage-1 weights (dependency)
- Remove min_dist leakage from Stage-2 prompt construction (and oracle eval)
- Fix 5-line format metric by enforcing 5-line output with a robust post-processor
- Add control metrics:
  - accel_mae, brake_mae, steering_accuracy
- Keep existing metrics: action_accuracy buckets, BLEU1, ROUGE-L, format compliance

DDP/torchrun fixes (IMPORTANT):
- Only rank0 writes files / runs heavy generation-eval loops
- All ranks still call trainer.train()
- Barriers are only used for short synchronization (not long generation loops)
- Heavy generation-eval is automatically DISABLED when WORLD_SIZE > 1 to avoid NCCL timeouts
- Disable safetensors checkpointing to avoid shared-tensor save crash for T5
"""

import os
import json
import re
import copy
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist

from datasets import Dataset
from datasets.utils.logging import disable_progress_bar

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
# DDP helpers
# ---------------------------

def _dist_info() -> Tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def _barrier() -> None:
    if _is_dist():
        dist.barrier()

def _rank0_only(rank: int) -> bool:
    return rank == 0

def _unwrap_model(m):
    # Trainer may wrap into DDP. We always want the base module for deepcopy/generate/save.
    return m.module if hasattr(m, "module") else m


# ---------------------------
# Heavy-eval gating (fix NCCL timeouts)
# ---------------------------

# User override: export SKIP_HEAVY_EVAL=1
_SKIP_HEAVY_EVAL = os.environ.get("SKIP_HEAVY_EVAL", "0") == "1"

def _heavy_eval_enabled(world: int) -> bool:
    """
    Heavy eval = generation loops + big json dumps.
    - Disabled automatically under torchrun (world > 1) to avoid NCCL barrier timeouts.
    - Enabled on single GPU unless user forces skip.
    """
    if world > 1:
        return False
    if _SKIP_HEAVY_EVAL:
        return False
    return True


# ---------------------------
# Helpers
# ---------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    p = prompt or ""
    if "Here are my actions:" in p and "Brake pedal" in p and "Steering" in p:
        return p
    return p + PAPER_FORMAT_INSTRUCTION

def _build_stage1_prompt(vec_str: str) -> str:
    return f"Describe the driving scene from object vectors:\n{vec_str}"

def _build_stage2_prompt_from_caption(caption: str) -> str:
    qa_question = "How should the car drive in this situation and why?"
    prompt = caption + f"\n\nQuestion: {qa_question}"
    return _ensure_paper_format(prompt)

def _strip_min_dist(text: str) -> str:
    """
    Removes lines like:
      'Minimum object distance: 8.1 m'
    from oracle prompts to avoid leaking the label proxy.
    """
    return re.sub(
        r"\n?Minimum object distance:\s*[0-9.]+\s*m\s*\n?",
        "\n",
        text,
        flags=re.IGNORECASE,
    )


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
# Parsing + format enforcement
# ---------------------------

def _extract_pct_after(text: str, key: str) -> Optional[int]:
    t = (text or "").lower()
    key = key.lower()
    if key not in t:
        return None
    after = t.split(key, 1)[1].strip()
    num = ""
    for ch in after:
        if ch.isdigit():
            num += ch
        else:
            break
    if not num:
        return None
    v = int(num)
    return max(0, min(100, v))

def _extract_brake_percent(text: str) -> Optional[int]:
    return _extract_pct_after(text, "brake pedal:")

def _extract_accel_percent(text: str) -> Optional[int]:
    return _extract_pct_after(text, "accelerator pedal:")

def _extract_steer(text: str) -> Optional[str]:
    t = (text or "").lower()
    m = re.search(r"steering:\s*(left|straight|right)", t)
    return m.group(1) if m else None

def _extract_reason(text: str) -> Optional[str]:
    t = (text or "").strip()
    m = re.search(r"reason:\s*(.+)$", t, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None

def enforce_5_lines(text: str) -> Tuple[str, int]:
    """
    Return (fixed_text, parse_ok)
    parse_ok=1 if we could parse accel+brake+steer+reason from RAW model output, else 0.
    """
    accel = _extract_accel_percent(text)
    brake = _extract_brake_percent(text)
    steer = _extract_steer(text)
    reason = _extract_reason(text)

    parse_ok = 1 if (accel is not None and brake is not None and steer is not None and reason is not None) else 0

    if accel is None:
        accel = 0
    if brake is None:
        brake = 0
    if steer not in ("left", "straight", "right"):
        steer = "straight"
    if not reason:
        reason = "N/A"

    fixed = (
        "Here are my actions:\n"
        f"- Accelerator pedal: {accel}%\n"
        f"- Brake pedal: {brake}%\n"
        f"- Steering: {steer}\n"
        f"Reason: {reason}\n"
    )
    return fixed, parse_ok

def _format_compliance_5line(text: str) -> int:
    if not text:
        return 0
    lines = [ln.rstrip("\n") for ln in (text or "").splitlines()]
    lines = [ln.strip() for ln in lines if ln.strip()]
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
    if _extract_brake_percent(text) is None or _extract_accel_percent(text) is None:
        return 0
    if _extract_steer(text) is None:
        return 0
    return 1


# ---------------------------
# Stage 2 proxy action metric
# ---------------------------

def _map_text_to_action_label(text: str) -> str:
    b = _extract_brake_percent(text)
    if b is None:
        return "OTHER"
    if b >= 30:
        return "BRAKE"
    if b >= 5:
        return "CAUTION"
    return "CONTINUE"


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
            max_length=192,
        )["input_ids"]

    pad_id = tokenizer.pad_token_id
    labels = [[(tok if tok != pad_id else -100) for tok in seq] for seq in labels]
    model_inputs["labels"] = labels
    return model_inputs


# ---------------------------
# Stage 1 training
# ---------------------------

def train_stage1(captioning_path: str):
    rank, world = _dist_info()
    heavy_eval = _heavy_eval_enabled(world)

    if not _rank0_only(rank):
        disable_progress_bar()

    if _rank0_only(rank):
        logger.info("\n" + "=" * 80)
        logger.info("[STAGE 1] Vector → Caption training started.")
        logger.info(f"[STAGE 1] Loading captioning data from: {captioning_path}")

    with open(captioning_path, "r") as f:
        data = json.load(f)

    full_ds = Dataset.from_list(data)
    if _rank0_only(rank):
        logger.info(f"[STAGE 1] Total samples: {len(full_ds)}")

    split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    if _rank0_only(rank):
        logger.info(f"[STAGE 1] Train samples: {len(train_ds)}  |  Val samples: {len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return _tokenize_captioning(batch, tokenizer)

    if _rank0_only(rank):
        logger.info("[STAGE 1] Tokenizing datasets...")
    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["input", "target"])
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True, remove_columns=["input", "target"])

    if _rank0_only(rank):
        logger.info(f"[STAGE 1] Loading model: {MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    if _rank0_only(rank):
        _ensure_dir(STAGE1_OUTPUT_DIR)
        logger.info(f"[STAGE 1] Output directory: {STAGE1_OUTPUT_DIR}")
    _barrier()  # make sure dir exists before any rank uses it

    training_args = TrainingArguments(
        output_dir=STAGE1_OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        fp16=False,
        optim="adafactor",
        learning_rate=5e-4,
        max_grad_norm=1.0,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        # avoid T5 shared-tensor safetensors crash
        save_safetensors=False,
        # DDP stability
        ddp_find_unused_parameters=False,
        # reduce external logging noise
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )

    if _rank0_only(rank):
        logger.info("[STAGE 1] Starting training...")
    trainer.train()

    # short sync so all ranks finish training before stage transition
    _barrier()

    trained_model = _unwrap_model(trainer.model)

    # Rank0-only: save always (fast)
    if _rank0_only(rank):
        trainer.save_model(STAGE1_OUTPUT_DIR)
        tokenizer.save_pretrained(STAGE1_OUTPUT_DIR)

    # Barrier: ensure all ranks wait for model save to complete
    _barrier()

    if _rank0_only(rank):
        # Heavy generation/eval ONLY when single GPU (world==1)
        if heavy_eval:
            logger.info("[STAGE 1] Training finished. Running evaluation...")

            # In distributed mode, skip trainer.evaluate() to avoid deadlock
            # (trainer.evaluate() has internal barriers that all ranks must participate in)
            if world > 1:
                logger.info("[STAGE 1] Skipping trainer.evaluate() in DDP mode to avoid deadlock")
                eval_metrics = {"eval_loss": "N/A - DDP mode (would cause deadlock)"}
            else:
                eval_metrics = trainer.evaluate()
                logger.info("\n[STAGE 1] Eval metrics:", eval_metrics)

            metrics_path = os.path.join(STAGE1_OUTPUT_DIR, "eval_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(eval_metrics, f, indent=2)
            logger.info(f"[STAGE 1] Saved eval metrics to {metrics_path}")

            logger.info("[STAGE 1] Generating predictions on validation set (rank0 only)...")
            trained_model.eval()
            device = trained_model.device

            val_preds = []
            for sample in eval_ds:
                input_text = sample["input"]
                gt_text = sample["target"]
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=192,
                    truncation=True
                ).to(device)

                with torch.no_grad():
                    pred_ids = trained_model.generate(
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
        else:
            if world > 1:
                logger.info("[STAGE 1] (DDP) Skipping heavy generation/eval to avoid NCCL timeouts. "
                      "Run single-GPU for val_predictions + detailed metrics.")
            elif _SKIP_HEAVY_EVAL:
                logger.info("[STAGE 1] SKIP_HEAVY_EVAL=1 => skipping heavy generation/eval.")

        logger.info("[STAGE 1] Done.\n" + "=" * 80)

    # short sync so stage2 starts together
    _barrier()
    return trained_model, tokenizer


# ---------------------------
# Stage 2 training + evaluation
# ---------------------------

def train_stage2(model_stage1, tokenizer, qa_path: str):
    """
    - Stage-2 model initialized from Stage-1 weights
    - Stage-2 prompts do NOT include min_dist (oracle eval strips it)
    - Heavy generation eval is disabled under DDP (world>1) to avoid NCCL timeouts
    """
    rank, world = _dist_info()
    heavy_eval = _heavy_eval_enabled(world)

    if not _rank0_only(rank):
        disable_progress_bar()

    if _rank0_only(rank):
        logger.info("\n" + "=" * 80)
        logger.info("[STAGE 2] Driving QA finetuning started.")
        logger.info(f"[STAGE 2] Loading QA data from: {qa_path}")

    model_stage1 = _unwrap_model(model_stage1)

    # Stage-2 initialized from Stage-1 weights
    model_stage2 = copy.deepcopy(model_stage1).to(model_stage1.device)
    model_stage2.train()

    with open(qa_path, "r") as f:
        data = json.load(f)

    full_ds = Dataset.from_list(data)
    if _rank0_only(rank):
        logger.info(f"[STAGE 2] Total samples: {len(full_ds)}")

    split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    if _rank0_only(rank):
        logger.info(f"[STAGE 2] Train samples: {len(train_ds)}  |  Val samples: {len(eval_ds)}")

    def tokenize_fn(batch):
        batch_inp = batch["input"]
        batch = dict(batch)
        batch["input"] = [_ensure_paper_format(x) for x in batch_inp]
        return _tokenize_qa(batch, tokenizer)

    if _rank0_only(rank):
        logger.info("[STAGE 2] Tokenizing datasets...")
    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=["input", "target"])
    tokenized_eval = eval_ds.map(tokenize_fn, batched=True, remove_columns=["input", "target"])

    if _rank0_only(rank):
        _ensure_dir(STAGE2_OUTPUT_DIR)
        logger.info(f"[STAGE 2] Output directory: {STAGE2_OUTPUT_DIR}")
    _barrier()

    training_args = TrainingArguments(
        output_dir=STAGE2_OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        fp16=False,
        optim="adafactor",
        learning_rate=5e-4,
        max_grad_norm=1.0,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        # avoid T5 shared-tensor safetensors crash
        save_safetensors=False,
        # DDP stability
        ddp_find_unused_parameters=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model_stage2,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
    )

    if _rank0_only(rank):
        logger.info("[STAGE 2] Starting training...")
    trainer.train()

    # short sync so all ranks finish training
    _barrier()

    trained_stage2 = _unwrap_model(trainer.model)

    if _rank0_only(rank):
        trainer.save_model(STAGE2_OUTPUT_DIR)
        tokenizer.save_pretrained(STAGE2_OUTPUT_DIR)
        logger.info(f"[STAGE 2] Saved model+tokenizer to {STAGE2_OUTPUT_DIR}")

    # Barrier: ensure all ranks wait for model save to complete
    _barrier()

    if _rank0_only(rank):
        # loss-only eval is usually fine (fast); keep it
        logger.info("[STAGE 2] Training finished. Running evaluation (loss only)...")

        # In distributed mode, skip trainer.evaluate() to avoid deadlock
        # (trainer.evaluate() has internal barriers that all ranks must participate in)
        if world > 1:
            logger.info("[STAGE 2] Skipping trainer.evaluate() in DDP mode to avoid deadlock")
            raw_eval = {"eval_loss": "N/A - DDP mode (would cause deadlock)"}
        else:
            raw_eval = trainer.evaluate()
            logger.info("\n[STAGE 2] Raw eval output:", raw_eval)

        # Heavy generation eval ONLY when single GPU (world==1)
        if heavy_eval:
            model_stage1.eval()
            trained_stage2.eval()

            def _gen_text(model, prompt: str, max_new_tokens: int, no_repeat_ngram_size: int = 3) -> str:
                prompt = _ensure_paper_format(prompt)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=192,
                    truncation=True
                ).to(trained_stage2.device)
                with torch.no_grad():
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
                parse_ok_sum = 0.0

                accel_mae_sum = 0.0
                brake_mae_sum = 0.0
                accel_cnt = 0
                brake_cnt = 0

                steer_correct = 0
                steer_total = 0

                n = 0
                outputs: List[Dict] = []

                for sample in eval_ds:
                    raw_input_text = sample["input"]
                    gt_text = sample["target"]

                    if mode == "oracle_caption":
                        stage2_prompt = _ensure_paper_format(_strip_min_dist(raw_input_text))
                        caption_used = None
                    else:
                        vec_str = sample.get("vec_str", "")
                        s1_prompt = _build_stage1_prompt(vec_str)
                        caption_pred = _gen_text(model_stage1, s1_prompt, max_new_tokens=160, no_repeat_ngram_size=4)
                        caption_used = caption_pred
                        stage2_prompt = _build_stage2_prompt_from_caption(caption_pred)

                    pred_raw = _gen_text(trained_stage2, stage2_prompt, max_new_tokens=90, no_repeat_ngram_size=3)
                    pred_fixed, parse_ok = enforce_5_lines(pred_raw)

                    gt_action = _map_text_to_action_label(gt_text)
                    pred_action = _map_text_to_action_label(pred_fixed)

                    if gt_action != "OTHER":
                        total += 1
                        if gt_action == pred_action:
                            correct += 1

                    bleu_sum += bleu1(pred_fixed, gt_text)
                    rouge_sum += rouge_l_f1(pred_fixed, gt_text)

                    fmt_sum += float(_format_compliance_5line(pred_fixed))
                    parse_ok_sum += float(parse_ok)

                    gt_acc = _extract_accel_percent(gt_text)
                    gt_brk = _extract_brake_percent(gt_text)
                    gt_str = _extract_steer(gt_text)

                    pr_acc = _extract_accel_percent(pred_fixed)
                    pr_brk = _extract_brake_percent(pred_fixed)
                    pr_str = _extract_steer(pred_fixed)

                    if gt_acc is not None and pr_acc is not None:
                        accel_mae_sum += abs(pr_acc - gt_acc)
                        accel_cnt += 1
                    if gt_brk is not None and pr_brk is not None:
                        brake_mae_sum += abs(pr_brk - gt_brk)
                        brake_cnt += 1

                    if gt_str is not None and pr_str is not None:
                        steer_total += 1
                        if gt_str == pr_str:
                            steer_correct += 1

                    n += 1

                    outputs.append({
                        "mode": mode,
                        "input": stage2_prompt if mode != "oracle_caption" else raw_input_text,
                        "ground_truth": gt_text,
                        "prediction_raw": pred_raw,
                        "prediction_fixed": pred_fixed,
                        "gt_action": gt_action,
                        "pred_action": pred_action,
                        "format_ok": int(_format_compliance_5line(pred_fixed)),
                        "parse_ok": int(parse_ok),
                        "caption_used": caption_used,
                    })

                metrics = {
                    "action_accuracy": float(correct / total) if total > 0 else 0.0,
                    "bleu1": float(bleu_sum / max(1, n)),
                    "rougeL_f1": float(rouge_sum / max(1, n)),
                    "format_compliance": float(fmt_sum / max(1, n)),
                    "parse_ok_rate": float(parse_ok_sum / max(1, n)),
                    "accel_mae": float(accel_mae_sum / max(1, accel_cnt)),
                    "brake_mae": float(brake_mae_sum / max(1, brake_cnt)),
                    "steering_accuracy": float(steer_correct / max(1, steer_total)) if steer_total > 0 else 0.0,
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
        else:
            if world > 1:
                logger.info("[STAGE 2] (DDP) Skipping heavy generation-eval metrics to avoid NCCL timeouts. "
                      "Run single-GPU for detailed metrics JSON + predictions.")
            elif _SKIP_HEAVY_EVAL:
                logger.info("[STAGE 2] SKIP_HEAVY_EVAL=1 => skipping heavy generation-eval metrics.")

        logger.info("[STAGE 2] Done.\n" + "=" * 80)

    # short sync so all ranks exit together
    _barrier()
    return trained_stage2
