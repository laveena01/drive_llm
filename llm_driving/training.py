# llm_driving/training.py

"""
PART-2 (training only):
- Use vector-prefix encoder (vectors -> prefix embeddings) instead of vector strings.
- Optional LoRA on FLAN-T5, plus option to freeze base weights.
- Stage-2 initialized from Stage-1 weights (still true).
- Keep your Part-1 metrics & format enforcement.

NOTE: inference not updated here (you said later).
"""

import os
import json
import re
import copy
from typing import List, Dict, Tuple, Optional

import numpy as np
from datasets import Dataset

import torch
import torch.nn as nn

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
    STAGE1_MAX_INPUT_LEN,
    STAGE1_MAX_TARGET_LEN,
    STAGE2_MAX_INPUT_LEN,
    STAGE2_MAX_TARGET_LEN,
    GEN_MAX_NEW_TOKENS_STAGE1,
    GEN_MAX_NEW_TOKENS_STAGE2,
    USE_VECTOR_PREFIX,
    PREFIX_LEN,
    MAX_OBJECTS,
    VECTOR_DIM,
    VEC_ENCODER_HIDDEN,
    VEC_ENCODER_LAYERS,
    VEC_ENCODER_HEADS,
    VEC_ENCODER_DROPOUT,
    FREEZE_BASE_MODEL,
    USE_LORA,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    STAGE1_TEXT_PROMPT,
    STAGE2_QUESTION,
)

from .vector_encoder import VectorPrefixEncoder, VectorEncoderConfig


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

def _build_stage2_prompt_from_caption(caption: str) -> str:
    prompt = caption + f"\n\nQuestion: {STAGE2_QUESTION}"
    return _ensure_paper_format(prompt)

def _strip_min_dist(text: str) -> str:
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
    accel = _extract_accel_percent(text)
    brake = _extract_brake_percent(text)
    steer = _extract_steer(text)
    reason = _extract_reason(text)

    parse_ok = 1 if (accel is not None and brake is not None and steer is not None and reason is not None) else 0

    if accel is None: accel = 0
    if brake is None: brake = 0
    if steer not in ("left", "straight", "right"): steer = "straight"
    if not reason: reason = "N/A"

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
# Vector-prefix wrapper model
# ---------------------------

class VectorPrefixSeq2Seq(nn.Module):
    """
    Wrap a HF seq2seq model:
      - builds encoder inputs_embeds = [prefix(vectors), embed(input_ids)]
      - passes through base model
    """

    def __init__(self, base_model: nn.Module, vector_encoder: VectorPrefixEncoder, prefix_len: int):
        super().__init__()
        self.base_model = base_model
        self.vector_encoder = vector_encoder
        self.prefix_len = prefix_len

        # For Trainer logging expectations sometimes
        self.config = getattr(base_model, "config", None)

    def forward(self, input_ids=None, attention_mask=None, labels=None, vectors=None, num_objects=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        if not USE_VECTOR_PREFIX:
            return self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

        if vectors is None or num_objects is None:
            raise ValueError("USE_VECTOR_PREFIX=True but vectors/num_objects not provided to forward().")

        # (B, P, d_model)
        prefix = self.vector_encoder(vectors, num_objects)

        # (B, T, d_model)
        embed = self.base_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([prefix, embed], dim=1)

        # prepend attention mask of ones for prefix tokens
        B = input_ids.shape[0]
        p_mask = torch.ones((B, self.prefix_len), dtype=attention_mask.dtype, device=attention_mask.device)
        attn = torch.cat([p_mask, attention_mask], dim=1)

        return self.base_model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels, **kwargs)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, vectors=None, num_objects=None, **gen_kwargs):
        if not USE_VECTOR_PREFIX:
            return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

        prefix = self.vector_encoder(vectors, num_objects)
        embed = self.base_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, embed], dim=1)

        B = input_ids.shape[0]
        p_mask = torch.ones((B, self.prefix_len), dtype=attention_mask.dtype, device=attention_mask.device)
        attn = torch.cat([p_mask, attention_mask], dim=1)

        return self.base_model.generate(inputs_embeds=inputs_embeds, attention_mask=attn, **gen_kwargs)

    def save_all(self, save_dir: str, tokenizer: AutoTokenizer):
        """
        Training-only convenience save:
        - base HF model is saved in save_dir (config.json etc)
        - vector encoder weights saved as vector_encoder.pt
        - vector meta saved as vector_encoder_meta.json
        """
        _ensure_dir(save_dir)
        self.base_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        torch.save(self.vector_encoder.state_dict(), os.path.join(save_dir, "vector_encoder.pt"))
        meta = {
            "prefix_len": self.prefix_len,
            "max_objects": MAX_OBJECTS,
            "vector_dim": VECTOR_DIM,
            "hidden_dim": VEC_ENCODER_HIDDEN,
            "layers": VEC_ENCODER_LAYERS,
            "heads": VEC_ENCODER_HEADS,
            "dropout": VEC_ENCODER_DROPOUT,
        }
        with open(os.path.join(save_dir, "vector_encoder_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)


# ---------------------------
# Tokenization + collator
# ---------------------------

def _tokenize_batch_text(batch, tokenizer, max_in: int, max_tgt: int):
    model_inputs = tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=max_in,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["target"],
            truncation=True,
            padding="max_length",
            max_length=max_tgt,
        )["input_ids"]

    pad_id = tokenizer.pad_token_id
    labels = [[(tok if tok != pad_id else -100) for tok in seq] for seq in labels]
    model_inputs["labels"] = labels
    return model_inputs


class VectorDataCollator:
    def __init__(self):
        pass

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # HF tokenizer outputs lists already, convert to torch
        batch = {}
        for k in ["input_ids", "attention_mask", "labels"]:
            batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)

        # vectors + num_objects
        if USE_VECTOR_PREFIX:
            vecs = torch.tensor([f["vectors"] for f in features], dtype=torch.float32)
            nobj = torch.tensor([f["num_objects"] for f in features], dtype=torch.long)
            batch["vectors"] = vecs
            batch["num_objects"] = nobj

        return batch


# ---------------------------
# Model builder (LoRA + freeze)
# ---------------------------

def _maybe_apply_lora(base_model: nn.Module) -> nn.Module:
    if not USE_LORA:
        return base_model

    try:
        from peft import LoraConfig, get_peft_model, TaskType
        lcfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
        )
        model = get_peft_model(base_model, lcfg)
        print("[LoRA] Enabled. Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        return model
    except Exception as e:
        print("[LoRA] Could not enable LoRA (peft missing or target modules mismatch). Falling back. Error:", e)
        return base_model


def _maybe_freeze_base(model: nn.Module):
    if not FREEZE_BASE_MODEL:
        return
    for n, p in model.named_parameters():
        p.requires_grad = False

    # If it's a PeftModel, LoRA params will still be trainable
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True


# ---------------------------
# Stage 1 training
# ---------------------------

def train_stage1(captioning_path: str):
    print("\n" + "=" * 80)
    print("[STAGE 1] Vector → Caption training started.")
    print(f"[STAGE 1] Loading captioning data from: {captioning_path}")

    with open(captioning_path, "r") as f:
        data = json.load(f)

    full_ds = Dataset.from_list(data)
    print(f"[STAGE 1] Total samples: {len(full_ds)}")

    split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    print(f"[STAGE 1] Train samples: {len(train_ds)}  |  Val samples: {len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def map_fn(batch):
        # override input to a stable text prompt when using vector prefix
        if USE_VECTOR_PREFIX:
            batch = dict(batch)
            batch["input"] = [STAGE1_TEXT_PROMPT for _ in batch["input"]]
        return _tokenize_batch_text(batch, tokenizer, STAGE1_MAX_INPUT_LEN, STAGE1_MAX_TARGET_LEN)

    print("[STAGE 1] Tokenizing datasets...")
    tokenized_train = train_ds.map(map_fn, batched=True)
    tokenized_eval = eval_ds.map(map_fn, batched=True)

    # Keep only needed columns
    keep_cols = ["input_ids", "attention_mask", "labels"]
    if USE_VECTOR_PREFIX:
        keep_cols += ["vectors", "num_objects"]

    tokenized_train = tokenized_train.remove_columns([c for c in tokenized_train.column_names if c not in keep_cols])
    tokenized_eval = tokenized_eval.remove_columns([c for c in tokenized_eval.column_names if c not in keep_cols])

    print(f"[STAGE 1] Loading base model: {MODEL_NAME}")
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    base = _maybe_apply_lora(base)
    _maybe_freeze_base(base)

    # Vector encoder (prefix)
    if USE_VECTOR_PREFIX:
        d_model = base.config.d_model
        vec_cfg = VectorEncoderConfig(
            max_objects=MAX_OBJECTS,
            vector_dim=VECTOR_DIM,
            hidden_dim=VEC_ENCODER_HIDDEN,
            prefix_len=PREFIX_LEN,
            t5_d_model=d_model,
            n_layers=VEC_ENCODER_LAYERS,
            n_heads=VEC_ENCODER_HEADS,
            dropout=VEC_ENCODER_DROPOUT,
        )
        vec_enc = VectorPrefixEncoder(vec_cfg)
        model = VectorPrefixSeq2Seq(base, vec_enc, PREFIX_LEN)
    else:
        model = base

    _ensure_dir(STAGE1_OUTPUT_DIR)
    print(f"[STAGE 1] Output directory: {STAGE1_OUTPUT_DIR}")

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
        save_safetensors=False,
        remove_unused_columns=False,  # IMPORTANT: keep vectors in batch
    )

    collator = VectorDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print("[STAGE 1] Starting training...")
    trainer.train()

    # Save (base model + tokenizer + vector encoder weights)
    if isinstance(model, VectorPrefixSeq2Seq):
        model.save_all(STAGE1_OUTPUT_DIR, tokenizer)
    else:
        trainer.save_model(STAGE1_OUTPUT_DIR)
        tokenizer.save_pretrained(STAGE1_OUTPUT_DIR)

    print("[STAGE 1] Training finished. Running evaluation...")
    eval_metrics = trainer.evaluate()
    print("\n[STAGE 1] Eval metrics:", eval_metrics)

    metrics_path = os.path.join(STAGE1_OUTPUT_DIR, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"[STAGE 1] Saved eval metrics to {metrics_path}")
    print("[STAGE 1] Done.\n" + "=" * 80)
    return model, tokenizer


# ---------------------------
# Stage 2 training + evaluation
# ---------------------------

def train_stage2(model_stage1, tokenizer, qa_path: str):
    print("\n" + "=" * 80)
    print("[STAGE 2] Driving QA finetuning started.")
    print(f"[STAGE 2] Loading QA data from: {qa_path}")

    # Depend on Stage-1 weights (exact)
    model_stage2 = copy.deepcopy(model_stage1)
    model_stage2.train()

    with open(qa_path, "r") as f:
        data = json.load(f)

    full_ds = Dataset.from_list(data)
    print(f"[STAGE 2] Total samples: {len(full_ds)}")

    split_ds = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    eval_ds = split_ds["test"]
    print(f"[STAGE 2] Train samples: {len(train_ds)}  |  Val samples: {len(eval_ds)}")

    def map_fn(batch):
        batch = dict(batch)
        batch["input"] = [_ensure_paper_format(x) for x in batch["input"]]
        return _tokenize_batch_text(batch, tokenizer, STAGE2_MAX_INPUT_LEN, STAGE2_MAX_TARGET_LEN)

    print("[STAGE 2] Tokenizing datasets...")
    tokenized_train = train_ds.map(map_fn, batched=True)
    tokenized_eval = eval_ds.map(map_fn, batched=True)

    keep_cols = ["input_ids", "attention_mask", "labels"]
    if USE_VECTOR_PREFIX:
        keep_cols += ["vectors", "num_objects"]

    tokenized_train = tokenized_train.remove_columns([c for c in tokenized_train.column_names if c not in keep_cols])
    tokenized_eval = tokenized_eval.remove_columns([c for c in tokenized_eval.column_names if c not in keep_cols])

    _ensure_dir(STAGE2_OUTPUT_DIR)
    print(f"[STAGE 2] Output directory: {STAGE2_OUTPUT_DIR}")

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
        save_safetensors=False,
        remove_unused_columns=False,
    )

    collator = VectorDataCollator()

    trainer = Trainer(
        model=model_stage2,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print("[STAGE 2] Starting training...")
    trainer.train()

    # Save stage2
    if isinstance(model_stage2, VectorPrefixSeq2Seq):
        model_stage2.save_all(STAGE2_OUTPUT_DIR, tokenizer)
    else:
        trainer.save_model(STAGE2_OUTPUT_DIR)
        tokenizer.save_pretrained(STAGE2_OUTPUT_DIR)

    print("[STAGE 2] Saved model+tokenizer to", STAGE2_OUTPUT_DIR)

    # ---- eval (loss) ----
    raw_eval = trainer.evaluate()
    print("\n[STAGE 2] Raw eval output:", raw_eval)

    # ---- extra eval with generation (oracle_caption vs stage1_caption) ----
    model_stage1.eval()
    model_stage2.eval()

    def _gen_text(model, prompt: str, vectors_np, num_objects: int, max_new_tokens: int, no_repeat_ngram_size: int = 3) -> str:
        prompt = _ensure_paper_format(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=STAGE2_MAX_INPUT_LEN, truncation=True)

        input_ids = inputs["input_ids"].to(next(model.parameters()).device)
        attention_mask = inputs["attention_mask"].to(next(model.parameters()).device)

        if USE_VECTOR_PREFIX:
            v = torch.tensor([vectors_np], dtype=torch.float32, device=input_ids.device)
            n = torch.tensor([num_objects], dtype=torch.long, device=input_ids.device)
            pred_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                vectors=v,
                num_objects=n,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=1.2,
            )
        else:
            pred_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
            vectors_np = sample["vectors"]
            num_obj = int(sample["num_objects"])

            if mode == "oracle_caption":
                stage2_prompt = _ensure_paper_format(_strip_min_dist(raw_input_text))
                caption_used = None
            else:
                # Stage1 caption from vectors
                s1_prompt = STAGE1_TEXT_PROMPT
                caption_pred = _gen_text(
                    model_stage1,
                    s1_prompt,
                    vectors_np=vectors_np,
                    num_objects=num_obj,
                    max_new_tokens=GEN_MAX_NEW_TOKENS_STAGE1,
                    no_repeat_ngram_size=4,
                )
                caption_used = caption_pred
                stage2_prompt = _build_stage2_prompt_from_caption(caption_pred)

            pred_raw = _gen_text(
                model_stage2,
                stage2_prompt,
                vectors_np=vectors_np,
                num_objects=num_obj,
                max_new_tokens=GEN_MAX_NEW_TOKENS_STAGE2,
                no_repeat_ngram_size=3,
            )
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

    print("[STAGE 2] Computing metrics: oracle_caption...")
    oracle_metrics, oracle_outputs = run_eval("oracle_caption")
    print("[STAGE 2] oracle_caption metrics:", oracle_metrics)

    print("[STAGE 2] Computing metrics: stage1_caption...")
    stage1_metrics, stage1_outputs = run_eval("stage1_caption")
    print("[STAGE 2] stage1_caption metrics:", stage1_metrics)

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

    with open(os.path.join(STAGE2_OUTPUT_DIR, "val_predictions_oracle_caption.json"), "w") as f:
        json.dump(oracle_outputs, f, indent=2)

    with open(os.path.join(STAGE2_OUTPUT_DIR, "val_predictions_stage1_caption.json"), "w") as f:
        json.dump(stage1_outputs, f, indent=2)

    print("[STAGE 2] Done.\n" + "=" * 80)
    return model_stage2
