# llm_driving/inference.py

import os
import json
import re
from typing import Dict, List, Optional, Tuple

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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
    ok = 1 if (accel is not None and brake is not None and steer is not None and reason is not None) else 0

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
    return fixed, ok

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

def _has_hf_files(dir_path: str) -> bool:
    # minimal "loadable" check
    return (
        os.path.isdir(dir_path)
        and (
            os.path.exists(os.path.join(dir_path, "config.json"))
            or os.path.exists(os.path.join(dir_path, "adapter_config.json"))
        )
    )

def _resolve_stage2_model_dir(run_dir_stage2: str) -> str:
    """
    Prefer:
      1) run_dir_stage2/  (if it contains config.json)
      2) latest checkpoint in run_dir_stage2/checkpoint-*/ (if that contains config.json)
    """
    run_dir_stage2 = run_dir_stage2.rstrip("/")

    if _has_hf_files(run_dir_stage2) and os.path.exists(os.path.join(run_dir_stage2, "config.json")):
        return run_dir_stage2

    # fallback: latest checkpoint-*
    if os.path.isdir(run_dir_stage2):
        ckpts = []
        for name in os.listdir(run_dir_stage2):
            if name.startswith("checkpoint-"):
                ckpt_dir = os.path.join(run_dir_stage2, name)
                if os.path.exists(os.path.join(ckpt_dir, "config.json")):
                    # parse global step
                    try:
                        step = int(name.split("-", 1)[1])
                    except Exception:
                        step = -1
                    ckpts.append((step, ckpt_dir))
        if ckpts:
            ckpts.sort(key=lambda x: x[0])
            return ckpts[-1][1]

    # if we get here, path either doesn't exist or isn't a HF model folder
    raise FileNotFoundError(
        f"[INF] Could not find a loadable HF model at: {run_dir_stage2}\n"
        f"Expected config.json in stage2/ or stage2/checkpoint-*/"
    )

def run_inference_two_stage(
    run_dir_stage2: str,
    qa_path: str,
    out_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict:
    # 1) resolve model dir robustly (root or latest checkpoint)
    model_dir = _resolve_stage2_model_dir(run_dir_stage2)

    print("[INF] Loading Stage-2 model from:", model_dir)
    tok2 = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    m2 = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)

    if hasattr(m2, "cuda"):
        try:
            m2 = m2.cuda()
        except Exception:
            pass

    # 2) load QA data from the SAME run_dir you passed (no config mismatch)
    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"[INF] QA file not found: {qa_path}")

    print("[INF] Loading QA data:", qa_path)
    qa_data = _load_json(qa_path)
    if limit is not None:
        qa_data = qa_data[:limit]

    full_ds = Dataset.from_list(qa_data)
    split = full_ds.train_test_split(test_size=0.2, seed=42)
    eval_ds = split["test"]

    preds = []
    correct = 0
    total = 0
    parse_ok_sum = 0

    for s in eval_ds:
        raw_inp = s["input"]
        gt = s["target"]

        prompt = raw_inp + PAPER_FORMAT_INSTRUCTION
        pred_raw = _gen(m2, tok2, prompt, max_len=192, max_new=90)
        pred_fixed, parse_ok = enforce_5_lines(pred_raw)
        parse_ok_sum += parse_ok

        gt_action = _map_text_to_action_label(gt)
        pred_action = _map_text_to_action_label(pred_fixed)

        if gt_action != "OTHER":
            total += 1
            if gt_action == pred_action:
                correct += 1

        preds.append({
            "input": raw_inp,
            "ground_truth": gt,
            "prediction_raw": pred_raw,
            "prediction_fixed": pred_fixed,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "parse_ok": int(parse_ok),
            "paper_view": _paper_view_block(raw_inp, pred_fixed),
        })

    action_acc = (correct / total) if total > 0 else 0.0
    parse_ok_rate = parse_ok_sum / max(1, len(preds))
    print(f"[INF] action_accuracy={action_acc:.4f} on {total} labeled samples | parse_ok_rate={parse_ok_rate:.3f}")

    result = {
        "metrics": {"action_accuracy": float(action_acc), "parse_ok_rate": float(parse_ok_rate)},
        "num_eval_samples": len(preds),
        "predictions": preds,
        "model_dir_used": model_dir,
        "qa_path_used": qa_path,
    }

    if out_path is None:
        out_path = os.path.join(run_dir_stage2, "inference_outputs.json")

    _save_json(out_path, result)
    print("[INF] Saved inference outputs to:", out_path)
    return result
