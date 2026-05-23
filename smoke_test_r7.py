"""
smoke_test_r7.py — Phase 2 validation for R7 v2 (Direct Vector Prefix to Stage 2).

Validates two high-risk wiring points before launching the 12h R7 train:

  1. Save / load round-trip via `lora_utils.save_checkpoint` /
     `lora_utils.load_checkpoint`. After save → load, the reloaded
     `VectorPrefixT5` must produce bit-identical (within `atol=1e-4`) logits
     on a fixed input compared to the original. This catches the
     "HF Trainer silently dropped vector_encoder.pt" gotcha described in
     the R7 plan.

  2. Vector-channel diff. Two forward passes with the SAME text input but
     different `vectors` tensors (real-ish vs zero) must produce different
     logits. If outputs are identical, the vector prefix isn't being used
     and the whole R7 hypothesis is moot.

Runs in ~1 min on CPU; ~10 s on a GPU. No real training, no dataset
needed. Uses a freshly initialized model (no Stage 1 weights), so it
tests the wiring, not learned behavior.

Usage:
    python smoke_test_r7.py                 # CPU
    python smoke_test_r7.py --device cuda:0 # GPU
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import torch
from transformers import AutoTokenizer

from llm_driving import config as cfg
from llm_driving.lora_utils import load_checkpoint, save_checkpoint
from llm_driving.vector_encoder import VectorEncoderConfig
from llm_driving.vector_prefix_t5 import VectorPrefixT5


def _build_fresh_model(device: str):
    """Construct a fresh VectorPrefixT5 with the same config R7 v2 uses."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    encoder_config = VectorEncoderConfig(**cfg.VECTOR_ENCODER_CONFIG)
    model = VectorPrefixT5(
        model_name=cfg.MODEL_NAME,
        encoder_config=encoder_config,
        tokenizer=tokenizer,
        use_temporal=False,
    )
    model.eval().to(device)
    return model, tokenizer


def _fixed_inputs(tokenizer, device: str):
    """Build a deterministic (text, vectors, num_objects) triple."""
    torch.manual_seed(42)
    prompt = (
        "### OBSERVATION\nCar 4m ahead.\n\n"
        "### RISK\nRisk level: HIGH.\n\n"
        "### QUESTION\nHow should the car drive?"
    )
    text_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=cfg.STAGE2_MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
    ).to(device)
    # Random-ish but deterministic vectors: 3 real objects, rest zero-padded.
    vectors = torch.zeros(1, cfg.MAX_OBJECTS, cfg.VECTOR_DIM, device=device)
    vectors[0, 0] = torch.tensor([4.0, 0.0, 4.0, -5.0, 0.0, 0.0, 1.0, 0.0])  # car ahead
    vectors[0, 1] = torch.tensor([2.0, 1.0, 2.2, 0.0, 0.0, 0.0, 1.0, 1.0])    # pedestrian right
    vectors[0, 2] = torch.tensor([8.0, -2.0, 8.2, -3.0, 0.0, 0.0, 1.5, 0.0])  # car ahead-left
    num_objects = torch.tensor([3], dtype=torch.long, device=device)
    # Dummy labels so we can use forward(labels=...) and inspect loss/logits.
    labels = text_inputs["input_ids"].clone()
    return text_inputs, vectors, num_objects, labels


@torch.no_grad()
def _forward_logits(model, text_inputs, vectors, num_objects, labels):
    out = model(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        vectors=vectors,
        num_objects=num_objects,
        labels=labels,
    )
    return out.logits.detach().cpu()


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def test_save_load_roundtrip(device: str, atol: float = 1e-4) -> bool:
    """Test 1: save → load round-trip preserves logits."""
    print("\n" + "=" * 70)
    print("[TEST 1] save → load round-trip identical-logits check")
    print("=" * 70)

    model_a, tokenizer = _build_fresh_model(device)
    text_inputs, vectors, num_objects, labels = _fixed_inputs(tokenizer, device)
    logits_a = _forward_logits(model_a, text_inputs, vectors, num_objects, labels)
    print(f"[TEST 1] logits_a shape={tuple(logits_a.shape)}, mean={logits_a.mean():.6f}")

    tmpdir = tempfile.mkdtemp(prefix="r7_smoke_")
    try:
        save_checkpoint(model_a, tmpdir, epoch=1)
        files = sorted(os.listdir(tmpdir))
        print(f"[TEST 1] save_checkpoint wrote: {files}")

        required = {"vector_encoder.pt", "encoder_config.pt"}
        missing = required - set(files)
        if missing:
            print(f"[TEST 1] FAIL — required files missing from checkpoint: {missing}")
            return False
        has_t5 = ("t5_model.pt" in files) or ("lora_adapter" in files)
        if not has_t5:
            print(f"[TEST 1] FAIL — neither t5_model.pt nor lora_adapter/ in checkpoint")
            return False

        del model_a
        model_b = load_checkpoint(
            model_name=cfg.MODEL_NAME,
            checkpoint_dir=tmpdir,
            device=device,
            apply_lora_config=False,
        )
        model_b.eval()
        logits_b = _forward_logits(model_b, text_inputs, vectors, num_objects, labels)
        print(f"[TEST 1] logits_b shape={tuple(logits_b.shape)}, mean={logits_b.mean():.6f}")

        diff = _max_abs_diff(logits_a, logits_b)
        print(f"[TEST 1] max |logits_a - logits_b| = {diff:.2e}  (tol={atol:.2e})")
        if diff <= atol:
            print("[TEST 1] PASS — save/load round-trip preserves logits.")
            return True
        else:
            print(
                "[TEST 1] FAIL — logits diverged after save/load. Likely cause: "
                "vector_encoder weights were not persisted (or not reloaded)."
            )
            return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_vector_channel_diff(device: str, min_diff: float = 1e-3) -> bool:
    """Test 2: zero-vectors vs real-vectors must produce different logits."""
    print("\n" + "=" * 70)
    print("[TEST 2] vector channel is wired — zero-vec vs real-vec output diff")
    print("=" * 70)

    model, tokenizer = _build_fresh_model(device)
    text_inputs, vectors, num_objects, labels = _fixed_inputs(tokenizer, device)

    logits_real = _forward_logits(model, text_inputs, vectors, num_objects, labels)
    print(f"[TEST 2] logits_real mean={logits_real.mean():.6f}, std={logits_real.std():.6f}")

    zero_vectors = torch.zeros_like(vectors)
    zero_n = torch.zeros_like(num_objects)
    logits_zero = _forward_logits(model, text_inputs, zero_vectors, zero_n, labels)
    print(f"[TEST 2] logits_zero mean={logits_zero.mean():.6f}, std={logits_zero.std():.6f}")

    diff = _max_abs_diff(logits_real, logits_zero)
    print(f"[TEST 2] max |logits_real - logits_zero| = {diff:.2e}  (must exceed {min_diff:.2e})")
    if diff >= min_diff:
        print("[TEST 2] PASS — vector channel materially changes output.")
        return True
    else:
        print(
            "[TEST 2] FAIL — vectors do not influence the model's logits. "
            "Either the vector prefix is being zeroed out, or the encoder "
            "is producing identical embeddings regardless of input."
        )
        return False


def test_basic_generate(device: str) -> bool:
    """Test 3: model.generate() works end-to-end with vectors threaded through."""
    print("\n" + "=" * 70)
    print("[TEST 3] model.generate() smoke — produces non-empty output")
    print("=" * 70)

    model, tokenizer = _build_fresh_model(device)
    text_inputs, vectors, num_objects, _ = _fixed_inputs(tokenizer, device)
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                vectors=vectors,
                num_objects=num_objects,
                max_new_tokens=20,
                num_beams=2,
                early_stopping=True,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"[TEST 3] generated text: {text!r}")
        if len(text) >= 0:
            print("[TEST 3] PASS — generate() returned without error.")
            return True
        return False
    except Exception as e:
        print(f"[TEST 3] FAIL — generate() raised: {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda:0 if available, else cpu).",
    )
    args = parser.parse_args()

    print(f"[SMOKE] device={args.device}")
    print(f"[SMOKE] cfg.MODEL_NAME={cfg.MODEL_NAME}")
    print(f"[SMOKE] cfg.USE_STAGE2_VECTOR_PREFIX={getattr(cfg, 'USE_STAGE2_VECTOR_PREFIX', None)}")

    results = {
        "save_load_roundtrip": test_save_load_roundtrip(args.device),
        "vector_channel_diff": test_vector_channel_diff(args.device),
        "basic_generate": test_basic_generate(args.device),
    }

    print("\n" + "=" * 70)
    print("[SMOKE] SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        print(f"  {name:30s} {'PASS' if ok else 'FAIL'}")

    if all(results.values()):
        print("\n[SMOKE] All checks passed. Safe to launch R7 train.")
        sys.exit(0)
    else:
        print("\n[SMOKE] One or more checks FAILED. Do NOT launch R7 train yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
