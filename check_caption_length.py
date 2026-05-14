"""
Step 4 / M3 — caption length unit test.

Goal: confirm that lanGen's temporal-aware captions (Part C) stay under
Stage 1's max-target-length budget even in the dense (10-object) case.

Stage 1 tokenizer: FLAN-T5 (sentencepiece BPE). STAGE1_MAX_TARGET_LEN=256.

If this script fails, options:
  (a) Reduce `TEMPORAL_CAPTION_TOP_N` in config.py
  (b) Shorten `describe_object` in langen.py
  (c) Bump `STAGE1_MAX_TARGET_LEN` (least preferred — costs Stage 1 memory)

Usage:
    python check_caption_length.py
"""

from __future__ import annotations

import sys
import numpy as np
from transformers import AutoTokenizer

from llm_driving.config import (
    MAX_OBJECTS,
    VECTOR_DIM,
    TEMPORAL_WINDOW,
    TEMPORAL_CAPTION_TOP_N,
    STAGE1_MAX_TARGET_LEN,
    MODEL_NAME,
)
from llm_driving.langen import lanGen


SAFETY_MARGIN_TOKENS = 6  # the caption must come in at <= 256 - margin


def _make_synthetic_dense_frame(K: int, M: int = MAX_OBJECTS) -> tuple:
    """
    Build a worst-case synthetic frame: M=10 objects at varying distances,
    a mix of types, with a tracked K-frame window where objects approach
    or decelerate (so all temporal templates fire).
    """
    np.random.seed(0)
    vectors = np.zeros((M, VECTOR_DIM), dtype=np.float32)
    for j in range(M):
        # rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id
        rel_x = float(np.random.uniform(-15, 15))
        rel_y = float(np.random.uniform(-10, 10))
        dist = float(np.sqrt(rel_x ** 2 + rel_y ** 2))
        rel_vx = float(np.random.uniform(-4, 4))
        rel_vy = float(np.random.uniform(-2, 2))
        heading = float(np.random.uniform(-np.pi, np.pi))
        size = float(np.random.uniform(0.5, 4.5))
        type_id = float(j % 4)  # mix of car / ped / light / obj
        vectors[j] = [rel_x, rel_y, dist, rel_vx, rel_vy, heading, size, type_id]

    # Build a K-frame window where objects approach over time + decelerate.
    window = np.zeros((K, M, VECTOR_DIM), dtype=np.float32)
    present = np.ones((K, M), dtype=bool)  # all objects visible in all K frames
    for k in range(K):
        # Past frame (k < K-1): objects farther away, faster.
        frac_past = (K - 1 - k) * 0.5  # seconds
        for j in range(M):
            v = vectors[j].copy()
            v[0] += v[3] * frac_past   # rel_x += vx * dt
            v[1] += v[4] * frac_past
            v[2] = float(np.sqrt(v[0] ** 2 + v[1] ** 2))
            # Make past speeds higher (deceleration toward now)
            v[3] *= (1 + 0.5 * (frac_past))
            v[4] *= (1 + 0.5 * (frac_past))
            window[k, j] = v

    frame = {
        "vectors": vectors,
        "num_objects": M,
    }
    return frame, window, present


def main() -> None:
    K = int(TEMPORAL_WINDOW)
    top_n = int(TEMPORAL_CAPTION_TOP_N)
    print(f"[M3] Building synthetic dense frame: K={K}, M={MAX_OBJECTS}, top_n={top_n}")
    frame, window, present = _make_synthetic_dense_frame(K)

    # Generate captions in both modes.
    cap_single = lanGen(frame)
    cap_temporal = lanGen(
        frame,
        vectors_window=window,
        object_present_mask=present,
        window_len=K,
        temporal_top_n=top_n,
    )

    print(f"\n[M3] Single-frame caption ({len(cap_single)} chars):")
    print(cap_single)
    print(f"\n[M3] Temporal-aware caption ({len(cap_temporal)} chars):")
    print(cap_temporal)

    # Tokenise both.
    print(f"\n[M3] Loading tokenizer for {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    n_single = len(tok(cap_single, add_special_tokens=True)["input_ids"])
    n_temporal = len(tok(cap_temporal, add_special_tokens=True)["input_ids"])
    print(f"\n[M3] Token counts:")
    print(f"  single-frame  : {n_single} tokens")
    print(f"  temporal      : {n_temporal} tokens")
    print(f"  limit         : {STAGE1_MAX_TARGET_LEN} (safety margin {SAFETY_MARGIN_TOKENS})")

    threshold = int(STAGE1_MAX_TARGET_LEN) - SAFETY_MARGIN_TOKENS
    if n_temporal > threshold:
        print(
            f"\n[M3] FAIL: temporal caption ({n_temporal} tokens) exceeds "
            f"safe budget ({threshold}). See options in script docstring."
        )
        sys.exit(1)

    print(f"\n[M3] PASS: temporal caption fits within {threshold} tokens.")


if __name__ == "__main__":
    main()
