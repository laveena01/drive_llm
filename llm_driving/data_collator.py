# llm_driving/data_collator.py
"""
Custom data collator for VectorPrefixT5 training.

Batches both tokenized text and raw vector tensors into properly padded
tensors. Works with HuggingFace Trainer or manual training loops.
"""

from __future__ import annotations
from typing import List, Dict, Any
import torch
import numpy as np
from transformers import PreTrainedTokenizerBase


class VectorPrefixDataCollator:
    """
    Collates samples that contain:
        - "input": text string
        - "target": text string
        - "vectors": list of lists (raw vectors per object)
        - "num_objects": int

    Optionally (Step 2 / Part B), when ``temporal_window > 0`` and the samples
    carry ``vectors_window`` / ``num_objects_window`` / ``window_len`` fields,
    the batch also includes:
        - vectors_window:     (B, K, MAX_OBJECTS, VECTOR_DIM) float tensor
        - num_objects_window: (B, K) long tensor
        - window_len:         (B,) long tensor

    Single-frame ``vectors`` / ``num_objects`` keys are *always* emitted so
    the same collator works for both ``USE_TEMPORAL=False`` and
    ``USE_TEMPORAL=True`` runs (the temporal path is selected by
    ``VectorPrefixT5`` based on its config flag, not by the collator).

    Produces batched tensors:
        - input_ids, attention_mask: tokenized text (padded)
        - labels: tokenized targets (padded with -100)
        - vectors: (B, MAX_OBJECTS, VECTOR_DIM) float tensor
        - num_objects: (B,) long tensor
        - vectors_window, num_objects_window, window_len: see above (optional)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_input_length: int = 128,
        max_target_length: int = 192,
        max_objects: int = 10,
        vector_dim: int = 8,
        padding: str = "max_length",
        temporal_window: int = 0,
        pretokenized: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.max_objects = max_objects
        self.vector_dim = vector_dim
        self.padding = padding
        # 0 disables the temporal path; > 0 enables it. Samples missing the
        # window fields are zero-padded so a partial dataset (or eval-time
        # samples that bypass the builder) does not crash.
        self.temporal_window = int(temporal_window)
        # R7 v2: when True, features already carry tokenized fields
        # (`input_ids`, `attention_mask`, `labels`) — produced upstream by
        # `_tokenize_qa` for Stage 2. The collator then only stacks those
        # tensors and adds vectors/num_objects, instead of re-tokenizing
        # raw `input`/`target` strings (which Stage 2 features don't have).
        # Also propagates `question_weight` if present so
        # WeightedLossTrainer can read it from the batch.
        self.pretokenized = bool(pretokenized)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of sample dicts into a batch dict of tensors.
        """
        batch_size = len(features)

        if self.pretokenized:
            # R7 v2 path: features already have tokenized input_ids /
            # attention_mask / labels from _tokenize_qa. Just stack them.
            def _stack_long(key: str) -> torch.Tensor:
                rows = [torch.as_tensor(f[key], dtype=torch.long) for f in features]
                return torch.stack(rows, dim=0)

            model_inputs: Dict[str, torch.Tensor] = {
                "input_ids": _stack_long("input_ids"),
                "attention_mask": _stack_long("attention_mask"),
                "labels": _stack_long("labels"),
            }
            # Propagate per-sample loss weight (used by WeightedLossTrainer).
            if any("question_weight" in f for f in features):
                weights = [float(f.get("question_weight", 1.0)) for f in features]
                model_inputs["question_weight"] = torch.tensor(
                    weights, dtype=torch.float
                )
        else:
            # --- Tokenize text inputs (Stage 1 captioning path) ---
            inputs_text = [f["input"] for f in features]
            targets_text = [f["target"] for f in features]

            model_inputs = self.tokenizer(
                inputs_text,
                max_length=self.max_input_length,
                padding=self.padding,
                truncation=True,
                return_tensors="pt",
            )

            labels = self.tokenizer(
                text_target=targets_text,
                max_length=self.max_target_length,
                padding=self.padding,
                truncation=True,
                return_tensors="pt",
            )

            # Replace padding token id with -100 so loss ignores padding
            label_ids = labels["input_ids"]
            label_ids[label_ids == self.tokenizer.pad_token_id] = -100
            model_inputs["labels"] = label_ids

        # --- Build single-frame vector tensors (always) ---
        vectors_batch = torch.zeros(batch_size, self.max_objects, self.vector_dim)
        num_objects_batch = torch.zeros(batch_size, dtype=torch.long)

        for i, f in enumerate(features):
            if "vectors" in f and f["vectors"]:
                vecs = f["vectors"]
                n = min(len(vecs), self.max_objects)
                for j in range(n):
                    vec = vecs[j]
                    vec_len = min(len(vec), self.vector_dim)
                    vectors_batch[i, j, :vec_len] = torch.tensor(vec[:vec_len], dtype=torch.float)

            if "num_objects" in f:
                num_objects_batch[i] = min(int(f["num_objects"]), self.max_objects)

        model_inputs["vectors"] = vectors_batch
        model_inputs["num_objects"] = num_objects_batch

        # --- Build temporal-window tensors (Step 2 / Part B + Step 4 / Part A) ---
        if self.temporal_window > 0:
            K = self.temporal_window
            vectors_window_batch = torch.zeros(
                batch_size, K, self.max_objects, self.vector_dim
            )
            num_objects_window_batch = torch.zeros(batch_size, K, dtype=torch.long)
            window_len_batch = torch.zeros(batch_size, dtype=torch.long)
            # Step 4 / Part A: per-(frame, slot) "object present" mask. True
            # means slot j of frame s carries a real object; False means
            # either scene-start padding or "the anchor object wasn't
            # visible in that past frame." Defaults to all-True for samples
            # that pre-date Part A (collator-side backward compatibility).
            object_present_mask_batch = torch.ones(
                batch_size, K, self.max_objects, dtype=torch.bool
            )

            for i, f in enumerate(features):
                vw = f.get("vectors_window", None)
                if vw is not None and len(vw) > 0:
                    # vw is shape (K_sample, MAX_OBJECTS, VECTOR_DIM); the
                    # builder right-aligns the window so we can copy directly.
                    k_sample = min(len(vw), K)
                    for s in range(k_sample):
                        slot = vw[s]
                        for j in range(min(len(slot), self.max_objects)):
                            vec = slot[j]
                            vec_len = min(len(vec), self.vector_dim)
                            vectors_window_batch[i, s, j, :vec_len] = torch.tensor(
                                vec[:vec_len], dtype=torch.float
                            )

                    now = f.get("num_objects_window", None)
                    if now is not None:
                        for s in range(min(len(now), K)):
                            num_objects_window_batch[i, s] = min(
                                int(now[s]), self.max_objects
                            )

                    wl = f.get("window_len", k_sample)
                    window_len_batch[i] = max(1, min(int(wl), K))

                    # Object-present mask (Step 4 / Part A). When absent,
                    # default to "all slots present in all real frames,
                    # padding-frames not present" — derived from window_len.
                    opm = f.get("object_present_mask", None)
                    if opm is not None and len(opm) > 0:
                        # Reset to False then copy in (covers any
                        # padding-slots that should be False).
                        object_present_mask_batch[i] = False
                        for s in range(min(len(opm), K)):
                            row = opm[s]
                            for j in range(min(len(row), self.max_objects)):
                                object_present_mask_batch[i, s, j] = bool(row[j])
                    else:
                        # Conservative fallback: scene-start padding frames
                        # have all slots False; real frames keep True.
                        object_present_mask_batch[i] = False
                        wl_int = int(window_len_batch[i].item())
                        for s in range(K - wl_int, K):
                            for j in range(int(num_objects_window_batch[i, s].item())):
                                object_present_mask_batch[i, s, j] = True
                else:
                    # Sample is missing the window — fall back to a 1-frame
                    # window built from the single-frame vectors so the
                    # tensor shape stays valid.
                    vectors_window_batch[i, K - 1] = vectors_batch[i]
                    num_objects_window_batch[i, K - 1] = num_objects_batch[i]
                    window_len_batch[i] = 1
                    object_present_mask_batch[i] = False
                    for j in range(int(num_objects_batch[i].item())):
                        object_present_mask_batch[i, K - 1, j] = True

            model_inputs["vectors_window"] = vectors_window_batch
            model_inputs["num_objects_window"] = num_objects_window_batch
            model_inputs["window_len"] = window_len_batch
            model_inputs["object_present_mask"] = object_present_mask_batch

        return model_inputs
