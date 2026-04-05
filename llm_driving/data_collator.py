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
    
    Produces batched tensors:
        - input_ids, attention_mask: tokenized text (padded)
        - labels: tokenized targets (padded with -100)
        - vectors: (B, MAX_OBJECTS, VECTOR_DIM) float tensor
        - num_objects: (B,) long tensor
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_input_length: int = 128,
        max_target_length: int = 192,
        max_objects: int = 10,
        vector_dim: int = 8,
        padding: str = "max_length",
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.max_objects = max_objects
        self.vector_dim = vector_dim
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of sample dicts into a batch dict of tensors.
        """
        batch_size = len(features)

        # --- Tokenize text inputs ---
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

        # --- Build vector tensors ---
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

        return model_inputs
