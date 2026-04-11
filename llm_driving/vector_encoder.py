# llm_driving/vector_encoder.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("llm_driving")

@dataclass
class VectorEncoderConfig:
    max_objects: int
    vector_dim: int
    hidden_dim: int
    prefix_len: int
    t5_d_model: int
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    debug: bool = False


class VectorPrefixEncoder(nn.Module):
    """
    Encodes object-level vectors into prefix embeddings for T5.

    Input:  vectors (B, MAX_OBJECTS, VECTOR_DIM)  +  num_objects (B,)
    Output: prefix  (B, PREFIX_LEN, t5_d_model)

    Architecture:
      1. obj_in: Linear projection per object  (VECTOR_DIM -> hidden_dim)
      2. encoder: TransformerEncoder with self-attention across objects
      3. Masked mean-pool over valid objects
      4. to_prefix: Linear projection to (prefix_len * t5_d_model), reshaped
    """

    def __init__(self, cfg: VectorEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # 1) Project each object vector to hidden_dim
        self.obj_in = nn.Linear(cfg.vector_dim, cfg.hidden_dim)

        # 2) Transformer encoder for self-attention across objects
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
        )

        # 3) Project pooled representation to prefix embeddings
        self.to_prefix = nn.Linear(cfg.hidden_dim, cfg.prefix_len * cfg.t5_d_model)

    def forward(self, vectors: torch.Tensor, num_objects: torch.Tensor) -> torch.Tensor:
        B, M, D = vectors.shape

        # OPTIONAL debug checks (never enable during real training runs)
        if getattr(self.cfg, "debug", False):
            if D != self.cfg.vector_dim or M != self.cfg.max_objects:
                logger.warning(
                    f"[VectorPrefixEncoder] unexpected input shape: vectors={tuple(vectors.shape)}, "
                    f"expected (*,{self.cfg.max_objects},{self.cfg.vector_dim})"
                )
            if not torch.isfinite(vectors).all():
                logger.warning("[VectorPrefixEncoder] vectors contain NaN/Inf.")

        device = vectors.device
        x = self.obj_in(vectors)

        idxs = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
        key_padding_mask = idxs >= num_objects.clamp(min=0).unsqueeze(1)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        valid = (~key_padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        pooled = (x * valid).sum(dim=1) / denom

        out = self.to_prefix(pooled).view(B, self.cfg.prefix_len, self.cfg.t5_d_model)
        return out


# ---------------------------------------------------------------------------
# Utility: parse vec_str back to numpy array
# ---------------------------------------------------------------------------

def parse_vec_str(
    vec_str: str,
    max_objects: int = 10,
    vector_dim: int = 8,
) -> Tuple[np.ndarray, int]:
    """
    Reconstruct (max_objects, vector_dim) array from a vec_str string.

    vec_str format: "v0,v1,...,v7; v0,v1,...,v7; ..."
    Returns (padded_vectors, num_valid_objects).
    """
    arr = np.zeros((max_objects, vector_dim), dtype=np.float32)
    if not vec_str or not vec_str.strip():
        return arr, 0

    objects = vec_str.strip().split("; ")
    n = min(len(objects), max_objects)

    for i, obj_str in enumerate(objects[:n]):
        vals = [float(v) for v in obj_str.strip().split(",")]
        fill = min(len(vals), vector_dim)
        arr[i, :fill] = vals[:fill]

    return arr, n
