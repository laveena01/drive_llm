# llm_driving/vector_encoder.py

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


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


class VectorPrefixEncoder(nn.Module):
    """
    Encodes (MAX_OBJECTS x VECTOR_DIM) -> (PREFIX_LEN x d_model) prefix embeddings.

    Design (simple + stable):
    - Linear per-object embedding
    - TransformerEncoder over objects (masked by num_objects)
    - Masked mean pool -> scene embedding
    - Project -> PREFIX_LEN * d_model
    """

    def __init__(self, cfg: VectorEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.obj_in = nn.Linear(cfg.vector_dim, cfg.hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.to_prefix = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.prefix_len * cfg.t5_d_model),
        )

    def forward(self, vectors: torch.Tensor, num_objects: torch.Tensor) -> torch.Tensor:
        """
        vectors: (B, MAX_OBJECTS, VECTOR_DIM) float
        num_objects: (B,) int
        returns: (B, PREFIX_LEN, d_model)
        """
        B, M, D = vectors.shape
        device = vectors.device

        # (B, M, H)
        x = self.obj_in(vectors)

        # Build padding mask: True where padded positions
        # key_padding_mask: (B, M), True means "ignore"
        idxs = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
        key_padding_mask = idxs >= num_objects.clamp(min=0).unsqueeze(1)

        # Transformer encode objects
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Masked mean pool
        valid = (~key_padding_mask).float().unsqueeze(-1)  # (B,M,1)
        denom = valid.sum(dim=1).clamp(min=1.0)            # (B,1)
        pooled = (x * valid).sum(dim=1) / denom            # (B,H)

        # Project to prefix tokens
        out = self.to_prefix(pooled).view(B, self.cfg.prefix_len, self.cfg.t5_d_model)
        return out
