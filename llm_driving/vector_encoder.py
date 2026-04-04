# llm_driving/vector_encoder.py

from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
import logging
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
    num_types: int = 4          # car=0, pedestrian=1, traffic_light=2, object=3
    type_embed_dim: int = 16    # learned embedding dim for categorical type_id
    debug: bool = False


class VectorPrefixEncoder(nn.Module):
    """
    Encodes object-level vectors into prefix embeddings for T5.

    Architecture:
        1. Split input into continuous features (7D) and categorical type_id
        2. Embed type_id via nn.Embedding, concatenate with continuous features
        3. Project to hidden_dim via Linear
        4. Process with TransformerEncoder (self-attention over objects)
        5. Pool via learned query tokens + cross-attention (paper-faithful)
        6. Project to T5 embedding space (hidden_dim → t5_d_model)

    Input:  (B, MAX_OBJECTS, VECTOR_DIM=8)  +  (B,) num_objects
    Output: (B, PREFIX_LEN, t5_d_model)
    """

    def __init__(self, cfg: VectorEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # --- Type embedding for categorical type_id ---
        self.type_embedding = nn.Embedding(cfg.num_types, cfg.type_embed_dim)

        # --- Object projection: (vector_dim - 1 + type_embed_dim) -> hidden_dim ---
        # Subtract 1 because type_id is removed from continuous dims and replaced with embedding
        continuous_dim = cfg.vector_dim - 1  # 7 continuous features
        self.obj_in = nn.Linear(continuous_dim + cfg.type_embed_dim, cfg.hidden_dim)

        # --- Transformer encoder for object-level self-attention ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
        )

        # --- Learned query tokens for cross-attention pooling ---
        self.query_tokens = nn.Parameter(
            torch.randn(cfg.prefix_len, cfg.hidden_dim) / sqrt(cfg.hidden_dim)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(cfg.hidden_dim)

        # --- Project from hidden_dim to t5_d_model ---
        self.to_prefix = nn.Linear(cfg.hidden_dim, cfg.t5_d_model)

    def forward(self, vectors: torch.Tensor, num_objects: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vectors:     (B, MAX_OBJECTS, VECTOR_DIM) raw object vectors.
                         Last dim is type_id (categorical).
            num_objects:  (B,) number of valid objects per sample.
        Returns:
            prefix:      (B, PREFIX_LEN, t5_d_model) prefix embeddings for T5.
        """
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

        # --- Split continuous features and categorical type_id ---
        continuous = vectors[:, :, :-1]                                     # (B, M, 7)
        type_ids = vectors[:, :, -1].long().clamp(0, self.cfg.num_types - 1)  # (B, M)
        type_embeds = self.type_embedding(type_ids)                         # (B, M, type_embed_dim)

        # --- Concatenate and project to hidden_dim ---
        x = torch.cat([continuous, type_embeds], dim=-1)    # (B, M, 7 + type_embed_dim)
        x = self.obj_in(x)                                 # (B, M, hidden_dim)

        # --- Create padding mask (True = padded, should be ignored) ---
        idxs = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
        key_padding_mask = idxs >= num_objects.clamp(min=0).unsqueeze(1)

        # --- Transformer encoder (self-attention over objects) ---
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, M, hidden_dim)

        # --- Cross-attention pooling with learned query tokens ---
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, prefix_len, hidden_dim)
        pooled, _ = self.cross_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )                                                           # (B, prefix_len, hidden_dim)
        pooled = self.cross_attn_norm(pooled + queries)             # residual + layer norm

        # --- Project to T5 embedding space ---
        out = self.to_prefix(pooled)                                # (B, prefix_len, t5_d_model)
        return out
