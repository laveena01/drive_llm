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
    tokens_per_object: int = 6  # prefix tokens allocated per object
    debug: bool = False


class VectorPrefixEncoder(nn.Module):
    """
    Encodes object-level vectors into prefix embeddings for T5.

    Architecture (Per-Object Slot Design):
        1. Split input into continuous features (7D) and categorical type_id
        2. Embed type_id via nn.Embedding, concatenate with continuous features
        3. Project to hidden_dim via Linear
        4. Process with TransformerEncoder (self-attention over objects)
        5. Expand each object's hidden state into `tokens_per_object` prefix tokens
           via a learned expansion layer (NOT cross-attention pooling)
        6. Add slot position embeddings so decoder distinguishes object 1 from object 7
        7. Project to T5 embedding space (hidden_dim → t5_d_model)

    Input:  (B, MAX_OBJECTS, VECTOR_DIM=8)  +  (B,) num_objects
    Output: (B, PREFIX_LEN, t5_d_model)

    Key difference from v1: each object gets its own dedicated slot of prefix
    tokens, preventing the mean-field collapse that caused repetition.
    """

    def __init__(self, cfg: VectorEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tokens_per_object = cfg.tokens_per_object

        # Compute how many prefix tokens come from objects vs padding
        self.object_prefix_len = cfg.max_objects * cfg.tokens_per_object  # 10 * 6 = 60
        self.extra_tokens = cfg.prefix_len - self.object_prefix_len       # 64 - 60 = 4

        assert self.extra_tokens >= 0, (
            f"prefix_len ({cfg.prefix_len}) must be >= "
            f"max_objects * tokens_per_object ({self.object_prefix_len})"
        )

        # --- Type embedding for categorical type_id ---
        self.type_embedding = nn.Embedding(cfg.num_types, cfg.type_embed_dim)

        # --- Object projection: (vector_dim - 1 + type_embed_dim) -> hidden_dim ---
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
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
        )

        # --- Per-object token expansion ---
        # Maps each object's hidden state to `tokens_per_object` prefix vectors
        self.token_expansion = nn.Linear(
            cfg.hidden_dim, cfg.tokens_per_object * cfg.hidden_dim
        )
        self.expansion_norm = nn.LayerNorm(cfg.hidden_dim)

        # --- Slot position embeddings ---
        # Distinguishes "slot 0 of object 3" from "slot 0 of object 7"
        # Total positions = max_objects * tokens_per_object + extra_tokens
        self.slot_position_embed = nn.Embedding(cfg.prefix_len, cfg.hidden_dim)

        # --- Object-level position embedding (added before self-attention) ---
        self.object_position_embed = nn.Embedding(cfg.max_objects, cfg.hidden_dim)

        # --- Learnable global context tokens (fill remaining prefix slots) ---
        if self.extra_tokens > 0:
            self.global_tokens = nn.Parameter(
                torch.randn(self.extra_tokens, cfg.hidden_dim) / sqrt(cfg.hidden_dim)
            )

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
        device = vectors.device

        if getattr(self.cfg, "debug", False):
            if D != self.cfg.vector_dim or M != self.cfg.max_objects:
                logger.warning(
                    f"[VectorPrefixEncoder] unexpected input shape: vectors={tuple(vectors.shape)}, "
                    f"expected (*,{self.cfg.max_objects},{self.cfg.vector_dim})"
                )

        # --- Split continuous features and categorical type_id ---
        continuous = vectors[:, :, :-1]                                      # (B, M, 7)
        type_ids = vectors[:, :, -1].long().clamp(0, self.cfg.num_types - 1) # (B, M)
        type_embeds = self.type_embedding(type_ids)                          # (B, M, type_embed_dim)

        # --- Concatenate and project to hidden_dim ---
        x = torch.cat([continuous, type_embeds], dim=-1)   # (B, M, 7 + type_embed_dim)
        x = self.obj_in(x)                                # (B, M, hidden_dim)

        # --- Add object-level positional embeddings ---
        obj_positions = torch.arange(M, device=device)
        x = x + self.object_position_embed(obj_positions).unsqueeze(0)  # (B, M, hidden_dim)

        # --- Create padding mask (True = padded, should be ignored) ---
        idxs = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
        num_obj_clamped = num_objects.clamp(min=0)
        key_padding_mask = idxs >= num_obj_clamped.unsqueeze(1)

        # --- Check if any sample has zero objects (all-masked causes NaN in Transformer) ---
        has_objects = num_obj_clamped > 0  # (B,)

        if has_objects.all():
            # Normal path: all samples have at least 1 object
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        elif not has_objects.any():
            # All samples have zero objects — skip encoder entirely, use zeros
            x = torch.zeros_like(x)
        else:
            # Mixed batch: run encoder only on samples with objects
            mask_with = has_objects.unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1)
            # Run encoder with at least 1 unmasked position per sample
            safe_mask = key_padding_mask.clone()
            safe_mask[~has_objects, 0] = False  # unmask first position to avoid NaN
            x_enc = self.encoder(x, src_key_padding_mask=safe_mask)
            x = x_enc * mask_with  # zero out encoder output for zero-object samples

        # --- Expand each object into tokens_per_object prefix tokens ---
        expanded = self.token_expansion(x)  # (B, M, tokens_per_object * hidden_dim)
        expanded = expanded.view(B, M, self.tokens_per_object, self.cfg.hidden_dim)

        # Reshape to (B, M * tokens_per_object, hidden_dim)
        object_prefix = expanded.reshape(B, self.object_prefix_len, self.cfg.hidden_dim)

        # --- Zero out prefix tokens for padded objects ---
        obj_indices = torch.arange(self.object_prefix_len, device=device).unsqueeze(0)
        obj_owner = obj_indices // self.tokens_per_object  # (1, object_prefix_len)
        n_expanded = num_obj_clamped.unsqueeze(1)  # (B, 1)
        token_mask = (obj_owner < n_expanded).unsqueeze(-1).float()  # (B, object_prefix_len, 1)

        # Apply LayerNorm only to valid tokens (avoids NaN from all-zero inputs)
        # Then zero out padded slots
        normed = self.expansion_norm(object_prefix)
        normed = torch.nan_to_num(normed, nan=0.0)  # zero-object slots: LN(0)=NaN → 0
        object_prefix = normed * token_mask

        # --- Append global context tokens ---
        if self.extra_tokens > 0:
            global_part = self.global_tokens.unsqueeze(0).expand(B, -1, -1)
            prefix = torch.cat([object_prefix, global_part], dim=1)  # (B, prefix_len, hidden_dim)
        else:
            prefix = object_prefix

        # --- Add slot position embeddings ---
        slot_positions = torch.arange(self.cfg.prefix_len, device=device)
        prefix = prefix + self.slot_position_embed(slot_positions).unsqueeze(0)

        # --- Project to T5 embedding space ---
        out = self.to_prefix(prefix)  # (B, prefix_len, t5_d_model)
        return out
