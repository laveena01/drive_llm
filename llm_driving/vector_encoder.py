# llm_driving/vector_encoder.py

from __future__ import annotations
from dataclasses import dataclass
from math import sqrt
from typing import Optional, Tuple
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
        7. Project to T5 embedding space (hidden_dim -> t5_d_model)

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

    def forward(
        self,
        vectors: torch.Tensor,
        num_objects: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            vectors:     (B, MAX_OBJECTS, VECTOR_DIM) raw object vectors.
                         Last dim is type_id (categorical).
            num_objects:  (B,) number of valid objects per sample.
            return_hidden: if True, return the pre-projection prefix in
                hidden_dim space `(B, PREFIX_LEN, hidden_dim)` instead of the
                T5-projected output. Used by `TemporalVectorEncoder` so the
                temporal attention runs in hidden_dim before sharing the
                final `to_prefix` projection.

        Returns:
            prefix:      (B, PREFIX_LEN, t5_d_model) prefix embeddings for T5,
                         or `(B, PREFIX_LEN, hidden_dim)` when `return_hidden`.
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
        normed = torch.nan_to_num(normed, nan=0.0)  # zero-object slots: LN(0)=NaN -> 0
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

        if return_hidden:
            return prefix  # (B, prefix_len, hidden_dim)

        # --- Project to T5 embedding space ---
        out = self.to_prefix(prefix)  # (B, prefix_len, t5_d_model)
        return out


# ---------------------------------------------------------------------------
# Step 2 / Part B — TemporalVectorEncoder
# ---------------------------------------------------------------------------


class TemporalVectorEncoder(nn.Module):
    """
    Adds a temporal axis to the vector-prefix pipeline.

    Pipeline:

        vectors_window: (B, K, MAX_OBJECTS, VECTOR_DIM)
        num_objects_window: (B, K)
        window_len: (B,)

            ── shared VectorPrefixEncoder applied per frame, returning
               hidden_dim prefix (no projection yet) ──>
        per_frame_hidden: (B, K, prefix_len, hidden_dim)

            ── add learned frame-position embedding along K ──>
            ── per prefix slot, attend across K frames with a small
               TransformerEncoder; key_padding_mask blanks scene-start
               padding slots so the temporal axis only sees real frames ──>
            ── readout: take the *last* frame (slot K-1, the current
               frame), giving a temporally-conditioned prefix in
               hidden_dim ──>
        out_hidden: (B, prefix_len, hidden_dim)

            ── reuse the inner encoder's `to_prefix` projection so the
               output lives in T5 embedding space ──>
        prefix: (B, prefix_len, t5_d_model)   ← matches single-frame shape

    Output shape into T5 is identical to the per-frame encoder, so
    `VectorPrefixT5._build_prefix_inputs` can swap one for the other
    without surgery on the rest of Stage 1.

    Notes:
    - The per-frame encoder is *shared* across the K frames (parameter-
      efficient, forces generalisation across timesteps).
    - Slot position embeddings come from the inner encoder; frame position
      embeddings are added here, on the K axis.
    - The readout takes the *last* frame deliberately so the model learns
      to summarise "what happened up to now" rather than averaging across
      time. Mean-pooling readouts collapse the temporal asymmetry.
    """

    def __init__(
        self,
        cfg: VectorEncoderConfig,
        temporal_window: int = 4,
        temporal_n_layers: int = 2,
        temporal_n_heads: int = 4,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()
        self.cfg = cfg
        if temporal_window < 1:
            raise ValueError(f"temporal_window must be >= 1, got {temporal_window}")
        self.K = int(temporal_window)

        # Shared per-frame encoder (re-used for every frame in the window).
        self.inner = VectorPrefixEncoder(cfg)

        # Frame-position embedding (one per slot in the K-window).
        self.frame_position_embed = nn.Embedding(self.K, cfg.hidden_dim)

        # Temporal transformer: small, attends across K tokens per slot.
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=temporal_n_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=temporal_n_layers)

    @property
    def to_prefix(self) -> nn.Linear:
        """Expose the inner encoder's projection so checkpoint paths that
        save/load this module find the same name."""
        return self.inner.to_prefix

    def forward(
        self,
        vectors_window: torch.Tensor,
        num_objects_window: torch.Tensor,
        window_len: torch.Tensor,
        object_present_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vectors_window:     (B, K, MAX_OBJECTS, VECTOR_DIM)
            num_objects_window: (B, K) long tensor; valid objects per past frame
            window_len:         (B,) long tensor; number of *real* frames in
                                the right-aligned window (1..K). Frames at
                                slots [0 .. K - window_len - 1] are padding.
            object_present_mask: optional (B, K, MAX_OBJECTS) bool. True
                                where slot j of frame k carries a real
                                (identity-tracked) object; False where the
                                anchor identity wasn't visible in that
                                past frame OR the frame is scene-start
                                padding. When None, falls back to the
                                per-sample scene-start padding mask.
                                (Step 4 / Part A.)

        Returns:
            prefix: (B, prefix_len, t5_d_model)
        """
        if vectors_window.dim() != 4:
            raise ValueError(
                f"vectors_window must be 4-D (B, K, M, D); got {tuple(vectors_window.shape)}"
            )
        B, K, M, D = vectors_window.shape
        if K != self.K:
            raise ValueError(
                f"vectors_window has K={K}, but TemporalVectorEncoder was "
                f"configured with K={self.K}"
            )
        device = vectors_window.device

        # 1) Per-frame encode (shared weights). Hidden-dim path so we can
        #    aggregate before the T5 projection.
        vw_flat = vectors_window.reshape(B * K, M, D)
        nw_flat = num_objects_window.reshape(B * K)
        per_frame_hidden = self.inner(
            vw_flat, nw_flat, return_hidden=True
        )  # (B*K, prefix_len, hidden_dim)
        prefix_len = per_frame_hidden.shape[1]
        H = per_frame_hidden.shape[2]
        per_frame_hidden = per_frame_hidden.reshape(B, K, prefix_len, H)

        # 2) Add frame-position embedding along K.
        frame_idx = torch.arange(K, device=device)
        frame_pos = self.frame_position_embed(frame_idx)  # (K, H)
        per_frame_hidden = per_frame_hidden + frame_pos.view(1, K, 1, H)

        # 3) Per prefix slot, attend across K frames. Reshape so each slot
        #    is its own sequence of length K. Each slot's embedding evolves
        #    over time.
        x = per_frame_hidden.permute(0, 2, 1, 3).reshape(B * prefix_len, K, H)

        # 4) Build temporal padding mask.
        # ---------------------------------------------------------------
        # Two cases:
        # (a) object_present_mask is None (Step 2 legacy / sort-by-distance):
        #     mask is per-sample (scene-start padding) — uniform across slots.
        # (b) object_present_mask is provided (Step 4 / Part A):
        #     mask is per-(sample, slot, frame) — object-prefix slots use
        #     the per-object identity mask; global tokens (last `extra_tokens`
        #     positions of the prefix) use the scene-start padding mask.
        # ---------------------------------------------------------------
        slot_idx = torch.arange(K, device=device).unsqueeze(0)              # (1, K)
        threshold = (K - window_len.to(device)).unsqueeze(1)                # (B, 1)
        scene_pad_mask = (slot_idx < threshold)                              # (B, K) True=pad
        # Defensive: never let *all* K frames be masked (would NaN attention);
        # the data path already guarantees window_len >= 1, so slot K-1 is real.
        scene_pad_mask = scene_pad_mask.clone()
        scene_pad_mask[:, K - 1] = False

        tokens_per_object = self.inner.tokens_per_object
        object_prefix_len = self.inner.object_prefix_len   # M * tokens_per_object
        extra_tokens = self.inner.extra_tokens

        if object_present_mask is None:
            # Legacy uniform per-sample mask, expanded to all prefix slots.
            kp_mask = (
                scene_pad_mask.unsqueeze(1)
                .expand(B, prefix_len, K)
                .reshape(B * prefix_len, K)
            )
        else:
            # Build per-slot per-frame mask.
            # object_present_mask: (B, K, M) bool — True = object present.
            # Convert to "padding mask" semantics (True = MASK OUT):
            obj_pad_mask = ~object_present_mask.to(torch.bool)              # (B, K, M)
            # Expand each object j to tokens_per_object prefix slots:
            obj_slots_pad_mask = obj_pad_mask.repeat_interleave(
                tokens_per_object, dim=2
            )                                                                # (B, K, M*tokens_per_object)
            if extra_tokens > 0:
                # Global tokens: scene-start padding mask, same for all global slots.
                # Shape (B, K, extra_tokens)
                global_pad_mask = scene_pad_mask.unsqueeze(2).expand(
                    B, K, extra_tokens
                )
                full_pad_mask = torch.cat(
                    [obj_slots_pad_mask, global_pad_mask], dim=2
                )  # (B, K, prefix_len)
            else:
                full_pad_mask = obj_slots_pad_mask  # (B, K, prefix_len)

            # Defensive: per (sample, slot), guarantee at least one frame
            # unmasked to avoid NaN softmax. Unmask the last frame for any
            # row that would otherwise be fully True.
            all_pad = full_pad_mask.all(dim=1)               # (B, prefix_len)
            if all_pad.any():
                # Unmask the last frame (K-1) on those rows.
                full_pad_mask = full_pad_mask.clone()
                full_pad_mask[:, K - 1, :] = torch.where(
                    all_pad, torch.zeros_like(all_pad), full_pad_mask[:, K - 1, :]
                )

            # Reshape to (B*prefix_len, K) expected by nn.TransformerEncoder:
            # current layout is (B, K, prefix_len) → want (B, prefix_len, K).
            kp_mask = full_pad_mask.permute(0, 2, 1).reshape(B * prefix_len, K)

        x = self.temporal_encoder(x, src_key_padding_mask=kp_mask)
        # (B*prefix_len, K, H)

        # 5) Read out the last (current) frame and project to T5 d_model.
        x = x.reshape(B, prefix_len, K, H)[:, :, K - 1, :]  # (B, prefix_len, H)
        out = self.inner.to_prefix(x)  # (B, prefix_len, t5_d_model)
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
