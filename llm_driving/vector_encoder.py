# llm_driving/vector_encoder.py

from __future__ import annotations
from dataclasses import dataclass
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
    debug: bool = False   # <-- OPTIONAL


class VectorPrefixEncoder(nn.Module):
    ...
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
