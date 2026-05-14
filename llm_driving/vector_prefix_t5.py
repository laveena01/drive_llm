# llm_driving/vector_prefix_t5.py
"""
VectorPrefixT5 — wraps FLAN-T5 + VectorPrefixEncoder.

Injects learned vector prefix embeddings into the T5 encoder input using
the `inputs_embeds` mechanism. No model surgery required.

Architecture:
    vectors --> VectorPrefixEncoder --> prefix_embeds (B, 64, 768)
    text    --> T5.shared (embedding) --> text_embeds  (B, L, 768)

    [prefix_embeds ; text_embeds] --> T5 encoder --> T5 decoder --> output

    Attention mask is extended with 1s for prefix positions.
    Labels (decoder-side) are unaffected by prefix.
"""

from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_driving.vector_encoder import (
    TemporalVectorEncoder,
    VectorPrefixEncoder,
    VectorEncoderConfig,
)
from llm_driving import config as cfg

logger = logging.getLogger("llm_driving")


class VectorPrefixT5(nn.Module):
    """
    Wraps a T5 model and a VectorPrefixEncoder.

    On forward():
        1. Encode vectors -> prefix embeddings
        2. Embed text input_ids -> text embeddings
        3. Concatenate [prefix ; text] embeddings
        4. Extend attention mask for prefix positions
        5. Forward through T5 with inputs_embeds

    On generate():
        Same prefix injection, then call T5.generate() with inputs_embeds.

    Step 2 / Part B: when ``cfg.USE_TEMPORAL`` is True, the wrapped encoder
    is `TemporalVectorEncoder` and the forward/generate paths consume
    ``vectors_window`` / ``num_objects_window`` / ``window_len`` instead of
    the single-frame ``vectors`` / ``num_objects`` (which are still accepted
    as kwargs for backward compatibility).
    """

    def __init__(
        self,
        model_name: str,
        encoder_config: VectorEncoderConfig,
        tokenizer: Optional[AutoTokenizer] = None,
        use_temporal: Optional[bool] = None,
    ):
        super().__init__()

        # Load base T5 model
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Load or accept tokenizer
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)

        # Decide encoder variant. Default reads `USE_TEMPORAL` from config so
        # an `accelerate launch ... main.py` run with the flag flipped does
        # the right thing without code changes elsewhere. Tests can override
        # explicitly via the constructor arg.
        self.use_temporal = (
            bool(getattr(cfg, "USE_TEMPORAL", False))
            if use_temporal is None
            else bool(use_temporal)
        )

        if self.use_temporal:
            self.vector_encoder = TemporalVectorEncoder(
                cfg=encoder_config,
                temporal_window=int(getattr(cfg, "TEMPORAL_WINDOW", 4)),
                temporal_n_layers=int(getattr(cfg, "TEMPORAL_TRANSFORMER_LAYERS", 2)),
                temporal_n_heads=int(getattr(cfg, "TEMPORAL_TRANSFORMER_HEADS", 4)),
                temporal_dropout=float(getattr(cfg, "TEMPORAL_TRANSFORMER_DROPOUT", 0.1)),
            )
            logger.info(
                "[VectorPrefixT5] Using TemporalVectorEncoder "
                f"(K={getattr(cfg, 'TEMPORAL_WINDOW', 4)}, "
                f"layers={getattr(cfg, 'TEMPORAL_TRANSFORMER_LAYERS', 2)}, "
                f"heads={getattr(cfg, 'TEMPORAL_TRANSFORMER_HEADS', 4)})"
            )
        else:
            self.vector_encoder = VectorPrefixEncoder(encoder_config)
            logger.info("[VectorPrefixT5] Using single-frame VectorPrefixEncoder")

        # Store config
        self.encoder_config = encoder_config
        self.prefix_len = encoder_config.prefix_len

    @property
    def config(self):
        """Expose T5 config for compatibility."""
        return self.t5.config

    @property
    def device(self):
        """Return device of the model parameters."""
        return next(self.parameters()).device

    def _build_prefix_inputs(
        self,
        vectors: Optional[torch.Tensor],
        num_objects: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vectors_window: Optional[torch.Tensor] = None,
        num_objects_window: Optional[torch.Tensor] = None,
        window_len: Optional[torch.Tensor] = None,
        object_present_mask: Optional[torch.Tensor] = None,
    ):
        """
        Build inputs_embeds and extended attention_mask by prepending
        vector prefix embeddings to text embeddings.

        Single-frame path (legacy, ``self.use_temporal=False``): consumes
        ``vectors`` (B, M, D) + ``num_objects`` (B,).

        Temporal path (Step 2 / Part B, ``self.use_temporal=True``): consumes
        ``vectors_window`` (B, K, M, D) + ``num_objects_window`` (B, K) +
        ``window_len`` (B,). The single-frame kwargs may also be passed
        through (e.g. by HF Trainer) but are unused on this path.

        Returns:
            inputs_embeds: (B, prefix_len + seq_len, d_model)
            attention_mask: (B, prefix_len + seq_len)
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # 1. Encode vectors -> prefix embeddings
        if self.use_temporal:
            if vectors_window is None or num_objects_window is None or window_len is None:
                raise ValueError(
                    "VectorPrefixT5 was constructed with use_temporal=True but "
                    "vectors_window / num_objects_window / window_len were not "
                    "provided. Check that the data collator was instantiated "
                    "with temporal_window > 0 and that batch keys are passed "
                    "through the training loop."
                )
            prefix_embeds = self.vector_encoder(
                vectors_window,
                num_objects_window,
                window_len,
                object_present_mask=object_present_mask,
            )
        else:
            if vectors is None or num_objects is None:
                raise ValueError(
                    "VectorPrefixT5 was constructed with use_temporal=False but "
                    "vectors / num_objects were not provided."
                )
            prefix_embeds = self.vector_encoder(vectors, num_objects)
        # prefix_embeds: (B, prefix_len, d_model)

        # 2. Embed text input_ids
        text_embeds = self.t5.shared(input_ids)  # (B, seq_len, d_model)

        # 3. Concatenate [prefix ; text]
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)  # (B, P+L, d_model)

        # 4. Extend attention mask
        if attention_mask is None:
            attention_mask = torch.ones(B, input_ids.shape[1], device=device, dtype=torch.long)

        prefix_mask = torch.ones(B, self.prefix_len, device=device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # (B, P+L)

        return inputs_embeds, extended_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        vectors: Optional[torch.Tensor] = None,
        num_objects: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        vectors_window: Optional[torch.Tensor] = None,
        num_objects_window: Optional[torch.Tensor] = None,
        window_len: Optional[torch.Tensor] = None,
        object_present_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with vector prefix injection.

        Single-frame path (use_temporal=False):
            vectors:         (B, MAX_OBJECTS, VECTOR_DIM) raw object vectors
            num_objects:     (B,) valid object count per sample

        Temporal path (use_temporal=True, Step 2 / Part B):
            vectors_window:     (B, K, MAX_OBJECTS, VECTOR_DIM)
            num_objects_window: (B, K)
            window_len:         (B,)

        Common:
            input_ids:       (B, seq_len) tokenized text input
            attention_mask:  (B, seq_len) text attention mask
            labels:          (B, target_len) decoder target token IDs
            decoder_input_ids: optional decoder input
            decoder_attention_mask: optional decoder attention mask

        Returns:
            Seq2SeqLMOutput with loss (if labels provided) and logits
        """
        inputs_embeds, extended_mask = self._build_prefix_inputs(
            vectors=vectors,
            num_objects=num_objects,
            input_ids=input_ids,
            attention_mask=attention_mask,
            vectors_window=vectors_window,
            num_objects_window=num_objects_window,
            window_len=window_len,
            object_present_mask=object_present_mask,
        )

        # Forward through T5 with inputs_embeds (NOT input_ids)
        return self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        vectors: Optional[torch.Tensor] = None,
        num_objects: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        vectors_window: Optional[torch.Tensor] = None,
        num_objects_window: Optional[torch.Tensor] = None,
        window_len: Optional[torch.Tensor] = None,
        object_present_mask: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ):
        """
        Generate text with vector prefix injection. Accepts either the
        single-frame inputs or the temporal-window inputs depending on
        ``self.use_temporal`` (the encoder enforces the right one).

        Returns:
            Generated token IDs (B, gen_len)
        """
        inputs_embeds, extended_mask = self._build_prefix_inputs(
            vectors=vectors,
            num_objects=num_objects,
            input_ids=input_ids,
            attention_mask=attention_mask,
            vectors_window=vectors_window,
            num_objects_window=num_objects_window,
            window_len=window_len,
            object_present_mask=object_present_mask,
        )

        # Use T5.generate() with inputs_embeds
        return self.t5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            **generate_kwargs,
        )

    def get_encoder_params(self):
        """Return vector encoder parameters (always trainable)."""
        return list(self.vector_encoder.parameters())

    def get_t5_params(self):
        """Return T5 model parameters."""
        return list(self.t5.parameters())

    def freeze_t5_base(self):
        """Freeze all T5 base model parameters."""
        for param in self.t5.parameters():
            param.requires_grad = False
        logger.info("[VectorPrefixT5] Froze T5 base model parameters")

    def unfreeze_t5_base(self):
        """Unfreeze all T5 base model parameters."""
        for param in self.t5.parameters():
            param.requires_grad = True
        logger.info("[VectorPrefixT5] Unfroze T5 base model parameters")

    def print_trainable_parameters(self):
        """Print trainable vs total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100.0 * trainable / total if total > 0 else 0.0
        msg = (
            f"trainable params: {trainable:,} || "
            f"all params: {total:,} || "
            f"trainable%: {pct:.2f}%"
        )
        logger.info(f"[VectorPrefixT5] {msg}")
        print(msg)
        return trainable, total
