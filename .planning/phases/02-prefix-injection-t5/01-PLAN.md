---
phase: 2
plan: 01
title: "VectorPrefixT5 wrapper with prefix injection"
wave: 1
depends_on: []
files_modified:
  - llm_driving/vector_prefix_t5.py
requirements_addressed: [PINJ-01, PINJ-02, PINJ-03, PINJ-04, PINJ-05]
autonomous: true
---

# Plan 01: VectorPrefixT5 Wrapper

## Objective

Create a VectorPrefixT5 nn.Module that wraps a FLAN-T5 model and a VectorPrefixEncoder. On forward/generate, it computes vector prefix embeddings, computes text embeddings, concatenates them, extends the attention mask, and passes `inputs_embeds` to T5.

## must_haves

1. Prefix embeddings prepended to text embeddings before T5 encoder
2. Attention mask extended with ones for prefix positions
3. Decoder labels unaffected (encoder-decoder model — labels are decoder-side)
4. model.generate() works with prefix injection
5. Forward pass produces valid output with random weights (smoke test)
