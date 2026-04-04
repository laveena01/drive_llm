---
phase: 1
plan: 01
title: "Complete VectorPrefixEncoder implementation"
wave: 1
depends_on: []
files_modified:
  - llm_driving/vector_encoder.py
  - llm_driving/config.py
requirements_addressed: [VENC-01, VENC-02, VENC-03, VENC-04]
autonomous: true
---

# Plan 01: Complete VectorPrefixEncoder Implementation

## Objective

Implement the full `VectorPrefixEncoder.__init__()` method and update `forward()` to use learned query tokens with cross-attention instead of mean pooling. Update config.py with correct constants (PREFIX_LEN=64, add NUM_TYPES and TYPE_EMBED_DIM).

## must_haves

1. VectorPrefixEncoder can be instantiated from VectorEncoderConfig
2. forward() takes (B, MAX_OBJECTS, VECTOR_DIM) + num_objects → returns (B, PREFIX_LEN, t5_d_model)
3. Padding mask excludes zero-padded object slots from attention
4. Learned type embedding for categorical type_id
5. Learned query tokens with cross-attention for pooling

## Tasks

<task id="01.1">
<title>Update config.py constants</title>
<read_first>
- llm_driving/config.py (current constants)
</read_first>
<action>
Update config.py vector-prefix section:

1. Change `PREFIX_LEN = 16` to `PREFIX_LEN = 64` (paper-faithful, discussed in Phase 1 context)
2. Add `T5_D_MODEL = 768` constant (flan-t5-base hidden size)
3. Add `NUM_OBJECT_TYPES = 4` (car=0, pedestrian=1, traffic_light=2, object=3)
4. Add `TYPE_EMBED_DIM = 16` (type embedding dimension)
5. Add `VECTOR_ENCODER_CONFIG` dict that aggregates all encoder hyperparams:

```python
VECTOR_ENCODER_CONFIG = dict(
    max_objects=MAX_OBJECTS,        # 10
    vector_dim=VECTOR_DIM,         # 8
    hidden_dim=VEC_ENCODER_HIDDEN, # 256
    prefix_len=PREFIX_LEN,         # 64
    t5_d_model=T5_D_MODEL,        # 768
    n_layers=VEC_ENCODER_LAYERS,   # 2
    n_heads=VEC_ENCODER_HEADS,     # 4
    dropout=VEC_ENCODER_DROPOUT,   # 0.1
    num_types=NUM_OBJECT_TYPES,    # 4
    type_embed_dim=TYPE_EMBED_DIM, # 16
)
```
</action>
<acceptance_criteria>
- config.py contains `PREFIX_LEN = 64`
- config.py contains `T5_D_MODEL = 768`
- config.py contains `NUM_OBJECT_TYPES = 4`
- config.py contains `TYPE_EMBED_DIM = 16`
- config.py contains `VECTOR_ENCODER_CONFIG = dict(`
- VECTOR_ENCODER_CONFIG includes all 10 keys: max_objects, vector_dim, hidden_dim, prefix_len, t5_d_model, n_layers, n_heads, dropout, num_types, type_embed_dim
</acceptance_criteria>
</task>

<task id="01.2">
<title>Update VectorEncoderConfig dataclass</title>
<read_first>
- llm_driving/vector_encoder.py (current stub)
- llm_driving/config.py (after 01.1 updates)
</read_first>
<action>
Update the VectorEncoderConfig dataclass in vector_encoder.py to include the two new fields:

```python
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
    num_types: int = 4
    type_embed_dim: int = 16
    debug: bool = False
```

Two new fields added: `num_types` (default 4) and `type_embed_dim` (default 16).
</action>
<acceptance_criteria>
- VectorEncoderConfig contains field `num_types: int = 4`
- VectorEncoderConfig contains field `type_embed_dim: int = 16`
- All 11 fields present in the dataclass
</acceptance_criteria>
</task>

<task id="01.3">
<title>Implement VectorPrefixEncoder.__init__</title>
<read_first>
- llm_driving/vector_encoder.py (current stub — forward() logic for reference)
- llm_driving/config.py (hyperparameters)
- Official Wayve encoder: github.com/wayveai/Driving-with-LLMs/blob/main/models/vector_encoder.py (reference for query token pattern)
</read_first>
<action>
Implement the `__init__` method of VectorPrefixEncoder with these modules:

```python
from math import sqrt

class VectorPrefixEncoder(nn.Module):
    def __init__(self, cfg: VectorEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # --- Type embedding for categorical type_id ---
        self.type_embedding = nn.Embedding(cfg.num_types, cfg.type_embed_dim)

        # --- Object projection: (vector_dim - 1 + type_embed_dim) -> hidden_dim ---
        # Subtract 1 because type_id is removed from continuous dims and replaced with embedding
        continuous_dim = cfg.vector_dim - 1  # 7 continuous features
        self.obj_in = nn.Linear(continuous_dim + cfg.type_embed_dim, cfg.hidden_dim)

        # --- Transformer encoder for object-level processing ---
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
```

Key design decisions:
- `obj_in` input size is `(7 + 16) = 23` (7 continuous dims + 16-dim type embedding)
- `norm_first=True` in TransformerEncoderLayer for pre-norm (better training stability)
- `activation="gelu"` (modern default)
- `dim_feedforward=hidden_dim * 4` (standard ratio)
- `query_tokens` initialized with `/ sqrt(hidden_dim)` (like paper's route_embedding)
- `cross_attn_norm` added for post-cross-attention layer normalization
- `to_prefix` is a simple per-token projection `(256 -> 768)`, NOT a massive reshape linear
</action>
<acceptance_criteria>
- `self.type_embedding` is `nn.Embedding(4, 16)`
- `self.obj_in` is `nn.Linear(23, 256)` (7 continuous + 16 type embed → hidden_dim)
- `self.encoder` is `nn.TransformerEncoder` with 2 layers
- `self.query_tokens` is `nn.Parameter` of shape `(64, 256)`
- `self.cross_attn` is `nn.MultiheadAttention(embed_dim=256, num_heads=4)`
- `self.cross_attn_norm` is `nn.LayerNorm(256)`
- `self.to_prefix` is `nn.Linear(256, 768)`
- Module can be instantiated: `VectorPrefixEncoder(VectorEncoderConfig(**VECTOR_ENCODER_CONFIG))` without error
</acceptance_criteria>
</task>

<task id="01.4">
<title>Update VectorPrefixEncoder.forward() for type embedding and cross-attention</title>
<read_first>
- llm_driving/vector_encoder.py (current forward stub)
- Phase 1 CONTEXT.md decisions D-03 (cross-attention) and D-04 (type embedding)
</read_first>
<action>
Replace the existing forward() method with this implementation:

```python
def forward(self, vectors: torch.Tensor, num_objects: torch.Tensor) -> torch.Tensor:
    """
    Args:
        vectors:     (B, MAX_OBJECTS, VECTOR_DIM) raw object vectors
        num_objects:  (B,) number of valid objects per sample
    Returns:
        prefix:      (B, PREFIX_LEN, t5_d_model) prefix embeddings for T5
    """
    B, M, D = vectors.shape

    # OPTIONAL debug checks
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
    continuous = vectors[:, :, :-1]                     # (B, M, 7)
    type_ids = vectors[:, :, -1].long().clamp(0, self.cfg.num_types - 1)  # (B, M)
    type_embeds = self.type_embedding(type_ids)         # (B, M, type_embed_dim)

    # --- Concatenate and project to hidden_dim ---
    x = torch.cat([continuous, type_embeds], dim=-1)    # (B, M, 7 + type_embed_dim)
    x = self.obj_in(x)                                 # (B, M, hidden_dim)

    # --- Create padding mask ---
    idxs = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
    key_padding_mask = idxs >= num_objects.clamp(min=0).unsqueeze(1)  # True = pad

    # --- Transformer encoder (self-attention over objects) ---
    x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, M, hidden_dim)

    # --- Cross-attention pooling with learned query tokens ---
    queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, prefix_len, hidden_dim)
    pooled, _ = self.cross_attn(
        query=queries,
        key=x,
        value=x,
        key_padding_mask=key_padding_mask,
    )                                                   # (B, prefix_len, hidden_dim)
    pooled = self.cross_attn_norm(pooled + queries)     # residual + layer norm

    # --- Project to T5 embedding space ---
    out = self.to_prefix(pooled)                        # (B, prefix_len, t5_d_model)
    return out
```

Key changes from the stub:
1. Type embedding: extract type_id, embed, concatenate with continuous features
2. Cross-attention: queries attend to transformer-encoded object features
3. Residual connection + LayerNorm on cross-attention output
4. `to_prefix` is now per-token (256→768), not a reshape linear
</action>
<acceptance_criteria>
- forward() splits vectors into continuous[:, :, :-1] and type_ids[:, :, -1]
- forward() calls self.type_embedding(type_ids)
- forward() calls self.cross_attn(query=queries, key=x, value=x, key_padding_mask=...)
- forward() applies residual connection: pooled + queries
- forward() returns shape (B, 64, 768) for default config
- forward() uses key_padding_mask to exclude padded objects
</acceptance_criteria>
</task>

## Verification

After all tasks complete:

1. **Shape test**: `VectorPrefixEncoder(cfg).forward(torch.randn(2, 10, 8), torch.tensor([5, 3]))` returns shape `(2, 64, 768)`
2. **Gradient test**: `.backward()` on the output sum computes gradients for all parameters
3. **Padding test**: With `num_objects=[1, 10]`, both batch elements produce valid output (no NaN)
4. **Zero objects**: With `num_objects=[0]`, output is valid (clamp(min=0) handles this)
5. **Type clamp**: type_id values > 3 are clamped to 3, negative values clamped to 0

---
*Plan created: 2026-04-04*
