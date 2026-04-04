---
phase: 1
plan: 02
title: "Unit test for VectorPrefixEncoder"
wave: 2
depends_on: [01]
files_modified:
  - tests/test_vector_encoder.py
requirements_addressed: [VENC-05]
autonomous: true
---

# Plan 02: Unit Test for VectorPrefixEncoder

## Objective

Create a comprehensive unit test that validates VectorPrefixEncoder output shapes, gradient flow, padding mask behavior, and edge cases.

## must_haves

1. Test passes with multiple batch sizes (1, 4)
2. Test passes with multiple object counts (0, 1, 5, 10)
3. Gradient flows through all encoder parameters
4. Padding correctly excludes zero-padded slots

## Tasks

<task id="02.1">
<title>Create test_vector_encoder.py</title>
<read_first>
- llm_driving/vector_encoder.py (implemented encoder from Plan 01)
- llm_driving/config.py (VECTOR_ENCODER_CONFIG dict)
</read_first>
<action>
Create `tests/test_vector_encoder.py` with these test cases:

```python
"""Unit tests for VectorPrefixEncoder."""
import pytest
import torch
from llm_driving.vector_encoder import VectorPrefixEncoder, VectorEncoderConfig
from llm_driving.config import VECTOR_ENCODER_CONFIG


@pytest.fixture
def encoder():
    cfg = VectorEncoderConfig(**VECTOR_ENCODER_CONFIG)
    return VectorPrefixEncoder(cfg)


@pytest.fixture
def cfg():
    return VectorEncoderConfig(**VECTOR_ENCODER_CONFIG)


class TestVectorPrefixEncoder:

    def test_output_shape_single(self, encoder, cfg):
        """Output shape is (1, prefix_len, t5_d_model) for single sample."""
        vectors = torch.randn(1, cfg.max_objects, cfg.vector_dim)
        num_objects = torch.tensor([5])
        out = encoder(vectors, num_objects)
        assert out.shape == (1, cfg.prefix_len, cfg.t5_d_model)

    def test_output_shape_batch(self, encoder, cfg):
        """Output shape is (B, prefix_len, t5_d_model) for batched input."""
        B = 4
        vectors = torch.randn(B, cfg.max_objects, cfg.vector_dim)
        num_objects = torch.tensor([3, 7, 1, 10])
        out = encoder(vectors, num_objects)
        assert out.shape == (B, cfg.prefix_len, cfg.t5_d_model)

    def test_gradient_flow(self, encoder, cfg):
        """Gradients flow through all encoder parameters."""
        vectors = torch.randn(2, cfg.max_objects, cfg.vector_dim)
        num_objects = torch.tensor([5, 3])
        out = encoder(vectors, num_objects)
        loss = out.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_zero_objects(self, encoder, cfg):
        """Zero objects should produce valid output (not NaN)."""
        vectors = torch.zeros(1, cfg.max_objects, cfg.vector_dim)
        num_objects = torch.tensor([0])
        out = encoder(vectors, num_objects)
        assert out.shape == (1, cfg.prefix_len, cfg.t5_d_model)
        assert torch.isfinite(out).all(), "Output contains NaN/Inf with zero objects"

    def test_all_objects(self, encoder, cfg):
        """All object slots filled should work."""
        vectors = torch.randn(1, cfg.max_objects, cfg.vector_dim)
        num_objects = torch.tensor([cfg.max_objects])
        out = encoder(vectors, num_objects)
        assert out.shape == (1, cfg.prefix_len, cfg.t5_d_model)
        assert torch.isfinite(out).all()

    def test_type_id_clamping(self, encoder, cfg):
        """type_id values outside valid range are clamped."""
        vectors = torch.randn(1, cfg.max_objects, cfg.vector_dim)
        vectors[0, 0, -1] = 99.0   # invalid type_id
        vectors[0, 1, -1] = -5.0   # negative type_id
        num_objects = torch.tensor([5])
        out = encoder(vectors, num_objects)
        assert torch.isfinite(out).all(), "Output should be finite even with invalid type_ids"

    def test_padding_mask_effect(self, encoder, cfg):
        """Padded object slots should not affect output of valid slots."""
        vectors = torch.randn(2, cfg.max_objects, cfg.vector_dim)
        # Sample 1: only 1 valid object
        # Sample 2: all objects valid but same vectors
        vectors[1] = vectors[0].clone()
        num_objects_a = torch.tensor([1, 1])
        out_a = encoder(vectors.clone(), num_objects_a)

        # Change a padded slot in sample 1 — should not change output
        vectors_modified = vectors.clone()
        vectors_modified[0, 5, :] = 999.0  # slot 5 is padded when num_objects=1
        out_b = encoder(vectors_modified, num_objects_a)

        # Sample 1 outputs should be identical (padded slots ignored)
        assert torch.allclose(out_a[0], out_b[0], atol=1e-5), "Padded slots affected the output"

    def test_different_num_objects_different_output(self, encoder, cfg):
        """Different num_objects should produce different outputs."""
        vectors = torch.randn(2, cfg.max_objects, cfg.vector_dim)
        vectors[1] = vectors[0].clone()
        num_objects = torch.tensor([1, 10])
        out = encoder(vectors, num_objects)
        # Different number of valid objects → different outputs
        assert not torch.allclose(out[0], out[1], atol=1e-3)

    def test_parameter_count(self, encoder):
        """Encoder should have a reasonable parameter count."""
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        # With hidden_dim=256, prefix_len=64, t5_d_model=768:
        # Should be roughly 1-3M params
        assert total_params > 100_000, f"Too few parameters: {total_params}"
        assert total_params < 10_000_000, f"Too many parameters: {total_params}"
        assert total_params == trainable_params, "All encoder params should be trainable"
```

Also create `tests/__init__.py` (empty file) if it doesn't exist.
</action>
<acceptance_criteria>
- tests/test_vector_encoder.py exists
- File contains at least 9 test methods in TestVectorPrefixEncoder class
- Running `python -m pytest tests/test_vector_encoder.py -v` passes all tests
- tests/__init__.py exists
</acceptance_criteria>
</task>

## Verification

Run tests:
```bash
python -m pytest tests/test_vector_encoder.py -v
```

All 9 tests should pass. If any fail, the encoder implementation needs correction.

---
*Plan created: 2026-04-04*
