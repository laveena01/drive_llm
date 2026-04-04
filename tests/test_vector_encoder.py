"""Unit tests for VectorPrefixEncoder."""
import pytest
import torch
from llm_driving.vector_encoder import VectorPrefixEncoder, VectorEncoderConfig
from llm_driving.config import VECTOR_ENCODER_CONFIG


@pytest.fixture
def cfg():
    return VectorEncoderConfig(**VECTOR_ENCODER_CONFIG)


@pytest.fixture
def encoder(cfg):
    model = VectorPrefixEncoder(cfg)
    model.eval()
    return model


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

    def test_gradient_flow(self, cfg):
        """Gradients flow through all encoder parameters."""
        encoder = VectorPrefixEncoder(cfg)
        encoder.train()
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
        vectors[0, 0, -1] = 99.0   # invalid type_id (too high)
        vectors[0, 1, -1] = -5.0   # negative type_id
        num_objects = torch.tensor([5])
        out = encoder(vectors, num_objects)
        assert torch.isfinite(out).all(), "Output should be finite even with invalid type_ids"

    def test_padding_mask_effect(self, encoder, cfg):
        """Padded object slots should not affect output."""
        vectors = torch.randn(2, cfg.max_objects, cfg.vector_dim)
        vectors[1] = vectors[0].clone()
        num_objects = torch.tensor([1, 1])
        out_a = encoder(vectors.clone(), num_objects)

        # Change a padded slot in sample 1 — should not change output
        vectors_modified = vectors.clone()
        vectors_modified[0, 5, :] = 999.0  # slot 5 is padded when num_objects=1
        out_b = encoder(vectors_modified, num_objects)

        # Sample 1 outputs should be identical (padded slots ignored)
        assert torch.allclose(out_a[0], out_b[0], atol=1e-5), \
            "Padded slots affected the output"

    def test_different_num_objects_different_output(self, encoder, cfg):
        """Different num_objects should produce different outputs."""
        vectors = torch.randn(2, cfg.max_objects, cfg.vector_dim)
        vectors[1] = vectors[0].clone()
        num_objects = torch.tensor([1, 10])
        out = encoder(vectors, num_objects)
        assert not torch.allclose(out[0], out[1], atol=1e-3), \
            "Different num_objects should produce different outputs"

    def test_parameter_count(self, encoder):
        """Encoder should have a reasonable parameter count."""
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )
        # With hidden_dim=256, prefix_len=64, t5_d_model=768:
        # Should be roughly 1-3M params
        assert total_params > 100_000, f"Too few parameters: {total_params}"
        assert total_params < 10_000_000, f"Too many parameters: {total_params}"
        assert total_params == trainable_params, "All encoder params should be trainable"
