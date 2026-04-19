import torch

from track_p.vq_codebook import VQCodebook


def test_codebook_has_correct_shape():
    cb = VQCodebook(size=64, dim=128)
    assert cb.embeddings.shape == (64, 128)


def test_quantize_returns_valid_indices():
    cb = VQCodebook(size=64, dim=128)
    z = torch.randn(8, 128)
    indices, quantized, loss = cb.quantize(z)
    assert indices.shape == (8,)
    assert (indices >= 0).all() and (indices < 64).all()
    assert quantized.shape == z.shape
    assert loss.dim() == 0  # scalar loss


def test_usage_counter_increments_on_quantize():
    cb = VQCodebook(size=64, dim=32, ema=False)
    z = torch.randn(100, 32)
    cb.quantize(z)
    total_usage = cb.usage_counter.sum().item()
    assert total_usage == 100


def test_ema_update_does_not_require_gradient_through_embeddings():
    cb = VQCodebook(size=16, dim=8, ema=True)
    z = torch.randn(32, 8, requires_grad=True)
    _, quantized, _ = cb.quantize(z)
    # Straight-through: gradient must flow back to z even though embeddings
    # are not differentiable leaves under EMA update.
    loss = quantized.sum()
    loss.backward()
    assert z.grad is not None


def test_codebook_rotation_revives_dead_codes():
    """After rotate_dead_codes, dead codes move to live input points
    and become selectable on the next forward. Zeghidour 2022 trick."""
    import torch

    cb = VQCodebook(size=16, dim=8, ema=False)
    cb.usage_counter[:10] = 100
    cb.usage_counter[10:] = 0

    live_before = cb.embeddings[:10].clone()
    dead_before = cb.embeddings[10:].clone()

    z = torch.randn(64, 8) * 0.05

    cb.rotate_dead_codes(z, dead_threshold=10)

    assert torch.allclose(cb.embeddings[:10], live_before)
    assert not torch.allclose(cb.embeddings[10:], dead_before)
