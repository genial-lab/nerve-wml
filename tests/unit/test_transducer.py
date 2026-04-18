import torch

from track_p.transducer import Transducer


def test_transducer_matrix_shape():
    t = Transducer(alphabet_size=64)
    assert t.logits.shape == (64, 64)


def test_forward_returns_valid_code_index():
    t = Transducer(alphabet_size=64)
    src_code = torch.tensor([5, 17, 42])
    dst_code = t.forward(src_code, hard=True)
    assert dst_code.shape == src_code.shape
    assert (dst_code >= 0).all() and (dst_code < 64).all()


def test_entropy_regularizer_nonzero_for_uniform_start():
    t = Transducer(alphabet_size=64)
    ent = t.entropy()
    # Uniform-ish init → high entropy per row → near log(64)
    assert ent.item() > 3.0  # log(64) ≈ 4.16
