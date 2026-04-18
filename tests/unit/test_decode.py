import torch

from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_w._decode import embed_inbound


def _letter(src: int, dst: int, code: int, role: Role, phase: Phase) -> Neuroletter:
    return Neuroletter(code=code, role=role, phase=phase, src=src, dst=dst, timestamp=0.0)


def test_embed_inbound_empty_returns_zero_vector():
    codebook = torch.randn(64, 128)
    out = embed_inbound([], codebook)
    assert out.shape == (128,)
    assert torch.allclose(out, torch.zeros(128))


def test_embed_inbound_single_letter_returns_code_row():
    codebook = torch.randn(64, 128)
    letter = _letter(src=0, dst=1, code=7, role=Role.PREDICTION, phase=Phase.GAMMA)
    out = embed_inbound([letter], codebook)
    assert torch.allclose(out, codebook[7])


def test_embed_inbound_mean_pools_multiple_letters():
    codebook = torch.randn(64, 128)
    letters = [
        _letter(0, 1, 3,  Role.PREDICTION, Phase.GAMMA),
        _letter(2, 1, 17, Role.PREDICTION, Phase.GAMMA),
    ]
    out = embed_inbound(letters, codebook)
    expected = (codebook[3] + codebook[17]) / 2
    assert torch.allclose(out, expected)
