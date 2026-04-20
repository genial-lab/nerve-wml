import torch

from track_w.mlp_wml import MlpWML
from track_w.streaming_hooks import rollout_mlp_emit_codes


def test_rollout_mlp_emit_codes_shape():
    wml = MlpWML(id=0, input_dim=16, d_hidden=32, seed=0)
    xs = torch.randn(4, 16, 16)  # [B, T, dim]
    codes = rollout_mlp_emit_codes(wml, xs)
    assert codes.shape == (4, 16)
    assert codes.dtype == torch.long
