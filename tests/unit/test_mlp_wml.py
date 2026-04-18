import torch

from track_w.mlp_wml import MlpWML


def test_mlp_wml_has_required_attrs():
    wml = MlpWML(id=0, d_hidden=128, seed=0)
    assert wml.id == 0
    assert wml.codebook.shape == (64, 128)
    assert hasattr(wml, "core")
    assert hasattr(wml, "emit_head_pi")
    assert hasattr(wml, "emit_head_eps")
    assert wml.threshold_eps == 0.30


def test_mlp_wml_parameters_include_codebook_and_core():
    wml = MlpWML(id=0, d_hidden=128, seed=0)
    param_ids = {id(p) for p in wml.parameters()}
    assert id(wml.codebook) in param_ids
    # At least one linear in core should be a parameter.
    core_params = [p for p in wml.core.parameters()]
    assert len(core_params) > 0
    assert all(id(p) in param_ids for p in core_params)


def test_mlp_wml_seed_is_local():
    """Constructing an MlpWML must NOT mutate the global torch RNG."""
    torch.manual_seed(42)
    expected = torch.rand(1).item()

    torch.manual_seed(42)
    _ = MlpWML(id=0, d_hidden=128, seed=99)
    observed = torch.rand(1).item()

    assert expected == observed
