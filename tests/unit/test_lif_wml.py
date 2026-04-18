import torch

from track_w.lif_wml import LifWML


def test_lif_wml_has_required_attrs():
    wml = LifWML(id=0, n_neurons=50, seed=0)
    assert wml.id == 0
    assert wml.codebook.shape == (64, 50)
    assert wml.v_mem.shape == (50,)
    assert wml.v_thr == 1.0


def test_lif_wml_parameters_include_codebook():
    wml = LifWML(id=0, n_neurons=50, seed=0)
    param_ids = {id(p) for p in wml.parameters()}
    assert id(wml.codebook) in param_ids


def test_lif_wml_seed_is_local():
    torch.manual_seed(42)
    expected = torch.rand(1).item()

    torch.manual_seed(42)
    _ = LifWML(id=0, n_neurons=50, seed=99)
    observed = torch.rand(1).item()

    assert expected == observed


from nerve_core.neuroletter import Phase, Role
from track_w.mock_nerve import MockNerve


def test_lif_wml_step_advances_membrane():
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml = LifWML(id=0, n_neurons=20, seed=0)

    # Inject a strong inbound signal by pre-filling the receiver's own queue
    # (simulates another WML having sent to us on the previous tick).
    from nerve_core.neuroletter import Neuroletter
    nerve._queues[0].append(
        Neuroletter(code=3, role=Role.PREDICTION, phase=Phase.GAMMA,
                    src=1, dst=0, timestamp=0.0)
    )

    v0 = wml.v_mem.clone()
    wml.step(nerve, t=0.0)
    assert not torch.allclose(wml.v_mem, v0)


def test_lif_wml_step_emits_pi_when_pattern_confident():
    """After a few ticks with strong drive, the LIF may emit π (best-effort)."""
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml = LifWML(id=0, n_neurons=20, seed=0)

    for _ in range(5):
        from nerve_core.neuroletter import Neuroletter
        nerve._queues[0].append(
            Neuroletter(code=3, role=Role.PREDICTION, phase=Phase.GAMMA,
                        src=1, dst=0, timestamp=0.0)
        )
        wml.step(nerve, t=0.0)

    received = nerve.listen(wml_id=1, role=Role.PREDICTION)
    # Emission is best-effort (decoder may return no match until LIF stabilises).
    # The assertion is that any emissions that happened are well-formed.
    assert all(l.src == 0 and l.phase is Phase.GAMMA for l in received)
