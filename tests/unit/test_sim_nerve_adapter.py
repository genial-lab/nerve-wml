from bridge.sim_nerve_adapter import SimNerveAdapter
from nerve_core.neuroletter import Neuroletter, Phase, Role


def test_adapter_round_trip():
    nerve = SimNerveAdapter(n_wmls=4, k=2)
    nerve.set_phase_active(gamma=True, theta=False)

    # Find an active edge for the round-trip send.
    active = [(i, j) for i in range(4) for j in range(4)
              if nerve.routing_weight(i, j) == 1.0]
    assert len(active) > 0
    src, dst = active[0]

    nerve.send(Neuroletter(5, Role.PREDICTION, Phase.GAMMA, src, dst, 0.0))
    received = nerve.listen(wml_id=dst)
    assert len(received) == 1


def test_adapter_honors_gamma_priority():
    nerve = SimNerveAdapter(n_wmls=4, k=2)
    nerve.set_phase_active(gamma=True, theta=True)

    # Pick two active edges that both terminate at the same dst.
    active_to = {}
    for i in range(4):
        for j in range(4):
            if nerve.routing_weight(i, j) == 1.0:
                active_to.setdefault(j, []).append(i)
    # Find a dst with at least 2 incoming edges.
    dst = next((d for d, srcs in active_to.items() if len(srcs) >= 2), None)
    if dst is None:
        # Fallback: use any one active edge for each phase, same dst.
        src_a = active_to[list(active_to.keys())[0]][0]
        dst = list(active_to.keys())[0]
        # Even with one src, both phases from same src still exercise gating.
        nerve.send(Neuroletter(5, Role.PREDICTION, Phase.GAMMA, src_a, dst, 0.0))
        nerve.send(Neuroletter(9, Role.ERROR,      Phase.THETA, src_a, dst, 0.0))
    else:
        srcs = active_to[dst]
        nerve.send(Neuroletter(5, Role.PREDICTION, Phase.GAMMA, srcs[0], dst, 0.0))
        nerve.send(Neuroletter(9, Role.ERROR,      Phase.THETA, srcs[1], dst, 0.0))

    # γ and θ both active → γ delivers, θ held.
    delivered = nerve.listen(wml_id=dst)
    assert [l.role for l in delivered] == [Role.PREDICTION]

    # Now turn γ off — θ should deliver.
    nerve.set_phase_active(gamma=False, theta=True)
    delivered = nerve.listen(wml_id=dst)
    assert [l.role for l in delivered] == [Role.ERROR]


def test_adapter_parameters_include_transducers():
    nerve = SimNerveAdapter(n_wmls=4, k=2)
    params = list(nerve.parameters())
    assert len(params) > 0  # router.logits + each transducer
