"""Runtime guards for the N-1..N-5 and W-1..W-4 invariants in the spec §4.5.

These are assert_*() helpers — cheap in prod, strict in tests.
"""
from __future__ import annotations

from .neuroletter import Neuroletter, Phase, Role


def assert_n1_silence_legal(inbound: list[Neuroletter]) -> None:
    """N-1: listen() returning [] is always valid (silence is information)."""
    # Nothing to assert — the contract is that empty is fine.
    return


def assert_n3_role_phase_consistent(
    letter: Neuroletter,
    *,
    strict: bool = True,
) -> None:
    """N-3 strict mode: PREDICTION↔GAMMA and ERROR↔THETA."""
    if not strict:
        return
    expected_phase = Phase.GAMMA if letter.role is Role.PREDICTION else Phase.THETA
    assert letter.phase is expected_phase, (
        f"N-3 violated: role={letter.role.name} with phase={letter.phase.name} "
        f"(expected {expected_phase.name} in strict mode)"
    )


def assert_n4_routing_weight_valid(weight: float, *, pruned: bool) -> None:
    """N-4: post-pruning weights are {0, 1}; pre-pruning they are continuous in [0, 1]."""
    if pruned:
        assert weight in (0.0, 1.0), (
            f"N-4 violated: post-pruning weight must be 0 or 1, got {weight}"
        )
    else:
        assert 0.0 <= weight <= 1.0, (
            f"N-4 violated: pre-pruning weight must be in [0, 1], got {weight}"
        )
