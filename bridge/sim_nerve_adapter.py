"""SimNerveAdapter — unify the Nerve API used by Track-W WMLs.

Track-W was developed against MockNerve, which exposes
`set_phase_active(gamma, theta)` so pilots can control the gate windows
directly. SimNerve uses γ/θ oscillators. The adapter keeps the
oscillator-driven gate as the default but also exposes set_phase_active
so Gate M can reuse Track-W's training loop verbatim.

Also wires up per-edge transducers — the only module that fine-tunes
during merge training. WML internals stay frozen.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor

from nerve_core.invariants import assert_n3_role_phase_consistent
from nerve_core.neuroletter import Neuroletter, Phase, Role
from track_p.router import SparseRouter
from track_p.transducer import Transducer


class SimNerveAdapter(torch.nn.Module):
    """SimNerve-backed Nerve with Track-W-compatible phase control.

    Default: manual γ/θ flags via set_phase_active (same as MockNerve).
    """

    ALPHABET_SIZE: int   = 64
    GAMMA_HZ:      float = 40.0
    THETA_HZ:      float = 6.0

    def __init__(self, n_wmls: int, k: int, *, seed: int | None = None) -> None:
        super().__init__()
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        self.n_wmls  = n_wmls
        self.router  = SparseRouter(n_wmls=n_wmls, k=k)
        with torch.no_grad():
            self.router.logits.data = torch.randn(
                n_wmls, n_wmls, generator=gen
            ) * 0.1
        self._edges: Tensor = self.router.sample_edges(
            tau=0.5, hard=True, generator=gen,
        )

        # Per-edge transducer (only active edges — dict indexed by "src_dst").
        self._transducers = torch.nn.ModuleDict()
        for src in range(n_wmls):
            for dst in range(n_wmls):
                if self._edges[src, dst].item() == 1.0:
                    self._transducers[f"{src}_{dst}"] = Transducer(alphabet_size=64)

        self._queues: dict[int, list[Neuroletter]] = defaultdict(list)
        self._clock        = 0.0
        self._gamma_active = True
        self._theta_active = False

    def send(self, letter: Neuroletter) -> None:
        assert_n3_role_phase_consistent(letter, strict=True)
        if self._edges[letter.src, letter.dst].item() == 0:
            return
        # Apply the src→dst transducer.
        key = f"{letter.src}_{letter.dst}"
        if key in self._transducers:
            src_code = torch.tensor([letter.code], dtype=torch.long)
            dst_code = int(self._transducers[key].forward(src_code, hard=True).item())
            letter = Neuroletter(
                code=dst_code, role=letter.role, phase=letter.phase,
                src=letter.src, dst=letter.dst, timestamp=letter.timestamp,
            )
        self._queues[letter.dst].append(letter)

    def listen(
        self,
        wml_id: int,
        role:   Role  | None = None,
        phase:  Phase | None = None,
    ) -> list[Neuroletter]:
        pending = self._queues.get(wml_id, [])
        delivered: list[Neuroletter] = []
        held:      list[Neuroletter] = []
        for letter in pending:
            if letter.phase is Phase.GAMMA and self._gamma_active:
                delivered.append(letter)
            elif letter.phase is Phase.THETA and self._theta_active and not self._gamma_active:
                delivered.append(letter)
            else:
                held.append(letter)
        self._queues[wml_id] = held
        if role is not None:
            delivered = [l for l in delivered if l.role is role]
        if phase is not None:
            delivered = [l for l in delivered if l.phase is phase]
        return delivered

    def time(self) -> float:
        return self._clock

    def tick(self, dt: float) -> None:
        self._clock += dt

    def set_phase_active(self, gamma: bool, theta: bool) -> None:
        self._gamma_active = gamma
        self._theta_active = theta

    def routing_weight(self, src: int, dst: int) -> float:
        return float(self._edges[src, dst].item())
