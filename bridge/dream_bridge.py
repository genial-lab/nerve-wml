"""DreamBridge — collect ε traces, encode for kiki_oniric, apply consolidation delta.

Env-gated by DREAM_CONSOLIDATION_ENABLED. When disabled (default), all methods
no-op so downstream code can depend on the bridge unconditionally without
runtime surprises.

Schema v0 for the trace ndarray:
    shape [n_events, 4] with columns [src_wml, dst_wml, code, phase_clock].

Plan 7 Tasks 2-4.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from nerve_core.neuroletter import Neuroletter, Phase, Role


class DreamBridge:
    """Bridge between a nerve-wml run and a kiki_oniric consolidation cycle."""

    def __init__(self, *, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = os.environ.get("DREAM_CONSOLIDATION_ENABLED", "0") == "1"
        self.enabled = enabled

    def collect_eps_trace(
        self,
        nerve,
        *,
        duration_ticks: int = 1000,
        dt: float = 1e-3,
    ) -> list[Neuroletter]:
        """Drive a SimNerveAdapter for duration_ticks; record all ε letters.

        Synthetic stimulus: one ε per tick to the first dst-1 pair with
        an active edge. This is enough for schema v0 — real WML emissions
        are a natural upgrade.
        """
        if not self.enabled:
            return []

        trace: list[Neuroletter] = []
        # Find a (src, dst) pair with active routing for synthetic stimulus.
        src_dst = None
        for src in range(nerve.n_wmls):
            for dst in range(nerve.n_wmls):
                if src != dst and nerve.routing_weight(src, dst) == 1.0:
                    src_dst = (src, dst)
                    break
            if src_dst is not None:
                break
        if src_dst is None:
            return []

        src, dst = src_dst
        nerve.set_phase_active(gamma=False, theta=True)
        for i in range(duration_ticks):
            nerve.send(Neuroletter(i % 64, Role.ERROR, Phase.THETA, src, dst, nerve.time()))
            nerve.tick(dt)
            for letter in nerve.listen(wml_id=dst):
                if letter.role is Role.ERROR:
                    trace.append(letter)
        return trace

    def to_dream_input(self, trace: list[Neuroletter]) -> np.ndarray:
        """Encode a list of Neuroletters into the schema v0 ndarray.

        Columns: [src_wml, dst_wml, code, phase_clock].
        phase_clock = round(timestamp * Nerve.GAMMA_HZ) so it's locale-free.
        """
        if not self.enabled or not trace:
            return np.zeros((0, 4), dtype=np.int32)

        GAMMA_HZ = 40.0  # noqa: N806 (matches Nerve.GAMMA_HZ constant)
        rows = []
        for letter in trace:
            rows.append([
                letter.src,
                letter.dst,
                letter.code,
                int(round(letter.timestamp * GAMMA_HZ)),
            ])
        return np.asarray(rows, dtype=np.int32)

    def apply_consolidation_output(
        self,
        nerve,
        delta: np.ndarray,
        *,
        alpha: float = 0.1,
    ) -> None:
        """Apply the kiki_oniric delta to the nerve's transducers in place.

        delta: shape [n_transducers, 64, 64]. Applied as
            transducer.logits += alpha * delta[i]
        N-4 invariant preserved (logits are continuous during training).
        """
        if not self.enabled:
            return
        if delta is None or delta.size == 0:
            return

        transducer_keys = list(nerve._transducers.keys())
        n = min(len(transducer_keys), delta.shape[0])
        for i in range(n):
            t = nerve._transducers[transducer_keys[i]]
            with torch.no_grad():
                t.logits.data += alpha * torch.from_numpy(delta[i]).float()
