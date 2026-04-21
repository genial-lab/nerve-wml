"""Per-nerve soft transducer mapping src local code → dst local code.

See spec §4.3. Each row of the 64×64 logits matrix is a distribution over
possible target codes. Gumbel-softmax during training, argmax at inference.

Gating variants (issue #5, v1.4.0):
    HARD (default, backward-compatible): forward returns [B] long dst codes
    via argmax. The gradient path is broken at argmax — this is the
    canonical v0.3 bouba_sens B-3-PASS behavior and must not change.
    GUMBEL_SOFTMAX (opt-in): forward returns [B, alphabet_size] soft
    distribution (differentiable, sum to 1 per row). Consumers that want
    a continuous interpolation between target codes during training
    opt in here; bouba_sens Sprint 7+ uses this to probe whether the
    hard 0/1 gating is the reason B-2 Me3 delta stays under threshold
    (ADR-0009 hypothesis 2).
"""
from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class TransducerGating(str, Enum):
    """Gating mode selector for :class:`Transducer`.

    Using ``str`` as the base makes the enum YAML/JSON-serializable without
    a custom encoder, which matters for bouba_sens grid configs.
    """

    HARD = "hard"
    GUMBEL_SOFTMAX = "gumbel_softmax"


class Transducer(nn.Module):
    """Learnable src → dst code transducer.

    Parameters
    ----------
    alphabet_size
        Size of the src/dst code alphabets (square logits matrix).
    init_scale
        Std of the Gaussian init applied to logits; kept small to stay
        near uniform and avoid premature collapse.
    gating
        :class:`TransducerGating` mode. ``HARD`` (default) preserves the
        pre-v1.4 behavior: argmax in ``forward`` returns integer dst
        codes and the gradient path is broken. ``GUMBEL_SOFTMAX`` makes
        ``forward`` return the differentiable soft distribution so
        downstream consumers can keep gradient flow through the code
        axis (issue #5).
    gumbel_tau
        Temperature for Gumbel-softmax. Only read when
        ``gating == GUMBEL_SOFTMAX`` (or when ``forward`` is called with
        ``hard=False`` explicitly). Low values sharpen toward the hard
        argmax; high values smooth toward uniform.
    """

    def __init__(
        self,
        alphabet_size: int = 64,
        init_scale: float = 0.1,
        *,
        gating: TransducerGating = TransducerGating.HARD,
        gumbel_tau: float = 1.0,
    ) -> None:
        super().__init__()
        self.alphabet_size = alphabet_size
        self.gating = TransducerGating(gating)
        self.gumbel_tau = float(gumbel_tau)
        # Near-uniform init to avoid premature collapse.
        self.logits = nn.Parameter(torch.randn(alphabet_size, alphabet_size) * init_scale)

    def forward(
        self,
        src_code: Tensor,
        *,
        hard: bool | None = None,
        tau: float | None = None,
    ) -> Tensor:
        """Map src codes to dst codes (or a dst code distribution).

        Parameters
        ----------
        src_code
            ``[B]`` long tensor of src code indices.
        hard
            Per-call override of the gating mode. When ``None`` (default)
            the instance's ``self.gating`` decides: ``HARD`` → argmax path,
            ``GUMBEL_SOFTMAX`` → soft distribution path. Pass ``True`` /
            ``False`` explicitly to force one path regardless of gating.
        tau
            Per-call override of ``self.gumbel_tau``. Only used on the
            soft path.

        Returns
        -------
        Tensor
            ``[B]`` long (argmax) on the hard path, or
            ``[B, alphabet_size]`` float (differentiable distribution,
            ``sum == 1`` per row) on the soft path.
        """
        row_logits = self.logits[src_code]  # [B, alphabet_size]
        effective_tau = self.gumbel_tau if tau is None else float(tau)
        if hard is None:
            effective_hard = self.gating == TransducerGating.HARD
        else:
            effective_hard = bool(hard)
        y = F.gumbel_softmax(row_logits, tau=effective_tau, hard=effective_hard)
        if effective_hard:
            return y.argmax(dim=-1)
        return y

    def entropy(self) -> Tensor:
        """Row-wise Shannon entropy of the transducer distribution.

        Higher = more uniform (used as a regularizer to avoid collapse to identity).
        Returns the mean entropy across all rows.
        """
        p = F.softmax(self.logits, dim=-1)                     # [size, size]
        ent_per_row = -(p * (p + 1e-9).log()).sum(dim=-1)
        return ent_per_row.mean()
