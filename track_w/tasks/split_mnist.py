"""SplitMnistLikeTask — two disjoint 2-class flow_proxy subtasks.

Used by W4 to measure forgetting. Task 0 uses classes {0, 1}; Task 1 uses
{2, 3}. After training on Task 0 then Task 1, we measure retention on Task 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .flow_proxy import FlowProxyTask


@dataclass
class SplitMnistLikeTask:
    seed: int = 0
    dim: int = 16
    subtasks: list = field(init=False)

    def __post_init__(self) -> None:
        # Two flow-proxy tasks with non-overlapping label ranges, implemented
        # by using different class counts and re-labelling at sample time.
        self.subtasks = [
            _LabelOffsetTask(FlowProxyTask(dim=self.dim, n_classes=2, seed=self.seed), offset=0),
            _LabelOffsetTask(FlowProxyTask(dim=self.dim, n_classes=2, seed=self.seed + 1), offset=2),
        ]


class _LabelOffsetTask:
    def __init__(self, inner: FlowProxyTask, offset: int) -> None:
        self._inner  = inner
        self._offset = offset
        self.dim     = inner.dim
        self.n_classes = 2

    def sample(self, batch: int = 64):
        x, y = self._inner.sample(batch=batch)
        return x, y + self._offset
