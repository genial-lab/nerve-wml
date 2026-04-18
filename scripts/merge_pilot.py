"""Merge pilot — Gate M. Swap MockNerve → SimNerveAdapter, fine-tune transducers.

Baseline: train WMLs against MockNerve via the Track-W inner loop.
Merged:   same WMLs, now against SimNerveAdapter after MergeTrainer.
Gate M:   merged accuracy ≥ 95 % of baseline accuracy.
"""
from __future__ import annotations

import torch

from bridge.merge_trainer import MergeTrainer
from bridge.sim_nerve_adapter import SimNerveAdapter
from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.tasks.flow_proxy import FlowProxyTask
from track_w.training import train_wml_on_task


def _eval_accuracy(wml, task, n_samples: int = 256) -> float:
    x, y = task.sample(batch=n_samples)
    with torch.no_grad():
        pred = wml.emit_head_pi(wml.core(x))[:, : task.n_classes].argmax(-1)
    return (pred == y).float().mean().item()


def run_merge_gate() -> dict:
    """Returns {acc_mock_baseline, acc_merged, all_passed}."""
    torch.manual_seed(0)
    task = FlowProxyTask(dim=16, n_classes=4, seed=0)

    # Baseline: train one WML against MockNerve.
    mock = MockNerve(n_wmls=2, k=1, seed=0)
    mock.set_phase_active(gamma=True, theta=False)
    baseline_wml = MlpWML(id=0, d_hidden=16, seed=0)
    train_wml_on_task(baseline_wml, mock, task, steps=300, lr=1e-2)
    acc_mock = _eval_accuracy(baseline_wml, task)

    # Merged: take the baseline WML, swap in SimNerveAdapter, fine-tune transducers.
    sim   = SimNerveAdapter(n_wmls=2, k=1, seed=0)
    sim.set_phase_active(gamma=True, theta=False)
    MergeTrainer(wmls=[baseline_wml], nerve=sim, task=task,
                 steps=60, lr=1e-2).train()
    acc_merged = _eval_accuracy(baseline_wml, task)

    all_passed = (acc_merged / max(acc_mock, 1e-6)) >= 0.95

    return {
        "acc_mock_baseline": acc_mock,
        "acc_merged":        acc_merged,
        "all_passed":        all_passed,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_merge_gate(), indent=2))
