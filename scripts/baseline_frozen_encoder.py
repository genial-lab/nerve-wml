"""Frozen-encoder baseline for nerve-wml Claim B ablation (review F3).

Isolates the contribution of the nerve-wml VQ protocol from the trivial
alignment expected of any two substrates sharing a task and an encoder.
Two substrates (MLP-head and LIF-head) are trained as linear classifiers
on top of a single FROZEN random encoder. Their output codes (argmax
over 12-class logits) are compared via plug-in MI/H(a). If the resulting
MI/H(a) is comparable to nerve-wml's Test (1) figure of 0.91-0.96, the
claim that nerve-wml's VQ protocol contributes to substrate-agnostic
transmission must be softened.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from track_w.tasks.hard_flow_proxy import HardFlowProxyTask


class _FrozenEncoder(nn.Module):
    """Random 2-layer MLP with frozen parameters, input_dim -> d_hidden."""

    def __init__(self, input_dim: int, d_hidden: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_frozen_baseline(
    seed: int = 0,
    steps: int = 800,
    d_hidden: int = 16,
    return_encoder: bool = False,
    distinct_encoders: bool = False,
) -> dict[str, Any]:
    """Train two trainable heads on frozen encoder(s).

    Two modes:
      * distinct_encoders=False (default): both heads see the SAME
        frozen encoder output (shared frontend control).
      * distinct_encoders=True: each head has its OWN frozen random
        encoder (initialised with different seed). This is the
        "stronger control" of review β: if MI/H is still high under
        independent encoders, Claim B loses even the task-shared
        interpretation. If MI/H drops, Claim B can be reformulated
        as "alignment across distinct encoders via VQ protocol".

    Returns a dict with acc_mlp, acc_lif, codes_mlp, codes_lif.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    encoder_a = _FrozenEncoder(input_dim=16, d_hidden=d_hidden, seed=seed + 100)
    if distinct_encoders:
        encoder_b = _FrozenEncoder(input_dim=16, d_hidden=d_hidden, seed=seed + 200)
    else:
        encoder_b = encoder_a
    encoder_initial = copy.deepcopy(encoder_a) if return_encoder else None

    head_a = nn.Linear(d_hidden, 12)
    head_b = nn.Linear(d_hidden, 12)
    opt = torch.optim.Adam(
        list(head_a.parameters()) + list(head_b.parameters()),
        lr=1e-2,
    )

    for _ in range(steps):
        x, y = task.sample(batch=64)
        with torch.no_grad():
            z_a = encoder_a(x)
            z_b = encoder_b(x) if distinct_encoders else z_a
        logits_a = head_a(z_a)
        logits_b = head_b(z_b)
        loss = F.cross_entropy(logits_a, y) + F.cross_entropy(logits_b, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    eval_task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    x_eval, y_eval = eval_task.sample(batch=1000)
    with torch.no_grad():
        z_a_eval = encoder_a(x_eval)
        z_b_eval = encoder_b(x_eval) if distinct_encoders else z_a_eval
        pred_a = head_a(z_a_eval).argmax(-1)
        pred_b = head_b(z_b_eval).argmax(-1)
        acc_mlp = (pred_a == y_eval).float().mean().item()
        acc_lif = (pred_b == y_eval).float().mean().item()

    result: dict[str, Any] = {
        "acc_mlp":   acc_mlp,
        "acc_lif":   acc_lif,
        "codes_mlp": pred_a.cpu().numpy().astype(np.int64),
        "codes_lif": pred_b.cpu().numpy().astype(np.int64),
    }
    if return_encoder:
        result["encoder_initial"] = encoder_initial
        result["encoder_final"] = encoder_a
    return result


def _run_condition(
    seeds: list[int], steps: int, distinct_encoders: bool,
) -> tuple[list[dict], dict]:
    from nerve_wml.methodology import mi_plugin_discrete, null_model_mi

    per_seed = []
    for s in seeds:
        r = train_frozen_baseline(
            seed=s, steps=steps, distinct_encoders=distinct_encoders,
        )
        mi = mi_plugin_discrete(r["codes_mlp"], r["codes_lif"])
        nm = null_model_mi(
            r["codes_mlp"], r["codes_lif"], n_shuffles=1000, seed=s,
        )
        per_seed.append({
            "seed":      s,
            "acc_mlp":   r["acc_mlp"],
            "acc_lif":   r["acc_lif"],
            "mi_plugin": mi,
            "null_z":    nm.z_score,
            "null_p":    nm.p_value,
        })
    summary = {
        "mi_plugin_mean": float(np.mean([r["mi_plugin"] for r in per_seed])),
        "acc_mlp_mean":   float(np.mean([r["acc_mlp"] for r in per_seed])),
        "acc_lif_mean":   float(np.mean([r["acc_lif"] for r in per_seed])),
    }
    return per_seed, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("papers/paper1/figures/baseline_frozen_encoder.json"),
    )
    args = parser.parse_args()

    print("Running shared-encoder control...")
    shared_per_seed, shared_summary = _run_condition(
        args.seeds, args.steps, distinct_encoders=False,
    )
    print(f"  shared MI/H: {shared_summary['mi_plugin_mean']:.4f}")

    print("Running distinct-encoders control (review β stronger control)...")
    distinct_per_seed, distinct_summary = _run_condition(
        args.seeds, args.steps, distinct_encoders=True,
    )
    print(f"  distinct MI/H: {distinct_summary['mi_plugin_mean']:.4f}")

    delta = shared_summary["mi_plugin_mean"] - distinct_summary["mi_plugin_mean"]

    out = {
        "shared_encoder": {
            "per_seed": shared_per_seed,
            "summary":  shared_summary,
        },
        "distinct_encoders": {
            "per_seed": distinct_per_seed,
            "summary":  distinct_summary,
        },
        "comparison": {
            "shared_minus_distinct":  delta,
            "nerve_wml_test1_range":  "0.91-0.96",
            "interpretation":
                "If distinct_encoders MI/H stays high (>=0.85), Claim B has "
                "no empirical floor -- shared-task-alignment explains "
                "nerve-wml Test (1). If distinct_encoders MI/H drops below "
                "0.60, Claim B can be reformulated as 'alignment across "
                "distinct encoders via VQ protocol', which is defensible.",
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print()
    print("=== Decision matrix ===")
    print(f"  shared MI/H:    {shared_summary['mi_plugin_mean']:.4f}")
    print(f"  distinct MI/H:  {distinct_summary['mi_plugin_mean']:.4f}")
    print(f"  Delta:          {delta:+.4f}")
    print(f"  nerve-wml T(1): 0.91-0.96")
    print()
    if distinct_summary["mi_plugin_mean"] >= 0.85:
        print("  Verdict: distinct MI/H >= 0.85 -> Claim B has no floor")
        print("           -> pivot narrative to 'VQ preserves, not creates'")
    elif distinct_summary["mi_plugin_mean"] < 0.60:
        print("  Verdict: distinct MI/H < 0.60 -> Claim B reformulable as")
        print("           'alignment across distinct encoders via VQ'")
    else:
        print("  Verdict: distinct MI/H in [0.60, 0.85] -> borderline,")
        print("           requires nuanced reformulation")
    print()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
