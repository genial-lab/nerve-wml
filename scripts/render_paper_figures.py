"""Regenerate paper figures from frozen golden artefacts.

Figure 1 (cycle_trace.pdf): first 60 cycles of the frozen trace, drawn as
a 2-row heatmap where row 0 = γ deliveries, row 1 = θ deliveries.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_cycle_trace(
    golden_path:   str = "tests/golden/cycle_trace_4wmls_seed0.npz",
    output_path:   str = "papers/paper1/figures/cycle_trace.pdf",
    n_to_show:     int = 60,
) -> None:
    data = np.load(golden_path)["codes"]
    shown = data[:n_to_show]

    # Column 0 = γ delivery; column 1 = θ delivery (per freeze_golden.py structure).
    row_gamma = (shown[:, 0] >= 0).astype(int)
    row_theta = (
        (shown[:, 1] >= 0).astype(int)
        if shown.shape[1] > 1
        else np.zeros(n_to_show, dtype=int)
    )

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.imshow(np.stack([row_gamma, row_theta]), aspect="auto", cmap="Greys")
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$\gamma$ (predictions)", r"$\theta$ (errors)"])
    ax.set_xlabel("cycle")
    ax.set_title("γ/θ multiplexing — first 60 cycles (golden trace)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_w4_forgetting_bars(
    *,
    output_path: str = "papers/paper1/figures/w4_forgetting.pdf",
    n_seeds:     int = 3,
    steps:       int = 400,
) -> None:
    """Figure 2: W4 forgetting across three regimes (baseline / shared / rehearsal)."""
    import torch
    from scripts.track_w_pilot import run_w4, run_w4_multi_seed

    # Baseline (disjoint heads): deterministic across seeds, so measure once per seed.
    baseline_forgetting = []
    for s in range(n_seeds):
        torch.manual_seed(s)
        r = run_w4(steps=steps)
        baseline_forgetting.append(
            (r["acc_task0_initial"] - r["acc_task0_after_task1"])
            / max(r["acc_task0_initial"], 1e-6)
        )

    # Shared / rehearsal: reuse the existing multi-seed helper.
    torch.manual_seed(0)
    shared_report = run_w4_multi_seed(seeds=list(range(n_seeds)), steps=steps)

    regimes = [
        ("Baseline\n(disjoint)",       baseline_forgetting),
        ("Shared head\n(no rehearsal)", shared_report["forgetting_shared"]),
        ("Shared head\n(+30% rehearsal)", shared_report["forgetting_rehearsal"]),
    ]

    means = [float(np.mean(v)) for _, v in regimes]
    errs  = [float(np.std(v))  for _, v in regimes]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    positions = range(len(regimes))
    ax.bar(list(positions), means, yerr=errs, capsize=6,
           color=["#1f77b4", "#d62728", "#2ca02c"])
    ax.set_xticks(list(positions))
    ax.set_xticklabels([label for label, _ in regimes])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Forgetting (task 0 retention loss)")
    ax.set_title("Gate W4: catastrophic forgetting across regimes")
    ax.axhline(y=0.20, color="gray", linestyle="--", linewidth=1,
               label="gate threshold 20%")
    ax.legend(loc="upper right")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_p1_dead_curve(
    *,
    output_path:      str = "papers/paper1/figures/p1_dead_curve.pdf",
    max_steps:        int = 16000,
    checkpoint_every: int = 1000,
) -> None:
    """Figure 3: dead-code fraction vs step, three VQ init variants."""
    import torch
    from scripts.track_p_pilot import run_p1_dead_vs_steps

    torch.manual_seed(0)
    curves = run_p1_dead_vs_steps(
        max_steps=max_steps,
        checkpoint_every=checkpoint_every,
    )

    labels_styles = {
        "mog_init":         ("MOG init (shortcut)",      "#1f77b4", "-"),
        "random_no_rot":    ("Random, no rotation",      "#d62728", "--"),
        "random_with_rot":  ("Random + rotation",        "#2ca02c", "-"),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for variant, (label, color, ls) in labels_styles.items():
        steps = [s for s, _ in curves[variant]]
        dead  = [d for _, d in curves[variant]]
        ax.plot(steps, dead, color=color, linestyle=ls, linewidth=2, label=label)

    ax.axhline(y=0.10, color="gray", linestyle=":", linewidth=1,
               label="gate threshold 10%")
    ax.set_xlabel("training step")
    ax.set_ylabel("dead-code fraction")
    ax.set_ylim(0, 0.7)
    ax.set_title("Gate P1: VQ dead-code convergence by init strategy")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)



def render_w2_histogram(
    *,
    output_path: str = "papers/paper1/figures/w2_histogram.pdf",
    n_seeds:     int = 5,
    steps:       int = 400,
) -> None:
    """Figure 4: MLP vs LIF accuracy histogram across seeds.

    Uses run_w2_multi_seed over [0, ..., n_seeds-1]. Both distributions
    are plotted as overlapping histograms with distinct colours.
    Paper §5 discusses that on FlowProxyTask 4-class both pools saturate
    to 1.0; for variance, a harder task would be needed (noted as follow-up).
    """
    import torch
    from scripts.track_w_pilot import run_w2_multi_seed

    torch.manual_seed(0)
    report = run_w2_multi_seed(seeds=list(range(n_seeds)), steps=steps)

    acc_mlp = np.array(report["acc_mlp"])
    acc_lif = np.array(report["acc_lif"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))

    bins = np.linspace(0, 1, 21)
    ax.hist(acc_mlp, bins=bins, alpha=0.6, label=f"MLP (n={n_seeds})", color="#1f77b4")
    ax.hist(acc_lif, bins=bins, alpha=0.6, label=f"LIF (n={n_seeds})", color="#ff7f0e")

    ax.axvline(acc_mlp.mean(), color="#1f77b4", linestyle="--", linewidth=1,
               label=f"MLP mean {acc_mlp.mean():.3f}")
    ax.axvline(acc_lif.mean(), color="#ff7f0e", linestyle="--", linewidth=1,
               label=f"LIF mean {acc_lif.mean():.3f}")

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("accuracy on FlowProxyTask (4-class)")
    ax.set_ylabel("seed count")
    ax.set_title("Gate W2: MLP vs LIF polymorphie across seeds")
    ax.legend(loc="upper left", fontsize=8)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    render_cycle_trace()
    print("paper figures rendered.")


if __name__ == "__main__":
    main()
