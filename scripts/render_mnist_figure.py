"""Render the paper's MNIST §Real-Data figure."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.run_mnist_pilots import run_mnist_mi, run_mnist_polymorphism  # noqa: E402

OUT = Path("papers/paper1/figures")


def main() -> None:
    poly = run_mnist_polymorphism(n_wmls=16, steps=300, seeds=list(range(3)))
    mi = run_mnist_mi(seeds=list(range(3)), steps=300, batch=1024)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    ax1.bar(
        [f"seed {s}" for s in poly["seeds"]],
        [g * 100 for g in poly["gaps"]],
        color="#1f5fa3", alpha=0.85,
    )
    ax1.axhline(5.0, color="#d96941", linestyle="--", label="5 % contract")
    ax1.axhline(8.0, color="gray", linestyle=":", label="8 % honest bound")
    ax1.set_ylabel(r"MLP $\leftrightarrow$ LIF gap on MNIST (%)")
    median_pct = poly["median_gap"] * 100
    ax1.set_title(f"(a) Claim A on MNIST — median {median_pct:.2f} %")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(
        [f"seed {r['seed']}" for r in mi],
        [r["mi_over_h"] for r in mi],
        color="#5fa341", alpha=0.85,
    )
    ax2.axhline(0.5, color="gray", linestyle="--", label="50 % threshold")
    mean_ratio = float(np.mean([r["mi_over_h"] for r in mi]))
    ax2.set_ylabel(r"MI / H(MLP) on MNIST")
    ax2.set_title(f"(b) Claim B on MNIST — mean {mean_ratio:.2f}")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "mnist_scaling.pdf", bbox_inches="tight")
    fig.savefig(OUT / "mnist_scaling.png", dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT / 'mnist_scaling.pdf'}")


if __name__ == "__main__":
    main()
