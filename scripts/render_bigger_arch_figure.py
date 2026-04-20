"""Figure: small-arch vs bigger-arch polymorphism gap + MI ratio."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.run_bigger_arch import (  # noqa: E402
    run_hard_bigger_arch_multiseed,
    run_mi_bigger_arch,
)
from scripts.track_w_pilot import run_w2_hard_n16_multiseed  # noqa: E402

OUT = Path("papers/paper1/figures")


def main() -> None:
    small = run_w2_hard_n16_multiseed(seeds=[0, 1, 2], steps=400)
    big = run_hard_bigger_arch_multiseed(seeds=[0, 1, 2], steps=300)
    big_mi = run_mi_bigger_arch(seeds=[0, 1, 2], steps=300, batch=1024)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = ["d_hidden=16", "d_hidden=128"]
    medians = [small["median_gap"], big["median_gap"]]
    maxes   = [small["max_gap"],    big["max_gap"]]
    x = np.arange(2)
    ax1.bar(x - 0.18, [m * 100 for m in medians], 0.36,
            label="median", color="#1f5fa3")
    ax1.bar(x + 0.18, [m * 100 for m in maxes], 0.36,
            label="max", color="#7a2c17")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.axhline(5.0, color="#d96941", linestyle="--", label="5 % contract")
    ax1.set_ylabel("Gap at N=16 (%)")
    ax1.set_title("(a) Architecture scale amplifies the gap")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    ratios = [r["mi_over_h"] for r in big_mi]
    ax2.bar(
        [f"seed {r['seed']}" for r in big_mi], ratios,
        color="#5fa341", alpha=0.85,
    )
    ax2.axhline(0.5, color="gray", linestyle="--", label="50 % threshold")
    mean_ratio = float(np.mean(ratios))
    ax2.set_ylabel(r"MI / H(MLP), d_hidden=128")
    ax2.set_title(f"(b) Claim B survives — mean {mean_ratio:.2f}")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "bigger_arch_scaling.pdf", bbox_inches="tight")
    fig.savefig(OUT / "bigger_arch_scaling.png", dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT / 'bigger_arch_scaling.pdf'}")


if __name__ == "__main__":
    main()
