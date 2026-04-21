"""Ablation of the gamma/theta gate (invariant N-3: role==ERROR iff phase==THETA).

Runs the W2-hard scaling law (N=16/32/64, 5 seeds each) twice:
one with strict_n3=True (baseline, Bastos-Friston 2012 cortical
microcircuit), one with strict_n3=False (control, gate decorative).

Decision threshold for "the gamma/theta gate carries empirical weight":
  |Delta_median_gap| > 0.5 percentage points at N>=32, directionally
  consistent across the two points.

Does NOT modify any runner. Uses a context manager to monkey-patch
MockNerve.__init__ locally -- runners remain bit-identical on master.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

from track_w.mock_nerve import MockNerve
from scripts.track_w_pilot import (
    run_w2_hard_n16_multiseed,
    run_w2_hard_n32_multiseed,
    run_w2_hard_n64_multiseed,
)


@contextlib.contextmanager
def _force_strict_n3(value: bool):
    """Temporarily force MockNerve(strict_n3=value) for any construction."""
    original = MockNerve.__init__

    def patched(self, *args, **kwargs):
        kwargs["strict_n3"] = value
        return original(self, *args, **kwargs)

    MockNerve.__init__ = patched
    try:
        yield
    finally:
        MockNerve.__init__ = original


def _collect(strict: bool, seeds: list[int]) -> dict:
    runners = {
        "N=16": run_w2_hard_n16_multiseed,
        "N=32": run_w2_hard_n32_multiseed,
        "N=64": run_w2_hard_n64_multiseed,
    }
    with _force_strict_n3(strict):
        out = {label: fn(seeds=seeds) for label, fn in runners.items()}
    return out


def main() -> None:
    seeds = list(range(5))
    print(f"Ablation N-3 -- 2 conditions x 3 N-points x {len(seeds)} seeds")
    print("This takes ~10-15 min on CPU.")
    print()

    print("[1/2] Running baseline (strict_n3=True, Bastos-Friston gate ON)...")
    baseline = _collect(strict=True, seeds=seeds)

    print("[2/2] Running control (strict_n3=False, gate OFF)...")
    control = _collect(strict=False, seeds=seeds)

    rows = []
    for label in baseline:
        b, c = baseline[label], control[label]
        rows.append({
            "N": label,
            "median_gap_strict": b["median_gap"],
            "median_gap_open":   c["median_gap"],
            "delta_median_gap":  c["median_gap"] - b["median_gap"],
            "mean_acc_mlp_strict": b["mean_acc_mlp"],
            "mean_acc_mlp_open":   c["mean_acc_mlp"],
            "delta_mean_acc_mlp":  c["mean_acc_mlp"] - b["mean_acc_mlp"],
            "mean_acc_lif_strict": b["mean_acc_lif"],
            "mean_acc_lif_open":   c["mean_acc_lif"],
            "delta_mean_acc_lif":  c["mean_acc_lif"] - b["mean_acc_lif"],
        })

    out_dir = Path("papers/paper1/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_n3.json"
    out_path.write_text(
        json.dumps(
            {"baseline": baseline, "control": control, "delta": rows},
            indent=2,
        )
    )

    print()
    header = (
        f"{'N':<6}{'gap_strict':>12}{'gap_open':>12}{'d_gap':>10}"
        f"{'acc_mlp_s':>12}{'acc_mlp_o':>12}{'d_mlp':>10}"
        f"{'acc_lif_s':>12}{'acc_lif_o':>12}{'d_lif':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['N']:<6}"
            f"{r['median_gap_strict']:>12.4f}{r['median_gap_open']:>12.4f}"
            f"{r['delta_median_gap']:>+10.4f}"
            f"{r['mean_acc_mlp_strict']:>12.4f}{r['mean_acc_mlp_open']:>12.4f}"
            f"{r['delta_mean_acc_mlp']:>+10.4f}"
            f"{r['mean_acc_lif_strict']:>12.4f}{r['mean_acc_lif_open']:>12.4f}"
            f"{r['delta_mean_acc_lif']:>+10.4f}"
        )

    big_rows = [r for r in rows if r["N"] in ("N=32", "N=64")]
    any_sig = any(abs(r["delta_median_gap"]) > 0.005 for r in big_rows)
    directional = all(
        (r["delta_median_gap"] > 0) == (big_rows[0]["delta_median_gap"] > 0)
        for r in big_rows
    )

    print()
    print(f"Output: {out_path}")
    print(
        f"Verdict: |d_gap| > 0.5 pp at N>=32 -> "
        f"{'SIGNAL' if any_sig else 'NO SIGNAL'} | "
        f"direction consistent -> {'YES' if directional else 'NO'}"
    )
    if any_sig and directional:
        print(
            "-> gate carries empirical weight. Proceed to Platonic RH (#4)."
        )
    else:
        print(
            "-> gate appears decorative. Repositioning on Bastos-Friston "
            "needs stronger evidence or different task."
        )


if __name__ == "__main__":
    main()
