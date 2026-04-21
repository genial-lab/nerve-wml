"""Cross-estimator MI robustness check.

Loads the NPZ produced by save_codes_for_checks.py (which now stores
both the argmax codes and the pre-VQ continuous embeddings) and
computes MI between MLP and LIF through three estimators:

  1. Plug-in entropy-normalised MI/H(a) on discrete codes
     (= the paper's baseline estimator).
  2. Miller-Madow bias-corrected plug-in on discrete codes.
  3. Kraskov-Stogbauer-Grassberger (KSG) k-NN on continuous
     pre-VQ embeddings; result in nats, reported both raw and
     normalised by H(mlp_codes) for direct comparison.

If the three estimators agree within epsilon, the MI/H claim is
cross-estimator robust per bouba_sens section 6.3 methodology.

GROSMAC-SAFE policy-wise (pure numpy + scipy), but KSG on N=5000
samples uses roughly 600 MB RAM; run on Tower or kxkm-ai.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nerve_wml.methodology.mi_estimators import (
    entropy_discrete,
    mi_kraskov_ksg_continuous,
    mi_miller_madow_discrete,
    mi_plugin_discrete,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--k", type=int, default=3,
                        help="Kraskov k-NN neighbours (default 3)")
    parser.add_argument(
        "--n-kraskov",
        type=int,
        default=2000,
        help="Subsample size for KSG (default 2000 to keep RAM ~100MB)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("papers/paper1/figures/mi_multi_estimator.json"),
    )
    args = parser.parse_args()

    if not args.codes.exists():
        raise FileNotFoundError(
            f"{args.codes} not found. Regenerate via "
            "scripts/save_codes_for_checks.py on Tower/kxkm-ai."
        )

    data = np.load(args.codes)
    mlp_codes = data["mlp_codes"]
    lif_codes = data["lif_codes"]
    if "mlp_embeddings" not in data or "lif_embeddings" not in data:
        raise KeyError(
            "NPZ is missing pre-VQ embeddings. Regenerate with the "
            "updated save_codes_for_checks.py."
        )
    mlp_emb = data["mlp_embeddings"]
    lif_emb = data["lif_embeddings"]

    per_seed = []
    for seed_idx, s in enumerate(args.seeds):
        codes_a = mlp_codes[seed_idx].astype(np.int64)
        codes_b = lif_codes[seed_idx].astype(np.int64)
        emb_a = mlp_emb[seed_idx].astype(np.float64)
        emb_b = lif_emb[seed_idx].astype(np.float64)

        rng = np.random.default_rng(s)
        n_total = emb_a.shape[0]
        if n_total > args.n_kraskov:
            idx = rng.choice(n_total, size=args.n_kraskov, replace=False)
            emb_a_sub = emb_a[idx]
            emb_b_sub = emb_b[idx]
        else:
            emb_a_sub = emb_a
            emb_b_sub = emb_b

        mi_plugin = mi_plugin_discrete(codes_a, codes_b)
        mi_mm = mi_miller_madow_discrete(codes_a, codes_b)
        mi_ksg_nats = mi_kraskov_ksg_continuous(emb_a_sub, emb_b_sub, k=args.k)
        h_a_nats = entropy_discrete(codes_a)
        mi_ksg_norm = mi_ksg_nats / h_a_nats if h_a_nats > 0 else 0.0

        per_seed.append({
            "seed":           s,
            "mi_plugin":      mi_plugin,
            "mi_miller_madow": mi_mm,
            "mi_kraskov_nats": mi_ksg_nats,
            "mi_kraskov_over_h_a": mi_ksg_norm,
            "h_a_nats":        h_a_nats,
            "n_kraskov":       min(n_total, args.n_kraskov),
        })

    plugin_mean = float(np.mean([r["mi_plugin"] for r in per_seed]))
    mm_mean = float(np.mean([r["mi_miller_madow"] for r in per_seed]))
    ksg_norm_mean = float(np.mean([r["mi_kraskov_over_h_a"] for r in per_seed]))

    discrete_delta = abs(plugin_mean - mm_mean)
    discrete_robust = discrete_delta < 0.10
    if ksg_norm_mean > 1e-9:
        amplification_ratio = plugin_mean / ksg_norm_mean
    else:
        amplification_ratio = float("inf")

    summary = {
        "discrete_plugin_mean":        plugin_mean,
        "discrete_miller_madow_mean":  mm_mean,
        "discrete_delta":              discrete_delta,
        "discrete_robust":             discrete_robust,
        "discrete_epsilon":            0.10,
        "continuous_ksg_over_ha_mean": ksg_norm_mean,
        "vq_amplification_ratio":      amplification_ratio,
        "n_seeds":                     len(args.seeds),
        "k":                           args.k,
        "n_kraskov":                   args.n_kraskov,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"per_seed": per_seed, "summary": summary}, indent=2)
    )

    print(
        f"Multi-estimator MI check -- {len(args.seeds)} seeds, k={args.k}, "
        f"n_kraskov={args.n_kraskov}"
    )
    print()
    header = (
        f"{'seed':>6}{'plugin':>10}{'miller_mm':>12}"
        f"{'ksg_nats':>10}{'ksg/h(a)':>12}{'h(a)_nats':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in per_seed:
        print(
            f"{r['seed']:>6}"
            f"{r['mi_plugin']:>10.4f}"
            f"{r['mi_miller_madow']:>12.4f}"
            f"{r['mi_kraskov_nats']:>10.4f}"
            f"{r['mi_kraskov_over_h_a']:>12.4f}"
            f"{r['h_a_nats']:>12.4f}"
        )
    print()
    print(
        f"Means: plugin={plugin_mean:.4f}, MM={mm_mean:.4f}, "
        f"KSG/H(a)={ksg_norm_mean:.4f}"
    )
    print()
    print(
        f"Discrete cross-estimator (plugin vs Miller-Madow): "
        f"Delta={discrete_delta:.4f}"
    )
    print(
        f"  -> {'ROBUST (within 0.10)' if discrete_robust else 'DIVERGENT'}"
    )
    print()
    print(
        f"VQ compression amplification (post-VQ / pre-VQ): "
        f"{amplification_ratio:.2f}x"
    )
    print(
        "  -> discrete codes carry ~"
        f"{amplification_ratio:.1f}x more shared-information fraction"
    )
    print("     than the continuous pre-VQ embeddings.")
    print()
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
