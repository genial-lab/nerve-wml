"""Direct measurement of inter-substrate information transmission.

Three tests that the v0.7 scaling law did NOT address:

  (1) Mutual information I(code_MLP(x) ; code_LIF(x)) for x ~ task.
      Tests whether two substrates independently trained on the same
      task encode COMMON information, or merely reach comparable
      accuracy via orthogonal codes.

  (2) Round-trip fidelity:
        x → MLP.emit → transducer(M→L) → LIF.receive → LIF.emit
          → transducer(L→M) → MLP.receive → argmax
      vs baseline: x → MLP.argmax. Measures how much task-relevant
      information survives a cross-substrate pass.

  (3) Cross-substrate merge: freeze a trained MLP. Train ONLY the
      transducer MLP→LIF with task supervision; keep LIF's readout
      frozen too. Measure LIF task accuracy when its sole input is
      MLP-emitted neuroletters through the transducer.

Together these pin whether the Nerve Protocol enables genuine
inter-substrate COMMUNICATION, or only substrate-equivalent task
competence (the scaling-law claim).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: N812, E402

from track_w._surrogate import spike_with_surrogate  # noqa: E402


def mutual_info_score(xs: np.ndarray, ys: np.ndarray) -> float:
    """Mutual information I(X;Y) in nats, empirical estimator from samples.

    MI(X,Y) = sum_{x,y} p(x,y) log(p(x,y) / (p(x) p(y))).
    Matches sklearn.metrics.mutual_info_score for discrete inputs.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xs_unique, xs_inv = np.unique(xs, return_inverse=True)
    ys_unique, ys_inv = np.unique(ys, return_inverse=True)
    joint = np.zeros((xs_unique.size, ys_unique.size), dtype=np.float64)
    for xi, yi in zip(xs_inv, ys_inv, strict=False):
        joint[xi, yi] += 1.0
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    mi = float((joint[mask] * np.log(joint[mask] / (px @ py)[mask])).sum())
    return mi
from track_w.lif_wml import LifWML  # noqa: E402
from track_w.mlp_wml import MlpWML  # noqa: E402
from track_w.mock_nerve import MockNerve  # noqa: E402
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask  # noqa: E402
from track_w.training import train_wml_on_task  # noqa: E402


def _train_pair(seed: int, steps: int = 800) -> tuple:
    """Train one MLP and one LIF on HardFlowProxyTask with RNG isolation.

    Mirrors run_w2_hard but returns the trained substrates (plus the
    LIF's input_encoder) so we can query their emissions afterwards.
    """
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)

    task_mlp = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    mlp = MlpWML(id=0, d_hidden=16, seed=seed)
    train_wml_on_task(mlp, nerve, task_mlp, steps=steps, lr=1e-2)

    task_lif = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    lif = LifWML(id=0, n_neurons=16, seed=seed + 10)
    input_encoder = torch.nn.Linear(16, lif.n_neurons)
    opt = torch.optim.Adam(
        list(lif.parameters()) + list(input_encoder.parameters()), lr=1e-2,
    )
    for _ in range(steps):
        x, y = task_lif.sample(batch=64)
        i_in = lif.input_proj(input_encoder(x))
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        logits = lif.emit_head_pi(spikes)[:, : task_lif.n_classes]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return mlp, lif, input_encoder


def _emit_codes(mlp, lif, input_encoder, x, n_classes):
    """Return (codes_mlp, codes_lif, labels_pred_mlp, labels_pred_lif)."""
    with torch.no_grad():
        pi_mlp = mlp.emit_head_pi(mlp.core(x))  # [B, alphabet_size]
        codes_mlp = pi_mlp.argmax(-1)           # [B], in [0, alphabet_size)
        labels_mlp = pi_mlp[:, :n_classes].argmax(-1)

        i_in = lif.input_proj(input_encoder(x))
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        pi_lif = lif.emit_head_pi(spikes)
        codes_lif = pi_lif.argmax(-1)
        labels_lif = pi_lif[:, :n_classes].argmax(-1)
    return codes_mlp, codes_lif, labels_mlp, labels_lif


def run_test_1_mutual_information(seeds=None, steps=800, batch=2048):
    """Test (1) — MI between per-substrate emitted codes on the same input."""
    if seeds is None:
        seeds = list(range(5))
    results = []
    for seed in seeds:
        mlp, lif, input_encoder = _train_pair(seed, steps=steps)
        task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
        x, y = task.sample(batch=batch)
        codes_mlp, codes_lif, labels_mlp, labels_lif = _emit_codes(
            mlp, lif, input_encoder, x, task.n_classes
        )
        # Code-level MI (full 64-wide alphabet).
        mi_codes = mutual_info_score(
            codes_mlp.numpy(), codes_lif.numpy()
        )
        # Label-level MI (12-class task projection).
        mi_labels = mutual_info_score(
            labels_mlp.numpy(), labels_lif.numpy()
        )
        # Reference: MI between each substrate's labels and the ground truth.
        mi_mlp_truth = mutual_info_score(labels_mlp.numpy(), y.numpy())
        mi_lif_truth = mutual_info_score(labels_lif.numpy(), y.numpy())
        # Entropy normalizers (nats).
        _, counts_mlp = np.unique(codes_mlp.numpy(), return_counts=True)
        h_mlp = -(counts_mlp / counts_mlp.sum() * np.log(counts_mlp / counts_mlp.sum())).sum()
        _, counts_lif = np.unique(codes_lif.numpy(), return_counts=True)
        h_lif = -(counts_lif / counts_lif.sum() * np.log(counts_lif / counts_lif.sum())).sum()

        results.append({
            "seed":          seed,
            "mi_codes":      float(mi_codes),
            "mi_labels":     float(mi_labels),
            "mi_mlp_truth":  float(mi_mlp_truth),
            "mi_lif_truth":  float(mi_lif_truth),
            "h_codes_mlp":   float(h_mlp),
            "h_codes_lif":   float(h_lif),
            "mi_over_h_mlp": float(mi_codes / max(h_mlp, 1e-9)),
        })
    return results


def run_test_2_round_trip_fidelity(seeds=None, steps=800, batch=512, transducer_steps=200):
    """Test (2) — fraction of label information surviving a round-trip pass.

    Round-trip: x → MLP → transducer(M→L, 64→64) → LIF.emit → transducer(L→M)
      → final readout against ground-truth labels.

    Transducers are trained with task supervision on a held-out split.
    """
    if seeds is None:
        seeds = list(range(3))
    results = []
    for seed in seeds:
        mlp, lif, input_encoder = _train_pair(seed, steps=steps)
        alphabet_size = mlp.emit_head_pi.out_features
        torch.manual_seed(seed + 100)
        transducer_ml = torch.nn.Linear(alphabet_size, alphabet_size)
        transducer_lm = torch.nn.Linear(alphabet_size, alphabet_size)
        opt = torch.optim.Adam(
            list(transducer_ml.parameters()) + list(transducer_lm.parameters()),
            lr=1e-2,
        )
        task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)

        # Freeze everything except the transducers.
        for p in list(mlp.parameters()) + list(lif.parameters()) + list(
            input_encoder.parameters()
        ):
            p.requires_grad_(False)

        # Train the MLP→LIF→MLP round-trip to reproduce ground-truth labels.
        for _ in range(transducer_steps):
            x, y = task.sample(batch=64)
            pi_mlp = mlp.emit_head_pi(mlp.core(x))         # [B, alphabet]
            pi_mlp_at_lif = transducer_ml(pi_mlp)          # [B, alphabet]
            # LIF receives the transduced logits — decode via its readout.
            # We inject the transduced logits as fake "emitted code logits"
            # the LIF must interpret. Since LIF's emit head is already
            # trained, we treat pi_mlp_at_lif as the substitute for LIF's
            # pre-readout features via a simple linear adapter.
            # (Round-trip is inherently lossy here because we project 64→64
            # instead of going through spike dynamics; what we measure is
            # whether code-level information SURVIVES two linear transducer
            # steps.)
            pi_lif_out = lif.emit_head_pi(
                pi_mlp_at_lif @ lif.emit_head_pi.weight
            )                                              # [B, alphabet]
            pi_back_at_mlp = transducer_lm(pi_lif_out)     # [B, alphabet]
            logits = pi_back_at_mlp[:, : task.n_classes]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Eval on a fresh batch.
        x, y = task.sample(batch=batch)
        with torch.no_grad():
            pi_mlp = mlp.emit_head_pi(mlp.core(x))
            pi_mlp_at_lif = transducer_ml(pi_mlp)
            pi_lif_out = lif.emit_head_pi(
                pi_mlp_at_lif @ lif.emit_head_pi.weight
            )
            pi_back_at_mlp = transducer_lm(pi_lif_out)
            pred_roundtrip = pi_back_at_mlp[:, : task.n_classes].argmax(-1)
            acc_roundtrip = (pred_roundtrip == y).float().mean().item()

            # Baseline: MLP alone (no round-trip).
            pred_direct = mlp.emit_head_pi(mlp.core(x))[
                :, : task.n_classes
            ].argmax(-1)
            acc_direct = (pred_direct == y).float().mean().item()

        results.append({
            "seed":           seed,
            "acc_direct":     acc_direct,
            "acc_roundtrip":  acc_roundtrip,
            "fidelity_ratio": acc_roundtrip / max(acc_direct, 1e-6),
        })
    return results


def run_test_3_cross_substrate_merge(seeds=None, steps=800, batch=512, merge_steps=400):
    """Test (3) — freeze MLP, train transducer M→L, measure LIF accuracy.

    The LIF never sees the raw task input x. It only receives MLP-emitted
    neuroletters (after transducer). The question: can the LIF complete
    the task using only MLP's emitted codes as input?
    """
    if seeds is None:
        seeds = list(range(3))
    results = []
    for seed in seeds:
        mlp, lif, input_encoder = _train_pair(seed, steps=steps)
        alphabet_size = mlp.emit_head_pi.out_features
        # Freeze MLP and LIF readouts.
        for p in list(mlp.parameters()) + list(lif.parameters()):
            p.requires_grad_(False)

        torch.manual_seed(seed + 200)
        # Transducer MLP→LIF operates on emitted-code logits and maps
        # them to features the LIF's emit_head_pi can consume directly.
        transducer = torch.nn.Linear(alphabet_size, lif.n_neurons)
        opt = torch.optim.Adam(transducer.parameters(), lr=1e-2)
        task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)

        for _ in range(merge_steps):
            x, y = task.sample(batch=64)
            with torch.no_grad():
                pi_mlp = mlp.emit_head_pi(mlp.core(x))           # [B, alphabet]
            features_for_lif = transducer(pi_mlp)                # [B, n_neurons]
            # LIF uses its frozen emit_head_pi on the transduced features.
            logits = lif.emit_head_pi(features_for_lif)[:, : task.n_classes]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        x, y = task.sample(batch=batch)
        with torch.no_grad():
            pi_mlp = mlp.emit_head_pi(mlp.core(x))
            features_for_lif = transducer(pi_mlp)
            pred_merge = lif.emit_head_pi(features_for_lif)[
                :, : task.n_classes
            ].argmax(-1)
            acc_merge = (pred_merge == y).float().mean().item()

            # Reference: MLP's own accuracy (upper bound).
            pred_mlp = mlp.emit_head_pi(mlp.core(x))[
                :, : task.n_classes
            ].argmax(-1)
            acc_mlp_alone = (pred_mlp == y).float().mean().item()

        results.append({
            "seed":            seed,
            "acc_mlp_alone":   acc_mlp_alone,
            "acc_cross_merge": acc_merge,
            "merge_ratio":     acc_merge / max(acc_mlp_alone, 1e-6),
        })
    return results


def main():
    print("=" * 70)
    print("Test (1): Mutual information between substrate emissions")
    print("=" * 70)
    r1 = run_test_1_mutual_information(seeds=list(range(5)), steps=800, batch=2048)
    for r in r1:
        print(f"  seed={r['seed']} | MI(codes)={r['mi_codes']:.4f} nats  "
              f"MI(labels)={r['mi_labels']:.4f} nats  "
              f"MI_mlp_truth={r['mi_mlp_truth']:.4f}  "
              f"MI_lif_truth={r['mi_lif_truth']:.4f}  "
              f"MI/H(MLP)={r['mi_over_h_mlp']:.4f}")
    print(f"  mean MI(codes)     = {np.mean([r['mi_codes'] for r in r1]):.4f}")
    print(f"  mean MI(labels)    = {np.mean([r['mi_labels'] for r in r1]):.4f}")
    print(f"  mean MI/H(MLP)     = {np.mean([r['mi_over_h_mlp'] for r in r1]):.4f}")
    print(f"  mean MI_mlp_truth  = {np.mean([r['mi_mlp_truth'] for r in r1]):.4f}  (upper reference: log(12)={np.log(12):.4f})")

    print()
    print("=" * 70)
    print("Test (2): Round-trip fidelity MLP → LIF → MLP")
    print("=" * 70)
    r2 = run_test_2_round_trip_fidelity(seeds=list(range(3)), steps=800)
    for r in r2:
        print(f"  seed={r['seed']} | acc_direct={r['acc_direct']:.3f} "
              f"acc_roundtrip={r['acc_roundtrip']:.3f} "
              f"fidelity_ratio={r['fidelity_ratio']:.3f}")

    print()
    print("=" * 70)
    print("Test (3): Cross-substrate merge (LIF fed by MLP-emitted codes only)")
    print("=" * 70)
    r3 = run_test_3_cross_substrate_merge(seeds=list(range(3)), steps=800)
    for r in r3:
        print(f"  seed={r['seed']} | acc_mlp_alone={r['acc_mlp_alone']:.3f} "
              f"acc_cross_merge={r['acc_cross_merge']:.3f} "
              f"merge_ratio={r['merge_ratio']:.3f}")


if __name__ == "__main__":
    main()
