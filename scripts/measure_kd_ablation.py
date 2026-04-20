"""KD match-compute ablation — is cross-merge just KD in disguise?

Three conditions at matched compute budget (same SGD steps, same
batch size, same optimizer), on HardFlowProxyTask with a teacher MLP
trained to competence + a frozen LIF student:

  (A) cross-merge (ours): train Linear transducer mapping the
      teacher's logits over the 64-code protocol alphabet to the
      LIF's input features. Supervision = ground-truth hard labels
      through the frozen LIF emit head. This passes
      log2(64) = 6 bits per emission.

  (B) KD-through-transducer: same frozen LIF, same linear transducer,
      but supervision = Hinton 2015 distillation loss using the
      teacher's 12-class softmax distribution at temperature T. This
      passes ~log2(12) ≈ 3.6 bits per emission (class distribution
      entropy).

  (C) vanilla KD: LIF trained end-to-end (not frozen) on the Hinton
      distillation loss. No transducer — this is the classical
      setup. Included as an upper-bound reference.

If acc(A) > acc(B) at matched compute, the 6-bit protocol channel
carries information the 3.6-bit class distribution cannot. That's
the empirical signature that cross-merge exploits something KD does
not, confirming it is not just KD in disguise.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: N812, E402

from track_w._surrogate import spike_with_surrogate  # noqa: E402
from track_w.lif_wml import LifWML  # noqa: E402
from track_w.mlp_wml import MlpWML  # noqa: E402
from track_w.mock_nerve import MockNerve  # noqa: E402
from track_w.tasks.hard_flow_proxy import HardFlowProxyTask  # noqa: E402
from track_w.training import train_wml_on_task  # noqa: E402


def _train_teacher(seed: int, steps: int) -> MlpWML:
    """Train the teacher MLP to competence on the shared task."""
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    mlp = MlpWML(id=0, d_hidden=16, seed=seed)
    train_wml_on_task(mlp, nerve, task, steps=steps, lr=1e-2)
    return mlp


def _eval_acc(logits_fn, task, batch: int = 512) -> float:
    x, y = task.sample(batch=batch)
    with torch.no_grad():
        logits = logits_fn(x)
        pred = logits[:, : task.n_classes].argmax(-1)
        return (pred == y).float().mean().item()


def condition_a_cross_merge(
    teacher: MlpWML, seed: int, steps: int,
) -> tuple[float, float]:
    """(A) Our cross-merge: frozen student, frozen teacher, learned
    transducer over the 64-code protocol alphabet, hard-label CE."""
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    lif = LifWML(id=0, n_neurons=16, seed=seed + 10)
    alphabet = teacher.emit_head_pi.out_features
    # Warm the LIF just enough to have a meaningful frozen emit head
    # (otherwise emit_head_pi is random and the transducer task is ill-posed).
    opt_warm = torch.optim.Adam(lif.parameters(), lr=1e-2)
    warm_steps = max(50, steps // 4)
    for _ in range(warm_steps):
        x, y = task.sample(batch=64)
        i_in = lif.input_proj(x)
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        logits = lif.emit_head_pi(spikes)[:, : task.n_classes]
        loss = F.cross_entropy(logits, y)
        opt_warm.zero_grad()
        loss.backward()
        opt_warm.step()

    # Freeze both teacher and student.
    for p in list(teacher.parameters()) + list(lif.parameters()):
        p.requires_grad_(False)

    torch.manual_seed(seed + 200)
    transducer = torch.nn.Linear(alphabet, lif.n_neurons)
    opt = torch.optim.Adam(transducer.parameters(), lr=1e-2)
    for _ in range(steps):
        x, y = task.sample(batch=64)
        with torch.no_grad():
            pi_mlp = teacher.emit_head_pi(teacher.core(x))       # [B, 64]
        feat = transducer(pi_mlp)                                # [B, n_neurons]
        logits = lif.emit_head_pi(feat)[:, : task.n_classes]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    def _fwd(x):
        with torch.no_grad():
            pi_mlp = teacher.emit_head_pi(teacher.core(x))
            feat = transducer(pi_mlp)
            return lif.emit_head_pi(feat)

    acc_cross = _eval_acc(_fwd, task)
    acc_teacher = _eval_acc(
        lambda x: teacher.emit_head_pi(teacher.core(x)), task,
    )
    return acc_cross, acc_teacher


def condition_b_kd_through_transducer(
    teacher: MlpWML, seed: int, steps: int, temperature: float = 4.0,
) -> float:
    """(B) KD-through-transducer: frozen student, frozen teacher, learned
    transducer. Supervision = Hinton KD loss on n_classes-dim softmax
    at temperature T. Compute-matched with condition (A)."""
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    lif = LifWML(id=0, n_neurons=16, seed=seed + 10)
    alphabet = teacher.emit_head_pi.out_features

    opt_warm = torch.optim.Adam(lif.parameters(), lr=1e-2)
    warm_steps = max(50, steps // 4)
    for _ in range(warm_steps):
        x, y = task.sample(batch=64)
        i_in = lif.input_proj(x)
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        logits = lif.emit_head_pi(spikes)[:, : task.n_classes]
        loss = F.cross_entropy(logits, y)
        opt_warm.zero_grad()
        loss.backward()
        opt_warm.step()

    for p in list(teacher.parameters()) + list(lif.parameters()):
        p.requires_grad_(False)

    torch.manual_seed(seed + 300)
    transducer = torch.nn.Linear(alphabet, lif.n_neurons)
    opt = torch.optim.Adam(transducer.parameters(), lr=1e-2)

    T = temperature
    alpha_soft = 0.5
    for _ in range(steps):
        x, y = task.sample(batch=64)
        with torch.no_grad():
            pi_mlp = teacher.emit_head_pi(teacher.core(x))
            teacher_soft = F.softmax(
                pi_mlp[:, : task.n_classes] / T, dim=-1,
            )
        feat = transducer(pi_mlp)
        student_logits = lif.emit_head_pi(feat)[:, : task.n_classes]
        student_soft_log = F.log_softmax(student_logits / T, dim=-1)
        kd_loss = F.kl_div(
            student_soft_log, teacher_soft,
            reduction="batchmean",
        ) * T * T
        hard_loss = F.cross_entropy(student_logits, y)
        loss = alpha_soft * kd_loss + (1 - alpha_soft) * hard_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    def _fwd(x):
        with torch.no_grad():
            pi_mlp = teacher.emit_head_pi(teacher.core(x))
            feat = transducer(pi_mlp)
            return lif.emit_head_pi(feat)

    return _eval_acc(_fwd, task)


def condition_c_vanilla_kd(
    teacher: MlpWML, seed: int, steps: int, temperature: float = 4.0,
) -> float:
    """(C) Vanilla Hinton KD: student trained end-to-end on soft+hard
    loss, no transducer. Upper-bound reference."""
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    lif = LifWML(id=0, n_neurons=16, seed=seed + 20)
    opt = torch.optim.Adam(lif.parameters(), lr=1e-2)

    T = temperature
    alpha_soft = 0.5
    for _ in range(steps):
        x, y = task.sample(batch=64)
        with torch.no_grad():
            teacher_soft = F.softmax(
                teacher.emit_head_pi(teacher.core(x))[:, : task.n_classes] / T,
                dim=-1,
            )
        i_in = lif.input_proj(x)
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        student_logits = lif.emit_head_pi(spikes)[:, : task.n_classes]
        student_soft_log = F.log_softmax(student_logits / T, dim=-1)
        kd_loss = F.kl_div(
            student_soft_log, teacher_soft, reduction="batchmean",
        ) * T * T
        hard_loss = F.cross_entropy(student_logits, y)
        loss = alpha_soft * kd_loss + (1 - alpha_soft) * hard_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    def _fwd(x):
        with torch.no_grad():
            i_in = lif.input_proj(x)
            spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
            return lif.emit_head_pi(spikes)

    return _eval_acc(_fwd, task)


def run_kd_ablation(
    seeds: list[int] | None = None,
    teacher_steps: int = 400,
    transfer_steps: int = 300,
) -> dict:
    """Matched-compute ablation. Same teacher_steps to train the MLP;
    same transfer_steps for each of (A), (B), (C)."""
    if seeds is None:
        seeds = [0, 1, 2]
    results = []
    for s in seeds:
        teacher = _train_teacher(seed=s, steps=teacher_steps)
        acc_a, acc_teacher = condition_a_cross_merge(
            teacher, seed=s, steps=transfer_steps,
        )
        acc_b = condition_b_kd_through_transducer(
            teacher, seed=s, steps=transfer_steps,
        )
        acc_c = condition_c_vanilla_kd(
            teacher, seed=s, steps=transfer_steps,
        )
        results.append({
            "seed":         s,
            "acc_teacher":  acc_teacher,
            "acc_cross":    acc_a,
            "acc_kd_trans": acc_b,
            "acc_kd_vanilla": acc_c,
            "gap_cross_vs_kd": acc_a - acc_b,
        })

    summary = {
        "seeds":           seeds,
        "mean_acc_teacher":    float(np.mean([r["acc_teacher"]    for r in results])),
        "mean_acc_cross":      float(np.mean([r["acc_cross"]      for r in results])),
        "mean_acc_kd_trans":   float(np.mean([r["acc_kd_trans"]   for r in results])),
        "mean_acc_kd_vanilla": float(np.mean([r["acc_kd_vanilla"] for r in results])),
        "mean_gap":       float(np.mean([r["gap_cross_vs_kd"] for r in results])),
        "per_seed":       results,
    }
    return summary


def main() -> None:
    r = run_kd_ablation(seeds=[0, 1, 2], teacher_steps=400, transfer_steps=300)
    print(f"KD match-compute ablation on HardFlowProxyTask (3 seeds):")
    print()
    for rec in r["per_seed"]:
        print(f"  seed={rec['seed']} | teacher={rec['acc_teacher']:.3f} "
              f"cross-merge={rec['acc_cross']:.3f} "
              f"KD-trans={rec['acc_kd_trans']:.3f} "
              f"KD-vanilla={rec['acc_kd_vanilla']:.3f} "
              f"Δ(cross-KD)={rec['gap_cross_vs_kd']*100:+.2f}%")
    print()
    print(f"  mean teacher      = {r['mean_acc_teacher']:.3f}")
    print(f"  mean cross-merge  = {r['mean_acc_cross']:.3f}")
    print(f"  mean KD-through-T = {r['mean_acc_kd_trans']:.3f}")
    print(f"  mean KD-vanilla   = {r['mean_acc_kd_vanilla']:.3f}")
    print(f"  mean gap (cross vs KD-trans) = {r['mean_gap']*100:+.2f}%")


if __name__ == "__main__":
    main()
