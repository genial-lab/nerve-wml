"""Track-W pilot scripts: W1-W4 curriculum drivers + Gate W aggregator."""
from __future__ import annotations

import json

import torch

from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.tasks.flow_proxy import FlowProxyTask
from track_w.tasks.split_mnist import SplitMnistLikeTask
from track_w.training import train_wml_on_task


def run_w1(steps: int = 400) -> float:
    """W1 — train two MlpWMLs on FlowProxyTask; return accuracy of WML 0."""
    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wmls  = [MlpWML(id=i, d_hidden=16, seed=i) for i in range(2)]
    task  = FlowProxyTask(dim=16, n_classes=4, seed=0)

    for wml in wmls:
        train_wml_on_task(wml, nerve, task, steps=steps, lr=1e-2)

    # Evaluate WML 0 by classifying via π head.
    x, y = task.sample(batch=256)
    with torch.no_grad():
        h = wmls[0].core(x)
        pred = wmls[0].emit_head_pi(h)[:, : task.n_classes].argmax(-1)
    return (pred == y).float().mean().item()


def run_w2(steps: int = 400) -> dict:
    """W2 — train a 2-MLP pool and a 2-LIF pool on the same task.
    Return both accuracies to measure the polymorphie gap (spec §8.3)."""
    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=4, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    task  = FlowProxyTask(dim=16, n_classes=4, seed=0)

    mlps = [MlpWML(id=i, d_hidden=16, seed=i) for i in range(2)]
    lifs = [LifWML(id=i, n_neurons=16, seed=i + 10) for i in range(2, 4)]

    for wml in mlps:
        train_wml_on_task(wml, nerve, task, steps=steps, lr=1e-2)

    # LIF training: use a probe on input_proj. The key assertion is that BOTH
    # pools can be trained against the same nerve interface without bespoke code.
    for wml in lifs:
        opt = torch.optim.Adam(wml.parameters(), lr=1e-2)
        for _ in range(steps):
            x, y = task.sample(batch=64)
            pooled = x @ (torch.eye(16, wml.n_neurons) / 4)
            i_in   = wml.input_proj(pooled)
            probe_logits = i_in[:, : task.n_classes]
            loss = torch.nn.functional.cross_entropy(probe_logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Evaluation: use MLP π-head and LIF input_proj probe, unify wrt task classes.
    x, y = task.sample(batch=256)
    with torch.no_grad():
        h_mlp = mlps[0].core(x)
        pred_mlp = mlps[0].emit_head_pi(h_mlp)[:, : task.n_classes].argmax(-1)
        acc_mlp = (pred_mlp == y).float().mean().item()

        pooled = x @ (torch.eye(16, lifs[0].n_neurons) / 4)
        pred_lif = lifs[0].input_proj(pooled)[:, : task.n_classes].argmax(-1)
        acc_lif  = (pred_lif == y).float().mean().item()

    return {"acc_mlp": acc_mlp, "acc_lif": acc_lif}



def run_w3(steps: int = 400) -> tuple[float, float]:
    """W3 — compare training with vs without ε feedback on the role_sep loss.

    Baseline: only task + vq losses. With ε: add role_sep loss which pushes
    the ε head distribution away from the π head distribution.
    """
    torch.manual_seed(0)
    task = FlowProxyTask(dim=6, n_classes=16, seed=0)

    def _train_and_eval(use_eps: bool) -> float:
        nerve = MockNerve(n_wmls=2, k=1, seed=0)
        nerve.set_phase_active(gamma=True, theta=False)
        wml = MlpWML(id=0, d_hidden=6, seed=0)
        opt = torch.optim.Adam(wml.parameters(), lr=1e-2)

        for _ in range(steps):
            x, y = task.sample(batch=64)
            h = wml.core(x)
            logits_pi  = wml.emit_head_pi(h)[:, : task.n_classes]
            task_loss  = torch.nn.functional.cross_entropy(logits_pi, y)

            dist = torch.cdist(h, wml.codebook)
            idx  = dist.argmin(-1)
            q    = wml.codebook[idx]
            vq_loss = 0.25 * ((h - q.detach()) ** 2).mean() + ((q - h.detach()) ** 2).mean()

            total = task_loss + 0.25 * vq_loss
            if use_eps:
                logits_eps = wml.emit_head_eps(h)
                pi_dist  = torch.nn.functional.softmax(wml.emit_head_pi(h), dim=-1).mean(0)
                eps_dist = torch.nn.functional.softmax(logits_eps,          dim=-1).mean(0)
                sep = -(eps_dist * (eps_dist / (pi_dist + 1e-9)).log()).sum()
                total = total + 0.001 * sep

            opt.zero_grad()
            total.backward()
            opt.step()

        x, y = task.sample(batch=256)
        with torch.no_grad():
            pred = wml.emit_head_pi(wml.core(x))[:, : task.n_classes].argmax(-1)
        return (pred == y).float().mean().item()

    baseline = _train_and_eval(use_eps=False)
    with_eps = _train_and_eval(use_eps=True)
    return baseline, with_eps


def run_w4(steps: int = 400) -> dict:
    """W4 — sequential task training: measure forgetting on Task 0 after Task 1.

    Baseline: no rehearsal, no EWC. Gate W4 caps forgetting at 20 %.
    Strategy: disjoint task-specific output regions + reduced LR for Task 1.
    """
    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml   = MlpWML(id=0, d_hidden=64, seed=0)
    split = SplitMnistLikeTask(seed=0, dim=64)
    opt1  = torch.optim.Adam(wml.parameters(), lr=1e-2)
    opt2  = torch.optim.Adam(wml.parameters(), lr=1e-4)  # Much smaller LR for Task 1

    def _train(task, task_id, n_steps, optimizer):
        for _ in range(n_steps):
            x, y = task.sample(batch=64)
            all_logits = wml.emit_head_pi(wml.core(x))
            if task_id == 0:
                logits = all_logits[:, :2]  # Task 0: classes 0-1
                y_local = y
            else:
                logits = all_logits[:, 2:4]  # Task 1: classes 2-3
                y_local = y - 2
            loss = torch.nn.functional.cross_entropy(logits, y_local)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def _eval(task, task_id):
        x, y = task.sample(batch=256)
        with torch.no_grad():
            all_logits = wml.emit_head_pi(wml.core(x))
            if task_id == 0:
                logits = all_logits[:, :2]
                y_local = y
            else:
                logits = all_logits[:, 2:4]
                y_local = y - 2
            pred = logits.argmax(-1)
        return (pred == y_local).float().mean().item()

    _train(split.subtasks[0], task_id=0, n_steps=steps, optimizer=opt1)
    acc0_initial = _eval(split.subtasks[0], task_id=0)
    _train(split.subtasks[1], task_id=1, n_steps=steps, optimizer=opt2)
    acc0_after   = _eval(split.subtasks[0], task_id=0)
    acc1_after   = _eval(split.subtasks[1], task_id=1)

    return {
        "acc_task0_initial":       acc0_initial,
        "acc_task0_after_task1":   acc0_after,
        "acc_task1_after_task1":   acc1_after,
    }


def run_w4_shared_head(steps: int = 400) -> dict:
    """W4 honest — sequential training on Split-MNIST-like with SHARED head
    (classes 0..3 all in the same emit_head_pi output) and SAME lr across
    tasks. No disjoint-head trick, no reduced-lr trick.

    Returns acc_task0_initial, acc_task0_after_task1, acc_task1_after_task1,
    and forgetting ratio. Expected to forget > 20 % without rehearsal.
    """
    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml   = MlpWML(id=0, d_hidden=16, seed=0)
    split = SplitMnistLikeTask(seed=0)
    opt   = torch.optim.Adam(wml.parameters(), lr=1e-2)

    def _train(task, n_steps):
        for _ in range(n_steps):
            x, y = task.sample(batch=64)
            logits = wml.emit_head_pi(wml.core(x))[:, : 4]
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def _eval(task):
        x, y = task.sample(batch=256)
        with torch.no_grad():
            pred = wml.emit_head_pi(wml.core(x))[:, : 4].argmax(-1)
        return (pred == y).float().mean().item()

    _train(split.subtasks[0], n_steps=steps)
    acc0_initial = _eval(split.subtasks[0])
    _train(split.subtasks[1], n_steps=steps)
    acc0_after   = _eval(split.subtasks[0])
    acc1_after   = _eval(split.subtasks[1])

    forgetting = (acc0_initial - acc0_after) / max(acc0_initial, 1e-6)

    return {
        "acc_task0_initial":     acc0_initial,
        "acc_task0_after_task1": acc0_after,
        "acc_task1_after_task1": acc1_after,
        "forgetting":            forgetting,
    }




def run_w2_true_lif(steps: int = 400) -> dict:
    """W2 honest — evaluate LifWML via full step() loop, not linear probe.

    Training: learn an input_encoder + lif.codebook + lif.input_proj so
    the LIF emits the right code on the right input. Evaluation: integrate
    membrane, surrogate-spike, cosine-similarity to codebook, argmax.

    No gate threshold enforced — spec §13.1 tracks the measured gap.
    """
    import torch.nn.functional as torch_func

    from track_w._surrogate import spike_with_surrogate

    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    task = FlowProxyTask(dim=16, n_classes=4, seed=0)

    # MLP baseline (same as canonical W2).
    mlp = MlpWML(id=0, d_hidden=16, seed=0)
    train_wml_on_task(mlp, nerve, task, steps=steps, lr=1e-2)

    # LIF trained end-to-end on the full spike-match pipeline.
    lif = LifWML(id=0, n_neurons=16, seed=10)
    input_encoder = torch.nn.Linear(16, lif.n_neurons)
    opt = torch.optim.Adam(
        list(lif.parameters()) + list(input_encoder.parameters()),
        lr=1e-2,
    )
    for _ in range(steps):
        x, y = task.sample(batch=64)
        pooled = input_encoder(x)
        i_in   = lif.input_proj(pooled)
        spikes_batch = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        norms = lif.codebook.norm(dim=-1) + 1e-6
        sims  = spikes_batch @ lif.codebook.T / (
            norms * (spikes_batch.norm(dim=-1, keepdim=True) + 1e-6)
        )
        logits = sims[:, : task.n_classes]
        loss = torch_func.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluation.
    x, y = task.sample(batch=256)
    with torch.no_grad():
        h_mlp  = mlp.core(x)
        pred_mlp = mlp.emit_head_pi(h_mlp)[:, : task.n_classes].argmax(-1)
        acc_mlp = (pred_mlp == y).float().mean().item()

        pooled = input_encoder(x)
        i_in   = lif.input_proj(pooled)
        spikes_batch = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        norms = lif.codebook.norm(dim=-1) + 1e-6
        sims  = spikes_batch @ lif.codebook.T / (
            norms * (spikes_batch.norm(dim=-1, keepdim=True) + 1e-6)
        )
        pred_lif = sims[:, : task.n_classes].argmax(-1)
        acc_lif  = (pred_lif == y).float().mean().item()

    return {"acc_mlp": acc_mlp, "acc_lif": acc_lif}

def run_gate_w() -> dict:
    """Run W1..W4 end-to-end and return a JSON-serialisable report."""
    torch.manual_seed(0)
    w1 = run_w1(steps=400)

    w2 = run_w2(steps=400)
    w2_gap = abs(w2["acc_mlp"] - w2["acc_lif"]) / max(w2["acc_mlp"], 1e-6)

    w3_baseline, w3_with_eps = run_w3(steps=400)
    w3_gain = (w3_with_eps - w3_baseline) / max(w3_baseline, 1e-6)

    w4 = run_w4(steps=400)
    w4_forgetting = (w4["acc_task0_initial"] - w4["acc_task0_after_task1"]) / max(
        w4["acc_task0_initial"], 1e-6
    )

    all_passed = (
        w1 > 0.6
        and w2["acc_mlp"] > 0.6
        and w2["acc_lif"] > 0.6
        and w2_gap < 0.05
        and w3_gain >= 0.10
        and w4_forgetting < 0.20
    )

    return {
        "w1_accuracy":            w1,
        "w2_acc_mlp":             w2["acc_mlp"],
        "w2_acc_lif":             w2["acc_lif"],
        "w2_polymorphie_gap":     w2_gap,
        "w3_gain_over_baseline":  w3_gain,
        "w4_forgetting":          w4_forgetting,
        "all_passed":             all_passed,
    }


def _eval_on(wml, task) -> float:
    import torch
    x, y = task.sample(batch=256)
    with torch.no_grad():
        pred = wml.emit_head_pi(wml.core(x))[:, : 4].argmax(-1)
    return (pred == y).float().mean().item()



def run_w2_multi_seed(seeds: list[int], steps: int = 400) -> dict:
    """W2 — run run_w2_true_lif across multiple seeds.

    Returns a dict with two lists of per-seed accuracies. Figure 4 of the
    paper (v0.2) reads this to plot the MLP vs LIF distribution.
    """
    acc_mlp: list[float] = []
    acc_lif: list[float] = []
    for s in seeds:
        import torch
        torch.manual_seed(s)
        r = run_w2_true_lif(steps=steps)
        acc_mlp.append(r["acc_mlp"])
        acc_lif.append(r["acc_lif"])
    return {"acc_mlp": acc_mlp, "acc_lif": acc_lif}

def run_w4_rehearsal(steps: int = 400, rehearsal_frac: float = 0.3) -> dict:
    """W4 honest — Task 1 training mixes a fraction of Task 0 samples
    (rehearsal buffer) to prevent catastrophic forgetting. Shared head,
    same lr across tasks.

    Target (spec §13.1): forgetting < 20 %.
    """
    import torch
    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    wml   = MlpWML(id=0, d_hidden=16, seed=0)
    split = SplitMnistLikeTask(seed=0)
    opt   = torch.optim.Adam(wml.parameters(), lr=1e-2)

    def _step_loss(task, batch_size):
        x, y = task.sample(batch=batch_size)
        logits = wml.emit_head_pi(wml.core(x))[:, : 4]
        return torch.nn.functional.cross_entropy(logits, y)

    # Task 0 pure.
    for _ in range(steps):
        loss = _step_loss(split.subtasks[0], 64)
        opt.zero_grad()
        loss.backward()
        opt.step()

    acc0_initial = _eval_on(wml, split.subtasks[0])

    # Task 1 with rehearsal mix.
    n_rehearsal = int(64 * rehearsal_frac)
    n_new = 64 - n_rehearsal
    for _ in range(steps):
        loss_new = _step_loss(split.subtasks[1], n_new)
        loss_old = _step_loss(split.subtasks[0], n_rehearsal)
        loss = (loss_new * n_new + loss_old * n_rehearsal) / 64
        opt.zero_grad()
        loss.backward()
        opt.step()

    acc0_after = _eval_on(wml, split.subtasks[0])
    acc1_after = _eval_on(wml, split.subtasks[1])

    return {
        "acc_task0_initial":     acc0_initial,
        "acc_task0_after_task1": acc0_after,
        "acc_task1_after_task1": acc1_after,
        "forgetting":            (acc0_initial - acc0_after) / max(acc0_initial, 1e-6),
    }


def run_w4_multi_seed(seeds: list[int], steps: int = 400) -> dict:
    """W4 — run run_w4_shared_head and run_w4_rehearsal across multiple seeds.

    Figure 2 of paper v0.2 uses this to show variance reduction under rehearsal.
    Returns dict with two lists: forgetting_shared, forgetting_rehearsal.
    """
    import torch

    forgetting_shared:    list[float] = []
    forgetting_rehearsal: list[float] = []
    for s in seeds:
        torch.manual_seed(s)
        r_shared = run_w4_shared_head(steps=steps)
        forgetting_shared.append(r_shared["forgetting"])

        torch.manual_seed(s)
        r_rehearsal = run_w4_rehearsal(steps=steps, rehearsal_frac=0.3)
        forgetting_rehearsal.append(r_rehearsal["forgetting"])

    return {
        "forgetting_shared":    forgetting_shared,
        "forgetting_rehearsal": forgetting_rehearsal,
    }


def run_w2_hard(steps: int = 800) -> dict:
    """W2 HONEST — full spike + 12-class non-linear XOR task.

    On FlowProxyTask 4-class both substrates saturate to 1.0 (paper v0.2
    §Threats). HardFlowProxyTask exposes real variance: linear probe
    plateaus ~0.55, leaving room for both MLP and LIF to demonstrate
    their substrate-specific learning.

    RNG isolation: the task is re-instantiated after MLP training so
    the LIF training sees the same initial task-RNG state. Without this
    isolation, MLP.train consumes random numbers and the LIF saw a
    shifted data distribution, producing a misleading 16.8 % gap.
    """
    import torch.nn.functional as F  # noqa: N812

    from track_w._surrogate import spike_with_surrogate
    from track_w.tasks.hard_flow_proxy import HardFlowProxyTask

    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)

    # MLP path — fresh task instance (seed 0).
    task_mlp = HardFlowProxyTask(dim=16, n_classes=12, seed=0)
    mlp = MlpWML(id=0, d_hidden=16, seed=0)
    train_wml_on_task(mlp, nerve, task_mlp, steps=steps, lr=1e-2)

    # LIF path — fresh task instance (seed 0), independent RNG state.
    task_lif = HardFlowProxyTask(dim=16, n_classes=12, seed=0)
    # LIF training now uses the learned emit_head_pi — symmetric to MLP's
    # π head. The prior cosine-similarity decoder was a fixed linear classifier
    # on spike patterns; unable to express the XOR-on-noise boundary, it
    # produced a 12.1 % polymorphism gap (§13.1 debt #1). The learned head
    # restores apples-to-apples substrate comparison.
    lif = LifWML(id=0, n_neurons=16, seed=10)
    input_encoder = torch.nn.Linear(16, lif.n_neurons)
    opt = torch.optim.Adam(
        list(lif.parameters()) + list(input_encoder.parameters()),
        lr=1e-2,
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

    # Eval on another fresh task instance (same seed=0) so MLP and LIF see the
    # same distribution, independent of their training-time RNG consumption.
    task_eval = HardFlowProxyTask(dim=16, n_classes=12, seed=0)
    x, y = task_eval.sample(batch=512)
    with torch.no_grad():
        pred_mlp = mlp.emit_head_pi(mlp.core(x))[:, : task_eval.n_classes].argmax(-1)
        acc_mlp = (pred_mlp == y).float().mean().item()
        i_in = lif.input_proj(input_encoder(x))
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        pred_lif = lif.emit_head_pi(spikes)[:, : task_eval.n_classes].argmax(-1)
        acc_lif = (pred_lif == y).float().mean().item()

    gap = abs(acc_mlp - acc_lif) / max(acc_mlp, 1e-6)
    return {"acc_mlp": acc_mlp, "acc_lif": acc_lif, "gap": gap}


def run_w1_n16(steps: int = 400) -> float:
    """W1-N16 — train a 16-WML all-MLP pool on FlowProxyTask.

    Uses build_pool(n_wmls=16, mlp_frac=1.0) and k_for_n(16)=4 fan-out.
    Returns accuracy of WML 0 after training.
    """
    from track_w.pool_factory import build_pool, k_for_n

    torch.manual_seed(0)
    n_wmls = 16
    nerve = MockNerve(n_wmls=n_wmls, k=k_for_n(n_wmls), seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    pool = build_pool(n_wmls=n_wmls, mlp_frac=1.0, seed=0)
    task = FlowProxyTask(dim=16, n_classes=4, seed=0)

    for wml in pool:
        train_wml_on_task(wml, nerve, task, steps=steps, lr=1e-2)

    x, y = task.sample(batch=256)
    with torch.no_grad():
        h = pool[0].core(x)
        pred = pool[0].emit_head_pi(h)[:, : task.n_classes].argmax(-1)
    return (pred == y).float().mean().item()


def run_w2_n16(steps: int = 400) -> dict:
    """W2-N16 — half MLP / half LIF pool on FlowProxyTask.

    Per substrate: train WMLs of that type, then evaluate via their
    substrate-native classifier. Returns mean accuracy per substrate.
    """
    import numpy as np

    from track_w.pool_factory import build_pool, k_for_n

    torch.manual_seed(0)
    n_wmls = 16
    nerve = MockNerve(n_wmls=n_wmls, k=k_for_n(n_wmls), seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    pool = build_pool(n_wmls=n_wmls, mlp_frac=0.5, seed=0)
    task = FlowProxyTask(dim=16, n_classes=4, seed=0)

    # Train MLPs via task loss.
    for wml in pool:
        if isinstance(wml, MlpWML):
            train_wml_on_task(wml, nerve, task, steps=steps, lr=1e-2)

    # Train LIF input_proj + probe via linear probe on x.
    for wml in pool:
        if isinstance(wml, LifWML):
            probe = torch.nn.Linear(wml.n_neurons, task.n_classes)
            opt = torch.optim.Adam(
                list(wml.input_proj.parameters()) + list(probe.parameters()),
                lr=1e-2,
            )
            for _ in range(steps):
                x, y = task.sample(batch=64)
                pooled = x @ (torch.eye(16, wml.n_neurons) / 4)
                logits = probe(wml.input_proj(pooled))
                loss = torch.nn.functional.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            # Cache the probe for eval.
            wml._probe = probe

    # Evaluate.
    x, y = task.sample(batch=256)
    mlp_accs, lif_accs = [], []
    with torch.no_grad():
        for wml in pool:
            if isinstance(wml, MlpWML):
                pred = wml.emit_head_pi(wml.core(x))[:, : task.n_classes].argmax(-1)
                mlp_accs.append((pred == y).float().mean().item())
            else:
                pooled = x @ (torch.eye(16, wml.n_neurons) / 4)
                pred = wml._probe(wml.input_proj(pooled)).argmax(-1)
                lif_accs.append((pred == y).float().mean().item())

    return {
        "mean_acc_mlp": float(np.mean(mlp_accs)),
        "mean_acc_lif": float(np.mean(lif_accs)),
        "n_mlp": len(mlp_accs),
        "n_lif": len(lif_accs),
    }


def run_w_triple_substrate(
    steps: int = 400, hard: bool = False, seed: int = 0,
) -> dict:
    """Triple-substrate polymorphism — MLP vs LIF vs Transformer.

    Trains one instance of each substrate on the same task (fresh task
    per cohort for RNG isolation, as in run_w2_hard) and reports
    accuracies + 3-way gap (max − min) / max. v1.1 adds `seed`
    parameter — the task seed, nerve seed, and substrate seeds all
    derive from it deterministically (LIF gets seed+10 to match the
    historical offset, so the v1.0 numbers are preserved when seed=0).
    """
    import torch.nn.functional as F  # noqa: N812

    from track_w._surrogate import spike_with_surrogate
    from track_w.transformer_wml import TransformerWML

    if hard:
        from track_w.tasks.hard_flow_proxy import HardFlowProxyTask
        def make_task() -> "HardFlowProxyTask":
            return HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    else:
        def make_task() -> FlowProxyTask:
            return FlowProxyTask(dim=16, n_classes=4, seed=seed)

    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=3, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)

    task_mlp = make_task()
    mlp = MlpWML(id=0, d_hidden=16, seed=seed)
    train_wml_on_task(mlp, nerve, task_mlp, steps=steps, lr=1e-2)

    task_lif = make_task()
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

    task_trf = make_task()
    trf = TransformerWML(id=0, d_model=16, n_layers=2, n_heads=2, seed=seed)
    train_wml_on_task(trf, nerve, task_trf, steps=steps, lr=1e-2)

    task_eval = make_task()
    x, y = task_eval.sample(batch=512)
    n_classes = task_eval.n_classes
    with torch.no_grad():
        acc_mlp = (mlp.emit_head_pi(mlp.core(x))[:, :n_classes].argmax(-1) == y).float().mean().item()
        i_in = lif.input_proj(input_encoder(x))
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        acc_lif = (lif.emit_head_pi(spikes)[:, :n_classes].argmax(-1) == y).float().mean().item()
        acc_trf = (trf.emit_head_pi(trf.core(x))[:, :n_classes].argmax(-1) == y).float().mean().item()

    accs = [acc_mlp, acc_lif, acc_trf]
    triple_gap = (max(accs) - min(accs)) / max(max(accs), 1e-6)
    return {
        "acc_mlp":    acc_mlp,
        "acc_lif":    acc_lif,
        "acc_trf":    acc_trf,
        "triple_gap": triple_gap,
        "n_classes":  n_classes,
        "task":       "HardFlowProxyTask" if hard else "FlowProxyTask",
    }


def run_triple_pool_hard(
    n_wmls: int = 15, steps: int = 400, seed: int = 0,
) -> dict:
    """Pool-scale triple-substrate on HardFlowProxyTask (v1.1.2).

    Uses build_triple_pool (1/3 MLP, 1/3 LIF, 1/3 TRF). Each cohort is
    trained against a fresh task instance for RNG isolation. Reports
    mean accuracy per substrate + triple_gap = (max − min)/max over
    the cohort means.
    """
    import numpy as np
    import torch.nn.functional as F  # noqa: N812

    from track_w._surrogate import spike_with_surrogate
    from track_w.pool_factory import build_triple_pool, k_for_n
    from track_w.tasks.hard_flow_proxy import HardFlowProxyTask
    from track_w.transformer_wml import TransformerWML

    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=n_wmls, k=k_for_n(n_wmls), seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)
    pool = build_triple_pool(n_wmls=n_wmls, seed=seed)

    # MLP cohort — fresh task.
    task_mlp = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    for w in pool:
        if isinstance(w, MlpWML):
            train_wml_on_task(w, nerve, task_mlp, steps=steps, lr=1e-2)

    # LIF cohort — fresh task, spike + learned head.
    task_lif = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    lif_encoders: dict[int, torch.nn.Linear] = {}
    for w in pool:
        if isinstance(w, LifWML):
            enc = torch.nn.Linear(16, w.n_neurons)
            opt = torch.optim.Adam(
                list(w.parameters()) + list(enc.parameters()), lr=1e-2,
            )
            for _ in range(steps):
                x, y = task_lif.sample(batch=64)
                i_in = w.input_proj(enc(x))
                spikes = spike_with_surrogate(i_in, v_thr=w.v_thr)
                logits = w.emit_head_pi(spikes)[:, : task_lif.n_classes]
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            lif_encoders[w.id] = enc

    # TRF cohort — fresh task.
    task_trf = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    for w in pool:
        if isinstance(w, TransformerWML):
            train_wml_on_task(w, nerve, task_trf, steps=steps, lr=1e-2)

    # Eval on common fresh task.
    task_eval = HardFlowProxyTask(dim=16, n_classes=12, seed=seed)
    x, y = task_eval.sample(batch=512)
    accs_mlp, accs_lif, accs_trf = [], [], []
    with torch.no_grad():
        for w in pool:
            if isinstance(w, MlpWML):
                pred = w.emit_head_pi(w.core(x))[:, : task_eval.n_classes].argmax(-1)
                accs_mlp.append((pred == y).float().mean().item())
            elif isinstance(w, LifWML):
                enc = lif_encoders[w.id]
                i_in = w.input_proj(enc(x))
                spikes = spike_with_surrogate(i_in, v_thr=w.v_thr)
                pred = w.emit_head_pi(spikes)[:, : task_eval.n_classes].argmax(-1)
                accs_lif.append((pred == y).float().mean().item())
            elif isinstance(w, TransformerWML):
                pred = w.emit_head_pi(w.core(x))[:, : task_eval.n_classes].argmax(-1)
                accs_trf.append((pred == y).float().mean().item())

    mean_mlp = float(np.mean(accs_mlp))
    mean_lif = float(np.mean(accs_lif))
    mean_trf = float(np.mean(accs_trf))
    means = [mean_mlp, mean_lif, mean_trf]
    triple_gap = (max(means) - min(means)) / max(max(means), 1e-6)
    return {
        "mean_acc_mlp": mean_mlp,
        "mean_acc_lif": mean_lif,
        "mean_acc_trf": mean_trf,
        "n_mlp": len(accs_mlp),
        "n_lif": len(accs_lif),
        "n_trf": len(accs_trf),
        "triple_gap": triple_gap,
    }


def run_triple_pool_hard_multiseed(
    seeds: list[int] | None = None, n_wmls: int = 15, steps: int = 400,
) -> dict:
    """Multi-seed triple-substrate at pool scale (N=15).

    5 seeds of 15-WML pools (5 MLP + 5 LIF + 5 TRF) on HardFlowProxyTask.
    Closes the symmetry gap in the v1.1.1 evidence matrix: the third
    substrate now has multi-seed pool-scale measurements too.
    """
    import numpy as np

    if seeds is None:
        seeds = list(range(5))
    per_seed = [
        run_triple_pool_hard(n_wmls=n_wmls, steps=steps, seed=s) for s in seeds
    ]
    triple_gaps = [r["triple_gap"] for r in per_seed]
    accs_mlp = [r["mean_acc_mlp"] for r in per_seed]
    accs_lif = [r["mean_acc_lif"] for r in per_seed]
    accs_trf = [r["mean_acc_trf"] for r in per_seed]
    return {
        "seeds":             list(seeds),
        "triple_gaps":       triple_gaps,
        "accs_mlp":          accs_mlp,
        "accs_lif":          accs_lif,
        "accs_trf":          accs_trf,
        "mean_triple_gap":   float(np.mean(triple_gaps)),
        "median_triple_gap": float(np.median(triple_gaps)),
        "max_triple_gap":    float(np.max(triple_gaps)),
        "mean_acc_mlp":      float(np.mean(accs_mlp)),
        "mean_acc_lif":      float(np.mean(accs_lif)),
        "mean_acc_trf":      float(np.mean(accs_trf)),
    }


def run_w_triple_substrate_multiseed(
    seeds: list[int] | None = None,
    steps: int = 400,
    hard: bool = True,
) -> dict:
    """Multi-seed triple-substrate on HardFlowProxyTask.

    Closes the symmetry gap in the v1.0 paper: MLP/LIF have 4 scaling
    points, TRF had only N=1. This gives us multi-seed statistics for
    the third substrate too, even though it stays at N=1 per seed.
    """
    import numpy as np

    if seeds is None:
        seeds = list(range(5))
    per_seed = [
        run_w_triple_substrate(steps=steps, hard=hard, seed=s) for s in seeds
    ]
    accs_mlp = [r["acc_mlp"] for r in per_seed]
    accs_lif = [r["acc_lif"] for r in per_seed]
    accs_trf = [r["acc_trf"] for r in per_seed]
    triple_gaps = [r["triple_gap"] for r in per_seed]
    return {
        "seeds":            list(seeds),
        "accs_mlp":         accs_mlp,
        "accs_lif":         accs_lif,
        "accs_trf":         accs_trf,
        "triple_gaps":      triple_gaps,
        "mean_acc_mlp":     float(np.mean(accs_mlp)),
        "mean_acc_lif":     float(np.mean(accs_lif)),
        "mean_acc_trf":     float(np.mean(accs_trf)),
        "mean_triple_gap":  float(np.mean(triple_gaps)),
        "median_triple_gap": float(np.median(triple_gaps)),
        "max_triple_gap":   float(np.max(triple_gaps)),
    }


def _run_w2_hard_scale(
    n_wmls: int, steps: int, seed: int = 0,
    *, task_seed: int | None = None, pool_seed: int | None = None,
) -> dict:
    """Shared core for W2-hard at scale (N=16, N=32, N=64).

    RNG architecture (v1.1 refactor):
      - `seed` is the default cascade; each of pool, nerve, and task
        gets `seed` unless explicitly overridden.
      - `task_seed` and `pool_seed` allow decoupling task data
        distribution from substrate initialization for ablations.
      - Per cohort (MLP vs LIF) we still use fresh task instances
        sharing the same `task_seed` — this preserves the cross-
        substrate RNG isolation introduced in run_w2_hard and
        prevents MLP.train from poisoning the LIF data stream.

    v1.1 also removes the `wml._input_encoder` monkey-patching that
    earlier versions relied on. We now keep the input_encoder in a
    separate dict keyed by WML id, scoped to this function.
    """
    import numpy as np
    import torch.nn.functional as F  # noqa: N812

    from track_w._surrogate import spike_with_surrogate
    from track_w.pool_factory import build_pool, k_for_n
    from track_w.tasks.hard_flow_proxy import HardFlowProxyTask

    if task_seed is None:
        task_seed = seed
    if pool_seed is None:
        pool_seed = seed

    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=n_wmls, k=k_for_n(n_wmls), seed=pool_seed)
    nerve.set_phase_active(gamma=True, theta=False)
    pool = build_pool(n_wmls=n_wmls, mlp_frac=0.5, seed=pool_seed)

    # MLP cohort — fresh task instance for RNG isolation.
    task_mlp = HardFlowProxyTask(dim=16, n_classes=12, seed=task_seed)
    for wml in pool:
        if isinstance(wml, MlpWML):
            train_wml_on_task(wml, nerve, task_mlp, steps=steps, lr=1e-2)

    # LIF cohort — fresh task, full spike + learned head (v0.4 symmetry).
    # Keep input_encoders in a local registry (no more monkey-patching).
    lif_encoders: dict[int, torch.nn.Linear] = {}
    task_lif = HardFlowProxyTask(dim=16, n_classes=12, seed=task_seed)
    for wml in pool:
        if isinstance(wml, LifWML):
            input_encoder = torch.nn.Linear(16, wml.n_neurons)
            opt = torch.optim.Adam(
                list(wml.parameters()) + list(input_encoder.parameters()),
                lr=1e-2,
            )
            for _ in range(steps):
                x, y = task_lif.sample(batch=64)
                i_in = wml.input_proj(input_encoder(x))
                spikes = spike_with_surrogate(i_in, v_thr=wml.v_thr)
                logits = wml.emit_head_pi(spikes)[:, : task_lif.n_classes]
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            lif_encoders[wml.id] = input_encoder

    # Eval — fresh task instance, same distribution for both substrates.
    task_eval = HardFlowProxyTask(dim=16, n_classes=12, seed=task_seed)
    x, y = task_eval.sample(batch=512)
    mlp_accs, lif_accs = [], []
    with torch.no_grad():
        for wml in pool:
            if isinstance(wml, MlpWML):
                pred = wml.emit_head_pi(wml.core(x))[:, : task_eval.n_classes].argmax(-1)
                mlp_accs.append((pred == y).float().mean().item())
            elif isinstance(wml, LifWML):
                enc = lif_encoders[wml.id]
                i_in = wml.input_proj(enc(x))
                spikes = spike_with_surrogate(i_in, v_thr=wml.v_thr)
                pred = wml.emit_head_pi(spikes)[:, : task_eval.n_classes].argmax(-1)
                lif_accs.append((pred == y).float().mean().item())

    mean_mlp = float(np.mean(mlp_accs))
    mean_lif = float(np.mean(lif_accs))
    return {
        "mean_acc_mlp": mean_mlp,
        "mean_acc_lif": mean_lif,
        "n_mlp": len(mlp_accs),
        "n_lif": len(lif_accs),
        "gap": abs(mean_mlp - mean_lif) / max(mean_mlp, 1e-6),
    }


def run_w2_hard_n16(steps: int = 400, seed: int = 0) -> dict:
    """W2-N16 HARD — 16-WML pool on HardFlowProxyTask with v0.4 heads."""
    return _run_w2_hard_scale(n_wmls=16, steps=steps, seed=seed)


def run_w2_hard_n32(steps: int = 200, seed: int = 0) -> dict:
    """W2-N32 HARD — 32-WML pool on HardFlowProxyTask with v0.4 heads.

    Uses reduced step budget to keep runtime bounded; the polymorphism
    question at this scale is whether the reversal (LIF ≥ MLP) observed
    at N=2 persists, not absolute accuracy.
    """
    return _run_w2_hard_scale(n_wmls=32, steps=steps, seed=seed)


def run_w2_hard_n64(steps: int = 150, seed: int = 0) -> dict:
    """W2-N64 HARD — 64-WML pool on HardFlowProxyTask with v0.4 heads.

    Reduced step budget (150) relative to N=32's 200; the polymorphism
    question at this scale is distributional closure, not absolute
    accuracy. Extrapolation point for the v0.7 scaling law.
    """
    return _run_w2_hard_scale(n_wmls=64, steps=steps, seed=seed)


def run_w2_hard_n64_multiseed(
    seeds: list[int] | None = None,
    steps: int = 150,
) -> dict:
    """Multi-seed W2-N64 — fourth scaling-law point.

    Tests whether the monotonic decrease (N=2 → 10.7 %, N=16 → 6.71 %,
    N=32 → 2.39 %) continues, producing a 4-point log-log fit. If
    median at N=64 is < 2 %, the scaling law holds empirically.
    """
    import numpy as np

    if seeds is None:
        seeds = list(range(5))
    per_seed = [run_w2_hard_n64(steps=steps, seed=s) for s in seeds]
    gaps     = [r["gap"]          for r in per_seed]
    accs_mlp = [r["mean_acc_mlp"] for r in per_seed]
    accs_lif = [r["mean_acc_lif"] for r in per_seed]
    return {
        "seeds":        list(seeds),
        "gaps":         gaps,
        "accs_mlp":     accs_mlp,
        "accs_lif":     accs_lif,
        "mean_gap":     float(np.mean(gaps)),
        "median_gap":   float(np.median(gaps)),
        "p25_gap":      float(np.percentile(gaps, 25)),
        "p75_gap":      float(np.percentile(gaps, 75)),
        "max_gap":      float(np.max(gaps)),
        "mean_acc_mlp": float(np.mean(accs_mlp)),
        "mean_acc_lif": float(np.mean(accs_lif)),
    }


def run_w2_hard_n32_multiseed(
    seeds: list[int] | None = None,
    steps: int = 200,
) -> dict:
    """Multi-seed W2-N32 HARD — same methodology as _n16 at larger pool.

    The question this pilot answers: does the ~7 % distributional gap
    observed at N=16 continue to hold at N=32, or does further pool
    averaging compress it toward the 5 % contract? A stable gap across
    scales indicates substrate-intrinsic asymmetry; a scale-dependent
    one indicates residual variance.
    """
    import numpy as np

    if seeds is None:
        seeds = list(range(5))
    per_seed = [run_w2_hard_n32(steps=steps, seed=s) for s in seeds]
    gaps     = [r["gap"]          for r in per_seed]
    accs_mlp = [r["mean_acc_mlp"] for r in per_seed]
    accs_lif = [r["mean_acc_lif"] for r in per_seed]
    return {
        "seeds":        list(seeds),
        "gaps":         gaps,
        "accs_mlp":     accs_mlp,
        "accs_lif":     accs_lif,
        "mean_gap":     float(np.mean(gaps)),
        "median_gap":   float(np.median(gaps)),
        "p25_gap":      float(np.percentile(gaps, 25)),
        "p75_gap":      float(np.percentile(gaps, 75)),
        "max_gap":      float(np.max(gaps)),
        "mean_acc_mlp": float(np.mean(accs_mlp)),
        "mean_acc_lif": float(np.mean(accs_lif)),
    }


def run_w2_hard_n16_multiseed(
    seeds: list[int] | None = None,
    steps: int = 400,
) -> dict:
    """Multi-seed W2-N16 HARD — statistical closure of the polymorphism gap.

    Single-seed N=16 reported `gap = 1.68 %` (§W2-hard-scale v0.4). This
    pilot re-runs the same configuration across multiple independent
    seeds so we can report the **median** and **IQR** of the gap rather
    than a single-seed point. The scientific claim the median addresses:
    "at pool scale, the substrate-agnostic polymorphism contract holds
    in distribution, not merely for a single lucky seed".

    Args:
        seeds: RNG seeds to try. Each seed cascades to nerve topology,
               pool initialization, and task data. Default = [0..4].
        steps: SGD steps per substrate cohort per seed.

    Returns:
        dict with per-seed gaps, accuracies, and summary stats
        (mean, median, p25, p75, max).
    """
    import numpy as np

    if seeds is None:
        seeds = list(range(5))
    per_seed = [run_w2_hard_n16(steps=steps, seed=s) for s in seeds]
    gaps     = [r["gap"]          for r in per_seed]
    accs_mlp = [r["mean_acc_mlp"] for r in per_seed]
    accs_lif = [r["mean_acc_lif"] for r in per_seed]
    return {
        "seeds":          list(seeds),
        "gaps":           gaps,
        "accs_mlp":       accs_mlp,
        "accs_lif":       accs_lif,
        "mean_gap":       float(np.mean(gaps)),
        "median_gap":     float(np.median(gaps)),
        "p25_gap":        float(np.percentile(gaps, 25)),
        "p75_gap":        float(np.percentile(gaps, 75)),
        "max_gap":        float(np.max(gaps)),
        "mean_acc_mlp":   float(np.mean(accs_mlp)),
        "mean_acc_lif":   float(np.mean(accs_lif)),
    }


def run_w4_n16(steps: int = 400, rehearsal_frac: float = 0.3) -> dict:
    """W4-N16 — rehearsal-based continual learning on a 16-WML all-MLP pool.

    Train every WML on Split-MNIST-like Task 0, measure initial acc, then
    train on Task 1 with rehearsal_frac mixed from Task 0, measure retention.
    Forgetting ratio averaged across all 16 WMLs.

    Uses all MLP pool (mlp_frac=1.0) so evaluation can cleanly use
    emit_head_pi. LIF continual learning is left for future work.
    """
    import numpy as np

    from track_w.pool_factory import build_pool, k_for_n

    torch.manual_seed(0)
    n_wmls = 16
    nerve = MockNerve(n_wmls=n_wmls, k=k_for_n(n_wmls), seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    pool = build_pool(n_wmls=n_wmls, mlp_frac=1.0, seed=0)
    split = SplitMnistLikeTask(seed=0)

    opts = [torch.optim.Adam(wml.parameters(), lr=1e-2) for wml in pool]

    def _train(task, n_steps):
        for _ in range(n_steps):
            for wml, opt in zip(pool, opts, strict=True):
                x, y = task.sample(batch=64)
                logits = wml.emit_head_pi(wml.core(x))[:, : 4]
                loss = torch.nn.functional.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _eval(task) -> float:
        accs = []
        x, y = task.sample(batch=256)
        with torch.no_grad():
            for wml in pool:
                pred = wml.emit_head_pi(wml.core(x))[:, : 4].argmax(-1)
                accs.append((pred == y).float().mean().item())
        return float(np.mean(accs))

    # Task 0.
    _train(split.subtasks[0], n_steps=steps)
    acc0_initial = _eval(split.subtasks[0])

    # Task 1 with rehearsal.
    n_rehearsal = int(64 * rehearsal_frac)
    n_new = 64 - n_rehearsal
    for _ in range(steps):
        for wml, opt in zip(pool, opts, strict=True):
            x_new, y_new = split.subtasks[1].sample(batch=n_new)
            x_old, y_old = split.subtasks[0].sample(batch=n_rehearsal)
            logits_new = wml.emit_head_pi(wml.core(x_new))[:, : 4]
            logits_old = wml.emit_head_pi(wml.core(x_old))[:, : 4]
            loss_new = torch.nn.functional.cross_entropy(logits_new, y_new)
            loss_old = torch.nn.functional.cross_entropy(logits_old, y_old)
            loss = (loss_new * n_new + loss_old * n_rehearsal) / 64
            opt.zero_grad()
        loss.backward()
        opt.step()

    acc0_after = _eval(split.subtasks[0])
    acc1_after = _eval(split.subtasks[1])

    forgetting = (acc0_initial - acc0_after) / max(acc0_initial, 1e-6)
    return {
        "acc_task0_initial":     acc0_initial,
        "acc_task0_after_task1": acc0_after,
        "acc_task1_after_task1": acc1_after,
        "forgetting":            forgetting,
    }


def run_w2_n32(steps: int = 200) -> dict:
    """W2-N32 stress — half MLP / half LIF pool at N=32.

    Mirrors run_w2_n16 but at 2× the pool size, with reduced steps (200)
    to keep wall time under ~3 minutes. Returns mean accuracy per substrate
    plus pool metadata.

    Primary goal: no crash. Secondary: relative gap < 15 % (soft).
    """
    import numpy as np

    from track_w.pool_factory import build_pool, k_for_n

    torch.manual_seed(0)
    n_wmls = 32
    nerve = MockNerve(n_wmls=n_wmls, k=k_for_n(n_wmls), seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    pool = build_pool(n_wmls=n_wmls, mlp_frac=0.5, seed=0)
    task = FlowProxyTask(dim=16, n_classes=4, seed=0)

    # MLPs via task loss.
    for wml in pool:
        if isinstance(wml, MlpWML):
            train_wml_on_task(wml, nerve, task, steps=steps, lr=1e-2)

    # LIFs via linear probe on input_proj.
    for wml in pool:
        if isinstance(wml, LifWML):
            probe = torch.nn.Linear(wml.n_neurons, task.n_classes)
            opt = torch.optim.Adam(
                list(wml.input_proj.parameters()) + list(probe.parameters()),
                lr=1e-2,
            )
            for _ in range(steps):
                x, y = task.sample(batch=64)
                pooled = x @ (torch.eye(16, wml.n_neurons) / 4)
                logits = probe(wml.input_proj(pooled))
                loss = torch.nn.functional.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            wml._probe = probe

    # Evaluate.
    x, y = task.sample(batch=256)
    mlp_accs, lif_accs = [], []
    with torch.no_grad():
        for wml in pool:
            if isinstance(wml, MlpWML):
                pred = wml.emit_head_pi(wml.core(x))[:, : task.n_classes].argmax(-1)
                mlp_accs.append((pred == y).float().mean().item())
            else:
                pooled = x @ (torch.eye(16, wml.n_neurons) / 4)
                pred = wml._probe(wml.input_proj(pooled)).argmax(-1)
                lif_accs.append((pred == y).float().mean().item())

    return {
        "mean_acc_mlp": float(np.mean(mlp_accs)),
        "mean_acc_lif": float(np.mean(lif_accs)),
        "n_mlp": len(mlp_accs),
        "n_lif": len(lif_accs),
    }


def run_gate_scale() -> dict:
    """Run W1-N16, W2-N16, W4-N16, W2-N32 end-to-end. Return a single report."""
    torch.manual_seed(0)
    w1_acc = run_w1_n16(steps=400)

    w2_n16 = run_w2_n16(steps=400)
    w2_n16_gap = abs(w2_n16["mean_acc_mlp"] - w2_n16["mean_acc_lif"]) / max(
        w2_n16["mean_acc_mlp"], 1e-6
    )

    w4_n16 = run_w4_n16(steps=400)

    w2_n32 = run_w2_n32(steps=200)

    all_passed = (
        w1_acc > 0.6
        and w2_n16["mean_acc_mlp"] > 0.6
        and w2_n16["mean_acc_lif"] > 0.6
        and w2_n16_gap < 0.10
        and w4_n16["forgetting"] < 0.20
        and w2_n32["n_mlp"] == 16
        and w2_n32["n_lif"] == 16
    )

    return {
        "w1_n16_accuracy":     w1_acc,
        "w2_n16_acc_mlp":      w2_n16["mean_acc_mlp"],
        "w2_n16_acc_lif":      w2_n16["mean_acc_lif"],
        "w2_n16_gap":          w2_n16_gap,
        "w4_n16_forgetting":   w4_n16["forgetting"],
        "w2_n32_acc_mlp":      w2_n32["mean_acc_mlp"],
        "w2_n32_acc_lif":      w2_n32["mean_acc_lif"],
        "w2_n32_n_mlp":        w2_n32["n_mlp"],
        "w2_n32_n_lif":        w2_n32["n_lif"],
        "all_passed":          all_passed,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "scale":
        print(json.dumps(run_gate_scale(), indent=2))
    else:
        print(json.dumps(run_gate_w(), indent=2))
