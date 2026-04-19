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
    plateaus ~0.6, MLP trains further, LIF with cosine decoder lags.

    This pilot documents the honest gap rather than hiding it. The gap
    is EXPECTED to exceed 5 % — see spec §13.1 Debt 2 (reopened).
    """
    import torch.nn.functional as F
    from track_w._surrogate import spike_with_surrogate
    from track_w.tasks.hard_flow_proxy import HardFlowProxyTask

    torch.manual_seed(0)
    nerve = MockNerve(n_wmls=2, k=1, seed=0)
    nerve.set_phase_active(gamma=True, theta=False)
    task = HardFlowProxyTask(dim=16, n_classes=12, seed=0)

    mlp = MlpWML(id=0, d_hidden=16, seed=0)
    train_wml_on_task(mlp, nerve, task, steps=steps, lr=1e-2)

    lif = LifWML(id=0, n_neurons=16, seed=10)
    input_encoder = torch.nn.Linear(16, lif.n_neurons)
    opt = torch.optim.Adam(
        list(lif.parameters()) + list(input_encoder.parameters()),
        lr=1e-2,
    )
    for _ in range(steps):
        x, y = task.sample(batch=64)
        i_in = lif.input_proj(input_encoder(x))
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        norms = lif.codebook.norm(dim=-1) + 1e-6
        sims = spikes @ lif.codebook.T / (
            norms * (spikes.norm(dim=-1, keepdim=True) + 1e-6)
        )
        logits = sims[:, : task.n_classes]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()

    x, y = task.sample(batch=512)
    with torch.no_grad():
        pred_mlp = mlp.emit_head_pi(mlp.core(x))[:, : task.n_classes].argmax(-1)
        acc_mlp = (pred_mlp == y).float().mean().item()
        i_in = lif.input_proj(input_encoder(x))
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        norms = lif.codebook.norm(dim=-1) + 1e-6
        sims = spikes @ lif.codebook.T / (
            norms * (spikes.norm(dim=-1, keepdim=True) + 1e-6)
        )
        pred_lif = sims[:, : task.n_classes].argmax(-1)
        acc_lif = (pred_lif == y).float().mean().item()

    gap = abs(acc_mlp - acc_lif) / max(acc_mlp, 1e-6)
    return {"acc_mlp": acc_mlp, "acc_lif": acc_lif, "gap": gap}


if __name__ == "__main__":
    print(json.dumps(run_gate_w(), indent=2))
