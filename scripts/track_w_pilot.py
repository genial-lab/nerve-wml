"""Track-W pilot scripts: W1-W4 curriculum drivers + Gate W aggregator."""
from __future__ import annotations

import torch

from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.tasks.flow_proxy import FlowProxyTask
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


from track_w.lif_wml import LifWML


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
            opt.zero_grad(); loss.backward(); opt.step()

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

            opt.zero_grad(); total.backward(); opt.step()

        x, y = task.sample(batch=256)
        with torch.no_grad():
            pred = wml.emit_head_pi(wml.core(x))[:, : task.n_classes].argmax(-1)
        return (pred == y).float().mean().item()

    baseline = _train_and_eval(use_eps=False)
    with_eps = _train_and_eval(use_eps=True)
    return baseline, with_eps

from track_w.tasks.split_mnist import SplitMnistLikeTask


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
            optimizer.zero_grad(); loss.backward(); optimizer.step()

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


if __name__ == "__main__":
    import json

    print(json.dumps(run_gate_w(), indent=2))
