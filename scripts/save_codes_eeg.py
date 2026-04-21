"""Train MLP+LIF substrates on Sleep-EDF EEG and save emitted codes.

Stage 2 of the Sleep-EDF protocol (see
docs/research-notes/sleep-edf-pipeline-protocol.md):

Loads tests/golden/sleep_edf_epochs.npz produced by
scripts/eeg_preprocess_sleep_edf.py, trains MLP and LIF
substrates with the same recipe as scripts/save_codes_for_checks.py
(homogeneous to run_w2_hard) but on real EEG epochs of shape
[N, 2, 3000] flattened to [N, 6000].

Output: tests/golden/codes_mlp_lif_eeg.npz with the same schema
as tests/golden/codes_mlp_lif.npz so that all measure_mi_*.py
scripts work without modification.

Run on kxkm-ai (GPU optional, CPU works for first pass):

    uv run python scripts/save_codes_eeg.py \\
        --epochs tests/golden/sleep_edf_epochs.npz \\
        --seeds 0 1 2 \\
        --out tests/golden/codes_mlp_lif_eeg.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from track_w._surrogate import spike_with_surrogate
from track_w.lif_wml import LifWML
from track_w.mlp_wml import MlpWML
from track_w.mock_nerve import MockNerve
from track_w.training import train_wml_on_task


class _EegTaskAdapter:
    """Wraps an x[N, d_in] / y[N] tensor pair into the
    HardFlowProxyTask-style sample(batch=B) -> (x, y) interface
    with optional class-balanced sampling to counter Sleep-EDF's
    severe class imbalance (Wake ~68%, N1/N3 ~4% each).

    When class_balanced=True, each batch contains approximately
    batch/n_classes samples per class, draw uniformly from the
    indices of that class. This prevents the loss from being
    dominated by Wake, which causes mode collapse under plain
    uniform sampling.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_classes: int,
        class_balanced: bool = True,
    ) -> None:
        self.x = x
        self.y = y
        self.n_classes = n_classes
        self._n = x.shape[0]
        self._class_balanced = class_balanced
        if class_balanced:
            self._idx_per_class: list[torch.Tensor] = [
                torch.where(y == c)[0] for c in range(n_classes)
            ]
            self._non_empty_classes = [
                c for c, idxs in enumerate(self._idx_per_class) if len(idxs) > 0
            ]

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._class_balanced:
            idx = torch.randint(0, self._n, (batch,))
            return self.x[idx], self.y[idx]
        classes = torch.tensor(
            [
                self._non_empty_classes[i % len(self._non_empty_classes)]
                for i in torch.randperm(batch).tolist()
            ],
            dtype=torch.long,
        )
        idx = torch.empty(batch, dtype=torch.long)
        for i, c in enumerate(classes.tolist()):
            class_idx = self._idx_per_class[c]
            idx[i] = class_idx[torch.randint(0, len(class_idx), (1,)).item()]
        return self.x[idx], self.y[idx]


def _train_pair_eeg(
    x_train:   torch.Tensor,
    y_train:   torch.Tensor,
    n_classes: int,
    d_in:      int,
    d_hidden:  int,
    seed:      int,
    steps:     int,
    lr:        float = 1e-2,
) -> tuple[MlpWML, LifWML]:
    """Train MLP + LIF on EEG.

    Both substrates are instantiated with input_dim=d_in so their
    internal projection (mlp.core first layer / lif.input_proj)
    handles the high-dim EEG input directly; no external encoder
    is needed, unlike save_codes_for_checks.py where d_in equals
    n_neurons and the encoder is a simple same-dim projection.
    """
    torch.manual_seed(seed)
    nerve = MockNerve(n_wmls=2, k=1, seed=seed)
    nerve.set_phase_active(gamma=True, theta=False)

    task_mlp = _EegTaskAdapter(x_train, y_train, n_classes=n_classes)
    mlp = MlpWML(id=0, d_hidden=d_hidden, input_dim=d_in, seed=seed)
    train_wml_on_task(mlp, nerve, task_mlp, steps=steps, lr=lr)

    task_lif = _EegTaskAdapter(x_train, y_train, n_classes=n_classes)
    lif = LifWML(id=0, n_neurons=d_hidden, input_dim=d_in, seed=seed + 10)
    opt = torch.optim.Adam(lif.parameters(), lr=lr)
    for _ in range(steps):
        x, y = task_lif.sample(batch=64)
        i_in = lif.input_proj(x)
        spikes = spike_with_surrogate(i_in, v_thr=lif.v_thr)
        logits = lif.emit_head_pi(spikes)[:, : task_lif.n_classes]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return mlp, lif


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=Path,
        required=True,
        help="NPZ produced by eeg_preprocess_sleep_edf.py.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--d-hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/golden/codes_mlp_lif_eeg.npz"),
    )
    args = parser.parse_args()

    if not args.epochs.exists():
        raise FileNotFoundError(
            f"{args.epochs} not found. Produce it via "
            "scripts/eeg_preprocess_sleep_edf.py on kxkm-ai first."
        )

    data = np.load(args.epochs)
    x_train_np = data["x_train"]
    y_train_np = data["y_train"]
    x_test_np = data["x_test"]
    y_test_np = data["y_test"]

    n_classes = int(max(y_train_np.max(), y_test_np.max())) + 1
    n_ch, n_samp = x_train_np.shape[1], x_train_np.shape[2]
    d_in = n_ch * n_samp

    x_train = torch.from_numpy(x_train_np.reshape(-1, d_in).astype(np.float32))
    y_train = torch.from_numpy(y_train_np.astype(np.int64))
    x_test = torch.from_numpy(x_test_np.reshape(-1, d_in).astype(np.float32))
    y_test = torch.from_numpy(y_test_np.astype(np.int64))

    # Per-feature z-score normalisation using TRAIN stats only.
    # Raw EEG amplitudes span 1e-5 to 1e-3 volts; unnormalised
    # inputs drove the MLP/LIF pair to immediate mode collapse
    # onto the Wake class at d_in=6000.
    train_mean = x_train.mean(dim=0, keepdim=True)
    train_std = x_train.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    print(
        f"EEG epochs: train {x_train.shape}, test {x_test.shape}, "
        f"{n_classes} classes, d_in={d_in}"
    )
    print(
        f"After z-score: train mean={x_train.mean().item():+.4f} "
        f"std={x_train.std().item():.4f}"
    )
    label_counts_train = torch.bincount(y_train, minlength=n_classes).tolist()
    print(f"Train label counts: {dict(enumerate(label_counts_train))}")

    all_codes_mlp: list[np.ndarray] = []
    all_codes_lif: list[np.ndarray] = []
    all_emb_mlp: list[np.ndarray] = []
    all_emb_lif: list[np.ndarray] = []
    accs_mlp: list[float] = []
    accs_lif: list[float] = []

    for seed in args.seeds:
        print(f"seed {seed}: training MLP + LIF on EEG ({args.steps} steps)...")
        mlp, lif = _train_pair_eeg(
            x_train=x_train,
            y_train=y_train,
            n_classes=n_classes,
            d_in=d_in,
            d_hidden=args.d_hidden,
            seed=seed,
            steps=args.steps,
            lr=args.lr,
        )

        with torch.no_grad():
            mlp_emb = mlp.core(x_test)
            mlp_codes = mlp.emit_head_pi(mlp_emb)[:, :n_classes].argmax(-1)
            lif_emb = lif.input_proj(x_test)
            spikes = spike_with_surrogate(lif_emb, v_thr=lif.v_thr)
            lif_codes = lif.emit_head_pi(spikes)[:, :n_classes].argmax(-1)
            acc_mlp = (mlp_codes == y_test).float().mean().item()
            acc_lif = (lif_codes == y_test).float().mean().item()

        all_codes_mlp.append(mlp_codes.cpu().numpy().astype(np.int64))
        all_codes_lif.append(lif_codes.cpu().numpy().astype(np.int64))
        all_emb_mlp.append(mlp_emb.cpu().numpy().astype(np.float32))
        all_emb_lif.append(lif_emb.cpu().numpy().astype(np.float32))
        accs_mlp.append(acc_mlp)
        accs_lif.append(acc_lif)
        print(
            f"  acc_mlp={acc_mlp:.4f}, acc_lif={acc_lif:.4f}, "
            f"alphabet_mlp={len(np.unique(all_codes_mlp[-1]))}/64, "
            f"lif={len(np.unique(all_codes_lif[-1]))}/64"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        mlp_codes=np.stack(all_codes_mlp),
        lif_codes=np.stack(all_codes_lif),
        mlp_embeddings=np.stack(all_emb_mlp),
        lif_embeddings=np.stack(all_emb_lif),
        acc_mlp=np.asarray(accs_mlp, dtype=np.float32),
        acc_lif=np.asarray(accs_lif, dtype=np.float32),
        seeds=np.asarray(args.seeds, dtype=np.int64),
        n_classes=n_classes,
    )
    print()
    print(f"Saved: {args.out}")
    print(
        f"Mean acc: MLP={np.mean(accs_mlp):.4f}, LIF={np.mean(accs_lif):.4f}"
    )
    print(
        f"Mean pairwise gap: {abs(np.mean(accs_mlp) - np.mean(accs_lif)):.4f}"
    )


if __name__ == "__main__":
    main()
