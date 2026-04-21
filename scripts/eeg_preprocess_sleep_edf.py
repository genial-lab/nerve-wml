"""Sleep-EDF EEG preprocessing for nerve-wml Voie 3 Semaines 3-5.

Loads PhysioNet Sleep-EDF Expanded recordings via MNE-Python,
filters / downsamples / segments into 30-second epochs, attaches
sleep-stage labels from the hypnogram, and saves a single NPZ
artefact ready for substrate training (scripts/save_codes_eeg.py).

Pipeline (Stage 1 of the Sleep-EDF protocol -- see
docs/research-notes/sleep-edf-pipeline-protocol.md):

  1. Download/load (psg, hypnogram) EDF pairs for N subjects.
  2. Pick 2 EEG channels (Fpz-Cz, Pz-Oz), bandpass 0.5-30 Hz,
     downsample to 100 Hz.
  3. Read sleep-stage annotations, map to 5 classes
     (W=0, N1=1, N2=2, N3=3, REM=4).
  4. Segment into 30-second non-overlapping epochs.
  5. Patient-wise train/val/test split (70/15/15).
  6. Save tests/golden/sleep_edf_epochs.npz.

Run on kxkm-ai (NOT GrosMac per user policy):

    ssh kxkm@kxkm-ai
    cd ~/nerve-wml
    uv run python scripts/eeg_preprocess_sleep_edf.py \\
        --subjects 0 1 2 3 4 5 6 7 8 9 \\
        --out tests/golden/sleep_edf_epochs.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import mne
from mne.datasets import sleep_physionet


SLEEP_STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # collapse N3+N4 into N3 (R&K -> AASM convention)
    "Sleep stage R": 4,
    # "Sleep stage ?": skip (uncertain)
    # "Movement time": skip
}


def _load_subject_epochs(
    psg_path: Path,
    hypnogram_path: Path,
    channels:        tuple[str, ...] = ("EEG Fpz-Cz", "EEG Pz-Oz"),
    bandpass:        tuple[float, float] = (0.5, 30.0),
    target_sfreq:    float = 100.0,
    epoch_length_s:  float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (epochs[N, n_ch, n_samples], labels[N]) for one subject.

    Pipeline: read EDF, pick 2 EEG channels, bandpass 0.5-30 Hz,
    downsample to 100 Hz, attach hypnogram annotations, segment
    into 30-second non-overlapping epochs labelled by dominant
    sleep stage.
    """
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    raw.pick(list(channels))
    raw.filter(*bandpass, verbose=False)
    raw.resample(target_sfreq, verbose=False)

    annotations = mne.read_annotations(str(hypnogram_path))
    raw.set_annotations(annotations, emit_warning=False)

    available_stages = set(annotations.description) & set(SLEEP_STAGE_MAP)
    event_id = {k: SLEEP_STAGE_MAP[k] for k in available_stages}
    if not event_id:
        return (
            np.empty((0, len(channels), int(epoch_length_s * target_sfreq))),
            np.empty(0, dtype=np.int64),
        )

    events, _ = mne.events_from_annotations(
        raw, event_id=event_id,
        chunk_duration=epoch_length_s, verbose=False,
    )
    if events.size == 0:
        return (
            np.empty((0, len(channels), int(epoch_length_s * target_sfreq))),
            np.empty(0, dtype=np.int64),
        )

    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=epoch_length_s - 1.0 / target_sfreq,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return (
        epochs.get_data().astype(np.float32),
        epochs.events[:, -1].astype(np.int64),
    )


def _split_per_subject(
    epochs:     np.ndarray,
    labels:     np.ndarray,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = 0,
) -> tuple[np.ndarray, ...]:
    """Time-respecting split within one subject (no shuffle leak)."""
    n = epochs.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    tr, va, te = perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
    return (
        epochs[tr], labels[tr],
        epochs[va], labels[va],
        epochs[te], labels[te],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=list(range(10)),
        help="Sleep-EDF age-subset subject IDs (0-77).",
    )
    parser.add_argument("--recording", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/golden/sleep_edf_epochs.npz"),
    )
    args = parser.parse_args()

    all_x_train: list[np.ndarray] = []
    all_y_train: list[np.ndarray] = []
    all_x_val:   list[np.ndarray] = []
    all_y_val:   list[np.ndarray] = []
    all_x_test:  list[np.ndarray] = []
    all_y_test:  list[np.ndarray] = []
    n_total_epochs = 0

    for subj_id in args.subjects:
        files = sleep_physionet.age.fetch_data(
            subjects=[subj_id], recording=[args.recording], verbose=False,
        )
        psg_path, hypnogram_path = map(Path, files[0])
        epochs, labels = _load_subject_epochs(psg_path, hypnogram_path)
        if epochs.shape[0] == 0:
            print(f"  subject {subj_id}: no usable epochs, skipped")
            continue
        x_tr, y_tr, x_va, y_va, x_te, y_te = _split_per_subject(
            epochs, labels, seed=subj_id,
        )
        all_x_train.append(x_tr); all_y_train.append(y_tr)
        all_x_val.append(x_va);   all_y_val.append(y_va)
        all_x_test.append(x_te);  all_y_test.append(y_te)
        n_total_epochs += epochs.shape[0]
        print(
            f"  subject {subj_id}: {epochs.shape[0]} epochs "
            f"({x_tr.shape[0]} train / {x_va.shape[0]} val / {x_te.shape[0]} test)"
        )

    if not all_x_train:
        raise RuntimeError("no subject produced usable epochs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        x_train=np.concatenate(all_x_train),
        y_train=np.concatenate(all_y_train),
        x_val=np.concatenate(all_x_val),
        y_val=np.concatenate(all_y_val),
        x_test=np.concatenate(all_x_test),
        y_test=np.concatenate(all_y_test),
        subjects=np.asarray(args.subjects, dtype=np.int64),
    )
    print()
    print(f"Saved: {args.out}")
    print(
        f"Total epochs: {n_total_epochs} "
        f"({sum(x.shape[0] for x in all_x_train)} train / "
        f"{sum(x.shape[0] for x in all_x_val)} val / "
        f"{sum(x.shape[0] for x in all_x_test)} test)"
    )
    label_counts = np.bincount(np.concatenate(all_y_train).astype(np.int64))
    print(f"Train label distribution: {dict(enumerate(label_counts.tolist()))}")


if __name__ == "__main__":
    main()
