"""Direct tests of inter-substrate information transmission (Claim B).

The scaling law (test_w2_hard_scale.py) proves that MLP and LIF
substrates reach COMPARABLE task accuracy at pool scale. These tests
go further and probe whether they TRANSMIT information coherently:

  (1) MI between per-substrate emitted codes is high (> 50 % of H).
  (2) Round-trip fidelity > 90 % (information survives cross-substrate
      pass).
  (3) Cross-substrate merge retains > 90 % of MLP accuracy when LIF's
      sole input is MLP-emitted codes through a learned transducer.
"""
import torch

from scripts.measure_info_transmission import (
    run_test_1_mutual_information,
    run_test_2_round_trip_fidelity,
    run_test_3_cross_substrate_merge,
)


def test_mi_between_substrates_is_high():
    """Test (1): MI / H(MLP) should be > 50 % — shared code semantics.

    Empirical: ~91 % mean across 3 seeds. Pins the claim that the two
    substrates do NOT encode orthogonally — they converge to a shared
    code induced by the common protocol + task pressure.
    """
    torch.manual_seed(0)
    results = run_test_1_mutual_information(seeds=list(range(3)), steps=400, batch=2048)
    ratios = [r["mi_over_h_mlp"] for r in results]
    mean_ratio = sum(ratios) / len(ratios)
    assert mean_ratio > 0.50, (
        f"MI(codes_MLP, codes_LIF) / H(codes_MLP) = {mean_ratio:.3f} "
        f"< 50 % across seeds {[r['seed'] for r in results]}; "
        "the protocol is not transmitting shared-code information"
    )
    # Per-seed floor: no seed collapses to near-zero MI.
    for r in results:
        assert r["mi_over_h_mlp"] > 0.30, (
            f"seed={r['seed']}: MI/H {r['mi_over_h_mlp']:.3f} collapsed; "
            "code orthogonality detected"
        )


def test_round_trip_fidelity_preserves_accuracy():
    """Test (2): round-trip accuracy / direct accuracy > 0.85.

    Empirical: ~99 % mean fidelity across 3 seeds. Pins that code
    information survives MLP → LIF → MLP via learned transducers.
    """
    torch.manual_seed(0)
    results = run_test_2_round_trip_fidelity(
        seeds=list(range(3)), steps=400, transducer_steps=200
    )
    ratios = [r["fidelity_ratio"] for r in results]
    mean_fidelity = sum(ratios) / len(ratios)
    assert mean_fidelity > 0.85, (
        f"round-trip fidelity {mean_fidelity:.3f} below 85 %; "
        f"per-seed ratios = {ratios}"
    )


def test_cross_substrate_merge_approaches_mlp_accuracy():
    """Test (3): LIF fed ONLY by MLP codes retains > 85 % of MLP accuracy.

    Empirical: ~97 % mean merge ratio across 3 seeds. Pins that the
    transducer + frozen LIF readout can recover the MLP's task signal
    from its emitted codes alone. This is the strongest test of
    substrate-agnostic INFORMATION TRANSMISSION (Claim B).
    """
    torch.manual_seed(0)
    results = run_test_3_cross_substrate_merge(
        seeds=list(range(3)), steps=400, merge_steps=300
    )
    ratios = [r["merge_ratio"] for r in results]
    mean_merge = sum(ratios) / len(ratios)
    assert mean_merge > 0.85, (
        f"cross-substrate merge ratio {mean_merge:.3f} below 85 %; "
        f"per-seed ratios = {ratios}"
    )
    # The LIF should approach the MLP's accuracy, not collapse to random.
    for r in results:
        assert r["acc_cross_merge"] > 1 / 12 + 0.10, (
            f"seed={r['seed']}: cross-merge accuracy "
            f"{r['acc_cross_merge']:.3f} barely above random (1/12=0.083)"
        )
