"""KD match-compute ablation — is cross-merge just KD in disguise?

Honest finding: at matched compute on HardFlowProxyTask, cross-merge
and KD-through-transducer are statistically indistinguishable. Our
contribution vs KD is therefore **methodological (isolating protocol
channel capacity from student learning capacity)**, not a task-
accuracy improvement.

See docs/positioning.md §"KD match-compute ablation" for interpretation.
"""
import pytest

from scripts.measure_kd_ablation import run_kd_ablation


@pytest.mark.slow
def test_cross_merge_is_comparable_to_kd_at_matched_compute():
    """Pin the honest finding: cross-merge is within ~5 pp of KD-
    through-transducer on mean accuracy. Not strictly better, not
    strictly worse — methodologically distinct, not empirically
    superior on this task."""
    r = run_kd_ablation(seeds=[0, 1, 2], teacher_steps=400, transfer_steps=300)
    # Cross-merge reaches within reasonable fraction of the teacher.
    assert r["mean_acc_cross"] > 0.40, (
        f"cross-merge collapsed to {r['mean_acc_cross']:.3f}"
    )
    # Cross-merge and KD-through-transducer are comparable (within 5 pp).
    assert abs(r["mean_gap"]) < 0.05, (
        f"cross-merge vs KD-through-transducer gap "
        f"{r['mean_gap']*100:+.2f}% exceeds 5 pp — "
        "surprising result, revisit the ablation"
    )
    # Vanilla KD is the upper reference (student can train end-to-end).
    assert r["mean_acc_kd_vanilla"] >= r["mean_acc_cross"] - 0.10, (
        f"vanilla KD {r['mean_acc_kd_vanilla']:.3f} much worse than "
        f"cross-merge {r['mean_acc_cross']:.3f} — ordering unexpected"
    )
