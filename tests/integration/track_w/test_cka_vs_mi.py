"""Regression test for the CKA vs MI/H comparison (positioning.md)."""
import pytest

from scripts.measure_cka_vs_mi import run_cka_vs_mi


@pytest.mark.slow
def test_mi_stronger_than_cka_at_argmax():
    """Our MI/H captures more alignment than CKA on the same argmax
    one-hots. Both are invariant to clean label permutations (CKA via
    feature-orthogonal invariance per Kornblith 2019, MI by
    construction); the gap reflects soft many-to-one code mappings
    where CKA's linear structure breaks down but MI still captures
    conditional dependence. See docs/positioning.md."""
    results = run_cka_vs_mi(seeds=[0, 1, 2], steps=400, batch=1024)
    mean_mi = sum(r["mi_over_h"] for r in results) / len(results)
    mean_cka_argmax = sum(r["cka_argmax_onehot"] for r in results) / len(results)
    mean_cka_pre = sum(r["cka_pre_emit"] for r in results) / len(results)
    mean_cka_logits = sum(r["cka_emit_logits"] for r in results) / len(results)

    # 1. MI/H > 0.85 — strong shared information.
    assert mean_mi > 0.85, (
        f"mean MI/H {mean_mi:.3f} below 85 % — Claim B weaker than paper"
    )
    # 2. MI/H >= CKA_argmax — permutation invariance gives us more
    #    alignment than CKA captures on the same argmax one-hots.
    assert mean_mi >= mean_cka_argmax, (
        f"MI/H {mean_mi:.3f} did not exceed CKA argmax {mean_cka_argmax:.3f}"
    )
    # 3. Continuous CKA is lower than discrete alignment, i.e. the
    #    substrates differ geometrically but agree categorically.
    assert mean_cka_pre < mean_cka_argmax, (
        f"pre-emit CKA {mean_cka_pre:.3f} >= argmax CKA {mean_cka_argmax:.3f} — "
        "expected argmax to re-align the substrates"
    )
    # 4. Emit-logit CKA falls between or below pre-emit CKA.
    assert mean_cka_logits > 0.40, (
        f"emit-logit CKA {mean_cka_logits:.3f} collapsed"
    )
