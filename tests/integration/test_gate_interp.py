"""Gate: interp-passed — end-to-end interpretation pipeline integration test.

Validates that the full pipeline (train → extract → cluster → render) completes
under timing budget, produces non-degenerate output, and maintains all invariants.
"""
import time
from pathlib import Path

import torch

from scripts.interpret_pilot import run_interp_pilot


def test_gate_interp_passes_all_criteria(tmp_path):
    """
    G1: runtime < 60 s
    G2: cluster entropy > 2 bits
    G3: all 64 codes present in HTML with data-code attribute
    G4: HTML file > 1 KB (non-trivial)
    G5: at least one code is active (mapped from inputs)
    """
    out = tmp_path / "report.html"
    torch.manual_seed(0)

    start = time.time()
    report = run_interp_pilot(output_path=str(out), steps=100, n_inputs=256)
    elapsed = time.time() - start

    # G1 runtime budget (loose — CI may be slower than laptop).
    assert elapsed < 60.0, f"interp pilot took {elapsed:.1f} s, expected < 60"

    # G2 cluster entropy > 2 bits (non-degenerate).
    assert report["entropy_bits"] > 2.0, (
        f"cluster entropy {report['entropy_bits']:.2f} bits < 2"
    )

    # G3 all 64 codes have a row in the HTML (with double quotes per visualise.py).
    text = Path(out).read_text()
    for c in range(64):
        assert f'data-code="{c}"' in text, (
            f"code {c} missing from HTML (expected data-code=\"{c}\")"
        )

    # G4 non-trivial file size.
    assert out.stat().st_size > 1_000, (
        f"HTML file {out.stat().st_size} bytes, expected > 1000"
    )

    # G5 at least one centroid is non-zero (some codes got mapped).
    assert report["n_active_codes"] > 0, (
        f"no active codes; n_active={report['n_active_codes']}"
    )
