from pathlib import Path

import pytest


def test_render_w4_forgetting_bars_creates_pdf(tmp_path):
    import torch
    torch.manual_seed(0)
    from scripts.render_paper_figures import render_w4_forgetting_bars

    out = tmp_path / "fig2.pdf"
    render_w4_forgetting_bars(output_path=str(out), n_seeds=3, steps=200)
    assert out.exists()
    assert out.stat().st_size > 1000  # at least 1 KB = real PDF


@pytest.mark.slow
def test_render_w4_forgetting_bars_matches_paper_location():
    """Rendering to the paper's figures directory produces a file
    where the LaTeX source expects it."""
    import torch
    torch.manual_seed(0)
    from scripts.render_paper_figures import render_w4_forgetting_bars

    out = "papers/paper1/figures/w4_forgetting.pdf"
    render_w4_forgetting_bars(output_path=out, n_seeds=3, steps=200)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 1000
