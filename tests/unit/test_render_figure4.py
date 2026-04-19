from pathlib import Path

import pytest


def test_render_w2_histogram_creates_pdf(tmp_path):
    import torch
    torch.manual_seed(0)
    from scripts.render_paper_figures import render_w2_histogram

    out = tmp_path / "fig4.pdf"
    render_w2_histogram(output_path=str(out), n_seeds=3, steps=200)
    assert out.exists()
    assert out.stat().st_size > 1000


@pytest.mark.slow
def test_render_w2_histogram_at_paper_location():
    import torch
    torch.manual_seed(0)
    from scripts.render_paper_figures import render_w2_histogram

    out = "papers/paper1/figures/w2_histogram.pdf"
    render_w2_histogram(output_path=out, n_seeds=5, steps=300)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 1000
