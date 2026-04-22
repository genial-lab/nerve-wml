"""Unit tests for the upstream axioms version compat-check."""
from __future__ import annotations

import warnings

import pytest

from nerve_core.axioms_compat import (
    PINNED_UPSTREAM_VERSION,
    UpstreamAxiomsVersionWarning,
    check_upstream_axioms_version,
)


def test_pinned_version_is_c_v0_8_0_partial():
    """Pinned version must match the upstream axiom registry at release."""
    assert PINNED_UPSTREAM_VERSION == "C-v0.8.0+PARTIAL"


def test_check_is_noop_when_module_absent():
    """Gracefully return when kiki_oniric is not installed — axioms
    extras is optional, and consumers without the extra must not see
    warnings or errors from the compat-check."""
    assert check_upstream_axioms_version(strict=False) is None


def test_check_warns_on_version_drift(monkeypatch):
    """If the installed version differs from PINNED, warn but do not raise."""
    import nerve_core.axioms_compat as mod

    def _fake_get_version():
        return "C-v99.0.0+DRIFT"

    monkeypatch.setattr(mod, "_read_upstream_version", _fake_get_version)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        check_upstream_axioms_version(strict=False)

    assert any(
        issubclass(w.category, UpstreamAxiomsVersionWarning)
        and "C-v99.0.0+DRIFT" in str(w.message)
        for w in recorded
    )


def test_check_raises_in_strict_mode_on_drift(monkeypatch):
    """strict=True promotes the drift warning to a RuntimeError."""
    import nerve_core.axioms_compat as mod

    monkeypatch.setattr(
        mod, "_read_upstream_version", lambda: "C-v99.0.0+DRIFT",
    )

    with pytest.raises(RuntimeError, match="C-v99.0.0\\+DRIFT"):
        check_upstream_axioms_version(strict=True)


def test_check_is_silent_on_exact_match(monkeypatch):
    """No warning when installed == pinned."""
    import nerve_core.axioms_compat as mod

    monkeypatch.setattr(
        mod, "_read_upstream_version", lambda: "C-v0.8.0+PARTIAL",
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        check_upstream_axioms_version(strict=False)

    assert not any(
        issubclass(w.category, UpstreamAxiomsVersionWarning)
        for w in recorded
    )
