"""Ensure nerve_core import triggers the axioms compat-check once."""
from __future__ import annotations

import importlib


def test_reloading_nerve_core_calls_compat_check(monkeypatch):
    """Reload of nerve_core re-executes its __init__.py body, which
    must invoke check_upstream_axioms_version() once."""
    import nerve_core
    import nerve_core.axioms_compat as compat

    calls: list[None] = []

    monkeypatch.setattr(
        compat,
        "check_upstream_axioms_version",
        lambda **_: calls.append(None),
    )

    importlib.reload(nerve_core)

    assert len(calls) >= 1
