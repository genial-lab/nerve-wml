"""Upstream `kiki_oniric.axioms` version compat-check.

Pinned against dream-of-kiki formal axis `C-v0.8.0+PARTIAL` at
nerve-wml v1.8.0 release (2026-04-22). If the installed upstream
differs, emit an :class:`UpstreamAxiomsVersionWarning`; promote to
:class:`RuntimeError` in ``strict=True`` mode.

The check is a no-op when the `axioms` extras group is not installed
(i.e. `kiki_oniric` cannot be imported) - extras-opt-in semantics.
"""
from __future__ import annotations

import warnings

#: Upstream formal axis version pinned at nerve-wml v1.8.0 release.
#: Bump this constant (and the tests) whenever we re-verify against
#: a new upstream tag.
PINNED_UPSTREAM_VERSION: str = "C-v0.8.0+PARTIAL"


class UpstreamAxiomsVersionWarning(UserWarning):
    """Emitted when installed kiki_oniric.axioms version diverges from
    the pinned PINNED_UPSTREAM_VERSION at nerve-wml release time."""


def _read_upstream_version() -> str | None:
    """Import kiki_oniric.axioms and read its _CURRENT_VERSION constant.

    Returns None when the module is not installed (extras absent).
    This is the single point we can monkeypatch in tests.
    """
    try:
        from kiki_oniric import axioms as _up  # type: ignore[import-not-found]
    except ImportError:
        return None
    return getattr(_up, "_CURRENT_VERSION", None)


def check_upstream_axioms_version(*, strict: bool = False) -> None:
    """Verify the installed `kiki_oniric.axioms` version matches PINNED.

    Parameters
    ----------
    strict
        If True, raise :class:`RuntimeError` on any divergence. If
        False (default), emit an :class:`UpstreamAxiomsVersionWarning`.

    Behaviour
    ---------
    - Module absent (extras not installed) -> silent no-op.
    - Module present, version matches PINNED -> silent no-op.
    - Module present, version differs -> warn (or raise in strict mode).
    """
    installed = _read_upstream_version()
    if installed is None:
        return
    if installed == PINNED_UPSTREAM_VERSION:
        return
    msg = (
        f"kiki_oniric.axioms version mismatch: installed={installed!r}, "
        f"pinned={PINNED_UPSTREAM_VERSION!r}. "
        f"nerve-wml v1.8.0 was released against {PINNED_UPSTREAM_VERSION}; "
        f"behaviour with a different upstream is not guaranteed. "
        f"Pin `dreamofkiki @ git+...@v0.9.1` in your consuming "
        f"project or bump nerve-wml to a newer release."
    )
    if strict:
        raise RuntimeError(msg)
    warnings.warn(msg, UpstreamAxiomsVersionWarning, stacklevel=2)
