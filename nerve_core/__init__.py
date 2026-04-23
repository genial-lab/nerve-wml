# nerve_core — shared contracts for nerve-wml.

from nerve_core.axioms_compat import (
    check_upstream_axioms_version as _check_upstream,
)

# Invoked once at package import; silent unless the axioms extra is
# installed AND the upstream version has drifted from the pinned one.
_check_upstream(strict=False)
