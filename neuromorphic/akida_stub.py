"""BrainChip Akida compiler stub.

Documents the expected API surface for Akida SDK integration without
taking it as a runtime dep.

    pip install akida
    from akida import Model
    # replace the NotImplementedError body with a real Model.from_* call.

Plan 6 Task 9.
"""
from __future__ import annotations


class AkidaCompiler:
    """Placeholder for Akida SDK compilation."""

    @staticmethod
    def compile(artefact: dict) -> None:
        """Compile an exported artefact to an Akida model.

        Expected signature once `akida` is installed:
            model = Model.from_dict(artefact)
            return model.compile(input_shape=(artefact["n_neurons"],))

        Currently raises so CI detects missing SDK.
        """
        raise NotImplementedError(
            "Akida compilation requires the `akida` package. Install via "
            "`pip install akida` and wire the call in neuromorphic.akida_stub. "
            "See docs/neuromorphic/deployment-guide.md §Akida."
        )
