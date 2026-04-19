"""Loihi 2 compiler stub.

Documents the expected API surface for `lava-nc` integration without
taking it as a runtime dep. When hardware is available:

    pip install lava-nc
    from lava.lib.dl.netx import hdf5
    # replace the NotImplementedError body with a real compile call.

Plan 6 Task 9.
"""
from __future__ import annotations


class LoihiCompiler:
    """Placeholder for `lava-nc` Loihi 2 compilation."""

    @staticmethod
    def compile(artefact: dict) -> None:
        """Compile an exported artefact to a Loihi 2 executable.

        Expected signature once lava-nc is installed:
            graph = hdf5.Network(artefact)
            return LoihiCompiler.compile_graph(graph, board="Oheo Gulch")

        Currently raises so CI detects missing hardware integration.
        """
        raise NotImplementedError(
            "Loihi 2 compilation requires the `lava-nc` package. Install via "
            "`pip install lava-nc` and wire the call in neuromorphic.loihi_stub. "
            "See docs/neuromorphic/deployment-guide.md §Loihi."
        )
