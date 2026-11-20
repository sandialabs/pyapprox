"""Torch backend methods compile into one graph under torch.compile.

Torch is optional, so the whole module is skipped without it.
"""

import pytest

# Skip the module without torch; the imports below are then ordinary ones.
pytest.importorskip("torch")

import torch

from pyapprox.util.backends.torch import TorchBkd


class TestBooleanReductionsCompile:
    """``any`` and ``all`` stay inside the graph (no host sync), so a
    function using them compiles with ``fullgraph=True``."""

    def test_any_and_all_compile_fullgraph(self, torch_bkd: TorchBkd) -> None:
        bkd = torch_bkd

        def masked(array: torch.Tensor) -> torch.Tensor:
            rows_ok = bkd.all(array > 0.0, axis=1, keepdims=True)
            some = bkd.any(array < -1.0)
            return bkd.where(rows_ok & ~some, array, bkd.zeros_like(array))

        compiled = torch.compile(masked, fullgraph=True)
        array = bkd.asarray([[1.0, 2.0], [-0.5, 3.0]])
        bkd.assert_allclose(compiled(array), masked(array))
        bkd.assert_allclose(compiled(array), bkd.asarray([[1.0, 2.0], [0.0, 0.0]]))
