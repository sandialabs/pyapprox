"""Torch backend methods keep a gradient graph intact.

Torch is optional, so the whole module is skipped without it.
"""

import pytest

# Skip the module without torch; the imports below are then ordinary ones.
pytest.importorskip("torch")

import torch

from pyapprox.util.backends.torch import TorchBkd


class TestIndexAddAutograd:
    """Gradients flow through ``index_add`` to both the array added into
    and the added values, repeated indices included."""

    def test_reaches_array_and_values(self, torch_bkd: TorchBkd) -> None:
        bkd = torch_bkd
        array = bkd.asarray([1.0, 2.0, 3.0]).requires_grad_()
        values = bkd.asarray([4.0, 5.0]).requires_grad_()
        index = bkd.asarray([2, 2], dtype=bkd.int64_dtype())
        weights = bkd.asarray([1.0, 2.0, 3.0])
        array_grad, values_grad = torch.autograd.grad(
            bkd.sum(bkd.index_add(array, index, values) * weights), (array, values)
        )
        bkd.assert_allclose(array_grad, weights)
        bkd.assert_allclose(values_grad, bkd.asarray([3.0, 3.0]))
