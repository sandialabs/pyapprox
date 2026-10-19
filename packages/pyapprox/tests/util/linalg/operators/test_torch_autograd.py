"""The dense backend stack keeps a torch gradient graph intact.

Each test differentiates a scalar through one component with autograd
and compares with the analytic gradient. Torch is optional, so the whole
module is skipped without it.
"""

import numpy as np
import pytest

# Skip the module without torch; the imports below are then ordinary ones.
pytest.importorskip("torch")

import torch

from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.linalg.backend_block_diagonal import (
    BackendBlockDiagonalOperator,
    BlockDiagonalLU,
)
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
    BackendRowSet,
    BackendVectorSpace,
)
from pyapprox.util.linalg.operators import (
    RowReplacedOperator,
    ScaledSum,
    SpaceDescriptor,
)
from tests._helpers.linalg.stacks import random_matrix

_U = SpaceDescriptor("u", 5)
_P = SpaceDescriptor("p", 3)


def _random_pair(seed: int, nx: int, ny: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A leaf ``x`` that requires a gradient, and a constant ``y``."""
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(nx, 1)), requires_grad=True)
    y = torch.tensor(rng.normal(size=(ny, 1)))
    return x, y


class TestTorchAutograd:
    def test_vector_space(self, torch_bkd: TorchBkd) -> None:
        """d(||2x - y||^2)/dx = 4 (2x - y)."""
        space = BackendVectorSpace(_U, torch_bkd)
        x, y = _random_pair(17, 5, 5)
        z = space.axpby(2.0, x, -1.0, y, out=space.zeros())
        (grad,) = torch.autograd.grad(torch.sum(space.dot(z, z)), x)
        torch_bkd.assert_allclose(grad, 4.0 * z.detach(), rtol=1e-12)

    def test_dense_operator(self, torch_bkd: TorchBkd) -> None:
        """d(y^T A x)/dx = A^T y for a 3x5 A."""
        op = BackendDenseOperator(
            torch_bkd.asarray(random_matrix(3, 5, 23)), _U, torch_bkd, _P
        )
        x, y = _random_pair(24, 5, 3)
        (grad,) = torch.autograd.grad(torch.sum(y * op.apply(x)), x)
        torch_bkd.assert_allclose(grad, op.apply_transpose(y), rtol=1e-12)

    def test_scaled_sum(self, torch_bkd: TorchBkd) -> None:
        """d(y^T S x)/dx = S^T y, S = 2A - B."""
        bkd = torch_bkd
        a, b = random_matrix(5, 5, 26), random_matrix(5, 5, 27)
        op = ScaledSum(
            [
                (2.0, BackendDenseOperator(bkd.asarray(a), _U, bkd)),
                (-1.0, BackendDenseOperator(bkd.asarray(b), _U, bkd)),
            ],
            BackendVectorSpace(_U, bkd),
        )
        x, y = _random_pair(28, 5, 5)
        (grad,) = torch.autograd.grad(torch.sum(y * op.apply(x)), x)
        bkd.assert_allclose(grad, op.apply_transpose(y), rtol=1e-12)

    def test_row_replaced_operator(self, torch_bkd: TorchBkd) -> None:
        """d(y^T R x)/dx = R^T y, R = P A + (I - P) over rows {0, 3}."""
        bkd = torch_bkd
        dense: BackendDenseOperator[torch.Tensor] = BackendDenseOperator(
            bkd.asarray(random_matrix(5, 5, 21)), _U, bkd
        )
        rows: BackendRowSet[torch.Tensor] = BackendRowSet(_U, [0, 3], bkd)
        op = RowReplacedOperator(dense, rows)
        x, y = _random_pair(22, 5, 5)
        (grad,) = torch.autograd.grad(torch.sum(y * op.apply(x)), x)
        bkd.assert_allclose(grad, op.apply_transpose(y), rtol=1e-12)

    def test_dense_solve(self, torch_bkd: TorchBkd) -> None:
        """d(y^T A^{-1} b)/db = A^{-T} y, A shifted by 5I to stay
        well conditioned."""
        bkd = torch_bkd
        a = random_matrix(5, 5, 16) + 5.0 * np.eye(5)
        handle = BackendDenseLU(bkd).setup(
            BackendDenseOperator(bkd.asarray(a), _U, bkd)
        )
        b, y = _random_pair(17, 5, 5)
        (grad,) = torch.autograd.grad(torch.sum(y * handle.apply(b)), b)
        bkd.assert_allclose(grad, handle.apply_transpose(y), rtol=1e-10)

    def test_block_diagonal_solve(self, torch_bkd: TorchBkd) -> None:
        """The derivative of a batched block solve with respect to the
        blocks equals that of a dense solve with the block-diagonal
        matrix."""
        bkd = torch_bkd
        space = SpaceDescriptor("w", 6)
        rng = np.random.default_rng(18)
        blocks = bkd.asarray(rng.normal(size=(3, 2, 2)) + 3.0 * np.eye(2))
        b = bkd.asarray(rng.normal(size=(6, 1)))

        def block_solve(stored: torch.Tensor) -> torch.Tensor:
            op = BackendBlockDiagonalOperator(stored, space, bkd)
            return BlockDiagonalLU(bkd).setup(op).apply(b)[:, 0]

        def dense_solve(stored: torch.Tensor) -> torch.Tensor:
            # The block-diagonal matrix, entry ((i, a), (j, b)) = d_ij B_i[a, b].
            dense = bkd.reshape(bkd.einsum("ij,iab->iajb", bkd.eye(3), stored), (6, 6))
            return bkd.solve(dense, b)[:, 0]

        bkd.assert_allclose(
            bkd.jacobian(block_solve, blocks),
            bkd.jacobian(dense_solve, blocks),
            rtol=1e-12,
        )
