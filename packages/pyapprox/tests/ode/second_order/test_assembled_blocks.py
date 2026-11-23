"""Tests for the assembled second-order state storage on two stacks.

The storage is stack-generic; these run it on the dense backend stack
(both backends) and on the SciPy CSR stack, comparing every assembled
operator against ``np.block`` and every solve against a dense one.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.ode.second_order.assembled_blocks import (
    AssembledStateBlocks,
    backend_dense_state_blocks,
)
from pyapprox.ode.second_order.scipy_blocks import scipy_csr_state_blocks
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseOperator,
    BackendRowPartition,
    DenseFormat,
)
from pyapprox.util.linalg.operators import OperatorProperties, SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipyCSROperator

_N = 4


def _matrices() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(0)
    stiffness = rng.normal(size=(_N, _N))
    damping = rng.normal(size=(_N, _N)) + 4.0 * np.eye(_N)
    return -(stiffness @ stiffness.T + np.eye(_N)), -damping


def _expected(
    lower_left: NDArray[np.float64], lower_right: NDArray[np.float64]
) -> NDArray[np.float64]:
    expected: NDArray[np.float64] = np.block(
        [[np.zeros((_N, _N)), np.eye(_N)], [lower_left, lower_right]]
    )
    return expected


class TestDense:
    def _storage(
        self, bkd: Backend[Array]
    ) -> AssembledStateBlocks[Array, Array, DenseFormat[Array]]:
        return backend_dense_state_blocks(bkd, _N)

    def test_split_and_join_round_trip(self, bkd: Backend[Array]) -> None:
        storage = self._storage(bkd)
        state = bkd.asarray(np.arange(2.0 * _N).reshape(2 * _N, 1))
        bkd.assert_allclose(storage.displacement(state), state[:_N], rtol=0.0)
        bkd.assert_allclose(storage.velocity(state), state[_N:], rtol=0.0)
        joined = storage.join(storage.displacement(state), storage.velocity(state))
        bkd.assert_allclose(joined, state, rtol=0.0)

    def test_assembled_operator_and_solve(self, bkd: Backend[Array]) -> None:
        """The companion operator [0, I; K, C] in the stack's format and
        solved by the stack's solver, a zero block of the right shape."""
        storage = self._storage(bkd)
        lower_left, lower_right = _matrices()
        half = storage.half_space().descriptor()
        op = storage.assemble(
            (
                (None, storage.identity()),
                (
                    BackendDenseOperator(bkd.asarray(lower_left), half, bkd),
                    BackendDenseOperator(bkd.asarray(lower_right), half, bkd),
                ),
            )
        )
        expected = bkd.asarray(_expected(lower_left, lower_right))
        bkd.assert_allclose(op.format().matrix(), expected, rtol=1e-14)
        x = bkd.asarray(np.random.default_rng(1).normal(size=(2 * _N, 2)))
        bkd.assert_allclose(op.apply(x), bkd.dot(expected, x), rtol=1e-13)
        bkd.assert_allclose(op.apply_transpose(x), bkd.dot(expected.T, x), rtol=1e-13)
        solution, info = storage.solver().setup(op).solve(op.apply(x))
        assert info.converged
        bkd.assert_allclose(solution, x, rtol=1e-10)

    def test_block_mass_keeps_the_mass_properties(self, bkd: Backend[Array]) -> None:
        storage = self._storage(bkd)
        half = storage.half_space().descriptor()
        mass = BackendDenseOperator(
            bkd.asarray(np.diag(np.arange(1.0, _N + 1.0))),
            half,
            bkd,
            properties=OperatorProperties(symmetric=True, spd=True, diagonal=True),
        )
        op = storage.assemble(((storage.identity(), None), (None, mass)))
        assert op.properties() == OperatorProperties(
            symmetric=True, spd=True, diagonal=True
        )

    def test_refusals(self, bkd: Backend[Array]) -> None:
        storage = self._storage(bkd)
        half = backend_state_space(bkd, _N)
        space = backend_state_space(bkd, 2 * _N)
        identity = storage.identity()
        quarter = backend_state_space(bkd, _N // 2)
        with pytest.raises(ValueError, match="two parts"):
            AssembledStateBlocks(
                half,
                space,
                BackendRowPartition(space, [quarter] * 4, bkd),
                identity,
                storage.solver(),
            )
        with pytest.raises(TypeError, match="RowPartitionProtocol"):
            AssembledStateBlocks(
                half,
                space,
                None,  # type: ignore[arg-type]
                identity,
                storage.solver(),
            )
        with pytest.raises(ValueError, match="cover"):
            BackendRowPartition(space, [half], bkd)


class TestCSR:
    def test_assembled_operator_and_solve(self) -> None:
        """The same companion operator, stored as CSR and solved by a
        sparse LU."""
        half = SpaceDescriptor("u", _N, dtype="float64")
        storage = scipy_csr_state_blocks(half, SpaceDescriptor("w", 2 * _N, "float64"))
        lower_left, lower_right = _matrices()
        op = storage.assemble(
            (
                (None, storage.identity()),
                (
                    ScipyCSROperator(csr_matrix(lower_left), half, scalar=np.float64),
                    ScipyCSROperator(csr_matrix(lower_right), half, scalar=np.float64),
                ),
            )
        )
        fmt = op.format()
        assert isinstance(fmt, CSRFormat)
        expected = _expected(lower_left, lower_right)
        np.testing.assert_allclose(fmt.matrix().toarray(), expected, rtol=1e-14)
        x = np.random.default_rng(1).normal(size=(2 * _N, 2))
        np.testing.assert_allclose(op.apply(x), expected @ x, rtol=1e-13)
        solution, info = storage.solver().setup(op).solve(op.apply(x))
        assert info.converged
        np.testing.assert_allclose(solution, x, rtol=1e-10)
        joined = storage.join(storage.displacement(x), storage.velocity(x))
        np.testing.assert_array_equal(joined, x)
