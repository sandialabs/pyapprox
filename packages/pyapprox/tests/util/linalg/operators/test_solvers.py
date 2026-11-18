"""Conformance tests for solvers and solver handles on every stack."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import (
    BackendDenseLU,
    BackendDenseOperator,
)
from pyapprox.util.linalg.operators import (
    SpaceDescriptor,
)
from pyapprox.util.linalg.scipy_stack import ScipyCSROperator, ScipySparseLU
from tests._helpers.linalg.operator_suites.solvers import (
    DiagonalSolverSuite,
    DirectSolverSuite,
)

_U = SpaceDescriptor("u", 5)


class TestDirectSolver(DirectSolverSuite):
    pass


class TestFormatPairingAtSetup:
    def test_sparse_lu_refuses_dense_storage(self, numpy_bkd: Backend[Array]) -> None:
        dense = BackendDenseOperator(numpy_bkd.asarray(np.eye(5)), _U, numpy_bkd)
        with pytest.raises(
            TypeError, match="needs an operator in CSRFormat, got one in DenseFormat"
        ):
            ScipySparseLU().setup(dense)  # type: ignore[arg-type]

    def test_dense_lu_refuses_csr_storage(self, numpy_bkd: Backend[Array]) -> None:
        sparse = ScipyCSROperator(csr_matrix(np.eye(5)), _U, scalar=np.float64)
        with pytest.raises(
            TypeError, match="needs an operator in DenseFormat, got one in CSRFormat"
        ):
            BackendDenseLU(numpy_bkd).setup(sparse)  # type: ignore[arg-type]


class TestDiagonalSolver(DiagonalSolverSuite):
    pass
