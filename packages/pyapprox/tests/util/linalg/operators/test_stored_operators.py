"""Conformance tests for the stored operators of every stack.

Every protocol test runs on ScipyCSROperator (NumPy arrays), on
BackendDenseOperator under NumPy and Torch, and on the opaque mock
stack, reading vectors only through ``case.array`` and ``case.vector``.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator
from pyapprox.util.linalg.operators import (
    OperatorProperties,
    SpaceDescriptor,
)
from pyapprox.util.linalg.scipy_stack import ScipyCSROperator
from tests._helpers.linalg.stack_case import random_matrix as _random
from tests._helpers.linalg.operator_suites.stored_operators import (
    StoredOperatorConformanceSuite,
)

_U = SpaceDescriptor("u", 5)
_P = SpaceDescriptor("p", 3)


class TestStoredOperatorConformance(StoredOperatorConformanceSuite):
    pass


class TestOperatorProperties:
    def test_implications_are_enforced(self) -> None:
        with pytest.raises(ValueError, match="spd=True requires symmetric"):
            OperatorProperties(spd=True)
        with pytest.raises(ValueError, match="diagonal=True requires symmetric"):
            OperatorProperties(diagonal=True)
        with pytest.raises(ValueError, match="identity=True requires diagonal"):
            OperatorProperties(symmetric=True, spd=True, identity=True)
        with pytest.raises(ValueError, match="identity=True requires spd"):
            OperatorProperties(symmetric=True, diagonal=True, identity=True)


class TestScipyCSROperator:
    def test_rejects_dense_matrix(self) -> None:
        with pytest.raises(TypeError, match="scipy sparse"):
            ScipyCSROperator(np.eye(5), _U, scalar=np.float64)

    def test_rejects_dtype_disagreeing_with_layout(self) -> None:
        with pytest.raises(ValueError, match="dtype"):
            ScipyCSROperator(
                csr_matrix(np.eye(5, dtype=np.float32)), _U, scalar=np.float64
            )

    def test_format_exposes_the_stored_matrix(self) -> None:
        matrix = csr_matrix(_random(3, 5, 22))
        op = ScipyCSROperator(matrix, _U, _P, scalar=np.float64)
        NumpyBkd().assert_allclose(op.format().matrix().toarray(), matrix.toarray())


class TestBackendDenseOperator:
    def test_rejects_layout_dtype_other_than_the_backends(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(ValueError, match="configured dtype"):
            BackendDenseOperator(
                bkd.asarray(np.eye(5)), SpaceDescriptor("u", 5, dtype="float32"), bkd
            )
