"""Conformance tests for the vector space of every stack.

Every test of the protocol runs on each stack of the shared ``case``
fixture: the NumPy stack (writes into ``out``), the backend space on
NumPy and Torch (ignores ``out``) and the opaque mock (writes into
``out``; read only through ``case.array``).
"""

import numpy as np
import pytest

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendVectorSpace
from pyapprox.util.linalg.operators import SpaceDescriptor
from pyapprox.util.linalg.scipy_stack import NumpyVectorSpace
from tests._helpers.linalg.operator_suites.vector_space import (
    VectorSpaceConformanceSuite,
)

_N = 6
_DESCRIPTOR = SpaceDescriptor("x", _N)


class TestVectorSpaceConformance(VectorSpaceConformanceSuite):
    pass


class TestNumpyVectorSpace:
    def test_writes_into_and_returns_out(self) -> None:
        space = NumpyVectorSpace(_DESCRIPTOR, np.float64)
        x, y = np.ones((_N, 1)), np.full((_N, 1), 2.0)
        assert space.axpby(1.0, x, 2.0, y, out=y) is y
        assert space.scale(3.0, x, out=x) is x
        NumpyBkd().assert_allclose(y, np.full((_N, 1), 5.0))
        NumpyBkd().assert_allclose(x, np.full((_N, 1), 3.0))

    def test_rejects_overlapping_out(self) -> None:
        """A view of a non-accumulating operand is an alias too."""
        space = NumpyVectorSpace(_DESCRIPTOR, np.float64)
        x = np.ones((_N, 2))
        with pytest.raises(ValueError, match="alias"):
            space.axpby(1.0, x[:, :1], 1.0, np.ones((_N, 1)), out=x[:, :1])
        with pytest.raises(ValueError, match="alias"):
            space.copy(x[:, 1:], out=x[:, 1:])

    def test_rejects_out_of_wrong_shape_or_dtype(self) -> None:
        space = NumpyVectorSpace(_DESCRIPTOR, np.float64)
        x = np.ones((_N, 2))
        with pytest.raises(ValueError, match="out must have shape"):
            space.scale(2.0, x, out=np.zeros((_N, 1)))
        with pytest.raises(ValueError, match="dtype"):
            space.norm(np.ones((_N, 1), dtype=np.float32))

    def test_dtype_follows_the_configured_scalar(self) -> None:
        space = NumpyVectorSpace(SpaceDescriptor("x", _N, dtype="float32"), np.float32)
        assert space.zeros(2).dtype == np.float32

    def test_refuses_a_space_of_another_scalar_type(self) -> None:
        with pytest.raises(ValueError, match="configured scalar type 'float64'"):
            NumpyVectorSpace(SpaceDescriptor("x", _N, dtype="float32"), np.float64)

    def test_complex_dot_conjugates_first_argument(self) -> None:
        bkd = NumpyBkd()
        space = NumpyVectorSpace(
            SpaceDescriptor("z", 2, dtype="complex128"), np.complex128
        )
        x = np.array([[1.0 + 1.0j], [2.0j]])
        bkd.assert_allclose(space.dot(x, x), np.array([[6.0 + 0.0j]]))
        # Norms come back in the space's scalar type: real-valued complex.
        norm = space.norm(x)
        assert norm.dtype == np.complex128
        bkd.assert_allclose(norm, np.array([[np.sqrt(6.0) + 0.0j]]))


class TestBackendVectorSpace:
    def test_rejects_layout_dtype_other_than_the_backends(
        self, bkd: Backend[Array]
    ) -> None:
        with pytest.raises(ValueError, match="configured dtype 'float64'"):
            BackendVectorSpace(SpaceDescriptor("x", _N, dtype="float32"), bkd)
