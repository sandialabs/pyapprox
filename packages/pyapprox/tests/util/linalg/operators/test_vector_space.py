"""Conformance tests for the vector space of every stack.

Every test of the protocol runs on each stack of the shared ``case``
fixture: the NumPy stack (writes into ``out``), the backend space on
NumPy and Torch (ignores ``out``) and the opaque mock (writes into
``out``; read only through ``case.array``).
"""

from typing import Any

import numpy as np
import pytest

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.backend_dense import BackendVectorSpace
from pyapprox.util.linalg.operators import SpaceDescriptor, VectorSpaceProtocol
from pyapprox.util.linalg.scipy_stack import NumpyVectorSpace
from tests._helpers.operator_stacks import StackCase as _Case

_N = 6
_DESCRIPTOR = SpaceDescriptor("x", _N)


class TestVectorSpaceConformance:
    def test_satisfies_protocol(self, case: _Case) -> None:
        space = case.space(_DESCRIPTOR)
        assert isinstance(space, VectorSpaceProtocol)
        assert space.descriptor() == _DESCRIPTOR

    def test_zeros(self, case: _Case) -> None:
        space, bkd = case.space(_DESCRIPTOR), case.bkd
        x = space.zeros()
        assert space.ncols(x) == 1
        assert space.ncols(space.zeros(3)) == 3
        assert space.ncols(space.zeros_like(case.vec(_DESCRIPTOR, 4, 0))) == 4
        bkd.assert_allclose(space.norm(x), bkd.zeros((1, 1)))

    def test_reduction_shapes_and_values(self, case: _Case) -> None:
        """dot is (kx, ky) and norm is (1, k), replicated small arrays."""
        space, bkd = case.space(_DESCRIPTOR), case.bkd
        x, y = case.vec(_DESCRIPTOR, 3, 1), case.vec(_DESCRIPTOR, 2, 2)
        xa, ya = case.array(x), case.array(y)
        dot = space.dot(x, y)
        assert tuple(dot.shape) == (3, 2)
        bkd.assert_allclose(dot, bkd.dot(xa.T, ya), rtol=1e-12)
        norm = space.norm(x)
        assert tuple(norm.shape) == (1, 3)
        bkd.assert_allclose(
            norm, bkd.sqrt(bkd.sum(xa * xa, axis=0, keepdims=True)), rtol=1e-12
        )
        single = case.vec(_DESCRIPTOR, 1, 3)
        assert tuple(space.dot(single, single).shape) == (1, 1)
        assert tuple(space.norm(single).shape) == (1, 1)

    def test_axpby_without_out(self, case: _Case) -> None:
        space = case.space(_DESCRIPTOR)
        x, y = case.vec(_DESCRIPTOR, 2, 3), case.vec(_DESCRIPTOR, 2, 4)
        y0 = case.array(y)
        result = space.axpby(2.0, x, -3.0, y)
        case.assert_vector(result, 2.0 * case.array(x) - 3.0 * y0, rtol=1e-12)
        case.assert_vector(y, y0)
        assert result is not y

    def test_axpby_into_accumulating_operand(self, case: _Case) -> None:
        """out may be y; callers use the returned value either way."""
        space = case.space(_DESCRIPTOR)
        x, y = case.vec(_DESCRIPTOR, 2, 5), case.vec(_DESCRIPTOR, 2, 6)
        expected = 0.5 * case.array(x) + 4.0 * case.array(y)
        y = space.axpby(0.5, x, 4.0, y, out=y)
        case.assert_vector(y, expected, rtol=1e-12)

    def test_axpby_into_separate_buffer(self, case: _Case) -> None:
        space = case.space(_DESCRIPTOR)
        x, y = case.vec(_DESCRIPTOR, 1, 7), case.vec(_DESCRIPTOR, 1, 8)
        out = space.zeros()
        result = space.axpby(1.0, x, 1.0, y, out=out)
        case.assert_vector(result, case.array(x) + case.array(y), rtol=1e-12)
        assert (result is out) == case.in_place

    def test_scale_and_copy(self, case: _Case) -> None:
        space = case.space(_DESCRIPTOR)
        x = case.vec(_DESCRIPTOR, 2, 9)
        expected = -2.0 * case.array(x)
        out = space.zeros(2)
        copied = space.copy(x, out=out)
        case.assert_vector(copied, case.array(x))
        assert (copied is out) == case.in_place
        copied = space.scale(-2.0, copied, out=copied)
        case.assert_vector(copied, expected, rtol=1e-12)
        case.assert_vector(space.copy(x), case.array(x))

    def test_aliasing_rules(self, case: _Case) -> None:
        """out may be the accumulating operand, never another operand."""
        space = case.space(_DESCRIPTOR)
        x, y = case.vec(_DESCRIPTOR, 1, 12), case.vec(_DESCRIPTOR, 1, 13)
        with pytest.raises(ValueError, match="alias"):
            space.axpby(1.0, x, 1.0, y, out=x)
        with pytest.raises(ValueError, match="alias"):
            space.axpby(1.0, y, 1.0, y, out=y)
        with pytest.raises(ValueError, match="alias"):
            space.copy(x, out=x)
        # x is y without out is a plain functional call.
        case.assert_vector(space.axpby(1.0, y, 1.0, y), 2.0 * case.array(y))

    def test_rejects_wrong_shapes(self, case: _Case) -> None:
        """A 1-D or wrong-length array is not a block of the space (the
        mock refuses anything that is not its own vector)."""
        space, bkd = case.space(_DESCRIPTOR), case.bkd
        with pytest.raises((ValueError, TypeError)):
            space.norm(bkd.zeros((_N,)))
        with pytest.raises((ValueError, TypeError)):
            space.dot(bkd.zeros((_N + 1, 1)), case.vec(_DESCRIPTOR, 1, 14))
        with pytest.raises(ValueError, match="same shape"):
            space.axpby(
                1.0, case.vec(_DESCRIPTOR, 1, 15), 1.0, case.vec(_DESCRIPTOR, 2, 16)
            )


class TestNumpyVectorSpace:
    def test_writes_into_and_returns_out(self) -> None:
        space = NumpyVectorSpace(_DESCRIPTOR)
        x, y = np.ones((_N, 1)), np.full((_N, 1), 2.0)
        assert space.axpby(1.0, x, 2.0, y, out=y) is y
        assert space.scale(3.0, x, out=x) is x
        NumpyBkd().assert_allclose(y, np.full((_N, 1), 5.0))
        NumpyBkd().assert_allclose(x, np.full((_N, 1), 3.0))

    def test_rejects_overlapping_out(self) -> None:
        """A view of a non-accumulating operand is an alias too."""
        space = NumpyVectorSpace(_DESCRIPTOR)
        x = np.ones((_N, 2))
        with pytest.raises(ValueError, match="alias"):
            space.axpby(1.0, x[:, :1], 1.0, np.ones((_N, 1)), out=x[:, :1])
        with pytest.raises(ValueError, match="alias"):
            space.copy(x[:, 1:], out=x[:, 1:])

    def test_rejects_out_of_wrong_shape_or_dtype(self) -> None:
        space = NumpyVectorSpace(_DESCRIPTOR)
        x = np.ones((_N, 2))
        with pytest.raises(ValueError, match="out must have shape"):
            space.scale(2.0, x, out=np.zeros((_N, 1)))
        with pytest.raises(ValueError, match="dtype"):
            space.norm(np.ones((_N, 1), dtype=np.float32))

    def test_dtype_follows_layout(self) -> None:
        space = NumpyVectorSpace(SpaceDescriptor("x", _N, dtype="float32"))
        assert space.zeros(2).dtype == np.float32

    def test_complex_dot_conjugates_first_argument(self) -> None:
        bkd = NumpyBkd()
        space = NumpyVectorSpace(SpaceDescriptor("z", 2, dtype="complex128"))
        x = np.array([[1.0 + 1.0j], [2.0j]])
        bkd.assert_allclose(space.dot(x, x), np.array([[6.0 + 0.0j]]))
        bkd.assert_allclose(space.norm(x), np.array([[np.sqrt(6.0)]]))


class TestBackendVectorSpace:
    def test_rejects_layout_dtype_other_than_the_backends(
        self, bkd: Backend[Any]
    ) -> None:
        with pytest.raises(ValueError, match="configured dtype 'float64'"):
            BackendVectorSpace(SpaceDescriptor("x", _N, dtype="float32"), bkd)

    def test_preserves_the_graph(self, torch_bkd: Backend[Any]) -> None:
        """d(||a x + b y||^2)/dx = 2 a (a x + b y) through autograd."""
        import torch

        space = BackendVectorSpace(_DESCRIPTOR, torch_bkd)
        rng = np.random.default_rng(17)
        x = torch.tensor(rng.normal(size=(_N, 1)), requires_grad=True)
        y = torch.tensor(rng.normal(size=(_N, 1)))
        z = space.axpby(2.0, x, -1.0, y, out=space.zeros())
        value = torch.sum(space.dot(z, z))
        (grad,) = torch.autograd.grad(value, x)
        torch_bkd.assert_allclose(grad, 4.0 * z.detach(), rtol=1e-12)
