"""Conformance tests for the vector spaces of the NumPy and backend stacks.

Every test of the protocol runs on each stack: the NumPy stack (writes
into ``out``) and the backend space on NumPy and Torch (ignores ``out``).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Backend
from pyapprox.util.linalg.backend_dense import BackendVectorSpace
from pyapprox.util.linalg.operators import Layout, VectorSpaceProtocol
from pyapprox.util.linalg.scipy_stack import NumpyVectorSpace

_N = 6
_LAYOUT = Layout("x", _N)


@dataclass
class _Case:
    space: VectorSpaceProtocol[Any, Any]
    bkd: Backend[Any]
    in_place: bool

    def vec(self, ncols: int, seed: int) -> Any:
        rng = np.random.default_rng(seed)
        return self.bkd.asarray(rng.normal(size=(_N, ncols)))


@pytest.fixture(params=["numpy-stack", "backend-numpy", "backend-torch"])
def case(request: pytest.FixtureRequest) -> _Case:
    if request.param == "numpy-stack":
        return _Case(NumpyVectorSpace(_LAYOUT), NumpyBkd(), True)
    fixture = "numpy_bkd" if request.param == "backend-numpy" else "torch_bkd"
    bkd = request.getfixturevalue(fixture)
    return _Case(BackendVectorSpace(_LAYOUT, bkd), bkd, False)


class TestVectorSpaceConformance:
    def test_satisfies_protocol(self, case: _Case) -> None:
        assert isinstance(case.space, VectorSpaceProtocol)
        assert case.space.layout() == _LAYOUT

    def test_zeros(self, case: _Case) -> None:
        space, bkd = case.space, case.bkd
        x = space.zeros()
        assert space.ncols(x) == 1
        assert space.ncols(space.zeros(3)) == 3
        assert space.ncols(space.zeros_like(case.vec(4, 0))) == 4
        bkd.assert_allclose(space.norm(x), bkd.zeros((1, 1)))

    def test_reduction_shapes_and_values(self, case: _Case) -> None:
        """dot is (kx, ky) and norm is (1, k), replicated small arrays."""
        space, bkd = case.space, case.bkd
        x, y = case.vec(3, 1), case.vec(2, 2)
        dot = space.dot(x, y)
        assert tuple(dot.shape) == (3, 2)
        bkd.assert_allclose(dot, bkd.dot(x.T, y), rtol=1e-12)
        norm = space.norm(x)
        assert tuple(norm.shape) == (1, 3)
        bkd.assert_allclose(
            norm, bkd.sqrt(bkd.sum(x * x, axis=0, keepdims=True)), rtol=1e-12
        )
        assert tuple(space.dot(x[:, :1], y[:, :1]).shape) == (1, 1)
        assert tuple(space.norm(y[:, :1]).shape) == (1, 1)

    def test_axpby_without_out(self, case: _Case) -> None:
        space, bkd = case.space, case.bkd
        x, y = case.vec(2, 3), case.vec(2, 4)
        y0 = bkd.copy(y)
        result = space.axpby(2.0, x, -3.0, y)
        bkd.assert_allclose(result, 2.0 * x - 3.0 * y0, rtol=1e-12)
        bkd.assert_allclose(y, y0)
        assert result is not y

    def test_axpby_into_accumulating_operand(self, case: _Case) -> None:
        """out may be y; callers use the returned value either way."""
        space, bkd = case.space, case.bkd
        x, y = case.vec(2, 5), case.vec(2, 6)
        expected = 0.5 * x + 4.0 * y
        y = space.axpby(0.5, x, 4.0, y, out=y)
        bkd.assert_allclose(y, expected, rtol=1e-12)

    def test_axpby_into_separate_buffer(self, case: _Case) -> None:
        space, bkd = case.space, case.bkd
        x, y = case.vec(1, 7), case.vec(1, 8)
        out = space.zeros()
        result = space.axpby(1.0, x, 1.0, y, out=out)
        bkd.assert_allclose(result, x + y, rtol=1e-12)
        assert (result is out) == case.in_place

    def test_scale_and_copy(self, case: _Case) -> None:
        space, bkd = case.space, case.bkd
        x = case.vec(2, 9)
        expected = -2.0 * x
        out = space.zeros(2)
        copied = space.copy(x, out=out)
        bkd.assert_allclose(copied, x)
        assert (copied is out) == case.in_place
        copied = space.scale(-2.0, copied, out=copied)
        bkd.assert_allclose(copied, expected, rtol=1e-12)
        bkd.assert_allclose(space.copy(x), x)

    def test_aliasing_rules(self, case: _Case) -> None:
        """out may be the accumulating operand, never another operand."""
        space = case.space
        x, y = case.vec(1, 12), case.vec(1, 13)
        with pytest.raises(ValueError, match="alias"):
            space.axpby(1.0, x, 1.0, y, out=x)
        with pytest.raises(ValueError, match="alias"):
            space.axpby(1.0, y, 1.0, y, out=y)
        with pytest.raises(ValueError, match="alias"):
            space.copy(x, out=x)
        # x is y without out is a plain functional call.
        case.bkd.assert_allclose(space.axpby(1.0, y, 1.0, y), 2.0 * y)

    def test_rejects_wrong_shapes(self, case: _Case) -> None:
        space, bkd = case.space, case.bkd
        with pytest.raises(ValueError, match="ncols"):
            space.norm(bkd.zeros((_N,)))
        with pytest.raises(ValueError, match="ncols"):
            space.dot(bkd.zeros((_N + 1, 1)), case.vec(1, 14))
        with pytest.raises(ValueError, match="same shape"):
            space.axpby(1.0, case.vec(1, 15), 1.0, case.vec(2, 16))


class TestNumpyVectorSpace:
    def test_writes_into_and_returns_out(self) -> None:
        space = NumpyVectorSpace(_LAYOUT)
        x, y = np.ones((_N, 1)), np.full((_N, 1), 2.0)
        assert space.axpby(1.0, x, 2.0, y, out=y) is y
        assert space.scale(3.0, x, out=x) is x
        NumpyBkd().assert_allclose(y, np.full((_N, 1), 5.0))
        NumpyBkd().assert_allclose(x, np.full((_N, 1), 3.0))

    def test_rejects_overlapping_out(self) -> None:
        """A view of a non-accumulating operand is an alias too."""
        space = NumpyVectorSpace(_LAYOUT)
        x = np.ones((_N, 2))
        with pytest.raises(ValueError, match="alias"):
            space.axpby(1.0, x[:, :1], 1.0, np.ones((_N, 1)), out=x[:, :1])
        with pytest.raises(ValueError, match="alias"):
            space.copy(x[:, 1:], out=x[:, 1:])

    def test_rejects_out_of_wrong_shape_or_dtype(self) -> None:
        space = NumpyVectorSpace(_LAYOUT)
        x = np.ones((_N, 2))
        with pytest.raises(ValueError, match="out must have shape"):
            space.scale(2.0, x, out=np.zeros((_N, 1)))
        with pytest.raises(ValueError, match="dtype"):
            space.norm(np.ones((_N, 1), dtype=np.float32))

    def test_dtype_follows_layout(self) -> None:
        space = NumpyVectorSpace(Layout("x", _N, dtype="float32"))
        assert space.zeros(2).dtype == np.float32

    def test_complex_dot_conjugates_first_argument(self) -> None:
        bkd = NumpyBkd()
        space = NumpyVectorSpace(Layout("z", 2, dtype="complex128"))
        x = np.array([[1.0 + 1.0j], [2.0j]])
        bkd.assert_allclose(space.dot(x, x), np.array([[6.0 + 0.0j]]))
        bkd.assert_allclose(space.norm(x), np.array([[np.sqrt(6.0)]]))


class TestBackendVectorSpace:
    def test_rejects_non_float64_layout(self, bkd: Backend[Any]) -> None:
        with pytest.raises(ValueError, match="float64"):
            BackendVectorSpace(Layout("x", _N, dtype="float32"), bkd)

    def test_preserves_the_graph(self, torch_bkd: Backend[Any]) -> None:
        """d(||a x + b y||^2)/dx = 2 a (a x + b y) through autograd."""
        import torch

        space = BackendVectorSpace(_LAYOUT, torch_bkd)
        rng = np.random.default_rng(17)
        x = torch.tensor(rng.normal(size=(_N, 1)), requires_grad=True)
        y = torch.tensor(rng.normal(size=(_N, 1)))
        z = space.axpby(2.0, x, -1.0, y, out=space.zeros())
        value = torch.sum(space.dot(z, z))
        (grad,) = torch.autograd.grad(value, x)
        torch_bkd.assert_allclose(grad, 4.0 * z.detach(), rtol=1e-12)
