"""Stack-generic operator suites: ``VectorSpaceConformanceSuite``.

Each runs on the stack its ``case`` fixture supplies
(``TypedStackCase``); a stack's test directory subclasses it and
defines the fixture.
"""

import pytest

from pyapprox.util.linalg.operators import InnerProductSpaceProtocol, SpaceDescriptor
from tests._helpers.linalg.stack_case import A, F, TypedStackCase, V

_N = 6
_DESCRIPTOR = SpaceDescriptor("x", _N)


class VectorSpaceConformanceSuite:
    def test_satisfies_protocol(self, case: TypedStackCase[V, F, A]) -> None:
        space = case.space(_DESCRIPTOR)
        assert isinstance(space, InnerProductSpaceProtocol)
        assert space.descriptor() == _DESCRIPTOR

    def test_zeros(self, case: TypedStackCase[V, F, A]) -> None:
        space, bkd = case.space(_DESCRIPTOR), case.bkd
        x = space.zeros()
        assert space.ncols(x) == 1
        assert space.ncols(space.zeros(3)) == 3
        assert space.ncols(space.zeros_like(case.vec(_DESCRIPTOR, 4, 0))) == 4
        bkd.assert_allclose(space.norm(x), bkd.zeros((1, 1)))

    def test_reduction_shapes_and_values(self, case: TypedStackCase[V, F, A]) -> None:
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

    def test_axpby_without_out(self, case: TypedStackCase[V, F, A]) -> None:
        space = case.space(_DESCRIPTOR)
        x, y = case.vec(_DESCRIPTOR, 2, 3), case.vec(_DESCRIPTOR, 2, 4)
        y0 = case.array(y)
        result = space.axpby(2.0, x, -3.0, y)
        case.assert_vector(result, 2.0 * case.array(x) - 3.0 * y0, rtol=1e-12)
        case.assert_vector(y, y0)
        assert result is not y

    def test_axpby_into_accumulating_operand(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        """out may be y; callers use the returned value either way."""
        space = case.space(_DESCRIPTOR)
        x, y = case.vec(_DESCRIPTOR, 2, 5), case.vec(_DESCRIPTOR, 2, 6)
        expected = 0.5 * case.array(x) + 4.0 * case.array(y)
        y = space.axpby(0.5, x, 4.0, y, out=y)
        case.assert_vector(y, expected, rtol=1e-12)

    def test_axpby_into_separate_buffer(self, case: TypedStackCase[V, F, A]) -> None:
        space = case.space(_DESCRIPTOR)
        x, y = case.vec(_DESCRIPTOR, 1, 7), case.vec(_DESCRIPTOR, 1, 8)
        out = space.zeros()
        result = space.axpby(1.0, x, 1.0, y, out=out)
        case.assert_vector(result, case.array(x) + case.array(y), rtol=1e-12)
        assert (result is out) == case.in_place

    def test_scale_and_copy(self, case: TypedStackCase[V, F, A]) -> None:
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

    def test_column_and_concatenate(self, case: TypedStackCase[V, F, A]) -> None:
        """A column is a copied block of one; concatenating the columns
        gives the block back."""
        space = case.space(_DESCRIPTOR)
        x = case.vec(_DESCRIPTOR, 3, 17)
        xa = case.array(x)
        columns = [space.column(x, k) for k in range(3)]
        for k, column in enumerate(columns):
            assert space.ncols(column) == 1
            case.assert_vector(column, xa[:, k : k + 1])
        case.assert_vector(space.concatenate(columns), xa)
        # Writing into a column leaves x unchanged.
        space.scale(0.0, columns[0], out=columns[0])
        case.assert_vector(x, xa)
        case.assert_vector(
            space.concatenate([columns[2], x]),
            case.bkd.hstack([xa[:, 2:3], xa]),
        )
        for k in (3, -1):
            with pytest.raises(IndexError, match=f"column {k} is outside"):
                space.column(x, k)

    def test_aliasing_rules(self, case: TypedStackCase[V, F, A]) -> None:
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

    def test_rejects_wrong_shapes(self, case: TypedStackCase[V, F, A]) -> None:
        """A 1-D or wrong-length array is not a block of the space (the
        mock refuses anything that is not its own vector)."""
        space, bkd = case.space(_DESCRIPTOR), case.bkd
        with pytest.raises((ValueError, TypeError)):
            space.norm(bkd.zeros((_N,)))  # type: ignore[arg-type]
        with pytest.raises((ValueError, TypeError)):
            space.dot(bkd.zeros((_N + 1, 1)), case.vec(_DESCRIPTOR, 1, 14))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="same shape"):
            space.axpby(
                1.0, case.vec(_DESCRIPTOR, 1, 15), 1.0, case.vec(_DESCRIPTOR, 2, 16)
            )
