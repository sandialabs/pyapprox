"""ExceptDofs: an essential condition with some DOFs left free.

Checks: the kept DOFs, values and time derivatives are the wrapped
condition's at the remaining positions, in its order; a missing
derivative stays missing; the wrapper is classified as essential and
pickles; refusals.
"""

import pickle
from typing import Callable, Generic, Optional

import pytest

from pyapprox.pde.boundary import EssentialBCProtocol, ExceptDofs, split_by_role
from pyapprox.util.backends.protocols import Array, Backend


class _Ramp(Generic[Array]):
    """u = (d + 1) t on DOFs d = 7, 3, 9, 4 (in that order); only the
    first time derivative is given."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._dofs = bkd.asarray([7, 3, 9, 4], dtype=bkd.index_dtype())
        self._rates = bkd.asarray([8.0, 4.0, 10.0, 5.0])

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def constrained_dofs(self) -> Array:
        return self._dofs

    def constrained_values(self, time: float) -> Array:
        return self._rates * time

    def is_time_invariant(self) -> bool:
        return False

    def rate(self, time: float) -> Array:
        return self._rates

    def constrained_values_derivative(
        self, order: int
    ) -> Optional[Callable[[float], Array]]:
        return self.rate if order == 1 else None

    def __repr__(self) -> str:
        return "_Ramp()"


def test_drops_the_exempt_dofs(bkd: Backend[Array]) -> None:
    bc = ExceptDofs(_Ramp(bkd), [9, 7])
    assert isinstance(bc, EssentialBCProtocol)
    assert not bc.is_time_invariant()
    kept = bkd.asarray([3, 4], dtype=bkd.index_dtype())
    bkd.assert_allclose(bc.constrained_dofs(), kept)
    bkd.assert_allclose(bc.constrained_values(2.0), bkd.asarray([8.0, 10.0]))
    rate = bc.constrained_values_derivative(1)
    assert rate is not None
    bkd.assert_allclose(rate(0.5), bkd.asarray([4.0, 5.0]))
    assert bc.constrained_values_derivative(2) is None


def test_no_exemption_is_the_condition(bkd: Backend[Array]) -> None:
    inner = _Ramp(bkd)
    bc = ExceptDofs(inner, [])
    bkd.assert_allclose(bc.constrained_dofs(), inner.constrained_dofs())
    bkd.assert_allclose(bc.constrained_values(1.5), inner.constrained_values(1.5))


def test_classified_as_essential_and_pickles(numpy_bkd: Backend[Array]) -> None:
    bc = ExceptDofs(_Ramp(numpy_bkd), [3])
    roles = split_by_role([bc])
    assert roles.essentials() == [bc] and roles.terms() == []
    copy = pickle.loads(pickle.dumps(bc))
    rate = copy.constrained_values_derivative(1)
    assert rate is not None
    numpy_bkd.assert_allclose(rate(0.0), numpy_bkd.asarray([8.0, 10.0, 5.0]))


def test_rejects_bad_input(bkd: Backend[Array]) -> None:
    with pytest.raises(ValueError, match=r"does not constrain DOFs \[5\]"):
        ExceptDofs(_Ramp(bkd), [3, 5])
    with pytest.raises(ValueError, match="repeated"):
        ExceptDofs(_Ramp(bkd), [3, 3])
    with pytest.raises(TypeError, match="EssentialBCProtocol"):
        ExceptDofs(object(), [3])  # type: ignore[arg-type]
