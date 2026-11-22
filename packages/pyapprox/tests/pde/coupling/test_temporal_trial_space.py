r"""The port data's space in time over a window, injected into
:class:`WindowPortRows`.

- :class:`StartContinuousLagrange` on fractions (1/2, 1) reproduces a
  quadratic in time and its rate exactly from the start value and the
  two nodal values, and its coefficients sum to one.
- :class:`WindowPortRows` reads the injected space: on three port rows of
  a five-row state, the quadratic trace and its rate at t = 0.1 + 0.3 H
  in a window of length H = 0.2 equal the polynomial's to 1e-13; a space
  that ignores the start (a jump at the window's start) gives the port
  data alone, whatever the committed trace.
- Refusals.
"""

from typing import Callable, Generic, Optional, Sequence

import numpy as np
import pytest

from pyapprox.ode.pinned_rows import PinnedRowsParamDerivatives
from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.pde.coupling.ports import BackendPortRows
from pyapprox.pde.coupling.responses import TimeWindow
from pyapprox.pde.coupling.transient import (
    StartContinuousLagrange,
    TemporalTrialSpaceProtocol,
    WindowPortRows,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet
from pyapprox.util.linalg.operators import RowSetProtocol

_PORTS = [1, 2, 4]
_START, _H = 0.1, 0.2


def _quadratic(s: float) -> list[float]:
    """Three port rows' values at fraction ``s``."""
    return [1.0 + 2.0 * s - 3.0 * s * s, -0.5 + s * s, 4.0 * s]


def _quadratic_rate(s: float) -> list[float]:
    """Their derivatives in ``s``."""
    return [2.0 - 6.0 * s, 2.0 * s, 4.0]


class _NoEssentials(Generic[Array]):
    """No essential rows: an empty declaration."""

    def __init__(self, bkd: Backend[Array]) -> None:
        states, _ = backend_spaces(bkd, 5, 1)
        self._states = states
        self._rows: BackendRowSet[Array] = BackendRowSet(states.descriptor(), [], bkd)

    def rows(self) -> RowSetProtocol[Array]:
        return self._rows

    def values(self, time: float) -> Array:
        return self._states.zeros(1)

    def rates(self) -> Optional[Callable[[float], Array]]:
        return self.values

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[Array]:
        return PinnedRowsParamDerivatives.independent()


class _Jump:
    """One unknown trace held over the window, the start ignored."""

    def nunknowns(self) -> int:
        return 1

    def coefficients(self, fraction: float) -> Sequence[float]:
        return (0.0, 1.0)

    def rate_coefficients(self, fraction: float) -> Sequence[float]:
        return (0.0, 0.0)


def _rows(
    bkd: Backend[Array], trial: TemporalTrialSpaceProtocol
) -> WindowPortRows[Array, Array]:
    states, _ = backend_spaces(bkd, 5, 1)
    descriptor = states.descriptor()
    return WindowPortRows(
        _NoEssentials(bkd),
        BackendRowSet(descriptor, _PORTS, bkd),
        BackendPortRows(descriptor, _PORTS, bkd),
        states,
        trial,
    )


def test_lagrange_reproduces_a_quadratic() -> None:
    space = StartContinuousLagrange((0.5, 1.0))
    assert isinstance(space, TemporalTrialSpaceProtocol)
    assert space.nunknowns() == 2 and space.fractions() == (0.5, 1.0)
    nodal = [_quadratic(s) for s in (0.0, 0.5, 1.0)]
    for s in (0.0, 0.2, 0.5, 0.9, 1.0):
        weights = space.coefficients(s)
        assert abs(sum(weights) - 1.0) <= 1e-14
        rates = space.rate_coefficients(s)
        for p in range(3):
            value = sum(w * nodal[k][p] for k, w in enumerate(weights))
            rate = sum(w * nodal[k][p] for k, w in enumerate(rates))
            assert abs(value - _quadratic(s)[p]) <= 1e-14
            assert abs(rate - _quadratic_rate(s)[p]) <= 1e-13


def test_window_port_rows_read_the_injected_space(bkd: Backend[Array]) -> None:
    rows = _rows(bkd, StartContinuousLagrange((0.5, 1.0)))
    assert rows.nport() == 2 * len(_PORTS)
    states, _ = backend_spaces(bkd, 5, 1)
    # Rows 1, 2 and 4 hold the quadratic at the start; rows 0 and 3 are
    # not port rows.
    q0 = _quadratic(0.0)
    start = bkd.asarray([[9.0], [q0[0]], [q0[1]], [9.0], [q0[2]]])
    # Interleaved, the unknown the fast index.
    data = bkd.asarray(
        [[_quadratic(s)[p]] for p in range(len(_PORTS)) for s in (0.5, 1.0)]
    )
    rows.set_window(TimeWindow(_START, _START + _H), start, data)
    time = _START + 0.3 * _H
    want = bkd.asarray([[v] for v in _quadratic(0.3)])
    want_rate = bkd.asarray([[v / _H] for v in _quadratic_rate(0.3)])
    ports = BackendPortRows(states.descriptor(), _PORTS, bkd)
    bkd.assert_allclose(ports.take(rows.values(time)), want, atol=1e-13)
    rate = rows.rates()
    assert rate is not None
    bkd.assert_allclose(ports.take(rate(time)), want_rate, atol=1e-12)


def test_a_space_may_ignore_the_start(bkd: Backend[Array]) -> None:
    rows = _rows(bkd, _Jump())
    states, _ = backend_spaces(bkd, 5, 1)
    ports = BackendPortRows(states.descriptor(), _PORTS, bkd)
    data = bkd.asarray([[1.0], [2.0], [3.0]])
    start = bkd.asarray(np.full((5, 1), 7.0))
    rows.set_window(TimeWindow(_START, _START + _H), start, data)
    for s in (0.0, 0.4, 1.0):
        bkd.assert_allclose(ports.take(rows.values(_START + s * _H)), data)


def test_refusals(bkd: Backend[Array]) -> None:
    for fractions in [(), (0.5,), (0.5, 0.5, 1.0), (0.0, 1.0), (1.0, 0.5)]:
        with pytest.raises(ValueError, match="fractions must increase"):
            StartContinuousLagrange(fractions)
    with pytest.raises(TypeError, match="TemporalTrialSpaceProtocol"):
        _rows(bkd, object())  # type: ignore[arg-type]
