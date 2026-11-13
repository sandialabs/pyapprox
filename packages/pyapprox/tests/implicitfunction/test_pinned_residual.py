r"""A residual with rows pinned to data, :math:`R = P F + (I - P)(u - g)`.

:math:`F_i(u; p) = (A u)_i + p_0 u_i^2 + p_1^2 c_i` on 3 states with
:math:`A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]`, :math:`c = (1, -2, 3)`,
at :math:`p = (0.7, -0.4)` and :math:`u = (0.3, -0.8, 1.1)`; the rows
:math:`\{0, 2\}` pinned to :math:`g = (p_0^2 + 1, \cdot, p_0 p_1)`:

- Value: the pinned rows read :math:`u_R - g_R`, the other row is
  :math:`F`'s.
- Every derivative passes finite differences (``DerivativeChecker``,
  central, relative, smallest error below 1e-6): the state and parameter
  Jacobians of :math:`R`, and the adjoint-weighted second derivatives as
  the Jacobians of :math:`R_u^\top \lambda` in :math:`u` and :math:`p` and
  of :math:`R_p^\top \lambda` in :math:`p`.
- Wrong-model control: the same data declared independent of the
  parameters give a parameter Jacobian whose error stays above 1e-2.
- Levels and zeros: a constant :math:`F_u` (``p_0 = 0`` and declared
  affine) stays constant with the stated-zero state curvature; data
  without second derivatives give a first-order bundle; a state-only
  :math:`F` gives a state-only bundle.
- Refusals: data whose parameter Jacobian is unavailable, a residual, pin
  or space of the wrong type, and rows of another space.
- The parameter version is :math:`F`'s.
"""

from typing import Callable

import pytest

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.pinned_rows import (
    PinnedValuesParamDerivatives,
    RowPin,
)
from pyapprox.implicitfunction.residual_derivatives import (
    DerivativeOperator,
    is_stated_zero,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.residual_derivatives import backend_spaces
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet
from pyapprox.util.linalg.operators import LinearizationPoint
from tests._helpers.implicitfunction_cases import (
    QUADRATIC_PARAM,
    QuadraticPinnedData,
    QuadraticResidual,
    pinned_quadratic,
)

_POINT = LinearizationPoint(0, ("test",))
_PARAM = QUADRATIC_PARAM
_STATE = [[0.3], [-0.8], [1.1]]
_ADJ = [[0.9], [-0.5], [0.6]]
_FD_TOL = 1e-6


def _min_error(
    bkd: Backend[Array],
    nqoi: int,
    nvars: int,
    fun: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
    sample: Array,
) -> float:
    errors = DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=nqoi, nvars=nvars, fun=fun, jacobian=jacobian, bkd=bkd
        )
    ).check_derivatives(sample, relative=True, central=True)[0]
    return bkd.to_float(bkd.min(errors))


def _dense(op: DerivativeOperator[Array], ncols: int, bkd: Backend[Array]) -> Array:
    return bkd.hstack([op.apply(bkd.eye(ncols)[:, j : j + 1]) for j in range(ncols)])


class TestTheValue:
    def test_pinned_rows_read_the_data(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd)
        state = bkd.asarray(_STATE)
        p0, p1 = _PARAM
        value = pinned(state)
        bkd.assert_allclose(value[0, 0], bkd.asarray(0.3 - (p0**2 + 1.0)))
        bkd.assert_allclose(value[2, 0], bkd.asarray(1.1 - p0 * p1))
        bkd.assert_allclose(value[1, 0], f(state)[1, 0])
        assert pinned.parameters_version() == 0
        f.set_param(bkd.asarray(_PARAM))
        assert pinned.parameters_version() == 1


class TestTheDerivatives:
    """Finite differences of the value and of the first derivatives."""

    def _at_param(
        self, f: QuadraticResidual[Array], fn: Callable[[], Array]
    ) -> Callable[[Array], Array]:
        def at(p: Array) -> Array:
            saved = f.p
            f.p = p
            try:
                return fn()
            finally:
                f.p = saved

        return at

    def test_the_state_jacobian(self, bkd: Backend[Array]) -> None:
        _, pinned = pinned_quadratic(bkd)
        jacobian = pinned.derivatives().jacobian
        assert jacobian is not None and not jacobian.affine()
        error = _min_error(
            bkd,
            3,
            3,
            pinned,
            lambda u: _dense(jacobian(u, _POINT), 3, bkd),
            bkd.asarray(_STATE),
        )
        assert error < _FD_TOL

    def test_the_parameter_jacobian(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd)
        param = pinned.derivatives().param
        assert param is not None
        state = bkd.asarray(_STATE)
        error = _min_error(
            bkd,
            3,
            2,
            self._at_param(f, lambda: pinned(state)),
            self._at_param(f, lambda: _dense(param(state), 2, bkd)),
            f.p,
        )
        assert error < _FD_TOL

    def test_the_second_derivatives(self, bkd: Backend[Array]) -> None:
        f, pinned = pinned_quadratic(bkd)
        bundle = pinned.derivatives()
        jacobian, param = bundle.jacobian, bundle.param
        state_state, state_param = bundle.state_state, bundle.state_param
        param_param = bundle.param_param
        assert jacobian is not None and param is not None
        assert state_state is not None and state_param is not None
        assert param_param is not None
        state, adj = bkd.asarray(_STATE), bkd.asarray(_ADJ)

        def ju(u: Array) -> Array:
            return jacobian(u, _POINT).apply_transpose(adj)

        def jp(u: Array) -> Array:
            return param(u).apply_transpose(adj)

        errors = [
            _min_error(
                bkd, 3, 3, ju, lambda u: _dense(state_state(u, adj), 3, bkd), state
            ),
            _min_error(
                bkd,
                3,
                2,
                self._at_param(f, lambda: ju(state)),
                self._at_param(f, lambda: _dense(state_param(state, adj), 2, bkd)),
                f.p,
            ),
            _min_error(
                bkd,
                2,
                2,
                self._at_param(f, lambda: jp(state)),
                self._at_param(f, lambda: _dense(param_param(state, adj), 2, bkd)),
                f.p,
            ),
        ]
        assert max(errors) < _FD_TOL

    def test_wrong_model_control(self, bkd: Backend[Array]) -> None:
        """Data declared independent of the parameters miss
        :math:`-(I - P) g_p`."""
        f, pinned = pinned_quadratic(
            bkd, bundle=PinnedValuesParamDerivatives.independent()
        )
        param = pinned.derivatives().param
        assert param is not None
        state = bkd.asarray(_STATE)
        error = _min_error(
            bkd,
            3,
            2,
            self._at_param(f, lambda: pinned(state)),
            self._at_param(f, lambda: _dense(param(state), 2, bkd)),
            f.p,
        )
        assert error > 1e-2


class TestLevelsAndZeros:
    def test_a_constant_jacobian_stays_constant(self, bkd: Backend[Array]) -> None:
        _, pinned = pinned_quadratic(bkd, affine=True)
        bundle = pinned.derivatives()
        assert bundle.jacobian is not None and bundle.jacobian.affine()
        assert is_stated_zero(bundle.state_state)
        assert not is_stated_zero(bundle.state_param)
        expected = bkd.asarray([[1.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 1.0]])
        bkd.assert_allclose(
            _dense(bundle.jacobian(bkd.asarray(_STATE), _POINT), 3, bkd), expected
        )

    def test_data_without_curvature_give_first_order(self, bkd: Backend[Array]) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        data = QuadraticPinnedData(f)
        first = PinnedValuesParamDerivatives.first_order(data.jacobian)
        pinned = PinnedResidual(f, RowPin(QuadraticPinnedData(f, first)), f.states)
        bundle = pinned.derivatives()
        assert bundle.param is not None and bundle.state_state is None

    def test_a_state_only_residual_stays_state_only(self, bkd: Backend[Array]) -> None:
        _, pinned = pinned_quadratic(bkd, order=0)
        bundle = pinned.derivatives()
        assert bundle.jacobian is not None and bundle.param is None


class TestRefusals:
    def test_unavailable_data_jacobian(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="does not declare the parameter"):
            pinned_quadratic(bkd, bundle=PinnedValuesParamDerivatives.unavailable())

    def test_wrong_types_and_spaces(self, bkd: Backend[Array]) -> None:
        f: QuadraticResidual[Array] = QuadraticResidual(bkd)
        pin = RowPin(QuadraticPinnedData(f))
        with pytest.raises(TypeError, match="ResidualProtocol"):
            PinnedResidual(3.0, pin, f.states)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="RowPin"):
            PinnedResidual(f, 3.0, f.states)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="VectorSpaceProtocol"):
            PinnedResidual(f, pin, 3.0)  # type: ignore[arg-type]
        other, _ = backend_spaces(bkd, 4, 2)
        rows = BackendRowSet(other.descriptor(), [0, 2], bkd)
        with pytest.raises(ValueError):
            PinnedResidual(f, RowPin(QuadraticPinnedData(f, rows=rows)), f.states)
