r"""Stack-generic suites: a residual with rows pinned to data,
:math:`R = P F + (I - P)(u - g)`.

The pinned quadratic of ``tests._helpers.implicitfunction_stack_cases``
at :math:`p = (0.7, -0.4)` and :math:`u = (0.3, -0.8, 1.1)`:

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
from pyapprox.implicitfunction.residual_derivatives import is_stated_zero
from pyapprox.util.linalg.operators import LinearizationPoint, SpaceDescriptor
from tests._helpers.implicitfunction_stack_cases import (
    QUADRATIC_PARAM,
    STATES,
    QuadraticPinnedData,
    QuadraticResidual,
    pinned_quadratic,
)
from tests._helpers.implicitfunction_suites.checks import (
    FD_TOL,
    min_fd_error,
    operator_matrix,
)
from tests._helpers.linalg.stack_case import A, F, TypedStackCase, V

_POINT = LinearizationPoint(0, ("test",))
_STATE = [[0.3], [-0.8], [1.1]]
_ADJ = [[0.9], [-0.5], [0.6]]


def _at_param(f: QuadraticResidual[V, F, A], fn: Callable[[], A]) -> Callable[[A], A]:
    """``fn`` evaluated with :math:`F`'s parameters set to a sample."""

    def at(p: A) -> A:
        saved = f.p
        f.p = p
        try:
            return fn()
        finally:
            f.p = saved

    return at


def _on_states(
    case: TypedStackCase[V, F, A], fn: Callable[[V], V]
) -> Callable[[A], A]:
    """``fn`` of a state, as a function of backend arrays."""

    def on(u: A) -> A:
        return case.array(fn(case.vector(u, STATES)))

    return on


class TheValueSuite:
    def test_pinned_rows_read_the_data(self, case: TypedStackCase[V, F, A]) -> None:
        f, pinned = pinned_quadratic(case)
        bkd = case.bkd
        state = case.vector(bkd.asarray(_STATE), STATES)
        p0, p1 = QUADRATIC_PARAM
        value = case.array(pinned(state))
        bkd.assert_allclose(value[0, 0], bkd.asarray(0.3 - (p0**2 + 1.0)))
        bkd.assert_allclose(value[2, 0], bkd.asarray(1.1 - p0 * p1))
        bkd.assert_allclose(value[1, 0], case.array(f(state))[1, 0])
        assert pinned.parameters_version() == 0
        f.set_param(bkd.asarray(QUADRATIC_PARAM))
        assert pinned.parameters_version() == 1


class TheDerivativesSuite:
    """Finite differences of the value and of the first derivatives."""

    def test_the_state_jacobian(self, case: TypedStackCase[V, F, A]) -> None:
        _, pinned = pinned_quadratic(case)
        jacobian = pinned.derivatives().jacobian
        assert jacobian is not None and not jacobian.affine()
        error = min_fd_error(
            case,
            3,
            3,
            _on_states(case, pinned),
            lambda u: operator_matrix(
                case, jacobian(case.vector(u, STATES), _POINT)
            ),
            case.bkd.asarray(_STATE),
        )
        assert error < FD_TOL

    def test_the_parameter_jacobian(self, case: TypedStackCase[V, F, A]) -> None:
        f, pinned = pinned_quadratic(case)
        param = pinned.derivatives().param
        assert param is not None
        state = case.vector(case.bkd.asarray(_STATE), STATES)
        error = min_fd_error(
            case,
            3,
            2,
            _at_param(f, lambda: case.array(pinned(state))),
            _at_param(f, lambda: operator_matrix(case, param(state))),
            f.p,
        )
        assert error < FD_TOL

    def test_the_second_derivatives(self, case: TypedStackCase[V, F, A]) -> None:
        f, pinned = pinned_quadratic(case)
        bundle = pinned.derivatives()
        jacobian, param = bundle.jacobian, bundle.param
        state_state, state_param = bundle.state_state, bundle.state_param
        param_param = bundle.param_param
        assert jacobian is not None and param is not None
        assert state_state is not None and state_param is not None
        assert param_param is not None
        bkd = case.bkd
        state = case.vector(bkd.asarray(_STATE), STATES)
        adj = case.vector(bkd.asarray(_ADJ), STATES)

        def ju(u: V) -> V:
            return jacobian(u, _POINT).apply_transpose(adj)

        def jp(u: V) -> V:
            return param(u).apply_transpose(adj)

        errors = [
            min_fd_error(
                case,
                3,
                3,
                _on_states(case, ju),
                lambda u: operator_matrix(
                    case, state_state(case.vector(u, STATES), adj)
                ),
                bkd.asarray(_STATE),
            ),
            min_fd_error(
                case,
                3,
                2,
                _at_param(f, lambda: case.array(ju(state))),
                _at_param(f, lambda: operator_matrix(case, state_param(state, adj))),
                f.p,
            ),
            min_fd_error(
                case,
                2,
                2,
                _at_param(f, lambda: case.array(jp(state))),
                _at_param(f, lambda: operator_matrix(case, param_param(state, adj))),
                f.p,
            ),
        ]
        assert max(errors) < FD_TOL

    def test_wrong_model_control(self, case: TypedStackCase[V, F, A]) -> None:
        """Data declared independent of the parameters miss
        :math:`-(I - P) g_p`."""
        f, pinned = pinned_quadratic(
            case, bundle=PinnedValuesParamDerivatives.independent()
        )
        param = pinned.derivatives().param
        assert param is not None
        state = case.vector(case.bkd.asarray(_STATE), STATES)
        error = min_fd_error(
            case,
            3,
            2,
            _at_param(f, lambda: case.array(pinned(state))),
            _at_param(f, lambda: operator_matrix(case, param(state))),
            f.p,
        )
        assert error > 1e-2


class LevelsAndZerosSuite:
    def test_a_constant_jacobian_stays_constant(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        _, pinned = pinned_quadratic(case, affine=True)
        bundle = pinned.derivatives()
        assert bundle.jacobian is not None and bundle.jacobian.affine()
        assert is_stated_zero(bundle.state_state)
        assert not is_stated_zero(bundle.state_param)
        bkd = case.bkd
        expected = bkd.asarray([[1.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 1.0]])
        state = case.vector(bkd.asarray(_STATE), STATES)
        bkd.assert_allclose(
            operator_matrix(case, bundle.jacobian(state, _POINT)), expected
        )

    def test_data_without_curvature_give_first_order(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        data = QuadraticPinnedData(f)
        first = PinnedValuesParamDerivatives.first_order(data.jacobian)
        pinned = PinnedResidual(f, RowPin(QuadraticPinnedData(f, first)), f.states)
        bundle = pinned.derivatives()
        assert bundle.param is not None and bundle.state_state is None

    def test_a_state_only_residual_stays_state_only(
        self, case: TypedStackCase[V, F, A]
    ) -> None:
        _, pinned = pinned_quadratic(case, order=0)
        bundle = pinned.derivatives()
        assert bundle.jacobian is not None and bundle.param is None


class RefusalsSuite:
    def test_unavailable_data_jacobian(self, case: TypedStackCase[V, F, A]) -> None:
        with pytest.raises(ValueError, match="does not declare the parameter"):
            pinned_quadratic(case, bundle=PinnedValuesParamDerivatives.unavailable())

    def test_wrong_types_and_spaces(self, case: TypedStackCase[V, F, A]) -> None:
        f: QuadraticResidual[V, F, A] = QuadraticResidual(case)
        pin = RowPin(QuadraticPinnedData(f))
        with pytest.raises(TypeError, match="ResidualProtocol"):
            PinnedResidual(3.0, pin, f.states)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="RowPin"):
            PinnedResidual(f, 3.0, f.states)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="VectorSpaceProtocol"):
            PinnedResidual(f, pin, 3.0)  # type: ignore[arg-type]
        rows = case.rows(SpaceDescriptor("state", 4), [0, 2])
        with pytest.raises(ValueError):
            PinnedResidual(f, RowPin(QuadraticPinnedData(f, rows=rows)), f.states)
