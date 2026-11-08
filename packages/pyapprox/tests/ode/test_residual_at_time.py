r"""An ODE right-hand side's bundle at one time: a time-free residual's.

:math:`f(y, t; p)` is a family of residuals indexed by :math:`t`;
``ODEResidualDerivatives.at_time(t)`` gives the bundle of one member.

- Lotka-Volterra (parameters (1, 0.1, 0.075, 1.5), state (10, 5)), a
  general Jacobian and every second derivative, its parameter curvature a
  stated zero: at :math:`t = 0.3` each field's action equals the ODE
  field's at that time, the Jacobian stays general, the stated zero stays
  stated, and the initial-state fields are left out.
- A Jacobian that sees only the time, :math:`(1 + t) A`, is constant at a
  fixed time and equals the ODE Jacobian there; a constant one stays
  constant; an affine second-order bundle keeps its stated-zero state
  curvature.
- A state-only bundle stays state-only, and no derivatives stay none.
- ``RightHandSideAtTime`` of the same Lotka-Volterra at :math:`t = 0.3`
  satisfies ``ResidualProtocol``; its value and every field, given
  single columns, equal the ODE's given the 1-D arrays; its version
  follows ``set_param``; a right-hand side without a version, or a
  non-form, is refused.
"""

from dataclasses import fields
from typing import Generic

import pytest

from pyapprox.implicitfunction.residual import ResidualProtocol
from pyapprox.implicitfunction.residual_derivatives import (
    ResidualDerivatives,
    is_stated_zero,
)
from pyapprox.ode.residual_at_time import RightHandSideAtTime
from pyapprox.ode.residual_derivatives import (
    STATED_ZERO,
    ODEResidualDerivatives,
    backend_spaces,
    constant_jacobian,
    dense_operator,
    state_free_jacobian,
)
from pyapprox.ode.residual_form import BackendColumnForm
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseOperator, DenseFormat
from pyapprox.util.linalg.operators import LinearizationPoint
from tests._helpers.ode.runge_kutta_cases import LV_PARAM, lotka_volterra_with_mass

_TIME = 0.3
_POINT = LinearizationPoint(0, ("test",))


class TestAGeneralBundle:
    def test_each_field_is_the_odes_at_that_time(self, bkd: Backend[Array]) -> None:
        model = lotka_volterra_with_mass(bkd)
        model.set_param(bkd.asarray(LV_PARAM))
        ode = model.derivatives()
        core = ode.at_time(_TIME)
        assert isinstance(core, ResidualDerivatives)
        assert ode.jacobian is not None and core.jacobian is not None
        assert not core.jacobian.affine()
        assert core.param_space is ode.param_space
        state, adj = bkd.asarray([10.0, 5.0]), bkd.asarray([0.9, -0.4])
        direction = bkd.asarray([[0.7], [-1.3]])
        pdir = bkd.asarray([[0.3], [-0.2], [0.5], [0.1]])
        bkd.assert_allclose(
            core.jacobian(state, _POINT).apply(direction),
            ode.jacobian(state, _TIME, _POINT).apply(direction),
        )
        assert core.jacobian(state, _POINT).linearization() == _POINT
        assert core.param is not None and ode.param is not None
        bkd.assert_allclose(
            core.param(state).apply(pdir), ode.param(state, _TIME).apply(pdir)
        )
        assert core.state_state is not None and ode.state_state is not None
        assert core.state_param is not None and ode.state_param is not None
        bkd.assert_allclose(
            core.state_state(state, adj).apply(direction),
            ode.state_state(state, _TIME, adj).apply(direction),
        )
        bkd.assert_allclose(
            core.state_param(state, adj).apply(pdir),
            ode.state_param(state, _TIME, adj).apply(pdir),
        )
        assert is_stated_zero(core.param_param)
        # The residual bundle has no initial-state fields at all.
        assert "initial_state" not in {f.name for f in fields(core)}


class _TimeScaled(Generic[Array]):
    """:math:`(1 + t) A`, a Jacobian that sees only the time."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd
        self._states, _ = backend_spaces(bkd, 2, 3)
        self.matrix = bkd.asarray([[1.0, 2.0], [3.0, 4.0]])

    def __call__(self, time: float) -> BackendDenseOperator[Array]:
        return BackendDenseOperator(
            (1.0 + time) * self.matrix, self._states.descriptor(), self._bkd
        )


class TestAffineBundles:
    def test_a_state_free_jacobian_is_constant_at_a_fixed_time(
        self, bkd: Backend[Array]
    ) -> None:
        scaled: _TimeScaled[Array] = _TimeScaled(bkd)
        ode: ODEResidualDerivatives[Array, DenseFormat[Array]] = (
            ODEResidualDerivatives.state_only(state_free_jacobian(scaled))
        )
        core = ode.at_time(_TIME)
        assert core.jacobian is not None and core.jacobian.affine()
        direction = bkd.asarray([[1.0], [1.0]])
        state = bkd.asarray([10.0, 5.0])
        bkd.assert_allclose(
            core.jacobian(state, _POINT).apply(direction),
            bkd.asarray([[3.0 * 1.3], [7.0 * 1.3]]),
        )
        assert core.param is None and core.state_state is None

    def test_an_affine_second_order_bundle_keeps_its_zeros(
        self, bkd: Backend[Array]
    ) -> None:
        scaled: _TimeScaled[Array] = _TimeScaled(bkd)
        states, params = backend_spaces(bkd, 2, 3)
        ones = bkd.ones((2, 3))

        def param_matrix(state: Array, time: float) -> Array:
            return ones

        ode = ODEResidualDerivatives.second_order(
            states,
            params,
            jacobian=constant_jacobian(scaled(0.0)),
            param=dense_operator(param_matrix, params, states, bkd),
            initial_state=STATED_ZERO,
            initial_state_curvature=STATED_ZERO,
            state_state=STATED_ZERO,
            state_param=STATED_ZERO,
            param_param=STATED_ZERO,
        )
        core = ode.at_time(_TIME)
        assert core.jacobian is not None and core.jacobian.affine()
        assert is_stated_zero(core.state_state)
        assert is_stated_zero(core.state_param)
        assert is_stated_zero(core.param_param)
        assert core.param is not None
        bkd.assert_allclose(
            core.param(bkd.zeros((2,))).apply(bkd.ones((3, 1))),
            bkd.asarray([[3.0], [3.0]]),
        )

    def test_no_derivatives_stay_none(self, bkd: Backend[Array]) -> None:
        assert ODEResidualDerivatives.none().at_time(_TIME).jacobian is None


class TestTheRightHandSideAtTime:
    """The member at t = 0.3 as a time-free residual of single columns."""

    def test_value_and_fields_take_columns(self, bkd: Backend[Array]) -> None:
        model = lotka_volterra_with_mass(bkd)
        model.set_param(bkd.asarray(LV_PARAM))
        residual = RightHandSideAtTime(model, _TIME, BackendColumnForm(bkd))
        assert isinstance(residual, ResidualProtocol)
        assert residual.time() == _TIME
        flat, adj_flat = bkd.asarray([10.0, 5.0]), bkd.asarray([0.9, -0.4])
        state, adj = bkd.reshape(flat, (2, 1)), bkd.reshape(adj_flat, (2, 1))
        bkd.assert_allclose(residual(state)[:, 0], model(flat, _TIME))
        ode, core = model.derivatives(), residual.derivatives()
        direction = bkd.asarray([[0.7], [-1.3]])
        pdir = bkd.asarray([[0.3], [-0.2], [0.5], [0.1]])
        assert ode.jacobian is not None and core.jacobian is not None
        bkd.assert_allclose(
            core.jacobian(state, _POINT).apply(direction),
            ode.jacobian(flat, _TIME, _POINT).apply(direction),
        )
        assert ode.param is not None and core.param is not None
        bkd.assert_allclose(
            core.param(state).apply(pdir), ode.param(flat, _TIME).apply(pdir)
        )
        assert ode.state_state is not None and core.state_state is not None
        assert ode.state_param is not None and core.state_param is not None
        bkd.assert_allclose(
            core.state_state(state, adj).apply(direction),
            ode.state_state(flat, _TIME, adj_flat).apply(direction),
        )
        bkd.assert_allclose(
            core.state_param(state, adj).apply(pdir),
            ode.state_param(flat, _TIME, adj_flat).apply(pdir),
        )
        # The stated-zero parameter curvature reads no vector and stays stated.
        assert is_stated_zero(core.param_param)

    def test_the_version_is_the_right_hand_sides(self, bkd: Backend[Array]) -> None:
        model = lotka_volterra_with_mass(bkd)
        residual = RightHandSideAtTime(model, _TIME, BackendColumnForm(bkd))
        version = residual.parameters_version()
        model.set_param(bkd.asarray(LV_PARAM))
        assert residual.parameters_version() == version + 1

    def test_refusals(self, bkd: Backend[Array]) -> None:
        model = lotka_volterra_with_mass(bkd)
        with pytest.raises(TypeError, match="VersionedRightHandSideProtocol"):
            RightHandSideAtTime(3.0, _TIME, BackendColumnForm(bkd))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ResidualFormProtocol"):
            RightHandSideAtTime(model, _TIME, 3.0)  # type: ignore[arg-type]
