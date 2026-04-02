r"""Backward Euler time stepping residual with adjoint support.

The Backward Euler method is a first-order implicit time integrator:

.. math::

    M (y_n - y_{n-1}) - \Delta t \, f(y_n, t_n) = 0

Split into three classes via mixin composition:

- BackwardEulerStepper: core + sensitivity + quadrature + implicit
- BackwardEulerAdjoint: + adjoint methods
- BackwardEulerHVP: + HVP methods
"""

from typing import Generic

from pyapprox.pde.sparse_utils import solve_maybe_sparse
from pyapprox.pde.time.mixins.adjoint import AdjointMixin
from pyapprox.pde.time.mixins.core import CoreStepperMixin
from pyapprox.pde.time.mixins.hvp import HVPMixin
from pyapprox.pde.time.mixins.implicit import ImplicitStepperMixin
from pyapprox.pde.time.mixins.quadrature import QuadratureMixin
from pyapprox.pde.time.mixins.sensitivity import SensitivityMixin
from pyapprox.pde.time.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array

# =========================================================================
# Base stepper: core + sensitivity + quadrature + implicit
# =========================================================================


class BackwardEulerStepper(
    SensitivityMixin[Array],
    QuadratureMixin[Array],
    ImplicitStepperMixin[Array],
    CoreStepperMixin[Array],
    Generic[Array],
):
    r"""Backward Euler time stepping residual (base level).

    First-order implicit method (A-stable):

    .. math::

        R(y_n) = M (y_n - y_{n-1}) - \Delta t \, f(y_n, t_n) = 0
    """

    def __init__(self, residual: ODEResidualProtocol[Array]) -> None:
        super().__init__(residual)

    def __call__(self, state: Array) -> Array:
        self._residual.set_time(self._time + self._deltat)
        return self._residual.apply_mass_matrix(
            state - self._prev_state
        ) - self._deltat * self._residual(state)

    def jacobian(self, state: Array) -> Array:
        r"""Compute :math:`dR/dy_n = M - \Delta t \, (df/dy)`."""
        self._residual.set_time(self._time + self._deltat)
        return self._residual.mass_matrix(
            state.shape[0]
        ) - self._deltat * self._residual.jacobian(state)

    # -- SensitivityMixin --

    def is_explicit(self) -> bool:
        return False

    def has_prev_state_hessian(self) -> bool:
        return False

    def sensitivity_off_diag_jacobian(
        self, fsol_nm1: Array, fsol_n: Array, deltat: float
    ) -> Array:
        r"""Compute :math:`dR_n/dy_{n-1} = -M`.

        The :math:`f(y_n)` term does not depend on :math:`y_{n-1}`.
        """
        return -self._residual.mass_matrix(fsol_nm1.shape[0])

    # -- QuadratureMixin --

    def _get_quadrature_class(self) -> type:
        from pyapprox.surrogates.affine.univariate.piecewisepoly import (
            PiecewiseConstantRight,
        )
        return PiecewiseConstantRight


# =========================================================================
# Adjoint level: + param_jacobian, adjoint methods
# =========================================================================


class BackwardEulerAdjoint(
    AdjointMixin[Array],
    BackwardEulerStepper[Array],
    Generic[Array],
):
    """Backward Euler with adjoint capability for gradient computation."""

    def __init__(
        self, residual: ODEResidualWithParamJacobianProtocol[Array]
    ) -> None:
        super().__init__(residual)

    def _param_jacobian_impl(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        r"""Compute :math:`dR/dp = -\Delta t \, (df/dp)|_{y_n, t_n}`."""
        self._residual.set_time(self._time + self._deltat)
        return -self._deltat * self._adjoint_residual.param_jacobian(fsol_n)

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        r"""Compute :math:`(dR/dy_n)^T = (M - \Delta t \, J)^T`."""
        self._residual.set_time(self._time)
        return (
            self._residual.mass_matrix(fsol_n.shape[0])
            - self._deltat * self._residual.jacobian(fsol_n)
        ).T

    def adjoint_off_diag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        r"""Compute :math:`(dR_{n+1}/dy_n)^T = -M^T`."""
        return -self._residual.mass_matrix(fsol_n.shape[0]).T

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        r"""Solve :math:`(dR/dy_N)^T \lambda_N = -dQ/dy_N` at final time."""
        drdu = self.jacobian(final_fwd_sol)
        return solve_maybe_sparse(self._bkd, drdu.T, -final_dqdu)


# =========================================================================
# HVP level: + four HVP methods
# =========================================================================


class BackwardEulerHVP(
    HVPMixin[Array],
    BackwardEulerAdjoint[Array],
    Generic[Array],
):
    r"""Backward Euler with HVP capability for Hessian-vector products.

    All HVP methods are simple scalings of the underlying ODE residual
    HVPs evaluated at :math:`(y_n, t_n)`:

    .. math::

        \frac{d^2 R}{d(\cdot)^2} = -\Delta t \, \frac{d^2 f}{d(\cdot)^2}
    """

    def __init__(self, residual: ODEResidualWithHVPProtocol[Array]) -> None:
        super().__init__(residual)

    def state_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dy_n^2) w = -\Delta t \, (d^2f/dy^2) w`."""
        self._residual.set_time(self._time + self._deltat)
        return -self._deltat * self._hvp_residual.state_state_hvp(
            fsol_n, adj_state, wvec
        )

    def state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dy_n \, dp) v = -\Delta t \, (d^2f/dy \, dp) v`."""
        self._residual.set_time(self._time + self._deltat)
        return -self._deltat * self._hvp_residual.state_param_hvp(
            fsol_n, adj_state, vvec
        )

    def param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dp \, dy_n) w = -\Delta t \, (d^2f/dp \, dy) w`."""
        self._residual.set_time(self._time + self._deltat)
        return -self._deltat * self._hvp_residual.param_state_hvp(
            fsol_n, adj_state, wvec
        )

    def param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dp^2) v = -\Delta t \, (d^2f/dp^2) v`."""
        self._residual.set_time(self._time + self._deltat)
        return -self._deltat * self._hvp_residual.param_param_hvp(
            fsol_n, adj_state, vvec
        )
