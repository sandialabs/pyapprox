"""Forward Euler time stepping residual with adjoint support.

The Forward Euler method is a first-order explicit time integrator:

    M·(y_n - y_{n-1}) - Δt·f(y_{n-1}, t_{n-1}) = 0

Split into three classes via mixin composition:
- ForwardEulerStepper: core + sensitivity + quadrature
- ForwardEulerAdjoint: + adjoint methods
- ForwardEulerHVP: + HVP methods
"""

from typing import Generic

from pyapprox.pde.sparse_utils import solve_maybe_sparse
from pyapprox.pde.time.mixins.adjoint import AdjointMixin
from pyapprox.pde.time.mixins.core import CoreStepperMixin
from pyapprox.pde.time.mixins.hvp import HVPMixin
from pyapprox.pde.time.mixins.quadrature import QuadratureMixin
from pyapprox.pde.time.mixins.sensitivity import SensitivityMixin
from pyapprox.pde.time.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array

# =========================================================================
# Base stepper: core + sensitivity + quadrature
# =========================================================================


class ForwardEulerStepper(
    SensitivityMixin[Array],
    QuadratureMixin[Array],
    CoreStepperMixin[Array],
    Generic[Array],
):
    r"""Forward Euler time stepping residual (base level).

    First-order explicit time integrator:

    .. math::

        R(y_n) = M (y_n - y_{n-1}) - \Delta t \, f(y_{n-1}, t_{n-1}) = 0
    """

    def __init__(self, residual: ODEResidualProtocol[Array]) -> None:
        super().__init__(residual)

    def __call__(self, state: Array) -> Array:
        self._residual.set_time(self._time)
        return self._residual.apply_mass_matrix(
            state - self._prev_state
        ) - self._deltat * self._residual(self._prev_state)

    def jacobian(self, state: Array) -> Array:
        return self._residual.mass_matrix(state.shape[0])

    def linsolve(self, state: Array, residual: Array) -> Array:
        return solve_maybe_sparse(self._bkd, self.jacobian(state), residual)

    # -- SensitivityMixin --

    def is_explicit(self) -> bool:
        return True

    def has_prev_state_hessian(self) -> bool:
        return False

    def sensitivity_off_diag_jacobian(
        self, fsol_nm1: Array, fsol_n: Array, deltat: float
    ) -> Array:
        r"""Compute :math:`dR_n/dy_{n-1}` for forward sensitivity propagation.

        .. math::

            \frac{dR_n}{dy_{n-1}} = -(M + \Delta t \, J)

        where :math:`J = df/dy|_{y_{n-1}}`.
        """
        self._residual.set_time(self._time)
        mass = self._residual.mass_matrix(fsol_nm1.shape[0])
        jac = self._residual.jacobian(fsol_nm1)
        return -mass - deltat * jac

    # -- QuadratureMixin --

    def _get_quadrature_class(self) -> type:
        from pyapprox.surrogates.affine.univariate.piecewisepoly import (
            PiecewiseConstantLeft,
        )
        return PiecewiseConstantLeft


# =========================================================================
# Adjoint level: + param_jacobian, adjoint methods
# =========================================================================


class ForwardEulerAdjoint(
    AdjointMixin[Array],
    ForwardEulerStepper[Array],
    Generic[Array],
):
    """Forward Euler with adjoint capability for gradient computation."""

    def __init__(
        self, residual: ODEResidualWithParamJacobianProtocol[Array]
    ) -> None:
        super().__init__(residual)

    def _param_jacobian_impl(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        r"""Compute :math:`dR/dp = -\Delta t \, (df/dp)|_{y_{n-1}}`.
        """
        self._adjoint_residual.set_time(self._time)
        return -self._deltat * self._adjoint_residual.param_jacobian(fsol_nm1)

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        r"""Return :math:`(dR/dy_n)^T = M^T` (explicit, so Jacobian is mass matrix)."""
        return self._adjoint_residual.mass_matrix(fsol_n.shape[0]).T

    def adjoint_off_diag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        r"""Compute :math:`(dR_{n+1}/dy_n)^T = -(\Delta t \, J + M)^T`."""
        self._adjoint_residual.set_time(self._time)
        return -(
            deltat_np1 * self._adjoint_residual.jacobian(fsol_n)
            + self._adjoint_residual.mass_matrix(fsol_n.shape[0])
        ).T

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        return -final_dqdu


# =========================================================================
# HVP level: + four HVP methods
# =========================================================================


class ForwardEulerHVP(
    HVPMixin[Array],
    ForwardEulerAdjoint[Array],
    Generic[Array],
):
    r"""Forward Euler with HVP capability for Hessian-vector products.

    For Forward Euler all HVP methods are simple scalings of the
    underlying ODE residual HVPs because :math:`R` depends linearly
    on :math:`f(y_{n-1})`:

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
        r"""Compute :math:`(d^2R/dy^2) w = -\Delta t \, (d^2f/dy^2) w`."""
        self._hvp_residual.set_time(self._time)
        return -self._deltat * self._hvp_residual.state_state_hvp(
            fsol_nm1, adj_state, wvec
        )

    def state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dy \, dp) v = -\Delta t \, (d^2f/dy \, dp) v`."""
        self._hvp_residual.set_time(self._time)
        return -self._deltat * self._hvp_residual.state_param_hvp(
            fsol_nm1, adj_state, vvec
        )

    def param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dp \, dy) w = -\Delta t \, (d^2f/dp \, dy) w`."""
        self._hvp_residual.set_time(self._time)
        return -self._deltat * self._hvp_residual.param_state_hvp(
            fsol_nm1, adj_state, wvec
        )

    def param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dp^2) v = -\Delta t \, (d^2f/dp^2) v`."""
        self._hvp_residual.set_time(self._time)
        return -self._deltat * self._hvp_residual.param_param_hvp(
            fsol_nm1, adj_state, vvec
        )
