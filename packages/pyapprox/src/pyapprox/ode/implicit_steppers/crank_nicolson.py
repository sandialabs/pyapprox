r"""Crank-Nicolson time stepping residual with adjoint support.

The Crank-Nicolson method is a second-order implicit time integrator:

.. math::

    M (y_n - y_{n-1}) - \frac{\Delta t}{2}
    \bigl[f(y_{n-1}, t_{n-1}) + f(y_n, t_n)\bigr] = 0

Split into three classes via mixin composition:

- CrankNicolsonStepper: core + sensitivity + quadrature + implicit
- CrankNicolsonAdjoint: + adjoint methods
- CrankNicolsonHVP: + HVP + PrevStepHVP methods
"""

from typing import Generic

from pyapprox.ode.mixins.adjoint import AdjointMixin
from pyapprox.ode.mixins.core import CoreStepperMixin
from pyapprox.ode.mixins.hvp import HVPMixin, PrevStepHVPMixin
from pyapprox.ode.mixins.implicit import ImplicitStepperMixin
from pyapprox.ode.mixins.quadrature import QuadratureMixin
from pyapprox.ode.mixins.sensitivity import SensitivityMixin
from pyapprox.ode.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithHVPProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.sparse_dispatch import solve_maybe_sparse

# =========================================================================
# Base stepper: core + sensitivity + quadrature + implicit
# =========================================================================


class CrankNicolsonStepper(
    SensitivityMixin[Array],
    QuadratureMixin[Array],
    ImplicitStepperMixin[Array],
    CoreStepperMixin[Array],
    Generic[Array],
):
    r"""Crank-Nicolson time stepping residual (base level).

    Second-order implicit method (A-stable):

    .. math::

        R(y_n) = M (y_n - y_{n-1})
        - \frac{\Delta t}{2}
        \bigl[f(y_{n-1}, t_{n-1}) + f(y_n, t_n)\bigr] = 0
    """

    def __init__(self, residual: ODEResidualProtocol[Array]) -> None:
        super().__init__(residual)

    def __call__(self, state: Array) -> Array:
        # f(y_{n-1}, t_{n-1})
        self._residual.set_time(self._time)
        current_res = self._residual(self._prev_state)

        # f(y_n, t_n)
        self._residual.set_time(self._time + self._deltat)
        next_res = self._residual(state)

        return self._residual.apply_mass_matrix(
            state - self._prev_state
        ) - 0.5 * self._deltat * (current_res + next_res)

    def jacobian(self, state: Array) -> Array:
        r"""Compute :math:`dR/dy_n = M - (\Delta t/2) \, (df/dy)|_{y_n}`."""
        self._residual.set_time(self._time + self._deltat)
        return self._residual.mass_matrix(
            state.shape[0]
        ) - 0.5 * self._deltat * self._residual.jacobian(state)

    # -- SensitivityMixin --

    def is_explicit(self) -> bool:
        return False

    def has_prev_state_hessian(self) -> bool:
        r"""Return True: :math:`R_{n+1}` depends on :math:`f(y_n)`.

        Crank-Nicolson :math:`R_n` depends on both :math:`y_{n-1}` and
        :math:`y_n`, so when computing :math:`d^2 L / dy_n^2` we need the
        contribution from :math:`R_{n+1}`'s dependence on :math:`y_n`.
        """
        return True

    def sensitivity_off_diag_jacobian(
        self, fsol_nm1: Array, fsol_n: Array, deltat: float
    ) -> Array:
        r"""Compute :math:`dR_n/dy_{n-1} = -(M + (\Delta t/2) \, J_{n-1})`.

        .. math::

            \frac{dR_n}{dy_{n-1}}
            = -\bigl(M + \tfrac{\Delta t}{2} \, J_{n-1}\bigr)

        where :math:`J_{n-1} = (df/dy)|_{y_{n-1}}`.
        """
        self._residual.set_time(self._time)
        mass = self._residual.mass_matrix(fsol_nm1.shape[0])
        jac = self._residual.jacobian(fsol_nm1)
        return -(mass + 0.5 * deltat * jac)

    # -- QuadratureMixin --

    def _get_quadrature_class(self) -> type:
        from pyapprox.surrogates.affine.univariate.piecewisepoly import (
            PiecewiseLinear,
        )
        return PiecewiseLinear


# =========================================================================
# Adjoint level: + param_jacobian, adjoint methods
# =========================================================================


class CrankNicolsonAdjoint(
    AdjointMixin[Array],
    CrankNicolsonStepper[Array],
    Generic[Array],
):
    """Crank-Nicolson with adjoint capability for gradient computation."""

    def __init__(
        self, residual: ODEResidualWithParamJacobianProtocol[Array]
    ) -> None:
        super().__init__(residual)

    def _param_jacobian_impl(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        r"""Compute :math:`dR/dp`.

        .. math::

            \frac{dR}{dp} = -\frac{\Delta t}{2}
            \Bigl[\frac{df}{dp}\Big|_{y_{n-1}, t_{n-1}}
            + \frac{df}{dp}\Big|_{y_n, t_n}\Bigr]
        """
        self._residual.set_time(self._time)
        current_param_jac = self._adjoint_residual.param_jacobian(fsol_nm1)

        self._residual.set_time(self._time + self._deltat)
        next_param_jac = self._adjoint_residual.param_jacobian(fsol_n)

        return -0.5 * self._deltat * (current_param_jac + next_param_jac)

    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        r"""Compute :math:`(dR/dy_n)^T = (M - (\Delta t/2) \, J)^T`."""
        self._residual.set_time(self._time)
        return (
            self._residual.mass_matrix(fsol_n.shape[0])
            - 0.5 * self._deltat * self._residual.jacobian(fsol_n)
        ).T

    def adjoint_off_diag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        r"""Compute :math:`(dR_{n+1}/dy_n)^T`.

        .. math::

            \Bigl(\frac{dR_{n+1}}{dy_n}\Bigr)^T
            = -\bigl(M + \tfrac{\Delta t_{n+1}}{2} \, J_n\bigr)^T
        """
        self._residual.set_time(self._time)
        return -(
            self._residual.mass_matrix(fsol_n.shape[0])
            + 0.5 * deltat_np1 * self._residual.jacobian(fsol_n)
        ).T

    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        r"""Solve :math:`(dR/dy_N)^T \lambda_N = -dQ/dy_N` at final time."""
        drdu = self.jacobian(final_fwd_sol)
        return solve_maybe_sparse(self._bkd, drdu.T, -final_dqdu)


# =========================================================================
# HVP level: + four HVP methods + three PrevStep HVP methods
# =========================================================================


class CrankNicolsonHVP(
    PrevStepHVPMixin[Array],
    HVPMixin[Array],
    CrankNicolsonAdjoint[Array],
    Generic[Array],
):
    r"""Crank-Nicolson with HVP capability for Hessian-vector products.

    The residual is:

    .. math::

        R(y_n) = M (y_n - y_{n-1})
        - \frac{\Delta t}{2}
        \bigl[f(y_{n-1}, t_{n-1}) + f(y_n, t_n)\bigr]

    Key insight: :math:`y_{n-1}` is fixed when differentiating w.r.t.
    :math:`y_n`, so the :math:`f(y_{n-1})` term drops from
    :math:`d^2 R / dy_n^2`.  However, both terms contribute to
    :math:`d^2 R / dp^2` because both depend on :math:`p`.

    The ``prev_*`` methods compute the :math:`R_{n+1}` contribution
    evaluated at :math:`y_n` (which acts as :math:`y_{n-1}` for
    :math:`R_{n+1}`).
    """

    def __init__(self, residual: ODEResidualWithHVPProtocol[Array]) -> None:
        super().__init__(residual)

    # -- HVPMixin: current-step HVP methods --

    def state_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dy_n^2) w = -(\Delta t/2) \, (d^2f/dy^2)|_{y_n} \, w`.

        The :math:`f(y_{n-1})` term does not contribute because
        :math:`y_{n-1}` is fixed.
        """
        self._residual.set_time(self._time + self._deltat)
        return (
            -0.5
            * self._deltat
            * self._hvp_residual.state_state_hvp(fsol_n, adj_state, wvec)
        )

    def state_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dy_n \, dp) v =
        -(\Delta t/2) \, (d^2f/dy \, dp)|_{y_n} \, v`.

        Only the :math:`f(y_n)` term contributes; :math:`y_{n-1}` is
        independent of :math:`y_n`.
        """
        self._residual.set_time(self._time + self._deltat)
        return (
            -0.5
            * self._deltat
            * self._hvp_residual.state_param_hvp(fsol_n, adj_state, vvec)
        )

    def param_state_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dp \, dy_n) w =
        -(\Delta t/2) \, (d^2f/dp \, dy)|_{y_n} \, w`.

        Only the :math:`f(y_n)` term contributes; :math:`y_{n-1}` is fixed.
        """
        self._residual.set_time(self._time + self._deltat)
        return (
            -0.5
            * self._deltat
            * self._hvp_residual.param_state_hvp(fsol_n, adj_state, wvec)
        )

    def param_param_hvp(
        self,
        fsol_nm1: Array,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2R/dp^2) v`.

        .. math::

            \frac{d^2 R}{dp^2} v = -\frac{\Delta t}{2}
            \Bigl[\frac{d^2 f}{dp^2}\Big|_{y_{n-1}} v
            + \frac{d^2 f}{dp^2}\Big|_{y_n} v\Bigr]

        Both terms contribute because both :math:`f(y_{n-1}, p)` and
        :math:`f(y_n, p)` depend on :math:`p`.
        """
        # Contribution from y_{n-1} term
        self._residual.set_time(self._time)
        hvp_nm1 = self._hvp_residual.param_param_hvp(
            fsol_nm1, adj_state, vvec
        )

        # Contribution from y_n term
        self._residual.set_time(self._time + self._deltat)
        hvp_n = self._hvp_residual.param_param_hvp(fsol_n, adj_state, vvec)

        return -0.5 * self._deltat * (hvp_nm1 + hvp_n)

    # -- PrevStepHVPMixin: cross-step HVP methods --
    #
    # These compute R_{n+1}'s dependence on y_n (which acts as y_{n-1}
    # for R_{n+1}).  Needed because has_prev_state_hessian() = True.

    def prev_state_state_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2 R_{n+1}/dy_n^2) w =
        -(\Delta t/2) \, (d^2f/dy^2)|_{y_n} \, w`.

        Evaluates the :math:`f(y_n)` contribution from :math:`R_{n+1}`.
        """
        return (
            -0.5
            * self._deltat
            * self._hvp_residual.state_state_hvp(fsol_n, adj_state, wvec)
        )

    def prev_state_param_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        vvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2 R_{n+1}/dy_n \, dp) v =
        -(\Delta t/2) \, (d^2f/dy \, dp)|_{y_n} \, v`."""
        return (
            -0.5
            * self._deltat
            * self._hvp_residual.state_param_hvp(fsol_n, adj_state, vvec)
        )

    def prev_param_state_hvp(
        self,
        fsol_n: Array,
        adj_state: Array,
        wvec: Array,
    ) -> Array:
        r"""Compute :math:`(d^2 R_{n+1}/dp \, dy_n) w =
        -(\Delta t/2) \, (d^2f/dp \, dy)|_{y_n} \, w`."""
        return (
            -0.5
            * self._deltat
            * self._hvp_residual.param_state_hvp(fsol_n, adj_state, wvec)
        )
