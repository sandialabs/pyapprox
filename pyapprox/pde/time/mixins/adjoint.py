"""Adjoint mixin for gradient computation via the discrete adjoint method."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, cast

from pyapprox.pde.sparse_utils import solve_maybe_sparse
from pyapprox.pde.time.protocols.ode_residual import (
    ODEResidualProtocol,
    ODEResidualWithParamJacobianProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class AdjointMixin(ABC, Generic[Array]):
    """Mixin providing adjoint methods for gradient computation.

    Requires the underlying ODE residual to support param_jacobian
    (ODEResidualWithParamJacobianProtocol). This is enforced by the
    concrete stepper's __init__ accepting the narrower type.

    Uses _adjoint_residual property for typed access to the narrowed
    residual, avoiding field redeclaration conflicts with CoreStepperMixin.
    """

    if TYPE_CHECKING:
        _residual: ODEResidualProtocol[Array]
        _bkd: Backend[Array]
        _time: float
        _deltat: float

    @property
    def _adjoint_residual(
        self,
    ) -> ODEResidualWithParamJacobianProtocol[Array]:
        """Typed access to the ODE residual as param-jacobian capable.

        Safe because concrete stepper __init__ enforces this type.
        """
        return cast(
            ODEResidualWithParamJacobianProtocol[Array], self._residual
        )

    @property
    def native_residual(self) -> ODEResidualWithParamJacobianProtocol[Array]:
        """Get the underlying ODE residual (narrowed to param jacobian capable)."""
        return self._adjoint_residual

    @abstractmethod
    def _param_jacobian_impl(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """Compute the parameter Jacobian dR/dp for one time step."""
        ...

    def param_jacobian(self, fsol_nm1: Array, fsol_n: Array) -> Array:
        """Compute the parameter Jacobian dR/dp for one time step."""
        return self._param_jacobian_impl(fsol_nm1, fsol_n)

    @abstractmethod
    def adjoint_diag_jacobian(self, fsol_n: Array) -> Array:
        """Compute the diagonal Jacobian block for adjoint solve: (dR/dy_n)^T."""
        ...

    @abstractmethod
    def adjoint_off_diag_jacobian(
        self, fsol_n: Array, deltat_np1: float
    ) -> Array:
        """Compute the off-diagonal Jacobian for adjoint coupling."""
        ...

    @abstractmethod
    def adjoint_initial_condition(
        self, final_fwd_sol: Array, final_dqdu: Array
    ) -> Array:
        """Compute the initial condition for backward adjoint solve."""
        ...

    def adjoint_final_solution(
        self,
        fsol_0: Array,
        asol_1: Array,
        dqdu_0: Array,
        deltat_1: float,
    ) -> Array:
        """Compute the adjoint at initial time (final step of backward sweep).

        Solves: M^T lambda_0 = -B_1^T lambda_1 - dQ/dy_0
        """
        drduT_diag = self._adjoint_residual.mass_matrix(fsol_0.shape[0]).T
        drduT_offdiag = self.adjoint_off_diag_jacobian(fsol_0, deltat_1)
        rhs = -drduT_offdiag @ asol_1 - dqdu_0
        return solve_maybe_sparse(self._bkd, drduT_diag, rhs)
