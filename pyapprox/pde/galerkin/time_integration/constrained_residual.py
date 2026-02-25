"""Constrained time step residual for Dirichlet BC enforcement.

Wraps a time stepper (BackwardEulerResidual, CrankNicolsonResidual, etc.)
and applies Dirichlet constraints after the stepper assembles the Newton
residual and Jacobian:

    R[d] = y[d] - g(t_{n+1})
    J[d, :] = e_d

This is the standard FEM approach: physics computes physics, BCs are
enforced at the linear algebra level. The wrapper satisfies
NewtonSolverResidualProtocol so Newton sees it directly.

TODO: Consider letting the residual specify its required solve type
(sparse vs dense) so callers don't need if/else dispatch on issparse.
"""

from typing import Generic

import numpy as np
from scipy.sparse import issparse

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.sparse_utils import apply_dirichlet_rows, solve_maybe_sparse


class ConstrainedTimeStepResidual(Generic[Array]):
    """Wrapper applying Dirichlet constraints to assembled Newton system.

    After the stepper assembles:
        R = M*(y - y_prev) - dt*F(y, t)
        J = M - dt*J_F(y, t)

    this wrapper replaces Dirichlet DOF rows with:
        R[d] = y[d] - g(t_{n+1})
        J[d, :] = e_d  (unit row vector)

    This gives the correct discrete lift M[i,d]*(g_{n+1} - g_n) at
    interior rows because M and y_prev are unmodified, and Newton
    drives y[d] → g(t_{n+1}) via the constraint rows.

    Satisfies NewtonSolverResidualProtocol: bkd(), __call__(), linsolve().

    Parameters
    ----------
    stepper : TimeSteppingResidualBase
        The time stepping residual (e.g., BackwardEulerResidual).
    adapter : GalerkinPhysicsODEAdapter
        The physics adapter with dirichlet_dof_info(time) method.
    """

    def __init__(self, stepper, adapter):
        self._stepper = stepper
        self._adapter = adapter
        self._bkd = stepper.bkd()
        self._bc_time: float = 0.0

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def set_bc_time(self, time: float) -> None:
        """Set the time at which Dirichlet values are evaluated.

        Parameters
        ----------
        time : float
            Time for g(t). Typically t_{n+1} for implicit methods.
        """
        self._bc_time = time

    def __call__(self, state: Array) -> Array:
        """Evaluate constrained residual.

        Assembles R via the stepper, then replaces Dirichlet rows with
        R[d] = y[d] - g(t_{n+1}).

        Parameters
        ----------
        state : Array
            Current Newton iterate. Shape: (nstates,)

        Returns
        -------
        Array
            Constrained residual. Shape: (nstates,)
        """
        R = self._stepper(state)
        d_dofs, d_vals = self._adapter.dirichlet_dof_info(self._bc_time)
        d_dofs_np = self._bkd.to_numpy(d_dofs).astype(np.intp)
        if len(d_dofs_np) > 0:
            R_np = self._bkd.to_numpy(R).copy()
            state_np = self._bkd.to_numpy(state)
            d_vals_np = self._bkd.to_numpy(d_vals)
            R_np[d_dofs_np] = state_np[d_dofs_np] - d_vals_np
            return self._bkd.asarray(R_np.astype(np.float64))
        return R

    def linsolve(self, state: Array, residual: Array) -> Array:
        """Solve constrained linear system.

        Assembles J via the stepper, replaces Dirichlet rows with
        J[d, :] = e_d, then solves J * dy = residual.

        Parameters
        ----------
        state : Array
            Current Newton iterate. Shape: (nstates,)
        residual : Array
            Right-hand side (the residual). Shape: (nstates,)

        Returns
        -------
        Array
            Newton step dy. Shape: (nstates,)
        """
        J = self._stepper.jacobian(state)
        d_dofs, _ = self._adapter.dirichlet_dof_info(self._bc_time)
        d_dofs_np = self._bkd.to_numpy(d_dofs).astype(np.intp)
        if len(d_dofs_np) > 0:
            if issparse(J):
                J = apply_dirichlet_rows(J, d_dofs_np)
            else:
                J_np = self._bkd.to_numpy(J).copy()
                J_np[d_dofs_np, :] = 0.0
                J_np[d_dofs_np, d_dofs_np] = 1.0
                J = self._bkd.asarray(J_np.astype(np.float64))
        return solve_maybe_sparse(self._bkd, J, residual)

    def __repr__(self) -> str:
        return (
            f"ConstrainedTimeStepResidual(\n"
            f"  stepper={self._stepper!r},\n"
            f"  bc_time={self._bc_time},\n"
            f")"
        )
