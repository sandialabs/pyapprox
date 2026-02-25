"""Custom time stepper for Stokes DAE.

Stokes has a singular mass matrix [M_vel, 0; 0, 0] because the continuity
equation div(u) = 0 is an algebraic constraint (no dp/dt). The standard
GalerkinPhysicsODEAdapter (which computes M^{-1}*F) cannot be used.

This module directly forms the implicit time-stepping residual:

Backward Euler:
    R(y_n) = M*(y_n - y_{n-1}) - dt*F(y_n, t_n) = 0

Crank-Nicolson:
    R(y_n) = M*(y_n - y_{n-1}) - (dt/2)*(F(y_{n-1}, t_{n-1}) + F(y_n, t_n)) = 0

where M = block_diag(M_vel, 0) and F = physics.residual (with BCs applied).
Dirichlet BC rows of M are zeroed so the time-stepping residual at those
DOFs reduces to the physics BC equation state[dof] - exact_value = 0.
"""

from typing import Generic, Optional

import numpy as np
from scipy.sparse import issparse

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.pde.sparse_utils import solve_maybe_sparse


class StokesTimeStepResidual(Generic[Array]):
    """Time-stepping residual for Stokes DAE.

    Directly forms the implicit time-stepping residual without requiring
    M^{-1} (which is undefined for Stokes due to singular mass matrix).

    The mass matrix is modified to zero out Dirichlet BC rows so that the
    time-stepping residual at those DOFs becomes the physics BC equation.

    Parameters
    ----------
    physics : StokesPhysics
        The Stokes physics object.
    method : str
        Time integration method: "backward_euler" or "crank_nicolson".
    """

    def __init__(self, physics, method: str = "backward_euler"):
        if method not in ("backward_euler", "crank_nicolson"):
            raise ValueError(
                f"method must be 'backward_euler' or 'crank_nicolson', "
                f"got '{method}'"
            )
        self._physics = physics
        self._bkd = physics.bkd()
        self._method = method
        self._time: float = 0.0
        self._deltat: float = 0.0
        self._prev_state: Optional[Array] = None
        self._prev_F: Optional[Array] = None  # F(y_{n-1}, t_{n-1}) for CN

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def _bc_modified_mass(self, time: float):
        """Return mass matrix with Dirichlet rows zeroed.

        Parameters
        ----------
        time : float
            Time at which to evaluate Dirichlet DOFs.

        Returns
        -------
        sparse or Array
            Modified mass matrix. Shape: (nstates, nstates)
        """
        M_orig = self._physics.mass_matrix()
        if issparse(M_orig):
            M = M_orig.copy()
        else:
            M = self._bkd.copy(M_orig)

        D_dofs_arr, _ = self._physics.dirichlet_dof_info(time)
        D_dofs = self._bkd.to_numpy(D_dofs_arr)
        if len(D_dofs) > 0:
            if issparse(M):
                # Zero rows in CSR format
                for dof in D_dofs:
                    M[dof, :] = 0.0
                M.eliminate_zeros()
            else:
                M_np = self._bkd.to_numpy(M)
                M_np[D_dofs, :] = 0.0
                M = self._bkd.asarray(M_np.astype(np.float64))

        return M

    def set_time(
        self, time: float, deltat: float, prev_state: Array
    ) -> None:
        """Set the current time step parameters.

        Parameters
        ----------
        time : float
            Time at the START of the time step (t_{n-1}).
        deltat : float
            Time step size.
        prev_state : Array
            Solution at previous time step y_{n-1}.
        """
        self._time = time
        self._deltat = deltat
        self._prev_state = prev_state

        # For Crank-Nicolson, precompute F(y_{n-1}, t_{n-1})
        if self._method == "crank_nicolson":
            self._prev_F = self._physics.residual(prev_state, time)

    def __call__(self, state: Array) -> Array:
        """Evaluate the time-stepping residual R(y_n).

        Backward Euler:
            R = M_bc*(y_n - y_{n-1}) - dt*F(y_n, t_n)

        Crank-Nicolson:
            R = M_bc*(y_n - y_{n-1}) - (dt/2)*(F(y_{n-1}, t_{n-1}) + F(y_n, t_n))

        where M_bc has Dirichlet rows zeroed so the BC equation is preserved.

        Parameters
        ----------
        state : Array
            Candidate solution at current time step y_n.

        Returns
        -------
        Array
            Residual vector. Shape: (nstates,)
        """
        t_n = self._time + self._deltat
        M = self._bc_modified_mass(t_n)

        diff = state - self._prev_state
        if issparse(M):
            M_diff = np.asarray(M @ diff).ravel()
            M_diff = self._bkd.asarray(M_diff)
        else:
            M_diff = self._bkd.dot(M, diff)

        F_n = self._physics.residual(state, t_n)

        if self._method == "backward_euler":
            return M_diff - self._deltat * F_n
        else:  # crank_nicolson
            return M_diff - (self._deltat / 2.0) * (self._prev_F + F_n)

    def jacobian(self, state: Array) -> Array:
        """Compute the Jacobian dR/dy_n.

        Backward Euler:
            dR/dy_n = M_bc - dt*dF/dy_n

        Crank-Nicolson:
            dR/dy_n = M_bc - (dt/2)*dF/dy_n

        Parameters
        ----------
        state : Array
            Candidate solution at current time step y_n.

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nstates, nstates)
        """
        t_n = self._time + self._deltat
        M = self._bc_modified_mass(t_n)
        J_F = self._physics.jacobian(state, t_n)

        if self._method == "backward_euler":
            return M - self._deltat * J_F
        else:  # crank_nicolson
            return M - (self._deltat / 2.0) * J_F

    def solve_step(self, state_guess: Optional[Array] = None) -> Array:
        """Solve one time step using Newton iteration.

        Parameters
        ----------
        state_guess : Array, optional
            Initial guess for y_n. If None, uses y_{n-1}.

        Returns
        -------
        Array
            Solution y_n at the new time level.
        """
        if state_guess is None:
            y = self._bkd.copy(self._prev_state)
        else:
            y = self._bkd.copy(state_guess)

        max_iter = 50
        tol = 1e-12

        for iteration in range(max_iter):
            R = self(y)
            R_np = self._bkd.to_numpy(R)
            r_norm = float(np.linalg.norm(R_np))

            if r_norm < tol:
                return y

            J = self.jacobian(y)
            dy = solve_maybe_sparse(self._bkd, J, -R)
            y = y + dy

        return y

    def __repr__(self) -> str:
        return (
            f"StokesTimeStepResidual("
            f"method='{self._method}', "
            f"physics={self._physics!r})"
        )
