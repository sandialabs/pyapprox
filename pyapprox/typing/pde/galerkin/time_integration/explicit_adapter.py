"""BC-clean ODE adapter for explicit Galerkin time stepping.

The standard GalerkinPhysicsODEAdapter computes f = M^{-1} * residual(),
but residual() includes Dirichlet row replacement (state[dof] - g(t)).
Applying M^{-1} to those rows creates positive eigenvalues, making
explicit stepping unconditionally unstable.

This adapter instead uses spatial_residual() (no Dirichlet enforcement)
and modifies the mass matrix to have identity rows at Dirichlet DOFs.
The result is f[D_dofs] = 0 (frozen) and f[interior] = M_int^{-1} * F_int
(correct). Dirichlet values are injected after each time step by the
caller (GalerkinModel).

This adapter satisfies ODEResidualProtocol, so all existing stepper
residuals (ForwardEulerResidual, HeunResidual, and any future ones like
RK4) work unchanged. Adding a new integrator requires only a new stepper
class and a registry entry — no adapter changes.
"""

from typing import Generic, Optional

import numpy as np
import scipy.linalg

from pyapprox.typing.util.backends.protocols import Array, Backend


class GalerkinExplicitODEAdapter(Generic[Array]):
    """BC-clean ODE adapter for explicit Galerkin time stepping.

    Satisfies ODEResidualProtocol so existing stepper residuals
    (ForwardEulerResidual, HeunResidual, future RK4, etc.) work unchanged.

    Computes f(y, t) = M_bc^{-1} * F_clean where:
    - F_clean = spatial_residual (no Dirichlet enforcement), zeroed at D_dofs
    - M_bc = mass matrix with identity rows at Dirichlet DOFs

    Parameters
    ----------
    physics : GalerkinPhysicsProtocol
        Galerkin physics with spatial_residual() and dirichlet_dof_info().
    lumped_mass : bool, default=False
        If True, use row-sum lumped mass matrix (diagonal) instead of
        consistent mass. Cheaper per step but less accurate.
    """

    def __init__(self, physics, lumped_mass: bool = False):
        if not hasattr(physics, "spatial_residual"):
            raise TypeError(
                f"{type(physics).__name__} does not have spatial_residual(). "
                "Explicit time stepping requires this method."
            )
        if not hasattr(physics, "dirichlet_dof_info"):
            raise TypeError(
                f"{type(physics).__name__} does not have dirichlet_dof_info(). "
                "Explicit time stepping requires this method."
            )

        self._physics = physics
        self._bkd = physics.bkd()
        self._time: float = 0.0
        self._lumped_mass = lumped_mass

        # Cache Dirichlet DOF indices (assumed constant in time)
        d_dofs, _ = physics.dirichlet_dof_info(0.0)
        self._d_dof_indices_np = self._bkd.to_numpy(d_dofs).astype(np.intp)

        # Build BC-modified mass matrix and factor/lump
        self._setup_mass(physics)

        # Cache identity for mass_matrix() method
        self._identity_cached: Optional[Array] = None

    def _setup_mass(self, physics) -> None:
        """Build and factor the BC-modified mass matrix."""
        M_np = self._bkd.to_numpy(physics.mass_matrix()).copy()
        d_dofs = self._d_dof_indices_np

        if len(d_dofs) > 0:
            M_np[d_dofs, :] = 0.0
            M_np[d_dofs, d_dofs] = 1.0

        if self._lumped_mass:
            self._m_diag_np = M_np.sum(axis=1)
            # Safety: Dirichlet rows already have sum = 1.0
        else:
            self._m_lu = scipy.linalg.lu_factor(M_np)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def set_time(self, time: float) -> None:
        """Set the current time for evaluation.

        Parameters
        ----------
        time : float
            Current time.
        """
        self._time = time

    def __call__(self, state: Array) -> Array:
        """Evaluate BC-clean ODE residual f(y, t) = M_bc^{-1} * F_clean.

        F_clean = spatial_residual(state, time) with D_dofs zeroed.
        Result: f[D_dofs] = 0, f[interior] = M_int^{-1} * F_int.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            BC-clean residual. Shape: (nstates,)
        """
        # Runtime assertion: Dirichlet DOF locations must not change
        d_dofs_now, _ = self._physics.dirichlet_dof_info(self._time)
        d_dofs_now_np = self._bkd.to_numpy(d_dofs_now).astype(np.intp)
        if not np.array_equal(d_dofs_now_np, self._d_dof_indices_np):
            raise RuntimeError(
                "Dirichlet DOF indices changed between construction and "
                f"evaluation at time {self._time}. The explicit adapter "
                "caches the mass matrix factorization and cannot handle "
                "time-varying Dirichlet DOF locations."
            )

        F = self._physics.spatial_residual(state, self._time)
        F_np = self._bkd.to_numpy(F).copy()

        if len(self._d_dof_indices_np) > 0:
            F_np[self._d_dof_indices_np] = 0.0

        if self._lumped_mass:
            f_np = F_np / self._m_diag_np
        else:
            f_np = scipy.linalg.lu_solve(self._m_lu, F_np)

        return self._bkd.asarray(f_np.astype(np.float64))

    def jacobian(self, state: Array) -> Array:
        """Compute state Jacobian df/dy = M_bc^{-1} * dF/du.

        Parameters
        ----------
        state : Array
            Current state. Shape: (nstates,)

        Returns
        -------
        Array
            Jacobian. Shape: (nstates, nstates)
        """
        J_F = self._physics.jacobian(state, self._time)
        J_np = self._bkd.to_numpy(J_F).copy()

        if self._lumped_mass:
            result_np = J_np / self._m_diag_np[:, None]
        else:
            result_np = scipy.linalg.lu_solve(self._m_lu, J_np)

        return self._bkd.asarray(result_np.astype(np.float64))

    def mass_matrix(self, nstates: int) -> Array:
        """Return the identity matrix.

        Since M is absorbed into f = M_bc^{-1} * F, the effective mass
        matrix for the stepper is identity.

        Parameters
        ----------
        nstates : int
            Number of states.

        Returns
        -------
        Array
            Identity matrix. Shape: (nstates, nstates)
        """
        if self._identity_cached is None:
            self._identity_cached = self._bkd.eye(nstates)
        return self._identity_cached

    def __repr__(self) -> str:
        return (
            f"GalerkinExplicitODEAdapter("
            f"physics={self._physics!r}, "
            f"lumped_mass={self._lumped_mass})"
        )
