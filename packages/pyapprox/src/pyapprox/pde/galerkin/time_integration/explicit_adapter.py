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
residuals (ForwardEulerHVP, HeunHVP, and any future ones like
RK4) work unchanged. Adding a new integrator requires only a new stepper
class and a registry entry — no adapter changes.
"""

from typing import Generic, Optional

import numpy as np
from scipy.sparse import spmatrix

from pyapprox.ode.mass_matrix import (
    IdentityMassMatrix,
    MassMatrixProtocol,
    create_mass_matrix,
)
from pyapprox.pde.galerkin.protocols.physics import GalerkinPhysicsProtocol
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinExplicitODEAdapter(Generic[Array]):
    """BC-clean ODE adapter for explicit Galerkin time stepping.

    Satisfies ODEResidualProtocol so existing stepper residuals
    (ForwardEulerHVP, HeunHVP, future RK4, etc.) work unchanged.

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

    def __init__(
        self,
        physics: GalerkinPhysicsProtocol[Array],
        lumped_mass: bool = False,
    ) -> None:
        self._physics = physics
        self._bkd = physics.bkd()
        self._time: float = 0.0
        self._lumped_mass = lumped_mass

        # Cache Dirichlet DOF indices (assumed constant in time)
        d_dofs, _ = physics.dirichlet_dof_info(0.0)
        self._d_dof_indices_np = self._bkd.to_numpy(d_dofs).astype(np.intp)

        # Build BC-modified mass matrix and factor/lump
        self._setup_mass(physics)

        # Effective mass is identity (real M absorbed into f)
        nstates = physics.mass_matrix().shape[0]
        self._mass = IdentityMassMatrix(nstates, self._bkd)

    def _setup_mass(self, physics: GalerkinPhysicsProtocol[Array]) -> None:
        """Build and factor the BC-modified mass matrix, preserving sparsity."""
        from scipy.sparse import csc_matrix, lil_matrix

        M_raw = physics.mass_matrix()
        d_dofs = self._d_dof_indices_np

        if self._lumped_mass:
            # Lumped mass: row-sum diagonal, Dirichlet rows already sum to 1.0
            if isinstance(M_raw, spmatrix):
                M_np = np.asarray(M_raw.toarray())
            else:
                M_np = self._bkd.to_numpy(M_raw).copy()
            if len(d_dofs) > 0:
                M_np[d_dofs, :] = 0.0
                M_np[d_dofs, d_dofs] = 1.0
            self._m_diag_np: Optional[np.ndarray] = M_np.sum(axis=1)
            self._m_bc: Optional[MassMatrixProtocol[Array]] = None
        else:
            # Consistent mass: use create_mass_matrix for cached factorization
            self._m_diag_np = None
            if isinstance(M_raw, spmatrix):
                M_mod = lil_matrix(M_raw)
                if len(d_dofs) > 0:
                    M_mod[d_dofs, :] = 0.0
                    M_mod[d_dofs, d_dofs] = 1.0
                self._m_bc = create_mass_matrix(csc_matrix(M_mod), self._bkd)
            else:
                M_np = self._bkd.to_numpy(M_raw).copy()
                if len(d_dofs) > 0:
                    M_np[d_dofs, :] = 0.0
                    M_np[d_dofs, d_dofs] = 1.0
                self._m_bc = create_mass_matrix(
                    self._bkd.asarray(M_np), self._bkd
                )

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
            if self._m_diag_np is None:
                raise RuntimeError("lumped mass diagonal not initialized")
            f_np = F_np / self._m_diag_np
            return self._bkd.asarray(f_np, dtype=self._bkd.double_dtype())
        if self._m_bc is None:
            raise RuntimeError("BC-modified mass matrix not initialized")
        return self._m_bc.solve(
            self._bkd.asarray(F_np, dtype=self._bkd.double_dtype())
        )

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
        if isinstance(J_F, spmatrix):
            J_np = J_F.toarray().copy()
        else:
            J_np = self._bkd.to_numpy(J_F).copy()

        if self._lumped_mass:
            if self._m_diag_np is None:
                raise RuntimeError("lumped mass diagonal not initialized")
            result_np = J_np / self._m_diag_np[:, None]
            return self._bkd.asarray(
                result_np, dtype=self._bkd.double_dtype()
            )
        if self._m_bc is None:
            raise RuntimeError("BC-modified mass matrix not initialized")
        J_arr = self._bkd.asarray(J_np, dtype=self._bkd.double_dtype())
        return self._m_bc.solve(J_arr)

    def mass_matrix(self) -> MassMatrixProtocol[Array]:
        """Return identity mass matrix (real M absorbed into f)."""
        return self._mass

    def __repr__(self) -> str:
        return (
            f"GalerkinExplicitODEAdapter("
            f"physics={self._physics!r}, "
            f"lumped_mass={self._lumped_mass})"
        )
