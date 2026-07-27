"""Periodic boundary conditions for spectral collocation methods.

Enforces u(left) = u(right) AND u'(left) = u'(right) on paired boundaries.

For a second-order PDE with N collocation points, the PDE provides N-2
interior equations. Two boundary equations are needed to close the system:
  - Value matching: u(left) = u(right)    [replaces primary boundary row]
  - Derivative matching: u'(left) = u'(right) [replaces partner boundary row]

The derivative matrix D for the periodic direction must be supplied.
"""

from typing import Generic, Optional

from pyapprox.pde.collocation.protocols.boundary import (
    BCPhysicalSensitivities,
)
from pyapprox.util.backends.protocols import Array, Backend


class PeriodicBC(Generic[Array]):
    """Periodic boundary condition with value and derivative matching.

    For second-order PDEs, enforces both:
      u(primary) = u(partner)       at primary boundary rows
      u'(primary) = u'(partner)     at partner boundary rows

    The derivative matrix for the periodic direction is required to compute
    the derivative matching condition.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of mesh points on the primary boundary (e.g., left).
        Shape: (nboundary_pts,)
    partner_indices : Array
        Indices of mesh points on the partner boundary (e.g., right).
        Shape: (nboundary_pts,)
        Must have same length as boundary_indices.
    derivative_matrix : Array
        Derivative matrix for the periodic direction.
        Shape: (nstates, nstates). For 1D this is D; for 2D periodic
        in x use D_x, for periodic in y use D_y.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        boundary_indices: Array,
        partner_indices: Array,
        derivative_matrix: Array,
    ):
        self._bkd = bkd
        self._boundary_indices = boundary_indices
        self._partner_indices = partner_indices
        self._D = derivative_matrix
        self._nboundary_pts = boundary_indices.shape[0]

        if partner_indices.shape[0] != self._nboundary_pts:
            raise ValueError(
                f"partner_indices length {partner_indices.shape[0]} must match "
                f"boundary_indices length {self._nboundary_pts}"
            )

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def boundary_indices(self) -> Array:
        """Return indices of mesh points on the primary boundary."""
        return self._boundary_indices

    def is_essential(self) -> bool:
        """Return True: periodic BCs directly constrain DOF values."""
        return True

    def partner_indices(self) -> Array:
        """Return indices of mesh points on the partner boundary."""
        return self._partner_indices

    def apply_to_residual(self, residual: Array, state: Array, time: float) -> Array:
        """Apply periodic BC to residual.

        Primary boundary rows: u(primary) - u(partner)
        Partner boundary rows: D[primary, :] @ u - D[partner, :] @ u

        Parameters
        ----------
        residual : Array
            Residual vector. Shape: (nstates,)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified residual. Shape: (nstates,)
        """
        idx1 = self._boundary_indices
        idx2 = self._partner_indices
        D = self._D
        residual = self._bkd.copy(residual)

        # Value matching at primary boundary
        residual[idx1] = state[idx1] - state[idx2]
        # Derivative matching at partner boundary
        residual[idx2] = (D[idx1, :] - D[idx2, :]) @ state
        return residual

    def apply_to_jacobian(self, jacobian: Array, state: Array, time: float) -> Array:
        """Apply periodic BC to Jacobian.

        Primary boundary rows: J[idx1, idx1] = 1, J[idx1, idx2] = -1
        Partner boundary rows: J[idx2, :] = D[idx1, :] - D[idx2, :]

        Parameters
        ----------
        jacobian : Array
            Jacobian matrix. Shape: (nstates, nstates)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified Jacobian. Shape: (nstates, nstates)
        """
        idx1 = self._boundary_indices
        idx2 = self._partner_indices
        D = self._D
        jacobian = self._bkd.copy(jacobian)

        # Vectorized row replacement (per-element writes are slow on
        # torch). Value matching rows (primary boundary):
        jacobian[idx1, :] = 0.0
        jacobian[idx1, idx1] = 1.0
        jacobian[idx1, idx2] = -1.0

        # Derivative matching rows (partner boundary)
        jacobian[idx2, :] = D[idx1, :] - D[idx2, :]
        return jacobian

    def apply_to_param_jacobian(
        self,
        param_jacobian: Array,
        state: Array,
        time: float,
        physical_sensitivities: Optional[
            BCPhysicalSensitivities[Array]
        ] = None,
    ) -> Array:
        """Apply periodic BC to parameter Jacobian.

        Sets parameter Jacobian rows at both primary and partner
        boundaries to zero (BCs do not depend on parameters).

        Parameters
        ----------
        param_jacobian : Array
            Parameter Jacobian. Shape: (nstates, nparams)
        state : Array
            Current solution. Shape: (nstates,)
        time : float
            Current time.

        Returns
        -------
        Array
            Modified parameter Jacobian. Shape: (nstates, nparams)
        """
        idx1 = self._boundary_indices
        idx2 = self._partner_indices
        param_jacobian = self._bkd.copy(param_jacobian)
        param_jacobian[idx1, :] = 0.0
        param_jacobian[idx2, :] = 0.0
        return param_jacobian
