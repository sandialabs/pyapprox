"""Periodic boundary conditions for spectral collocation methods.

Enforces u(x_left) = u(x_right) on paired boundaries.
"""

from typing import Generic

from pyapprox.typing.util.backends.protocols import Array, Backend


class PeriodicBC(Generic[Array]):
    """Periodic boundary condition.

    Enforces that solution values match at paired boundary points.
    Typically used to enforce u(x_left) = u(x_right).

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
    """

    def __init__(
        self,
        bkd: Backend[Array],
        boundary_indices: Array,
        partner_indices: Array,
    ):
        self._bkd = bkd
        self._boundary_indices = boundary_indices
        self._partner_indices = partner_indices
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

    def partner_indices(self) -> Array:
        """Return indices of mesh points on the partner boundary."""
        return self._partner_indices

    def apply_to_residual(
        self, residual: Array, state: Array, time: float
    ) -> Array:
        """Apply periodic BC to residual.

        Sets residual at primary boundary points to:
        u(primary) - u(partner)

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
        residual = self._bkd.copy(residual)

        for i in range(self._nboundary_pts):
            residual[idx1[i]] = state[idx1[i]] - state[idx2[i]]
        return residual

    def apply_to_jacobian(
        self, jacobian: Array, state: Array, time: float
    ) -> Array:
        """Apply periodic BC to Jacobian.

        Sets Jacobian rows at primary boundary:
        J[idx1, idx1] = 1, J[idx1, idx2] = -1, rest = 0

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
        jacobian = self._bkd.copy(jacobian)
        nstates = jacobian.shape[0]

        for i in range(self._nboundary_pts):
            # Zero out row
            for j in range(nstates):
                jacobian[idx1[i], j] = 0.0
            # Set coefficients
            jacobian[idx1[i], idx1[i]] = 1.0
            jacobian[idx1[i], idx2[i]] = -1.0
        return jacobian

    def apply_to_param_jacobian(
        self, param_jacobian: Array, state: Array, time: float
    ) -> Array:
        """Apply periodic BC to parameter Jacobian.

        Sets parameter Jacobian rows at primary boundary to zero.

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
        param_jacobian = self._bkd.copy(param_jacobian)
        nparams = param_jacobian.shape[1]
        for i in range(self._nboundary_pts):
            for j in range(nparams):
                param_jacobian[idx1[i], j] = 0.0
        return param_jacobian
