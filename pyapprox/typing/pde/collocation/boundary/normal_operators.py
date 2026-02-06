"""Normal operators for Robin/Neumann boundary conditions.

Provides operators that compute normal-direction terms at boundary points:
- GradientNormalOperator: grad(u) . n using derivative matrices
- FluxNormalOperator: flux(u) . n using a FluxProviderProtocol
- _LegacyNormalOperator: backward compatibility wrapper for (D, normal_sign) API
"""

from typing import Generic, List

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.collocation.protocols.boundary import (
    FluxProviderProtocol,
)


class GradientNormalOperator(Generic[Array]):
    """Computes grad(u) . n at boundary points.

    Uses physical derivative matrices and per-point outward normals
    to compute the directional derivative.

    grad(u) . n = sum_d (D_d @ u)[boundary_idx] * normals[:, d]

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of boundary points in the full mesh. Shape: (nboundary_pts,)
    normals : Array
        Outward unit normal vectors at boundary points.
        Shape: (nboundary_pts, ndim)
    derivative_matrices : List[Array]
        Physical derivative matrices (d/dx_d) for each dimension.
        Each shape: (npts, npts). These are FULL matrices, not
        boundary-extracted rows.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        boundary_indices: Array,
        normals: Array,
        derivative_matrices: List[Array],
    ):
        self._bkd = bkd
        self._boundary_indices = boundary_indices
        self._normals = normals
        self._derivative_matrices = derivative_matrices
        self._nboundary_pts = boundary_indices.shape[0]
        self._ndim = len(derivative_matrices)

        # Precompute normal derivative matrix: (nboundary_pts, npts)
        # normal_deriv_matrix[i, :] = sum_d normals[i, d] * D_d[idx[i], :]
        npts = derivative_matrices[0].shape[0]
        normal_deriv_matrix = bkd.zeros((self._nboundary_pts, npts))
        normal_deriv_matrix = bkd.copy(normal_deriv_matrix)
        for i in range(self._nboundary_pts):
            idx = int(boundary_indices[i])
            for d in range(self._ndim):
                normal_deriv_matrix[i, :] = (
                    normal_deriv_matrix[i, :]
                    + normals[i, d] * derivative_matrices[d][idx, :]
                )
        self._normal_deriv_matrix = normal_deriv_matrix

    def __call__(self, state: Array) -> Array:
        """Compute grad(u) . n at boundary points.

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (npts,)

        Returns
        -------
        Array
            Normal derivative at boundary points. Shape: (nboundary_pts,)
        """
        return self._normal_deriv_matrix @ state

    def jacobian(self, state: Array) -> Array:
        """Return Jacobian of grad(u) . n w.r.t. state.

        For the gradient operator this is state-independent (linear operator).

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (npts,) (unused for linear operator)

        Returns
        -------
        Array
            Jacobian. Shape: (nboundary_pts, npts)
        """
        return self._normal_deriv_matrix


class FluxNormalOperator(Generic[Array]):
    """Computes flux(u) . n at boundary points.

    Delegates flux computation to a FluxProviderProtocol and dots
    the result with boundary normals.

    flux(u) . n = sum_d flux_d(u)[boundary_idx] * normals[:, d]

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    boundary_indices : Array
        Indices of boundary points. Shape: (nboundary_pts,)
    normals : Array
        Outward unit normal vectors. Shape: (nboundary_pts, ndim)
    flux_provider : FluxProviderProtocol
        Object providing compute_flux() and compute_flux_jacobian().
    """

    def __init__(
        self,
        bkd: Backend[Array],
        boundary_indices: Array,
        normals: Array,
        flux_provider: FluxProviderProtocol[Array],
    ):
        self._bkd = bkd
        self._boundary_indices = boundary_indices
        self._normals = normals
        self._flux_provider = flux_provider
        self._nboundary_pts = boundary_indices.shape[0]

    def __call__(self, state: Array) -> Array:
        """Compute flux(u) . n at boundary points.

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (npts,)

        Returns
        -------
        Array
            Flux dot normal at boundary points. Shape: (nboundary_pts,)
        """
        bkd = self._bkd
        flux_components = self._flux_provider.compute_flux(state)
        result = bkd.zeros((self._nboundary_pts,))
        for d, flux_d in enumerate(flux_components):
            result = result + flux_d[self._boundary_indices] * self._normals[:, d]
        return result

    def jacobian(self, state: Array) -> Array:
        """Return Jacobian of flux(u) . n w.r.t. state.

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (npts,)

        Returns
        -------
        Array
            Jacobian. Shape: (nboundary_pts, npts)
        """
        bkd = self._bkd
        flux_jac_components = self._flux_provider.compute_flux_jacobian(state)
        npts = flux_jac_components[0].shape[0]
        result = bkd.zeros((self._nboundary_pts, npts))
        for d, jac_d in enumerate(flux_jac_components):
            for i in range(self._nboundary_pts):
                idx = int(self._boundary_indices[i])
                result[i, :] = (
                    result[i, :]
                    + self._normals[i, d] * jac_d[idx, :]
                )
        return result


class _LegacyNormalOperator(Generic[Array]):
    """Backward-compatible normal operator for (derivative_matrix, normal_sign).

    Wraps the old API where a single derivative matrix (already extracted
    for boundary rows) is multiplied by a scalar normal sign.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    derivative_matrix : Array
        Boundary-extracted derivative matrix rows.
        Shape: (nboundary_pts, npts)
    normal_sign : float
        Sign of outward normal (+1 or -1).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        derivative_matrix: Array,
        normal_sign: float,
    ):
        self._bkd = bkd
        self._derivative_matrix = derivative_matrix
        self._normal_sign = normal_sign

    def __call__(self, state: Array) -> Array:
        """Compute normal_sign * D @ state.

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (npts,)

        Returns
        -------
        Array
            Normal derivative at boundary points. Shape: (nboundary_pts,)
        """
        return self._normal_sign * (self._derivative_matrix @ state)

    def jacobian(self, state: Array) -> Array:
        """Return normal_sign * D.

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (npts,) (unused)

        Returns
        -------
        Array
            Jacobian. Shape: (nboundary_pts, npts)
        """
        return self._normal_sign * self._derivative_matrix
