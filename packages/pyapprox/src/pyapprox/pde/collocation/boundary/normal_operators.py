"""Normal operators for Robin/Neumann boundary conditions.

Provides operators that compute normal-direction terms at boundary points:
- GradientNormalOperator: grad(u) . n using derivative matrices
- FluxNormalOperator: flux(u) . n using a FluxProviderProtocol
- TractionNormalOperator: one component of σ·n for 2D linear elasticity
"""

from typing import Generic, List

from pyapprox.pde.collocation.protocols.boundary import (
    FluxProviderProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


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
        for d in range(self._ndim):
            normal_deriv_matrix = normal_deriv_matrix + (
                normals[:, d : d + 1]
                * derivative_matrices[d][boundary_indices, :]
            )
        self._normal_deriv_matrix = normal_deriv_matrix

    def normals(self) -> Array:
        """Return outward unit normals. Shape: (nboundary_pts, ndim)."""
        return self._normals

    def has_coefficient_dependence(self) -> bool:
        """Return False: grad(u)·n does not depend on a parameterized coefficient."""
        return False

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

    def normals(self) -> Array:
        """Return outward unit normals. Shape: (nboundary_pts, ndim)."""
        return self._normals

    def has_coefficient_dependence(self) -> bool:
        """Return True: flux = -D·grad(u) depends on parameterized D."""
        return True

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
        # Vectorized row gather (per-row python loops are slow on torch)
        for d, jac_d in enumerate(flux_jac_components):
            result = result + (
                self._normals[:, d : d + 1]
                * jac_d[self._boundary_indices, :]
            )
        return result


class TractionNormalOperator(Generic[Array]):
    """Computes one component of traction t = σ·n at boundary points.

    For 2D linear elasticity with component-stacked state [u_x, u_y]:
        σ = λ*tr(ε)*I + 2μ*ε
        t = σ·n

    This operator returns t_x (component=0) or t_y (component=1).

    Note: 2D only. 3D would require a separate operator with 3 components
    and 6 independent stress components.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_boundary_indices : Array
        **Mesh point indices** (0..npts-1) on boundary. Shape: (nboundary,)
        These index into derivative matrices, NOT into the component-stacked
        state vector. The factory function handles the state-index offset.
    normals : Array
        Outward unit normal vectors at boundary points.
        Shape: (nboundary, 2)
    derivative_matrices : List[Array]
        [Dx, Dy], each shape (npts, npts). Physical derivative matrices.
    lamda : float
        Lamé's first parameter λ.
    mu : float
        Shear modulus μ.
    component : int
        0 for t_x, 1 for t_y.
    npts : int
        Number of mesh points (state length = 2*npts).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        mesh_boundary_indices: Array,
        normals: Array,
        derivative_matrices: List[Array],
        lamda: float,
        mu: float,
        component: int,
        npts: int,
    ):
        self._bkd = bkd
        self._mesh_boundary_indices = mesh_boundary_indices
        self._normals = normals
        self._npts = npts
        self._component = component

        Dx = derivative_matrices[0]
        Dy = derivative_matrices[1]
        lam = lamda
        lam_2mu = lamda + 2.0 * mu

        # Precompute traction Jacobian: (nboundary, 2*npts)
        # Traction is linear in state, so Jacobian is constant.
        #
        # For component 0 (t_x):
        #   d(t_x)/d(u) = nx*(λ+2μ)*Dx[idx,:] + ny*μ*Dy[idx,:]
        #   d(t_x)/d(v) = nx*λ*Dy[idx,:] + ny*μ*Dx[idx,:]
        #
        # For component 1 (t_y):
        #   d(t_y)/d(u) = nx*μ*Dy[idx,:] + ny*λ*Dx[idx,:]
        #   d(t_y)/d(v) = nx*μ*Dx[idx,:] + ny*(λ+2μ)*Dy[idx,:]
        nx = normals[:, 0:1]
        ny = normals[:, 1:2]
        dx_rows = Dx[mesh_boundary_indices, :]
        dy_rows = Dy[mesh_boundary_indices, :]
        if component == 0:
            # d(t_x)/d(u), d(t_x)/d(v)
            jac_u = nx * lam_2mu * dx_rows + ny * mu * dy_rows
            jac_v = nx * lam * dy_rows + ny * mu * dx_rows
        else:
            # d(t_y)/d(u), d(t_y)/d(v)
            jac_u = nx * mu * dy_rows + ny * lam * dx_rows
            jac_v = nx * mu * dx_rows + ny * lam_2mu * dy_rows

        self._jacobian = bkd.concatenate([jac_u, jac_v], axis=1)

    def normals(self) -> Array:
        """Return outward unit normals. Shape: (nboundary, 2)."""
        return self._normals

    def has_coefficient_dependence(self) -> bool:
        """Return False: traction uses fixed material constants."""
        return False

    def __call__(self, state: Array) -> Array:
        """Compute one component of traction t = σ·n at boundary points.

        Parameters
        ----------
        state : Array
            Full component-stacked solution [u_x, u_y]. Shape: (2*npts,)

        Returns
        -------
        Array
            Traction component at boundary points. Shape: (nboundary,)
        """
        return self._jacobian @ state

    def jacobian(self, state: Array) -> Array:
        """Return Jacobian of traction component w.r.t. state.

        State-independent (traction is linear in displacement).

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (2*npts,) (unused)

        Returns
        -------
        Array
            Jacobian. Shape: (nboundary, 2*npts)
        """
        return self._jacobian


