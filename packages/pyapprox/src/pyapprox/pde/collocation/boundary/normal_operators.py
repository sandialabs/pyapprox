"""Normal operators for Robin/Neumann boundary conditions.

Provides operators that compute normal-direction terms at boundary points:
- GradientNormalOperator: grad(u) . n using derivative matrices
- FluxNormalOperator: flux(u) . n using a FluxProviderProtocol
- TractionNormalOperator: one component of σ·n for linear elasticity
"""

from typing import Generic, List, Protocol, runtime_checkable

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
                normals[:, d : d + 1] * derivative_matrices[d][boundary_indices, :]
            )
        self._normal_deriv_matrix = normal_deriv_matrix

    def normals(self) -> Array:
        """Return outward unit normals. Shape: (nboundary_pts, ndim)."""
        return self._normals

    def has_coefficient_dependence(self) -> bool:
        """Return False: grad(u)·n does not depend on a parameterized coefficient."""
        return False

    def is_linear_in_state(self) -> bool:
        """Return True: grad(u)·n is a fixed matrix applied to the state."""
        return True

    def is_time_dependent(self) -> bool:
        """Return False: grad(u)·n reads no coefficient."""
        return False

    def __call__(self, state: Array, time: float) -> Array:
        """Compute grad(u) . n at boundary points.

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (npts,)
        time : float
            Unused: grad(u) . n has no coefficient.

        Returns
        -------
        Array
            Normal derivative at boundary points. Shape: (nboundary_pts,)
        """
        return self._normal_deriv_matrix @ state

    def jacobian(self, state: Array, time: float) -> Array:
        """Return Jacobian of grad(u) . n w.r.t. state.

        For the gradient operator this is state-independent (linear operator).

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (npts,) (unused for linear operator)
        time : float
            Unused: grad(u) . n has no coefficient.

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

    def is_linear_in_state(self) -> bool:
        """Whether the flux provider declares the flux linear in the
        state."""
        return self._flux_provider.flux_is_linear_in_state()

    def is_time_dependent(self) -> bool:
        """Whether the flux provider declares the flux time-dependent."""
        return self._flux_provider.flux_is_time_dependent()

    def __call__(self, state: Array, time: float) -> Array:
        """Compute flux(u) . n at boundary points.

        Parameters
        ----------
        state : Array
            Full solution vector, all components stacked. Shape: (nstates,)
        time : float
            Time at which the flux coefficients are evaluated.

        Returns
        -------
        Array
            Flux dot normal at boundary points. Shape: (nboundary_pts,)
        """
        bkd = self._bkd
        flux_components = self._flux_provider.compute_flux(state, time)
        result = bkd.zeros((self._nboundary_pts,))
        for d, flux_d in enumerate(flux_components):
            result = result + flux_d[self._boundary_indices] * self._normals[:, d]
        return result

    def jacobian(self, state: Array, time: float) -> Array:
        """Return Jacobian of flux(u) . n w.r.t. state.

        Parameters
        ----------
        state : Array
            Full solution vector, all components stacked. Shape: (nstates,)
        time : float
            Time at which the flux coefficients are evaluated.

        Returns
        -------
        Array
            Jacobian, one column per state entry (``nstates`` = ``npts``
            for a scalar physics). Shape: (nboundary_pts, nstates)
        """
        bkd = self._bkd
        flux_jac_components = self._flux_provider.compute_flux_jacobian(state, time)
        nstates = flux_jac_components[0].shape[1]
        result = bkd.zeros((self._nboundary_pts, nstates))
        # Vectorized row gather (per-row python loops are slow on torch)
        for d, jac_d in enumerate(flux_jac_components):
            result = result + (
                self._normals[:, d : d + 1] * jac_d[self._boundary_indices, :]
            )
        return result


@runtime_checkable
class CauchyStressSourceProtocol(Protocol, Generic[Array]):
    """What the traction operator needs: the Cauchy stress at every node
    and its state Jacobian, at a time. A linear elasticity physics
    satisfies it, so the traction reads the material the physics holds
    when it is evaluated."""

    def stress(self, state: Array, time: float) -> List[List[Array]]:
        """``sigma[i][j]`` at every node, each shape (npts,)."""
        ...

    def stress_jacobian(self, state: Array, time: float) -> List[List[Array]]:
        """``dsigma[i][j]/du``, each shape (npts, nstates)."""
        ...

    def stress_is_time_dependent(self) -> bool:
        """Whether any coefficient the stress reads is declared
        time-dependent."""
        ...

    def stress_is_linear_in_state(self) -> bool:
        """Whether the stress is linear in the state, so its Jacobian is
        the same at every state."""
        ...


class TractionNormalOperator(Generic[Array]):
    """Computes one component of traction t = σ·n at boundary points.

    The traction component i is :math:`t_i = \\sum_j \\sigma_{ij} n_j`,
    with :math:`\\sigma` taken from the stress source at evaluation time,
    so a material the parameterization changes reaches the boundary rows
    as it reaches the interior.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    mesh_boundary_indices : Array
        **Mesh point indices** (0..npts-1) on boundary. Shape: (nboundary,)
        These index into the stress fields, NOT into the component-stacked
        state vector. The factory function handles the state-index offset.
    normals : Array
        Outward unit normal vectors at boundary points.
        Shape: (nboundary, ndim)
    stress_source : CauchyStressSourceProtocol
        The Cauchy stress and its Jacobian, usually the physics.
    component : int
        The traction component i returned.

    Raises
    ------
    TypeError
        If ``stress_source`` does not satisfy ``CauchyStressSourceProtocol``.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        mesh_boundary_indices: Array,
        normals: Array,
        stress_source: CauchyStressSourceProtocol[Array],
        component: int,
    ) -> None:
        if not isinstance(stress_source, CauchyStressSourceProtocol):
            raise TypeError(
                "stress_source must satisfy CauchyStressSourceProtocol, got "
                f"{type(stress_source).__name__}"
            )
        self._bkd = bkd
        self._mesh_boundary_indices = mesh_boundary_indices
        self._normals = normals
        self._stress_source = stress_source
        self._component = component

    def normals(self) -> Array:
        """Return outward unit normals. Shape: (nboundary, ndim)."""
        return self._normals

    def has_coefficient_dependence(self) -> bool:
        """Return True: the traction depends on the Lame parameters."""
        return True

    def is_linear_in_state(self) -> bool:
        """Whether the stress source declares the stress linear in the
        state."""
        return self._stress_source.stress_is_linear_in_state()

    def is_time_dependent(self) -> bool:
        """Whether the stress source declares the stress time-dependent."""
        return self._stress_source.stress_is_time_dependent()

    def __call__(self, state: Array, time: float) -> Array:
        """Compute one component of traction t = σ·n at boundary points.

        Parameters
        ----------
        state : Array
            Full component-stacked solution. Shape: (nstates,)
        time : float
            The time at which the stress source reads its material.

        Returns
        -------
        Array
            Traction component at boundary points. Shape: (nboundary,)
        """
        idx = self._mesh_boundary_indices
        stress_row = self._stress_source.stress(state, time)[self._component]
        traction = stress_row[0][idx] * self._normals[:, 0]
        for j in range(1, len(stress_row)):
            traction = traction + stress_row[j][idx] * self._normals[:, j]
        return traction

    def jacobian(self, state: Array, time: float) -> Array:
        """Return Jacobian of traction component w.r.t. state:
        :math:`\\sum_j n_j \\, \\partial \\sigma_{ij}/\\partial u` at the
        boundary rows.

        Parameters
        ----------
        state : Array
            Full solution vector. Shape: (nstates,)
        time : float
            The time at which the stress source reads its material.

        Returns
        -------
        Array
            Jacobian. Shape: (nboundary, nstates)
        """
        idx = self._mesh_boundary_indices
        row = self._stress_source.stress_jacobian(state, time)[self._component]
        jac = self._normals[:, 0:1] * row[0][idx]
        for j in range(1, len(row)):
            jac = jac + self._normals[:, j : j + 1] * row[j][idx]
        return jac
