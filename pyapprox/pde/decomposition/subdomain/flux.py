"""Flux computation utilities for domain decomposition.

Provides helper functions and classes for computing interface fluxes
in DtN domain decomposition.
"""

from typing import Generic

from pyapprox.pde.collocation.protocols.basis import BasisProtocol
from pyapprox.util.backends.protocols import Array, Backend


class FluxComputer(Generic[Array]):
    """Computes flux (gradient) at boundary points.

    For a scalar field u, computes the flux:
        flux = -D * grad(u) · n

    where D is the diffusion coefficient and n is the outward normal.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    basis : BasisProtocol
        Collocation basis (provides derivative matrices).
    diffusion : float, optional
        Diffusion coefficient. Default: 1.0
    """

    def __init__(
        self,
        bkd: Backend[Array],
        basis: BasisProtocol[Array],
        diffusion: float = 1.0,
    ):
        self._bkd = bkd
        self._basis = basis
        self._diffusion = diffusion
        self._ndim = basis.ndim()

        # Cache derivative matrices
        self._D_matrices = [
            basis.derivative_matrix(1, dim) for dim in range(self._ndim)
        ]

    def compute_gradient(self, state: Array) -> list[Any]:
        """Compute gradient of state at all mesh points.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)

        Returns
        -------
        list
            List of gradient components. Each has shape: (nstates,)
        """
        return [D @ state for D in self._D_matrices]

    def compute_flux_at_indices(
        self, state: Array, indices: Array, normal: Array
    ) -> Array:
        """Compute flux at specified mesh indices.

        Parameters
        ----------
        state : Array
            Solution state. Shape: (nstates,)
        indices : Array
            Mesh indices where flux is computed. Shape: (n_boundary,)
        normal : Array
            Outward normal vector. Shape: (ndim,)

        Returns
        -------
        Array
            Flux values. Shape: (n_boundary,)
        """
        # Compute gradient
        grad = self.compute_gradient(state)

        # Extract gradient at boundary points and compute normal flux
        n_boundary = indices.shape[0]
        flux = self._bkd.zeros((n_boundary,))

        for dim in range(self._ndim):
            grad_dim_boundary = grad[dim][indices]
            flux = flux + grad_dim_boundary * self._bkd.to_float(normal[dim])

        # Scale by diffusion coefficient (flux = -D * grad·n, but we return
        # grad·n since the sign convention depends on residual formulation)
        return self._diffusion * flux

    def compute_flux_derivative(self, indices: Array, normal: Array) -> Array:
        """Compute derivative of flux w.r.t. state (Jacobian).

        For flux = D * grad(u) · n, the Jacobian is:
            d(flux)/d(state) = D * sum_i n_i * D_i[indices, :]

        Parameters
        ----------
        indices : Array
            Mesh indices where flux is computed. Shape: (n_boundary,)
        normal : Array
            Outward normal vector. Shape: (ndim,)

        Returns
        -------
        Array
            Flux Jacobian. Shape: (n_boundary, nstates)
        """
        n_boundary = indices.shape[0]
        nstates = self._basis.npts()

        # Build flux Jacobian
        flux_jac = self._bkd.zeros((n_boundary, nstates))

        for dim in range(self._ndim):
            D_dim = self._D_matrices[dim]
            # Extract rows corresponding to boundary indices
            for i, idx in enumerate(indices):
                idx_int = self._bkd.to_int(idx)
                for j in range(nstates):
                    n_d = self._bkd.to_float(normal[dim])
                    flux_jac[i, j] = flux_jac[i, j] + n_d * D_dim[idx_int, j]

        return self._diffusion * flux_jac


def compute_flux_mismatch(
    flux_left: Array, flux_right: Array, bkd: Backend[Array]
) -> Array:
    """Compute flux mismatch across interface.

    For DtN decomposition, the transmission condition requires flux
    conservation: flux_left + flux_right = 0 (where normals point outward).

    Parameters
    ----------
    flux_left : Array
        Flux from left subdomain. Shape: (npts,)
    flux_right : Array
        Flux from right subdomain. Shape: (npts,)
    bkd : Backend
        Computational backend.

    Returns
    -------
    Array
        Flux mismatch. Shape: (npts,)
    """
    # Residual: flux_left + flux_right should be zero
    return flux_left + flux_right


def flux_mismatch_norm(
    flux_left: Array, flux_right: Array, bkd: Backend[Array]
) -> float:
    """Compute L2 norm of flux mismatch.

    Parameters
    ----------
    flux_left : Array
        Flux from left subdomain. Shape: (npts,)
    flux_right : Array
        Flux from right subdomain. Shape: (npts,)
    bkd : Backend
        Computational backend.

    Returns
    -------
    float
        L2 norm of flux mismatch.
    """
    mismatch = compute_flux_mismatch(flux_left, flux_right, bkd)
    return bkd.to_float(bkd.norm(mismatch))
