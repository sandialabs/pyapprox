"""Mesh-based Karhunen-Loève Expansion using kernel matrices."""

from typing import Generic, Optional, Union

import numpy as np

from pyapprox.surrogates.kernels.protocols import KernelProtocol
from pyapprox.surrogates.kle.utils import (
    eigendecomposition_unweighted,
    eigendecomposition_weighted,
)
from pyapprox.util.backends.protocols import Array, Backend


class MeshKLE(Generic[Array]):
    """Karhunen-Loève Expansion computed from a kernel on mesh coordinates.

    Given mesh coordinates and a kernel, computes the KLE basis by
    eigendecomposition of the kernel matrix K(x_i, x_j).

    The expansion is:
        f(x) = mean(x) + sigma * sum_{i=1}^{nterms} sqrt(lambda_i) * phi_i(x) * z_i

    Memory
    ------
    The kernel matrix is dense with shape ``(ncoords, ncoords)``.
    For large meshes this can be the dominant memory cost.  A mesh
    with N = 10,000 coordinates produces a 10K x 10K matrix (~800 MB
    in float64); N = 40,000 (e.g. all quadrature points of a 10K-
    element mesh) requires ~12 GB.  Prefer using mesh nodes or element
    centroids as collocation points and interpolating to quadrature
    points afterwards. See ``pde.field_maps.kle_factory`` for
    memory-efficient FEM-aware constructors.

    Quadrature weights
    ------------------
    Quadrature weights (or lumped-mass row sums) must be included when
    discretizing the Fredholm eigenvalue problem; omitting them yields
    eigenvalues that lack physical variance units and eigenvectors
    biased toward regions of mesh refinement.  The symmetric weighting
    trick

        C_tilde_{ij} = sqrt(w_i) * C(x_i, x_j) * sqrt(w_j)

    reduces the generalized eigenproblem to a standard one while
    preserving correct L^2 orthogonality.

    Parameters
    ----------
    mesh_coords : Array, shape (nphys_vars, ncoords)
        Spatial coordinates of the mesh points.
    kernel : KernelProtocol[Array]
        Kernel object satisfying KernelProtocol. Called as
        kernel(mesh_coords, mesh_coords) to build the covariance matrix.
    sigma : float
        Variance scaling factor applied to eigenvectors.
    mean_field : float or Array
        Mean field. Scalar is broadcast to all coordinates.
    use_log : bool
        If True, return exp(mean + basis @ coef) instead of
        mean + basis @ coef.
    nterms : int or None
        Number of KLE terms. None uses all mesh points. When
        nterms < ncoords, a partial eigensolve (scipy eigsh) is used
        for O(N*k) cost instead of O(N^3). This converts to NumPy
        internally, so the Torch autograd graph is not preserved
        through the eigendecomposition. This is acceptable because
        KLE basis construction is a one-time setup cost.
    quad_weights : Array or None, shape (ncoords,)
        Quadrature weights for weighted eigendecomposition.  Should be
        provided whenever the collocation points come from a
        non-uniform discretization (FEM nodes, quadrature points, etc.).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        mesh_coords: Array,
        kernel: KernelProtocol[Array],
        sigma: float = 1.0,
        mean_field: Union[float, Array] = 0.0,
        use_log: bool = False,
        nterms: Optional[int] = None,
        quad_weights: Optional[Array] = None,
        bkd: Backend[Array] = None,
    ):
        if bkd is None:
            raise ValueError("bkd must be provided")
        if not isinstance(kernel, KernelProtocol):
            raise TypeError(
                f"kernel must satisfy KernelProtocol, got {type(kernel).__name__}"
            )
        self._bkd = bkd
        self._mesh_coords = mesh_coords
        self._kernel = kernel
        self._sigma = sigma
        self._use_log = use_log
        self._quad_weights = quad_weights
        if quad_weights is not None and quad_weights.ndim != 1:
            raise ValueError(f"quad_weights must be 1D, got ndim={quad_weights.ndim}")

        # Set mean field
        ncoords = mesh_coords.shape[1]
        if np.isscalar(mean_field):
            self._mean_field = bkd.full((ncoords,), 1) * mean_field
        else:
            self._mean_field = mean_field

        # Set nterms
        if nterms is None:
            nterms = ncoords
        if nterms > ncoords:
            raise ValueError(f"nterms={nterms} exceeds ncoords={ncoords}")
        self._nterms = nterms

        # Compute basis
        self._compute_basis()

    def _compute_basis(self) -> None:
        """Compute the KLE basis via eigendecomposition of the kernel matrix."""
        # Build kernel matrix using the kernel's __call__
        K = self._kernel(self._mesh_coords, self._mesh_coords)

        if self._quad_weights is None:
            eig_vals, eig_vecs = eigendecomposition_unweighted(
                K, self._nterms, self._bkd
            )
        else:
            eig_vals, eig_vecs = eigendecomposition_weighted(
                K, self._quad_weights, self._nterms, self._bkd
            )

        eig_vals = self._bkd.maximum(eig_vals, self._bkd.asarray([0.0]))
        self._sqrt_eig_vals = self._bkd.sqrt(eig_vals)
        self._unweighted_eig_vecs = eig_vecs
        # Pre-multiply by sqrt(eigenvalues) and sigma
        self._eig_vecs = eig_vecs * self._sqrt_eig_vals * self._sigma

    def __call__(self, coef: Array) -> Array:
        """Evaluate the KLE at given coefficients.

        Parameters
        ----------
        coef : Array, shape (nterms, nsamples)
            Random coefficients for each sample.

        Returns
        -------
        Array, shape (ncoords, nsamples)
            Field values at mesh coordinates for each sample.
        """
        if coef.ndim != 2:
            raise ValueError(f"coef.ndim={coef.ndim} but should be 2")
        if coef.shape[0] != self._nterms:
            raise ValueError(f"coef.shape[0]={coef.shape[0]} != nterms={self._nterms}")
        if self._use_log:
            return self._bkd.exp(self._mean_field[:, None] + self._eig_vecs @ coef)
        return self._mean_field[:, None] + self._eig_vecs @ coef

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nterms(self) -> int:
        """Return the number of KLE terms."""
        return self._nterms

    def nvars(self) -> int:
        """Return the number of KLE terms (alias for nterms)."""
        return self._nterms

    def eigenvectors(self) -> Array:
        """Return unweighted eigenvectors, shape (ncoords, nterms)."""
        return self._unweighted_eig_vecs

    def weighted_eigenvectors(self) -> Array:
        """Return eigenvectors scaled by sqrt(eigenvalues) * sigma.

        Shape (ncoords, nterms).
        """
        return self._eig_vecs

    def eigenvalues(self) -> Array:
        """Return eigenvalues, shape (nterms,)."""
        return self._sqrt_eig_vals**2

    def mean_field(self) -> Array:
        """Return the mean field, shape (ncoords,)."""
        return self._mean_field

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nterms={self._nterms}, sigma={self._sigma})"
