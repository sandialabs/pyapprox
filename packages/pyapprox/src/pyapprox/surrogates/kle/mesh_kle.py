"""Mesh-based Karhunen-Loève Expansion using kernel matrices."""

from typing import Generic, Optional, Union

import numpy as np

from pyapprox.surrogates.kernels.protocols import KernelProtocol
from pyapprox.surrogates.kle.eigensolvers import (
    DenseEigenSolver,
    KLEEigenSolverProtocol,
    usable_nterms,
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
        Number of KLE terms. None uses all mesh points. Under the
        default solver, nterms < ncoords takes a partial eigensolve
        (scipy eigsh) costing O(N*k) instead of O(N^3). That converts
        to NumPy internally, so the Torch autograd graph is not
        preserved through the eigendecomposition, which is acceptable
        because KLE basis construction is a one-time setup cost.
    quad_weights : Array or None, shape (ncoords,)
        Quadrature weights for weighted eigendecomposition.  Should be
        provided whenever the collocation points come from a
        non-uniform discretization (FEM nodes, quadrature points, etc.).
    eigensolver : KLEEigenSolverProtocol or None
        How the eigenpairs are computed. Defaults to
        :class:`DenseEigenSolver`, which assembles the kernel matrix and
        so costs O(N^2) memory -- the ceiling on problem size.
        :class:`PivotedCholeskyEigenSolver` and
        :class:`RandomizedEigenSolver` never form the matrix and lift
        that ceiling.

        The default is dense so that existing callers are unaffected.
        That does mean the users who most need a matrix-free solver are
        the ones who get dense assembly unless they know to ask.
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
        eigensolver: Optional[KLEEigenSolverProtocol[Array]] = None,
        bkd: Backend[Array] = None,
    ):
        if bkd is None:
            raise ValueError("bkd must be provided")
        if not isinstance(kernel, KernelProtocol):
            raise TypeError(
                f"kernel must satisfy KernelProtocol, got {type(kernel).__name__}"
            )
        if eigensolver is not None and not isinstance(
            eigensolver, KLEEigenSolverProtocol
        ):
            raise TypeError(
                "eigensolver must satisfy KLEEigenSolverProtocol, got "
                f"{type(eigensolver).__name__}"
            )
        self._eigensolver: KLEEigenSolverProtocol[Array] = (
            eigensolver if eigensolver is not None else DenseEigenSolver(bkd)
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
        if nterms is not None and nterms > ncoords:
            raise ValueError(f"nterms={nterms} exceeds ncoords={ncoords}")
        self._nterms = (
            self._usable_nterms() if nterms is None else nterms
        )

        # Compute basis
        self._compute_basis()

    def _usable_nterms(self) -> int:
        """Terms the kernel can supply at these coordinates.

        ``nterms=None`` used to mean ``ncoords``, which is almost never
        what a caller wants: smooth kernels are severely rank
        deficient, so most of those terms carried no variance and
        entered the basis as columns of zeros. On a squared exponential
        at lengthscale 0.3 with 60 points, 45 of the 60 were empty.

        Resolving the count needs the whole spectrum but not its
        eigenvectors, so this is an eigenvalue-only solve rather than a
        second full decomposition. It runs only when the caller
        declines to say how many terms they want.
        """
        kmat = self._kernel(self._mesh_coords, self._mesh_coords)
        if self._quad_weights is not None:
            sqrt_weights = self._bkd.sqrt(self._quad_weights)
            kmat = (sqrt_weights[:, None] * kmat) * sqrt_weights[None, :]
        nusable = usable_nterms(self._bkd.eigvalsh(kmat), self._bkd)
        if nusable < 1:
            raise ValueError(
                "the kernel supplies no modes carrying variance at these "
                "coordinates, so no KLE basis exists"
            )
        return nusable

    def _compute_basis(self) -> None:
        """Compute the KLE basis from the configured eigensolver.

        The weighted and unweighted cases are no longer distinguished
        here: the solver takes the quadrature weights and returns
        eigenpairs already in the unweighted convention, so the
        un-weighting, clipping, sorting and sign fixing happen once in
        one place rather than being re-derived per call site.
        """
        eig_vals, eig_vecs = self._eigensolver.solve(
            self._kernel,
            self._mesh_coords,
            self._nterms,
            quad_weights=self._quad_weights,
        )
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
