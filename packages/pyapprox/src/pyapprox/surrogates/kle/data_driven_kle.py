"""Data-driven Karhunen-Loève Expansion using SVD of field samples."""

from typing import Generic, Optional, Union

import numpy as np

from pyapprox.surrogates.kle.snapshot_eigensolvers import (
    SnapshotEigenSolverProtocol,
    default_snapshot_eigensolver,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    InnerProductProtocol,
)


class DataDrivenKLE(Generic[Array]):
    """Karhunen-Loève Expansion computed from field sample data.

    Decomposes the field samples directly rather than forming a sample
    covariance and eigendecomposing that, which would square the
    condition number. How the decomposition is done is the injected
    eigensolver's business; what stays here is the sample-covariance
    scaling, which needs to know these columns are samples.

    Parameters
    ----------
    field_samples : Array, shape (ncoords, nsamples)
        Field realizations at mesh coordinates.
    mean_field : float or Array
        Mean field. Scalar is broadcast to all coordinates.
    use_log : bool
        If True, return exp(mean + basis @ coef).
    nterms : int or None
        Number of KLE terms. None uses min(ncoords, nsamples).
    quad_weights : Array or None, shape (ncoords,)
        Quadrature weights, the diagonal of the metric the basis is
        orthonormal in. Cannot be combined with ``metric``, which says
        the same thing more generally.
    bkd : Backend[Array]
        Computational backend.
    metric : InnerProductProtocol, optional
        The inner product the eigenvectors are orthonormal in. Prefer
        this to ``quad_weights``: an assembled FEM mass matrix is a
        metric that no vector of weights can express.
    eigensolver : SnapshotEigenSolverProtocol, optional
        How the basis is extracted. Defaults to the solver matching the
        metric -- the SVD when it is diagonal, the method of snapshots
        otherwise -- mirroring the seam ``MeshKLE`` offers for the
        kernel-driven case.
    """

    def __init__(
        self,
        field_samples: Array,
        mean_field: Union[float, Array] = 0.0,
        use_log: bool = False,
        nterms: Optional[int] = None,
        quad_weights: Optional[Array] = None,
        bkd: Backend[Array] = None,
        metric: Optional[InnerProductProtocol[Array]] = None,
        eigensolver: Optional[SnapshotEigenSolverProtocol[Array]] = None,
    ):
        if bkd is None:
            raise ValueError("bkd must be provided")
        self._bkd = bkd
        self._field_samples = field_samples
        self._use_log = use_log
        self._quad_weights = quad_weights
        if quad_weights is not None and quad_weights.ndim != 1:
            raise ValueError(f"quad_weights must be 1D, got ndim={quad_weights.ndim}")

        # quad_weights is the diagonal special case of metric. Promote it
        # rather than carrying two representations of one concept; both
        # at once would leave which of them the basis honors ambiguous.
        if quad_weights is not None and metric is not None:
            raise ValueError(
                "pass either quad_weights or metric, not both: "
                "quad_weights is the diagonal case of metric, and "
                "supplying both leaves it undefined which the basis is "
                "orthonormal in"
            )
        if quad_weights is not None:
            metric = DiagonalInnerProduct(quad_weights, bkd)
        self._metric = metric
        self._eigensolver = (
            default_snapshot_eigensolver(bkd, metric)
            if eigensolver is None
            else eigensolver
        )

        # Set mean field
        ncoords = field_samples.shape[0]
        if np.isscalar(mean_field):
            self._mean_field = bkd.full((ncoords,), 1) * mean_field
        else:
            self._mean_field = mean_field

        # Set nterms. The binding limit is the rank of the sample matrix,
        # min(ncoords, nsamples), not ncoords alone: no decomposition
        # yields more than that many nonzero modes, so asking for more
        # returns columns of zeros -- modes the caller asked for and did
        # not get. Callers who centered their data lose one further
        # term, since subtracting the sample mean makes the columns
        # linearly dependent. That one is left to the solver's spectrum
        # check rather than counted here, because this class is handed a
        # matrix without being told whether it was centered.
        nsamples = int(field_samples.shape[1])
        max_nterms = min(int(ncoords), nsamples)
        if nterms is None:
            nterms = max_nterms
        if nterms > max_nterms:
            raise ValueError(
                f"nterms={nterms} exceeds the rank of the sample matrix, "
                f"min(ncoords={ncoords}, nsamples={nsamples})="
                f"{max_nterms}"
            )
        if nterms < 1:
            raise ValueError(f"nterms={nterms} must be positive")
        self._nterms = nterms

        self._compute_basis()

    def _compute_basis(self) -> None:
        """Extract the basis through the injected eigensolver.

        The decomposition itself lives in the solver, which is what lets
        a caller who cannot symmetrize -- an assembled FEM mass matrix
        has no cheap square root -- swap in the method of snapshots
        without this class knowing. What stays here is the KLE's own
        convention: the sample-covariance scaling, which depends on
        knowing these are samples rather than an arbitrary matrix.

        C = A A^T / (n-1)  (sample covariance)
        A = U S V^T  =>  C = U S^2 U^T / (n-1)
        so the eigenvalues of C are S^2/(n-1) and its eigenvectors U.
        """
        bkd = self._bkd
        eig_vals, eig_vecs = self._eigensolver.solve(
            self._field_samples, self._nterms, self._metric
        )
        # Solvers return eigenvalues of A A^T, so the singular values
        # are their square roots. Clipped to zero inside
        # finalize_eigenpairs, so the sqrt cannot produce NaN.
        self._singular_values = bkd.sqrt(eig_vals)

        # The two spectra this class reports differ by exactly this
        # factor, and this is the only place it is applied:
        # sqrt_eig_vals = s / sqrt(n - 1), so eigenvalues() returns
        # s**2 / (n - 1) while singular_values() returns s untouched.
        # The (n - 1) rather than n is Bessel's correction, making the
        # eigenvalues those of the unbiased sample covariance
        # A A^T / (n - 1) rather than of A A^T itself.
        nsamples = self._field_samples.shape[1]
        self._sqrt_eig_vals = self._singular_values / bkd.sqrt(
            bkd.full((1,), nsamples - 1)[0]
        )
        self._eig_vecs = eig_vecs * self._sqrt_eig_vals
        self._unweighted_eig_vecs = eig_vecs

    def __call__(self, coef: Array) -> Array:
        """Evaluate the KLE at given coefficients.

        Parameters
        ----------
        coef : Array, shape (nterms, nsamples)
            Random coefficients.

        Returns
        -------
        Array, shape (ncoords, nsamples)
            Field values.
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
        """Return eigenvectors scaled by sqrt(eigenvalues).

        Shape (ncoords, nterms).
        """
        return self._eig_vecs

    def eigenvalues(self) -> Array:
        r"""Sample-covariance eigenvalues, shape ``(nterms,)``.

        Variances, related to :meth:`singular_values` by

        .. math:: \lambda_i = s_i^2 / (n - 1)

        for ``n = nsamples``. The division happens once, in
        :meth:`_compute_basis`, where ``_sqrt_eig_vals`` is formed.

        Which of the two a caller wants is a real choice rather than a
        detail: the eigenvalues are what a KLE truncation weighs,
        because they say how much *variance* each mode carries, while
        the singular values are the natural scale for POD-style energy
        fractions. Both are offered here so the answer does not depend
        on which module a caller imported from.
        """
        return self._sqrt_eig_vals**2

    def singular_values(self) -> Array:
        r"""Singular values of the sample matrix, shape ``(nterms,)``.

        The raw ``s_i``, without the ``1/\sqrt{n-1}`` that turns them
        into standard deviations of the sample covariance. See
        :meth:`eigenvalues` for the relation between the two and for
        when each is the one to use.
        """
        return self._singular_values

    def mean_field(self) -> Array:
        """Return the mean field, shape (ncoords,)."""
        return self._mean_field

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nterms={self._nterms})"
