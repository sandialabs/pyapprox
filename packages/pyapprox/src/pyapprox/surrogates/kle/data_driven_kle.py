"""Data-driven Karhunen-Loève Expansion using SVD of field samples."""

from typing import Generic, Optional, Union

from pyapprox.surrogates.kle.snapshot_eigensolvers import (
    SnapshotEigenSolverProtocol,
    default_snapshot_eigensolver,
)
from pyapprox.surrogates.kle.truncation import resolve_nterms
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
        Number of KLE terms. Cannot be combined with
        ``variance_fraction``; when neither is given, every mode
        carrying variance is kept.
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
    variance_fraction : float, optional
        Keep the fewest modes carrying this fraction of the total
        variance, instead of a fixed ``nterms``. The two are alternative
        answers to one question, so giving both is an error.
    center : bool
        Subtract the sample mean before decomposing, and report it as
        the mean field. The default of False decomposes the samples as
        given, which is uncentered POD; that was previously the only
        behaviour, leaving centering an unstated obligation on the
        caller. Cannot be combined with an explicit ``mean_field``: both
        say what the expansion is taken about.
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
        variance_fraction: Optional[float] = None,
        center: bool = False,
    ):
        if bkd is None:
            raise ValueError("bkd must be provided")
        self._bkd = bkd
        sample_mean: Optional[Array] = None
        if center:
            if not (
                isinstance(mean_field, float) and mean_field == 0.0
            ):
                raise ValueError(
                    "pass either center=True or an explicit mean_field, "
                    "not both: each states what the expansion is taken "
                    "about, and together they leave that undefined"
                )
            sample_mean = bkd.mean(field_samples, axis=1)
            field_samples = field_samples - sample_mean[:, None]
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

        # Set mean field. isinstance narrows the Union where a numpy
        # scalar check does not, so the scalar branch is known to be
        # multiplying by a number rather than by anything the annotation
        # admits.
        ncoords = field_samples.shape[0]
        if sample_mean is not None:
            self._mean_field: Array = sample_mean
        elif isinstance(mean_field, (int, float)):
            self._mean_field = bkd.full((ncoords,), float(mean_field))
        else:
            self._mean_field = mean_field

        if nterms is not None and variance_fraction is not None:
            raise ValueError(
                "pass either nterms or variance_fraction, not both: they "
                "are two answers to one question and giving both leaves "
                "it undefined which truncates"
            )
        # The count cannot be settled before the spectrum when it is a
        # variance fraction, and the decomposition yields the whole
        # spectrum anyway, so both cases are resolved after the solve.
        self._compute_basis(nterms, variance_fraction)

    def _compute_basis(
        self,
        nterms: Optional[int],
        variance_fraction: Optional[float],
    ) -> None:
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
        # Solve for every mode carrying variance, then truncate. The
        # solver forms the whole spectrum regardless, so a count given
        # up front would save nothing and a variance fraction could not
        # be answered at all without a second decomposition.
        eig_vals, eig_vecs = self._eigensolver.solve(
            self._field_samples, None, self._metric
        )
        if nterms is None and variance_fraction is None:
            self._nterms = int(eig_vals.shape[0])
        else:
            self._nterms = resolve_nterms(
                eig_vals,
                bkd,
                nterms=nterms,
                variance_fraction=variance_fraction,
            )
        eig_vals = eig_vals[: self._nterms]
        eig_vecs = eig_vecs[:, : self._nterms]

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
