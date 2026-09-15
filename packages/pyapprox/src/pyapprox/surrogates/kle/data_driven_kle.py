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

        An *approximate* solver constrains how the basis may be
        truncated, because it never forms the whole spectrum. Pass
        ``nterms``: both ``variance_fraction`` and the default of
        keeping every mode carrying variance are questions about the
        modes discarded as well as those kept, and such a solver has no
        answer to give.
    variance_fraction : float, optional
        Keep the fewest modes carrying this fraction of the total
        variance, instead of a fixed ``nterms``. The two are alternative
        answers to one question, so giving both is an error. Requires an
        exact ``eigensolver``.
    center : bool
        Subtract the sample mean before decomposing, and report it as
        the mean field. The default of False decomposes the samples as
        given, which is uncentered POD; that was previously the only
        behaviour, leaving centering an unstated obligation on the
        caller. Cannot be combined with an explicit ``mean_field``: both
        say what the expansion is taken about.
    center_by : Array, optional
        Shape ``(ncoords,)`` or ``(ncoords, 1)``. Subtract this field
        rather than the sample mean, and report it as the mean field.

        For when the sample mean of ``field_samples`` is not the field
        to expand about -- samples drawn to cover a parameter space
        rather than to represent a distribution have a mean that is an
        artifact of the design. Passing an already-centered matrix with
        ``mean_field`` set does the same thing, but leaves the
        subtraction to the caller and so lets the two disagree.

        Mutually exclusive with both ``center`` and ``mean_field``: all
        three answer what the expansion is taken about.
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
        center_by: Optional[Array] = None,
    ):
        if bkd is None:
            raise ValueError("bkd must be provided")
        self._bkd = bkd
        explicit_mean_field = not (
            isinstance(mean_field, float) and mean_field == 0.0
        )
        if center and center_by is not None:
            raise ValueError(
                "pass either center=True or center_by, not both: "
                "center takes the sample mean while center_by takes the "
                "field given, and together they leave it undefined which "
                "is subtracted"
            )
        if center_by is not None and explicit_mean_field:
            raise ValueError(
                "pass either center_by or mean_field, not both: "
                "center_by subtracts the field and reports it as the "
                "mean field, so supplying mean_field as well leaves it "
                "undefined what the expansion is taken about"
            )
        sample_mean: Optional[Array] = None
        if center:
            if explicit_mean_field:
                raise ValueError(
                    "pass either center=True or an explicit mean_field, "
                    "not both: each states what the expansion is taken "
                    "about, and together they leave that undefined"
                )
            sample_mean = bkd.mean(field_samples, axis=1)
            field_samples = field_samples - sample_mean[:, None]
        elif center_by is not None:
            sample_mean = bkd.reshape(
                center_by, (int(center_by.shape[0]),)
            )
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
        # Ask for the count when one was given. A variance fraction
        # cannot be answered without the whole spectrum -- it is a
        # question about the modes that were discarded as much as the
        # ones kept -- so that case still requests everything and slices
        # afterwards.
        #
        # The count is not merely a hint. An approximate solver never
        # forms the whole spectrum and so cannot supply the numerical
        # rank; passing None makes it unusable here rather than merely
        # unhelped.
        decomposition = self._eigensolver.solve(
            self._field_samples,
            None if variance_fraction is not None else nterms,
            self._metric,
        )
        eig_vals = decomposition.eigenvalues
        eig_vecs = decomposition.eigenvectors
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
        # Only the unweighted basis is stored. The weighted one differs
        # from it by a per-column scaling, so holding both would keep two
        # (ncoords, nterms) arrays alive for the object's lifetime --
        # which at a large ambient dimension is the dominant cost of
        # owning a fitted KLE. The unweighted form is the one kept
        # because it is what the protocol's eigenvectors() returns, and
        # because recovering it from the weighted one would divide by
        # sqrt_eig_vals and amplify error in the smallest modes.
        self._unweighted_eig_vecs = eig_vecs

        # Both places that apply the scaling rely on it being one entry
        # per mode, and the failure if it is not would be silent rather
        # than loud: a length-one vector broadcasts across every mode,
        # producing an array of the right shape with one scale applied
        # uniformly. A length that matches neither raises on its own, so
        # this guard is what covers the case that would not.
        if (
            self._sqrt_eig_vals.ndim != 1
            or int(self._sqrt_eig_vals.shape[0]) != self._nterms
        ):
            raise ValueError(
                f"sqrt_eig_vals must be 1D with one entry per mode "
                f"({self._nterms}), got shape "
                f"{tuple(self._sqrt_eig_vals.shape)}"
            )

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
        # (V * s) @ coef and V @ (s * coef) agree to rounding, but the
        # second scales the (nterms, nsamples) factor rather than the
        # ambient one, so no (ncoords, nterms) temporary is built. Timed
        # at 200000 coordinates and 50 terms the two are within 1%, so
        # the smaller allocation is free.
        scaled = self._sqrt_eig_vals[:, None] * coef
        field = self._mean_field[:, None] + self._unweighted_eig_vecs @ scaled
        if self._use_log:
            return self._bkd.exp(field)
        return field

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

        Built on each call rather than stored, since it is the stored
        basis times a ``(nterms,)`` vector and keeping both would double
        what a fitted KLE occupies. Callers that only want to *apply* it
        should scale their coefficients instead, as :meth:`__call__`
        does, and avoid the array entirely.
        """
        return self._unweighted_eig_vecs * self._sqrt_eig_vals

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
