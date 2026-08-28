"""Data-driven Karhunen-Loève Expansion using SVD of field samples."""

from typing import Generic, Optional, Union

import numpy as np

from pyapprox.surrogates.kle.utils import adjust_sign_eig
from pyapprox.util.backends.protocols import Array, Backend

_MACHINE_EPS = float(np.finfo(np.float64).eps)


def _reject_negligible_singular_values(
    retained: Array, spectrum: Array, bkd: Backend[Array]
) -> None:
    """Refuse a basis whose trailing modes carry no variance.

    Mirrors the check ``eigensolvers`` applies to a kernel spectrum, for
    the reason that motivates it there: a KLE scales each vector by
    ``sqrt(eigenvalue)``, so a term whose singular value is zero to
    machine precision contributes a column of zeros -- a mode the caller
    asked for and did not get.

    For snapshot data the usual cause is centering. Subtracting the
    sample mean makes the columns linearly dependent, dropping the rank
    to ``nsamples - 1``, so a caller who centers and then asks for
    ``nsamples`` terms gets a silent empty mode. That cannot be caught
    by counting arguments, since this class is handed a matrix without
    being told whether it was centered; the spectrum reports it.
    """
    largest = bkd.to_float(bkd.max(spectrum))
    if largest <= 0.0:
        raise ValueError(
            "field_samples has no variance: every singular value is zero"
        )
    tolerance = largest * int(spectrum.shape[0]) * _MACHINE_EPS
    nusable = int((bkd.to_numpy(retained) > tolerance).sum())
    if nusable < int(retained.shape[0]):
        raise ValueError(
            f"nterms={int(retained.shape[0])} exceeds the numerical rank "
            f"of field_samples ({nusable}); the trailing terms would be "
            "zero columns. Centered data has rank at most nsamples - 1."
        )


class DataDrivenKLE(Generic[Array]):
    """Karhunen-Loève Expansion computed from field sample data.

    Uses SVD of the (optionally weighted) field samples for numerical
    stability, rather than eigendecomposition of the sample covariance.

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
        Quadrature weights for weighted SVD.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        field_samples: Array,
        mean_field: Union[float, Array] = 0.0,
        use_log: bool = False,
        nterms: Optional[int] = None,
        quad_weights: Optional[Array] = None,
        bkd: Backend[Array] = None,
    ):
        if bkd is None:
            raise ValueError("bkd must be provided")
        self._bkd = bkd
        self._field_samples = field_samples
        self._use_log = use_log
        self._quad_weights = quad_weights
        if quad_weights is not None and quad_weights.ndim != 1:
            raise ValueError(f"quad_weights must be 1D, got ndim={quad_weights.ndim}")

        # Set mean field
        ncoords = field_samples.shape[0]
        if np.isscalar(mean_field):
            self._mean_field = bkd.full((ncoords,), 1) * mean_field
        else:
            self._mean_field = mean_field

        # Set nterms. The binding limit is the rank of the sample matrix,
        # min(ncoords, nsamples), not ncoords alone: a thin SVD returns
        # only that many singular vectors, so asking for more silently
        # yields columns of zeros -- modes the caller asked for and did
        # not get, scaled by a zero singular value. Callers who centered
        # their data before passing it lose one further term, since
        # subtracting the sample mean makes the columns linearly
        # dependent; that is checked below against the spectrum rather
        # than assumed here, because this class cannot see whether
        # centering happened.
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

        # Compute basis via SVD
        self._compute_basis()

    def _compute_basis(self) -> None:
        """Compute KLE basis using SVD of (weighted) field samples.

        SVD-based approach is more numerically stable than computing
        the covariance matrix then taking its eigendecomposition.

        C = A^T A / (n-1)  (sample covariance)
        A = U S V^T  =>  C = V S^2 V^T / (n-1)
        So eigenvalues of C are S^2/(n-1) and eigenvectors are V.
        """
        bkd = self._bkd
        if self._quad_weights is None:
            field_samples = self._field_samples
        else:
            sqrt_weights = bkd.sqrt(self._quad_weights)
            field_samples = sqrt_weights[:, None] * self._field_samples

        # Thin SVD. Only min(ncoords, nsamples) left vectors can have a
        # nonzero singular value, and Vh is discarded, so the full
        # decomposition's (nsamples, nsamples) Vh is pure waste -- and it
        # dominates for the usual snapshot shape of few coordinates and
        # many samples: 4.4s against 0.15s at (65, 10000).
        U, S, _Vh = bkd.svd(field_samples, full_matrices=False)
        eig_vecs = adjust_sign_eig(U[:, : self._nterms], bkd)

        if self._quad_weights is not None:
            sqrt_weights = bkd.sqrt(self._quad_weights)
            eig_vecs = (1.0 / sqrt_weights[:, None]) * eig_vecs

        self._singular_values = S[: self._nterms]
        _reject_negligible_singular_values(self._singular_values, S, bkd)

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
