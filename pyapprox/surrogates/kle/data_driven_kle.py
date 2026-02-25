"""Data-driven Karhunen-Loève Expansion using SVD of field samples."""

from typing import Generic, Optional, Union

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.kle.utils import adjust_sign_eig


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
            raise ValueError(
                f"quad_weights must be 1D, got ndim={quad_weights.ndim}"
            )

        # Set mean field
        ncoords = field_samples.shape[0]
        if np.isscalar(mean_field):
            self._mean_field = bkd.full((ncoords,), 1) * mean_field
        else:
            self._mean_field = mean_field

        # Set nterms
        if nterms is None:
            nterms = ncoords
        if nterms > ncoords:
            raise ValueError(
                f"nterms={nterms} exceeds ncoords={ncoords}"
            )
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

        U, S, Vh = bkd.svd(field_samples)
        eig_vecs = adjust_sign_eig(U[:, :self._nterms], bkd)

        if self._quad_weights is not None:
            sqrt_weights = bkd.sqrt(self._quad_weights)
            eig_vecs = (1.0 / sqrt_weights[:, None]) * eig_vecs

        # Divide S by sqrt(n-1) to be consistent with sample covariance
        nsamples = self._field_samples.shape[1]
        self._sqrt_eig_vals = S[:self._nterms] / bkd.sqrt(
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
            raise ValueError(
                f"coef.shape[0]={coef.shape[0]} != nterms={self._nterms}"
            )
        if self._use_log:
            return self._bkd.exp(
                self._mean_field[:, None] + self._eig_vecs @ coef
            )
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
        """Return eigenvalues, shape (nterms,)."""
        return self._sqrt_eig_vals ** 2

    def mean_field(self) -> Array:
        """Return the mean field, shape (ncoords,)."""
        return self._mean_field

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nterms={self._nterms})"
