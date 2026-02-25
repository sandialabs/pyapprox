"""KLE via Galerkin projection: C_h Phi = M Phi Lambda."""

from typing import Generic, Optional, Union

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.kle.utils import eigendecomposition_generalized


class GalerkinKLE(Generic[Array]):
    """Karhunen-Loeve Expansion via Galerkin projection.

    Solves the generalized eigenproblem C_h Phi = M Phi Lambda where C_h
    is the covariance matrix assembled from a kernel and M is the FEM mass
    matrix. Eigenvectors are nodal FEM coefficients.

    The expansion is:
        f(x) = mean(x) + sigma * sum_{k=1}^{nterms} sqrt(lam_k) * phi_k(x) * z_k

    .. warning::
        The eigensolve is always performed in NumPy regardless of backend.
        For Torch backend this breaks the autograd computation graph.
        KLE basis construction is typically a one-time setup cost and does
        not need to be differentiated through.

    Parameters
    ----------
    cov_matrix : Array, shape (N, N)
        Covariance matrix C_h (symmetric positive semi-definite).
    mass_matrix : Array, shape (N, N)
        Mass matrix M (symmetric positive definite).
    nterms : int
        Number of KLE terms.
    sigma : float
        Standard deviation scaling factor.
    mean_field : float or Array
        Mean field. Scalar is broadcast to all coordinates.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        cov_matrix: Array,
        mass_matrix: Array,
        nterms: int,
        sigma: float = 1.0,
        mean_field: Union[float, Array] = 0.0,
        bkd: Backend[Array] = None,
    ):
        if bkd is None:
            raise ValueError("bkd must be provided")
        self._bkd = bkd
        self._sigma = sigma
        self._nterms = nterms

        N = cov_matrix.shape[0]
        if nterms > N:
            raise ValueError(f"nterms={nterms} exceeds N={N}")

        if np.isscalar(mean_field):
            self._mean_field = bkd.full((N,), 1) * mean_field
        else:
            self._mean_field = mean_field

        eig_vals, eig_vecs = eigendecomposition_generalized(
            cov_matrix, mass_matrix, nterms, bkd,
        )
        self._eig_vals = eig_vals
        self._sqrt_eig_vals = bkd.sqrt(eig_vals)
        self._unweighted_eig_vecs = eig_vecs
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
            raise ValueError(
                f"coef.shape[0]={coef.shape[0]} != nterms={self._nterms}"
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
        """Return eigenvectors scaled by sqrt(eigenvalues) * sigma.

        Shape (ncoords, nterms).
        """
        return self._eig_vecs

    def eigenvalues(self) -> Array:
        """Return eigenvalues, shape (nterms,)."""
        return self._eig_vals

    def mean_field(self) -> Array:
        """Return the mean field, shape (ncoords,)."""
        return self._mean_field

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nterms={self._nterms}, "
            f"sigma={self._sigma})"
        )
