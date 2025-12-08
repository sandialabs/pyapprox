"""Numeric orthonormal polynomials for arbitrary measures.

This module provides polynomials orthogonal with respect to arbitrary
discrete or continuous probability measures:
- DiscreteNumericOrthonormalPolynomial1D: From weighted samples
- Lanczos algorithm for discrete measures
"""

from typing import Generic, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.univariate.orthopoly_base import (
    OrthonormalPolynomial1D,
)


def lanczos_recursion(
    samples: Array,
    weights: Array,
    nterms: int,
    bkd: Backend[Array],
    ortho_tol: float = 1e-10,
) -> Array:
    """Compute recursion coefficients using Lanczos algorithm.

    The Lanczos algorithm computes recursion coefficients for orthogonal
    polynomials from discrete sample points and weights.

    Parameters
    ----------
    samples : Array
        Sample points. Shape: (nsamples,)
    weights : Array
        Sample weights (must sum to 1). Shape: (nsamples,)
    nterms : int
        Number of recursion coefficients to compute.
    bkd : Backend[Array]
        Computational backend.
    ortho_tol : float
        Tolerance for orthonormality check. Default: 1e-10.

    Returns
    -------
    Array
        Recursion coefficients. Shape: (nterms, 2)

    Raises
    ------
    ValueError
        If unable to compute requested number of terms.
    """
    nsamples = len(samples)
    if nterms > nsamples:
        raise ValueError(
            f"Cannot compute {nterms} terms from {nsamples} samples"
        )

    # Normalize weights
    weights = weights / bkd.sum(weights)

    ab = bkd.zeros((nterms, 2))

    # Initialize with weighted samples
    sqrt_weights = bkd.sqrt(weights)

    # Store polynomial values at sample points
    # p_{-1} = 0, p_0 = 1/sqrt(sum(weights)) = 1 (normalized)
    p_prev = bkd.zeros_like(samples)
    p_curr = bkd.ones_like(samples)

    # b_0 = 1 (probability measure normalization)
    ab[0, 1] = 1.0

    for nn in range(nterms):
        # a_n = <x * p_n, p_n> = sum(weights * samples * p_n^2)
        ab[nn, 0] = bkd.sum(weights * samples * p_curr * p_curr)

        if nn == nterms - 1:
            break

        # Compute unnormalized p_{n+1}
        p_next = (samples - ab[nn, 0]) * p_curr

        if nn > 0:
            p_next = p_next - ab[nn, 1] * p_prev

        # b_{n+1} = sqrt(<p_{n+1}, p_{n+1}>)
        norm_sq = bkd.sum(weights * p_next * p_next)
        if norm_sq < ortho_tol:
            raise ValueError(
                f"Lanczos breakdown at term {nn+1}: norm^2 = {float(norm_sq)}"
            )

        ab[nn + 1, 1] = bkd.sqrt(bkd.asarray([norm_sq]))[0]

        # Normalize p_{n+1}
        p_prev = p_curr
        p_curr = p_next / ab[nn + 1, 1]

    return ab


class DiscreteNumericOrthonormalPolynomial1D(
    OrthonormalPolynomial1D[Array], Generic[Array]
):
    """Orthonormal polynomials from discrete samples and weights.

    Uses the Lanczos algorithm to compute recursion coefficients
    from arbitrary discrete probability measures.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    samples : Array
        Sample points defining the measure. Shape: (nsamples,)
    weights : Array
        Weights at sample points. Shape: (nsamples,)
    ortho_tol : float
        Tolerance for orthonormality. Default: 1e-10.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Discrete uniform on {0, 1, 2, 3, 4}
    >>> samples = bkd.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> weights = bkd.ones((5,)) / 5
    >>> poly = DiscreteNumericOrthonormalPolynomial1D(bkd, samples, weights)
    >>> poly.set_nterms(5)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        samples: Array,
        weights: Array,
        ortho_tol: float = 1e-10,
    ):
        self._samples = samples
        self._weights = weights / bkd.sum(weights)  # Normalize
        self._ortho_tol = ortho_tol
        self._recursion_coef: Optional[Array] = None
        super().__init__(bkd)

    @property
    def samples(self) -> Array:
        """Return sample points."""
        return self._samples

    @property
    def weights(self) -> Array:
        """Return normalized weights."""
        return self._weights

    def _get_recursion_coefficients(self, nterms: int) -> Array:
        """Compute recursion coefficients using Lanczos algorithm.

        Parameters
        ----------
        nterms : int
            Number of coefficients needed.

        Returns
        -------
        Array
            Recursion coefficients. Shape: (nterms, 2)
        """
        # Cache recursion coefficients
        if self._recursion_coef is None or self._recursion_coef.shape[0] < nterms:
            self._recursion_coef = lanczos_recursion(
                self._samples,
                self._weights,
                nterms,
                self._bkd,
                self._ortho_tol,
            )

        return self._recursion_coef[:nterms, :]

    def check_orthonormality(self, tol: float = 1e-8) -> bool:
        """Check if polynomials are orthonormal with respect to measure.

        Parameters
        ----------
        tol : float
            Tolerance for orthonormality check.

        Returns
        -------
        bool
            True if orthonormal within tolerance.
        """
        # Evaluate polynomials at sample points
        samples_2d = self._bkd.reshape(self._samples, (1, -1))
        values = self(samples_2d)  # (nsamples, nterms)

        # Compute inner product matrix
        # <p_i, p_j> = sum_k w_k * p_i(x_k) * p_j(x_k)
        weighted_values = values * self._bkd.reshape(self._weights, (-1, 1))
        gram = self._bkd.dot(values.T, weighted_values)

        # Should be identity
        eye = self._bkd.eye(self.nterms())
        return self._bkd.allclose(gram, eye, atol=tol)

    def __repr__(self) -> str:
        return (
            f"DiscreteNumericOrthonormalPolynomial1D("
            f"nsamples={len(self._samples)}, nterms={self.nterms()})"
        )


class WeightedSamplePolynomial1D(
    DiscreteNumericOrthonormalPolynomial1D[Array], Generic[Array]
):
    """Convenience class for uniform weights.

    Creates polynomials orthogonal with respect to the empirical
    distribution of the provided samples (equal weights).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    samples : Array
        Sample points. Shape: (nsamples,)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        samples: Array,
    ):
        nsamples = len(samples)
        weights = bkd.ones((nsamples,)) / nsamples
        super().__init__(bkd, samples, weights)

    def __repr__(self) -> str:
        return (
            f"WeightedSamplePolynomial1D("
            f"nsamples={len(self._samples)}, nterms={self.nterms()})"
        )
