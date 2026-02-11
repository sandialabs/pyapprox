"""Lagrange basis functions using arbitrary quadrature rules.

This module provides a Lagrange basis implementation that uses sample points
from a quadrature rule. The basis functions are the Lagrange interpolating
polynomials at the quadrature points.

The key design is that LagrangeBasis1D accepts any quadrature rule as a
callable (npoints) -> (samples, weights). This allows using:
- Gauss quadrature points from orthogonal polynomials
- Leja sequence points from LejaSequence1D.quadrature_rule
- Clenshaw-Curtis or other nested quadrature rules
- Any custom point set

Example with Leja sequence:
    leja_seq = LejaSequence1D(bkd, poly, weighting, bounds=(-1.0, 1.0))
    basis = LagrangeBasis1D(bkd, leja_seq.quadrature_rule)
"""

from typing import Callable, Generic, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.univariate.lagrange_dispatch import (
    get_lagrange_eval_impl,
    _generic_lagrange_eval,
)


def compute_barycentric_weights(
    abscissa: Array, bkd: Backend[Array]
) -> Array:
    """Compute barycentric weights for Lagrange interpolation.

    Parameters
    ----------
    abscissa : Array
        Interpolation nodes. Shape: (nabscissa,)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Barycentric weights w_j = 1/prod_{i!=j}(x_j - x_i).
        Shape: (nabscissa,)
    """
    nabscissa = abscissa.shape[0]
    if nabscissa == 1:
        return bkd.ones((1,))
    denoms = abscissa[:, None] - abscissa[None, :]
    # Set diagonal to 1.0 so prod gives product over i != j
    denoms = denoms + bkd.eye(nabscissa)
    return 1.0 / bkd.prod(denoms, axis=1)


def univariate_lagrange_polynomial(
    abscissa: Array,
    samples: Array,
    bkd: Backend[Array],
    bary_weights: Optional[Array] = None,
) -> Array:
    """Evaluate Lagrange basis polynomials at sample points.

    Uses the barycentric formula L_j(x) = w_j * P(x) / (x - x_j) where
    P(x) = prod_i(x - x_i) and w_j are barycentric weights.

    Parameters
    ----------
    abscissa : Array
        Interpolation nodes. Shape: (nabscissa,)
    samples : Array
        Evaluation points. Shape: (nsamples,)
    bkd : Backend[Array]
        Computational backend.
    bary_weights : Array, optional
        Precomputed barycentric weights. If None, computed on the fly.

    Returns
    -------
    Array
        Lagrange basis values. Shape: (nsamples, nabscissa)
    """
    if bary_weights is None:
        bary_weights = compute_barycentric_weights(abscissa, bkd)
    return _generic_lagrange_eval(abscissa, samples, bary_weights, bkd)


def univariate_lagrange_first_derivative(
    abscissa: Array, samples: Array, bkd: Backend[Array]
) -> Array:
    """Evaluate first derivatives of Lagrange basis polynomials.

    Parameters
    ----------
    abscissa : Array
        Interpolation nodes. Shape: (nabscissa,)
    samples : Array
        Evaluation points. Shape: (nsamples,)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        First derivatives. Shape: (nsamples, nabscissa)
    """
    nsamples = samples.shape[0]
    nabscissa = abscissa.shape[0]
    denoms = abscissa[:, None] - abscissa[None, :]
    numers = samples[:, None] - abscissa[None, :]
    derivs = bkd.zeros((nsamples, nabscissa))

    for ii in range(nabscissa):
        denom = bkd.prod(denoms[ii, :ii]) * bkd.prod(denoms[ii, ii + 1 :])
        # Product rule for the jth 1D basis function
        numer_deriv = bkd.zeros((nsamples,))
        for jj in range(nabscissa):
            # Compute derivative of kth component of product rule sum
            if ii != jj:
                # Product over all k != ii, k != jj
                term = bkd.ones((nsamples,))
                for kk in range(nabscissa):
                    if kk != ii and kk != jj:
                        term = term * numers[:, kk]
                numer_deriv = numer_deriv + term
        derivs[:, ii] = numer_deriv / denom

    return derivs


def univariate_lagrange_second_derivative(
    abscissa: Array, samples: Array, bkd: Backend[Array]
) -> Array:
    """Evaluate second derivatives of Lagrange basis polynomials.

    Parameters
    ----------
    abscissa : Array
        Interpolation nodes. Shape: (nabscissa,)
    samples : Array
        Evaluation points. Shape: (nsamples,)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    Array
        Second derivatives. Shape: (nsamples, nabscissa)
    """
    nsamples = samples.shape[0]
    nabscissa = abscissa.shape[0]
    denoms = abscissa[:, None] - abscissa[None, :]
    numers = samples[:, None] - abscissa[None, :]
    derivs = bkd.zeros((nsamples, nabscissa))

    for ii in range(nabscissa):
        denom = bkd.prod(denoms[ii, :ii]) * bkd.prod(denoms[ii, ii + 1 :])
        numer_deriv = bkd.zeros((nsamples,))
        for jj in range(nabscissa):
            for kk in range(nabscissa):
                if ii != jj and ii != kk and jj != kk:
                    # Product over all m != ii, m != jj, m != kk
                    term = bkd.ones((nsamples,))
                    for mm in range(nabscissa):
                        if mm != ii and mm != jj and mm != kk:
                            term = term * numers[:, mm]
                    numer_deriv = numer_deriv + term
        derivs[:, ii] = numer_deriv / denom

    return derivs


class LagrangeBasis1D(Generic[Array]):
    """Lagrange basis using arbitrary quadrature points.

    This basis uses sample points from a quadrature rule as interpolation
    nodes. The quadrature rule is provided as a callable that takes npoints
    and returns (samples, weights).

    This design allows using any point set:
    - Gauss quadrature from orthogonal polynomials
    - Leja sequence points (nested, well-conditioned)
    - Clenshaw-Curtis points
    - Any custom point set

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    quadrature_rule : Callable[[int], Tuple[Array, Array]]
        Function that takes npoints and returns (samples, weights) where
        samples has shape (1, npoints) and weights has shape (npoints, 1).

    Examples
    --------
    Using with Leja sequence:

    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import JacobiPolynomial1D
    >>> from pyapprox.typing.surrogates.affine.leja import (
    ...     LejaSequence1D, ChristoffelWeighting
    ... )
    >>> bkd = NumpyBkd()
    >>> poly = JacobiPolynomial1D(0.0, 0.0, bkd)
    >>> weighting = ChristoffelWeighting(bkd)
    >>> leja_seq = LejaSequence1D(bkd, poly, weighting, bounds=(-1.0, 1.0))
    >>> basis = LagrangeBasis1D(bkd, leja_seq.quadrature_rule)
    >>> basis.set_nterms(5)
    >>> samples = bkd.asarray([[0.0, 0.5, -0.5]])
    >>> values = basis(samples)  # Shape: (3, 5)

    Using with Gauss quadrature:

    >>> poly = JacobiPolynomial1D(0.0, 0.0, bkd)
    >>> poly.set_nterms(10)  # Ensure enough terms
    >>> basis = LagrangeBasis1D(bkd, poly.gauss_quadrature_rule)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        quadrature_rule: Callable[[int], Tuple[Array, Array]],
    ):
        self._bkd = bkd
        self._quadrature_rule = quadrature_rule
        self._nterms: int = 0
        self._abscissa: Optional[Array] = None
        self._weights: Optional[Array] = None
        self._bary_weights: Optional[Array] = None
        self._eval_impl = get_lagrange_eval_impl(bkd)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def set_nterms(self, nterms: int) -> None:
        """Set the number of basis terms.

        Parameters
        ----------
        nterms : int
            Number of basis terms (equal to number of interpolation points).
        """
        if nterms <= 0:
            raise ValueError("nterms must be positive")

        self._nterms = nterms
        samples, weights = self._quadrature_rule(nterms)
        self._abscissa = samples.flatten()
        self._weights = weights
        self._bary_weights = compute_barycentric_weights(
            self._abscissa, self._bkd
        )

    def nterms(self) -> int:
        """Return the number of basis terms."""
        return self._nterms

    def __call__(self, samples: Array) -> Array:
        """Evaluate Lagrange basis at sample points.

        Parameters
        ----------
        samples : Array
            Evaluation points. Shape: (1, nsamples)

        Returns
        -------
        Array
            Basis values. Shape: (nsamples, nterms)
        """
        if self._abscissa is None:
            raise ValueError("Must call set_nterms before evaluation")
        return self._eval_impl(
            self._abscissa, samples[0], self._bary_weights, self._bkd
        )

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate first derivatives of Lagrange basis.

        Parameters
        ----------
        samples : Array
            Evaluation points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (1, nsamples).
        """
        if self._abscissa is None:
            raise ValueError("Must call set_nterms before evaluation")
        return univariate_lagrange_first_derivative(
            self._abscissa, samples[0], self._bkd
        )

    def hessian_batch(self, samples: Array) -> Array:
        """Evaluate second derivatives of Lagrange basis.

        Parameters
        ----------
        samples : Array
            Evaluation points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            Second derivatives. Shape: (nsamples, nterms)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (1, nsamples).
        """
        if self._abscissa is None:
            raise ValueError("Must call set_nterms before evaluation")
        return univariate_lagrange_second_derivative(
            self._abscissa, samples[0], self._bkd
        )

    def derivatives(self, samples: Array, order: int) -> Array:
        """Evaluate derivatives of specified order.

        Parameters
        ----------
        samples : Array
            Evaluation points. Shape: (1, nsamples)
        order : int
            Derivative order. 0 = values, 1 = first, 2 = second.

        Returns
        -------
        Array
            Derivatives. Shape: (nsamples, nterms)
        """
        if order == 0:
            return self(samples)
        elif order == 1:
            return self.jacobian_batch(samples)
        elif order == 2:
            return self.hessian_batch(samples)
        else:
            raise ValueError(
                f"Derivative order {order} not supported. Max is 2."
            )

    def quadrature_rule(self) -> Tuple[Array, Array]:
        """Return quadrature points and weights for current nterms.

        Must call set_nterms before using this method.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, nterms)
        weights : Array
            Quadrature weights. Shape: (nterms, 1)

        Raises
        ------
        ValueError
            If set_nterms has not been called.
        """
        if self._abscissa is None:
            raise ValueError("Must call set_nterms before quadrature_rule")
        # Return cached samples and weights in standard shapes
        return self._bkd.reshape(self._abscissa, (1, -1)), self._weights

    def get_samples(self, nterms: int) -> Array:
        """Return interpolation nodes for the given number of terms.

        This method satisfies the InterpolationBasis1DProtocol requirement.

        Parameters
        ----------
        nterms : int
            Number of interpolation points.

        Returns
        -------
        Array
            Interpolation nodes. Shape: (1, nterms)
        """
        samples, _ = self._quadrature_rule(nterms)
        return samples

    def __repr__(self) -> str:
        return f"LagrangeBasis1D(nterms={self._nterms})"
