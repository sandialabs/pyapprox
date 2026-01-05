"""Lagrange basis functions using arbitrary quadrature rules.

This module provides a Lagrange basis implementation that uses sample points
from a quadrature rule. The basis functions are the Lagrange interpolating
polynomials at the quadrature points.

The key design is that LagrangeBasis1D accepts any quadrature rule as a
callable (npoints) -> (samples, weights). This allows using:
- Gauss quadrature points from orthogonal polynomials
- Leja sequence points from LejaSequence1D.get_sequence
- Clenshaw-Curtis or other nested quadrature rules
- Any custom point set

Example with Leja sequence:
    leja_seq = LejaSequence1D(bkd, poly, weighting, bounds=(-1.0, 1.0))
    basis = LagrangeBasis1D(bkd, leja_seq.get_sequence)
"""

from typing import Callable, Generic, Optional, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend


def univariate_lagrange_polynomial(
    abscissa: Array, samples: Array, bkd: Backend[Array]
) -> Array:
    """Evaluate Lagrange basis polynomials at sample points.

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
        Lagrange basis values. Shape: (nsamples, nabscissa)
    """
    nabscissa = abscissa.shape[0]
    denoms = abscissa[:, None] - abscissa[None, :]
    numers = samples[:, None] - abscissa[None, :]
    values = []
    for ii in range(nabscissa):
        # l_j(x) = prod_{i!=j} (x-x_i)/(x_j-x_i)
        denom = bkd.prod(denoms[ii, :ii]) * bkd.prod(denoms[ii, ii + 1 :])
        numer = bkd.prod(numers[:, :ii], axis=1) * bkd.prod(
            numers[:, ii + 1 :], axis=1
        )
        values.append(numer / denom)
    return bkd.stack(values, axis=1)


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
    >>> basis = LagrangeBasis1D(bkd, leja_seq.get_sequence)
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
        return univariate_lagrange_polynomial(
            self._abscissa, samples[0], self._bkd
        )

    def jacobians(self, samples: Array) -> Array:
        """Evaluate first derivatives of Lagrange basis.

        Parameters
        ----------
        samples : Array
            Evaluation points. Shape: (1, nsamples)

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)
        """
        if self._abscissa is None:
            raise ValueError("Must call set_nterms before evaluation")
        return univariate_lagrange_first_derivative(
            self._abscissa, samples[0], self._bkd
        )

    def hessians(self, samples: Array) -> Array:
        """Evaluate second derivatives of Lagrange basis.

        Parameters
        ----------
        samples : Array
            Evaluation points. Shape: (1, nsamples)

        Returns
        -------
        Array
            Second derivatives. Shape: (nsamples, nterms)
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
            return self.jacobians(samples)
        elif order == 2:
            return self.hessians(samples)
        else:
            raise ValueError(
                f"Derivative order {order} not supported. Max is 2."
            )

    def gauss_quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Return quadrature points and weights.

        Parameters
        ----------
        npoints : int
            Number of quadrature points.

        Returns
        -------
        points : Array
            Quadrature points. Shape: (1, npoints)
        weights : Array
            Quadrature weights. Shape: (npoints, 1)
        """
        return self._quadrature_rule(npoints)

    def __repr__(self) -> str:
        return f"LagrangeBasis1D(nterms={self._nterms})"
