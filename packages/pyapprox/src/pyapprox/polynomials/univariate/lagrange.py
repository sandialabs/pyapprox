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

from pyapprox.polynomials.univariate.lagrange_dispatch import (
    _generic_lagrange_eval,
    _generic_lagrange_hessian,
    _generic_lagrange_jacobian,
    get_lagrange_eval_impl,
    get_lagrange_hessian_impl,
    get_lagrange_jacobian_impl,
)
from pyapprox.util.backends.protocols import Array, Backend


def compute_barycentric_weights(abscissa: Array, bkd: Backend[Array]) -> Array:
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
    abscissa: Array,
    samples: Array,
    bkd: Backend[Array],
    bary_weights: Optional[Array] = None,
) -> Array:
    """Evaluate first derivatives of Lagrange basis polynomials.

    Uses the vectorized identity L'_j(x) = L_j(x) * sum_{k!=j} 1/(x-x_k).

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
        First derivatives. Shape: (nsamples, nabscissa)
    """
    if bary_weights is None:
        bary_weights = compute_barycentric_weights(abscissa, bkd)
    return _generic_lagrange_jacobian(abscissa, samples, bary_weights, bkd)


def univariate_lagrange_second_derivative(
    abscissa: Array,
    samples: Array,
    bkd: Backend[Array],
    bary_weights: Optional[Array] = None,
) -> Array:
    """Evaluate second derivatives of Lagrange basis polynomials.

    Uses L''_j(x) = L_j(x) * (S_j^2 - T_j) where
    S_j = sum_{k!=j} 1/(x-x_k), T_j = sum_{k!=j} 1/(x-x_k)^2.

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
        Second derivatives. Shape: (nsamples, nabscissa)
    """
    if bary_weights is None:
        bary_weights = compute_barycentric_weights(abscissa, bkd)
    return _generic_lagrange_hessian(abscissa, samples, bary_weights, bkd)


def lagrange_differentiation_matrix(
    abscissa: Array,
    bkd: Backend[Array],
    bary_weights: Optional[Array] = None,
) -> Array:
    """The differentiation matrix of the Lagrange interpolant on its own
    nodes, :math:`D_{ij} = \\ell_j'(x_i)`.

    Off the diagonal :math:`D_{ij} = (w_j / w_i) / (x_i - x_j)`; the
    diagonal makes every row sum to zero, since a constant has zero
    derivative. Only ratios of the weights enter, so weights known up to
    a common factor (closed forms for Chebyshev nodes) serve as well as
    the exact ones.

    Parameters
    ----------
    abscissa : Array
        Interpolation nodes, distinct. Shape: (nabscissa,)
    bkd : Backend[Array]
        Computational backend.
    bary_weights : Array, optional
        Barycentric weights, up to a common factor. Default: computed from
        the nodes by :func:`compute_barycentric_weights`.

    Returns
    -------
    Array
        Differentiation matrix; zero for one node. Shape:
        (nabscissa, nabscissa)
    """
    nabscissa = abscissa.shape[0]
    if nabscissa == 1:
        return bkd.zeros((1, 1))
    if bary_weights is None:
        bary_weights = compute_barycentric_weights(abscissa, bkd)
    differences = abscissa[:, None] - abscissa[None, :]
    ratios = bary_weights[None, :] / bary_weights[:, None]
    # Shift the zero diagonal of the differences before dividing, then
    # drop it: the row sums set it.
    eye = bkd.eye(nabscissa)
    off_diagonal = ratios / (differences + eye) * (1.0 - eye)
    matrix: Array = off_diagonal - bkd.diag(bkd.sum(off_diagonal, axis=1))
    return matrix


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

    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.polynomials.univariate import JacobiPolynomial1D
    >>> from pyapprox.polynomials.leja import (
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
        # (abscissa, quadrature weights, barycentric weights), set together
        # by set_nterms
        self._nodal: Optional[Tuple[Array, Array, Array]] = None
        self._eval_impl = get_lagrange_eval_impl(bkd)
        self._jac_impl = get_lagrange_jacobian_impl(bkd)
        self._hess_impl = get_lagrange_hessian_impl(bkd)

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
        abscissa = self._bkd.flatten(samples)
        self._nodal = (
            abscissa,
            weights,
            compute_barycentric_weights(abscissa, self._bkd),
        )

    def nterms(self) -> int:
        """Return the number of basis terms."""
        return self._nterms

    def _nodal_state(self, purpose: str) -> Tuple[Array, Array, Array]:
        """The abscissa, quadrature and barycentric weights.

        Raises
        ------
        ValueError
            If set_nterms has not been called.
        """
        if self._nodal is None:
            raise ValueError(f"Must call set_nterms before {purpose}")
        return self._nodal

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
        abscissa, _, bary_weights = self._nodal_state("evaluation")
        return self._eval_impl(abscissa, samples[0], bary_weights, self._bkd)

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
        abscissa, _, bary_weights = self._nodal_state("evaluation")
        return self._jac_impl(abscissa, samples[0], bary_weights, self._bkd)

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
        abscissa, _, bary_weights = self._nodal_state("evaluation")
        return self._hess_impl(abscissa, samples[0], bary_weights, self._bkd)

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
            raise ValueError(f"Derivative order {order} not supported. Max is 2.")

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
        abscissa, weights, _ = self._nodal_state("quadrature_rule")
        # Return cached samples and weights in standard shapes
        return self._bkd.reshape(abscissa, (1, -1)), weights

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
