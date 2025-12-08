"""Tensor product quadrature rules.

This module provides tensor product quadrature rules that combine
univariate quadrature rules in multiple dimensions.
"""

from typing import Callable, Generic, List, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.protocols import (
    IndexGrowthRuleProtocol,
)

from .protocols import UnivariateQuadratureRuleProtocol


class TensorProductQuadratureRule(Generic[Array]):
    """Tensor product of univariate quadrature rules.

    Creates a multivariate quadrature rule by taking the tensor product
    of univariate rules with specified numbers of points per dimension.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    univariate_rules : List[UnivariateQuadratureRuleProtocol[Array]]
        Univariate quadrature rules for each dimension.
    npoints_1d : List[int]
        Number of points per dimension.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.univariate import LegendrePolynomial1D
    >>> bkd = NumpyBkd()
    >>> basis = LegendrePolynomial1D(bkd)
    >>> basis.set_nterms(5)
    >>> rule = TensorProductQuadratureRule(
    ...     bkd,
    ...     [lambda n: basis.gauss_quadrature_rule(n)] * 2,
    ...     [3, 3]
    ... )
    """

    def __init__(
        self,
        bkd: Backend[Array],
        univariate_rules: List[UnivariateQuadratureRuleProtocol[Array]],
        npoints_1d: List[int],
    ):
        self._bkd = bkd
        self._univariate_rules = univariate_rules
        self._npoints_1d = npoints_1d
        self._nvars = len(univariate_rules)

        # Precompute samples and weights
        self._samples, self._weights = self._build_tensor_product()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nsamples(self) -> int:
        """Return the number of samples."""
        return self._samples.shape[1]

    def _build_tensor_product(self) -> Tuple[Array, Array]:
        """Build tensor product samples and weights."""
        # Get 1D rules
        samples_1d: List[Array] = []
        weights_1d: List[Array] = []

        for dim in range(self._nvars):
            npts = self._npoints_1d[dim]
            s, w = self._univariate_rules[dim](npts)
            # Ensure 1D arrays
            if s.ndim > 1:
                s = s.flatten()
            if w.ndim > 1:
                w = w.flatten()
            samples_1d.append(s)
            weights_1d.append(w)

        # Compute total number of points
        total = 1
        for npts in self._npoints_1d:
            total *= npts

        # Build tensor product grid
        samples = self._bkd.zeros((self._nvars, total))
        weights = self._bkd.ones((total,))

        # Use nested loops to build tensor product
        repeat_inner = 1
        for dim in range(self._nvars - 1, -1, -1):
            npts = self._npoints_1d[dim]
            s1d = samples_1d[dim]
            w1d = weights_1d[dim]

            repeat_outer = total // (npts * repeat_inner)

            idx = 0
            for _ in range(repeat_outer):
                for pt_idx in range(npts):
                    for _ in range(repeat_inner):
                        samples[dim, idx] = s1d[pt_idx]
                        weights[idx] = weights[idx] * float(w1d[pt_idx])
                        idx += 1

            repeat_inner *= npts

        return samples, weights

    def __call__(self) -> Tuple[Array, Array]:
        """Return quadrature samples and weights."""
        return self._bkd.copy(self._samples), self._bkd.copy(self._weights)

    def integrate(self, func: Callable[[Array], Array]) -> Array:
        """Integrate a function using this quadrature rule.

        Parameters
        ----------
        func : Callable[[Array], Array]
            Function to integrate. Takes samples (nvars, nsamples) and
            returns values (nsamples, nqoi).

        Returns
        -------
        Array
            Integral estimate of shape (nqoi,)
        """
        values = func(self._samples)
        # values: (nsamples, nqoi), weights: (nsamples,)
        return self._bkd.sum(self._weights[:, None] * values, axis=0)

    def __repr__(self) -> str:
        return (
            f"TensorProductQuadratureRule(nvars={self._nvars}, "
            f"npoints_1d={self._npoints_1d}, nsamples={self.nsamples()})"
        )


class ParameterizedTensorProductQuadratureRule(Generic[Array]):
    """Tensor product quadrature parameterized by level.

    Creates tensor product quadrature rules on demand for different
    levels using a growth rule to determine points per dimension.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    univariate_rules : List[UnivariateQuadratureRuleProtocol[Array]]
        Univariate quadrature rules for each dimension.
    growth_rule : IndexGrowthRuleProtocol
        Rule mapping level to number of points.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule
    >>> bkd = NumpyBkd()
    >>> growth = LinearGrowthRule(scale=1, shift=1)
    >>> # rule = ParameterizedTensorProductQuadratureRule(bkd, [...], growth)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        univariate_rules: List[UnivariateQuadratureRuleProtocol[Array]],
        growth_rule: IndexGrowthRuleProtocol,
    ):
        self._bkd = bkd
        self._univariate_rules = univariate_rules
        self._growth_rule = growth_rule
        self._nvars = len(univariate_rules)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def __call__(self, level: int) -> Tuple[Array, Array]:
        """Generate quadrature rule for given level.

        Parameters
        ----------
        level : int
            Quadrature level (same in all dimensions).

        Returns
        -------
        Tuple[Array, Array]
            (samples, weights) with shapes (nvars, nsamples) and (nsamples,)
        """
        npoints_1d = [self._growth_rule(level) for _ in range(self._nvars)]
        rule = TensorProductQuadratureRule(
            self._bkd,
            self._univariate_rules,
            npoints_1d,
        )
        return rule()

    def integrate(
        self,
        func: Callable[[Array], Array],
        level: int,
    ) -> Array:
        """Integrate a function at given quadrature level."""
        samples, weights = self(level)
        values = func(samples)
        return self._bkd.sum(weights[:, None] * values, axis=0)

    def __repr__(self) -> str:
        return (
            f"ParameterizedTensorProductQuadratureRule(nvars={self._nvars})"
        )
