"""Probability-measure quadrature factory for integration w.r.t. distributions.

Provides a factory that builds tensor product quadrature rules for
arbitrary subsets of variables using Gauss quadrature with actual
marginal distributions. Weights sum to 1 (probability measure).

This factory is designed for computing expectations E[f(X)] w.r.t.
probability distributions, e.g. for marginalizing functions over
input distributions in sensitivity analysis.
"""

from typing import Generic, List

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.protocols import MarginalProtocol
from pyapprox.surrogates.quadrature.tensor_product import (
    TensorProductQuadratureRule,
)


class ProbabilityMeasureQuadratureFactory(Generic[Array]):
    """Factory building probability-measure quadrature rules for variable subsets.

    Given ``integrate_indices``, builds a
    :class:`TensorProductQuadratureRule` using
    :func:`gauss_quadrature_rule` with the actual marginal distributions.
    Weights sum to 1 (probability measure).

    This factory is intended for **probability-measure integration**,
    where one integrates out a subset of variables with respect to
    the probability distribution defined by the marginals. For
    Lebesgue-measure integration, use
    :class:`TensorProductQuadratureFactory` instead.

    Parameters
    ----------
    marginals : List[MarginalProtocol[Array]]
        Marginal distributions for ALL variables.
    npoints_1d : List[int]
        Number of quadrature points per variable (for ALL variables).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        marginals: List[MarginalProtocol[Array]],
        npoints_1d: List[int],
        bkd: Backend[Array],
    ):
        if len(marginals) != len(npoints_1d):
            raise ValueError(
                f"marginals and npoints_1d must have the same length, "
                f"got {len(marginals)} and {len(npoints_1d)}"
            )
        self._marginals = marginals
        self._npoints_1d = npoints_1d
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(
        self, integrate_indices: List[int]
    ) -> TensorProductQuadratureRule[Array]:
        """Build a quadrature rule for the specified variable indices.

        Parameters
        ----------
        integrate_indices : List[int]
            Indices of variables to integrate out.

        Returns
        -------
        TensorProductQuadratureRule[Array]
            Quadrature rule with probability-measure weights (summing to 1)
            on the domain of the specified variables.
        """
        from pyapprox.surrogates.quadrature import (
            gauss_quadrature_rule,
        )

        bkd = self._bkd
        npts = [self._npoints_1d[i] for i in integrate_indices]

        rules = []
        for i in integrate_indices:
            marginal = self._marginals[i]
            rules.append(
                lambda n, m=marginal: gauss_quadrature_rule(m, n, bkd)
            )

        return TensorProductQuadratureRule(bkd, rules, npts)
