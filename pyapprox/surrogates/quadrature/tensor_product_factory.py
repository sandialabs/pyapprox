"""Tensor product quadrature factory for Lebesgue-measure integration.

Provides a factory that builds tensor product quadrature rules for
arbitrary subsets of variables, using Gauss-Legendre rules mapped to
the domain of each variable.

This factory is designed for **Lebesgue-measure integration** (e.g.,
function marginalization), NOT for computing expectations E[f(X)]
w.r.t. probability distributions. For the latter, use
:func:`~pyapprox.surrogates.quadrature.gauss_quadrature_rule`
with :class:`~pyapprox.surrogates.quadrature.TensorProductQuadratureRule`.
"""

from typing import Callable, Generic, List, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.quadrature.tensor_product import (
    TensorProductQuadratureRule,
)


class _LebesgueMappedQuadratureRule(Generic[Array]):
    """Tensor product rule with Lebesgue-measure weights.

    Uses :func:`gauss_quadrature_rule` with uniform marginals to produce
    quadrature points in the physical domain, then scales weights by the
    domain volume so that they integrate against the Lebesgue measure.

    Parameters
    ----------
    tp_rule : TensorProductQuadratureRule[Array]
        Tensor product rule with samples in physical domain and
        probability-measure weights (summing to 1).
    volume : float
        Domain volume = prod(ub_i - lb_i).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        tp_rule: TensorProductQuadratureRule[Array],
        volume: float,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        samples, weights = tp_rule()
        self._nvars = samples.shape[0]
        self._samples = samples
        self._weights = weights * volume

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nsamples(self) -> int:
        """Return the number of quadrature samples."""
        return self._samples.shape[1]

    def __call__(self) -> Tuple[Array, Array]:
        """Return quadrature samples and weights.

        Returns
        -------
        Tuple[Array, Array]
            (samples, weights) with shapes (nvars, nsamples) and (nsamples,).
        """
        return self._bkd.copy(self._samples), self._bkd.copy(self._weights)

    def integrate(self, func: Callable[[Array], Array]) -> Array:
        """Integrate a function using this quadrature rule.

        Parameters
        ----------
        func : Callable[[Array], Array]
            Function mapping (nvars, nsamples) -> (nsamples, nqoi).

        Returns
        -------
        Array
            Integral estimate of shape (nqoi,).
        """
        values = func(self._samples)
        return self._bkd.sum(self._weights[:, None] * values, axis=0)


class TensorProductQuadratureFactory(Generic[Array]):
    """Factory building Lebesgue-measure quadrature rules for variable subsets.

    Uses Gauss-Legendre rules mapped to the domain of each variable via
    uniform marginals. The resulting weights are scaled by the domain
    volume so that they integrate against the Lebesgue measure.

    This factory is intended for **function marginalization**, where one
    integrates out a subset of variables with respect to the Lebesgue
    measure. For computing expectations E[f(X)] w.r.t. probability
    distributions, use
    :func:`~pyapprox.surrogates.quadrature.gauss_quadrature_rule`
    with
    :class:`~pyapprox.surrogates.quadrature.TensorProductQuadratureRule`
    instead.

    Parameters
    ----------
    npoints_1d : List[int]
        Number of quadrature points per variable.
    domain : Array
        Shape (nvars, 2) with [lb, ub] per variable.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        npoints_1d: List[int],
        domain: Array,
        bkd: Backend[Array],
    ):
        self._npoints_1d = npoints_1d
        self._domain = domain
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(
        self, integrate_indices: List[int]
    ) -> _LebesgueMappedQuadratureRule[Array]:
        """Build a quadrature rule for the specified variable indices.

        Parameters
        ----------
        integrate_indices : List[int]
            Indices of variables to integrate out.

        Returns
        -------
        _LebesgueMappedQuadratureRule[Array]
            Quadrature rule with Lebesgue-measure weights on the domain
            of the specified variables.
        """
        from pyapprox.probability.univariate.uniform import (
            UniformMarginal,
        )
        from pyapprox.surrogates.quadrature import (
            gauss_quadrature_rule,
        )

        bkd = self._bkd
        npts = [self._npoints_1d[i] for i in integrate_indices]

        # Build uniform marginals for each dimension
        rules = []
        volume = 1.0
        for i in integrate_indices:
            lb_i = float(self._domain[i, 0])
            ub_i = float(self._domain[i, 1])
            marginal = UniformMarginal(lower=lb_i, upper=ub_i, bkd=bkd)
            rules.append(
                lambda n, m=marginal: gauss_quadrature_rule(m, n, bkd)
            )
            volume *= (ub_i - lb_i)

        tp = TensorProductQuadratureRule(bkd, rules, npts)
        return _LebesgueMappedQuadratureRule(tp, volume, bkd)
