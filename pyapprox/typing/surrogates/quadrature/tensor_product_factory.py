"""Tensor product quadrature factory for function marginalization.

Provides a factory that builds tensor product quadrature rules for
arbitrary subsets of variables, with affine mapping from [-1, 1] to
the domain of each variable.
"""

from typing import Callable, Generic, List, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.quadrature.protocols import (
    UnivariateQuadratureRuleProtocol,
)
from pyapprox.typing.surrogates.quadrature.tensor_product import (
    TensorProductQuadratureRule,
)


class _AffinelyMappedQuadratureRule(Generic[Array]):
    """Wraps a [-1,1] tensor product rule with affine domain mapping.

    Satisfies MultivariateQuadratureRuleProtocol.

    Gauss-Legendre weights integrate the probability measure (1/2)dx
    on [-1,1], so weights sum to 1. For Lebesgue integration on
    [lb, ub], the scaling factor per dimension is (ub - lb).

    Parameters
    ----------
    tp_rule : TensorProductQuadratureRule[Array]
        Tensor product rule with samples on [-1, 1]^d.
    lb : Array
        Lower bounds, shape (nvars,).
    ub : Array
        Upper bounds, shape (nvars,).
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        tp_rule: TensorProductQuadratureRule[Array],
        lb: Array,
        ub: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        raw_samples, raw_weights = tp_rule()
        self._nvars = raw_samples.shape[0]

        # Affine map: x_i = (ub_i - lb_i)/2 * t_i + (ub_i + lb_i)/2
        half_width = (ub - lb) / 2.0
        center = (ub + lb) / 2.0
        self._samples = half_width[:, None] * raw_samples + center[:, None]

        # Legendre probability-measure weights sum to 1 per dimension.
        # Lebesgue integral: int_{lb}^{ub} f(x)dx = prod(ub-lb) * sum(w_i f(x_i))
        width = ub - lb
        weight_scale = bkd.prod(width)
        self._weights = raw_weights * weight_scale

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
    """Factory building tensor product quadrature rules for variable subsets.

    Accepts univariate quadrature rules on [-1, 1] and a domain
    specification. When called with a list of variable indices, builds
    a tensor product rule for those variables, affinely mapped from
    [-1, 1] to the specified domain.

    Parameters
    ----------
    univariate_rules : List[UnivariateQuadratureRuleProtocol[Array]]
        One univariate quadrature rule per variable, on [-1, 1].
    npoints_1d : List[int]
        Number of quadrature points per variable.
    domain : Array
        Shape (nvars, 2) with [lb, ub] per variable.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        univariate_rules: List[UnivariateQuadratureRuleProtocol[Array]],
        npoints_1d: List[int],
        domain: Array,
        bkd: Backend[Array],
    ):
        if len(univariate_rules) != len(npoints_1d):
            raise ValueError(
                f"univariate_rules has {len(univariate_rules)} entries "
                f"but npoints_1d has {len(npoints_1d)}"
            )
        self._univariate_rules = univariate_rules
        self._npoints_1d = npoints_1d
        self._domain = domain
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def __call__(
        self, integrate_indices: List[int]
    ) -> _AffinelyMappedQuadratureRule[Array]:
        """Build a quadrature rule for the specified variable indices.

        Parameters
        ----------
        integrate_indices : List[int]
            Indices of variables to integrate out.

        Returns
        -------
        _AffinelyMappedQuadratureRule[Array]
            Quadrature rule with samples on the domain of the
            specified variables.
        """
        rules = [self._univariate_rules[i] for i in integrate_indices]
        npts = [self._npoints_1d[i] for i in integrate_indices]
        tp = TensorProductQuadratureRule(self._bkd, rules, npts)

        lb = self._bkd.asarray(
            [float(self._domain[i, 0]) for i in integrate_indices]
        )
        ub = self._bkd.asarray(
            [float(self._domain[i, 1]) for i in integrate_indices]
        )
        return _AffinelyMappedQuadratureRule(tp, lb, ub, self._bkd)
