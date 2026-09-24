"""Exact moments of a combination surrogate, via cross moments.

The combination interpolant is I_K f = sum_k c_k I_k f, so its second
moment is a double sum over pairs:

    E[(I_K f)^2] = sum_{k,l} c_k c_l E[I_k f . I_l f]

``QuadratureMoments`` instead computes sum_k c_k Q_k[f^2]: it applies
the sparse-grid quadrature rule to the raw moment of the *data*, f^2 at
the grid points. That never forms I_K f, so the pairwise products
I_k f . I_l f above never appear --- what it returns is the rule's
approximation of E[f^2], not the second moment of the interpolant.

``PCEMoments`` gets the exact value by converting to an orthonormal
expansion, where the variance is a sum of squared coefficients. That
route needs globally polynomial interpolation bases. The cross-moment
route needs only that each I_k f can be evaluated, so it covers
piecewise bases too, at the cost of O(nsubspaces^2) pair integrals
rather than O(nsubspaces).
"""

from typing import Dict, Generic, Optional, Tuple
from weakref import WeakKeyDictionary

from pyapprox.surrogates.affine.protocols import (
    PhysicalDomainBasis1DProtocol,
)
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.statistics.subspace_moments import (
    variance_from_raw_moments,
)
from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.cartesian import (
    cartesian_product_samples,
    outer_product_weights,
)


# Pair cache: outer key weak so a dropped subspace is not retained,
# inner key the partner's identity.
_PairCache = WeakKeyDictionary[
    "TensorProductSubspace[Array]", Dict[int, Array]
]


class CrossMomentMoments(Generic[Array]):
    """Exact moments of the fitted surrogate, for any interpolation basis.

    mean     = sum_k c_k E[I_k f]
    variance = sum_{k,l} c_k c_l E[I_k f . I_l f] - mean^2

    The pair integrals are evaluated on a tensor product Gauss rule
    sized for the product of the two subspaces' bases. For polynomial
    bases that rule is exact; for piecewise bases it is a quadrature
    approximation whose accuracy rises with ``extra_points``, since no
    finite Gauss rule integrates across a breakpoint exactly.

    Use this when the surrogate's own variance is wanted and the basis
    is not globally polynomial. For polynomial bases ``PCEMoments`` is
    cheaper and exact, being linear rather than quadratic in the number
    of subspaces.

    Parameters
    ----------
    surrogate : CombinationSurrogate[Array]
        Fitted surrogate whose subspaces have values.
    bases_1d : Sequence[PhysicalDomainBasis1DProtocol[Array]]
        Univariate bases supplying the integration rule, one per
        dimension, from ``create_bases_1d(marginals, bkd)``. They must
        return physical-domain quadrature points.
    extra_points : int, optional
        Additional Gauss points per dimension beyond the degree the
        product of two bases requires. Default 2, which covers the
        rounding in that estimate; raise it for piecewise bases, where
        the integrand is not a polynomial.

    Raises
    ------
    TypeError
        If surrogate is not a CombinationSurrogate.
    ValueError
        If the number of bases does not match the surrogate.
    """

    def __init__(
        self,
        surrogate: CombinationSurrogate[Array],
        bases_1d: Tuple[PhysicalDomainBasis1DProtocol[Array], ...],
        extra_points: int = 2,
    ) -> None:
        if not isinstance(surrogate, CombinationSurrogate):
            raise TypeError(
                "surrogate must be a CombinationSurrogate, got "
                f"{type(surrogate).__name__}"
            )
        bases = list(bases_1d)
        if len(bases) != surrogate.nvars():
            raise ValueError(
                f"surrogate has {surrogate.nvars()} variables but "
                f"{len(bases)} bases were given"
            )
        if extra_points < 0:
            raise ValueError(
                f"extra_points must be non-negative, got {extra_points}"
            )
        self._surrogate = surrogate
        self._bases_1d = bases
        self._extra_points = extra_points
        self._bkd: Backend[Array] = surrogate.bkd()
        self._mean: Optional[Array] = None
        self._second: Optional[Array] = None
        # E[I_a f . I_b f] is symmetric in its arguments and depends only
        # on the two subspaces, so it is memoized against the pair. The
        # outer key is weak so a dropped subspace is not retained; the
        # inner key is the partner's identity.
        self._pairs: "_PairCache[Array]" = WeakKeyDictionary()

    def surrogate(self) -> CombinationSurrogate[Array]:
        """Return the surrogate these moments describe."""
        return self._surrogate

    def _rule(
        self,
        a: TensorProductSubspace[Array],
        b: TensorProductSubspace[Array],
    ) -> Tuple[Array, Array]:
        """Tensor product Gauss rule for the product of two subspaces.

        Per dimension the product of an n_a-point and an n_b-point
        Lagrange basis has degree at most (n_a - 1) + (n_b - 1), which a
        Gauss rule of ceil((n_a + n_b - 1) / 2) points integrates
        exactly.
        """
        samples_1d = []
        weights_1d = []
        for dim, basis in enumerate(self._bases_1d):
            npts_a = a.get_samples_1d(dim).shape[1]
            npts_b = b.get_samples_1d(dim).shape[1]
            npts = (npts_a + npts_b) // 2 + 1 + self._extra_points
            # The rule is derived from the basis's recursion coefficients,
            # which only exist once nterms is set.
            basis.set_nterms(npts)
            points, weights = basis.gauss_quadrature_rule(npts)
            samples_1d.append(self._bkd.flatten(points))
            weights_1d.append(self._bkd.flatten(weights))
        return (
            cartesian_product_samples(samples_1d, self._bkd),
            outer_product_weights(weights_1d, self._bkd),
        )

    def _cross_moment(
        self,
        a: TensorProductSubspace[Array],
        b: TensorProductSubspace[Array],
    ) -> Array:
        """Return E[I_a f . I_b f], shape (nqoi,), memoized per pair."""
        cached = self._pairs.setdefault(a, {})
        key = id(b)
        if key in cached:
            return cached[key]
        samples, weights = self._rule(a, b)
        moment = (a(samples) * b(samples)) @ weights
        cached[key] = moment
        # Symmetric, so record the transpose too rather than integrating
        # the same pair again from the other side.
        self._pairs.setdefault(b, {})[id(a)] = moment
        return moment

    def mean(self) -> Array:
        """Return E[I_K f], shape (nqoi,)."""
        if self._mean is None:
            bkd = self._bkd
            coefs = self._surrogate.coefficients()
            total = bkd.zeros((self._surrogate.nqoi(),))
            for j, subspace in enumerate(self._surrogate.subspaces()):
                if bkd.to_float(coefs[j]) == 0.0:
                    continue
                samples, weights = self._rule(subspace, subspace)
                total = total + coefs[j] * (subspace(samples) @ weights)
            self._mean = total
        return self._mean

    def second_moment(self) -> Array:
        """Return E[(I_K f)^2], shape (nqoi,)."""
        if self._second is None:
            bkd = self._bkd
            coefs = self._surrogate.coefficients()
            subspaces = self._surrogate.subspaces()
            nonzero = [
                (j, s)
                for j, s in enumerate(subspaces)
                if bkd.to_float(coefs[j]) != 0.0
            ]
            total = bkd.zeros((self._surrogate.nqoi(),))
            for j, a in nonzero:
                for k, b in nonzero:
                    total = total + coefs[j] * coefs[k] * self._cross_moment(
                        a, b
                    )
            self._second = total
        return self._second

    def variance(self) -> Array:
        """Return Var[I_K f], shape (nqoi,)."""
        return variance_from_raw_moments(self.mean(), self.second_moment())

    def __repr__(self) -> str:
        return f"CrossMomentMoments(nqoi={self._surrogate.nqoi()})"
