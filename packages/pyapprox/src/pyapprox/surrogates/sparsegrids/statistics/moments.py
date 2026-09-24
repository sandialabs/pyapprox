"""Moments of a fitted combination surrogate.

A surrogate evaluates; what you compute from it lives here. Each class
takes a fitted surrogate and exposes memoized ``mean`` and ``variance``.

The mean is the same under every definition below: sum_k c_k Q_k f. It
equals E[I_K f] whenever each 1D rule is interpolatory, which holds for
the Gauss, Leja and Clenshaw-Curtis factories.

The variance is not. Two definitions are offered and they disagree;
pick by what the number is for.

- ``QuadratureMoments``: the sparse-grid rule applied to f^2. Cheap,
  and not guaranteed nonnegative because the rule is signed.
- ``PCEMoments``: the exact variance of the fitted surrogate, via an
  orthonormal PCE conversion. Reach for this one when "the variance of
  my surrogate" is the question.

A third quantity, sum_k c_k Var_k, drives refinement inside
``VarianceChangeIndicator``. It is a proxy for ranking candidates, not
an estimate of either definition here, and is deliberately not offered
as a moments class.
"""

from typing import Callable, Generic, Optional, Sequence

from pyapprox.surrogates.affine.expansions import pce_statistics
from pyapprox.surrogates.affine.expansions.pce import (
    PolynomialChaosExpansion,
)
from pyapprox.surrogates.affine.protocols import (
    PhysicalDomainBasis1DProtocol,
)
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.converters.pce import (
    SparseGridToPCEConverter,
)
from pyapprox.surrogates.sparsegrids.statistics.subspace_moments import (
    subspace_mean,
    subspace_raw_moment,
)
from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array


def _combine(
    surrogate: CombinationSurrogate[Array],
    statistic: Callable[[TensorProductSubspace[Array]], Array],
) -> Array:
    """Sum a per-subspace statistic against the Smolyak coefficients."""
    bkd = surrogate.bkd()
    coefs = surrogate.coefficients()
    total = bkd.zeros((surrogate.nqoi(),))
    for j, subspace in enumerate(surrogate.subspaces()):
        total = total + coefs[j] * statistic(subspace)
    return total


class QuadratureMoments(Generic[Array]):
    """Moments under the sparse-grid quadrature rule.

    mean     = sum_k c_k Q_k f
    variance = Q_K[f^2] - (Q_K f)^2

    Built from the raw moments, so both terms combine linearly through
    the Smolyak coefficients.

    The variance is **not guaranteed nonnegative**: Q_K is a signed
    rule, so it can return a negative number where the true variance is
    positive. It is also not the variance of the fitted surrogate ---
    on f = x^2 + y^2 over [0,1]^2 with Gauss rules and
    K = {(0,0),(1,0),(2,0),(0,1),(0,2)} it gives 7.375/45 where the
    true value is 8/45. Use ``PCEMoments`` when the surrogate's own
    variance is wanted.

    Parameters
    ----------
    surrogate : CombinationSurrogate[Array]
        Fitted surrogate whose subspaces have values.

    Raises
    ------
    TypeError
        If surrogate is not a CombinationSurrogate.
    """

    def __init__(self, surrogate: CombinationSurrogate[Array]) -> None:
        if not isinstance(surrogate, CombinationSurrogate):
            raise TypeError(
                "surrogate must be a CombinationSurrogate, got "
                f"{type(surrogate).__name__}"
            )
        self._surrogate = surrogate
        self._mean: Optional[Array] = None
        self._second: Optional[Array] = None

    def surrogate(self) -> CombinationSurrogate[Array]:
        """Return the surrogate these moments describe."""
        return self._surrogate

    def mean(self) -> Array:
        """Return E[f] under the sparse-grid rule, shape (nqoi,)."""
        if self._mean is None:
            self._mean = _combine(self._surrogate, subspace_mean)
        return self._mean

    def second_moment(self) -> Array:
        """Return Q_K[f^2], shape (nqoi,)."""
        if self._second is None:
            self._second = _combine(
                self._surrogate, lambda s: subspace_raw_moment(s, 2)
            )
        return self._second

    def variance(self) -> Array:
        """Return Q_K[f^2] - (Q_K f)^2, shape (nqoi,)."""
        mean = self.mean()
        return self.second_moment() - mean**2

    def __repr__(self) -> str:
        return f"QuadratureMoments(nqoi={self._surrogate.nqoi()})"


class PCEMoments(Generic[Array]):
    """Exact moments of the fitted surrogate, via PCE conversion.

    Converts the combination surrogate to an orthonormal polynomial
    chaos expansion, whose coefficients give the mean and variance of
    I_K f exactly: E = c_0 and Var = sum over nonzero multi-indices of
    c_alpha^2.

    This is the one to use when the question is "what is the variance of
    my surrogate". ``QuadratureMoments`` answers a different question
    and gives a different number.

    Parameters
    ----------
    surrogate : CombinationSurrogate[Array]
        Fitted surrogate whose subspaces have values.
    orthonormal_bases_1d : Sequence[PhysicalDomainBasis1DProtocol[Array]]
        Univariate bases, from ``create_bases_1d(marginals, bkd)``. They
        must return physical-domain quadrature points, or the spectral
        projection is wrong for non-canonical domains.

    Raises
    ------
    TypeError
        If surrogate is not a CombinationSurrogate.
    """

    def __init__(
        self,
        surrogate: CombinationSurrogate[Array],
        orthonormal_bases_1d: Sequence[PhysicalDomainBasis1DProtocol[Array]],
    ) -> None:
        if not isinstance(surrogate, CombinationSurrogate):
            raise TypeError(
                "surrogate must be a CombinationSurrogate, got "
                f"{type(surrogate).__name__}"
            )
        self._surrogate = surrogate
        self._bases_1d = list(orthonormal_bases_1d)
        self._pce: Optional[PolynomialChaosExpansion[Array]] = None

    def surrogate(self) -> CombinationSurrogate[Array]:
        """Return the surrogate these moments describe."""
        return self._surrogate

    def pce(self) -> PolynomialChaosExpansion[Array]:
        """Return the converted expansion, converting once."""
        if self._pce is None:
            converter = SparseGridToPCEConverter(
                self._surrogate.bkd(), self._bases_1d
            )
            self._pce = converter.convert(self._surrogate)
        return self._pce

    def mean(self) -> Array:
        """Return E[I_K f], shape (nqoi,)."""
        return pce_statistics.mean(self.pce())

    def variance(self) -> Array:
        """Return Var[I_K f], shape (nqoi,)."""
        return pce_statistics.variance(self.pce())

    def __repr__(self) -> str:
        return f"PCEMoments(nqoi={self._surrogate.nqoi()})"
