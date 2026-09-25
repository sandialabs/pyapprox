"""Refinement driven by the surrogate's own variance.

``SummedSubspaceVarianceIndicator`` refines on sum_k c_k Var_k, a proxy.
``QuadratureVarianceIndicator`` refines on Q_K[f^2] - (Q_K f)^2, the
sparse-grid rule applied to f^2. This one refines on Var[I_K f], the
exact variance of the interpolant itself, which is what
``PCEMoments`` reports.

Converting each subspace to an orthonormal expansion makes the mean and
variance read off the coefficients: E = c_0 and Var = sum over nonzero
multi-indices of c_alpha^2. Both are linear in the subspaces through
the coefficients, so a candidate's change is a signed sum over its
backward box, like every other indicator here.
"""

from typing import Dict, Generic, Optional, Sequence, Tuple
from weakref import WeakKeyDictionary

from pyapprox.surrogates.affine.protocols import (
    PhysicalDomainBasis1DProtocol,
)
from pyapprox.surrogates.sparsegrids.candidate_info import (
    Candidate,
    SelectionSourceProtocol,
    SmolyakSelection,
)
from pyapprox.surrogates.sparsegrids.converters.pce import (
    TensorProductSubspaceToPCEConverter,
    merge_coefficients_by_index,
    subspace_coefficients_by_index,
)
from pyapprox.surrogates.sparsegrids.statistics.cache import SubspaceCache
from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array, Backend

# Multi-index to (nqoi,) coefficients. Subspaces carry different index
# sets, so combining them merges by key rather than adding elementwise.
KeyedCoefficients = Dict[Tuple[int, ...], Array]

# Selected-set coefficients, keyed weakly by the snapshot they came from
# so a superseded round is not kept alive by having been measured.
_SnapshotCache = WeakKeyDictionary["SmolyakSelection[Array]", KeyedCoefficients[Array]]


class PCEVarianceIndicator(Generic[Array]):
    """Change in mean and in the standard deviation of ``I_K f``.

    The variance is Var[I_K f] = sum over nonzero alpha of c_alpha^2,
    matching ``PCEMoments``. Unlike the sparse-grid rule's variance this
    is a sum of squares and so is never negative, and unlike the summed
    per-subspace variance it is the surrogate's own.

    The coefficients combine linearly through the Smolyak coefficients,
    so the candidate's change comes from a signed box sum without
    rebuilding a surrogate. The variance change is taken as

        Delta V = sum over nonzero alpha of
                  (2 c_alpha Delta c_alpha + Delta c_alpha^2)

    which is what Var_new - Var_old gives, without forming either: the
    cancellation that subtracting two nearly equal variances suffers is
    never entered.

    The error combines the change in the mean with the change in the
    standard deviation, using one quantity of interest for both terms:

        q* = argmax_q |Delta V_q|
        error = |Delta m_{q*}| + |sigma_new - sigma_old|_{q*}

    matching ``QuadratureVarianceIndicator`` so the two definitions can
    be compared on equal footing. Both terms are then changes in
    quantities carried on the same scale as f.

    The mean term is what keeps refinement moving on a target whose
    variance is already resolved, or has none: for a constant function
    every Delta V is zero, and an indicator built on the variance alone
    would report no error anywhere and stall.

    **Restricted to globally polynomial interpolation bases**, which
    means Gauss, Leja and Clenshaw-Curtis. The conversion projects each
    Lagrange function onto the orthonormal polynomials by Gauss
    quadrature, which is exact only because those functions are
    polynomials; a piecewise basis has breakpoints the quadrature
    cannot see. The converter raises for such a basis rather than
    scoring against coefficients that do not describe the interpolant.
    ``SummedSubspaceVarianceIndicator`` and
    ``QuadratureVarianceIndicator`` carry no such restriction.

    Multi-fidelity: each subspace converts its own fidelity's values, so
    the combined coefficients are the multi-index combination across
    fidelities, and the reported change is that of the combined
    surrogate rather than of any one fidelity.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    orthonormal_bases_1d : Sequence[PhysicalDomainBasis1DProtocol[Array]]
        Univariate bases, from ``create_bases_1d(marginals, bkd)``. They
        must return physical-domain quadrature points, or the spectral
        projection is wrong for non-canonical domains.
    cache : SubspaceCache[Array, KeyedCoefficients[Array]], optional
        Supplies each subspace's keyed coefficients. Pass one to share
        conversions across indicators.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        orthonormal_bases_1d: Sequence[PhysicalDomainBasis1DProtocol[Array]],
        cache: Optional["SubspaceCache[Array, KeyedCoefficients[Array]]"] = None,
    ) -> None:
        self._bkd = bkd
        self._nvars = len(orthonormal_bases_1d)
        self._converter = TensorProductSubspaceToPCEConverter(
            bkd, orthonormal_bases_1d
        )
        if cache is None:
            cache = SubspaceCache(self._coefficients)
        self._cache = cache
        # The selected set changes only on promotion, so its combined
        # coefficients are computed once per round and reused by every
        # candidate scored against it.
        self._snapshots: "_SnapshotCache[Array]" = WeakKeyDictionary()
        self._constant_key = (0,) * self._nvars

    def _coefficients(
        self, subspace: TensorProductSubspace[Array]
    ) -> KeyedCoefficients[Array]:
        """Convert one subspace and key its coefficients by multi-index."""
        indices, coefficients = self._converter.convert_subspace(subspace)
        return subspace_coefficients_by_index(
            indices, coefficients, self._nvars, self._bkd
        )

    def _box_sum(
        self,
        terms: Sequence[Tuple[int, TensorProductSubspace[Array]]],
    ) -> KeyedCoefficients[Array]:
        """Signed sum of the cached coefficients over a box.

        ``statistics.cache.box_sum`` folds a statistic that adds
        elementwise. These do not: two subspaces hold different
        multi-indices, so they merge by key.
        """
        if len(terms) == 0:
            raise ValueError("cannot sum over an empty box")
        total: KeyedCoefficients[Array] = {}
        for sign, subspace in terms:
            merge_coefficients_by_index(
                total,
                {
                    key: sign * coefs
                    for key, coefs in self._cache.get(subspace).items()
                },
            )
        return total

    def _selected_coefficients(
        self, selection: SmolyakSelection[Array]
    ) -> KeyedCoefficients[Array]:
        """Return the selected set's coefficients, combined once a round."""
        cached = self._snapshots.get(selection)
        if cached is None:
            if len(selection.terms) == 0:
                raise ValueError(
                    "cannot score against an empty selected set"
                )
            cached = self._box_sum(selection.terms)
            self._snapshots[selection] = cached
        return cached

    def _variance(self, coefs: KeyedCoefficients[Array]) -> Array:
        """Sum of squares over the nonconstant multi-indices."""
        total = self._zeros_like_entry(coefs)
        for key, value in coefs.items():
            if key != self._constant_key:
                total = total + value**2
        return total

    def _variance_delta(
        self, selected: KeyedCoefficients[Array], delta: KeyedCoefficients[Array]
    ) -> Array:
        """Return sum over nonzero alpha of 2 c dc + dc^2.

        A multi-index the candidate introduces has no selected
        coefficient, and contributes dc^2 alone.
        """
        total = self._zeros_like_entry(delta)
        for key, dcoef in delta.items():
            if key == self._constant_key:
                continue
            coef = selected.get(key)
            if coef is None:
                total = total + dcoef**2
            else:
                total = total + dcoef * (2.0 * coef + dcoef)
        return total

    def _zeros_like_entry(self, coefs: KeyedCoefficients[Array]) -> Array:
        """Return a zero accumulator shaped like one coefficient entry.

        The (nqoi,) shape is taken from the coefficients themselves
        rather than stored, so this indicator needs no separate notion
        of how many quantities of interest the grid carries.
        """
        for value in coefs.values():
            return self._bkd.zeros(tuple(value.shape))
        raise ValueError("no coefficients to size from")

    def _mean(self, coefs: KeyedCoefficients[Array]) -> Array:
        """Return the constant coefficient, or zeros when absent."""
        constant = coefs.get(self._constant_key)
        if constant is None:
            return self._zeros_like_entry(coefs)
        return constant

    def __call__(
        self,
        candidate: Candidate[Array],
        grid: SelectionSourceProtocol[Array],
    ) -> float:
        """Return the combined mean and standard deviation change."""
        selected = self._selected_coefficients(grid.selection())
        delta = self._box_sum(candidate.box)

        delta_mean = self._mean(delta)
        variance = self._variance(selected)
        delta_variance = self._variance_delta(selected, delta)

        qstar = self._bkd.to_int(
            self._bkd.argmax(self._bkd.abs(delta_variance))
        )
        mean_term = self._bkd.to_float(self._bkd.abs(delta_mean[qstar]))
        # A sum of squares cannot be negative, so no absolute value is
        # needed here, unlike the signed quadrature rule's variance.
        # Rounding can still put the sum a little below zero, and the
        # maximum keeps the square root real.
        sigma_old = self._bkd.sqrt(
            self._bkd.maximum(variance[qstar], self._bkd.asarray(0.0))
        )
        sigma_new = self._bkd.sqrt(
            self._bkd.maximum(
                variance[qstar] + delta_variance[qstar],
                self._bkd.asarray(0.0),
            )
        )
        sigma_term = self._bkd.to_float(self._bkd.abs(sigma_new - sigma_old))
        return mean_term + sigma_term

    def __repr__(self) -> str:
        return "PCEVarianceIndicator()"
