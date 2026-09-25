"""Refinement driven by the sparse-grid rule's variance.

``SummedSubspaceVarianceIndicator`` refines on sum_k c_k Var_k, a signed sum of
per-subspace variances that is a pure box sum and needs nothing but the
candidate. This one refines on V = Q_K[f^2] - (Q_K f)^2, the quantity
``QuadratureMoments`` reports, which is not a sum of per-subspace
variances and so needs the selected set's current mean.

Provided so the two definitions can be compared: which one drives
refinement better is an empirical question, and answering it needs both
available.
"""

from typing import Generic, Optional
from weakref import WeakKeyDictionary

from pyapprox.surrogates.sparsegrids.candidate_info import (
    Candidate,
    SelectionSourceProtocol,
    SmolyakSelection,
)
from pyapprox.surrogates.sparsegrids.statistics.cache import (
    SubspaceCache,
    box_sum,
)
from pyapprox.surrogates.sparsegrids.statistics.subspace_moments import (
    subspace_mean,
    subspace_raw_moment,
    variance_delta,
    variance_from_raw_moments,
)
from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array, Backend

# Selected-set moments, keyed weakly by the snapshot they came from so a
# superseded round is not kept alive by having been measured.
_SnapshotCache = WeakKeyDictionary["SmolyakSelection[Array]", Array]


class QuadratureVarianceIndicator(Generic[Array]):
    """Change in mean and in the sparse-grid rule's variance.

    The variance is V = Q_K[f^2] - (Q_K f)^2, matching
    ``QuadratureMoments``. Because Q_K is a signed rule this is not
    guaranteed nonnegative, so the reported change can be driven by a
    quantity that is not a variance. That is a reason to compare this
    against the alternatives rather than adopt it by default.

    Both terms come from raw moments, which combine linearly through
    the Smolyak coefficients where a central moment does not. The
    variance change is taken as

        Delta V = Delta M2 - Delta m (2 m + Delta m)

    which is the change V_new - V_old would give without forming
    either; see ``variance_delta``.

    The error combines the change in the mean with the change in the
    standard deviation, using one quantity of interest for both terms:

        q* = argmax_q |Delta V_q|
        error = |Delta m_{q*}| + |sigma_new - sigma_old|_{q*}

    where sigma = sqrt(|V|) before and after the candidate is added.
    Both terms are then changes in quantities carried on the same
    scale as f, which sqrt(|Delta V|) is not: a square root of a change
    is not the change in anything, and it overstates small movements,
    since sqrt(eps) is far larger than eps. The selected variance this
    needs is free here, because the snapshot already holds the
    selected second moment.

    The mean term is what keeps refinement moving on a target whose
    variance is already resolved, or has none: for a constant function
    every Delta V is zero, and an indicator built on the variance alone
    would report no error anywhere and stall.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    cache : SubspaceCache[Array, Array], optional
        Supplies each subspace's stacked [mean, second raw moment].
        Pass one to share memoized statistics across indicators.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        cache: Optional["SubspaceCache[Array, Array]"] = None,
    ) -> None:
        self._bkd = bkd
        if cache is None:
            cache = SubspaceCache(self._moments)
        self._cache = cache
        # The selected set changes only on promotion, so its combined
        # moments are computed once per round and reused by every
        # candidate scored against it.
        self._snapshots: "_SnapshotCache[Array]" = WeakKeyDictionary()

    def _moments(self, subspace: TensorProductSubspace[Array]) -> Array:
        """Stack [Q_k f, Q_k f^2] so one box sum yields both changes."""
        return self._bkd.stack(
            [subspace_mean(subspace), subspace_raw_moment(subspace, 2)],
            axis=0,
        )

    def _selected_moments(
        self, selection: SmolyakSelection[Array]
    ) -> Array:
        """Return [m, M2] for the selected set, combined once per round."""
        cached = self._snapshots.get(selection)
        if cached is None:
            if len(selection.terms) == 0:
                raise ValueError(
                    "cannot score against an empty selected set"
                )
            cached = box_sum(self._cache, selection.terms)
            self._snapshots[selection] = cached
        return cached

    def __call__(
        self,
        candidate: Candidate[Array],
        grid: SelectionSourceProtocol[Array],
    ) -> float:
        """Return the combined mean and variance change."""
        selected = self._selected_moments(grid.selection())
        delta = box_sum(self._cache, candidate.box)

        mean = selected[0]
        delta_mean = delta[0]
        variance = variance_from_raw_moments(mean, selected[1])
        delta_variance = variance_delta(mean, delta_mean, delta[1])

        qstar = self._bkd.to_int(
            self._bkd.argmax(self._bkd.abs(delta_variance))
        )
        mean_term = self._bkd.to_float(self._bkd.abs(delta_mean[qstar]))
        # abs() because Q_K is signed and can report a negative
        # variance; the standard deviation of such a value is taken
        # from its magnitude rather than left undefined.
        sigma_old = self._bkd.sqrt(self._bkd.abs(variance[qstar]))
        sigma_new = self._bkd.sqrt(
            self._bkd.abs(variance[qstar] + delta_variance[qstar])
        )
        sigma_term = self._bkd.to_float(self._bkd.abs(sigma_new - sigma_old))
        return mean_term + sigma_term

    def __repr__(self) -> str:
        return "QuadratureVarianceIndicator()"
