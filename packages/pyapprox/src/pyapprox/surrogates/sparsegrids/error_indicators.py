"""Error indicators for adaptive sparse grid refinement.

An indicator answers one question: how much does the surrogate move
when this candidate is added? It returns that error alone. The fitter
turns errors into queue order through an injected
``PriorityProtocol``, so cost policy lives in one place instead of
being repeated in every indicator.

Each candidate carries its backward box, and the change in any quantity
linear in the subspaces is a signed sum over that box. Indicators are
therefore scored without building a surrogate.

Available indicators:

- ``L2SurplusIndicator``: RMS surplus on the candidate's new samples.
  The recommended default for dimension-adaptive refinement.
- ``L2GlobalSurplusIndicator``: RMS surplus over every sample in the
  grid. Biased on separable functions, where refining one dimension
  adds points at which the other dimension's surplus is already
  resolved, diluting the average.
- ``SummedSubspaceVarianceIndicator``: change in mean and in the summed
  per-subspace variance.
"""

import warnings
from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.surrogates.sparsegrids.candidate_info import (
    Candidate,
    SampleSourceProtocol,
)
from pyapprox.surrogates.sparsegrids.smolyak import evaluate_box
from pyapprox.surrogates.sparsegrids.statistics.cache import (
    SubspaceCache,
    box_sum,
)
from pyapprox.surrogates.sparsegrids.statistics.subspace_moments import (
    subspace_mean,
    subspace_variance,
)
from pyapprox.surrogates.sparsegrids.subspace import (
    TensorProductSubspace,
)
from pyapprox.util.backends.protocols import Array, Backend

# A new point holding less than this fraction of the rule's total
# weight moves no moment the rule computes, to rounding.
_NEGLIGIBLE_WEIGHT_FRACTION = 1e-12


class QuadratureRefinementStalled(UserWarning):
    """A candidate's new points carry no quadrature weight.

    Refinement continues, but declines to refine the affected
    dimension. Indicators take ``raise_on_stall`` to turn this into
    ``QuadratureRefinementStalledError`` instead.
    """


class QuadratureRefinementStalledError(RuntimeError):
    """``QuadratureRefinementStalled`` raised rather than warned.

    Requested through an indicator's ``raise_on_stall``, so a run that
    would spend its remaining budget without refining the affected
    dimension stops instead of finishing.
    """


def _check_new_points_carry_weight(
    candidate: Candidate[Array],
    error: float,
    already_reported: bool,
    indicator_name: str,
    bkd: Backend[Array],
    raise_on_stall: bool,
) -> bool:
    """Report when zero error is caused by weightless new points.

    An indicator scored from quadrature moments reports no change when
    the candidate's new points carry no quadrature weight: the rule
    integrates as though they were absent, so every moment is
    unchanged and the signed box sum is exactly zero. The candidate
    then looks perfectly resolved and is never promoted, and
    refinement stalls in that dimension while the others absorb the
    budget.

    Zero error has innocent causes too, such as a dimension the target
    does not depend on, so the weights are checked rather than the
    error alone. Both conditions together identify the pathology; the
    error alone would misreport a genuinely resolved candidate.

    A nested sequence grown one point at a time is the common case,
    since a lone new point placed near an already-resolved location,
    typically a domain endpoint, takes almost none of the weight.

    Returns whether the condition has now been reported, so a caller
    can keep this to once per indicator rather than once per candidate.

    Raises
    ------
    QuadratureRefinementStalledError
        If raise_on_stall is set. Refinement that continues from here
        spends its remaining budget without refining this dimension,
        so a caller who knows that is not worth paying for can ask to
        stop instead.
    """
    if already_reported or error != 0.0:
        return already_reported
    local_indices = list(candidate.new_sample_local_indices)
    if len(local_indices) == 0:
        return already_reported
    weights = bkd.abs(bkd.flatten(candidate.subspace.get_quadrature_weights()))
    total = float(bkd.sum(weights))
    if total <= 0.0:
        return already_reported
    new_weight = float(
        bkd.sum(weights[bkd.asarray(local_indices, dtype=bkd.int64_dtype())])
    )
    if new_weight / total >= _NEGLIGIBLE_WEIGHT_FRACTION:
        return already_reported
    message = (
        f"{indicator_name} scored a candidate as zero error because its "
        f"new points hold {new_weight / total:.1e} of the quadrature "
        "rule's weight, so every moment the rule computes is unchanged "
        "and this dimension will not be refined further. Adding points "
        "more than one at a time usually avoids this: use a growth "
        "rule with a step of two or more, such as "
        "LinearGrowthRule(scale=2, shift=1), or a two-point Leja "
        "sequence. A surplus indicator such as L2SurplusIndicator does "
        "not use quadrature weights and is unaffected."
    )
    if raise_on_stall:
        raise QuadratureRefinementStalledError(message)
    warnings.warn(message, QuadratureRefinementStalled, stacklevel=3)
    return True


@runtime_checkable
class ErrorIndicatorProtocol(Protocol[Array]):
    """Scores one candidate.

    Parameters
    ----------
    candidate : Candidate[Array]
        The candidate and its backward box.
    grid : object
        Read-only view of the grid. Implementations should narrow this
        to the protocol they use.

    Returns
    -------
    float
        The error. Higher means the candidate changes the surrogate
        more. Cost is not applied here.
    """

    def __call__(self, candidate: Candidate[Array], grid: object) -> float: ...


def _rms(diff: Array, bkd: Backend[Array]) -> float:
    """Root-mean-square over every entry of a (nqoi, npoints) array."""
    npoints = max(diff.shape[1], 1)
    return bkd.to_float(bkd.sqrt(bkd.sum(diff * diff) / npoints))


class L2SurplusIndicator(Generic[Array]):
    """RMS surplus on the samples the candidate adds.

    error = ||Delta I(x_new)||_2 / sqrt(n_new)

    Measuring only where the candidate contributes new points avoids the
    dilution ``L2GlobalSurplusIndicator`` suffers on separable
    functions.

    This is the change in the interpolant, which is not in general the
    interpolation error. The two coincide for nested rules, where the
    combination reproduces the data at the grid's own nodes so the
    candidate's new points are the only ones not yet reproduced. Leja
    and Clenshaw-Curtis are nested. Gauss rules share no points between
    consecutive levels, so the combination does not reproduce the data
    even at its own nodes and this error is a weaker proxy for the
    residual there. It remains a well-defined measure of how much a
    candidate moves the surrogate, which is what refinement ranks on.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, candidate: Candidate[Array], grid: object) -> float:
        """Return the RMS surplus on the candidate's new samples."""
        samples = candidate.subspace.get_samples()
        local = self._bkd.asarray(
            list(candidate.new_sample_local_indices),
            dtype=self._bkd.int64_dtype(),
        )
        new_samples = samples[:, local]
        return _rms(evaluate_box(candidate.box, new_samples), self._bkd)


class L2GlobalSurplusIndicator(Generic[Array]):
    """RMS surplus over every sample in the grid.

    error = ||Delta I(x_all)||_2 / sqrt(n_all)

    For separable functions this is biased: refining one dimension adds
    many points at which the other dimension's surplus is already
    captured, pulling the average down. Prefer ``L2SurplusIndicator``
    there.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(
        self, candidate: Candidate[Array], grid: SampleSourceProtocol[Array]
    ) -> float:
        """Return the RMS surplus over the grid's samples."""
        samples = grid.get_samples("all")
        if isinstance(samples, dict):
            # Multi-fidelity: score against this candidate's own
            # fidelity, whose subspaces are the ones its box holds.
            cfg = candidate.config_idx
            samples = samples[cfg if cfg is not None else ()]
        return _rms(evaluate_box(candidate.box, samples), self._bkd)


class SummedSubspaceVarianceIndicator(Generic[Array]):
    """Change in mean and in the summed per-subspace variance.

    The quantity is sum_k c_k Var_k, where Var_k is subspace k's
    variance under its own tensor product rule. The name says that
    rather than "variance" because it is neither published variance
    definition: not Var[I_K f], which carries cross terms between
    subspaces, and not Q_K[f^2] - (Q_K f)^2, which applies the
    sparse-grid rule to f^2. Use ``CrossMomentMoments`` or
    ``PCEMoments`` for the first and ``QuadratureMoments`` for the
    second.

    It does coincide with the true variance when every subspace
    resolves f exactly in its own directions, because the Smolyak
    telescoping is then exact for the second moment as well as the
    first. That is not a condition refinement can assume, since an
    unresolved grid is the reason to refine, so this is a proxy for
    ranking candidates rather than an estimate of anything.

    Its advantage is that it needs no selected-set state: the change is
    a pure box sum over the candidate's own backward box.

    The error uses a single quantity of interest for both terms:

        q* = argmax_q |Delta V_q|
        error = |Delta m_{q*}| + sqrt(|Delta V_{q*}|)

    Taking the mean change from the same QoI that dominates the
    variance change keeps the two terms describing one quantity. When
    every Delta V_q is zero, q* is 0 and mean changes in other QoIs are
    ignored.

    The sqrt puts the variance term in the units of the mean term so
    the two can be added. **It is not the change in standard
    deviation**: sqrt of a change is not the change in anything, and it
    overstates small movements, since sqrt(eps) is far larger than eps.
    It also carries a round-off floor near sqrt(eps) |f|, about
    1e-8 |f|. ``QuadratureVarianceIndicator`` and
    ``PCEVarianceIndicator`` report |sigma_new - sigma_old| instead,
    each for the variance it defines.

    The reason this one does not is the property above: |sigma_new -
    sigma_old| needs the selected set's current variance, and reading
    that would give up scoring from the candidate's own box alone.
    This is the cheapest of the three, and the approximation is the
    price.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    cache : SubspaceCache[Array, Array], optional
        Supplies each subspace's stacked [mean, variance]. Pass one to
        share memoized statistics across indicators; by default each
        indicator keeps its own.
    raise_on_stall : bool, optional
        Raise ``QuadratureRefinementStalledError`` instead of warning
        when a candidate's new points carry no quadrature weight. That
        dimension is then never refined, so the rest of the budget is
        spent without it; set this to stop rather than pay for a run
        whose outcome is already determined. Default False.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        cache: Optional["SubspaceCache[Array, Array]"] = None,
        raise_on_stall: bool = False,
    ) -> None:
        self._bkd = bkd
        if cache is None:
            cache = SubspaceCache(self._moments)
        self._cache = cache
        self._raise_on_stall = raise_on_stall
        self._reported_stall = False

    def _moments(self, subspace: TensorProductSubspace[Array]) -> Array:
        """Stack [mean, variance] so one box sum yields both changes."""
        return self._bkd.stack(
            [subspace_mean(subspace), subspace_variance(subspace)], axis=0
        )

    def __call__(self, candidate: Candidate[Array], grid: object) -> float:
        """Return |Delta m| + sqrt(|Delta V|) for the dominant QoI."""
        delta = box_sum(self._cache, candidate.box)
        delta_mean, delta_variance = delta[0], delta[1]
        qstar = self._bkd.to_int(self._bkd.argmax(self._bkd.abs(delta_variance)))
        mean_change = self._bkd.to_float(self._bkd.abs(delta_mean[qstar]))
        # The square root, not the variance change itself: it carries
        # the same units as the mean term so the two can be added.
        root_variance_change = self._bkd.to_float(
            self._bkd.sqrt(self._bkd.abs(delta_variance[qstar]))
        )
        error = mean_change + root_variance_change
        self._reported_stall = _check_new_points_carry_weight(
            candidate,
            error,
            self._reported_stall,
            type(self).__name__,
            self._bkd,
            self._raise_on_stall,
        )
        return error
