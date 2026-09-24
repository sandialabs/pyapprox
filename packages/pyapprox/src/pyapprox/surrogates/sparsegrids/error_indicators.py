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
- ``VarianceChangeIndicator``: change in mean and in the summed
  per-subspace variance.
"""

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


class VarianceChangeIndicator(Generic[Array]):
    """Change in mean and in the summed per-subspace variance.

    The variance here is sum_k c_k Var_k, a **refinement proxy rather
    than an estimate of the surrogate's variance**. It ranks candidates
    by how much unresolved structure they expose, which is what
    refinement needs, and it is a pure box sum requiring no
    selected-set state. It is not Var[I_K f] and does not converge to
    it. For the variance of a fitted surrogate use ``PCEMoments``, or
    ``QuadratureMoments`` for the sparse-grid rule applied to f^2.

    The error uses a single quantity of interest for both terms:

        q* = argmax_q |Delta V_q|
        error = |Delta m_{q*}| + sqrt(|Delta V_{q*}|)

    Taking the mean change from the same QoI that dominates the
    variance change keeps the two terms describing one quantity. When
    every Delta V_q is zero, q* is 0 and mean changes in other QoIs are
    ignored.

    The sqrt puts the variance term on the scale of the mean term, at
    the cost of a round-off floor near sqrt(eps) |f|, about 1e-8 |f|.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    cache : SubspaceCache[Array, Array], optional
        Supplies each subspace's stacked [mean, variance]. Pass one to
        share memoized statistics across indicators; by default each
        indicator keeps its own.
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

    def _moments(self, subspace: TensorProductSubspace[Array]) -> Array:
        """Stack [mean, variance] so one box sum yields both changes."""
        return self._bkd.stack(
            [subspace_mean(subspace), subspace_variance(subspace)], axis=0
        )

    def __call__(self, candidate: Candidate[Array], grid: object) -> float:
        """Return the combined mean and variance change."""
        delta = box_sum(self._cache, candidate.box)
        delta_mean, delta_variance = delta[0], delta[1]
        qstar = self._bkd.to_int(self._bkd.argmax(self._bkd.abs(delta_variance)))
        mean_change = self._bkd.to_float(self._bkd.abs(delta_mean[qstar]))
        variance_change = self._bkd.to_float(
            self._bkd.sqrt(self._bkd.abs(delta_variance[qstar]))
        )
        return mean_change + variance_change
