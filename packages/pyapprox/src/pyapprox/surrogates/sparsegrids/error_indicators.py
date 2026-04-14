"""Error indicators for adaptive sparse grid refinement.

Decoupled from grid internals: each indicator receives a CandidateInfo
and returns (priority, error). The adaptive fitter builds CandidateInfo
objects and passes them here.

All indicators return priority = error / subspace_cost when
subspace_cost is set, giving a surplus-per-unit-work selection metric.
When subspace_cost is None or zero, priority falls back to the raw
error.

Available indicators:
- L2SurplusIndicator: RMS surplus on the candidate's new samples
  (recommended default)
- L2GlobalSurplusIndicator: RMS surplus on all samples in the grid;
  biased on separable functions where one dimension adds few new points
- VarianceChangeIndicator: change in mean and variance when candidate
  is added
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.surrogates.sparsegrids.candidate_info import CandidateInfo
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ErrorIndicatorProtocol(Protocol[Array]):
    """Protocol for error indicators used in adaptive refinement.

    Parameters
    ----------
    info : CandidateInfo[Array]
        All information about the candidate subspace.

    Returns
    -------
    Tuple[float, float]
        (priority, error) where higher priority means refine sooner.
        Implementations divide priority by subspace_cost when set.
    """

    def __call__(self, info: CandidateInfo[Array]) -> Tuple[float, float]: ...


def _weight_by_cost(error: float, info: CandidateInfo[Array]) -> float:
    """Return error / subspace_cost, or error if no valid cost."""
    if info.subspace_cost is not None and info.subspace_cost > 0:
        return error / info.subspace_cost
    return error


class L2GlobalSurplusIndicator(Generic[Array]):
    """RMS surplus evaluated on all samples in the grid.

    error    = ||I_{sel+k}(x_all) - I_sel(x_all)||_2 / sqrt(n_all)
    priority = error / subspace_cost

    Note: for separable functions this indicator is biased --- refining
    one dimension adds many new samples where the other-dimension
    surplus is already captured, diluting the RMS. Prefer
    L2SurplusIndicator in those cases.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, info: CandidateInfo[Array]) -> Tuple[float, float]:
        """Compute cost-weighted L2 surplus on all samples.

        Parameters
        ----------
        info : CandidateInfo[Array]
            Candidate information.

        Returns
        -------
        Tuple[float, float]
            (priority, error) where priority = error / subspace_cost
            and error is the uncost-weighted RMS surplus.
        """
        samples = info.all_samples
        nsamples = samples.shape[1]

        old_vals = info.selected_surrogate(samples)
        new_vals = info.sel_plus_candidate_surrogate(samples)
        diff = new_vals - old_vals

        error = self._bkd.to_float(
            self._bkd.sqrt(self._bkd.sum(diff * diff) / max(nsamples, 1))
        )
        return (_weight_by_cost(error, info), error)


class L2SurplusIndicator(Generic[Array]):
    """Cost-weighted RMS surplus on the candidate's new samples.

    error    = ||I_{sel+k}(x_new) - I_sel(x_new)||_2 / sqrt(n_new)
    priority = error / subspace_cost

    subspace_cost = model_cost(config_idx) * n_new_samples, so for a
    single-fidelity grid with unit cost priority reduces to
    error / n_new_samples.

    This is the recommended default for dimension-adaptive refinement:
    it measures surplus-per-unit-work without the dilution that
    L2GlobalSurplusIndicator suffers on separable functions.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, info: CandidateInfo[Array]) -> Tuple[float, float]:
        """Compute cost-weighted L2 surplus on new samples.

        Parameters
        ----------
        info : CandidateInfo[Array]
            Candidate information.

        Returns
        -------
        Tuple[float, float]
            (priority, error) where priority = error / subspace_cost
            and error is the uncost-weighted RMS surplus on new samples.
        """
        samples = info.new_samples
        nsamples = samples.shape[1]

        old_vals = info.selected_surrogate(samples)
        new_vals = info.sel_plus_candidate_surrogate(samples)
        diff = new_vals - old_vals

        error = self._bkd.to_float(
            self._bkd.sqrt(self._bkd.sum(diff * diff) / max(nsamples, 1))
        )
        return (_weight_by_cost(error, info), error)


class VarianceChangeIndicator(Generic[Array]):
    """Cost-weighted change in mean and variance when candidate is added.

    error    = max_qoi(|mean_new - mean_old|) +
               max_qoi(sqrt(|var_new - var_old|))
    priority = error / subspace_cost

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, info: CandidateInfo[Array]) -> Tuple[float, float]:
        """Compute cost-weighted variance change indicator.

        Parameters
        ----------
        info : CandidateInfo[Array]
            Candidate information.

        Returns
        -------
        Tuple[float, float]
            (priority, error) where priority = error / subspace_cost
            and error is the uncost-weighted mean+variance change.
        """
        old_mean = info.selected_surrogate.mean()
        new_mean = info.sel_plus_candidate_surrogate.mean()
        old_var = info.selected_surrogate.variance()
        new_var = info.sel_plus_candidate_surrogate.variance()

        mean_change = self._bkd.to_float(
            self._bkd.max(self._bkd.abs(new_mean - old_mean))
        )
        var_change = self._bkd.to_float(
            self._bkd.max(self._bkd.sqrt(self._bkd.abs(new_var - old_var)))
        )
        error = mean_change + var_change
        return (_weight_by_cost(error, info), error)
