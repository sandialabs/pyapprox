"""Error indicators for adaptive sparse grid refinement.

Decoupled from grid internals: each indicator receives a CandidateInfo
and returns (priority, error). The adaptive fitter builds CandidateInfo
objects and passes them here.

Available indicators:
- L2SurrogateDifferenceIndicator: L2 difference on all samples
- L2NewSamplesIndicator: L2 difference on new samples only
- VarianceChangeIndicator: change in variance when candidate is added
- CostWeightedIndicator: wraps any indicator, divides priority by cost
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.sparsegrids.candidate_info import CandidateInfo


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
    """

    def __call__(self, info: CandidateInfo[Array]) -> Tuple[float, float]:
        ...


class L2SurrogateDifferenceIndicator(Generic[Array]):
    """L2 difference between surrogates evaluated on all samples.

    Computes ||I_{sel+k}(x_all) - I_sel(x_all)||_2 / n_all
    where x_all includes samples from selected subspaces plus candidate.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, info: CandidateInfo[Array]) -> Tuple[float, float]:
        """Compute L2 surrogate difference on all samples.

        Parameters
        ----------
        info : CandidateInfo[Array]
            Candidate information.

        Returns
        -------
        Tuple[float, float]
            (priority, error).
        """
        samples = info.all_samples
        nsamples = samples.shape[1]

        old_vals = info.selected_surrogate(samples)
        new_vals = info.sel_plus_candidate_surrogate(samples)
        diff = new_vals - old_vals

        error = float(
            self._bkd.to_numpy(
                self._bkd.sqrt(
                    self._bkd.sum(diff * diff)
                    / max(nsamples, 1)
                )
            )
        )
        return (error, error)


class L2NewSamplesIndicator(Generic[Array]):
    """L2 difference between surrogates evaluated on new samples only.

    Computes ||I_{sel+k}(x_new) - I_sel(x_new)||_2 / n_new
    where x_new are samples unique to the candidate subspace.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, info: CandidateInfo[Array]) -> Tuple[float, float]:
        """Compute L2 surrogate difference on new samples.

        Parameters
        ----------
        info : CandidateInfo[Array]
            Candidate information.

        Returns
        -------
        Tuple[float, float]
            (priority, error).
        """
        samples = info.new_samples
        nsamples = samples.shape[1]

        old_vals = info.selected_surrogate(samples)
        new_vals = info.sel_plus_candidate_surrogate(samples)
        diff = new_vals - old_vals

        error = float(
            self._bkd.to_numpy(
                self._bkd.sqrt(
                    self._bkd.sum(diff * diff)
                    / max(nsamples, 1)
                )
            )
        )
        return (error, error)


class VarianceChangeIndicator(Generic[Array]):
    """Change in mean and variance when candidate is added.

    error = max_qoi(|mean_new - mean_old|) +
            max_qoi(sqrt(|var_new - var_old|))

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(self, info: CandidateInfo[Array]) -> Tuple[float, float]:
        """Compute variance change indicator.

        Parameters
        ----------
        info : CandidateInfo[Array]
            Candidate information.

        Returns
        -------
        Tuple[float, float]
            (priority, error).
        """
        old_mean = info.selected_surrogate.mean()
        new_mean = info.sel_plus_candidate_surrogate.mean()
        old_var = info.selected_surrogate.variance()
        new_var = info.sel_plus_candidate_surrogate.variance()

        mean_change = float(
            self._bkd.to_numpy(self._bkd.max(self._bkd.abs(new_mean - old_mean)))
        )
        var_change = float(
            self._bkd.to_numpy(
                self._bkd.max(
                    self._bkd.sqrt(self._bkd.abs(new_var - old_var))
                )
            )
        )
        error = mean_change + var_change
        return (error, error)


class CostWeightedIndicator(Generic[Array]):
    """Wraps a base indicator and divides priority by subspace cost.

    priority = base_priority / subspace_cost

    For single-fidelity grids where subspace_cost is None,
    falls back to the base indicator unchanged.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    base : ErrorIndicatorProtocol[Array]
        The base error indicator to wrap.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        base: ErrorIndicatorProtocol[Array],
    ) -> None:
        self._bkd = bkd
        self._base = base

    def __call__(self, info: CandidateInfo[Array]) -> Tuple[float, float]:
        """Compute cost-weighted indicator.

        Parameters
        ----------
        info : CandidateInfo[Array]
            Candidate information.

        Returns
        -------
        Tuple[float, float]
            (priority, error) where priority = error / cost.
        """
        priority, error = self._base(info)
        if info.subspace_cost is not None and info.subspace_cost > 0:
            priority = priority / info.subspace_cost
        return (priority, error)
