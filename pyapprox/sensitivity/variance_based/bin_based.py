"""Bin-based variance sensitivity analysis.

This module provides bin-based estimation of Sobol sensitivity indices
using the algorithm from Borgonovo et al. (2016).

References
----------
Borgonovo, E., Hazen, G. and Plischke, E. (2016). A Common Rationale for
Global Sensitivity Measures and Their Estimation. Risk Analysis, 36(10):1871-1895.
https://doi.org/10.1111/risa.12555
"""

import warnings
from typing import Dict, Generic, List, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.protocols import JointDistributionProtocol
from pyapprox.sensitivity.variance_based.base import (
    VarianceBasedSensitivityAnalysis,
)


class BinBasedSensitivityAnalysis(
    VarianceBasedSensitivityAnalysis[Array], Generic[Array]
):
    """Bin-based variance sensitivity analysis for main effects and interactions.

    Estimates Sobol indices by partitioning each variable's domain into bins
    and computing conditional expectations within each n-dimensional cell.

    This method works with any existing sample set and does not require the
    special matrix structure needed by Saltelli/Jansen estimators.

    Note: This method does NOT compute total effects, which would require
    (d-1)-dimensional binning and is infeasible for d>3. Use
    SampleBasedSensitivityAnalysis for total effects.

    Parameters
    ----------
    distribution : JointDistributionProtocol[Array]
        Joint distribution with marginals() method. Each marginal must have
        invcdf() for computing bin boundaries.
    bkd : Backend[Array]
        Backend for array operations.
    nbins : List[int], optional
        Number of bins per dimension for each interaction order.
        nbins[0] is for 1st order, nbins[1] for 2nd order, etc.
        If None, uses adaptive formula: nbins ~ nsamples^(1/(3*order)).
    eps : float, optional
        Tolerance for bin boundaries. Bins span [eps, 1-eps] in CDF space
        to avoid issues with unbounded distributions. Default is 0.0.
    clip_negative : bool, optional
        Whether to clip negative Sobol indices to 0. Default is True.
        Negative values can occur due to estimation error.

    Examples
    --------
    >>> from pyapprox.sensitivity.variance_based import (
    ...     BinBasedSensitivityAnalysis,
    ... )
    >>> # Assuming `prior` is a JointDistributionProtocol
    >>> sa = BinBasedSensitivityAnalysis(prior, bkd)
    >>> samples = prior.rvs(10000)
    >>> values = my_function(samples)  # Shape: (nqoi, nsamples)
    >>> sa.compute(samples, values)
    >>> main = sa.main_effects()
    >>> sobol = sa.sobol_indices()

    References
    ----------
    .. [BHPRA2016] Borgonovo, E., Hazen, G. and Plischke, E. (2016).
       A Common Rationale for Global Sensitivity Measures and Their Estimation.
       Risk Analysis, 36(10):1871-1895.
    """

    def __init__(
        self,
        distribution: JointDistributionProtocol[Array],
        bkd: Backend[Array],
        nbins: Optional[List[int]] = None,
        eps: float = 0.0,
        clip_negative: bool = True,
    ) -> None:
        if not isinstance(distribution, JointDistributionProtocol):
            raise TypeError(
                "distribution must satisfy JointDistributionProtocol, "
                f"got {type(distribution).__name__}"
            )
        super().__init__(distribution.nvars(), bkd)
        self._distribution = distribution
        self._nbins = nbins
        self._eps = eps
        self._clip_negative = clip_negative
        self._mean: Optional[Array] = None
        self._variance: Optional[Array] = None
        self._main_effects_: Optional[Array] = None
        self._sobol_indices_: Optional[Array] = None
        self._raw_variances: Dict[Tuple[int, ...], Array] = {}

    def _get_nbins(self, nsamples: int, order: int) -> int:
        """Get number of bins for a given interaction order.

        Parameters
        ----------
        nsamples : int
            Number of samples available.
        order : int
            Interaction order (1 for main effects, 2 for pairwise, etc.).

        Returns
        -------
        int
            Number of bins per dimension for this order.
        """
        if self._nbins is not None and order <= len(self._nbins):
            return self._nbins[order - 1]
        # Adaptive: total cells ~ nsamples^(1/3)
        # nbins^order ~ nsamples^(1/3) => nbins ~ nsamples^(1/(3*order))
        return max(2, int(nsamples ** (1.0 / (3 * order))))

    def _compute_bin_boundaries(self, nbins: int) -> List[Array]:
        """Compute bin boundaries for each marginal.

        Parameters
        ----------
        nbins : int
            Number of bins per dimension.

        Returns
        -------
        List[Array]
            List of length nvars, each element is Array of shape (nbins+1,)
            containing bin boundaries in the original variable space.
        """
        marginals = self._distribution.marginals()
        boundaries: List[Array] = []
        probs = self._bkd.linspace(self._eps, 1 - self._eps, nbins + 1)
        probs_2d = self._bkd.reshape(probs, (1, -1))

        for marginal in marginals:
            bounds = marginal.invcdf(probs_2d)
            boundaries.append(self._bkd.reshape(bounds, (-1,)))

        return boundaries

    def _compute_nd_bin_indices(
        self,
        samples: Array,
        active_vars: List[int],
        bin_boundaries: List[Array],
        nbins: int,
    ) -> Array:
        """Compute flat n-dimensional bin index for each sample.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        active_vars : List[int]
            Indices of active variables.
        bin_boundaries : List[Array]
            Bin boundaries for each marginal.
        nbins : int
            Number of bins per dimension.

        Returns
        -------
        Array
            Flattened bin index for each sample. Shape: (nsamples,)
            Index ranges from 0 to nbins^nactive - 1.
        """
        nsamples = samples.shape[1]
        flat_idx = self._bkd.zeros((nsamples,))

        for k, var_idx in enumerate(active_vars):
            bounds = bin_boundaries[var_idx]
            var_samples = samples[var_idx, :]

            # Find bin index using searchsorted
            # searchsorted returns index where element would be inserted
            # We use right side and subtract 1 to get bin index
            bin_idx = self._bkd.searchsorted(bounds, var_samples, side="right") - 1
            bin_idx = self._bkd.clip(bin_idx, 0, nbins - 1)

            # Add contribution to flat index
            multiplier = nbins**k
            flat_idx = flat_idx + bin_idx * multiplier

        return flat_idx

    def _compute_interaction_variance(
        self,
        samples: Array,
        values: Array,
        interaction_index: Array,
        bin_boundaries_dict: Dict[int, List[Array]],
        nsamples: int,
    ) -> Array:
        """Compute Var[E[Y|X_I]] for a given interaction term.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Output values. Shape: (nqoi, nsamples)
        interaction_index : Array
            Binary indicator of active variables. Shape: (nvars,)
        bin_boundaries_dict : Dict[int, List[Array]]
            Bin boundaries for each order (keyed by order).
        nsamples : int
            Number of samples.

        Returns
        -------
        Array
            Conditional variance for this interaction. Shape: (nqoi,)
        """
        # Find active variables
        mask = self._bkd.asarray(interaction_index > 0)
        active_vars_arr = self._bkd.where(mask)[0]
        active_vars = [int(v) for v in self._bkd.to_numpy(active_vars_arr)]
        nactive = len(active_vars)

        if nactive == 0:
            nqoi = values.shape[0]
            return self._bkd.zeros((nqoi,))

        # Check cache
        cache_key = tuple(sorted(active_vars))
        if cache_key in self._raw_variances:
            return self._raw_variances[cache_key]

        # Get bin boundaries for this order
        order = nactive
        if order not in bin_boundaries_dict:
            nbins = self._get_nbins(nsamples, order)
            bin_boundaries_dict[order] = self._compute_bin_boundaries(nbins)

        bin_boundaries = bin_boundaries_dict[order]
        nbins = self._get_nbins(nsamples, order)

        # Compute total number of bins
        total_bins = nbins**nactive

        # Compute flattened bin index for each sample
        bin_indices = self._compute_nd_bin_indices(
            samples, active_vars, bin_boundaries, nbins
        )
        bin_indices_np = self._bkd.to_numpy(bin_indices).astype(int)

        # Initialize accumulators
        nqoi = values.shape[0]
        bin_sums = self._bkd.zeros((total_bins, nqoi))
        bin_counts = self._bkd.zeros((total_bins,))

        # Accumulate using loop (vectorized accumulation via scatter_add not
        # available in all backends)
        values_T = self._bkd.transpose(values)  # Shape: (nsamples, nqoi)
        for ii in range(nsamples):
            bin_idx = bin_indices_np[ii]
            if 0 <= bin_idx < total_bins:
                bin_sums[bin_idx, :] = bin_sums[bin_idx, :] + values_T[ii, :]
                bin_counts[bin_idx] = bin_counts[bin_idx] + 1

        # Warn if many empty cells
        zero = self._bkd.zeros((1,))
        empty_mask = bin_counts == zero[0]
        n_empty = int(self._bkd.to_numpy(self._bkd.sum(self._bkd.asarray(empty_mask))))
        if n_empty / total_bins > 0.3:
            warnings.warn(
                f"{n_empty}/{total_bins} cells empty for order-{order} "
                f"interaction. Consider fewer bins or more samples.",
                stacklevel=3,
            )

        # Compute global mean
        global_mean = self._bkd.mean(values, axis=1)  # Shape: (nqoi,)

        # Compute weighted variance of bin means
        conditional_variance = self._bkd.zeros((nqoi,))

        for bin_idx in range(total_bins):
            count = bin_counts[bin_idx]
            if count > 0:
                bin_mean = bin_sums[bin_idx, :] / count
                weight = count / nsamples
                conditional_variance = (
                    conditional_variance + weight * (bin_mean - global_mean) ** 2
                )

        # Cache result
        self._raw_variances[cache_key] = conditional_variance

        return conditional_variance

    def compute(self, samples: Array, values: Array) -> None:
        """Compute sensitivity indices from samples and function values.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Output values. Shape: (nqoi, nsamples)

        Raises
        ------
        ValueError
            If samples and values have incompatible shapes.
        """
        # Validate inputs
        if samples.ndim != 2:
            raise ValueError(f"samples must be 2D, got {samples.ndim}D")
        if values.ndim != 2:
            raise ValueError(f"values must be 2D, got {values.ndim}D")
        if samples.shape[1] != values.shape[1]:
            raise ValueError(
                f"samples and values must have same number of columns, "
                f"got {samples.shape[1]} and {values.shape[1]}"
            )

        nvars, nsamples = samples.shape
        nqoi = values.shape[0]

        # Clear cache
        self._raw_variances = {}

        # Compute global statistics
        self._mean = self._bkd.mean(values, axis=1)
        self._variance = self._bkd.var(values, axis=1)

        # Ensure interaction terms are set
        if not hasattr(self, "_interaction_terms"):
            self.set_interaction_terms_of_interest(self._default_interaction_terms())

        # Bin boundaries dict, keyed by order
        bin_boundaries_dict: Dict[int, List[Array]] = {}

        # Compute uncorrected variance ratios for each interaction term
        nterms = self._interaction_terms.shape[1]
        interaction_variances = self._bkd.zeros((nterms, nqoi))

        for ii in range(nterms):
            interaction_index = self._interaction_terms[:, ii]
            conditional_var = self._compute_interaction_variance(
                samples, values, interaction_index, bin_boundaries_dict, nsamples
            )
            # Divide by total variance to get ratio
            # Add small epsilon to avoid division by zero
            eps = self._bkd.asarray([1e-15])
            interaction_variances[ii, :] = conditional_var / (self._variance + eps)

        # Use base class method to correct for lower-order contributions
        # This performs ANOVA subtraction: S_ij = R_ij - S_i - S_j
        self._sobol_indices_ = self._correct_interaction_variance_ratios(
            interaction_variances
        )

        # Optionally clip negative indices
        if self._clip_negative:
            self._sobol_indices_ = self._bkd.maximum(
                self._sobol_indices_, self._bkd.zeros_like(self._sobol_indices_)
            )

        # Extract main effects (terms with exactly one active variable)
        term_sums = self._bkd.sum(self._interaction_terms, axis=0)
        one = self._bkd.ones((1,))
        main_effect_mask = self._bkd.asarray(term_sums == one[0])
        main_effect_indices = self._bkd.where(main_effect_mask)[0]
        self._main_effects_ = self._sobol_indices_[main_effect_indices, :]

    def main_effects(self) -> Array:
        """Return first-order (main effect) Sobol indices.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - main effect index for each variable and QoI.

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if self._main_effects_ is None:
            raise RuntimeError("Must call compute() first")
        return self._main_effects_

    def total_effects(self) -> Array:
        """Not implemented for bin-based method.

        Total effects require (d-1)-dimensional binning which is infeasible
        for d>3. Use SampleBasedSensitivityAnalysis for total effects.

        Raises
        ------
        NotImplementedError
            Always, as total effects are not supported.
        """
        raise NotImplementedError(
            "Bin-based method cannot compute total effects (would require "
            "(d-1)-dimensional binning). Use SampleBasedSensitivityAnalysis."
        )

    def sobol_indices(self) -> Array:
        """Return all computed Sobol indices (main effects + interactions).

        Returns
        -------
        Array
            Shape (nterms, nqoi) - Sobol index for each interaction term.

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if self._sobol_indices_ is None:
            raise RuntimeError("Must call compute() first")
        return self._sobol_indices_

    def mean(self) -> Array:
        """Return the mean of the output.

        Returns
        -------
        Array
            Shape (nqoi,) - mean for each QoI.

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if self._mean is None:
            raise RuntimeError("Must call compute() first")
        return self._mean

    def variance(self) -> Array:
        """Return the variance of the output.

        Returns
        -------
        Array
            Shape (nqoi,) - variance for each QoI.

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if self._variance is None:
            raise RuntimeError("Must call compute() first")
        return self._variance

    def nqoi(self) -> int:
        """Return the number of quantities of interest.

        Returns
        -------
        int
            Number of QoIs.

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if self._mean is None:
            raise RuntimeError("Must call compute() first")
        return int(self._mean.shape[0])

    def bootstrap(
        self,
        samples: Array,
        values: Array,
        nbootstraps: int = 10,
        seed: Optional[int] = None,
    ) -> Dict[str, Array]:
        """Compute bootstrap uncertainty estimates for main effects.

        Parameters
        ----------
        samples : Array
            Input samples. Shape: (nvars, nsamples)
        values : Array
            Output values. Shape: (nqoi, nsamples)
        nbootstraps : int, optional
            Number of bootstrap resamples. Default is 10.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, Array]
            Dictionary with keys:
            - 'median': Shape (nvars, nqoi)
            - 'min': Shape (nvars, nqoi)
            - 'max': Shape (nvars, nqoi)
            - 'quantile_25': Shape (nvars, nqoi)
            - 'quantile_75': Shape (nvars, nqoi)
            - 'std': Shape (nvars, nqoi)
        """
        import numpy as np

        if seed is not None:
            np.random.seed(seed)

        nsamples = samples.shape[1]

        # Compute original indices
        self.compute(samples, values)
        main_effects_list = [self._bkd.to_numpy(self.main_effects())]

        # Bootstrap resamples
        for _ in range(nbootstraps):
            # Sample with replacement
            indices = np.random.choice(nsamples, nsamples, replace=True)
            psamples = samples[:, indices]
            pvalues = values[:, indices]

            # Clear cache for fresh computation
            self._raw_variances = {}
            self._sobol_indices_ = None
            self._main_effects_ = None

            self.compute(psamples, pvalues)
            main_effects_list.append(self._bkd.to_numpy(self.main_effects()))

        # Stack all bootstrap results
        all_effects = np.stack(main_effects_list, axis=0)  # (nbootstraps+1, nvars, nqoi)

        # Compute statistics
        stats = {
            "median": self._bkd.asarray(np.median(all_effects, axis=0)),
            "min": self._bkd.asarray(np.min(all_effects, axis=0)),
            "max": self._bkd.asarray(np.max(all_effects, axis=0)),
            "quantile_25": self._bkd.asarray(np.percentile(all_effects, 25, axis=0)),
            "quantile_75": self._bkd.asarray(np.percentile(all_effects, 75, axis=0)),
            "std": self._bkd.asarray(np.std(all_effects, axis=0)),
        }

        return stats

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nvars={self._nvars}, "
            f"nbins={self._nbins}, eps={self._eps})"
        )
