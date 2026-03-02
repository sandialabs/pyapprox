"""Sample-based sensitivity analysis.

This module provides Monte Carlo and quasi-Monte Carlo based
sensitivity analysis methods (Sobol indices from samples).

References
----------
I.M. Sobol. Mathematics and Computers in Simulation 55 (2001) 271-280

Saltelli, Annoni et. al, Variance based sensitivity analysis of model
output. Design and estimator for the total sensitivity index. 2010.
https://doi.org/10.1016/j.cpc.2009.09.018
"""

from abc import abstractmethod
from typing import Generic, List, Optional, Tuple

from pyapprox.expdesign.quadrature.halton import (
    DistributionWithInvCDF,
    HaltonSampler,
)
from pyapprox.expdesign.quadrature.sobol import SobolSampler
from pyapprox.sensitivity.variance_based.base import (
    VarianceBasedSensitivityAnalysis,
)
from pyapprox.util.backends.protocols import Array, Backend


class SampleBasedSensitivityAnalysis(
    VarianceBasedSensitivityAnalysis[Array], Generic[Array]
):
    """Sample-based (Monte Carlo) sensitivity analysis.

    Computes Sobol indices using the Saltelli/Jansen estimators from
    two independent sample sets A and B, along with mixed matrices A_B^I.

    This is an abstract base class. Use one of the concrete implementations:
    - MonteCarloSensitivityAnalysis
    - SobolSequenceSensitivityAnalysis
    - HaltonSequenceSensitivityAnalysis

    Parameters
    ----------
    distribution : DistributionWithInvCDF[Array]
        Distribution from which to sample. Must have `nvars()` and
        `invcdf()` methods (or `rvs()` for Monte Carlo).
    bkd : Backend[Array]
        Backend for array operations.

    Notes
    -----
    The estimators used cannot guarantee that Sobol indices are in [0, 1]
    or sum to 1 due to finite sample effects. Main effects could be negative
    and total effects could exceed 1.

    Examples
    --------
    >>> from pyapprox.sensitivity.variance_based import (
    ...     SobolSequenceSensitivityAnalysis,
    ... )
    >>> # Assuming `dist` is a distribution with invcdf() method
    >>> sa = SobolSequenceSensitivityAnalysis(dist, bkd)
    >>> samples = sa.generate_samples(1000)
    >>> values = my_function(samples)  # Shape: (nqoi, nsamples)
    >>> sa.compute(values)
    >>> main = sa.main_effects()
    >>> total = sa.total_effects()
    """

    def __init__(
        self,
        distribution: DistributionWithInvCDF[Array],
        bkd: Backend[Array],
    ) -> None:
        if not isinstance(distribution, DistributionWithInvCDF):
            raise TypeError(
                "distribution must satisfy DistributionWithInvCDF, "
                f"got {type(distribution).__name__}"
            )
        super().__init__(distribution.nvars(), bkd)
        self._distribution = distribution
        self._samplesA: Optional[Array] = None
        self._samplesB: Optional[Array] = None
        self._samplesAB: Optional[List[Array]] = None
        self._mean: Optional[Array] = None
        self._variance: Optional[Array] = None
        self._main_effects_: Optional[Array] = None
        self._total_effects_: Optional[Array] = None
        self._sobol_indices_: Optional[Array] = None

    @abstractmethod
    def _get_AB_samples(self, nsamples: int) -> Tuple[Array, Array]:
        """Generate the two independent sample sets A and B.

        Parameters
        ----------
        nsamples : int
            Number of samples in each set.

        Returns
        -------
        samplesA : Array
            Shape (nvars, nsamples) - first sample set.
        samplesB : Array
            Shape (nvars, nsamples) - second sample set.
        """
        ...

    def _sobol_index_samples(self, sobol_index: Array) -> Array:
        """Generate mixed sample matrix A_B^I.

        Given two sample sets A and B, generate the set A_B^I where
        rows from B replace rows from A for indices in I (where sobol_index=1).

        Parameters
        ----------
        sobol_index : Array
            Shape (nvars,) - binary indicator of which variables to take from B.

        Returns
        -------
        Array
            Shape (nvars, nsamples) - mixed sample matrix.
        """
        assert self._samplesA is not None and self._samplesB is not None
        # Start with copy of A, replace rows where sobol_index==1 with B
        samples = self._bkd.copy(self._samplesA)
        for ii in range(self._nvars):
            if sobol_index[ii] > 0:
                samples[ii, :] = self._samplesB[ii, :]
        return samples

    def generate_samples(self, nsamples: int) -> Array:
        """Generate all samples needed for sensitivity analysis.

        Generates samples A, B, and all mixed matrices A_B^I for the
        interaction terms of interest.

        Parameters
        ----------
        nsamples : int
            Number of samples in each of the base sets A and B.
            Total samples returned = nsamples * (2 + nterms).

        Returns
        -------
        Array
            Shape (nvars, nsamples * (2 + nterms)) - all samples concatenated.
            Order: [A, B, A_B^{I_1}, A_B^{I_2}, ...]
        """
        if not hasattr(self, "_interaction_terms"):
            self.set_interaction_terms_of_interest(self._default_interaction_terms())
        self._samplesA, self._samplesB = self._get_AB_samples(nsamples)
        self._samplesAB = []
        for ii in range(self._interaction_terms.shape[1]):
            sobol_index = self._interaction_terms[:, ii]
            self._samplesAB.append(self._sobol_index_samples(sobol_index))
        return self._bkd.hstack([self._samplesA, self._samplesB] + self._samplesAB)

    def _unpack_values(self, values: Array) -> Tuple[Array, Array, List[Array]]:
        """Unpack function values into A, B, and AB components.

        Parameters
        ----------
        values : Array
            Shape (nqoi, nsamples_total) - function values at all samples.

        Returns
        -------
        valuesA : Array
            Shape (nqoi, nsamples) - values at sample set A.
        valuesB : Array
            Shape (nqoi, nsamples) - values at sample set B.
        valuesAB : List[Array]
            List of values at each mixed sample set A_B^I.
        """
        assert self._samplesA is not None
        assert self._samplesB is not None
        assert self._samplesAB is not None

        cnt = 0
        valuesA = values[:, cnt : cnt + self._samplesA.shape[1]]
        cnt += self._samplesA.shape[1]
        valuesB = values[:, cnt : cnt + self._samplesB.shape[1]]
        cnt += self._samplesB.shape[1]
        valuesAB: List[Array] = []
        for ii in range(self._interaction_terms.shape[1]):
            valuesAB.append(values[:, cnt : cnt + self._samplesAB[ii].shape[1]])
            cnt += self._samplesAB[ii].shape[1]
        return valuesA, valuesB, valuesAB

    def compute(self, values: Array) -> None:
        """Compute sensitivity indices from function values.

        Uses the Saltelli/Jansen estimators to compute main effects,
        total effects, and higher-order Sobol indices.

        Parameters
        ----------
        values : Array
            Shape (nqoi, nsamples_total) - function values at all samples
            returned by generate_samples(). The order must match.

        Notes
        -----
        Due to finite sample effects:
        - Main effects may be negative
        - Main effects may exceed 1
        - Total effects may exceed 1
        - Sum of indices may not equal 1
        """
        valuesA, valuesB, valuesAB = self._unpack_values(values)
        self._mean = self._bkd.mean(valuesA, axis=1)
        self._variance = self._bkd.var(valuesA, axis=1)
        nterms = self._interaction_terms.shape[1]
        nqoi = valuesA.shape[0]
        interaction_values = self._bkd.zeros((nterms, nqoi))
        self._total_effects_ = self._bkd.zeros((self._nvars, nqoi))

        for ii in range(nterms):
            sobol_index = self._interaction_terms[:, ii]
            # Saltelli estimator for first-order/interaction indices
            interaction_values[ii, :] = (
                self._bkd.mean(valuesB * (valuesAB[ii] - valuesA), axis=1)
                / self._variance
            )
            if self._bkd.to_int(sobol_index.sum()) == 1:
                idx = self._bkd.to_int(self._bkd.where(sobol_index == 1)[0][0])
                # Jansen estimator (entry f in Table 2 of Saltelli, Annoni et al.)
                self._total_effects_[idx] = (
                    0.5
                    * self._bkd.mean((valuesA - valuesAB[ii]) ** 2, axis=1)
                    / self._variance
                )

        # Correct interaction values to get true Sobol indices
        self._sobol_indices_ = self._correct_interaction_variance_ratios(
            interaction_values
        )
        # Extract main effects (terms where only one variable is active)
        main_effect_mask = self._interaction_terms.sum(axis=0) == 1
        self._main_effects_ = self._sobol_indices_[main_effect_mask, :]

    def mean(self) -> Array:
        """Return the mean of the output estimated from sample set A.

        Returns
        -------
        Array
            Shape (nqoi,) - mean for each QoI.
        """
        if self._mean is None:
            raise RuntimeError("Must call compute() first")
        return self._mean

    def variance(self) -> Array:
        """Return the variance of the output estimated from sample set A.

        Returns
        -------
        Array
            Shape (nqoi,) - variance for each QoI.
        """
        if self._variance is None:
            raise RuntimeError("Must call compute() first")
        return self._variance

    def main_effects(self) -> Array:
        """Return first-order (main effect) Sobol indices.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - main effect index for each variable and QoI.
        """
        if self._main_effects_ is None:
            raise RuntimeError("Must call compute() first")
        return self._main_effects_

    def total_effects(self) -> Array:
        """Return total-order Sobol indices.

        Returns
        -------
        Array
            Shape (nvars, nqoi) - total effect index for each variable and QoI.
        """
        if self._total_effects_ is None:
            raise RuntimeError("Must call compute() first")
        return self._total_effects_

    def sobol_indices(self) -> Array:
        """Return all computed Sobol indices (main effects + interactions).

        Returns
        -------
        Array
            Shape (nterms, nqoi) - Sobol index for each interaction term.
        """
        if self._sobol_indices_ is None:
            raise RuntimeError("Must call compute() first")
        return self._sobol_indices_

    def nqoi(self) -> int:
        """Return the number of quantities of interest."""
        if self._mean is None:
            raise RuntimeError("Must call compute() first")
        return int(self._mean.shape[0])


class MonteCarloSensitivityAnalysis(
    SampleBasedSensitivityAnalysis[Array], Generic[Array]
):
    """Monte Carlo-based sensitivity analysis.

    Uses random sampling from the distribution to generate sample sets.

    Parameters
    ----------
    distribution : DistributionWithRVS[Array]
        Distribution with `rvs(nsamples)` method for random sampling.
    bkd : Backend[Array]
        Backend for array operations.

    Examples
    --------
    >>> from pyapprox.sensitivity.variance_based import (
    ...     MonteCarloSensitivityAnalysis,
    ... )
    >>> sa = MonteCarloSensitivityAnalysis(dist, bkd)
    >>> samples = sa.generate_samples(1000)
    >>> values = my_function(samples)
    >>> sa.compute(values)
    >>> main = sa.main_effects()
    """

    def _get_AB_samples(self, nsamples: int) -> Tuple[Array, Array]:
        """Generate two independent random sample sets."""
        # Distribution must have rvs method
        return self._distribution.rvs(nsamples), self._distribution.rvs(  # type: ignore[attr-defined]
            nsamples
        )


class SobolSequenceSensitivityAnalysis(
    SampleBasedSensitivityAnalysis[Array], Generic[Array]
):
    """Sobol sequence-based sensitivity analysis.

    Uses Sobol quasi-Monte Carlo sequences for improved convergence
    compared to random sampling.

    Parameters
    ----------
    distribution : DistributionWithInvCDF[Array]
        Distribution with `invcdf()` method for transforming uniform samples.
    bkd : Backend[Array]
        Backend for array operations.
    start_index : int, optional
        Starting index in the Sobol sequence. Default is 0.
    scramble : bool, optional
        Whether to use Owen scrambling. Default is True.
    seed : int, optional
        Random seed for scrambling. Default is None.

    Examples
    --------
    >>> from pyapprox.sensitivity.variance_based import (
    ...     SobolSequenceSensitivityAnalysis,
    ... )
    >>> sa = SobolSequenceSensitivityAnalysis(dist, bkd)
    >>> samples = sa.generate_samples(1024)  # Powers of 2 recommended
    >>> values = my_function(samples)
    >>> sa.compute(values)
    >>> main = sa.main_effects()
    """

    def __init__(
        self,
        distribution: DistributionWithInvCDF[Array],
        bkd: Backend[Array],
        start_index: int = 0,
        scramble: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(distribution, bkd)
        # Create Sobol sampler with 2x the dimensions for A and B
        self._sampler = SobolSampler(
            2 * self._nvars,
            bkd,
            distribution=None,  # Will transform manually
            start_index=start_index,
            scramble=scramble,
            seed=seed,
        )

    def _get_AB_samples(self, nsamples: int) -> Tuple[Array, Array]:
        """Generate A and B from a single Sobol sequence."""
        # Generate 2*nvars dimensional Sobol points
        uniform_samples, _ = self._sampler.sample(nsamples)
        # Split into A and B, transform each through distribution
        uniform_A = uniform_samples[: self._nvars, :]
        uniform_B = uniform_samples[self._nvars :, :]
        samplesA = self._distribution.invcdf(uniform_A)
        samplesB = self._distribution.invcdf(uniform_B)
        return samplesA, samplesB


class HaltonSequenceSensitivityAnalysis(
    SampleBasedSensitivityAnalysis[Array], Generic[Array]
):
    """Halton sequence-based sensitivity analysis.

    Uses Halton quasi-Monte Carlo sequences for improved convergence
    compared to random sampling.

    Parameters
    ----------
    distribution : DistributionWithInvCDF[Array]
        Distribution with `invcdf()` method for transforming uniform samples.
    bkd : Backend[Array]
        Backend for array operations.
    start_index : int, optional
        Starting index in the Halton sequence. Default is 0.
    scramble : bool, optional
        Whether to use Owen scrambling. Default is True.
    seed : int, optional
        Random seed for scrambling. Default is None.

    Examples
    --------
    >>> from pyapprox.sensitivity.variance_based import (
    ...     HaltonSequenceSensitivityAnalysis,
    ... )
    >>> sa = HaltonSequenceSensitivityAnalysis(dist, bkd)
    >>> samples = sa.generate_samples(1000)
    >>> values = my_function(samples)
    >>> sa.compute(values)
    >>> main = sa.main_effects()
    """

    def __init__(
        self,
        distribution: DistributionWithInvCDF[Array],
        bkd: Backend[Array],
        start_index: int = 0,
        scramble: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(distribution, bkd)
        # Create Halton sampler with 2x the dimensions for A and B
        self._sampler = HaltonSampler(
            2 * self._nvars,
            bkd,
            distribution=None,  # Will transform manually
            start_index=start_index,
            scramble=scramble,
            seed=seed,
        )

    def _get_AB_samples(self, nsamples: int) -> Tuple[Array, Array]:
        """Generate A and B from a single Halton sequence."""
        # Generate 2*nvars dimensional Halton points
        uniform_samples, _ = self._sampler.sample(nsamples)
        # Split into A and B, transform each through distribution
        uniform_A = uniform_samples[: self._nvars, :]
        uniform_B = uniform_samples[self._nvars :, :]
        samplesA = self._distribution.invcdf(uniform_A)
        samplesB = self._distribution.invcdf(uniform_B)
        return samplesA, samplesB
