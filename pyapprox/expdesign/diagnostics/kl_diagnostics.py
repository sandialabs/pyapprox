"""
KL-OED diagnostics for MSE and convergence analysis.

Provides tools for computing bias, variance, and MSE of numerical
EIG estimates compared to exact analytical values.
"""

#TODO: classes should use bkd.fun not np.fun, e.g. compute mse
#TODO: float(self._bkd.to_numpy( should be bkd.to_float

from typing import Dict, Generic, List, Tuple

import numpy as np

from pyapprox.expdesign.benchmarks import LinearGaussianOEDBenchmark
from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.expdesign.objective import KLOEDObjective
from pyapprox.util.backends.protocols import Array, Backend


class KLOEDDiagnostics(Generic[Array]):
    """
    Diagnostics for KL-OED estimator performance.

    Computes bias, variance, and MSE of numerical EIG estimates by
    comparing to exact analytical values from LinearGaussianOEDBenchmark.

    Parameters
    ----------
    benchmark : LinearGaussianOEDBenchmark[Array]
        Benchmark problem with exact EIG computation.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.expdesign.benchmarks import LinearGaussianOEDBenchmark
    >>> from pyapprox.expdesign.diagnostics import KLOEDDiagnostics
    >>>
    >>> bkd = NumpyBkd()
    >>> benchmark = LinearGaussianOEDBenchmark(5, 2, 0.5, 0.5, bkd)
    >>> diagnostics = KLOEDDiagnostics(benchmark)
    >>>
    >>> weights = bkd.ones((5, 1)) / 5
    >>> bias, var, mse = diagnostics.compute_mse(
    ...     nouter=100, ninner=50, nrealizations=10, design_weights=weights
    ... )
    """

    def __init__(self, benchmark: LinearGaussianOEDBenchmark[Array]) -> None:
        self._benchmark = benchmark
        self._bkd = benchmark.bkd()

        # TODO: add validation check that benchmark meets protocol, e.g.
        # has exact_eig

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def exact_eig(self, design_weights: Array) -> float:
        """
        Compute exact expected information gain.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        eig : float
            Exact expected information gain.
        """
        return self._benchmark.exact_eig(design_weights)

    def compute_numerical_eig(
        self,
        nouter: int,
        ninner: int,
        design_weights: Array,
        seed: int = 42,
    ) -> float:
        """
        Compute numerical EIG estimate using KLOEDObjective.

        Parameters
        ----------
        nouter : int
            Number of outer loop samples.
        ninner : int
            Number of inner loop samples.
        design_weights : Array
            Design weights. Shape: (nobs, 1)
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        eig : float
            Numerical EIG estimate.
        """
        np.random.seed(seed)

        # Generate samples from the benchmark
        _, outer_shapes = self._benchmark.generate_data(nouter, seed=seed)
        _, inner_shapes = self._benchmark.generate_data(ninner, seed=seed + 1000)

        # Generate latent samples for reparameterization (standard normal)
        latent_samples = self._benchmark.generate_latent_samples(nouter, seed=seed)

        # Create noise variances
        noise_variances = self._bkd.full(
            (self._benchmark.nobs(),), self._benchmark.noise_var()
        )

        # Create objective
        inner_likelihood = GaussianOEDInnerLoopLikelihood(noise_variances, self._bkd)
        objective = KLOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            None,  # Uniform outer weights
            None,  # Uniform inner weights
            self._bkd,
        )

        # Compute EIG (objective returns negative EIG for minimization)
        eig = objective.expected_information_gain(design_weights)
        return eig

    def compute_mse(
        self,
        nouter: int,
        ninner: int,
        nrealizations: int,
        design_weights: Array,
        base_seed: int = 42,
    ) -> Tuple[float, float, float]:
        """
        Compute MSE of numerical EIG estimate.

        Parameters
        ----------
        nouter : int
            Number of outer loop samples.
        ninner : int
            Number of inner loop samples.
        nrealizations : int
            Number of independent realizations for variance estimation.
        design_weights : Array
            Design weights. Shape: (nobs, 1)
        base_seed : int
            Base random seed (each realization uses base_seed + i).

        Returns
        -------
        bias : float
            Bias = E[estimate] - exact
        variance : float
            Variance of the estimator.
        mse : float
            Mean squared error = bias^2 + variance
        """
        exact = self.exact_eig(design_weights)

        eig_values = []
        for i in range(nrealizations):
            seed = base_seed + i * 10000
            eig = self.compute_numerical_eig(nouter, ninner, design_weights, seed=seed)
            eig_values.append(eig)

        eig_array = np.array(eig_values)
        mean_eig = float(np.mean(eig_array))
        var_eig = float(np.var(eig_array))

        bias = mean_eig - exact
        mse = bias**2 + var_eig

        return bias, var_eig, mse

    def compute_mse_for_sample_combinations(
        self,
        outer_sample_counts: List[int],
        inner_sample_counts: List[int],
        nrealizations: int,
        design_weights: Array,
        base_seed: int = 42,
    ) -> Dict[str, List[Array]]:
        """
        Compute MSE for different combinations of sample counts.

        Parameters
        ----------
        outer_sample_counts : List[int]
            List of outer loop sample counts.
        inner_sample_counts : List[int]
            List of inner loop sample counts.
        nrealizations : int
            Number of realizations per combination.
        design_weights : Array
            Design weights. Shape: (nobs, 1)
        base_seed : int
            Base random seed.

        Returns
        -------
        values : Dict[str, List[Array]]
            Dictionary with keys:
            - "sqbias": List of squared-bias arrays (one per ninner)
            - "variance": List of variance arrays (one per ninner)
            - "mse": List of MSE arrays (one per ninner)
        """
        values: Dict[str, List[Array]] = {"sqbias": [], "variance": [], "mse": []}

        for ninner in inner_sample_counts:
            bias_list = []
            var_list = []
            mse_list = []

            for nouter in outer_sample_counts:
                bias, variance, mse = self.compute_mse(
                    nouter, ninner, nrealizations, design_weights, base_seed
                )
                bias_list.append(bias)
                var_list.append(variance)
                mse_list.append(mse)

            values["sqbias"].append(self._bkd.asarray(bias_list) ** 2)
            values["variance"].append(self._bkd.asarray(var_list))
            values["mse"].append(self._bkd.asarray(mse_list))

        return values

    @staticmethod
    def compute_convergence_rate(
        sample_counts: List[int],
        values: List[float],
    ) -> float:
        """
        Compute convergence rate from log-log linear fit.

        For Monte Carlo estimators, MSE typically decays as O(n^{-r})
        where r is the convergence rate. This is estimated by fitting
        a line to log(values) vs log(sample_counts).

        Parameters
        ----------
        sample_counts : List[int]
            Sample counts (x-axis).
        values : List[float]
            Values (e.g., MSE) corresponding to sample counts.

        Returns
        -------
        rate : float
            Convergence rate (negative of log-log slope).
            Rate of 1.0 indicates O(1/n) convergence.
        """
        log_n = np.log(np.array(sample_counts))
        log_vals = np.log(np.array(values))

        # Linear fit: log(val) = slope * log(n) + intercept
        slope, _ = np.polyfit(log_n, log_vals, 1)

        # Rate is negative slope (since MSE decreases with n)
        return -slope
