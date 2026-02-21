"""
Prediction OED diagnostics for MSE and convergence analysis.

Provides tools for computing bias, variance, and MSE of numerical
prediction OED estimates compared to exact analytical values.
"""

from typing import (
    Generic, List, Dict, Tuple, Type, Callable, Any, Optional,
    Protocol, runtime_checkable, Union,
)

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend

from pyapprox.typing.expdesign.benchmarks import (
    NonLinearGaussianOEDBenchmark,
    LinearGaussianPredOEDBenchmark,
)
from pyapprox.typing.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.typing.expdesign.objective import PredictionOEDObjective
from pyapprox.typing.expdesign.deviation import StandardDeviationMeasure
from pyapprox.typing.expdesign.statistics import SampleAverageMean
from pyapprox.typing.expdesign.statistics.avar import SampleAverageSmoothedAVaR
from pyapprox.typing.expdesign.statistics.base import SampleStatistic
from pyapprox.typing.expdesign.analytical import (
    ConjugateGaussianOEDPredictionUtilityBase,
    ConjugateGaussianOEDExpectedStdDev,
    ConjugateGaussianOEDExpectedAVaRDev,
    ConjugateGaussianOEDForLogNormalExpectedStdDev,
    ConjugateGaussianOEDForLogNormalAVaRStdDev,
)


@runtime_checkable
class PredictionOEDBenchmarkProtocol(Protocol[Array]):
    """Protocol for prediction OED benchmarks.

    Both NonLinearGaussianOEDBenchmark and LinearGaussianPredOEDBenchmark
    satisfy this protocol.
    """

    def bkd(self) -> Backend[Array]: ...
    def nobs(self) -> int: ...
    def nparams(self) -> int: ...
    def npred(self) -> int: ...
    def noise_var(self) -> float: ...
    def noise_std(self) -> float: ...
    def prior_var(self) -> float: ...
    def prior_std(self) -> float: ...
    def design_matrix(self) -> Array: ...
    def qoi_matrix(self) -> Array: ...
    def noise_variances(self) -> Array: ...
    def prior_mean(self) -> Array: ...
    def prior_covariance(self) -> Array: ...
    def generate_observation_data(
        self, nsamples: int, seed: int = 42,
    ) -> tuple[Array, Array]: ...
    def generate_latent_samples(
        self, nsamples: int, seed: int = 42,
    ) -> Array: ...


# Type for benchmarks accepted by diagnostics
PredBenchmarkType = Union[
    NonLinearGaussianOEDBenchmark[Any],
    LinearGaussianPredOEDBenchmark[Any],
]

# Type alias for utility factory return type
UtilityFactoryResult = Tuple[
    Type[ConjugateGaussianOEDPredictionUtilityBase[Any]],
    Tuple[Any, ...],
    Callable[[Backend[Any]], SampleStatistic[Any]],
]

# Registry for utility factories
# Maps utility_type -> factory function that returns (exact_cls, exact_args, noise_stat_factory)
_UTILITY_REGISTRY: Dict[str, Callable[..., UtilityFactoryResult]] = {}


def register_utility(
    name: str,
) -> Callable[
    [Callable[..., UtilityFactoryResult]],
    Callable[..., UtilityFactoryResult],
]:
    """
    Decorator to register a utility factory.

    The factory function should accept keyword arguments and return a tuple:
    (exact_utility_class, exact_utility_args, noise_stat_factory)

    where noise_stat_factory is a callable: Backend -> SampleStatistic

    Parameters
    ----------
    name : str
        Name to register the utility under.

    Example
    -------
    >>> @register_utility("my_utility")
    ... def _create_my_utility(**kwargs):
    ...     param = kwargs.get("param", 1.0)
    ...     def noise_stat_factory(bkd):
    ...         return SampleAverageMean(bkd)
    ...     return MyUtilityClass, (param,), noise_stat_factory
    """
    def decorator(
        func: Callable[..., UtilityFactoryResult]
    ) -> Callable[..., UtilityFactoryResult]:
        _UTILITY_REGISTRY[name] = func
        return func
    return decorator


def get_registered_utility_types() -> List[str]:
    """Get list of registered utility type names."""
    return list(_UTILITY_REGISTRY.keys())


def _mean_noise_stat_factory(bkd: Backend[Any]) -> SampleStatistic[Any]:
    """Factory for SampleAverageMean noise statistic."""
    return SampleAverageMean(bkd)


def _avar_noise_stat_factory(
    beta: float, delta: float = 100000
) -> Callable[[Backend[Any]], SampleStatistic[Any]]:
    """Factory for SampleAverageSmoothedAVaR noise statistic."""
    def factory(bkd: Backend[Any]) -> SampleStatistic[Any]:
        return SampleAverageSmoothedAVaR(beta, bkd, delta=delta)
    return factory


# Register built-in utility types for lognormal (NonLinearGaussianOEDBenchmark)
@register_utility("stdev")
def _create_stdev_utility(**kwargs: Any) -> UtilityFactoryResult:
    """Create lognormal expected std dev utility with mean noise statistic."""
    return (
        ConjugateGaussianOEDForLogNormalExpectedStdDev,
        (),
        _mean_noise_stat_factory,
    )


@register_utility("avar_stdev")
def _create_avar_stdev_utility(**kwargs: Any) -> UtilityFactoryResult:
    """Create lognormal AVaR std dev utility with AVaR noise statistic."""
    beta = kwargs.get("beta", 0.5)
    delta = kwargs.get("delta", 100000)
    return (
        ConjugateGaussianOEDForLogNormalAVaRStdDev,
        (beta,),
        _avar_noise_stat_factory(beta, delta),
    )


# Register utility types for linear QoI (LinearGaussianPredOEDBenchmark)
@register_utility("linear_stdev")
def _create_linear_stdev_utility(**kwargs: Any) -> UtilityFactoryResult:
    """Create linear Gaussian expected std dev utility."""
    return (
        ConjugateGaussianOEDExpectedStdDev,
        (),
        _mean_noise_stat_factory,
    )


@register_utility("linear_avar")
def _create_linear_avar_utility(**kwargs: Any) -> UtilityFactoryResult:
    """Create linear Gaussian AVaR deviation utility."""
    beta = kwargs.get("beta", 0.5)
    delta = kwargs.get("delta", 100000)
    return (
        ConjugateGaussianOEDExpectedAVaRDev,
        (beta,),
        _avar_noise_stat_factory(beta, delta),
    )


class PredictionOEDDiagnostics(Generic[Array]):
    """
    Diagnostics for prediction OED estimator performance.

    Computes bias, variance, and MSE of numerical prediction OED estimates
    by comparing to exact analytical values from conjugate Gaussian formulas.

    Supports both nonlinear (lognormal) benchmarks and linear benchmarks.

    Parameters
    ----------
    benchmark : PredBenchmarkType
        Benchmark problem (NonLinearGaussianOEDBenchmark or
        LinearGaussianPredOEDBenchmark).
    exact_utility_cls : Type[ConjugateGaussianOEDPredictionUtilityBase]
        Class for computing exact utility.
    exact_utility_args : tuple
        Additional arguments for the exact utility class
        (e.g., beta for AVaR).
    noise_stat_factory : Callable[[Backend], SampleStatistic]
        Factory for the noise statistic.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.expdesign.benchmarks import NonLinearGaussianOEDBenchmark
    >>> from pyapprox.typing.expdesign.diagnostics import PredictionOEDDiagnostics
    >>> from pyapprox.typing.expdesign.analytical import (
    ...     ConjugateGaussianOEDForLogNormalExpectedStdDev
    ... )
    >>>
    >>> bkd = NumpyBkd()
    >>> benchmark = NonLinearGaussianOEDBenchmark(
    ...     nobs=2, degree=3, noise_std=0.5, prior_std=0.5, bkd=bkd
    ... )
    >>> diagnostics = PredictionOEDDiagnostics(
    ...     benchmark, ConjugateGaussianOEDForLogNormalExpectedStdDev, ()
    ... )
    >>>
    >>> weights = bkd.ones((2, 1)) / 2
    >>> bias, var, mse = diagnostics.compute_mse(
    ...     nouter=100, ninner=50, nrealizations=10, design_weights=weights
    ... )
    """

    def __init__(
        self,
        benchmark: PredBenchmarkType,
        exact_utility_cls: Type[ConjugateGaussianOEDPredictionUtilityBase[Array]],
        exact_utility_args: Tuple[Any, ...],
        noise_stat_factory: Callable[[Backend[Array]], SampleStatistic[Array]],
    ) -> None:
        self._benchmark = benchmark
        self._bkd = benchmark.bkd()
        self._exact_utility_cls = exact_utility_cls
        self._exact_utility_args = exact_utility_args
        self._noise_stat_factory = noise_stat_factory

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def exact_utility(self, design_weights: Array) -> float:
        """
        Compute exact utility using conjugate Gaussian formulas.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1)

        Returns
        -------
        utility : float
            Exact expected utility value.
        """
        # Create exact utility object
        # Note: Subclasses may have additional arguments beyond the base class signature
        if self._exact_utility_args:
            utility = self._exact_utility_cls(  # type: ignore[call-arg]
                self._benchmark.prior_mean(),
                self._benchmark.prior_covariance(),
                self._benchmark.qoi_matrix(),
                *self._exact_utility_args,
                self._bkd,
            )
        else:
            utility = self._exact_utility_cls(
                self._benchmark.prior_mean(),
                self._benchmark.prior_covariance(),
                self._benchmark.qoi_matrix(),
                self._bkd,
            )

        # Set observation matrix
        utility.set_observation_matrix(self._benchmark.design_matrix())

        # Compute weighted noise covariance
        nobs = self._benchmark.nobs()
        noise_var = self._benchmark.noise_var()
        weights_flat = self._bkd.reshape(design_weights, (nobs,))
        # Effective noise variance = base_noise_var / weight
        effective_noise_var = noise_var / weights_flat
        noise_cov = self._bkd.diag(effective_noise_var)

        utility.set_noise_covariance(noise_cov)

        return utility.value()

    def _generate_qoi_vals(
        self, ninner: int, seed: int,
    ) -> Array:
        """Generate QoI values for the inner loop.

        Returns qoi_vals with shape (ninner, npred).

        For NonLinearGaussianOEDBenchmark: uses exp(B @ theta).
        For LinearGaussianPredOEDBenchmark: uses B @ theta.
        """
        if isinstance(self._benchmark, NonLinearGaussianOEDBenchmark):
            _, _, exp_qoi = self._benchmark.generate_qoi_data(
                ninner, seed=seed
            )
            # exp_qoi shape: (npred, ninner), need (ninner, npred)
            return exp_qoi.T
        elif isinstance(self._benchmark, LinearGaussianPredOEDBenchmark):
            _, linear_qoi = self._benchmark.generate_qoi_data(
                ninner, seed=seed
            )
            # linear_qoi shape: (npred, ninner), need (ninner, npred)
            return linear_qoi.T
        else:
            raise TypeError(
                f"Unsupported benchmark type: {type(self._benchmark)}"
            )

    def compute_numerical_utility(
        self,
        nouter: int,
        ninner: int,
        design_weights: Array,
        seed: int = 42,
    ) -> float:
        """
        Compute numerical prediction OED utility estimate.

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
        utility : float
            Numerical utility estimate.
        """
        # Generate outer loop data (observations)
        _, outer_shapes = self._benchmark.generate_observation_data(
            nouter, seed=seed
        )

        # Generate QoI values
        qoi_vals = self._generate_qoi_vals(ninner, seed=seed + 1000)

        # Generate inner shapes (observation model at inner prior samples)
        _, inner_shapes = self._benchmark.generate_observation_data(
            ninner, seed=seed + 1000
        )

        # Generate latent samples for reparameterization trick
        latent_samples = self._benchmark.generate_latent_samples(
            nouter, seed=seed
        )

        # Create noise variances
        noise_variances = self._benchmark.noise_variances()

        # Create likelihood
        inner_likelihood = GaussianOEDInnerLoopLikelihood(
            noise_variances, self._bkd
        )

        # Create deviation and risk measures
        deviation_measure = StandardDeviationMeasure(
            self._benchmark.npred(), self._bkd
        )
        risk_measure = SampleAverageMean(self._bkd)
        noise_stat = self._noise_stat_factory(self._bkd)

        # Create objective
        objective = PredictionOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            qoi_vals,
            deviation_measure,
            risk_measure,
            noise_stat,
            None,  # Uniform outer weights
            None,  # Uniform inner weights
            None,  # Uniform QoI weights
            self._bkd,
        )

        # Compute utility
        value = objective(design_weights)
        return float(self._bkd.to_numpy(value)[0, 0])

    def compute_mse(
        self,
        nouter: int,
        ninner: int,
        nrealizations: int,
        design_weights: Array,
        base_seed: int = 42,
    ) -> Tuple[float, float, float]:
        """
        Compute MSE of numerical prediction OED estimate.

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
            Base random seed (each realization uses base_seed + i * 10000).

        Returns
        -------
        bias : float
            Bias = E[estimate] - exact
        variance : float
            Variance of the estimator.
        mse : float
            Mean squared error = bias^2 + variance
        """
        exact = self.exact_utility(design_weights)

        utility_values = []
        for i in range(nrealizations):
            seed = base_seed + i * 10000
            utility = self.compute_numerical_utility(
                nouter, ninner, design_weights, seed=seed
            )
            utility_values.append(utility)

        utility_array = np.array(utility_values)
        mean_utility = float(np.mean(utility_array))
        var_utility = float(np.var(utility_array))

        bias = mean_utility - exact
        mse = bias ** 2 + var_utility

        return bias, var_utility, mse

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
        values: Dict[str, List[Array]] = {
            "sqbias": [], "variance": [], "mse": [],
        }

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
        where r is the convergence rate.

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
        return float(-slope)


def create_prediction_oed_diagnostics(
    benchmark: PredBenchmarkType,
    utility_type: str,
    **kwargs: Any,
) -> PredictionOEDDiagnostics[Any]:
    """
    Factory function to create PredictionOEDDiagnostics.

    Uses the registry to look up the appropriate exact utility class and
    noise statistic factory.

    Parameters
    ----------
    benchmark : PredBenchmarkType
        Benchmark problem (NonLinearGaussianOEDBenchmark or
        LinearGaussianPredOEDBenchmark).
    utility_type : str
        Type of utility. Registered types: "stdev", "avar_stdev",
        "linear_stdev", "linear_avar".
    **kwargs
        Additional arguments for the utility type (e.g., beta for avar).

    Returns
    -------
    PredictionOEDDiagnostics
        Configured diagnostics object.

    Raises
    ------
    ValueError
        If utility_type is not registered.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.expdesign.benchmarks import (
    ...     NonLinearGaussianOEDBenchmark, LinearGaussianPredOEDBenchmark,
    ... )
    >>> from pyapprox.typing.expdesign.diagnostics import (
    ...     create_prediction_oed_diagnostics
    ... )
    >>>
    >>> bkd = NumpyBkd()
    >>>
    >>> # Lognormal benchmark
    >>> bench_nl = NonLinearGaussianOEDBenchmark(
    ...     nobs=2, degree=3, noise_std=0.5, prior_std=0.5, bkd=bkd
    ... )
    >>> diag = create_prediction_oed_diagnostics(bench_nl, "stdev")
    >>>
    >>> # Linear benchmark
    >>> bench_l = LinearGaussianPredOEDBenchmark(
    ...     nobs=2, degree=3, noise_std=0.5, prior_std=0.5, npred=1, bkd=bkd
    ... )
    >>> diag = create_prediction_oed_diagnostics(bench_l, "linear_stdev")
    """
    if utility_type not in _UTILITY_REGISTRY:
        available = get_registered_utility_types()
        raise ValueError(
            f"Unknown utility_type: {utility_type}. "
            f"Available types: {available}"
        )

    factory = _UTILITY_REGISTRY[utility_type]
    exact_cls, exact_args, noise_stat_factory = factory(**kwargs)

    return PredictionOEDDiagnostics(
        benchmark, exact_cls, exact_args, noise_stat_factory
    )
