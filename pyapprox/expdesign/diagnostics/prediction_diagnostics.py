"""Prediction OED diagnostics — pure functions of raw sample arrays.

Computes numerical prediction OED utility estimates from pre-generated
samples. Sampling is the caller's responsibility.

Also provides exact utility computation via conjugate Gaussian formulas
and a registry of utility types.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
)

from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDExpectedAVaRDev,
    ConjugateGaussianOEDExpectedStdDev,
    ConjugateGaussianOEDForLogNormalAVaRStdDev,
    ConjugateGaussianOEDForLogNormalExpectedStdDev,
    ConjugateGaussianOEDPredictionUtilityBase,
)
from pyapprox.expdesign.deviation import StandardDeviationMeasure
from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.expdesign.objective import PredictionOEDObjective
from pyapprox.expdesign.statistics import SampleAverageMean
from pyapprox.expdesign.statistics.avar import SampleAverageSmoothedAVaR
from pyapprox.expdesign.statistics.base import SampleStatistic
from pyapprox.util.backends.protocols import Array, Backend

# --- Utility type registry ---

# Type alias for utility factory return type
UtilityFactoryResult = Tuple[
    Type[ConjugateGaussianOEDPredictionUtilityBase[Any]],
    Tuple[Any, ...],
    Callable[[Backend[Any]], SampleStatistic[Any]],
]

_UTILITY_REGISTRY: Dict[str, Callable[..., UtilityFactoryResult]] = {}


def register_utility(
    name: str,
) -> Callable[
    [Callable[..., UtilityFactoryResult]],
    Callable[..., UtilityFactoryResult],
]:
    """Decorator to register a utility factory.

    The factory function should accept keyword arguments and return a tuple:
    (exact_utility_class, exact_utility_args, noise_stat_factory)

    where noise_stat_factory is a callable: Backend -> SampleStatistic.
    """

    def decorator(
        func: Callable[..., UtilityFactoryResult],
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


# Register built-in utility types for lognormal (nonlinear QoI)
@register_utility("nonlinear_mean_stdev")
def _create_nonlinear_mean_stdev_utility(**kwargs: Any) -> UtilityFactoryResult:
    """Create lognormal expected std dev utility with mean noise statistic."""
    return (
        ConjugateGaussianOEDForLogNormalExpectedStdDev,
        (),
        _mean_noise_stat_factory,
    )


@register_utility("nonlinear_avar_stdev")
def _create_nonlinear_avar_stdev_utility(**kwargs: Any) -> UtilityFactoryResult:
    """Create lognormal AVaR std dev utility with AVaR noise statistic."""
    beta = kwargs.get("beta", 0.5)
    delta = kwargs.get("delta", 100000)
    return (
        ConjugateGaussianOEDForLogNormalAVaRStdDev,
        (beta,),
        _avar_noise_stat_factory(beta, delta),
    )


# Register utility types for linear QoI
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


# --- Diagnostics class ---


class PredictionOEDDiagnostics(Generic[Array]):
    """Numerical prediction OED utility estimation from raw sample arrays.

    This class does NOT generate samples. The caller provides
    outer_shapes, latent_samples, inner_shapes, and qoi_vals directly.

    Parameters
    ----------
    noise_variances : Array
        Per-observation noise variances. Shape: (nobs,).
    npred : int
        Number of prediction QoI locations.
    noise_stat_factory : Callable[[Backend], SampleStatistic]
        Factory for the noise statistic.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        noise_variances: Array,
        npred: int,
        noise_stat_factory: Callable[[Backend[Array]], SampleStatistic[Array]],
        bkd: Backend[Array],
    ) -> None:
        self._noise_variances = noise_variances
        self._npred = npred
        self._noise_stat_factory = noise_stat_factory
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def compute_numerical_utility(
        self,
        design_weights: Array,
        outer_shapes: Array,
        latent_samples: Array,
        inner_shapes: Array,
        qoi_vals: Array,
        outer_quad_weights: Optional[Array] = None,
        inner_quad_weights: Optional[Array] = None,
        qoi_quad_weights: Optional[Array] = None,
    ) -> float:
        """Compute numerical prediction OED utility estimate from raw samples.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (nobs, 1).
        outer_shapes : Array
            obs_map(prior_samples_outer). Shape: (nobs, nouter).
        latent_samples : Array
            Standard normal latent samples. Shape: (nobs, nouter).
        inner_shapes : Array
            obs_map(prior_samples_inner). Shape: (nobs, ninner).
        qoi_vals : Array
            qoi_map(prior_samples_inner). Shape: (ninner, npred).
        outer_quad_weights : Array, optional
            Quadrature weights for outer expectation. Shape: (nouter,).
        inner_quad_weights : Array, optional
            Quadrature weights for inner expectation. Shape: (ninner,).
        qoi_quad_weights : Array, optional
            Quadrature weights for QoI aggregation. Shape: (1, npred).

        Returns
        -------
        utility : float
            Numerical utility estimate.
        """
        inner_likelihood = GaussianOEDInnerLoopLikelihood(
            self._noise_variances, self._bkd
        )
        deviation_measure = StandardDeviationMeasure(self._npred, self._bkd)
        risk_measure = SampleAverageMean(self._bkd)
        noise_stat = self._noise_stat_factory(self._bkd)

        objective = PredictionOEDObjective(
            inner_likelihood,
            outer_shapes,
            latent_samples,
            inner_shapes,
            qoi_vals,
            deviation_measure,
            risk_measure,
            noise_stat,
            outer_quad_weights,
            inner_quad_weights,
            qoi_quad_weights,
            self._bkd,
        )

        value = objective(design_weights)
        return float(self._bkd.to_numpy(value)[0, 0])


def compute_exact_prediction_utility(
    prior_mean: Array,
    prior_covariance: Array,
    design_matrix: Array,
    qoi_matrix: Array,
    noise_var: float,
    design_weights: Array,
    exact_utility_cls: Type[ConjugateGaussianOEDPredictionUtilityBase[Array]],
    exact_utility_args: Tuple[Any, ...],
    bkd: Backend[Array],
) -> float:
    """Compute exact prediction utility using conjugate Gaussian formulas.

    Parameters
    ----------
    prior_mean : Array
        Prior mean. Shape: (nparams, 1).
    prior_covariance : Array
        Prior covariance. Shape: (nparams, nparams).
    design_matrix : Array
        Observation design matrix A. Shape: (nobs, nparams).
    qoi_matrix : Array
        QoI design matrix B. Shape: (npred, nparams).
    noise_var : float
        Base noise variance.
    design_weights : Array
        Design weights. Shape: (nobs, 1).
    exact_utility_cls : Type
        Class for computing exact utility.
    exact_utility_args : tuple
        Additional arguments for the exact utility class.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    utility : float
        Exact expected utility value.
    """
    nobs = design_matrix.shape[0]

    if exact_utility_args:
        utility = exact_utility_cls(  # type: ignore[call-arg]
            prior_mean, prior_covariance, qoi_matrix,
            *exact_utility_args, bkd,
        )
    else:
        utility = exact_utility_cls(
            prior_mean, prior_covariance, qoi_matrix, bkd,
        )

    utility.set_observation_matrix(design_matrix)

    weights_flat = bkd.reshape(design_weights, (nobs,))
    effective_noise_var = noise_var / weights_flat
    noise_cov = bkd.diag(effective_noise_var)
    utility.set_noise_covariance(noise_cov)

    return utility.value()


def create_prediction_oed_diagnostics(
    noise_variances: Array,
    npred: int,
    utility_type: str,
    bkd: Backend[Array],
    **kwargs: Any,
) -> PredictionOEDDiagnostics[Array]:
    """Factory function to create PredictionOEDDiagnostics.

    Uses the registry to look up the appropriate noise statistic factory.

    Parameters
    ----------
    noise_variances : Array
        Per-observation noise variances. Shape: (nobs,).
    npred : int
        Number of prediction QoI locations.
    utility_type : str
        Type of utility. Registered types: "stdev", "avar_stdev",
        "linear_stdev", "linear_avar".
    bkd : Backend[Array]
        Computational backend.
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
    """
    if utility_type not in _UTILITY_REGISTRY:
        available = get_registered_utility_types()
        raise ValueError(
            f"Unknown utility_type: {utility_type}. Available types: {available}"
        )

    factory = _UTILITY_REGISTRY[utility_type]
    _, _, noise_stat_factory = factory(**kwargs)

    return PredictionOEDDiagnostics(
        noise_variances, npred, noise_stat_factory, bkd,
    )


def get_utility_factory(
    utility_type: str,
    **kwargs: Any,
) -> UtilityFactoryResult:
    """Get utility factory result (exact class, args, noise stat factory).

    Parameters
    ----------
    utility_type : str
        Registered utility type name.
    **kwargs
        Additional arguments for the utility type.

    Returns
    -------
    exact_cls : Type
        Exact utility class.
    exact_args : tuple
        Additional arguments for exact utility class.
    noise_stat_factory : Callable
        Factory for noise statistic.
    """
    if utility_type not in _UTILITY_REGISTRY:
        available = get_registered_utility_types()
        raise ValueError(
            f"Unknown utility_type: {utility_type}. Available types: {available}"
        )
    factory = _UTILITY_REGISTRY[utility_type]
    return factory(**kwargs)
