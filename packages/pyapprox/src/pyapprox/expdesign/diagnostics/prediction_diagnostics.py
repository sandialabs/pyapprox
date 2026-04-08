"""Prediction OED diagnostics — pure functions of raw sample arrays.

Computes numerical prediction OED utility estimates from pre-generated
samples. Sampling is the caller's responsibility.

Also provides exact utility computation via conjugate Gaussian formulas
and a registry of utility types.
"""

from dataclasses import dataclass, field
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
    ConjugateGaussianOEDDataMeanQoIAVaRStdDev,
    ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev,
    ConjugateGaussianOEDDataMeanQoIMeanEntropicDev,
    ConjugateGaussianOEDDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev,
    ConjugateGaussianOEDPredictionUtilityBase,
)
from pyapprox.expdesign.deviation import (
    AVaRDeviationMeasure,
    DeviationMeasure,
    EntropicDeviationMeasure,
    StandardDeviationMeasure,
)
from pyapprox.expdesign.likelihood import GaussianOEDInnerLoopLikelihood
from pyapprox.expdesign.objective import PredictionOEDObjective
from pyapprox.risk import SampleAverageMean
from pyapprox.risk.avar import SampleAverageSmoothedAVaR
from pyapprox.risk.base import SampleStatistic
from pyapprox.util.backends.protocols import Array, Backend

# --- UtilityConfig dataclass ---


@dataclass
class UtilityConfig:
    """Configuration for a prediction OED utility type.

    Bundles together the factories for deviation measure, risk measure,
    and noise statistic, along with the exact analytical class and its
    extra constructor arguments.

    Attributes
    ----------
    deviation_factory : Callable[[int, Backend], DeviationMeasure]
        Factory that takes ``(npred, bkd)`` and returns a deviation measure.
    risk_factory : Callable[[Backend], SampleStatistic]
        Factory that takes ``(bkd,)`` and returns a risk measure.
    noise_stat_factory : Callable[[Backend], SampleStatistic]
        Factory that takes ``(bkd,)`` and returns a noise statistic.
    exact_cls : Type[ConjugateGaussianOEDPredictionUtilityBase]
        Analytical class for computing exact utility.
    exact_args : tuple
        Extra positional arguments for ``exact_cls`` (inserted between
        ``qoi_mat`` and ``bkd``).
    """

    deviation_factory: Callable[[int, Backend[Any]], DeviationMeasure[Any]]
    risk_factory: Callable[[Backend[Any]], SampleStatistic[Any]]
    noise_stat_factory: Callable[[Backend[Any]], SampleStatistic[Any]]
    exact_cls: Type[ConjugateGaussianOEDPredictionUtilityBase[Any]]
    exact_args: Tuple[Any, ...] = field(default_factory=tuple)


# --- Utility type registry ---

_UTILITY_REGISTRY: Dict[str, Callable[..., UtilityConfig]] = {}


def register_utility(
    name: str,
) -> Callable[
    [Callable[..., UtilityConfig]],
    Callable[..., UtilityConfig],
]:
    """Decorator to register a utility factory.

    The factory function should accept keyword arguments and return a
    ``UtilityConfig`` instance.
    """

    def decorator(
        func: Callable[..., UtilityConfig],
    ) -> Callable[..., UtilityConfig]:
        _UTILITY_REGISTRY[name] = func
        return func

    return decorator


def get_registered_utility_types() -> List[str]:
    """Get list of registered utility type names."""
    return list(_UTILITY_REGISTRY.keys())


# --- Registered utility types ---

# Linear model utilities


@register_utility("linear_mean_mean_stdev")
def _create_linear_mean_mean_stdev(**kwargs: Any) -> UtilityConfig:
    """Linear Gaussian, mean noise stat, mean risk, stdev deviation."""
    return UtilityConfig(
        deviation_factory=lambda npred, bkd: StandardDeviationMeasure(npred, bkd),
        risk_factory=lambda bkd: SampleAverageMean(bkd),
        noise_stat_factory=lambda bkd: SampleAverageMean(bkd),
        exact_cls=ConjugateGaussianOEDDataMeanQoIMeanStdDev,
    )


@register_utility("linear_mean_mean_entropic")
def _create_linear_mean_mean_entropic(**kwargs: Any) -> UtilityConfig:
    """Linear Gaussian, mean noise stat, mean risk, entropic deviation."""
    lamda = kwargs.get("lamda", 0.5)
    return UtilityConfig(
        deviation_factory=lambda npred, bkd: EntropicDeviationMeasure(
            npred, lamda, bkd
        ),
        risk_factory=lambda bkd: SampleAverageMean(bkd),
        noise_stat_factory=lambda bkd: SampleAverageMean(bkd),
        exact_cls=ConjugateGaussianOEDDataMeanQoIMeanEntropicDev,
        exact_args=(lamda,),
    )


@register_utility("linear_avar_mean_avar")
def _create_linear_avar_mean_avar(**kwargs: Any) -> UtilityConfig:
    """Linear Gaussian, AVaR noise stat, mean risk, AVaR deviation."""
    beta = kwargs.get("beta", 0.5)
    delta = kwargs.get("delta", 100000)
    return UtilityConfig(
        deviation_factory=lambda npred, bkd: AVaRDeviationMeasure(
            npred, beta, bkd, delta
        ),
        risk_factory=lambda bkd: SampleAverageMean(bkd),
        noise_stat_factory=lambda bkd: SampleAverageSmoothedAVaR(
            beta, bkd, delta=delta
        ),
        exact_cls=ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev,
        exact_args=(beta,),
    )


@register_utility("linear_mean_avar_stdev")
def _create_linear_mean_avar_stdev(**kwargs: Any) -> UtilityConfig:
    """Linear Gaussian, mean noise stat, AVaR risk, stdev deviation."""
    beta = kwargs.get("beta", 0.5)
    delta = kwargs.get("delta", 100000)
    return UtilityConfig(
        deviation_factory=lambda npred, bkd: StandardDeviationMeasure(npred, bkd),
        risk_factory=lambda bkd: SampleAverageSmoothedAVaR(
            beta, bkd, delta=delta
        ),
        noise_stat_factory=lambda bkd: SampleAverageMean(bkd),
        exact_cls=ConjugateGaussianOEDDataMeanQoIAVaRStdDev,
        exact_args=(beta,),
    )


# Nonlinear (lognormal) model utilities


@register_utility("nonlinear_mean_mean_stdev")
def _create_nonlinear_mean_mean_stdev(**kwargs: Any) -> UtilityConfig:
    """Lognormal QoI, mean noise stat, mean risk, stdev deviation."""
    return UtilityConfig(
        deviation_factory=lambda npred, bkd: StandardDeviationMeasure(npred, bkd),
        risk_factory=lambda bkd: SampleAverageMean(bkd),
        noise_stat_factory=lambda bkd: SampleAverageMean(bkd),
        exact_cls=ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
    )


@register_utility("nonlinear_avar_mean_stdev")
def _create_nonlinear_avar_mean_stdev(**kwargs: Any) -> UtilityConfig:
    """Lognormal QoI, AVaR noise stat, mean risk, stdev deviation."""
    beta = kwargs.get("beta", 0.5)
    delta = kwargs.get("delta", 100000)
    return UtilityConfig(
        deviation_factory=lambda npred, bkd: StandardDeviationMeasure(npred, bkd),
        risk_factory=lambda bkd: SampleAverageMean(bkd),
        noise_stat_factory=lambda bkd: SampleAverageSmoothedAVaR(
            beta, bkd, delta=delta
        ),
        exact_cls=ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev,
        exact_args=(beta,),
    )


# Naming convention: {model}_{L3_data_risk}_{L2_qoi_risk}_{L1_deviation}
# L3 = noise_stat (over data realizations), L2 = risk (over QoI components)


@register_utility("nonlinear_mean_avar_stdev")
def _create_nonlinear_mean_avar_stdev(**kwargs: Any) -> UtilityConfig:
    """Lognormal QoI, mean noise stat, AVaR risk over QoI, stdev deviation.

    E_y[AVaR_alpha({Std(W_1|y), ..., Std(W_Q|y)})] for vector lognormal QoI.
    Requires degree-1 basis with equal posterior variance (equal-K condition).
    """
    alpha = kwargs.get("alpha", 0.5)
    delta = kwargs.get("delta", 100000)
    return UtilityConfig(
        deviation_factory=lambda npred, bkd: StandardDeviationMeasure(npred, bkd),
        risk_factory=lambda bkd: SampleAverageSmoothedAVaR(
            alpha, bkd, delta=delta
        ),
        noise_stat_factory=lambda bkd: SampleAverageMean(bkd),
        exact_cls=ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev,
        exact_args=(alpha,),
    )


@register_utility("nonlinear_mean_stdev_mean_stdev")
def _create_nonlinear_mean_stdev_mean_stdev(**kwargs: Any) -> UtilityConfig:
    """Lognormal QoI, mean+c*stdev noise stat, mean risk, stdev deviation.

    E_y[Std(W|y)] + c * Std_y[Std(W|y)] — safety margin utility.
    """
    from pyapprox.risk import SampleAverageMeanPlusStdev

    safety_factor = kwargs.get("safety_factor", 1.0)
    return UtilityConfig(
        deviation_factory=lambda npred, bkd: StandardDeviationMeasure(npred, bkd),
        risk_factory=lambda bkd: SampleAverageMean(bkd),
        noise_stat_factory=lambda bkd: SampleAverageMeanPlusStdev(
            safety_factor, bkd
        ),
        exact_cls=ConjugateGaussianOEDForLogNormalDataMeanStdDevQoIMeanStdDev,
        exact_args=(safety_factor,),
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
    deviation_factory : Callable[[int, Backend], DeviationMeasure]
        Factory that takes ``(npred, bkd)`` and returns a deviation measure.
    risk_factory : Callable[[Backend], SampleStatistic]
        Factory that takes ``(bkd,)`` and returns a risk measure.
    noise_stat_factory : Callable[[Backend], SampleStatistic]
        Factory that takes ``(bkd,)`` and returns a noise statistic.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        noise_variances: Array,
        npred: int,
        deviation_factory: Callable[[int, Backend[Array]], DeviationMeasure[Array]],
        risk_factory: Callable[[Backend[Array]], SampleStatistic[Array]],
        noise_stat_factory: Callable[[Backend[Array]], SampleStatistic[Array]],
        bkd: Backend[Array],
    ) -> None:
        self._noise_variances = noise_variances
        self._npred = npred
        self._deviation_factory = deviation_factory
        self._risk_factory = risk_factory
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
        deviation_measure = self._deviation_factory(self._npred, self._bkd)
        risk_measure = self._risk_factory(self._bkd)
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


def get_utility_factory(
    utility_type: str,
    **kwargs: Any,
) -> UtilityConfig:
    """Get utility configuration for a registered utility type.

    Parameters
    ----------
    utility_type : str
        Registered utility type name. Available types:

        - ``"linear_mean_mean_stdev"``: StdDev deviation, mean risk, mean
          noise stat. Exact class: ``ConjugateGaussianOEDDataMeanQoIMeanStdDev``.
        - ``"linear_mean_mean_entropic"``: Entropic deviation, mean risk,
          mean noise stat. Kwarg: ``lamda``. Exact class:
          ``ConjugateGaussianOEDDataMeanQoIMeanEntropicDev``.
        - ``"linear_avar_mean_avar"``: AVaR deviation, mean risk, AVaR
          noise stat. Kwargs: ``beta``, ``delta``. Exact class:
          ``ConjugateGaussianOEDDataAVaRQoIMeanAVaRDev``.
        - ``"linear_mean_avar_stdev"``: StdDev deviation, AVaR risk, mean
          noise stat. Kwargs: ``beta``, ``delta``. Exact class:
          ``ConjugateGaussianOEDDataMeanQoIAVaRStdDev``.
        - ``"nonlinear_mean_mean_stdev"``: StdDev deviation, mean risk,
          mean noise stat (lognormal QoI). Exact class:
          ``ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev``.
        - ``"nonlinear_avar_mean_stdev"``: StdDev deviation, mean risk,
          AVaR noise stat (lognormal QoI). Kwargs: ``beta``, ``delta``.
          Exact class: ``ConjugateGaussianOEDForLogNormalDataAVaRQoIMeanStdDev``.

    **kwargs
        Additional arguments for the utility type (e.g., ``beta`` for AVaR,
        ``lamda`` for entropic).

    Returns
    -------
    UtilityConfig
        Configuration containing deviation/risk/noise_stat factories,
        exact analytical class, and extra arguments.
    """
    if utility_type not in _UTILITY_REGISTRY:
        available = get_registered_utility_types()
        raise ValueError(
            f"Unknown utility_type: {utility_type}. Available types: {available}"
        )
    factory = _UTILITY_REGISTRY[utility_type]
    return factory(**kwargs)


def create_prediction_oed_diagnostics(
    noise_variances: Array,
    npred: int,
    utility_type: str,
    bkd: Backend[Array],
    **kwargs: Any,
) -> PredictionOEDDiagnostics[Array]:
    """Factory function to create PredictionOEDDiagnostics.

    Uses the registry to look up the appropriate component factories.

    Parameters
    ----------
    noise_variances : Array
        Per-observation noise variances. Shape: (nobs,).
    npred : int
        Number of prediction QoI locations.
    utility_type : str
        Registered utility type name (see ``get_utility_factory``).
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
    config = get_utility_factory(utility_type, **kwargs)

    return PredictionOEDDiagnostics(
        noise_variances, npred,
        config.deviation_factory, config.risk_factory, config.noise_stat_factory,
        bkd,
    )
