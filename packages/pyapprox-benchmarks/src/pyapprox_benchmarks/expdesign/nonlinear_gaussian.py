"""Nonlinear Gaussian prediction OED benchmark.

Linear observation model with nonlinear QoI: qoi = exp(B @ theta).
The QoI is lognormal, enabling analytical utility computation via
conjugate Gaussian formulas.
"""

from __future__ import annotations

from typing import Any, Generic

from pyapprox_benchmarks.functions.algebraic.linear_gaussian_oed import (
    _build_vandermonde,
    build_exp_qoi_map,
)
from pyapprox_benchmarks.problems.inverse import (
    build_linear_gaussian_inference_problem,
)
from pyapprox_benchmarks.problems.oed import PredictionOEDProblem
from pyapprox.expdesign.diagnostics import (
    compute_exact_prediction_utility,
    get_utility_factory,
)
from pyapprox.util.backends.protocols import Array, Backend


class NonLinearGaussianPredOEDBenchmark(Generic[Array]):
    """Fixed prediction OED benchmark with exp(B@theta) QoI.

    Ground truth: exact prediction utility via conjugate Gaussian
    formulas for lognormal QoI. Access via ``exact_risk_utility()``.

    Access inference details via ``benchmark.problem()``.

    Parameters
    ----------
    problem : PredictionOEDProblem[Array]
        The prediction OED problem.
    bkd : Backend[Array]
        Computational backend.
    noise_std : float
        Noise standard deviation (benchmark configuration).
    prior_std : float
        Prior standard deviation (benchmark configuration).
    design_matrix : Array
        Observation design matrix A. Shape: (nobs, nparams).
    qoi_matrix : Array
        QoI design matrix B. Shape: (npred, nparams).
    obs_locations : Array
        Observation locations. Shape: (nobs,).
    qoi_locations : Array
        QoI prediction locations. Shape: (npred,).
    qoi_quad_weights : Array
        QoI quadrature weights. Shape: (npred, 1).
    """

    def __init__(
        self,
        problem: PredictionOEDProblem[Array],
        bkd: Backend[Array],
        noise_std: float,
        prior_std: float,
        design_matrix: Array,
        qoi_matrix: Array,
        obs_locations: Array,
        qoi_locations: Array,
        qoi_quad_weights: Array,
    ) -> None:
        self._problem = problem
        self._bkd = bkd
        self._noise_std = noise_std
        self._prior_std = prior_std
        self._design_matrix = design_matrix
        self._qoi_matrix = qoi_matrix
        self._obs_locations = obs_locations
        self._qoi_locations = qoi_locations
        self._qoi_quad_weights = qoi_quad_weights

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def problem(self) -> PredictionOEDProblem[Array]:
        """Get the prediction OED problem."""
        return self._problem

    # --- Ground truth ---

    def exact_risk_utility(
        self, weights: Array, utility_type: str, **kwargs: Any,
    ) -> float:
        """Compute exact prediction utility via conjugate Gaussian formulas.

        Parameters
        ----------
        weights : Array
            Design weights. Shape: (nobs, 1).
        utility_type : str
            Registered utility type name, e.g.
            ``"nonlinear_mean_mean_stdev"``,
            ``"nonlinear_avar_mean_stdev"``.
        **kwargs : Any
            Extra arguments for the utility type (e.g. ``beta=0.5``).

        Returns
        -------
        float
            Exact expected utility value.
        """
        config = get_utility_factory(utility_type, **kwargs)
        return compute_exact_prediction_utility(
            self._problem.prior_mean(),
            self._problem.prior_covariance(),
            self._design_matrix,
            self._qoi_matrix,
            self.noise_var(),
            weights,
            config.exact_cls,
            config.exact_args,
            self._bkd,
        )

    # --- Benchmark configuration (not on problem) ---

    def noise_std(self) -> float:
        """Noise standard deviation."""
        return self._noise_std

    def prior_std(self) -> float:
        """Prior standard deviation."""
        return self._prior_std

    def noise_var(self) -> float:
        """Noise variance."""
        return self._noise_std**2

    def prior_var(self) -> float:
        """Prior variance."""
        return self._prior_std**2

    def qoi_quad_weights(self) -> Array:
        """Get QoI quadrature weights. Shape: (npred, 1)."""
        return self._qoi_quad_weights

    # --- Matrices (TODO: move to problem once maps expose them) ---

    def design_matrix(self) -> Array:
        """Get observation design matrix A. Shape: (nobs, nparams)."""
        return self._design_matrix

    def qoi_matrix(self) -> Array:
        """Get QoI design matrix B. Shape: (npred, nparams)."""
        return self._qoi_matrix

    def design_locations(self) -> Array:
        """Get observation locations. Shape: (nobs,)."""
        return self._obs_locations

    def qoi_locations(self) -> Array:
        """Get QoI prediction locations. Shape: (npred,)."""
        return self._qoi_locations


def build_nonlinear_gaussian_pred_benchmark(
    nobs: int,
    degree: int,
    noise_std: float,
    prior_std: float,
    bkd: Backend[Array],
    npred: int = 1,
    min_degree: int = 0,
) -> NonLinearGaussianPredOEDBenchmark[Array]:
    """Build a NonLinearGaussianPredOEDBenchmark.

    Parameters
    ----------
    nobs : int
        Number of observation locations.
    degree : int
        Maximum polynomial degree.
    noise_std : float
        Noise standard deviation.
    prior_std : float
        Prior standard deviation.
    bkd : Backend[Array]
        Computational backend.
    npred : int
        Number of prediction locations (default 1).
    min_degree : int
        Minimum polynomial degree (default 0).

    Returns
    -------
    NonLinearGaussianPredOEDBenchmark
        Configured benchmark.
    """
    inference = build_linear_gaussian_inference_problem(
        nobs, degree, noise_std, prior_std, bkd, min_degree
    )
    obs_locations = bkd.linspace(-1.0, 1.0, nobs)
    design_matrix = _build_vandermonde(obs_locations, min_degree, degree, bkd)
    qoi_locations = bkd.linspace(-2.0 / 3.0, 2.0 / 3.0, npred)
    qoi_matrix = _build_vandermonde(qoi_locations, min_degree, degree, bkd)
    qoi_quad_weights = bkd.full((npred, 1), 1.0 / npred)

    qoi_map = build_exp_qoi_map(qoi_locations, min_degree, degree, bkd)
    conditions = bkd.reshape(obs_locations, (nobs, 1))
    problem = PredictionOEDProblem(
        inference, qoi_map, conditions, bkd
    )
    return NonLinearGaussianPredOEDBenchmark(
        problem, bkd, noise_std, prior_std,
        design_matrix, qoi_matrix, obs_locations, qoi_locations,
        qoi_quad_weights,
    )
