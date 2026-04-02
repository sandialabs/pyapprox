"""Linear Gaussian KL-OED benchmark with analytical EIG.

Polynomial regression problem where:
- Design locations are in [-1, 1]
- Forward model is polynomial basis evaluation
- Prior and noise are isotropic Gaussian
- Exact EIG via conjugate Gaussian posterior
"""

from __future__ import annotations

from typing import Generic

from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.expdesign.benchmarks.functions.linear_gaussian import (
    _build_vandermonde,
    build_linear_gaussian_inference_problem,
)
from pyapprox.expdesign.benchmarks.ground_truth import OEDGroundTruth
from pyapprox.expdesign.benchmarks.problems.kl_problem import KLOEDProblem
from pyapprox.expdesign.utils import compute_exact_eig
from pyapprox.util.backends.protocols import Array, Backend


class LinearGaussianKLOEDBenchmark(Generic[Array]):
    """Fixed KL-OED benchmark with analytical EIG.

    Composes a KLOEDProblem + OEDGroundTruth. Access inference details
    via ``benchmark.problem()``.

    Parameters
    ----------
    problem : KLOEDProblem[Array]
        The OED problem.
    ground_truth : OEDGroundTruth
        Ground truth with exact_eig callable.
    bkd : Backend[Array]
        Computational backend.
    noise_std : float
        Noise standard deviation (benchmark configuration).
    prior_std : float
        Prior standard deviation (benchmark configuration).
    design_matrix : Array
        Design matrix A. Shape: (nobs, nparams).
        TODO: move to problem once maps expose their matrices.
    obs_locations : Array
        Observation locations. Shape: (nobs,).
        TODO: move to problem once design_conditions generalizes.
    """

    def __init__(
        self,
        problem: KLOEDProblem[Array],
        ground_truth: OEDGroundTruth,
        bkd: Backend[Array],
        noise_std: float,
        prior_std: float,
        design_matrix: Array,
        obs_locations: Array,
    ) -> None:
        self._problem = problem
        self._ground_truth = ground_truth
        self._bkd = bkd
        self._noise_std = noise_std
        self._prior_std = prior_std
        self._design_matrix = design_matrix
        self._obs_locations = obs_locations

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def problem(self) -> KLOEDProblem[Array]:
        """Get the OED problem."""
        return self._problem

    def ground_truth(self) -> OEDGroundTruth:
        """Get the ground truth."""
        return self._ground_truth

    def exact_eig(self, weights: Array) -> float:
        """Compute exact EIG (delegates to ground truth)."""
        assert self._ground_truth.exact_eig is not None
        return self._ground_truth.exact_eig(weights)

    def d_optimal_objective(self, weights: Array) -> float:
        """D-optimal objective (negative EIG)."""
        return -self.exact_eig(weights)

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

    # --- Matrices (TODO: move to problem once maps expose them) ---

    def design_matrix(self) -> Array:
        """Get design matrix A. Shape: (nobs, nparams)."""
        return self._design_matrix

    def design_locations(self) -> Array:
        """Get design locations. Shape: (nobs,)."""
        return self._obs_locations


def build_linear_gaussian_kl_benchmark(
    nobs: int,
    degree: int,
    noise_std: float,
    prior_std: float,
    bkd: Backend[Array],
    min_degree: int = 0,
) -> LinearGaussianKLOEDBenchmark[Array]:
    """Build a LinearGaussianKLOEDBenchmark.

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
    min_degree : int
        Minimum polynomial degree (default 0).

    Returns
    -------
    LinearGaussianKLOEDBenchmark
        Configured benchmark.
    """
    inference = build_linear_gaussian_inference_problem(
        nobs, degree, noise_std, prior_std, bkd, min_degree
    )
    obs_locations = bkd.linspace(-1.0, 1.0, nobs)
    design_matrix = _build_vandermonde(obs_locations, min_degree, degree, bkd)
    conditions = bkd.reshape(obs_locations, (nobs, 1))
    problem = KLOEDProblem(inference, conditions, bkd)
    ground_truth = OEDGroundTruth(
        exact_eig=lambda w: compute_exact_eig(problem, w)
    )
    return LinearGaussianKLOEDBenchmark(
        problem, ground_truth, bkd, noise_std, prior_std,
        design_matrix, obs_locations,
    )


@BenchmarkRegistry.register(
    "linear_gaussian_kl_oed",
    category="oed",
    description="Linear Gaussian KL-OED benchmark with analytical EIG",
)
def _linear_gaussian_kl_oed_factory(
    bkd: Backend[Array],
) -> LinearGaussianKLOEDBenchmark[Array]:
    return build_linear_gaussian_kl_benchmark(
        nobs=10, degree=3, noise_std=1.0, prior_std=1.0, bkd=bkd,
    )
