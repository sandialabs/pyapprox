"""Linear Gaussian KL-OED benchmark with analytical EIG.

Polynomial regression problem where:
- Design locations are in [-1, 1]
- Forward model is polynomial basis evaluation
- Prior and noise are isotropic Gaussian
- Exact EIG via conjugate Gaussian posterior
"""

from __future__ import annotations

from typing import Callable, Generic

from pyapprox_benchmarks.functions.algebraic.linear_gaussian_oed import (
    _build_vandermonde,
)
from pyapprox_benchmarks.problems.inverse import (
    build_linear_gaussian_inference_problem,
)
from pyapprox_benchmarks.problems.oed import KLOEDProblem
from pyapprox.expdesign.utils import compute_exact_eig
from pyapprox.util.backends.protocols import Array, Backend


class LinearGaussianKLOEDBenchmark(Generic[Array]):
    """Fixed KL-OED benchmark with analytical EIG.

    Access inference details via ``benchmark.problem()``.

    Parameters
    ----------
    problem : KLOEDProblem[Array]
        The OED problem.
    exact_eig_fn : callable
        Callable computing exact EIG from weights.
    bkd : Backend[Array]
        Computational backend.
    noise_std : float
        Noise standard deviation (benchmark configuration).
    prior_std : float
        Prior standard deviation (benchmark configuration).
    design_matrix : Array
        Design matrix A. Shape: (nobs, nparams).
    obs_locations : Array
        Observation locations. Shape: (nobs,).
    """

    def __init__(
        self,
        problem: KLOEDProblem[Array],
        exact_eig_fn: Callable[[Array], float],
        bkd: Backend[Array],
        noise_std: float,
        prior_std: float,
        design_matrix: Array,
        obs_locations: Array,
    ) -> None:
        self._problem = problem
        self._exact_eig_fn = exact_eig_fn
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

    def exact_eig(self, weights: Array) -> float:
        """Compute exact EIG for the given design weights."""
        return self._exact_eig_fn(weights)

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
    exact_eig_fn = lambda w: compute_exact_eig(problem, w)  # noqa: E731
    return LinearGaussianKLOEDBenchmark(
        problem, exact_eig_fn, bkd, noise_std, prior_std,
        design_matrix, obs_locations,
    )
