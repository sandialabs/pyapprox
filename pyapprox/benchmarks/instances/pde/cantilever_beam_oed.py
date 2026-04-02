"""Cantilever beam 2D load identification OED benchmark.

Composes KLOEDProblem + OEDGroundTruth for sensor placement on a
2D cantilever beam. The forward model maps two load parameters
(constant + slope traction) to y-displacements at sensor locations.

Structure mirrors expdesign/benchmarks/instances/linear_gaussian.py:
- obs_map built by cantilever_beam_obs_map.build_cantilever_beam_obs_map
- BayesianInferenceProblem holds obs_map + prior + noise
- KLOEDProblem adds design conditions
- OEDGroundTruth provides exact EIG via conjugate Gaussian

Migration note: This file stays in instances/pde/ during the full
benchmarks/ refactor. The obs_map builder (cantilever_beam_obs_map.py)
moves to benchmarks/functions/pde/. Problem classes (BayesianInferenceProblem,
KLOEDProblem) are imported from expdesign/benchmarks/problems/ and will
eventually move to a shared location (e.g. pyapprox/inverse/).
"""

from typing import Generic, Optional

from pyapprox.benchmarks.instances.pde.cantilever_beam import (
    _DEFAULT_MESH_PATH,
)
from pyapprox.benchmarks.instances.pde.cantilever_beam_obs_map import (
    build_cantilever_beam_design_matrix,
    build_cantilever_beam_obs_map,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.expdesign.benchmarks.ground_truth import OEDGroundTruth
from pyapprox.expdesign.benchmarks.problems.inference_problem import (
    GaussianInferenceProblem,
)
from pyapprox.expdesign.benchmarks.problems.kl_problem import KLOEDProblem
from pyapprox.expdesign.utils import compute_exact_eig
from pyapprox.probability.gaussian import DenseCholeskyMultivariateGaussian
from pyapprox.util.backends.protocols import Array, Backend


class CantileverBeam2DLoadOEDBenchmark(Generic[Array]):
    """OED benchmark for 2D cantilever beam load identification.

    Composes a KLOEDProblem + OEDGroundTruth. Access inference details
    via ``benchmark.problem()``.

    The beam has distributed surface traction:
        t_y(x) = theta_1 * (-1) + theta_2 * (-x / L)

    The design matrix A (nobs, 2) maps load parameters to
    y-displacements at sensor locations via FEM superposition.

    Parameters
    ----------
    problem : KLOEDProblem[Array]
        The OED problem.
    ground_truth : OEDGroundTruth
        Ground truth with exact_eig callable.
    bkd : Backend[Array]
        Computational backend.
    noise_std : float
        Noise standard deviation.
    design_matrix : Array
        Design matrix A. Shape: (nobs, 2).
    sensor_xs : Array
        Sensor x-coordinates. Shape: (nobs,).
    """

    def __init__(
        self,
        problem: KLOEDProblem[Array],
        ground_truth: OEDGroundTruth,
        bkd: Backend[Array],
        noise_std: float,
        design_matrix: Array,
        sensor_xs: Array,
    ) -> None:
        self._problem = problem
        self._ground_truth = ground_truth
        self._bkd = bkd
        self._noise_std = noise_std
        self._design_matrix = design_matrix
        self._sensor_xs = sensor_xs

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

    # --- Benchmark configuration ---

    def noise_std(self) -> float:
        """Noise standard deviation."""
        return self._noise_std

    def noise_var(self) -> float:
        """Noise variance."""
        return self._noise_std**2

    # --- Matrices ---

    def design_matrix(self) -> Array:
        """Get design matrix A. Shape: (nobs, 2)."""
        return self._design_matrix

    def sensor_xs(self) -> Array:
        """Get sensor x-coordinates. Shape: (nobs,)."""
        return self._sensor_xs

    def design_locations(self) -> Array:
        """Get sensor x-coordinates. Shape: (nobs,)."""
        return self._sensor_xs


def build_cantilever_beam_oed_benchmark(
    bkd: Backend[Array],
    mesh_path: str = _DEFAULT_MESH_PATH,
    length: float = 100.0,
    height: float = 30.0,
    E_mean: float = 1e4,
    poisson_ratio: float = 0.3,
    prior_mean: Optional[Array] = None,
    prior_covariance: Optional[Array] = None,
    noise_std: float = 0.01,
    sensor_xs: Optional[Array] = None,
) -> CantileverBeam2DLoadOEDBenchmark[Array]:
    """Build a cantilever beam 2D load OED benchmark.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    mesh_path : str
        Path to JSON mesh file.
    length : float
        Beam length.
    height : float
        Beam height.
    E_mean : float
        Young's modulus.
    poisson_ratio : float
        Poisson ratio.
    prior_mean : Array or None
        Prior mean for theta. Shape (2, 1). Default: zeros.
    prior_covariance : Array or None
        Prior covariance. Shape (2, 2). Default: eye(2).
    noise_std : float
        Observation noise standard deviation.
    sensor_xs : Array or None
        Sensor x-coordinates. Shape (nobs,).
        Default: 5 equally spaced in [length/5, length].

    Returns
    -------
    CantileverBeam2DLoadOEDBenchmark
        Configured benchmark.
    """
    # Build design matrix from FEM
    design_matrix, sensor_xs = build_cantilever_beam_design_matrix(
        bkd, mesh_path, length, height, E_mean, poisson_ratio, sensor_xs,
    )
    nobs = design_matrix.shape[0]
    nparams = 2

    # Prior
    if prior_mean is None:
        prior_mean = bkd.zeros((nparams, 1))
    if prior_covariance is None:
        prior_covariance = bkd.eye(nparams)

    # Build obs_map
    obs_map = build_cantilever_beam_obs_map(design_matrix, bkd)

    # Build prior distribution
    prior_dist = DenseCholeskyMultivariateGaussian(
        prior_mean, prior_covariance, bkd,
    )

    # Noise
    noise_variances = bkd.full((nobs,), noise_std**2)

    # Compose problem hierarchy
    inference = GaussianInferenceProblem(
        obs_map=obs_map,
        prior=prior_dist,
        noise_variances=noise_variances,
        bkd=bkd,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
    )
    conditions = bkd.reshape(sensor_xs, (nobs, 1))
    problem = KLOEDProblem(inference, conditions, bkd)
    ground_truth = OEDGroundTruth(
        exact_eig=lambda w: compute_exact_eig(problem, w),
    )

    return CantileverBeam2DLoadOEDBenchmark(
        problem, ground_truth, bkd, noise_std, design_matrix, sensor_xs,
    )


@BenchmarkRegistry.register(
    "cantilever_beam_2d_load_oed",
    category="oed",
    description=(
        "2D cantilever beam load identification OED with analytical EIG"
    ),
)
def _cantilever_beam_2d_load_oed_factory(
    bkd: Backend[Array],
) -> CantileverBeam2DLoadOEDBenchmark[Array]:
    return build_cantilever_beam_oed_benchmark(bkd)
