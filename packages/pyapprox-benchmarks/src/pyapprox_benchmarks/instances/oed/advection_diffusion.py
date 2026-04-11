"""Obstructed advection-diffusion OED benchmarks.

Thin benchmark shells over the problem classes in
:mod:`pyapprox_benchmarks.problems.oed.advection_diffusion`. Two
benchmarks are registered:

- ``"obstructed_advection_diffusion_oed"`` — full 13-dim parameter
  space (10 KLE terms + 2 inlet shape + 1 Reynolds). Preserves
  **transitional Pattern A forwarders** (``prior()``, ``obs_map()``,
  ``qoi_map()``, ``evaluate_nodal()``, ``solve_for_plotting()``, …)
  so existing tutorial and paper consumers keep working while
  they migrate to ``benchmark.problem().fun()``. These forwarders
  will be removed in a follow-up commit.
- ``"obstructed_advection_diffusion_oed_fixed_velocity"`` — pins
  velocity at construction, pre-caches Stokes, reduces to the
  ``nkle_terms``-dim KLE prior. **Pattern B only** (no
  forwarders). Callers go through ``benchmark.problem()``.

See ``instances/oed/__init__.py`` for the full Pattern A vs B
convention story.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Literal,
    Optional,
    Tuple,
)

if TYPE_CHECKING:
    from pyapprox.interface.functions.protocols import FunctionProtocol
    from pyapprox.probability.protocols import DistributionProtocol

from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.ground_truth import OEDGroundTruth
from pyapprox_benchmarks.problems.oed.advection_diffusion import (
    AdvectionDiffusionOEDProblem,
    FixedVelocityAdvectionDiffusionOEDProblem,
)
from pyapprox_benchmarks.registry import BenchmarkRegistry

# ---------------------------------------------------------------------------
# ObstructedAdvectionDiffusionOEDBenchmark (Pattern A forwarders preserved)
# ---------------------------------------------------------------------------


class ObstructedAdvectionDiffusionOEDBenchmark(Generic[Array]):
    """Obstructed advection-diffusion prediction OED benchmark.

    Composition shell over
    :class:`AdvectionDiffusionOEDProblem` and
    :class:`OEDGroundTruth`. Registered under
    ``"obstructed_advection_diffusion_oed"``.

    **Preferred access**: go through :meth:`problem` for the prior,
    maps, and PDE evaluation methods::

        bench = BenchmarkRegistry.create("obstructed_advection_diffusion_oed")
        problem = bench.problem()
        prior = problem.prior()
        obs_map = problem.obs_map()

    The direct accessors (``prior()``, ``obs_map()``, ``qoi_map()``,
    ``design_conditions()``, ``nparams()``, ``nobservations()``,
    ``evaluate_nodal()``, ``evaluate_both()``, ``solve_for_plotting()``,
    ``mesh_nodes()``, ``nnodes()``) on this class are **transitional
    Pattern A forwarders**. They exist only so this migration can
    ship without breaking existing tutorial / paper consumers in the
    same commit, and they will be removed once those consumers have
    moved to ``benchmark.problem().fun()``. See the Pattern A vs
    Pattern B note in :mod:`pyapprox_benchmarks.instances.oed`.
    """

    def __init__(
        self,
        problem: AdvectionDiffusionOEDProblem[Array],
        ground_truth: OEDGroundTruth,
        bkd: Backend[Array],
    ) -> None:
        self._problem = problem
        self._ground_truth = ground_truth
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def problem(self) -> AdvectionDiffusionOEDProblem[Array]:
        """Get the prediction OED problem."""
        return self._problem

    def ground_truth(self) -> OEDGroundTruth:
        """Get the ground truth."""
        return self._ground_truth

    def noise_std(self) -> float:
        """Noise standard deviation (delegated to the problem)."""
        return self._problem.noise_std()

    def noise_var(self) -> float:
        """Noise variance (delegated to the problem)."""
        return self._problem.noise_std() ** 2

    # ------------------------------------------------------------------
    # Transitional Pattern A forwarders (to be removed in a later commit)
    # ------------------------------------------------------------------

    def prior(self) -> "DistributionProtocol[Array]":
        """Return the prior distribution (transitional forwarder)."""
        return self._problem.prior()

    def obs_map(self) -> "FunctionProtocol[Array]":
        """Return the observation map (transitional forwarder)."""
        return self._problem.obs_map()

    def qoi_map(self) -> "FunctionProtocol[Array]":
        """Return the QoI map (transitional forwarder)."""
        return self._problem.qoi_map()

    def design_conditions(self) -> Array:
        """Return sensor locations (transitional forwarder)."""
        return self._problem.design_conditions()

    def nparams(self) -> int:
        """Return number of parameters (transitional forwarder)."""
        return self._problem.nparams()

    def nobservations(self) -> int:
        """Return number of observation sensors (transitional forwarder)."""
        return self._problem.nobs()

    def evaluate_nodal(self, samples: Array) -> Array:
        """Evaluate full nodal concentration (transitional forwarder)."""
        return self._problem.evaluate_nodal(samples)

    def evaluate_both(
        self, samples: Array,
    ) -> Tuple[Array, Array]:
        """Evaluate obs and prediction maps (transitional forwarder)."""
        return self._problem.evaluate_both(samples)

    def solve_for_plotting(
        self, sample: Array,
    ) -> Dict[str, Any]:
        """Plotting-data forward solve (transitional forwarder)."""
        return self._problem.solve_for_plotting(sample)

    def mesh_nodes(self) -> Array:
        """Return ADR mesh node coordinates (transitional forwarder)."""
        return self._problem.mesh_nodes()

    def nnodes(self) -> int:
        """Return number of ADR mesh nodes (transitional forwarder)."""
        return self._problem.nnodes()


def build_obstructed_advection_diffusion_oed_benchmark(
    bkd: Backend[Array],
    *,
    noise_std: float = 0.1,
    nstokes_refine: int = 3,
    nadvec_diff_refine: int = 3,
    nkle_terms: int = 10,
    nsensors: int = 20,
    diffusivity: float = 0.1,
    final_time: float = 1.5,
    deltat: float = 0.25,
    kle_subdomain: Optional[Tuple[float, float, float, float]] = (
        0.0, 0.25, 0.0, 1.0,
    ),
    kle_correlation_length: float = 0.1,
    kle_sigma: float = 0.3,
    source_mode: Literal["forcing", "initial_condition"] = "forcing",
) -> ObstructedAdvectionDiffusionOEDBenchmark[Array]:
    """Build an :class:`ObstructedAdvectionDiffusionOEDBenchmark`.

    Constructs the underlying :class:`AdvectionDiffusionOEDProblem`,
    wraps it with an empty :class:`OEDGroundTruth` (no closed-form
    reference), and returns the composition shell. See
    :class:`AdvectionDiffusionOEDProblem` for parameter documentation.
    """
    problem = AdvectionDiffusionOEDProblem(
        bkd,
        noise_std=noise_std,
        nstokes_refine=nstokes_refine,
        nadvec_diff_refine=nadvec_diff_refine,
        nkle_terms=nkle_terms,
        nsensors=nsensors,
        diffusivity=diffusivity,
        final_time=final_time,
        deltat=deltat,
        kle_subdomain=kle_subdomain,
        kle_correlation_length=kle_correlation_length,
        kle_sigma=kle_sigma,
        source_mode=source_mode,
    )
    return ObstructedAdvectionDiffusionOEDBenchmark(
        problem=problem,
        ground_truth=OEDGroundTruth(),
        bkd=bkd,
    )


@BenchmarkRegistry.register(
    "obstructed_advection_diffusion_oed",
    category="oed",
    description=(
        "Obstructed advection-diffusion OED benchmark with Stokes coupling"
    ),
)
def _obstructed_advection_diffusion_oed_factory(
    bkd: Backend[Array],
) -> ObstructedAdvectionDiffusionOEDBenchmark[Array]:
    return build_obstructed_advection_diffusion_oed_benchmark(
        bkd, noise_std=0.1,
    )


# ---------------------------------------------------------------------------
# FixedVelocityObstructedAdvectionDiffusionOEDBenchmark (Pattern B only)
# ---------------------------------------------------------------------------


class FixedVelocityObstructedAdvectionDiffusionOEDBenchmark(Generic[Array]):
    """Fixed-velocity advection-diffusion OED benchmark.

    Pattern B only. Callers must go through :meth:`problem` to reach
    the prior, observation map, QoI map, design conditions, and PDE
    evaluation methods::

        bench = BenchmarkRegistry.create(
            "obstructed_advection_diffusion_oed_fixed_velocity",
        )
        problem = bench.problem()
        prior = problem.prior()
        vals = problem.evaluate_nodal(samples)

    The pinned velocity parameters (``vel_shape_a``, ``vel_shape_b``,
    ``reynolds_num``) live on the problem object, not the benchmark
    shell — they define the problem, not the benchmark packaging.

    This shell mirrors :class:`LinearGaussianPredOEDBenchmark` and is
    the template every future OED benchmark should follow. See the
    convention note in :mod:`pyapprox_benchmarks.instances.oed` for
    the Pattern A vs Pattern B story.
    """

    def __init__(
        self,
        problem: FixedVelocityAdvectionDiffusionOEDProblem[Array],
        ground_truth: OEDGroundTruth,
        bkd: Backend[Array],
    ) -> None:
        self._problem = problem
        self._ground_truth = ground_truth
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def problem(
        self,
    ) -> FixedVelocityAdvectionDiffusionOEDProblem[Array]:
        """Get the prediction OED problem."""
        return self._problem

    def ground_truth(self) -> OEDGroundTruth:
        """Get the ground truth."""
        return self._ground_truth

    def noise_std(self) -> float:
        """Noise standard deviation (delegated to the problem)."""
        return self._problem.noise_std()

    def noise_var(self) -> float:
        """Noise variance (delegated to the problem)."""
        return self._problem.noise_std() ** 2


def build_fixed_velocity_obstructed_advection_diffusion_oed_benchmark(
    bkd: Backend[Array],
    *,
    vel_shape_a: float = 2.5,
    vel_shape_b: float = 2.5,
    reynolds_num: float = 12.5,
    noise_std: float = 0.1,
    nstokes_refine: int = 3,
    nadvec_diff_refine: int = 3,
    nkle_terms: int = 10,
    nsensors: int = 20,
    diffusivity: float = 0.1,
    final_time: float = 1.5,
    deltat: float = 0.25,
    kle_subdomain: Optional[Tuple[float, float, float, float]] = (
        0.0, 0.25, 0.0, 1.0,
    ),
    kle_correlation_length: float = 0.1,
    kle_sigma: float = 0.3,
    source_mode: Literal["forcing", "initial_condition"] = "forcing",
) -> FixedVelocityObstructedAdvectionDiffusionOEDBenchmark[Array]:
    """Build a :class:`FixedVelocityObstructedAdvectionDiffusionOEDBenchmark`.

    Constructs the underlying
    :class:`FixedVelocityAdvectionDiffusionOEDProblem` (which pins
    velocity, pre-caches Stokes, and reduces the prior), wraps it
    with an empty :class:`OEDGroundTruth`, and returns the composition
    shell. See :class:`AdvectionDiffusionOEDProblem` /
    :class:`FixedVelocityAdvectionDiffusionOEDProblem` for parameter
    documentation.
    """
    problem = FixedVelocityAdvectionDiffusionOEDProblem(
        bkd,
        vel_shape_a=vel_shape_a,
        vel_shape_b=vel_shape_b,
        reynolds_num=reynolds_num,
        noise_std=noise_std,
        nstokes_refine=nstokes_refine,
        nadvec_diff_refine=nadvec_diff_refine,
        nkle_terms=nkle_terms,
        nsensors=nsensors,
        diffusivity=diffusivity,
        final_time=final_time,
        deltat=deltat,
        kle_subdomain=kle_subdomain,
        kle_correlation_length=kle_correlation_length,
        kle_sigma=kle_sigma,
        source_mode=source_mode,
    )
    return FixedVelocityObstructedAdvectionDiffusionOEDBenchmark(
        problem=problem,
        ground_truth=OEDGroundTruth(),
        bkd=bkd,
    )


@BenchmarkRegistry.register(
    "obstructed_advection_diffusion_oed_fixed_velocity",
    category="oed",
    description=(
        "Fixed-velocity obstructed advection-diffusion OED benchmark. "
        "Velocity pinned at construction; Stokes solved once; OED "
        "problem reduced to the KLE prior."
    ),
)
def _obstructed_advection_diffusion_oed_fixed_velocity_factory(
    bkd: Backend[Array],
) -> FixedVelocityObstructedAdvectionDiffusionOEDBenchmark[Array]:
    return build_fixed_velocity_obstructed_advection_diffusion_oed_benchmark(
        bkd, noise_std=0.1,
    )
