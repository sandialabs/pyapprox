"""Obstructed advection-diffusion OED problem wrappers.

Thin shells over the problem classes in
:mod:`pyapprox_benchmarks.problems.oed.advection_diffusion`.

These have no analytical ground truth and are therefore Problems,
not Benchmarks. The shell classes provide convenience accessors
(``noise_std``, ``noise_var``, transitional forwarders) while
consumers migrate to ``problem()`` access.
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

from pyapprox_benchmarks.problems.oed.advection_diffusion import (
    AdvectionDiffusionOEDProblem,
    FixedVelocityAdvectionDiffusionOEDProblem,
)

# ---------------------------------------------------------------------------
# ObstructedAdvectionDiffusionOEDProblemWrapper (Pattern A forwarders preserved)
# ---------------------------------------------------------------------------


class ObstructedAdvectionDiffusionOEDProblemWrapper(Generic[Array]):
    """Obstructed advection-diffusion prediction OED problem wrapper.

    No analytical ground truth — this is a Problem wrapper, not a
    Benchmark.  Kept as a class for transitional forwarders while
    consumers migrate to ``problem()`` access.

    Parameters
    ----------
    problem : AdvectionDiffusionOEDProblem[Array]
        The prediction OED problem.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        problem: AdvectionDiffusionOEDProblem[Array],
        bkd: Backend[Array],
    ) -> None:
        self._problem = problem
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def problem(self) -> AdvectionDiffusionOEDProblem[Array]:
        """Get the prediction OED problem."""
        return self._problem

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


def build_obstructed_advection_diffusion_oed_problem(
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
) -> ObstructedAdvectionDiffusionOEDProblemWrapper[Array]:
    """Build an :class:`ObstructedAdvectionDiffusionOEDProblemWrapper`."""
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
    return ObstructedAdvectionDiffusionOEDProblemWrapper(
        problem=problem,
        bkd=bkd,
    )


# ---------------------------------------------------------------------------
# FixedVelocityObstructedAdvectionDiffusionOEDProblemWrapper (Pattern B only)
# ---------------------------------------------------------------------------


class FixedVelocityObstructedAdvectionDiffusionOEDProblemWrapper(Generic[Array]):
    """Fixed-velocity advection-diffusion OED problem wrapper.

    No analytical ground truth — this is a Problem wrapper, not a
    Benchmark. Callers go through ``problem()`` for all access.

    Parameters
    ----------
    problem : FixedVelocityAdvectionDiffusionOEDProblem[Array]
        The prediction OED problem.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        problem: FixedVelocityAdvectionDiffusionOEDProblem[Array],
        bkd: Backend[Array],
    ) -> None:
        self._problem = problem
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def problem(
        self,
    ) -> FixedVelocityAdvectionDiffusionOEDProblem[Array]:
        """Get the prediction OED problem."""
        return self._problem

    def noise_std(self) -> float:
        """Noise standard deviation (delegated to the problem)."""
        return self._problem.noise_std()

    def noise_var(self) -> float:
        """Noise variance (delegated to the problem)."""
        return self._problem.noise_std() ** 2


def build_fixed_velocity_obstructed_advection_diffusion_oed_problem(
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
) -> FixedVelocityObstructedAdvectionDiffusionOEDProblemWrapper[Array]:
    """Build a :class:`FixedVelocityObstructedAdvectionDiffusionOEDProblemWrapper`."""
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
    return FixedVelocityObstructedAdvectionDiffusionOEDProblemWrapper(
        problem=problem,
        bkd=bkd,
    )
