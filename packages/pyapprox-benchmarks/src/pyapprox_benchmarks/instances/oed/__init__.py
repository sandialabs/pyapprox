"""OED benchmark instances.

Convention note — benchmark surface Pattern A vs Pattern B
==========================================================

OED benchmarks in this package currently expose their inference
substrate two different ways:

**Pattern A (majority, direct forwarding on the benchmark):**

    bench = BenchmarkRegistry.get(
        "obstructed_advection_diffusion_oed", bkd=bkd,
    )
    prior = bench.prior()
    obs_map = bench.obs_map()
    qoi_map = bench.qoi_map()

Used by :mod:`advection_diffusion`, :mod:`lotka_volterra`, and by
the majority of tutorials across ``pyapprox-tutorials``,
``pyapprox-papers``, and the expdesign test suite. Convenient for
tutorials but conflates two concerns: the *problem* (prior, maps,
forward solver) and the *benchmark* (registry metadata, ground
truth, config).

**Pattern B (ideal, composition through ``.problem()``):**

    bench = BenchmarkRegistry.get("linear_gaussian_pred_oed", bkd=bkd)
    problem = bench.problem()
    prior = problem.prior()
    obs_map = problem.obs_map()
    qoi_map = problem.qoi_map()

Used by :mod:`linear_gaussian`, :mod:`linear_gaussian_pred`,
:mod:`nonlinear_gaussian`, and :mod:`pde.cantilever_beam_oed`. The
benchmark is a thin ``(problem, ground_truth, bkd, config)`` shell;
the problem object is independently usable (picklable, testable in
isolation, composable with other OED machinery) without needing the
benchmark shell.

**Ideal end state.** Every benchmark should be Pattern B.
``benchmark.prior()`` / ``obs_map()`` / ``qoi_map()`` should *not*
exist on the benchmark — callers should always go through
``benchmark.problem()``. This keeps the benchmark layer purely
about registration + ground truth and avoids duplicating the
problem's public API on every benchmark.

**Transition plan.** Migrating Pattern A → Pattern B in one shot
would touch dozens of tutorial files across three repos. Instead,
migrate one benchmark at a time:

1. Split its problem substrate into a proper
   ``problems/oed/<name>.py`` class.
2. Rewrite the benchmark as a thin shell that *temporarily* keeps
   Pattern A forwarders (``prior()``, ``obs_map()``, etc.) as
   aliases for ``self._problem.prior()`` / etc., so existing
   consumers keep working.
3. In a follow-up PR, migrate consumers of that benchmark to
   Pattern B and delete the forwarders.

The advection-diffusion benchmark is the first such migration
(see :mod:`advection_diffusion` and
:mod:`pyapprox_benchmarks.problems.oed.advection_diffusion`). It
currently sits at step 2: problem substrate extracted, forwarders
preserved. Lotka-Volterra is the next candidate.

The :mod:`advection_diffusion` module also registers a
``"obstructed_advection_diffusion_oed_fixed_velocity"`` benchmark
that is **Pattern B only** — it is new and has no legacy consumers,
so it ships in its final desired form and is the template every
future OED benchmark should follow.

Do not add new Pattern A benchmarks. New benchmarks should follow
Pattern B from the start, modelled on
:mod:`linear_gaussian_pred`.
"""

from pyapprox_benchmarks.instances.oed.advection_diffusion import (
    FixedVelocityObstructedAdvectionDiffusionOEDBenchmark,
    ObstructedAdvectionDiffusionOEDBenchmark,
    build_fixed_velocity_obstructed_advection_diffusion_oed_benchmark,
    build_obstructed_advection_diffusion_oed_benchmark,
)
from pyapprox_benchmarks.instances.oed.linear_gaussian import (
    LinearGaussianKLOEDBenchmark,
    build_linear_gaussian_kl_benchmark,
)
from pyapprox_benchmarks.instances.oed.linear_gaussian_pred import (
    LinearGaussianPredOEDBenchmark,
    build_linear_gaussian_pred_benchmark,
)
from pyapprox_benchmarks.instances.oed.lotka_volterra import (
    LotkaVolterraOEDBenchmark,
)
from pyapprox_benchmarks.instances.oed.nonlinear_gaussian import (
    NonLinearGaussianPredOEDBenchmark,
    build_nonlinear_gaussian_pred_benchmark,
)

__all__ = [
    "FixedVelocityObstructedAdvectionDiffusionOEDBenchmark",
    "LinearGaussianKLOEDBenchmark",
    "LinearGaussianPredOEDBenchmark",
    "LotkaVolterraOEDBenchmark",
    "NonLinearGaussianPredOEDBenchmark",
    "ObstructedAdvectionDiffusionOEDBenchmark",
    "build_fixed_velocity_obstructed_advection_diffusion_oed_benchmark",
    "build_linear_gaussian_kl_benchmark",
    "build_linear_gaussian_pred_benchmark",
    "build_nonlinear_gaussian_pred_benchmark",
    "build_obstructed_advection_diffusion_oed_benchmark",
]
