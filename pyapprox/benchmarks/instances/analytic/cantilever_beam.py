"""Analytical cantilever beam benchmark instance.

Provides the analytical 1D cantilever beam model as a benchmark with
Beta marginals for the Young's moduli E1 (skin) and E2 (core).
"""

from pyapprox.benchmarks.benchmark import BenchmarkWithPrior, BoxDomain
from pyapprox.benchmarks.functions.algebraic.cantilever_beam import (
    CantileverBeam1DAnalytical,
)
from pyapprox.benchmarks.ground_truth import SensitivityGroundTruth
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.beta import BetaMarginal
from pyapprox.util.backends.protocols import Array, Backend

# TODO: Rename this file cantilever_beam_1d to reflect its contents

def cantilever_beam_1d_analytical(
    bkd: Backend[Array],
    length: float = 100.0,
    height: float = 30.0,
    skin_thickness: float = 5.0,
    q0: float = 10.0,
) -> BenchmarkWithPrior[Array, SensitivityGroundTruth[Array]]:
    """Create an analytical 1D cantilever beam benchmark.

    The model computes [tip_deflection, max_curvature] from (E1, E2)
    using closed-form formulas for a composite beam under linearly
    increasing load q(x) = q0*x/L.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    length : float
        Beam length.
    height : float
        Cross-section height.
    skin_thickness : float
        Thickness of each skin layer.
    q0 : float
        Load magnitude.

    Returns
    -------
    BenchmarkWithPrior
        Benchmark instance.
    """
    func = CantileverBeam1DAnalytical(
        length=length,
        height=height,
        skin_thickness=skin_thickness,
        q0=q0,
        bkd=bkd,
    )

    # Beta(2,5) marginals on [18000, 22000] for E1, [4500, 5500] for E2
    prior = IndependentJoint(
        [
            BetaMarginal(2.0, 5.0, bkd, lb=18000.0, ub=22000.0),
            BetaMarginal(2.0, 5.0, bkd, lb=4500.0, ub=5500.0),
        ],
        bkd,
    )

    bounds = bkd.array([[18000.0, 22000.0], [4500.0, 5500.0]])
    domain = BoxDomain(_bounds=bounds, _bkd=bkd)

    # TODO: This should not use SensitivityGroundTruth(). It has no
    # known SA ground truth, TODO Implement, StatisticsGroundTruth and
    # compute reference values for this class using Sympy?
    return BenchmarkWithPrior(
        _name="cantilever_beam_1d_analytical",
        _function=func,
        _domain=domain,
        _ground_truth=SensitivityGroundTruth(),
        _prior=prior,
        _description=(
            "Analytical 1D cantilever beam: (E1, E2) -> [tip deflection, max curvature]"
        ),
    )


# TODO: Analytic is not a benchmark category. Change to
# "statistics" once statistics ground truth is implemented
# TODO: can we write runtime checks that category matches the
# Benchmark class type, or should we remove category and just
# determine it from Benchmark class. We should allow a benchmark
# to have multiple categories, e.g. ODEBenchmark and
# StatisticsBenchmark, that will be picked up by
# BenchmarkRegistry.list_category("ode") and
# BenchmarkRegistry.list_category("statistics") will both return
# a list containing an ODE benchmark that returns both
# ODEGroundTruth and StatisticsGroundTruth.
# TODO: BenchmarkProtocols seem to be trying to achieve a similar
# but not quite the same thing as category. Protocols seem to be
# useful for a level below benchmark, e.g. not necessarily having
# a ground_truth but allowing the package to provide configurable
# reusable problem setups, e.g. a forward uq setup that has a
# prior. Right now we have function instances and benchmark
# instances. A function instance can be configurable and a
# benchmark should be fixed and has a function instance. Do we
# need an intermediate level, e.g. a benchmark problem that does
# not have a ground truth (should such be configurable or not) or
# is this intermediate problem level unnecessary. Currently some
# benchmarks like this one use empty ground truths e.g.
# SensitivityGroundTruth() to make this a benchmark (code wise
# but not conceptually) we should not do this. This is really
# just a problem instance with a function instance and a prior.
# A user may want to know if a benchmark has a prior. Is there a
# unified way of combining all these goals currently attempted
# with these three mechanisms: benchmark category, ground truth
# and protocols. Or a cleaner split.
# Should Benchmarks just be compositions of different components,
# e.g. function instance, problem setup (prior, obs model for
# InverseProblemBenchmark), and ground truth (true posterior pdf,
# posterior mean, var etc). Use of string categories is not very
# extensible, new benchmark types are not easily added, however
# searching over properties of benchmarks seems easier. I.e.
# what ground truths does it have, what problem components does
# it have (I think this is currently handled by protocols).
# Right now benchmarks like RosenbrockBenchmark or
# MultiOutputEnsembleBenchmark are really just problem instances
# (which can be configured) perhaps we should rename as such.
# The former has a ground truth but perhaps that should be
# separated from problem instance. The latter does not even have
# a ground truth. Once we decide on better convention update
# CONVENTIONS.md in this module.
# TODO: There are many examples of combinatorial explosion of
# benchmarks (which have ground truth computations for
# configurable versions of benchmarks. Right now we only
# instantiate a few in the registry, e.g. genz_oscillatory_2d,
# sobol_g_4d, rosenbrock_10d. We have a ConfigurableBenchmark
# class for these special cases, they still have ground truth
# but avoid code bloat.
# Right definition of what is an instance is unclear. We need
# clear types of instances: configurable function/model instances,
# forward models based on algebraic ode, pde etc. Problem
# instances, e.g. function instance with extra information to
# solve outerloop problems (prior, obs model, prediction model
# for goal oriented inference), and benchmark instances that add
# ground truths. Or some better split. We should try to provide
# estimated cost for function/model instances. How do we handle
# cost for configurable functions/models, e.g. those that use
# different timesteppers. Should we only provide costs when they
# are used in a benchmark instance (which has fixed config).
# Should we allow a benchmark to define its own function/model
# without making it a function instance reusable elsewhere. Pros:
# allows benchmarks where model will likely never be reused. Cons:
# function is not reusable without setting up entire benchmark.
# Look at benchmark.tests.harness and test files like
# test_harness_forward_uq of how benchmarks can be used for
# testing and benchmarking for research development. Benchmarks
# should be structured such that we can obtain all benchmarks
# that have the properties required by algorithm types to conduct
# analysis without knowing differences between benchmarks under
# the hood.
# TODO: Should benchmarks be defined here or do we allow some to
# be defined in submodule where most code lives or makes most
# sense, e.g. pde based benchmark for expdesign lives in
# expdesign module (not pde), e.g. there is a benchmarks
# submodule in expdesign. Decide and document rule, and place in
# benchmarks.CONVENTIONS.md. Once we decide we must ensure we
# apply rule consistently, e.g. move bayesian benchmarks to
# inverse, opt to optimization etc or move all to this module
# (or follow the split rule if we make one).
@BenchmarkRegistry.register(
    "cantilever_beam_1d_analytical",
    category="analytic",
    description=(
        "Analytical 1D cantilever beam with closed-form tip deflection "
        "and max curvature"
    ),
)
def _cantilever_beam_1d_analytical_factory(
    bkd: Backend[Array],
) -> BenchmarkWithPrior[Array, SensitivityGroundTruth[Array]]:
    return cantilever_beam_1d_analytical(bkd)
