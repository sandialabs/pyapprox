"""Hastings-Powell three-species ecology ODE benchmark instance."""

from pyapprox.benchmarks.benchmark import BoxDomain
from pyapprox.benchmarks.functions.ode import ODEBenchmark, ODETimeConfig
from pyapprox.benchmarks.ground_truth import ODEGroundTruth
from pyapprox.benchmarks.instances.ode.lotka_volterra import (
    ODEBenchmarkWrapper,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.benchmarks.functions.ode.hastings_ecology import (
    HastingsEcologyResidual,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


def hastings_ecology_3species(
    bkd: Backend[Array],
) -> ODEBenchmarkWrapper:
    """Create the Hastings-Powell three-species ecology benchmark.

    Three-species food chain model with saturating functional response.
    Parameters vary within +-5% of nominal values.

    The system is governed by:
        dY_1/dT = Y_1*(1 - Y_1) - a_1*Y_1*Y_2/(1 + b_1*Y_1)
        dY_2/dT = a_1*Y_1*Y_2/(1 + b_1*Y_1) - a_2*Y_2*Y_3/(1 + b_2*Y_2) - d_1*Y_2
        dY_3/dT = a_2*Y_2*Y_3/(1 + b_2*Y_2) - d_2*Y_3

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    ODEBenchmarkWrapper
        The Hastings ecology benchmark instance.

    Notes
    -----
    Parameters (9 total):
        [a_1, b_1, a_2, b_2, d_1, d_2, y_1_0, y_2_0, y_3_0]

    States (3 total):
        [Y_1, Y_2, Y_3] (population densities)

    References
    ----------
    Hastings, Alan, and Thomas Powell. "Chaos in a Three-Species Food Chain."
    Ecology 72, no. 3 (1991): 896-903.
    """
    nstates = 3
    nparams = 9

    # Create residual
    residual = HastingsEcologyResidual(bkd)

    # Nominal values from legacy code - shape (nparams, 1)
    nominal_values = bkd.reshape(bkd.array(
        [5.0, 3.0, 0.1, 2.0, 0.4, 0.01, 0.75, 0.15, 10.0]
    ), (-1, 1))

    # Prior: U[0.95*nominal, 1.05*nominal] for each parameter
    prior_ranges = [
        (0.95 * float(val), 1.05 * float(val)) for val in nominal_values[:, 0]
    ]

    prior = IndependentJoint(
        [UniformMarginal(lo, hi, bkd) for lo, hi in prior_ranges],
        bkd,
    )

    # Domain bounds
    bounds = bkd.array(prior_ranges)

    # Initial condition from parameters (last 3 params are initial conditions)
    # Shape: (nstates, 1)
    initial_condition = nominal_values[6:, :]

    inner = ODEBenchmark(
        _name="hastings_ecology_3species",
        _residual=residual,
        _domain=BoxDomain(_bounds=bounds, _bkd=bkd),
        _ground_truth=ODEGroundTruth(
            nstates=nstates,
            nparams=nparams,
            initial_condition=initial_condition,
            nominal_parameters=nominal_values,
            init_time=0.0,
            final_time=100.0,
            deltat=2.5,
        ),
        _time_config=ODETimeConfig(
            init_time=0.0,
            final_time=100.0,
            deltat=2.5,
        ),
        _prior=prior,
        _description="Hastings-Powell three-species food chain",
        _reference="Hastings & Powell (1991)",
    )

    return ODEBenchmarkWrapper(inner, estimated_cost=4.1e-04)


@BenchmarkRegistry.register(
    "hastings_ecology_3species",
    category="ode",
    description="Hastings-Powell three-species ecology ODE system",
)
def _hastings_ecology_3species_factory(
    bkd: Backend[Array],
) -> ODEBenchmarkWrapper:
    return hastings_ecology_3species(bkd)
