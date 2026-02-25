"""Coupled springs 2-mass ODE benchmark instance."""

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.benchmarks.benchmark import BoxDomain
from pyapprox.benchmarks.ground_truth import ODEGroundTruth
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.benchmarks.functions.ode import ODEBenchmark, ODETimeConfig
from pyapprox.pde.time.benchmarks.coupled_springs import (
    CoupledSpringsResidual,
)
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.benchmarks.instances.ode.lotka_volterra import (
    ODEBenchmarkWrapper,
)


def coupled_springs_2mass(
    bkd: Backend[Array],
) -> ODEBenchmarkWrapper:
    """Create the two-mass coupled springs benchmark.

    Two masses connected by springs with friction. The left end of the
    left spring is fixed.

    The system is governed by:
        x'_1 = y_1
        y'_1 = (-b_1*y_1 - k_1*(x_1 - L_1) + k_2*(x_2 - x_1 - L_2)) / m_1
        x'_2 = y_2
        y'_2 = (-b_2*y_2 - k_2*(x_2 - x_1 - L_2)) / m_2

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    ODEBenchmarkWrapper
        The coupled springs benchmark instance.

    Notes
    -----
    Parameters (12 total):
        [m_1, m_2, k_1, k_2, L_1, L_2, b_1, b_2, x_1_0, y_1_0, x_2_0, y_2_0]

    States (4 total):
        [x_1, y_1, x_2, y_2] (positions and velocities)
    """
    nstates = 4
    nparams = 12

    # Create residual
    residual = CoupledSpringsResidual(bkd)

    # Prior ranges from legacy code (specific per-parameter bounds)
    # [min, max] pairs for each parameter
    prior_ranges = [
        (0.9, 1.1),    # m_1
        (1.4, 1.6),    # m_2
        (7.0, 9.0),    # k_1
        (39.0, 41.0),  # k_2
        (0.4, 0.6),    # L_1
        (0.9, 1.1),    # L_2
        (0.7, 0.9),    # b_1
        (0.4, 0.6),    # b_2
        (0.4, 0.6),    # x_1_0
        (-0.1, 0.1),   # y_1_0
        (2.2, 2.3),    # x_2_0
        (-0.1, 0.1),   # y_2_0
    ]

    # Build prior from ranges
    prior = IndependentJoint(
        [
            UniformMarginal(lo, hi, bkd)
            for lo, hi in prior_ranges
        ],
        bkd,
    )

    # Domain bounds
    bounds = bkd.array(prior_ranges)

    # Nominal parameters (center of prior) - shape (nparams, 1)
    nominal_parameters = bkd.array([(lo + hi) / 2 for lo, hi in prior_ranges]).reshape(-1, 1)

    # Initial condition from parameters (last 4 params are initial conditions)
    # Shape: (nstates, 1)
    initial_condition = nominal_parameters[8:, :]

    inner = ODEBenchmark(
        _name="coupled_springs_2mass",
        _residual=residual,
        _domain=BoxDomain(_bounds=bounds, _bkd=bkd),
        _ground_truth=ODEGroundTruth(
            nstates=nstates,
            nparams=nparams,
            initial_condition=initial_condition,
            nominal_parameters=nominal_parameters,
            init_time=0.0,
            final_time=10.0,
            deltat=0.1,
        ),
        _time_config=ODETimeConfig(
            init_time=0.0,
            final_time=10.0,
            deltat=0.1,
        ),
        _prior=prior,
        _description="Two-mass coupled springs with friction",
        _reference="Classical mechanics",
    )

    return ODEBenchmarkWrapper(inner, estimated_cost=1.0e-03)


@BenchmarkRegistry.register(
    "coupled_springs_2mass",
    category="ode",
    description="Two-mass coupled springs ODE system",
)
def _coupled_springs_2mass_factory(
    bkd: Backend[Array],
) -> ODEBenchmarkWrapper:
    return coupled_springs_2mass(bkd)
