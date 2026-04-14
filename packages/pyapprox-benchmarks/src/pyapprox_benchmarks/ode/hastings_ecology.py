"""Hastings-Powell three-species ecology ODE forward UQ problem."""

from __future__ import annotations

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.ode import ODETimeConfig
from pyapprox_benchmarks.functions.ode.hastings_ecology import (
    HastingsEcologyResidual,
)
from pyapprox_benchmarks.problems.ode import ODEForwardUQProblem
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


def build_hastings_ecology_3species(
    bkd: Backend[Array],
) -> ODEForwardUQProblem[Array]:
    """Create the Hastings-Powell three-species ecology forward UQ problem.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    ODEForwardUQProblem
        The Hastings ecology forward UQ problem.

    Notes
    -----
    Parameters (9): [a_1, b_1, a_2, b_2, d_1, d_2, y_1_0, y_2_0, y_3_0].
    States (3): [Y_1, Y_2, Y_3] (population densities).
    Prior: U[0.95*nominal, 1.05*nominal] per parameter.

    References
    ----------
    Hastings & Powell, "Chaos in a Three-Species Food Chain", Ecology (1991).
    """
    nstates = 3

    residual = HastingsEcologyResidual(bkd)

    nominal_values = [5.0, 3.0, 0.1, 2.0, 0.4, 0.01, 0.75, 0.15, 10.0]

    prior_ranges = [
        (0.95 * v, 1.05 * v) for v in nominal_values
    ]

    prior = IndependentJoint(
        [UniformMarginal(lo, hi, bkd) for lo, hi in prior_ranges],
        bkd,
    )

    bounds = bkd.array(prior_ranges)
    nominal_parameters = bkd.reshape(bkd.array(nominal_values), (-1, 1))

    # Initial condition from nominal parameters (last 3 are ICs)
    initial_condition = nominal_parameters[6:, :]

    return ODEForwardUQProblem(
        name="hastings_ecology_3species",
        residual=residual,
        prior=prior,
        domain=BoxDomain(_bounds=bounds, _bkd=bkd),
        time_config=ODETimeConfig(
            init_time=0.0,
            final_time=100.0,
            deltat=2.5,
        ),
        nstates=nstates,
        initial_condition=initial_condition,
        nominal_parameters=nominal_parameters,
        bkd=bkd,
        description="Hastings-Powell three-species food chain",
        reference="Hastings & Powell (1991)",
        estimated_evaluation_cost=4.1e-04,
    )
