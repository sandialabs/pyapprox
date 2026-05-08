"""Coupled springs 2-mass ODE forward UQ problem."""

from __future__ import annotations

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.ode import ODETimeConfig
from pyapprox_benchmarks.functions.ode.coupled_springs import (
    CoupledSpringsResidual,
)
from pyapprox_benchmarks.problems.ode import ODEForwardUQProblem
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


def build_coupled_springs_2mass(
    bkd: Backend[Array],
) -> ODEForwardUQProblem[Array]:
    """Create the two-mass coupled springs forward UQ problem.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    ODEForwardUQProblem
        The coupled springs forward UQ problem.

    Notes
    -----
    Parameters (12): [m_1, m_2, k_1, k_2, L_1, L_2, b_1, b_2,
                       x_1_0, y_1_0, x_2_0, y_2_0].
    States (4): [x_1, y_1, x_2, y_2] (positions and velocities).
    """
    nstates = 4

    residual = CoupledSpringsResidual(bkd)

    prior_ranges = [
        (0.9, 1.1),   # m_1
        (1.4, 1.6),   # m_2
        (7.0, 9.0),   # k_1
        (39.0, 41.0), # k_2
        (0.4, 0.6),   # L_1
        (0.9, 1.1),   # L_2
        (0.7, 0.9),   # b_1
        (0.4, 0.6),   # b_2
        (0.4, 0.6),   # x_1_0
        (-0.1, 0.1),  # y_1_0
        (2.2, 2.3),   # x_2_0
        (-0.1, 0.1),  # y_2_0
    ]

    prior = IndependentJoint(
        [UniformMarginal(lo, hi, bkd) for lo, hi in prior_ranges],
        bkd,
    )

    bounds = bkd.array(prior_ranges)

    nominal_parameters = bkd.reshape(
        bkd.array([(lo + hi) / 2 for lo, hi in prior_ranges]),
        (-1, 1),
    )

    # Initial condition from nominal parameters (last 4 are ICs)
    initial_condition = nominal_parameters[8:, :]

    return ODEForwardUQProblem(
        name="coupled_springs_2mass",
        residual=residual,
        prior=prior,
        domain=BoxDomain(_bounds=bounds, _bkd=bkd),
        time_config=ODETimeConfig(
            init_time=0.0,
            final_time=10.0,
            deltat=0.1,
        ),
        nstates=nstates,
        initial_condition=initial_condition,
        nominal_parameters=nominal_parameters,
        bkd=bkd,
        description="Two-mass coupled springs with friction",
        reference="Classical mechanics",
        estimated_evaluation_cost=1.0e-03,
    )
