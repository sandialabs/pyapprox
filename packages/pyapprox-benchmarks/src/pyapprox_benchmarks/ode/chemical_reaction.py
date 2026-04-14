"""Chemical reaction surface adsorption ODE forward UQ problem."""

from __future__ import annotations

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.ode import ODETimeConfig
from pyapprox_benchmarks.functions.ode.chemical_reaction import (
    ChemicalReactionResidual,
)
from pyapprox_benchmarks.problems.ode import ODEForwardUQProblem
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


def build_chemical_reaction_surface(
    bkd: Backend[Array],
) -> ODEForwardUQProblem[Array]:
    """Create the chemical reaction surface adsorption forward UQ problem.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    ODEForwardUQProblem
        The chemical reaction forward UQ problem.

    Notes
    -----
    Parameters (6): [a, b, c, d, e, f].
    States (3): [u, v, w] (monomer, dimer, inert surface coverages).
    Initial condition: [0, 0, 0] (empty surface).

    References
    ----------
    Vigil et al., Phys. Rev. E. (1996).
    """
    nstates = 3

    residual = ChemicalReactionResidual(bkd)

    nominal_values = [1.6, 20.75, 0.04, 1.0, 0.36, 0.016]

    prior_ranges = [
        (0.0, 4.0),                            # a
        (5.0, 35.0),                            # b
        (0.9 * nominal_values[2], 1.1 * nominal_values[2]),  # c
        (0.9 * nominal_values[3], 1.1 * nominal_values[3]),  # d
        (0.9 * nominal_values[4], 1.1 * nominal_values[4]),  # e
        (0.9 * nominal_values[5], 1.1 * nominal_values[5]),  # f
    ]

    prior = IndependentJoint(
        [UniformMarginal(lo, hi, bkd) for lo, hi in prior_ranges],
        bkd,
    )

    bounds = bkd.array(prior_ranges)
    nominal_parameters = bkd.reshape(bkd.array(nominal_values), (-1, 1))
    initial_condition = bkd.array([[0.0], [0.0], [0.0]])

    return ODEForwardUQProblem(
        name="chemical_reaction_surface",
        residual=residual,
        prior=prior,
        domain=BoxDomain(_bounds=bounds, _bkd=bkd),
        time_config=ODETimeConfig(
            init_time=0.0,
            final_time=100.0,
            deltat=0.1,
        ),
        nstates=nstates,
        initial_condition=initial_condition,
        nominal_parameters=nominal_parameters,
        bkd=bkd,
        description="Chemical reaction surface adsorption",
        reference="Vigil et al. (1996)",
        estimated_evaluation_cost=9.0e-03,
    )
