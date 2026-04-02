"""Chemical reaction surface adsorption ODE benchmark instance."""

from pyapprox.benchmarks.benchmark import BoxDomain
from pyapprox.benchmarks.functions.ode import ODEBenchmark, ODETimeConfig
from pyapprox.benchmarks.ground_truth import ODEGroundTruth
from pyapprox.benchmarks.instances.ode.lotka_volterra import (
    ODEBenchmarkWrapper,
)
from pyapprox.benchmarks.registry import BenchmarkRegistry
from pyapprox.benchmarks.functions.ode.chemical_reaction import (
    ChemicalReactionResidual,
)
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


def chemical_reaction_surface(
    bkd: Backend[Array],
) -> ODEBenchmarkWrapper:
    """Create the chemical reaction surface adsorption benchmark.

    Surface adsorption model describing species adsorbing onto a surface
    from the gas phase.

    The system is governed by:
        du/dt = a*z - c*u - 4*d*u*v
        dv/dt = 2*b*z^2 - 4*d*u*v
        dw/dt = e*z - f*w

    where z = 1 - u - v - w (fraction of unoccupied surface).

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    ODEBenchmarkWrapper
        The chemical reaction benchmark instance.

    Notes
    -----
    Parameters (6 total):
        [a, b, c, d, e, f]

    States (3 total):
        [u, v, w] (monomer, dimer, inert surface coverages)

    Initial condition is fixed at [0, 0, 0] (empty surface).

    References
    ----------
    Vigil et al., Phys. Rev. E., 1996.
    Makeev et al., J. Chem. Phys., 2002.
    """
    nstates = 3
    nparams = 6

    # Create residual
    residual = ChemicalReactionResidual(bkd)

    # Nominal values from legacy code - shape (nparams, 1)
    nominal_values = bkd.reshape(
        bkd.array([1.6, 20.75, 0.04, 1.0, 0.36, 0.016]),
        (-1, 1),
    )

    # Prior ranges from legacy code:
    # a: U[0, 4], b: U[5, 35], c,d,e,f: U[0.9*nominal, 1.1*nominal]
    prior_ranges = [
        (0.0, 4.0),  # a
        (5.0, 35.0),  # b
        (0.9 * float(nominal_values[2, 0]), 1.1 * float(nominal_values[2, 0])),  # c
        (0.9 * float(nominal_values[3, 0]), 1.1 * float(nominal_values[3, 0])),  # d
        (0.9 * float(nominal_values[4, 0]), 1.1 * float(nominal_values[4, 0])),  # e
        (0.9 * float(nominal_values[5, 0]), 1.1 * float(nominal_values[5, 0])),  # f
    ]

    prior = IndependentJoint(
        [UniformMarginal(lo, hi, bkd) for lo, hi in prior_ranges],
        bkd,
    )

    # Domain bounds
    bounds = bkd.array(prior_ranges)

    # Initial condition is fixed at zeros (empty surface) - shape (nstates, 1)
    initial_condition = bkd.array([[0.0], [0.0], [0.0]])

    inner = ODEBenchmark(
        _name="chemical_reaction_surface",
        _residual=residual,
        _domain=BoxDomain(_bounds=bounds, _bkd=bkd),
        _ground_truth=ODEGroundTruth(
            nstates=nstates,
            nparams=nparams,
            initial_condition=initial_condition,
            nominal_parameters=nominal_values,
            init_time=0.0,
            final_time=100.0,
            deltat=0.1,
        ),
        _time_config=ODETimeConfig(
            init_time=0.0,
            final_time=100.0,
            deltat=0.1,
        ),
        _prior=prior,
        _description="Chemical reaction surface adsorption",
        _reference="Vigil et al. (1996)",
    )

    return ODEBenchmarkWrapper(inner, estimated_cost=9.0e-03)

@BenchmarkRegistry.register(
    "chemical_reaction_surface",
    category="ode",
    description="Chemical reaction surface adsorption ODE system",
)
def _chemical_reaction_surface_factory(
    bkd: Backend[Array],
) -> ODEBenchmarkWrapper:
    return chemical_reaction_surface(bkd)
