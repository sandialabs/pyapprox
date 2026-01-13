"""Fixed ODE benchmark instances.

These are pre-configured ODE benchmark instances with standard parameters
and known ground truth values. They wrap existing residuals from
pyapprox.typing.pde.time.benchmarks.
"""

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.benchmarks.benchmark import BoxDomain
from pyapprox.typing.benchmarks.ground_truth import ODEGroundTruth
from pyapprox.typing.benchmarks.registry import BenchmarkRegistry
from pyapprox.typing.benchmarks.functions.ode import ODEBenchmark, ODETimeConfig
from pyapprox.typing.pde.time.benchmarks.lotka_volterra import (
    LotkaVolterraResidual,
)
from pyapprox.typing.pde.time.benchmarks.coupled_springs import (
    CoupledSpringsResidual,
)
from pyapprox.typing.pde.time.benchmarks.hastings_ecology import (
    HastingsEcologyResidual,
)
from pyapprox.typing.pde.time.benchmarks.chemical_reaction import (
    ChemicalReactionResidual,
)
from pyapprox.typing.probability.univariate.uniform import UniformMarginal
from pyapprox.typing.probability.joint.independent import IndependentJoint


def lotka_volterra_3species(
    bkd: Backend[Array],
) -> ODEBenchmark[Array, ODEGroundTruth[Array]]:
    """Create the 3-species competitive Lotka-Volterra benchmark.

    Standard 3-species competitive Lotka-Volterra benchmark with uniform
    prior U[0.3, 0.7]^12 for all parameters.

    The system is governed by:
        dx_i/dt = r_i * x_i * (1 - sum_j a_ij * x_j), i = 1, 2, 3

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.

    Returns
    -------
    ODEBenchmark[Array, ODEGroundTruth]
        The Lotka-Volterra benchmark instance.

    Notes
    -----
    Parameters are organized as:
        p = [r_0, r_1, r_2, a_00, a_01, ..., a_22]

    Total: 12 parameters (3 growth rates + 9 competition coefficients).

    References
    ----------
    TODO: Find specific paper reference for this configuration.
    Original Lotka-Volterra: Lotka (1925), Volterra (1926).
    """
    nspecies = 3
    nparams = nspecies + nspecies * nspecies  # 3 + 9 = 12

    # Create residual
    residual = LotkaVolterraResidual(nspecies, bkd)

    # Prior: U[0.3, 0.7]^12 for all parameters
    # Note: scipy.stats.uniform(0.3, 0.4) means U[0.3, 0.3+0.4] = U[0.3, 0.7]
    prior = IndependentJoint(
        [UniformMarginal(0.3, 0.7, bkd) for _ in range(nparams)],
        bkd,
    )

    # Domain bounds
    bounds = bkd.array([[0.3, 0.7]] * nparams)

    # Initial condition (from legacy code) - shape (nstates, 1)
    initial_condition = bkd.array([[0.3], [0.4], [0.3]])

    # Nominal parameters (center of prior) - shape (nparams, 1)
    nominal_parameters = bkd.full((nparams, 1), 0.5)

    return ODEBenchmark(
        _name="lotka_volterra_3species",
        _residual=residual,
        _domain=BoxDomain(_bounds=bounds, _bkd=bkd),
        _ground_truth=ODEGroundTruth(
            nstates=nspecies,
            nparams=nparams,
            initial_condition=initial_condition,
            nominal_parameters=nominal_parameters,
            init_time=0.0,
            final_time=10.0,
            deltat=1.0,
        ),
        _time_config=ODETimeConfig(
            init_time=0.0,
            final_time=10.0,
            deltat=1.0,
        ),
        _prior=prior,
        _description="3-species competitive Lotka-Volterra",
        _reference="Lotka (1925), Volterra (1926)",  # TODO: find specific paper
    )


@BenchmarkRegistry.register(
    "lotka_volterra_3species",
    category="ode",
    description="3-species competitive Lotka-Volterra ODE system",
)
def _lotka_volterra_3species_factory(
    bkd: Backend[Array],
) -> ODEBenchmark[Array, ODEGroundTruth[Array]]:
    return lotka_volterra_3species(bkd)


def coupled_springs_2mass(
    bkd: Backend[Array],
) -> ODEBenchmark[Array, ODEGroundTruth[Array]]:
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
    ODEBenchmark[Array, ODEGroundTruth]
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

    return ODEBenchmark(
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


@BenchmarkRegistry.register(
    "coupled_springs_2mass",
    category="ode",
    description="Two-mass coupled springs ODE system",
)
def _coupled_springs_2mass_factory(
    bkd: Backend[Array],
) -> ODEBenchmark[Array, ODEGroundTruth[Array]]:
    return coupled_springs_2mass(bkd)


def hastings_ecology_3species(
    bkd: Backend[Array],
) -> ODEBenchmark[Array, ODEGroundTruth[Array]]:
    """Create the Hastings-Powell three-species ecology benchmark.

    Three-species food chain model with saturating functional response.
    Parameters vary within ±5% of nominal values.

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
    ODEBenchmark[Array, ODEGroundTruth]
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
    nominal_values = bkd.array([5.0, 3.0, 0.1, 2.0, 0.4, 0.01, 0.75, 0.15, 10.0]).reshape(-1, 1)

    # Prior: U[0.95*nominal, 1.05*nominal] for each parameter
    prior_ranges = [
        (0.95 * float(val), 1.05 * float(val)) for val in nominal_values[:, 0]
    ]

    prior = IndependentJoint(
        [
            UniformMarginal(lo, hi, bkd)
            for lo, hi in prior_ranges
        ],
        bkd,
    )

    # Domain bounds
    bounds = bkd.array(prior_ranges)

    # Initial condition from parameters (last 3 params are initial conditions)
    # Shape: (nstates, 1)
    initial_condition = nominal_values[6:, :]

    return ODEBenchmark(
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


@BenchmarkRegistry.register(
    "hastings_ecology_3species",
    category="ode",
    description="Hastings-Powell three-species ecology ODE system",
)
def _hastings_ecology_3species_factory(
    bkd: Backend[Array],
) -> ODEBenchmark[Array, ODEGroundTruth[Array]]:
    return hastings_ecology_3species(bkd)


def chemical_reaction_surface(
    bkd: Backend[Array],
) -> ODEBenchmark[Array, ODEGroundTruth[Array]]:
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
    ODEBenchmark[Array, ODEGroundTruth]
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
    nominal_values = bkd.array([1.6, 20.75, 0.04, 1.0, 0.36, 0.016]).reshape(-1, 1)

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
        [
            UniformMarginal(lo, hi, bkd)
            for lo, hi in prior_ranges
        ],
        bkd,
    )

    # Domain bounds
    bounds = bkd.array(prior_ranges)

    # Initial condition is fixed at zeros (empty surface) - shape (nstates, 1)
    initial_condition = bkd.array([[0.0], [0.0], [0.0]])

    return ODEBenchmark(
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


@BenchmarkRegistry.register(
    "chemical_reaction_surface",
    category="ode",
    description="Chemical reaction surface adsorption ODE system",
)
def _chemical_reaction_surface_factory(
    bkd: Backend[Array],
) -> ODEBenchmark[Array, ODEGroundTruth[Array]]:
    return chemical_reaction_surface(bkd)
