"""Lotka-Volterra 3-species ODE forward UQ problem."""

from __future__ import annotations

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.ode import ODETimeConfig
from pyapprox_benchmarks.functions.ode.lotka_volterra import (
    LotkaVolterraResidual,
)
from pyapprox_benchmarks.problems.ode import ODEForwardUQProblem
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend


def build_lotka_volterra_3species(
    bkd: Backend[Array],
    final_time: float = 10.0,
    deltat: float = 1.0,
) -> ODEForwardUQProblem[Array]:
    """Create the 3-species competitive Lotka-Volterra forward UQ problem.

    Parameters
    ----------
    bkd : Backend[Array]
        Backend for array operations.
    final_time : float, optional
        Final simulation time. Default 10.0.
    deltat : float, optional
        Time step. Default 1.0.

    Returns
    -------
    ODEForwardUQProblem
        The Lotka-Volterra forward UQ problem.

    Notes
    -----
    Parameters (12): [r_0, r_1, r_2, a_00, a_01, ..., a_22].
    Prior: U[0.3, 0.7]^12.
    """
    nspecies = 3
    nparams = nspecies + nspecies * nspecies  # 12

    residual = LotkaVolterraResidual(nspecies, bkd)

    prior = IndependentJoint(
        [UniformMarginal(0.3, 0.7, bkd) for _ in range(nparams)],
        bkd,
    )

    bounds = bkd.array([[0.3, 0.7]] * nparams)
    initial_condition = bkd.array([[0.3], [0.4], [0.3]])
    nominal_parameters = bkd.full((nparams, 1), 0.5)

    return ODEForwardUQProblem(
        name="lotka_volterra_3species",
        residual=residual,
        prior=prior,
        domain=BoxDomain(_bounds=bounds, _bkd=bkd),
        time_config=ODETimeConfig(
            init_time=0.0,
            final_time=final_time,
            deltat=deltat,
        ),
        nstates=nspecies,
        initial_condition=initial_condition,
        nominal_parameters=nominal_parameters,
        bkd=bkd,
        description="3-species competitive Lotka-Volterra",
        reference="Lotka (1925), Volterra (1926)",
        estimated_evaluation_cost=3.7e-05,
    )
