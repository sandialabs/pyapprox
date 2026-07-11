"""Periodic viscous Burgers operator-inference problem.

The periodic Burgers semi-discretization is exactly quadratic in the
state, so the Galerkin-reduced dynamics are exactly polynomial with
degree set ``(1, 2)`` and operator inference can recover the intrusive
reduced operators exactly.
"""

from __future__ import annotations

import math
from typing import Generic

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.physics import GalerkinPhysicsProtocol
from pyapprox.probability.joint.independent import IndependentJoint
from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.util.backends.protocols import Array, Backend

from pyapprox_benchmarks.benchmark import BoxDomain
from pyapprox_benchmarks.functions.pde.burgers import (
    build_periodic_burgers_physics,
    build_periodic_line_basis,
)
from pyapprox_benchmarks.problems.pde.opinf_problem import PDEOpInfProblem


class _PeriodicBurgersPhysicsFactory(Generic[Array]):
    """Rebuild Burgers physics per viscosity on the shared basis."""

    def __init__(
        self, basis: GalerkinBasisProtocol[Array], bkd: Backend[Array]
    ) -> None:
        self._basis = basis
        self._bkd = bkd

    def __call__(
        self, parameters: Array
    ) -> GalerkinPhysicsProtocol[Array]:
        viscosity = float(self._bkd.to_numpy(parameters)[0, 0])
        return build_periodic_burgers_physics(
            self._basis, viscosity, self._bkd
        )


def build_periodic_burgers_opinf_problem(
    bkd: Backend[Array],
    *,
    nx: int = 128,
    degree: int = 1,
    nominal_viscosity: float = 0.1,
    viscosity_bounds: tuple[float, float] = (0.05, 0.5),
    ic_offset: float = 0.5,
    ic_amplitude: float = 0.2,
    final_time: float = 1.0,
    deltat: float = 1e-3,
) -> PDEOpInfProblem[Array]:
    """Create the periodic Burgers operator-inference problem.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nx : int, optional
        Number of elements (= P1 dofs on the periodic mesh).
    degree : int, optional
        Lagrange polynomial degree.
    nominal_viscosity : float, optional
        Nominal kinematic viscosity.
    viscosity_bounds : tuple[float, float], optional
        Uniform prior bounds on the viscosity (the parametric Sec. 5
        extension; recovery itself is parameter-agnostic).
    ic_offset, ic_amplitude : float, optional
        Initial condition ``offset + amplitude * sin(2*pi*x)`` --
        space-periodic by construction.
    final_time, deltat : float, optional
        Default FOM time integration settings (Crank-Nicolson).  These
        are snapshot-generation choices, NOT recovery inputs: the
        exact method derives its own single-step size.

    Returns
    -------
    PDEOpInfProblem
        Problem with degree set ``(1, 2)``, no inputs, zero lift.
    """
    bounds = (0.0, 1.0)
    basis = build_periodic_line_basis(nx, bounds, bkd, degree=degree)

    coords = basis.dof_coordinates()
    initial_condition = ic_offset + ic_amplitude * bkd.sin(
        2.0 * math.pi * coords[0]
    )

    prior = IndependentJoint(
        [UniformMarginal(viscosity_bounds[0], viscosity_bounds[1], bkd)],
        bkd,
    )
    domain_bounds = bkd.array(
        [[viscosity_bounds[0], viscosity_bounds[1]]]
    )

    return PDEOpInfProblem(
        name="periodic_burgers_opinf",
        physics_factory=_PeriodicBurgersPhysicsFactory(basis, bkd),
        basis=basis,
        prior=prior,
        domain=BoxDomain(_bounds=domain_bounds, _bkd=bkd),
        time_config=TimeIntegrationConfig(
            method="crank_nicolson",
            init_time=0.0,
            final_time=final_time,
            deltat=deltat,
        ),
        initial_condition=initial_condition,
        nominal_parameters=bkd.array([[nominal_viscosity]]),
        degree_set=(1, 2),
        ninputs=0,
        lift_vector=bkd.zeros((basis.ndofs(),)),
        bkd=bkd,
        description=(
            "Viscous Burgers with periodic boundary conditions; the "
            "Galerkin semi-discretization is exactly quadratic in the "
            "state (no boundary lift, no inputs)."
        ),
        reference=(
            "Rosenberger, Sanderse, Stabile (2025), Exact operator "
            "inference with minimal data."
        ),
    )
