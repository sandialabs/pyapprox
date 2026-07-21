"""Homogeneous Chafee-Infante operator-inference problem.

The Chafee-Infante semi-discretization ``du/dt = gamma*u_xx +
lambda*u - u**3`` is exactly cubic in the state, so the
Galerkin-reduced dynamics are exactly polynomial with degree set
``(1, 3)``.

This is step (i) of the boundary-input escalation ladder documented
in ``pyapprox_benchmarks.functions.pde.chafee_infante``: homogeneous
(zero Dirichlet left, natural Neumann right, no input, zero lift).
The lumped-mass-input and consistent-mass-input variants are separate
follow-on problems of increasing complexity.
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
from pyapprox_benchmarks.functions.pde.chafee_infante import (
    build_chafee_infante_physics,
    build_line_basis,
)
from pyapprox_benchmarks.problems.pde.opinf_problem import PDEOpInfProblem


class _ChafeeInfantePhysicsFactory(Generic[Array]):
    """Rebuild Chafee-Infante physics per (gamma, lambda) on the shared basis."""

    def __init__(
        self,
        basis: GalerkinBasisProtocol[Array],
        bkd: Backend[Array],
        bc_kind: str,
    ) -> None:
        self._basis = basis
        self._bkd = bkd
        self._bc_kind = bc_kind

    def __call__(
        self, parameters: Array
    ) -> GalerkinPhysicsProtocol[Array]:
        parameters_np = self._bkd.to_numpy(parameters)
        diffusivity = float(parameters_np[0, 0])
        bifurcation = float(parameters_np[1, 0])
        return build_chafee_infante_physics(
            self._basis,
            diffusivity,
            bifurcation,
            self._bkd,
            bc_kind=self._bc_kind,
        )


def build_chafee_infante_opinf_problem(
    bkd: Backend[Array],
    *,
    nx: int = 128,
    degree: int = 1,
    nominal_diffusivity: float = 1.0,
    nominal_bifurcation: float = 1.0,
    diffusivity_bounds: tuple[float, float] = (0.5, 2.0),
    bifurcation_bounds: tuple[float, float] = (0.5, 5.0),
    bc_kind: str = "dirichlet_neumann",
    ic_amplitude: float = 0.5,
    final_time: float = 1.0,
    deltat: float = 1e-3,
) -> PDEOpInfProblem[Array]:
    """Create the homogeneous Chafee-Infante operator-inference problem.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nx : int, optional
        Number of elements (``nx + 1`` P1 dofs; the Dirichlet dof is
        constrained).
    degree : int, optional
        Lagrange polynomial degree.
    nominal_diffusivity : float, optional
        Nominal diffusion coefficient gamma.
    nominal_bifurcation : float, optional
        Nominal bifurcation parameter lambda.  Keep lambda <= ~5: the
        upper prior bound is chosen so operator-recovery probes stay
        well conditioned (larger lambda degrades cond(P) -- correct
        paper Sec. 4.2 behavior, not a bug).
    diffusivity_bounds, bifurcation_bounds : tuple[float, float], optional
        Uniform prior bounds on (gamma, lambda) (the parametric Sec. 5
        extension; recovery itself is parameter-agnostic).
    bc_kind : str, optional
        Boundary configuration, forwarded to the physics builder.  The
        recovered reduced operator is boundary-condition-specific, so
        this is exposed here rather than baked in.
    ic_amplitude : float, optional
        Initial condition ``amplitude * sin(pi*x/2)``: zero at the
        Dirichlet boundary, zero slope at the Neumann boundary.
    final_time, deltat : float, optional
        Default FOM time integration settings (Crank-Nicolson).  These
        are snapshot-generation choices, NOT recovery inputs: the
        exact method derives its own single-step size.

    Returns
    -------
    PDEOpInfProblem
        Problem with degree set ``(1, 3)``, no inputs, zero lift.
    """
    bounds = (0.0, 1.0)
    basis = build_line_basis(nx, bounds, bkd, degree=degree)

    coords = basis.dof_coordinates()
    initial_condition = ic_amplitude * bkd.sin(
        0.5 * math.pi * coords[0]
    )

    prior = IndependentJoint(
        [
            UniformMarginal(
                diffusivity_bounds[0], diffusivity_bounds[1], bkd
            ),
            UniformMarginal(
                bifurcation_bounds[0], bifurcation_bounds[1], bkd
            ),
        ],
        bkd,
    )
    domain_bounds = bkd.array(
        [
            [diffusivity_bounds[0], diffusivity_bounds[1]],
            [bifurcation_bounds[0], bifurcation_bounds[1]],
        ]
    )

    return PDEOpInfProblem(
        name="homogeneous_chafee_infante_opinf",
        physics_factory=_ChafeeInfantePhysicsFactory(basis, bkd, bc_kind),
        basis=basis,
        prior=prior,
        domain=BoxDomain(_bounds=domain_bounds, _bkd=bkd),
        time_config=TimeIntegrationConfig(
            method="crank_nicolson",
            init_time=0.0,
            final_time=final_time,
            deltat=deltat,
            newton_tol=1e-10,
            newton_maxiter=20,
            lumped_mass=False,
            verbosity=0,
        ),
        initial_condition=initial_condition,
        nominal_parameters=bkd.array(
            [[nominal_diffusivity], [nominal_bifurcation]]
        ),
        degree_set=(1, 3),
        ninputs=0,
        lift_vector=bkd.zeros((basis.ndofs(),)),
        bkd=bkd,
        description=(
            "Homogeneous Chafee-Infante (cubic bistable reaction-"
            "diffusion) with zero Dirichlet at the left boundary and "
            "natural Neumann at the right; the Galerkin semi-"
            "discretization is exactly cubic in the state (no lift, "
            "no inputs)."
        ),
        reference=(
            "Rosenberger, Sanderse, Stabile (2025), Exact operator "
            "inference with minimal data."
        ),
    )
