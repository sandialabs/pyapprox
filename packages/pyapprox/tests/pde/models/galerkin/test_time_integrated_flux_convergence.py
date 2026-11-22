"""Convergence of the time-integrated essential-boundary flux.

Advection-diffusion-reaction on [-1, 1]^2 with a manufactured transient
solution: Dirichlet on the left edge, natural (diffusive-flux) data
on the other three edges, exact initial state, stated as a
``pde.verification`` problem and stepped by ``solve_galerkin_transient``.
The flux term ``time_integrated_essential_boundary_flux`` with unit
weights on the left DOFs, summed over the steps, converges to

    Q* = int_0^T int_left law(x, t, u*, grad u*, n) ds dt,

the flux law on exact inputs, integrated at high order in space (facet
quadrature) and time (Gauss-Legendre).

The two error sources are separated:

- temporal: a solution quadratic in space, which P2 elements represent
  exactly, leaves only the time-stepping error; the rate is the
  stepper's order;
- spatial: a smooth solution on P1 with Crank-Nicolson at a small step
  gives the h^{2p} rate of the steady flux. Wrong-model control: the
  steady flux integrated over time drops M u_t on the boundary rows and
  converges only at first order in h.
"""

from typing import Any, Callable, Dict, List, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
import sympy as sp
from skfem import Functional, asm

from pyapprox.implicitfunction.setups import KeepStateFree, RecomputeAlways
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta
from pyapprox.ode.tableau import (
    ButcherTableau,
    backward_euler,
    crank_nicolson,
    implicit_midpoint,
)
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.expressions.symbols import coordinates
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import NormalFluxLawProtocol
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    time_integrated_essential_boundary_flux,
)
from pyapprox.pde.models.galerkin.transient_system import (
    galerkin_transient_system,
    solve_galerkin_transient,
)
from pyapprox.pde.models.galerkin.verification import (
    SkfemPoints,
    boundary_conditions,
    exact_dof_values,
    forcing,
    state_law,
    vector_field,
)
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
)
from pyapprox.pde.verification.domains import Box
from pyapprox.pde.verification.problem import ManufacturedProblem
from pyapprox.pde.verification.terms import (
    Divergence,
    Source,
    TermProtocol,
    TimeDerivative,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from tests._helpers.adjoint_checks import NumpyArray

_DIFF = 0.7
_VEL = ("1.0", "0.5")
_REACT = "1.0*u"
_FINAL_TIME = 0.4
_SQUARE = Box([(-1.0, 1.0)] * 2)
# Quadratic in space: P2 represents it exactly, leaving only time error.
_QUADRATIC = "(x**2 + 0.5*x*y - y**2 + 0.3*x)*(1 + sin(2*t))"
_SMOOTH = "sin(1.3*x)*cos(0.7*y)*(1 + sin(2*t)) + 0.5*x*y"


def _problem(solution: str) -> ManufacturedProblem:
    r""":math:`u_t + \nabla\cdot(-D\nabla u) + v\cdot\nabla u - R(u) = f`,
    Dirichlet on the left, natural (diffusive-flux) data elsewhere."""
    terms: List[TermProtocol] = [
        TimeDerivative("u"),
        Divergence("u", [f"-({_DIFF})*u_x", f"-({_DIFF})*u_y"]),
        Source("u", f"({_VEL[0]})*u_x + ({_VEL[1]})*u_y"),
        Source("u", f"-({_REACT})"),
    ]
    conditions: Dict[str, ConditionProtocol] = {
        "left": Dirichlet(),
        "right": Neumann(),
        "bottom": Neumann(),
        "top": Neumann(),
    }
    return ManufacturedProblem({"u": solution}, terms, _SQUARE, conditions)


class _EdgeFluxIntegrand:
    """law(x, u*, grad u*, n, t) at the facet quadrature points."""

    __name__ = "edge_flux"

    def __init__(
        self,
        law: NormalFluxLawProtocol,
        problem: ManufacturedProblem,
        time: float,
    ) -> None:
        self._law = law
        solution = problem.solution("u")
        gradient = [sp.diff(solution, c) for c in coordinates(("x", "y"))]
        self._solution = SkfemPoints([solution], problem)
        self._gradient = SkfemPoints(gradient, problem)
        self._time = time

    def __call__(self, w: Any) -> Any:
        shape = w.x.shape[1:]
        x = np.asarray(w.x).reshape(2, -1)
        normal = np.asarray(w.n).reshape(2, -1)
        u = self._solution(x, t=self._time)
        grad_u = self._gradient(x, t=self._time)[None, :, :]
        flux = self._law.normal_flux(x, u, grad_u, normal, self._time)
        return np.asarray(flux)[0].reshape(shape)


def _method(tableau: ButcherTableau) -> RungeKutta[NumpyArray, NumpyArray]:
    return RungeKutta(tableau, Newton(atol=1e-12, rtol=0.0, maxiters=20))


def _fluxes(
    bkd: NumpyBkd,
    solution: str,
    nx: int,
    degree: int,
    make: Callable[[], ButcherTableau],
    deltat: float,
) -> Tuple[float, float, float]:
    """(the flux term's Q, the steady flux integrated by the trapezoid
    rule over the stored states, Q*)."""
    problem = _problem(solution)
    mesh = StructuredMesh2D(nx=nx, ny=nx, bounds=[(-1.0, 1.0), (-1.0, 1.0)], bkd=bkd)
    basis = LagrangeBasis(mesh, degree=degree)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=_DIFF,
        bkd=bkd,
        velocity=vector_field(problem, _VEL),
        reaction=state_law(problem, _REACT, "u"),
        forcing=forcing(problem, "u"),
    )
    law = physics.flux_law()
    system = compose_galerkin_system(physics, boundary_conditions(problem, basis, "u"))
    transient = galerkin_transient_system(skfem_discretization(system), ScipySparseLU())
    method = _method(make())
    store: InMemoryTrajectoryStore[NumpyArray] = InMemoryTrajectoryStore()
    solve_galerkin_transient(
        transient,
        method,
        KeepStateFree(transient.solved.solver),
        FixedStep(deltat, (0.0, _FINAL_TIME)),
        exact_dof_values(problem, basis, ["u"], 0.0)[:, None],
        store,
    )
    dofs = basis.get_dofs("left")
    term = time_integrated_essential_boundary_flux(
        transient,
        method,
        RecomputeAlways(transient.unconstrained.solver),
        dofs,
        bkd.ones((1, dofs.shape[0])),
    )
    flux = sum(
        float(term.value(step, store.window(step))[0, 0])
        for step in range(store.nsteps())
    )

    def steady(n: int) -> float:
        rows = system.spatial_operator().spatial_residual(
            store.state(n)[:, 0], store.time(n)
        )
        return float(np.sum(np.asarray(rows)[dofs]))

    steady_integrated = -sum(
        0.5 * (store.time(n + 1) - store.time(n)) * (steady(n) + steady(n + 1))
        for n in range(store.nsteps())
    )

    facet_basis = basis.skfem_basis().boundary("left", intorder=10)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    target = sum(
        0.5
        * _FINAL_TIME
        * wt
        * float(
            asm(
                Functional(
                    _EdgeFluxIntegrand(law, problem, 0.5 * _FINAL_TIME * (nd + 1))
                ),
                facet_basis,
            )
        )
        for nd, wt in zip(nodes, weights)
    )
    return flux, steady_integrated, target


def _rates(errors: List[float]) -> NumpyArray:
    errs = np.asarray(errors)
    rates: NumpyArray = np.log2(errs[:-1] / errs[1:])
    return rates


class TestTemporalConvergence:
    """P2 on 4 x 4 cells, dt = 0.02, 0.01, 0.005 over [0, 0.4]. Measured:
    backward Euler 1.7e-2 -> 4.3e-3 (rates 1.00), Crank-Nicolson
    4.7e-5 -> 2.9e-6 (2.00), implicit midpoint 1.8e-6 -> 1.2e-7 (1.93,
    1.97). Implicit midpoint reaches its order from below: its stage's
    essential DOFs integrate the data's rate rather than hold the data,
    and at dt = 0.04 the rate is still 1.83."""

    @pytest.mark.parametrize(
        "method,order",
        [(backward_euler, 1), (crank_nicolson, 2), (implicit_midpoint, 2)],
        ids=["backward_euler", "crank_nicolson", "implicit_midpoint"],
    )
    def test_converges_at_the_stepper_order(
        self, numpy_bkd: NumpyBkd, method: Callable[[], ButcherTableau], order: int
    ) -> None:
        errors = [
            abs(q - t)
            for q, _, t in (
                _fluxes(numpy_bkd, _QUADRATIC, 4, 2, method, dt)
                for dt in (0.02, 0.01, 0.005)
            )
        ]
        rates = _rates(errors)
        assert np.all(np.abs(rates - order) < 0.1), (errors, rates)


class TestSpatialConvergence:
    """P1 on 4, 8, 16 cells per side, Crank-Nicolson with dt = 0.005.
    Measured: the flux term 3.4e-2 -> 2.1e-3 (rate 2.0, h^{2p}); the
    steady flux integrated over time by the trapezoid rule, Crank-Nicolson's
    own, 0.31 -> 0.08 (rate 1.0)."""

    def test_converges_at_twice_the_energy_rate(self, numpy_bkd: NumpyBkd) -> None:
        results = [
            _fluxes(numpy_bkd, _SMOOTH, nx, 1, crank_nicolson, 0.005)
            for nx in (4, 8, 16)
        ]
        flux_rates = _rates([abs(q - t) for q, _, t in results])
        steady_rates = _rates([abs(s - t) for _, s, t in results])
        assert np.all(flux_rates > 1.8), flux_rates
        # Wrong-model control: dropping M u_t on the boundary rows.
        assert np.all(steady_rates < 1.2), steady_rates
