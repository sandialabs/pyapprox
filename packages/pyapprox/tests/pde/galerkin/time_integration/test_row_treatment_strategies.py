"""Exact discrete conservation under the essential-row strategies.

1D heat equation on [0, 1], P1 on 10 cells, u_t = 0.1 u'' + f with
f = (1 + x)(1 + sin 5t), u(0, t) = 0 and a moving right end
u(1, t) = 0.5 sin 4t (its rate supplied), from 0.3 sin(pi x), to
T = 0.1. Three strategies:

- A, today's: the coupled mass with the BC wrapper pinning each step's
  end inside the solve (the forward model's path);
- B: the symmetric mass with the mass-consistent projection, stepped
  without the wrapper;
- C: the symmetric mass with the essential pin, the control.

Total content 1^T M u is conserved exactly when each stored step's
interior rows hold with F evaluated where the scheme evaluated it (the
boundary flux is then whatever the essential rows carry):
M_I.(y_{n+1} - y_n) = dt Phi_I, Phi being F(z_{n+1}) for backward Euler,
F(y_n) for forward Euler, the average of F(y_n) and F(z_{n+1}) for
Crank-Nicolson, F at the average of y_n and z_{n+1} for implicit
midpoint, and Heun's average over y_n and its predictor
y_n + dt M^{-1} f(y_n), with y_n the stored start of the step and
z_{n+1} the step's own unknown (the Newton solution, before any
projection; the stored state under A, which pins it inside the solve).
A and B are both exact for every stepper here: A because every stepper
ends its step with a mass solve for y_{n+1}, Heun included, which the
wrapper pins; B through the projection. For explicit steppers they are
one scheme; for implicit ones two schemes of the same order. What A
adds for the one-step methods is that their scheme points are stored
states, so a flux computed from stored states alone is the scheme's;
under B an implicit step's unknown is not stored, and evaluating F at
the stored state instead leaves a content gap one order above the
method. Heun's stage is stored under neither. dt = 0.001 keeps Heun stable; at 0.002
it grows and the identity is lost in round-off of a growing solution.
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Callable, Dict, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.ode.implicit_steppers.integrator import TimeIntegrator
from pyapprox.ode.protocols import StateConstraintProtocol
from pyapprox.ode.stepper_table import create_stepper
from pyapprox.pde.boundary import BoundarySignal
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.galerkin.time_integration import (
    GalerkinTransientModel,
    SymmetricNeutralizedRows,
    skfem_ode_residual,
)
from pyapprox.pde.galerkin.time_integration.state_constraints import (
    EssentialPin,
    MassConsistentProjection,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.rootfinding.newton import NewtonSolver

_Np = NDArray[Any]
_FINAL_TIME = 0.1
_DELTAT = 0.001
_Strategy = Literal["A", "B", "C"]


def _forcing(x: _Np, t: float) -> _Np:
    value: _Np = (1.0 + x[0]) * (1.0 + np.sin(5.0 * t))
    return value


def _right_value(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 0.5 * np.sin(4.0 * t))
    return value


def _right_rate(x: _Np, t: float) -> _Np:
    value: _Np = np.full(x.shape[1], 2.0 * np.cos(4.0 * t))
    return value


class _Heat:
    """The system, its dense consistent mass and the interior DOFs."""

    def __init__(self, bkd: NumpyBkd) -> None:
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        physics = AdvectionDiffusionReaction(
            basis=basis, diffusivity=0.1, bkd=bkd, forcing=TimeDependent(_forcing)
        )
        right = BoundarySignal(
            TimeDependent(_right_value), [TimeDependent(_right_rate)]
        )
        self.system: GalerkinSystem[_Np] = compose_galerkin_system(
            physics,
            [
                DirichletBC(basis, "left", 0.0, bkd),
                DirichletBC(basis, "right", right, bkd),
            ],
        )
        raw = self.system.mass_matrix()
        self.mass: _Np = (
            np.asarray(raw.toarray()) if isinstance(raw, spmatrix) else np.asarray(raw)
        )
        dofs = np.asarray(self.system.constraint_set().dofs())
        self.interior = np.setdiff1d(np.arange(self.mass.shape[0]), dofs)
        x = np.asarray(basis.dof_coordinates())[0]
        self.init_state: _Np = 0.3 * np.sin(np.pi * x)

    def spatial(self, state: _Np, time: float) -> _Np:
        return np.asarray(self.system.spatial_operator().spatial_residual(state, time))


def _solve(
    heat: _Heat, strategy: _Strategy, method: str, deltat: float
) -> tuple[_Np, _Np]:
    """The stored trajectory and times under ``strategy``."""
    traj, times, _ = _solve_with_unknowns(heat, strategy, method, deltat)
    return traj, times


def _solve_with_unknowns(
    heat: _Heat, strategy: _Strategy, method: str, deltat: float
) -> tuple[_Np, _Np, _Np]:
    """The stored trajectory, times and each step's own unknown (column
    n + 1 for step n): the stored state under A, which pins the unknown
    inside the solve, the unprojected Newton solution otherwise."""
    if strategy == "A":
        model = GalerkinTransientModel(skfem_ode_residual(heat.system))
        config: TimeIntegrationConfig[_Np] = TimeIntegrationConfig(
            method=method,
            init_time=0.0,
            final_time=_FINAL_TIME,
            deltat=deltat,
            newton_tol=1e-13,
            newton_maxiter=20,
            verbosity=0,
        )
        traj, times = model.solve_transient(heat.init_state, config)
        return traj, times, traj
    adapter = skfem_ode_residual(heat.system, row_treatment=SymmetricNeutralizedRows)
    constraint: StateConstraintProtocol[_Np] = (
        MassConsistentProjection(adapter, 0)
        if strategy == "B"
        else EssentialPin(heat.system, 0)
    )
    newton = NewtonSolver(create_stepper(method, adapter))
    newton.set_options(maxiters=20, atol=1e-13, rtol=0.0)
    integrator = TimeIntegrator(0.0, _FINAL_TIME, deltat, newton, constraint=constraint)
    traj, times = integrator.solve(heat.init_state)
    return traj, times, integrator.unprojected_states()


def _rule(heat: _Heat, method: str) -> Callable[[_Np, _Np, float, float], _Np]:
    """The method's evaluation rule Phi(y_n, z_{n+1}, t_n, dt): from the
    step's stored start and its own unknown (Heun's predictor rebuilt from
    y_n with the scheme's own mass and right-hand side)."""
    adapter = skfem_ode_residual(heat.system, row_treatment=SymmetricNeutralizedRows)

    def heun(y0: _Np, y1: _Np, t0: float, dt: float) -> _Np:
        predictor = y0 + dt * adapter.mass_matrix().solve(adapter(y0, t0))
        return 0.5 * (heat.spatial(y0, t0) + heat.spatial(predictor, t0 + dt))

    rules: Dict[str, Callable[[_Np, _Np, float, float], _Np]] = {
        "backward_euler": lambda y0, y1, t0, dt: heat.spatial(y1, t0 + dt),
        "forward_euler": lambda y0, y1, t0, dt: heat.spatial(y0, t0),
        "crank_nicolson": lambda y0, y1, t0, dt: 0.5
        * (heat.spatial(y0, t0) + heat.spatial(y1, t0 + dt)),
        "implicit_midpoint": lambda y0, y1, t0, dt: heat.spatial(
            0.5 * (y0 + y1), t0 + 0.5 * dt
        ),
        "heun": heun,
    }
    return rules[method]


def _interior_residuals(
    heat: _Heat,
    strategy: _Strategy,
    method: str,
    deltat: Optional[float] = None,
    at_unknowns: bool = True,
) -> tuple[float, float]:
    """The worst interior residual of any stored step relative to its
    mass increment, and the content gap 1^T sum_n r_I; F at each step's
    own unknown, or with ``at_unknowns=False`` at the stored state that
    replaced it."""
    traj, times, unknowns = _solve_with_unknowns(
        heat, strategy, method, _DELTAT if deltat is None else deltat
    )
    points = unknowns if at_unknowns else traj
    rule = _rule(heat, method)
    rows = heat.mass[heat.interior]
    worst, gap = 0.0, 0.0
    for n in range(traj.shape[1] - 1):
        y0, y1 = traj[:, n], traj[:, n + 1]
        t0, dt = float(times[n]), float(times[n + 1] - times[n])
        increment = rows @ (y1 - y0)
        residual = increment - dt * rule(y0, points[:, n + 1], t0, dt)[heat.interior]
        worst = max(worst, float(np.abs(residual).max() / np.abs(increment).max()))
        gap += float(residual.sum())
    return worst, gap


class TestExactConservation:
    @pytest.mark.parametrize(
        "method",
        [
            "backward_euler",
            "forward_euler",
            "crank_nicolson",
            "implicit_midpoint",
            "heun",
        ],
    )
    def test_todays_strategy_for_every_stepper(
        self, numpy_bkd: NumpyBkd, method: str
    ) -> None:
        """Every stepper here ends its step with a mass solve for y_{n+1},
        Heun included, and the wrapper pins the end inside it."""
        worst, _ = _interior_residuals(_Heat(numpy_bkd), "A", method)
        assert worst < 1e-10

    @pytest.mark.parametrize(
        "method",
        [
            "backward_euler",
            "forward_euler",
            "crank_nicolson",
            "implicit_midpoint",
            "heun",
        ],
    )
    def test_projection_for_every_stepper(
        self, numpy_bkd: NumpyBkd, method: str
    ) -> None:
        """Heun is multistage, where today's strategy cannot be exact."""
        worst, _ = _interior_residuals(_Heat(numpy_bkd), "B", method)
        assert worst < 1e-10

    @pytest.mark.parametrize("method", ["forward_euler", "heun"])
    def test_plain_pin_is_not_exact(self, numpy_bkd: NumpyBkd, method: str) -> None:
        """Control: the residual left by pinning alone is far above round-off."""
        worst, _ = _interior_residuals(_Heat(numpy_bkd), "C", method)
        assert worst > 1e-7


class TestStoredStateFlux:
    @pytest.mark.parametrize(
        "method, order",
        [("backward_euler", 1), ("crank_nicolson", 2), ("implicit_midpoint", 2)],
    )
    def test_differs_from_the_schemes_under_the_projection(
        self, numpy_bkd: NumpyBkd, method: str, order: int
    ) -> None:
        """What today's strategy adds: under the projection an implicit
        step's unknown is not stored, so F evaluated at the stored state
        is not the scheme's flux. The content gap that leaves shrinks as
        dt^{p+1} (halving dt divides it by about 4 or 8), so a flux read
        from stored states needs today's strategy, or the step unknowns."""
        heat = _Heat(numpy_bkd)
        _, coarse = _interior_residuals(
            heat, "B", method, 2 * _DELTAT, at_unknowns=False
        )
        _, fine = _interior_residuals(heat, "B", method, _DELTAT, at_unknowns=False)
        assert abs(coarse) > 1e-12
        ratio = coarse / fine
        assert 0.85 * 2 ** (order + 1) < ratio < 1.15 * 2 ** (order + 1)


class TestEquivalence:
    @pytest.mark.parametrize("method", ["forward_euler", "heun"])
    def test_explicit_trajectories_agree(
        self, numpy_bkd: NumpyBkd, method: str
    ) -> None:
        """An explicit step evaluates F only at states it already knows, so
        pinning inside its final mass solve and projecting after it solve
        the same equations: one scheme."""
        heat = _Heat(numpy_bkd)
        coupled, _ = _solve(heat, "A", method, _DELTAT)
        projected, _ = _solve(heat, "B", method, _DELTAT)
        numpy_bkd.assert_allclose(projected, coupled, rtol=1e-10, atol=1e-13)

    def test_implicit_trajectories_differ(self, numpy_bkd: NumpyBkd) -> None:
        """Backward Euler: today's strategy pins the unknown inside the
        Newton solve, so F sees g(t_{n+1}); the projection solves with the
        boundary at its quadrature value and corrects after. Two exactly
        conservative schemes of the same order, not one."""
        heat = _Heat(numpy_bkd)
        coupled, _ = _solve(heat, "A", "backward_euler", _DELTAT)
        projected, _ = _solve(heat, "B", "backward_euler", _DELTAT)
        assert np.abs(projected - coupled).max() > 1e-8
