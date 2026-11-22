"""Tests for the PDE operator-inference problems.

NumPy backend only (Galerkin uses skfem which is NumPy-based); numpy
appears ONLY at the skfem seam (sparse mass-matrix contractions) --
analysis is backend-generic.

The discriminating checks:
- problem metadata (degree set, inputs, lift) matches the exact
  polynomial structure of each semi-discretization;
- conservation/stability structure survives the BENCHMARK path (the
  builders, not hand-assembled physics);
- manufactured-solution convergence gates under the SAME boundary
  treatments the problems ship with (periodic for Burgers;
  Dirichlet-left/natural-Neumann-right for Chafee-Infante).
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.ode.step_plan import FixedStep
from pyapprox.ode.stepping_methods import RungeKutta, SteppingMethodProtocol
from pyapprox.ode.tableau import crank_nicolson, forward_euler
from pyapprox.ode.trajectory_store import InMemoryTrajectoryStore
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.protocols.system import (
    GalerkinTransientSystemProtocol,
)
from pyapprox.pde.galerkin.time_integration.skfem_discretization import (
    skfem_discretization,
)
from pyapprox.pde.models.galerkin.transient_system import (
    GalerkinTransientSystem,
    galerkin_transient_system,
    solve_galerkin_transient,
)
from pyapprox.pde.models.galerkin.verification import exact_dof_values, forcing
from pyapprox.pde.verification.conditions import (
    ConditionProtocol,
    Dirichlet,
    Neumann,
    Periodic,
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
from pyapprox.util.linalg.scipy_stack import CSRFormat, ScipySparseLU
from pyapprox.util.rootfinding.nonlinear import Newton
from pyapprox_benchmarks.functions.pde.burgers import (
    build_periodic_burgers_physics,
    build_periodic_line_basis,
)
from pyapprox_benchmarks.functions.pde.chafee_infante import (
    build_chafee_infante_system,
    build_line_basis,
)
from pyapprox_benchmarks.problems.pde import (
    build_chafee_infante_opinf_problem,
    build_periodic_burgers_opinf_problem,
)
from pyapprox_benchmarks.protocols import DomainProtocol

_Np = NDArray[Any]
_UNIT = Box([(0.0, 1.0)])
_CRANK_NICOLSON: RungeKutta[_Np, _Np] = RungeKutta(
    crank_nicolson(), Newton(atol=1e-10, rtol=0.0, maxiters=20)
)


def _mass_row_sums(system: GalerkinTransientSystemProtocol[_Np], bkd: NumpyBkd) -> _Np:
    """1^T M as a backend array -- the single skfem-seam contraction."""
    return bkd.asarray(np.asarray(system.mass_matrix().sum(axis=0)).ravel())


def _convergence_rates(errors: List[float]) -> List[float]:
    return [math.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]


def _transient(
    system: GalerkinTransientSystemProtocol[_Np],
) -> GalerkinTransientSystem[_Np, CSRFormat]:
    return galerkin_transient_system(skfem_discretization(system), ScipySparseLU())


def _solve(
    transient: GalerkinTransientSystem[_Np, CSRFormat],
    method: SteppingMethodProtocol[_Np, _Np],
    steps: FixedStep,
    init_state: _Np,
) -> Tuple[_Np, _Np]:
    """Every stored state, shape (nstates, ntimes), and the times, from
    ``init_state`` (shape (nstates,))."""
    store: InMemoryTrajectoryStore[_Np] = InMemoryTrajectoryStore()
    solve_galerkin_transient(
        transient,
        method,
        RecomputeAlways(transient.solved.solver),
        steps,
        init_state[:, None],
        store,
    )
    states = np.concatenate([store.state(n) for n in range(store.ntimes())], axis=1)
    return states, np.asarray(store.times())


def _problem(
    solution: str, terms: List[TermProtocol], conditions: Dict[str, ConditionProtocol]
) -> ManufacturedProblem:
    return ManufacturedProblem({"u": solution}, terms, _UNIT, conditions)


class TestPeriodicBurgersOpInfProblem:
    def test_metadata(self) -> None:
        bkd = NumpyBkd()
        problem = build_periodic_burgers_opinf_problem(bkd, nx=32)
        # periodic mesh: endpoints identified, nstates == nx
        assert problem.nstates() == 32
        assert problem.degree_set() == (1, 2)
        assert problem.ninputs() == 0
        assert problem.input_func() is None
        assert problem.nparams() == 1
        assert isinstance(problem.domain(), DomainProtocol)
        lift = problem.lift_vector()
        assert lift.shape[0] == 32
        assert bkd.to_float(bkd.max(bkd.abs(lift))) == 0.0

    def test_system_rebuilds_physics_on_shared_basis(self) -> None:
        bkd = NumpyBkd()
        problem = build_periodic_burgers_opinf_problem(bkd, nx=16)
        nominal = problem.system()
        other = problem.system(bkd.array([[0.3]]))
        assert nominal is not other
        assert nominal.nstates() == problem.nstates()
        assert other.nstates() == problem.nstates()

    def test_method_must_satisfy_its_protocol(self) -> None:
        bkd = NumpyBkd()
        problem = build_periodic_burgers_opinf_problem(bkd, nx=8)
        with pytest.raises(TypeError, match="SteppingMethodProtocol"):
            type(problem)(
                problem.name(),
                lambda param: problem.system(param),
                problem.basis(),
                problem.prior(),
                problem.domain(),
                "crank_nicolson",  # type: ignore[arg-type]
                problem.steps(),
                problem.initial_condition(),
                problem.nominal_parameters(),
                problem.degree_set(),
                problem.ninputs(),
                problem.lift_vector(),
                bkd,
            )

    def test_momentum_conservation_through_problem_path(self) -> None:
        """Forward Euler conserves total momentum 1^T M u_k on the
        periodic mesh -- exercised through the problem builder, not
        hand-assembled physics."""
        bkd = NumpyBkd()
        problem = build_periodic_burgers_opinf_problem(bkd, nx=32)
        system = problem.system()
        states, _ = _solve(
            problem.transient(system),
            RungeKutta(forward_euler()),
            FixedStep(1e-4, (0.0, 0.02)),
            problem.initial_condition(),
        )
        row_sums = _mass_row_sums(system, bkd)
        momenta = bkd.sum(row_sums[:, None] * states, axis=0)
        drift = bkd.to_float(bkd.max(bkd.abs(momenta - momenta[0])))
        assert drift < 1e-12

    def test_manufactured_solution_recovered_at_fem_rate(self) -> None:
        """MMS gate under the SAME periodic boundary treatment the
        problem ships with, through the physics builder: a
        space-periodic exact solution is recovered at the P1 rate."""
        bkd = NumpyBkd()
        problem = _problem(
            "(0.5 + 0.2*sin(2*pi*x))*exp(-t)",
            [
                TimeDerivative("u"),
                Divergence("u", ["-(0.1)*u_x"]),
                Source("u", "u*u_x"),
            ],
            {"left": Periodic(), "right": Periodic()},
        )
        errors = []
        for nx in (16, 32, 64):
            basis = build_periodic_line_basis(nx, (0.0, 1.0), bkd)
            physics = build_periodic_burgers_physics(
                basis, 0.1, bkd, forcing=forcing(problem, "u")
            )
            states, times = _solve(
                _transient(compose_galerkin_system(physics)),
                _CRANK_NICOLSON,
                FixedStep(1e-3, (0.0, 0.1)),
                exact_dof_values(problem, basis, ["u"], 0.0),
            )
            exact = exact_dof_values(problem, basis, ["u"], float(times[-1]))
            errors.append(
                bkd.to_float(bkd.norm(states[:, -1] - exact) / bkd.norm(exact))
            )
        assert min(_convergence_rates(errors)) > 1.7, errors


class TestChafeeInfanteOpInfProblem:
    def test_metadata(self) -> None:
        bkd = NumpyBkd()
        problem = build_chafee_infante_opinf_problem(bkd, nx=32)
        # standard mesh: nx + 1 P1 dofs (Dirichlet dof included)
        assert problem.nstates() == 33
        assert problem.degree_set() == (1, 3)
        assert problem.ninputs() == 0
        assert problem.nparams() == 2
        assert isinstance(problem.domain(), DomainProtocol)
        lift = problem.lift_vector()
        assert bkd.to_float(bkd.max(bkd.abs(lift))) == 0.0
        # initial condition satisfies the homogeneous Dirichlet BC
        u0 = problem.initial_condition()
        assert bkd.to_float(bkd.abs(u0[0])) == 0.0

    def test_unsupported_bc_kind_raises(self) -> None:
        bkd = NumpyBkd()
        basis = build_line_basis(8, (0.0, 1.0), bkd)
        with pytest.raises(ValueError, match="bc_kind"):
            build_chafee_infante_system(basis, 1.0, 1.0, bkd, bc_kind="periodic")

    def test_solve_fom_decays_below_bifurcation_threshold(self) -> None:
        """With lambda=1 below the first Dirichlet-Neumann eigenvalue
        gamma*(pi/2)^2 of the diffusion, u=0 is stable: the solution
        decays and the Dirichlet dof stays pinned at zero. Stepped with
        the problem's own default method."""
        bkd = NumpyBkd()
        problem = build_chafee_infante_opinf_problem(bkd, nx=32)
        u0 = problem.initial_condition()
        states, _ = _solve(
            problem.transient(problem.system()),
            problem.method(),
            FixedStep(5e-3, (0.0, 0.5)),
            u0,
        )
        boundary_drift = bkd.to_float(bkd.max(bkd.abs(states[0, :])))
        assert boundary_drift < 1e-12
        assert bkd.to_float(bkd.norm(states[:, -1])) < 0.5 * bkd.to_float(bkd.norm(u0))

    def test_manufactured_solution_recovered_at_fem_rate(self) -> None:
        """MMS gate under the SAME boundary treatment the problem
        ships with (zero Dirichlet left, natural Neumann right),
        through the system builder: sin(pi*x/2) vanishes at x=0 and
        has zero slope at x=1, so both BCs are satisfied exactly and
        the exact solution is recovered at the P1 rate."""
        bkd = NumpyBkd()
        problem = _problem(
            "sin(pi*x/2)*exp(-t)",
            [
                TimeDerivative("u"),
                Divergence("u", ["-u_x"]),
                Source("u", "-(u - u**3)"),
            ],
            {"left": Dirichlet(), "right": Neumann()},
        )
        errors = []
        for nx in (16, 32, 64):
            basis = build_line_basis(nx, (0.0, 1.0), bkd)
            system = build_chafee_infante_system(
                basis, 1.0, 1.0, bkd, forcing=forcing(problem, "u")
            )
            states, times = _solve(
                _transient(system),
                _CRANK_NICOLSON,
                FixedStep(1e-3, (0.0, 0.1)),
                exact_dof_values(problem, basis, ["u"], 0.0),
            )
            exact = exact_dof_values(problem, basis, ["u"], float(times[-1]))
            errors.append(
                bkd.to_float(bkd.norm(states[:, -1] - exact) / bkd.norm(exact))
            )
        assert min(_convergence_rates(errors)) > 1.7, errors
