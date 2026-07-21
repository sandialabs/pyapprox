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

import numpy as np
from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.manufactured.burgers import (
    ManufacturedBurgers1D,
)
from pyapprox.pde.galerkin.manufactured.adapter import (
    create_adr_manufactured_test,
)
from pyapprox.pde.galerkin.time_integration.galerkin_model import (
    GalerkinModel,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks.functions.pde.burgers import (
    build_periodic_burgers_physics,
    build_periodic_line_basis,
)
from pyapprox_benchmarks.functions.pde.chafee_infante import (
    build_chafee_infante_physics,
    build_line_basis,
)
from pyapprox_benchmarks.problems.pde import (
    build_chafee_infante_opinf_problem,
    build_periodic_burgers_opinf_problem,
)
from pyapprox_benchmarks.protocols import DomainProtocol


def _mass_row_sums(physics, bkd):
    """1^T M as a backend array -- the single skfem-seam contraction."""
    return bkd.asarray(
        np.asarray(physics.mass_matrix().sum(axis=0)).ravel()
    )


def _convergence_rates(errors):
    return [
        math.log2(errors[k] / errors[k + 1])
        for k in range(len(errors) - 1)
    ]


class TestPeriodicBurgersOpInfProblem:
    def test_metadata(self):
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

    def test_model_rebuilds_physics_on_shared_basis(self):
        bkd = NumpyBkd()
        problem = build_periodic_burgers_opinf_problem(bkd, nx=16)
        model_nominal = problem.model()
        model_other = problem.model(bkd.array([[0.3]]))
        assert model_nominal.physics() is not model_other.physics()
        assert model_nominal.nstates() == problem.nstates()
        assert model_other.nstates() == problem.nstates()

    def test_momentum_conservation_through_problem_path(self):
        """Forward Euler conserves total momentum 1^T M u_k on the
        periodic mesh -- exercised through the problem builder, not
        hand-assembled physics."""
        bkd = NumpyBkd()
        problem = build_periodic_burgers_opinf_problem(bkd, nx=32)
        model = problem.model()
        config = TimeIntegrationConfig(
            method="forward_euler", init_time=0.0, final_time=0.02,
            deltat=1e-4,
        )
        states, _ = model.solve_transient(
            problem.initial_condition(), config
        )
        row_sums = _mass_row_sums(model.physics(), bkd)
        momenta = bkd.sum(row_sums[:, None] * states, axis=0)
        drift = bkd.to_float(bkd.max(bkd.abs(momenta - momenta[0])))
        assert drift < 1e-12

    def test_manufactured_solution_recovered_at_fem_rate(self):
        """MMS gate under the SAME periodic boundary treatment the
        problem ships with, through the physics builder: a
        space-periodic exact solution is recovered at the P1 rate."""
        bkd = NumpyBkd()
        man_sol = ManufacturedBurgers1D(
            sol_str="(0.5 + 0.2*sin(2*pi*x))*exp(-T)",
            visc_str="0.1",
            bkd=bkd,
            oned=True,
        )
        solution = man_sol.functions["solution"]
        forcing = man_sol.functions["forcing"]

        errors = []
        for nx in (16, 32, 64):
            basis = build_periodic_line_basis(nx, (0.0, 1.0), bkd)
            physics = build_periodic_burgers_physics(
                basis, 0.1, bkd, forcing=forcing
            )
            coords = basis.dof_coordinates()
            u0 = bkd.ravel(solution(coords, 0.0))
            config = TimeIntegrationConfig(
                method="crank_nicolson", init_time=0.0,
                final_time=0.1, deltat=1e-3,
            )
            states, times = GalerkinModel(physics, bkd).solve_transient(
                u0, config
            )
            exact = bkd.ravel(
                solution(coords, bkd.to_float(times[-1]))
            )
            errors.append(
                bkd.to_float(
                    bkd.norm(states[:, -1] - exact) / bkd.norm(exact)
                )
            )
        assert min(_convergence_rates(errors)) > 1.7, errors


class TestChafeeInfanteOpInfProblem:
    def test_metadata(self):
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

    def test_unsupported_bc_kind_raises(self):
        bkd = NumpyBkd()
        basis = build_line_basis(8, (0.0, 1.0), bkd)
        with pytest.raises(ValueError, match="bc_kind"):
            build_chafee_infante_physics(
                basis, 1.0, 1.0, bkd, bc_kind="periodic"
            )

    def test_solve_fom_decays_below_bifurcation_threshold(self):
        """With lambda=1 below the first Dirichlet-Neumann eigenvalue
        gamma*(pi/2)^2 of the diffusion, u=0 is stable: the solution
        decays and the Dirichlet dof stays pinned at zero."""
        bkd = NumpyBkd()
        problem = build_chafee_infante_opinf_problem(bkd, nx=32)
        model = problem.model()
        config = TimeIntegrationConfig(
            method="crank_nicolson", init_time=0.0, final_time=0.5,
            deltat=5e-3,
        )
        u0 = problem.initial_condition()
        states, _ = model.solve_transient(u0, config)
        boundary_drift = bkd.to_float(bkd.max(bkd.abs(states[0, :])))
        assert boundary_drift < 1e-12
        assert bkd.to_float(bkd.norm(states[:, -1])) < 0.5 * bkd.to_float(
            bkd.norm(u0)
        )

    def test_manufactured_solution_recovered_at_fem_rate(self):
        """MMS gate under the SAME boundary treatment the problem
        ships with (zero Dirichlet left, natural Neumann right),
        through the physics builder: sin(pi*x/2) vanishes at x=0 and
        has zero slope at x=1, so both BCs are satisfied exactly and
        the exact solution is recovered at the P1 rate."""
        bkd = NumpyBkd()
        functions, _ = create_adr_manufactured_test(
            bounds=[0.0, 1.0],
            sol_str="sin(pi*x/2)*exp(-T)",
            diff_str="1.0",
            react_str="u - u**3",
            vel_strs=["1e-16*x"],
            bkd=bkd,
            time_dependent=True,
        )
        solution = functions["solution"]
        forcing = functions["forcing"]

        errors = []
        for nx in (16, 32, 64):
            basis = build_line_basis(nx, (0.0, 1.0), bkd)
            physics = build_chafee_infante_physics(
                basis, 1.0, 1.0, bkd, forcing=forcing
            )
            coords = basis.dof_coordinates()
            u0 = bkd.ravel(solution(coords, 0.0))
            config = TimeIntegrationConfig(
                method="crank_nicolson", init_time=0.0,
                final_time=0.1, deltat=1e-3,
            )
            states, times = GalerkinModel(physics, bkd).solve_transient(
                u0, config
            )
            exact = bkd.ravel(
                solution(coords, bkd.to_float(times[-1]))
            )
            errors.append(
                bkd.to_float(
                    bkd.norm(states[:, -1] - exact) / bkd.norm(exact)
                )
            )
        assert min(_convergence_rates(errors)) > 1.7, errors
