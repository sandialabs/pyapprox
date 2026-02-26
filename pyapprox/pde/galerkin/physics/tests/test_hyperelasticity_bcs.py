"""Boundary condition tests for HyperelasticityPhysics.

Tests Neumann, Robin, and mixed BC combinations in 1D and 2D using
manufactured solutions with non-zero boundary values.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyapprox.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinHyperelasticityAdapter,
    create_hyperelasticity_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
)
from pyapprox.pde.galerkin.physics import HyperelasticityPhysics
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend


def _get_exact_displacement(funcs, basis, bkd):
    """Evaluate manufactured solution at DOF locations for vector basis."""
    ndim = basis.ncomponents()
    dof_coords = bkd.to_numpy(basis.dof_coordinates())
    n_dofs = basis.ndofs()
    sol = funcs["solution"](dof_coords)
    exact = np.zeros(n_dofs)
    for i in range(n_dofs):
        exact[i] = sol[i, i % ndim]
    return exact


# =========================================================================
# 1D BC Tests
# =========================================================================


class TestHyperelasticityBCs1DBase:
    """1D boundary condition tests with non-zero boundary values."""

    def _setup(self, bkd) -> None:
        self._stress = NeoHookeanStress(1.0, 1.0)
        # Solution non-zero at x=1: u(0)=0, u(1)=0.05
        self._sol_strs = ["0.1*x**2*(1-x) + 0.05*x"]

    def _setup_problem(self, bkd, bc_types, nx=30, degree=2, robin_alpha=1.0) :
        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=[0.0, 1.0],
            sol_strs=self._sol_strs,
            stress_model=self._stress,
            bkd=bkd,
        )
        mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=degree)
        adapter = GalerkinHyperelasticityAdapter(basis, functions, bkd)
        body_force = adapter.forcing_for_galerkin()
        bc_set = adapter.create_boundary_conditions(bc_types, robin_alpha=robin_alpha)
        physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=self._stress,
            bkd=bkd,
            body_force=body_force,
            boundary_conditions=bc_set.all_conditions(),
        )
        return physics, functions, basis

    def _check_newton_solve(self, bkd, physics, functions, basis, tol=1e-4) :
        exact = _get_exact_displacement(functions, basis, bkd)
        solver = SteadyStateSolver(physics, tol=1e-10, max_iter=20, line_search=True)
        init_guess = bkd.asarray(exact + 0.005)
        result = solver.solve(init_guess)
        assert result.converged, f"Newton did not converge: {result.residual_norm:.2e}"
        u_np = bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(exact)
        rel_error = np.linalg.norm(u_np - exact) / max(u_norm, 1e-30)
        assert rel_error < tol

    def test_bc_dirichlet_neumann_1d(self, numpy_bkd) -> None:
        """Dirichlet left, Neumann right."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_problem(bkd, ["D", "N"])
        self._check_newton_solve(bkd, physics, functions, basis)

    def test_bc_dirichlet_robin_1d(self, numpy_bkd) -> None:
        """Dirichlet left, Robin right."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_problem(bkd, ["D", "R"])
        self._check_newton_solve(bkd, physics, functions, basis)

    def test_bc_robin_dirichlet_1d(self, numpy_bkd) -> None:
        """Robin left, Dirichlet right."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_problem(bkd, ["R", "D"])
        self._check_newton_solve(bkd, physics, functions, basis)

    def test_bc_residual_at_exact_neumann_1d(self, numpy_bkd) -> None:
        """Residual at exact solution should be small with Neumann BC."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_problem(bkd, ["D", "N"])
        exact = _get_exact_displacement(functions, basis, bkd)
        state = bkd.asarray(exact)
        res = physics.residual(state, 0.0)
        res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
        assert res_norm < 1e-3




# =========================================================================
# 2D BC Tests
# =========================================================================


class TestHyperelasticityBCs2DBase:
    """2D boundary condition tests with non-zero boundary values."""

    def _setup(self, bkd) -> None:
        self._stress = NeoHookeanStress(1.0, 1.0)
        # Non-zero on right/top: u vanishes at x=0, y=0 but not at x=1, y=1
        self._sol_strs = [
            "0.1*x**2*(1-x)*y**2*(1-y) + 0.02*x*y",
            "0.05*x**2*(1-x)*y**2*(1-y) + 0.01*x*y",
        ]

    def _setup_problem(self, bkd, bc_types, nx=12, ny=12, degree=2, robin_alpha=1.0) :
        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0],
            sol_strs=self._sol_strs,
            stress_model=self._stress,
            bkd=bkd,
        )
        mesh = StructuredMesh2D(nx=nx, ny=ny, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=degree)
        adapter = GalerkinHyperelasticityAdapter(basis, functions, bkd)
        body_force = adapter.forcing_for_galerkin()
        bc_set = adapter.create_boundary_conditions(bc_types, robin_alpha=robin_alpha)
        physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=self._stress,
            bkd=bkd,
            body_force=body_force,
            boundary_conditions=bc_set.all_conditions(),
        )
        return physics, functions, basis

    def _check_newton_solve(self, bkd, physics, functions, basis, tol=1e-3) :
        exact = _get_exact_displacement(functions, basis, bkd)
        solver = SteadyStateSolver(physics, tol=1e-10, max_iter=20, line_search=True)
        init_guess = bkd.asarray(exact + 0.005)
        result = solver.solve(init_guess)
        assert result.converged, f"Newton did not converge: {result.residual_norm:.2e}"
        u_np = bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(exact)
        rel_error = np.linalg.norm(u_np - exact) / max(u_norm, 1e-30)
        assert rel_error < tol

    def test_bc_mixed_DN_2d(self, numpy_bkd) -> None:
        """Dirichlet left/bottom, Neumann right/top."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_problem(bkd, ["D", "N", "D", "N"])
        self._check_newton_solve(bkd, physics, functions, basis)

    def test_bc_mixed_DR_2d(self, numpy_bkd) -> None:
        """Dirichlet left/bottom, Robin right/top."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_problem(bkd, ["D", "R", "D", "R"])
        self._check_newton_solve(bkd, physics, functions, basis)

    def test_bc_mixed_DNR_2d(self, numpy_bkd) -> None:
        """Dirichlet left, Neumann right, Dirichlet bottom, Robin top."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_problem(bkd, ["D", "N", "D", "R"])
        self._check_newton_solve(bkd, physics, functions, basis)

    def test_bc_residual_at_exact_mixed_2d(self, numpy_bkd) -> None:
        """Residual at exact solution should be small with mixed BCs."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_problem(bkd, ["D", "N", "D", "R"])
        exact = _get_exact_displacement(functions, basis, bkd)
        state = bkd.asarray(exact)
        res = physics.residual(state, 0.0)
        res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
        assert res_norm < 1e-3


