"""Boundary condition tests for HyperelasticityPhysics.

Tests Neumann, Robin, and mixed BC combinations in 1D and 2D using
manufactured solutions with non-zero boundary values.
"""

from typing import Any, Generic

import unittest

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
)
from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.typing.pde.galerkin.physics import HyperelasticityPhysics
from pyapprox.typing.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.typing.pde.galerkin.manufactured.adapter import (
    create_hyperelasticity_manufactured_test,
    GalerkinHyperelasticityAdapter,
)
from pyapprox.typing.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)


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


class TestHyperelasticityBCs1DBase(Generic[Array], unittest.TestCase):
    """1D boundary condition tests with non-zero boundary values."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._stress = NeoHookeanStress(1.0, 1.0)
        # Solution non-zero at x=1: u(0)=0, u(1)=0.05
        self._sol_strs = ["0.1*x**2*(1-x) + 0.05*x"]

    def _setup_problem(self, bc_types, nx=30, degree=2, robin_alpha=1.0):
        bkd = self._bkd
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
        bc_set = adapter.create_boundary_conditions(
            bc_types, robin_alpha=robin_alpha
        )
        physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=self._stress,
            bkd=bkd,
            body_force=body_force,
            boundary_conditions=bc_set.all_conditions(),
        )
        return physics, functions, basis

    def _check_newton_solve(self, physics, functions, basis, tol=1e-4):
        exact = _get_exact_displacement(functions, basis, self._bkd)
        solver = SteadyStateSolver(
            physics, tol=1e-10, max_iter=20, line_search=True
        )
        init_guess = self._bkd.asarray(exact + 0.005)
        result = solver.solve(init_guess)
        self.assertTrue(
            result.converged,
            f"Newton did not converge: {result.residual_norm:.2e}",
        )
        u_np = self._bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(exact)
        rel_error = np.linalg.norm(u_np - exact) / max(u_norm, 1e-30)
        self.assertLess(
            rel_error, tol,
            f"Newton solve rel error: {rel_error:.2e}",
        )

    def test_bc_dirichlet_neumann_1d(self) -> None:
        """Dirichlet left, Neumann right."""
        physics, functions, basis = self._setup_problem(["D", "N"])
        self._check_newton_solve(physics, functions, basis)

    def test_bc_dirichlet_robin_1d(self) -> None:
        """Dirichlet left, Robin right."""
        physics, functions, basis = self._setup_problem(["D", "R"])
        self._check_newton_solve(physics, functions, basis)

    def test_bc_robin_dirichlet_1d(self) -> None:
        """Robin left, Dirichlet right."""
        physics, functions, basis = self._setup_problem(["R", "D"])
        self._check_newton_solve(physics, functions, basis)

    def test_bc_residual_at_exact_neumann_1d(self) -> None:
        """Residual at exact solution should be small with Neumann BC."""
        physics, functions, basis = self._setup_problem(["D", "N"])
        exact = _get_exact_displacement(functions, basis, self._bkd)
        state = self._bkd.asarray(exact)
        res = physics.residual(state, 0.0)
        res_norm = float(np.linalg.norm(self._bkd.to_numpy(res)))
        self.assertLess(res_norm, 1e-3)


class TestHyperelasticityBCs1DNumpy(
    TestHyperelasticityBCs1DBase[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


try:
    import torch
    from pyapprox.typing.util.backends.torch import TorchBkd

    class TestHyperelasticityBCs1DTorch(
        TestHyperelasticityBCs1DBase[torch.Tensor]
    ):
        __test__ = True

        def bkd(self) -> TorchBkd:
            return TorchBkd()

except ImportError:
    pass


# =========================================================================
# 2D BC Tests
# =========================================================================


class TestHyperelasticityBCs2DBase(Generic[Array], unittest.TestCase):
    """2D boundary condition tests with non-zero boundary values."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._stress = NeoHookeanStress(1.0, 1.0)
        # Non-zero on right/top: u vanishes at x=0, y=0 but not at x=1, y=1
        self._sol_strs = [
            "0.1*x**2*(1-x)*y**2*(1-y) + 0.02*x*y",
            "0.05*x**2*(1-x)*y**2*(1-y) + 0.01*x*y",
        ]

    def _setup_problem(self, bc_types, nx=12, ny=12, degree=2,
                       robin_alpha=1.0):
        bkd = self._bkd
        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0],
            sol_strs=self._sol_strs,
            stress_model=self._stress,
            bkd=bkd,
        )
        mesh = StructuredMesh2D(
            nx=nx, ny=ny, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd
        )
        basis = VectorLagrangeBasis(mesh, degree=degree)
        adapter = GalerkinHyperelasticityAdapter(basis, functions, bkd)
        body_force = adapter.forcing_for_galerkin()
        bc_set = adapter.create_boundary_conditions(
            bc_types, robin_alpha=robin_alpha
        )
        physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=self._stress,
            bkd=bkd,
            body_force=body_force,
            boundary_conditions=bc_set.all_conditions(),
        )
        return physics, functions, basis

    def _check_newton_solve(self, physics, functions, basis, tol=1e-3):
        exact = _get_exact_displacement(functions, basis, self._bkd)
        solver = SteadyStateSolver(
            physics, tol=1e-10, max_iter=20, line_search=True
        )
        init_guess = self._bkd.asarray(exact + 0.005)
        result = solver.solve(init_guess)
        self.assertTrue(
            result.converged,
            f"Newton did not converge: {result.residual_norm:.2e}",
        )
        u_np = self._bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(exact)
        rel_error = np.linalg.norm(u_np - exact) / max(u_norm, 1e-30)
        self.assertLess(
            rel_error, tol,
            f"Newton solve rel error: {rel_error:.2e}",
        )

    def test_bc_mixed_DN_2d(self) -> None:
        """Dirichlet left/bottom, Neumann right/top."""
        physics, functions, basis = self._setup_problem(
            ["D", "N", "D", "N"]
        )
        self._check_newton_solve(physics, functions, basis)

    def test_bc_mixed_DR_2d(self) -> None:
        """Dirichlet left/bottom, Robin right/top."""
        physics, functions, basis = self._setup_problem(
            ["D", "R", "D", "R"]
        )
        self._check_newton_solve(physics, functions, basis)

    def test_bc_mixed_DNR_2d(self) -> None:
        """Dirichlet left, Neumann right, Dirichlet bottom, Robin top."""
        physics, functions, basis = self._setup_problem(
            ["D", "N", "D", "R"]
        )
        self._check_newton_solve(physics, functions, basis)

    def test_bc_residual_at_exact_mixed_2d(self) -> None:
        """Residual at exact solution should be small with mixed BCs."""
        physics, functions, basis = self._setup_problem(
            ["D", "N", "D", "R"]
        )
        exact = _get_exact_displacement(functions, basis, self._bkd)
        state = self._bkd.asarray(exact)
        res = physics.residual(state, 0.0)
        res_norm = float(np.linalg.norm(self._bkd.to_numpy(res)))
        self.assertLess(res_norm, 1e-3)


class TestHyperelasticityBCs2DNumpy(
    TestHyperelasticityBCs2DBase[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


try:
    import torch
    from pyapprox.typing.util.backends.torch import TorchBkd

    class TestHyperelasticityBCs2DTorch(
        TestHyperelasticityBCs2DBase[torch.Tensor]
    ):
        __test__ = True

        def bkd(self) -> TorchBkd:
            return TorchBkd()

except ImportError:
    pass


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401
