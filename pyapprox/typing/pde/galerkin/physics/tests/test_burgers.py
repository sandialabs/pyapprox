"""Tests for BurgersPhysics.

This module contains:
1. Unit tests for matrix properties (TestBurgersBase)
2. Parametrized steady-state tests (2 legacy cases: DD, NR)
3. Parametrized transient test (1 legacy case: DD with backward Euler)

Legacy test cases from test_finite_elements.py lines 1160-1240.
The periodic case is skipped (known legacy bug).
"""

import unittest
from typing import Generic, Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
from pyapprox.typing.pde.galerkin.physics.burgers import BurgersPhysics
from pyapprox.typing.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.typing.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolutionAdapter,
)
from pyapprox.typing.pde.collocation.manufactured_solutions.burgers import (
    ManufacturedBurgers1D,
)
from pyapprox.typing.pde.galerkin.time_integration import (
    GalerkinPhysicsODEAdapter,
)
from pyapprox.typing.pde.time.implicit_steppers import (
    BackwardEulerResidual,
    CrankNicolsonResidual,
)


# =========================================================================
# Part A: Unit Tests
# =========================================================================


class TestBurgersBase(Generic[Array], unittest.TestCase):
    """Base test class for BurgersPhysics unit tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()
        self.mesh = StructuredMesh1D(
            nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst
        )
        self.basis = LagrangeBasis(self.mesh, degree=2)
        self.physics = BurgersPhysics(
            basis=self.basis,
            viscosity=1.0,
            bkd=self.bkd_inst,
            forcing=lambda x: np.ones(x.shape[-1]),
        )

    def test_nstates(self) -> None:
        """Test DOF count matches basis."""
        self.assertEqual(self.physics.nstates(), self.basis.ndofs())

    def test_residual_shape(self) -> None:
        """Test residual has shape (nstates,)."""
        state = self.bkd_inst.asarray(np.zeros(self.physics.nstates()))
        res = self.physics.residual(state, 0.0)
        self.assertEqual(res.shape, (self.physics.nstates(),))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has shape (nstates, nstates)."""
        state = self.bkd_inst.asarray(np.zeros(self.physics.nstates()))
        jac = self.physics.jacobian(state, 0.0)
        n = self.physics.nstates()
        self.assertEqual(jac.shape, (n, n))

    def test_mass_matrix_shape(self) -> None:
        """Test mass matrix shape."""
        M = self.physics.mass_matrix()
        n = self.physics.nstates()
        self.assertEqual(M.shape, (n, n))

    def test_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric."""
        M = self.bkd_inst.to_numpy(self.physics.mass_matrix())
        np.testing.assert_allclose(M, M.T, atol=1e-14)

    def test_mass_matrix_positive_definite(self) -> None:
        """Test mass matrix is positive definite."""
        M = self.bkd_inst.to_numpy(self.physics.mass_matrix())
        eigvals = np.linalg.eigvalsh(M)
        self.assertTrue(np.all(eigvals > 0))

    def test_is_not_linear(self) -> None:
        """Test is_linear() returns False."""
        self.assertFalse(self.physics.is_linear())


class TestBurgersNumpy(TestBurgersBase[NDArray[Any]]):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


try:
    import torch
    from pyapprox.typing.util.backends.torch import TorchBkd

    class TestBurgersTorch(TestBurgersBase[torch.Tensor]):
        __test__ = True

        def setUp(self) -> None:
            torch.set_default_dtype(torch.float64)
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> TorchBkd:
            return self._bkd

except ImportError:
    pass


# =========================================================================
# Part B: Parametrized Steady-State Tests
# =========================================================================


def _viscosity_for_galerkin(funcs):
    """Create viscosity callable adapted for skfem coordinate shapes."""
    visc_func = funcs["viscosity"]

    def adapted_viscosity(x):
        orig_shape = x.shape
        ndim = orig_shape[0]
        trailing = orig_shape[1:]
        flat = x.reshape(ndim, -1)
        vals = visc_func(flat)
        if hasattr(vals, "shape") and vals.ndim > 1:
            vals = vals[:, 0] if vals.shape[1] == 1 else vals.flatten()
        return vals.reshape(trailing)

    return adapted_viscosity


# Format: (name, bounds, bndry_types, sol_str, visc_str)
BURGERS_STEADY_CASES: List[Tuple[str, List[float], List[str], str, str]] = [
    ("DD", [0.0, 1.0], ["D", "D"], "x*(1.0-x)", "10+1e-16*x"),
    ("NR", [0.0, 1.0], ["N", "R"], "x*(1.0-x)", "10+1e-16*x"),
]


class TestParametrizedBurgersSteady(ParametrizedTestCase):
    """Parametrized 1D Burgers steady-state tests with P2 elements.

    Replicates legacy test_finite_elements.py test_steady_burgers (2 cases).
    Uses manufactured solutions with Newton solver for the nonlinear problem.
    """

    @parametrize(
        "name,bounds,bndry_types,sol_str,visc_str",
        BURGERS_STEADY_CASES,
    )
    def test_steady_burgers(
        self,
        name: str,
        bounds: List[float],
        bndry_types: List[str],
        sol_str: str,
        visc_str: str,
    ) -> None:
        """Test steady-state Burgers with manufactured solution."""
        bkd = NumpyBkd()
        nx = 8 * 2**3  # 64 elements (nrefine=3 with base 8)

        # Create mesh and basis (P2 elements)
        mesh = StructuredMesh1D(
            nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd
        )
        basis = LagrangeBasis(mesh, degree=2)

        # Create manufactured solution
        man_sol = ManufacturedBurgers1D(sol_str, visc_str, bkd, oned=True)
        funcs = man_sol.functions

        # Create adapter and boundary conditions
        adapter = GalerkinManufacturedSolutionAdapter(
            basis, funcs, bkd
        )
        bc_set = adapter.create_boundary_conditions(
            bndry_types, robin_alpha=1.0
        )

        # Get functions for physics
        exact_sol_func = adapter.solution_function()
        forcing_func = adapter.forcing_for_galerkin()
        viscosity_func = _viscosity_for_galerkin(funcs)

        # Create physics
        physics = BurgersPhysics(
            basis=basis,
            viscosity=viscosity_func,
            bkd=bkd,
            forcing=forcing_func,
            boundary_conditions=bc_set.all_conditions(),
        )

        # Initial guess: exact + 1 perturbation
        dof_coords = bkd.to_numpy(basis.dof_coordinates())
        u_exact_vals = exact_sol_func(dof_coords)
        if u_exact_vals.ndim > 1:
            u_exact_vals = (
                u_exact_vals[:, 0]
                if u_exact_vals.shape[1] == 1
                else u_exact_vals.flatten()
            )
        init_guess = bkd.asarray(u_exact_vals + 1.0)

        # Solve with Newton
        solver = SteadyStateSolver(
            physics, tol=1e-12, max_iter=10, line_search=True
        )
        result = solver.solve(init_guess)

        self.assertTrue(
            result.converged,
            f"Test {name}: Newton did not converge "
            f"(residual_norm={result.residual_norm:.2e})",
        )

        # Compute error
        u_num = bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(u_exact_vals)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_vals) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_vals)

        self.assertLess(
            rel_error, 1e-8,
            f"Test {name}: rel_error={rel_error:.2e} should be < 1e-8",
        )


# =========================================================================
# Part C: Parametrized Transient Test
# =========================================================================


# Format: (name, bounds, bndry_types, sol_str, visc_str)
BURGERS_TRANSIENT_CASES: List[
    Tuple[str, List[float], List[str], str, str, str]
] = [
    (
        "DD_BE",
        [0.0, 1.0],
        ["D", "D"],
        "x*(1.0-x)*(1+T)",
        "10+1e-16*x",
        "backward_euler",
    ),
    (
        "DD_CN",
        [0.0, 1.0],
        ["D", "D"],
        "x*(1.0-x)*(1+T)",
        "10+1e-16*x",
        "crank_nicolson",
    ),
]


class TestParametrizedBurgersTransient(ParametrizedTestCase):
    """Parametrized 1D Burgers transient tests with P2 + backward Euler.

    Replicates legacy test_finite_elements.py test_transient_burgers DD case.
    Uses GalerkinPhysicsODEAdapter + BackwardEulerResidual for time stepping
    with Newton iteration at each step.
    """

    @parametrize(
        "name,bounds,bndry_types,sol_str,visc_str,method",
        BURGERS_TRANSIENT_CASES,
    )
    def test_transient_burgers(
        self,
        name: str,
        bounds: List[float],
        bndry_types: List[str],
        sol_str: str,
        visc_str: str,
        method: str,
    ) -> None:
        """Test transient Burgers with manufactured solution."""
        bkd = NumpyBkd()
        nx = 8 * 2**3  # 64 elements

        # Create mesh and basis (P2 elements)
        mesh = StructuredMesh1D(
            nx=nx, bounds=(bounds[0], bounds[1]), bkd=bkd
        )
        basis = LagrangeBasis(mesh, degree=2)

        # Create manufactured solution (time-dependent)
        man_sol = ManufacturedBurgers1D(sol_str, visc_str, bkd, oned=True)
        funcs = man_sol.functions

        # Create adapter with time_dependent=True
        adapter = GalerkinManufacturedSolutionAdapter(
            basis, funcs, bkd, time_dependent=True
        )
        bc_set = adapter.create_boundary_conditions(
            bndry_types, robin_alpha=1.0
        )

        # Get functions for physics
        # Use full forcing (includes du/dT) for transient Galerkin
        forcing_func = adapter.forcing_for_galerkin()
        viscosity_func = _viscosity_for_galerkin(funcs)

        # Create physics with time-dependent forcing and BCs
        physics = BurgersPhysics(
            basis=basis,
            viscosity=viscosity_func,
            bkd=bkd,
            forcing=forcing_func,
            boundary_conditions=bc_set.all_conditions(),
        )

        # Create ODE adapter and time stepper
        ode_adapter = GalerkinPhysicsODEAdapter(physics)
        if method == "backward_euler":
            stepper = BackwardEulerResidual(ode_adapter)
        else:
            stepper = CrankNicolsonResidual(ode_adapter)

        # Initial condition from exact solution at t=0
        exact_sol_func = adapter.solution_function()
        dof_coords = bkd.to_numpy(basis.dof_coordinates())

        def exact_at_time(t):
            u = exact_sol_func(dof_coords, t)
            if hasattr(u, "shape") and u.ndim > 1:
                return u[:, 0] if u.shape[1] == 1 else u.flatten()
            return u

        y = bkd.asarray(exact_at_time(0.0))

        # Time stepping parameters (match legacy)
        dt = 1.0
        nsteps = 5
        t = 0.0

        for step in range(nsteps):
            stepper.set_time(t, dt, y)

            # Newton iteration for this time step
            y_new = bkd.copy(y)
            for newton_iter in range(5):
                res = stepper(y_new)
                res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
                if res_norm < 1e-10:
                    break
                jac = stepper.jacobian(y_new)
                dy = bkd.solve(jac, -res)
                y_new = y_new + dy

            y = y_new
            t += dt

        # Compare to exact at final time
        u_exact_final = exact_at_time(t)
        u_num = bkd.to_numpy(y)

        u_norm = np.linalg.norm(u_exact_final)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact_final) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact_final)

        # Transient tolerance is looser than steady-state due to
        # backward Euler temporal discretization error (dt=1.0)
        self.assertLess(
            rel_error, 1e-6,
            f"Test {name}: rel_error={rel_error:.2e} at t={t} "
            f"should be < 1e-6",
        )


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


if __name__ == "__main__":
    unittest.main()
