"""Tests for Helmholtz physics.

This module contains:
1. Unit tests for matrix properties (TestHelmholtzBase)
2. Parametrized manufactured solution tests inspired by legacy
   test_finite_elements.py test_helmholtz (3 cases).

The typing Helmholtz class solves the screened Poisson (modified Helmholtz)
equation: -div(grad(u)) + k^2*u = f, with stiffness K = K_lap + k^2*M.
The manufactured solution is created using the ADR machinery with D=1,
no velocity, and reaction R(u) = -k^2*u so that the forcing matches.
"""

import unittest
from typing import Generic, Any, List, Tuple, Callable, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse
from unittest_parametrize import ParametrizedTestCase, parametrize

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import StructuredMesh1D, StructuredMesh2D
from pyapprox.typing.pde.galerkin.basis import LagrangeBasis
from pyapprox.typing.pde.galerkin.physics import Helmholtz
from pyapprox.typing.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.typing.pde.galerkin.manufactured import (
    GalerkinManufacturedSolutionAdapter,
)
from pyapprox.typing.pde.collocation.manufactured_solutions import (
    ManufacturedAdvectionDiffusionReaction,
    ManufacturedHelmholtz,
)


def _create_screened_poisson_manufactured(
    bounds: List[float],
    sol_str: str,
    sqwavenum_str: str,
    bkd: Backend,
) -> Tuple[Dict[str, Callable], Callable, int]:
    """Create manufactured solution for the screened Poisson equation.

    The typing Helmholtz class solves -div(grad(u)) + k^2*u = f.
    The collocation ADR machinery computes forcing for
    -div(D*grad(u)) - R(u) = f. Setting D=1 and R(u) = -k^2*u gives:
        f = -div(grad(u)) - (-k^2*u) = -div(grad(u)) + k^2*u
    which matches the screened Poisson convention.

    Returns
    -------
    functions : Dict
        Manufactured solution functions (solution, forcing, diffusive_flux, etc.)
    sqwavenum_func : Callable
        The squared wavenumber k^2(x) as a callable.
    nvars : int
        Number of spatial dimensions.
    """
    nvars = len(bounds) // 2

    # Build zero velocity strings for all dimensions
    coord_names = ["x", "y", "z"][:nvars]
    vel_strs = [f"1e-16*{c}" for c in coord_names]

    # ADR with D=1, no velocity, reaction = -k^2*u
    # This produces forcing for: -div(grad(u)) + k^2*u = f
    man_sol = ManufacturedAdvectionDiffusionReaction(
        sol_str=sol_str,
        nvars=nvars,
        diff_str="1",
        react_str=f"-({sqwavenum_str})*u",
        vel_strs=vel_strs,
        bkd=bkd,
        oned=True,
    )

    # Get k^2(x) callable from ManufacturedHelmholtz (correct shape handling)
    man_helm = ManufacturedHelmholtz(
        sol_str=sol_str,
        nvars=nvars,
        sqwavenum_str=sqwavenum_str,
        bkd=bkd,
        oned=True,
    )
    sqwavenum_func = man_helm.functions["sqwavenum"]

    return man_sol.functions, sqwavenum_func, nvars


class TestHelmholtzBase(Generic[Array], unittest.TestCase):
    """Base test class for Helmholtz physics."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    def test_1d_mass_matrix_symmetric(self) -> None:
        """Test mass matrix is symmetric in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(
            basis=basis, wavenumber=2*np.pi, bkd=self.bkd_inst
        )

        M = physics.mass_matrix()
        M_np = M.toarray() if issparse(M) else self.bkd_inst.to_numpy(M)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_1d_stiffness_symmetric(self) -> None:
        """Test stiffness matrix is symmetric in 1D."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(
            basis=basis, wavenumber=2*np.pi, bkd=self.bkd_inst
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)
        jac_np = jac.toarray() if issparse(jac) else self.bkd_inst.to_numpy(jac)

        # For Helmholtz, -jacobian = K = K_laplacian + k^2*M, should be symmetric
        K = -jac_np
        np.testing.assert_array_almost_equal(K, K.T, decimal=10)

    def test_1d_residual_shape(self) -> None:
        """Test residual has correct shape."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(
            basis=basis, wavenumber=2*np.pi, bkd=self.bkd_inst
        )

        u0 = physics.initial_condition(lambda x: np.sin(np.pi * x[0]))
        res = physics.residual(u0, 0.0)

        self.assertEqual(res.shape, (physics.nstates(),))

    def test_1d_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(
            basis=basis, wavenumber=2*np.pi, bkd=self.bkd_inst
        )

        u0 = physics.initial_condition(lambda x: np.sin(np.pi * x[0]))
        jac = physics.jacobian(u0, 0.0)

        self.assertEqual(jac.shape, (physics.nstates(), physics.nstates()))

    def test_2d_physics(self) -> None:
        """Test Helmholtz works in 2D."""
        mesh = StructuredMesh2D(
            nx=5, ny=5, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=self.bkd_inst
        )
        basis = LagrangeBasis(mesh, degree=1)
        physics = Helmholtz(
            basis=basis, wavenumber=np.pi, bkd=self.bkd_inst
        )

        # Initial condition
        u0 = physics.initial_condition(
            lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        )

        # Check shapes
        self.assertEqual(u0.shape, (physics.nstates(),))

        res = physics.residual(u0, 0.0)
        self.assertEqual(res.shape, (physics.nstates(),))

    def test_with_forcing(self) -> None:
        """Test Helmholtz with forcing term."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        def forcing(x):
            return np.ones(x.shape[1])

        physics = Helmholtz(
            basis=basis, wavenumber=2*np.pi, forcing=forcing, bkd=self.bkd_inst
        )

        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = self.bkd_inst.to_numpy(res)

        # With forcing and u=0, residual should be non-zero
        self.assertTrue(np.linalg.norm(res_np) > 0)

    def test_wavenumber_property(self) -> None:
        """Test wavenumber property."""
        mesh = StructuredMesh1D(nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)
        k = 3.5
        physics = Helmholtz(basis=basis, wavenumber=k, bkd=self.bkd_inst)

        self.assertEqual(physics.wavenumber(), k)

    def test_steady_state_solve(self) -> None:
        """Test solving steady-state Helmholtz with forcing."""
        mesh = StructuredMesh1D(nx=20, bounds=(0.0, 1.0), bkd=self.bkd_inst)
        basis = LagrangeBasis(mesh, degree=1)

        # Point source in the middle
        def forcing(x):
            return np.exp(-100 * (x[0] - 0.5)**2)

        physics = Helmholtz(
            basis=basis,
            wavenumber=2*np.pi,  # wavelength = 1
            forcing=forcing,
            bkd=self.bkd_inst,
        )

        solver = SteadyStateSolver(physics, tol=1e-10)
        result = solver.solve_linear()

        self.assertTrue(result.converged)
        self.assertLess(result.residual_norm, 1e-8)


class TestHelmholtzNumpy(TestHelmholtzBase[NDArray[Any]]):
    """NumPy backend tests."""

    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# Try to import torch for dual-backend testing
try:
    import torch
    from pyapprox.typing.util.backends.torch import TorchBkd

    class TestHelmholtzTorch(TestHelmholtzBase[torch.Tensor]):
        """PyTorch backend tests."""

        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_steady_state_solve(self) -> None:
            pass

except ImportError:
    pass


# ---------------------------------------------------------------------------
# Parametrized manufactured solution tests
# Replicates the 3 legacy test cases from test_finite_elements.py::test_helmholtz
# ---------------------------------------------------------------------------

# Format: (name, bounds, sol_str, sqwavenum_str, bndry_types)
HELMHOLTZ_MANUFACTURED_CASES: List[
    Tuple[str, List[float], str, str, List[str]]
] = [
    # 1D linear solution, constant k^2, all Dirichlet
    ("1d_lin_DD", [0.0, 1.0], "x", "4+1e-16*x", ["D", "D"]),
    # 1D quadratic solution, x-dependent k^2, Neumann-Robin
    ("1d_quad_NR", [0.0, 1.0], "x**2", "1*x", ["N", "R"]),
    # 2D quartic solution, nearly constant k^2, mixed BCs
    ("2d_x2y2_DDRN", [0.0, 0.5, 0.0, 1.0], "y**2*x**2", "1+1e-16*x",
     ["D", "D", "R", "N"]),
]


class TestParametrizedHelmholtzManufactured(ParametrizedTestCase):
    """Parametrized Helmholtz manufactured solution tests with P2 elements.

    Uses P2 (quadratic) Lagrange elements and manufactured solutions to
    verify exact reproduction of polynomial solutions up to degree 2.

    The typing Helmholtz class solves the screened Poisson equation:
        -div(grad(u)) + k^2*u = f
    The manufactured solution is built using the ADR machinery with D=1,
    no velocity, and reaction R(u) = -k^2*u so that the forcing satisfies
    f = -div(grad(u)) + k^2*u (matching the physics class convention).

    These tests replicate the 3 legacy test cases from
    pyapprox/pde/galerkin/tests/test_finite_elements.py::test_helmholtz.
    """

    @parametrize(
        "name,bounds,sol_str,sqwavenum_str,bndry_types",
        HELMHOLTZ_MANUFACTURED_CASES,
    )
    def test_manufactured_helmholtz(
        self,
        name: str,
        bounds: List[float],
        sol_str: str,
        sqwavenum_str: str,
        bndry_types: List[str],
    ) -> None:
        """Test manufactured solution for Helmholtz equation."""
        bkd = NumpyBkd()

        # Create manufactured solution with sign convention matching
        # the screened Poisson equation: -div(grad(u)) + k^2*u = f
        functions, sqwavenum_func, nvars = _create_screened_poisson_manufactured(
            bounds=bounds,
            sol_str=sol_str,
            sqwavenum_str=sqwavenum_str,
            bkd=bkd,
        )

        # Create mesh and P2 basis
        if nvars == 1:
            mesh = StructuredMesh1D(
                nx=10, bounds=(bounds[0], bounds[1]), bkd=bkd
            )
        else:
            mesh = StructuredMesh2D(
                nx=5, ny=5,
                bounds=[[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
                bkd=bkd,
            )
        basis = LagrangeBasis(mesh, degree=2)  # P2 elements

        # Create adapter and boundary conditions
        adapter = GalerkinManufacturedSolutionAdapter(basis, functions, bkd)
        bc_set = adapter.create_boundary_conditions(bndry_types, robin_alpha=1.0)

        # Get exact solution and forcing adapted for Galerkin
        exact_sol_func = adapter.solution_function()
        forcing_func = adapter.forcing_for_galerkin()

        # Create Helmholtz physics with callable squared wavenumber
        physics = Helmholtz(
            basis=basis,
            wavenumber=sqwavenum_func,
            forcing=forcing_func,
            boundary_conditions=bc_set.all_conditions(),
            bkd=bkd,
        )

        # Solve
        solver = SteadyStateSolver(physics, tol=1e-12)
        result = solver.solve_linear()

        # Compute error at DOF locations
        dof_coords = bkd.to_numpy(basis.dof_coordinates())
        u_num = bkd.to_numpy(result.solution)
        u_exact = exact_sol_func(dof_coords)
        if u_exact.ndim > 1:
            u_exact = (
                u_exact[:, 0] if u_exact.shape[1] == 1 else u_exact.flatten()
            )

        # Compute relative error
        u_norm = np.linalg.norm(u_exact)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - u_exact) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - u_exact)

        # P2 elements should exactly reproduce polynomial solutions ≤ degree 2
        self.assertLess(
            rel_error, 1e-8,
            f"Test {name}: rel_error={rel_error:.2e} should be < 1e-8"
        )


from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


if __name__ == "__main__":
    unittest.main()
