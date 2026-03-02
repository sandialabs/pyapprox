"""Steady-state manufactured solution tests for HyperelasticityPhysics.

Tests:
- Residual at exact solution ≈ 0 (1D, 2D, 3D)
- Jacobian matches finite differences (1D, 2D)
- Newton solve recovers exact solution (1D, 2D)
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


import numpy as np
from scipy.sparse import issparse

from pyapprox.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
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


def _to_dense(mat, bkd):
    """Convert a matrix (possibly sparse) to a dense numpy array."""
    if issparse(mat):
        return mat.toarray()
    return bkd.to_numpy(mat)


# =========================================================================
# Helpers
# =========================================================================


def _make_vector_dirichlet_value_func(sol_func, ndim):
    """Create a DirichletBC value_func for vector basis.

    The manufactured solution returns (npts, ncomponents).
    For interleaved DOFs: DOF j corresponds to component j % ndim.
    """

    def value_func(coords, time=0.0):
        nbndry_dofs = coords.shape[1]
        vals = sol_func(coords)  # (nbndry_dofs, ncomponents)
        result = np.zeros(nbndry_dofs)
        for j in range(nbndry_dofs):
            result[j] = vals[j, j % ndim]
        return result

    return value_func


def _get_exact_displacement(funcs, basis, bkd, time=0.0):
    """Evaluate manufactured solution at DOF locations for vector basis.

    Returns exact DOF values as a flat interleaved array.
    """
    ndim = basis.ncomponents()
    dof_coords = bkd.to_numpy(basis.dof_coordinates())
    n_dofs = basis.ndofs()
    sol = funcs["solution"](dof_coords)  # (ndofs, ncomponents)
    exact = np.zeros(n_dofs)
    for i in range(n_dofs):
        exact[i] = sol[i, i % ndim]
    return exact


# =========================================================================
# 1D Tests
# =========================================================================


class TestHyperelasticity1DBase:
    """Base class for 1D hyperelasticity tests."""

    def _setup(self, bkd) -> None:
        self._stress = NeoHookeanStress(1.0, 1.0)

    def _setup_1d_problem(self, bkd, nx=20, degree=2):
        """Create 1D MMS problem with all-Dirichlet BCs."""
        bounds = [0.0, 1.0]
        sol_strs = ["0.1*x**2*(1-x)**2"]

        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=bounds,
            sol_strs=sol_strs,
            stress_model=self._stress,
            bkd=bkd,
        )

        mesh = StructuredMesh1D(nx=nx, bounds=(0.0, 1.0), bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=degree)

        adapter = GalerkinHyperelasticityAdapter(basis, functions, bkd)
        body_force = adapter.forcing_for_galerkin()
        value_func = _make_vector_dirichlet_value_func(functions["solution"], nvars)
        bc_list = [
            DirichletBC(basis, name, value_func, bkd) for name in ["left", "right"]
        ]

        physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=self._stress,
            bkd=bkd,
            body_force=body_force,
            boundary_conditions=bc_list,
        )
        return physics, functions, basis

    def test_residual_at_exact_solution_1d(self, numpy_bkd) -> None:
        """1D residual should be small at exact manufactured solution."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_1d_problem(bkd, nx=40, degree=2)
        exact = _get_exact_displacement(functions, basis, bkd)
        state = bkd.asarray(exact)
        res = physics.residual(state, 0.0)
        res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
        assert res_norm < 1e-4

    def test_jacobian_fd_check_1d(self, numpy_bkd) -> None:
        """1D analytical Jacobian matches finite differences."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_1d_problem(bkd, nx=10, degree=1)
        n = physics.nstates()
        np.random.seed(42)
        state = bkd.asarray(0.01 * np.random.randn(n))
        jac = _to_dense(physics.jacobian(state, 0.0), bkd)
        res0 = bkd.to_numpy(physics.residual(state, 0.0))
        eps = 1e-7
        fd_jac = np.zeros((n, n))
        state_np = bkd.to_numpy(state)
        for j in range(n):
            state_pert = state_np.copy()
            state_pert[j] += eps
            res_pert = bkd.to_numpy(physics.residual(bkd.asarray(state_pert), 0.0))
            fd_jac[:, j] = (res_pert - res0) / eps
        rel_err = np.max(np.abs(jac - fd_jac)) / (np.max(np.abs(fd_jac)) + 1e-30)
        assert rel_err < 1e-4

    def test_newton_solve_1d(self, numpy_bkd) -> None:
        """1D Newton solve recovers exact manufactured solution."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_1d_problem(bkd, nx=40, degree=2)
        exact = _get_exact_displacement(functions, basis, bkd)

        solver = SteadyStateSolver(physics, tol=1e-10, max_iter=20, line_search=True)
        init_guess = bkd.asarray(exact + 0.01)
        result = solver.solve(init_guess)

        assert result.converged, (
            f"1D Newton did not converge: {result.residual_norm:.2e}"
        )
        u_num = bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(exact)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - exact) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - exact)
        assert rel_error < 1e-6


# =========================================================================
# 2D Tests
# =========================================================================


class TestHyperelasticity2DBase:
    """Base class for 2D hyperelasticity tests."""

    def _setup(self, bkd) -> None:
        self._stress = NeoHookeanStress(1.0, 1.0)

    def _setup_2d_problem(self, bkd, nx=8, ny=8, degree=2):
        """Create 2D MMS problem with all-Dirichlet BCs."""
        bounds = [0.0, 1.0, 0.0, 1.0]
        sol_strs = [
            "0.1*x**2*(1-x)**2*y**2*(1-y)**2",
            "0.05*x**2*(1-x)**2*y**2*(1-y)**2",
        ]

        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=bounds,
            sol_strs=sol_strs,
            stress_model=self._stress,
            bkd=bkd,
        )

        mesh = StructuredMesh2D(
            nx=nx,
            ny=ny,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=degree)

        adapter = GalerkinHyperelasticityAdapter(basis, functions, bkd)
        body_force = adapter.forcing_for_galerkin()
        value_func = _make_vector_dirichlet_value_func(functions["solution"], nvars)
        bc_list = [
            DirichletBC(basis, name, value_func, bkd)
            for name in ["left", "right", "bottom", "top"]
        ]

        physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=self._stress,
            bkd=bkd,
            body_force=body_force,
            boundary_conditions=bc_list,
        )
        return physics, functions, basis

    def test_residual_at_exact_solution_2d(self, numpy_bkd) -> None:
        """2D residual should be small at exact manufactured solution."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_2d_problem(bkd, nx=8, ny=8, degree=2)
        exact = _get_exact_displacement(functions, basis, bkd)
        state = bkd.asarray(exact)
        res = physics.residual(state, 0.0)
        res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
        assert res_norm < 1e-4

    def test_jacobian_fd_check_2d(self, numpy_bkd) -> None:
        """2D analytical Jacobian matches finite differences."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_2d_problem(bkd, nx=3, ny=3, degree=1)
        n = physics.nstates()
        np.random.seed(42)
        state = bkd.asarray(0.01 * np.random.randn(n))
        jac = _to_dense(physics.jacobian(state, 0.0), bkd)
        res0 = bkd.to_numpy(physics.residual(state, 0.0))
        eps = 1e-7
        fd_jac = np.zeros((n, n))
        state_np = bkd.to_numpy(state)
        for j in range(n):
            state_pert = state_np.copy()
            state_pert[j] += eps
            res_pert = bkd.to_numpy(physics.residual(bkd.asarray(state_pert), 0.0))
            fd_jac[:, j] = (res_pert - res0) / eps
        rel_err = np.max(np.abs(jac - fd_jac)) / (np.max(np.abs(fd_jac)) + 1e-30)
        assert rel_err < 1e-4

    def test_newton_solve_2d(self, numpy_bkd) -> None:
        """2D Newton solve recovers exact manufactured solution."""
        bkd = numpy_bkd
        self._setup(bkd)
        physics, functions, basis = self._setup_2d_problem(bkd, nx=12, ny=12, degree=2)
        exact = _get_exact_displacement(functions, basis, bkd)

        solver = SteadyStateSolver(physics, tol=1e-10, max_iter=20, line_search=True)
        init_guess = bkd.asarray(exact + 0.01)
        result = solver.solve(init_guess)

        assert result.converged, (
            f"2D Newton did not converge: {result.residual_norm:.2e}"
        )
        u_num = bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(exact)
        if u_norm > 1e-10:
            rel_error = np.linalg.norm(u_num - exact) / u_norm
        else:
            rel_error = np.linalg.norm(u_num - exact)
        assert rel_error < 1e-4


# =========================================================================
# 3D Tests (residual only — no tangent/Jacobian for 3D)
# =========================================================================


class TestHyperelasticity3DBase:
    """Base class for 3D hyperelasticity residual tests."""

    def _setup(self, bkd) -> None:
        self._stress = NeoHookeanStress(1.0, 1.0)

    @pytest.mark.slow_on("NumpyBkd")
    def test_residual_at_exact_solution_3d(self, numpy_bkd) -> None:
        """3D residual should be small at exact manufactured solution."""
        bkd = numpy_bkd
        self._setup(bkd)
        from pyapprox.pde.galerkin.mesh import StructuredMesh3D

        bounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        sol_strs = [
            "0.1*x**2*(1-x)**2*y**2*(1-y)**2*z**2*(1-z)**2",
            "0.05*x**2*(1-x)**2*y**2*(1-y)**2*z**2*(1-z)**2",
            "0.02*x**2*(1-x)**2*y**2*(1-y)**2*z**2*(1-z)**2",
        ]

        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=bounds,
            sol_strs=sol_strs,
            stress_model=self._stress,
            bkd=bkd,
        )

        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        adapter = GalerkinHyperelasticityAdapter(basis, functions, bkd)
        body_force = adapter.forcing_for_galerkin()
        value_func = _make_vector_dirichlet_value_func(functions["solution"], nvars)
        bc_names = ["left", "right", "bottom", "top", "front", "back"]
        bc_list = [DirichletBC(basis, name, value_func, bkd) for name in bc_names]

        physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=self._stress,
            bkd=bkd,
            body_force=body_force,
            boundary_conditions=bc_list,
        )

        exact = _get_exact_displacement(functions, basis, bkd)
        state = bkd.asarray(exact)
        res = physics.residual(state, 0.0)
        res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
        assert res_norm < 1e-3


class TestHyperelasticityShapes:
    """Basic shape and property tests."""

    def test_1d_shapes(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        bkd = NumpyBkd()
        stress = NeoHookeanStress(1.0, 1.0)
        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = HyperelasticityPhysics(basis, stress, bkd)
        n = physics.nstates()
        state = bkd.asarray(np.zeros(n))
        assert physics.residual(state, 0.0).shape == (n,)
        assert physics.jacobian(state, 0.0).shape == (n, n)
        assert physics.mass_matrix().shape == (n, n)
        assert physics.ndim() == 1

    def test_2d_shapes(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        bkd = NumpyBkd()
        stress = NeoHookeanStress(1.0, 1.0)
        mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = HyperelasticityPhysics(basis, stress, bkd)
        n = physics.nstates()
        state = bkd.asarray(np.zeros(n))
        assert physics.residual(state, 0.0).shape == (n,)
        assert physics.jacobian(state, 0.0).shape == (n, n)
        assert physics.mass_matrix().shape == (n, n)
        assert physics.ndim() == 2

    def test_mass_matrix_symmetric(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        bkd = NumpyBkd()
        stress = NeoHookeanStress(1.0, 1.0)
        mesh = StructuredMesh2D(nx=4, ny=4, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = HyperelasticityPhysics(basis, stress, bkd)
        M = _to_dense(physics.mass_matrix(), bkd)
        np.testing.assert_array_almost_equal(M, M.T)

    def test_zero_state_zero_residual(self, numpy_bkd) -> None:
        """With no body force, u=0 gives zero residual (F=I, P=0)."""
        bkd = numpy_bkd
        bkd = NumpyBkd()
        stress = NeoHookeanStress(1.0, 1.0)
        mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = HyperelasticityPhysics(basis, stress, bkd)
        state = bkd.asarray(np.zeros(physics.nstates()))
        res = bkd.to_numpy(physics.residual(state, 0.0))
        np.testing.assert_array_almost_equal(res, 0.0)

    def test_tangent_not_available_3d(self, numpy_bkd) -> None:
        """3D tangent stiffness raises NotImplementedError."""
        bkd = numpy_bkd
        from pyapprox.pde.galerkin.mesh import StructuredMesh3D

        bkd = NumpyBkd()
        stress = NeoHookeanStress(1.0, 1.0)
        mesh = StructuredMesh3D(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = HyperelasticityPhysics(basis, stress, bkd)
        state = bkd.asarray(np.zeros(physics.nstates()))
        with pytest.raises(NotImplementedError):
            physics.jacobian(state, 0.0)

    def test_repr(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        bkd = NumpyBkd()
        stress = NeoHookeanStress(1.0, 1.0)
        mesh = StructuredMesh1D(nx=3, bounds=(0.0, 1.0), bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = HyperelasticityPhysics(basis, stress, bkd)
        r = repr(physics)
        assert "HyperelasticityPhysics" in r
        assert "NeoHookeanStress" in r
