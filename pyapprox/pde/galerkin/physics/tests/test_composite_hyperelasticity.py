"""Tests for CompositeHyperelasticityPhysics.

Tests:
- Uniform material matches HyperelasticityPhysics
- Manufactured solution residual at exact ≈ 0 (1D, 2D)
- Jacobian matches finite differences (1D, 2D)
- Newton solve recovers exact solution (1D, 2D)
- Two-material stiffness differs from uniform
- Small-strain limit matches CompositeLinearElasticity
"""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from scipy.sparse import issparse
from skfem.models.elasticity import lame_parameters
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
from pyapprox.pde.galerkin.physics import (
    CompositeHyperelasticityPhysics,
    CompositeLinearElasticity,
    HyperelasticityPhysics,
)
from pyapprox.pde.galerkin.solvers.steady_state import SteadyStateSolver
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)


def _to_dense(mat, bkd):
    """Convert a matrix (possibly sparse) to a dense numpy array."""
    if issparse(mat):
        return mat.toarray()
    return bkd.to_numpy(mat)


def _make_vector_dirichlet_value_func(sol_func, ndim):
    def value_func(coords, time=0.0):
        nbndry_dofs = coords.shape[1]
        vals = sol_func(coords)
        result = np.zeros(nbndry_dofs)
        for j in range(nbndry_dofs):
            result[j] = vals[j, j % ndim]
        return result

    return value_func


def _get_exact_displacement(funcs, basis, bkd):
    ndim = basis.ncomponents()
    dof_coords = bkd.to_numpy(basis.dof_coordinates())
    n_dofs = basis.ndofs()
    sol = funcs["solution"](dof_coords)
    exact = np.zeros(n_dofs)
    for i in range(n_dofs):
        exact[i] = sol[i, i % ndim]
    return exact


# =========================================================================
# Uniform material matches HyperelasticityPhysics
# =========================================================================


class TestCompositeMatchesUniform:
    """Verify CompositeHyperelasticityPhysics with one material matches
    HyperelasticityPhysics."""
    def test_1d_residual_matches(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        E, nu = 2.0, 0.3
        lam, mu = lame_parameters(E, nu)
        stress = NeoHookeanStress(lam, mu)

        mesh = StructuredMesh1D(nx=5, bounds=(0.0, 1.0), bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)

        uniform_physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=stress,
            bkd=bkd,
        )
        composite_physics = CompositeHyperelasticityPhysics.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        np.random.seed(42)
        n = uniform_physics.nstates()
        state = bkd.asarray(0.01 * np.random.randn(n))

        res_u = bkd.to_numpy(uniform_physics.residual(state, 0.0))
        res_c = bkd.to_numpy(composite_physics.residual(state, 0.0))
        bkd.assert_allclose(
            bkd.asarray(res_c),
            bkd.asarray(res_u),
            atol=1e-12,
        )

    def test_2d_residual_matches(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        E, nu = 1.0, 0.25
        lam, mu = lame_parameters(E, nu)
        stress = NeoHookeanStress(lam, mu)

        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        uniform_physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=stress,
            bkd=bkd,
        )
        composite_physics = CompositeHyperelasticityPhysics.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        np.random.seed(42)
        n = uniform_physics.nstates()
        state = bkd.asarray(0.01 * np.random.randn(n))

        res_u = bkd.to_numpy(uniform_physics.residual(state, 0.0))
        res_c = bkd.to_numpy(composite_physics.residual(state, 0.0))
        bkd.assert_allclose(
            bkd.asarray(res_c),
            bkd.asarray(res_u),
            atol=1e-12,
        )

    def test_2d_jacobian_matches(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        E, nu = 1.0, 0.25
        lam, mu = lame_parameters(E, nu)
        stress = NeoHookeanStress(lam, mu)

        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        uniform_physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=stress,
            bkd=bkd,
        )
        composite_physics = CompositeHyperelasticityPhysics.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        np.random.seed(42)
        n = uniform_physics.nstates()
        state = bkd.asarray(0.01 * np.random.randn(n))

        jac_u = _to_dense(uniform_physics.jacobian(state, 0.0), bkd)
        jac_c = _to_dense(composite_physics.jacobian(state, 0.0), bkd)
        bkd.assert_allclose(
            bkd.asarray(jac_c),
            bkd.asarray(jac_u),
            atol=1e-12,
        )




# =========================================================================
# Manufactured solution tests (1D)
# =========================================================================


class TestCompositeHyperelasticity1D:
    def _setup_1d_problem(self, bkd, E=2.0, nu=0.3, nx=20, degree=2) :
        lam, mu = lame_parameters(E, nu)
        stress = NeoHookeanStress(lam, mu)
        bounds = [0.0, 1.0]
        sol_strs = ["0.1*x**2*(1-x)**2"]

        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=bounds,
            sol_strs=sol_strs,
            stress_model=stress,
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

        physics = CompositeHyperelasticityPhysics.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
            body_force=body_force,
            boundary_conditions=bc_list,
        )
        return physics, functions, basis

    def test_residual_at_exact_1d(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        physics, functions, basis = self._setup_1d_problem(bkd, nx=40, degree=2)
        exact = _get_exact_displacement(functions, basis, bkd)
        state = bkd.asarray(exact)
        res = physics.residual(state, 0.0)
        res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
        assert res_norm < 1e-4

    def test_jacobian_fd_check_1d(self, numpy_bkd) -> None:
        bkd = numpy_bkd
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
            res_pert = bkd.to_numpy(
                physics.residual(bkd.asarray(state_pert), 0.0)
            )
            fd_jac[:, j] = (res_pert - res0) / eps
        rel_err = np.max(np.abs(jac - fd_jac)) / (np.max(np.abs(fd_jac)) + 1e-30)
        assert rel_err < 1e-4

    def test_newton_solve_1d(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        physics, functions, basis = self._setup_1d_problem(bkd, nx=40, degree=2)
        exact = _get_exact_displacement(functions, basis, bkd)

        solver = SteadyStateSolver(physics, tol=1e-10, max_iter=20, line_search=True)
        init_guess = bkd.asarray(exact + 0.01)
        result = solver.solve(init_guess)

        assert result.converged
        u_num = bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(exact)
        rel_error = np.linalg.norm(u_num - exact) / max(u_norm, 1e-10)
        assert rel_error < 1e-6




# =========================================================================
# Manufactured solution tests (2D)
# =========================================================================


class TestCompositeHyperelasticity2D:
    def _setup_2d_problem(self, bkd, E=1.0, nu=0.25, nx=8, ny=8, degree=2) :
        lam, mu = lame_parameters(E, nu)
        stress = NeoHookeanStress(lam, mu)
        bounds = [0.0, 1.0, 0.0, 1.0]
        sol_strs = [
            "0.1*x**2*(1-x)**2*y**2*(1-y)**2",
            "0.05*x**2*(1-x)**2*y**2*(1-y)**2",
        ]

        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=bounds,
            sol_strs=sol_strs,
            stress_model=stress,
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

        physics = CompositeHyperelasticityPhysics.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
            body_force=body_force,
            boundary_conditions=bc_list,
        )
        return physics, functions, basis

    def test_residual_at_exact_2d(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        physics, functions, basis = self._setup_2d_problem(bkd, nx=8, ny=8, degree=2)
        exact = _get_exact_displacement(functions, basis, bkd)
        state = bkd.asarray(exact)
        res = physics.residual(state, 0.0)
        res_norm = float(np.linalg.norm(bkd.to_numpy(res)))
        assert res_norm < 1e-4

    def test_jacobian_fd_check_2d(self, numpy_bkd) -> None:
        bkd = numpy_bkd
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
            res_pert = bkd.to_numpy(
                physics.residual(bkd.asarray(state_pert), 0.0)
            )
            fd_jac[:, j] = (res_pert - res0) / eps
        rel_err = np.max(np.abs(jac - fd_jac)) / (np.max(np.abs(fd_jac)) + 1e-30)
        assert rel_err < 1e-4

    def test_newton_solve_2d(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        physics, functions, basis = self._setup_2d_problem(bkd, nx=12, ny=12, degree=2)
        exact = _get_exact_displacement(functions, basis, bkd)

        solver = SteadyStateSolver(physics, tol=1e-10, max_iter=20, line_search=True)
        init_guess = bkd.asarray(exact + 0.01)
        result = solver.solve(init_guess)

        assert result.converged
        u_num = bkd.to_numpy(result.solution)
        u_norm = np.linalg.norm(exact)
        rel_error = np.linalg.norm(u_num - exact) / max(u_norm, 1e-10)
        assert rel_error < 1e-4




# =========================================================================
# Multi-material tests
# =========================================================================


class TestCompositeMultiMaterial:
    """Tests specific to multi-material composites."""

    def test_two_material_differs_from_uniform(self, numpy_bkd) -> None:
        """Two-material residual differs from single uniform material."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=4,
            ny=4,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        nelems = mesh.nelements()

        # Uniform
        uniform = CompositeHyperelasticityPhysics.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        # Two-material: split left/right halves
        left_elems = np.arange(nelems // 2)
        right_elems = np.arange(nelems // 2, nelems)
        composite = CompositeHyperelasticityPhysics(
            basis=basis,
            material_map={
                "left": (1.0, 0.3),
                "right": (5.0, 0.2),
            },
            element_materials={
                "left": left_elems,
                "right": right_elems,
            },
            bkd=bkd,
        )

        np.random.seed(42)
        n = uniform.nstates()
        state = bkd.asarray(0.01 * np.random.randn(n))

        res_uniform = bkd.to_numpy(uniform.residual(state, 0.0))
        res_composite = bkd.to_numpy(composite.residual(state, 0.0))

        assert not np.allclose(res_uniform, res_composite), "Two-material residual should differ from uniform"

    def test_zero_state_zero_residual(self, numpy_bkd) -> None:
        """With no body force, u=0 gives zero residual (F=I, P=0)."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        nelems = mesh.nelements()

        physics = CompositeHyperelasticityPhysics(
            basis=basis,
            material_map={
                "mat1": (2.0, 0.3),
                "mat2": (5.0, 0.2),
            },
            element_materials={
                "mat1": np.arange(nelems // 2),
                "mat2": np.arange(nelems // 2, nelems),
            },
            bkd=bkd,
        )

        state = bkd.asarray(np.zeros(physics.nstates()))
        res = bkd.to_numpy(physics.residual(state, 0.0))
        np.testing.assert_array_almost_equal(res, 0.0)

    def test_small_strain_matches_linear_elasticity(self, numpy_bkd) -> None:
        """For small strains, Neo-Hookean ≈ linear elasticity."""
        bkd = numpy_bkd
        E, nu = 10.0, 0.3
        mesh = StructuredMesh2D(
            nx=4,
            ny=4,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        hyperelastic = CompositeHyperelasticityPhysics.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )
        linear = CompositeLinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        # Small displacement
        np.random.seed(42)
        n = hyperelastic.nstates()
        state = bkd.asarray(1e-6 * np.random.randn(n))

        res_hyper = bkd.to_numpy(hyperelastic.residual(state, 0.0))
        res_linear = bkd.to_numpy(linear.residual(state, 0.0))

        # Should agree to several digits for small strain
        bkd.assert_allclose(
            bkd.asarray(res_hyper),
            bkd.asarray(res_linear),
            rtol=2e-3,
        )

    def test_nparams_via_parameterization(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        nelems = mesh.nelements()

        physics = CompositeHyperelasticityPhysics(
            basis=basis,
            material_map={"a": (1.0, 0.3), "b": (2.0, 0.2)},
            element_materials={
                "a": np.arange(nelems // 2),
                "b": np.arange(nelems // 2, nelems),
            },
            bkd=bkd,
        )
        param = create_galerkin_lame_parameterization(physics, bkd)
        assert param.nparams() == 4

    def test_apply_changes_residual(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)

        physics = CompositeHyperelasticityPhysics.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )
        param = create_galerkin_lame_parameterization(physics, bkd)

        np.random.seed(42)
        n = physics.nstates()
        state = bkd.asarray(0.01 * np.random.randn(n))
        res1 = bkd.to_numpy(physics.residual(state, 0.0)).copy()

        # Change parameters via parameterization
        param.apply(physics, bkd.asarray(np.array([5.0, 0.2])))
        res2 = bkd.to_numpy(physics.residual(state, 0.0))

        assert not np.allclose(res1, res2)

    def test_poisson_ratio_validation(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(nx=4, bounds=(0.0, 1.0), bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)

        with pytest.raises(ValueError):
            CompositeHyperelasticityPhysics.from_uniform(
                basis=basis,
                youngs_modulus=1.0,
                poisson_ratio=0.5,
                bkd=bkd,
            )

    def test_mass_matrix_symmetric(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        physics = CompositeHyperelasticityPhysics.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )
        M = _to_dense(physics.mass_matrix(), bkd)
        np.testing.assert_array_almost_equal(M, M.T)
