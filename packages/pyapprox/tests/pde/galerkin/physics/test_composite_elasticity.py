"""Tests for CompositeLinearElasticity physics."""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


import numpy as np
from scipy.sparse import issparse

from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)


def _to_dense(mat, bkd):
    """Convert a matrix (possibly sparse) to a dense numpy array."""
    if issparse(mat):
        return mat.toarray()
    return bkd.to_numpy(mat)


def _uniform_material(
    basis,
    E,
    nu,
    bkd,
    body_force=None,
    boundary_conditions=None,
):
    """Create CompositeLinearElasticity with a single uniform material."""
    nelems = basis.skfem_basis().mesh.nelements
    return CompositeLinearElasticity(
        basis=basis,
        material_map={"uniform": (E, nu)},
        element_materials={"uniform": np.arange(nelems)},
        bkd=bkd,
        body_force=body_force,
        boundary_conditions=boundary_conditions,
    )


class TestCompositeLinearElasticityBase:
    # ---- Tests matching existing LinearElasticity tests ----

    def test_1d_stiffness_symmetric(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(
            nx=5,
            bounds=(0.0, 1.0),
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_1d_residual_shape(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(
            nx=5,
            bounds=(0.0, 1.0),
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        assert res.shape == (physics.nstates(),)

    def test_1d_rigid_body_motion(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh1D(
            nx=5,
            bounds=(0.0, 1.0),
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        u = bkd.asarray(np.ones(physics.nstates()))
        K = physics.stiffness_matrix()
        Ku = _to_dense(K, bkd) @ bkd.to_numpy(u)
        assert np.linalg.norm(Ku) < 1e-10

    def test_1d_manufactured_solution(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        from skfem.models.elasticity import lame_parameters as _lame

        from pyapprox.pde.galerkin.boundary.implementations import (
            DirichletBC,
        )
        from pyapprox.pde.galerkin.manufactured.adapter import (
            create_elasticity_manufactured_test,
        )

        E, nu = 1.0, 0.3
        lam, mu = _lame(E, nu)

        functions, nvars = create_elasticity_manufactured_test(
            bounds=[0.0, 1.0],
            sol_strs=["0.1*x*(1-x) + 0.2"],
            lambda_str=str(lam),
            mu_str=str(mu),
            bkd=bkd,
        )

        mesh = StructuredMesh1D(
            nx=10,
            bounds=(0.0, 1.0),
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=2)

        sol_func = functions["solution"]
        forcing_func = functions["forcing"]

        def body_force(x, time):
            vals = forcing_func(x)
            return vals.T

        def dirichlet_value(coords, time=0.0):
            vals = sol_func(coords)
            nbndry_dofs = coords.shape[1]
            result = np.zeros(nbndry_dofs)
            for j in range(nbndry_dofs):
                result[j] = vals[j, j % nvars]
            return result

        bc_list = [
            DirichletBC(basis, "left", dirichlet_value, bkd),
            DirichletBC(basis, "right", dirichlet_value, bkd),
        ]
        physics = _uniform_material(
            basis,
            E,
            nu,
            bkd,
            body_force=body_force,
            boundary_conditions=bc_list,
        )

        solver = SteadyStateSolver(physics, tol=1e-12, max_iter=5, line_search=False)
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        result = solver.solve(u0)

        assert result.converged, f"1D did not converge: {result.residual_norm}"

        u_np = bkd.to_numpy(result.solution)
        dof_coords = bkd.to_numpy(basis.dof_coordinates())
        sol_vals = functions["solution"](dof_coords)
        n_dofs = basis.ndofs()
        exact = np.zeros(n_dofs)
        for i in range(n_dofs):
            exact[i] = sol_vals[i, i % nvars]
        rel_error = np.linalg.norm(u_np - exact) / np.linalg.norm(exact)
        assert rel_error < 1e-8

    def test_2d_stiffness_symmetric(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_2d_mass_matrix_symmetric(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        M = physics.mass_matrix()
        M_np = _to_dense(M, bkd)
        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_2d_residual_shape(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        assert res.shape == (physics.nstates(),)

    def test_2d_jacobian_shape(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)
        assert jac.shape == (physics.nstates(), physics.nstates())

    def test_3d_stiffness_symmetric(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_3d_mass_matrix_symmetric(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        M = physics.mass_matrix()
        M_np = _to_dense(M, bkd)
        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_3d_residual_shape(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        assert res.shape == (physics.nstates(),)

    def test_3d_jacobian_shape(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)
        assert jac.shape == (physics.nstates(), physics.nstates())

    def test_2d_with_body_force(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        def body_force(x, time):
            f = np.zeros_like(x)
            f[1, :] = -1.0
            return f

        physics = _uniform_material(
            basis,
            1.0,
            0.3,
            bkd,
            body_force=body_force,
        )
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = bkd.to_numpy(res)
        assert np.linalg.norm(res_np) > 0

    def test_3d_with_body_force(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        def body_force(x, time):
            f = np.zeros_like(x)
            f[2, :] = -1.0
            return f

        physics = _uniform_material(
            basis,
            1.0,
            0.3,
            bkd,
            body_force=body_force,
        )
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = bkd.to_numpy(res)
        assert np.linalg.norm(res_np) > 0

    def test_lame_parameters(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        E = 100.0
        nu = 0.25
        physics = _uniform_material(basis, E, nu, bkd)
        expected_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        expected_mu = E / (2 * (1 + nu))
        assert abs(physics.lame_lambda() - expected_lambda) < 10**(-10)
        assert abs(physics.lame_mu() - expected_mu) < 10**(-10)

    def test_poisson_ratio_validation(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        nelems = basis.skfem_basis().mesh.nelements
        with pytest.raises(ValueError):
            CompositeLinearElasticity(
                basis=basis,
                material_map={"m": (1.0, 0.5)},
                element_materials={"m": np.arange(nelems)},
                bkd=bkd,
            )
        with pytest.raises(ValueError):
            CompositeLinearElasticity(
                basis=basis,
                material_map={"m": (1.0, -1.0)},
                element_materials={"m": np.arange(nelems)},
                bkd=bkd,
            )

    def test_rigid_body_motion_2d(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)

        def u_translation(x):
            result = np.zeros_like(x)
            result[0, :] = 1.0
            result[1, :] = 2.0
            return result

        u = physics.initial_condition(u_translation)
        K = physics.stiffness_matrix()
        Ku = _to_dense(K, bkd) @ bkd.to_numpy(u)
        assert np.linalg.norm(Ku) < 1e-10

    def test_rigid_body_motion_3d(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.25, bkd)

        def u_translation(x):
            result = np.zeros_like(x)
            result[0, :] = 1.0
            result[1, :] = 2.0
            result[2, :] = 3.0
            return result

        u = physics.initial_condition(u_translation)
        K = physics.stiffness_matrix()
        Ku = _to_dense(K, bkd) @ bkd.to_numpy(u)
        assert np.linalg.norm(Ku) < 1e-10

    def test_stiffness_action_consistency(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)

        def u_linear(x):
            return x

        u = physics.initial_condition(u_linear)
        K = physics.stiffness_matrix()
        Ku = _to_dense(K, bkd) @ bkd.to_numpy(u)
        assert Ku.shape == (physics.nstates(),)
        assert np.linalg.norm(Ku) > 0

    def test_stiffness_action_consistency_3d(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.25, bkd)

        def u_linear(x):
            return x

        u = physics.initial_condition(u_linear)
        K = physics.stiffness_matrix()
        Ku = _to_dense(K, bkd) @ bkd.to_numpy(u)
        assert Ku.shape == (physics.nstates(),)
        assert np.linalg.norm(Ku) > 0

    # ---- Multi-material specific tests ----

    def test_two_material_differs_from_uniform(self, numpy_bkd) -> None:
        """Two-material stiffness differs from uniform."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=10,
            ny=5,
            bounds=[[0.0, 2.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        nelems = basis.skfem_basis().mesh.nelements

        # Split into left and right halves
        left_elems = np.arange(nelems // 2)
        right_elems = np.arange(nelems // 2, nelems)

        physics_uniform = _uniform_material(
            basis,
            1.0,
            0.3,
            bkd,
        )
        physics_composite = CompositeLinearElasticity(
            basis=basis,
            material_map={
                "left": (1.0, 0.3),
                "right": (10.0, 0.25),
            },
            element_materials={
                "left": left_elems,
                "right": right_elems,
            },
            bkd=bkd,
        )

        K_uni = _to_dense(physics_uniform.stiffness_matrix(), bkd)
        K_comp = _to_dense(physics_composite.stiffness_matrix(), bkd)
        # Should be different
        assert np.linalg.norm(K_uni - K_comp) > 1e-6
        # But both should be symmetric
        np.testing.assert_allclose(K_comp, K_comp.T, atol=1e-12)

    def test_apply_invalidates_cache(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        param = create_galerkin_lame_parameterization(physics, bkd)

        K1 = _to_dense(physics.stiffness_matrix(), bkd)
        param.apply(physics, bkd.asarray(np.array([2.0, 0.25])))
        K2 = _to_dense(physics.stiffness_matrix(), bkd)
        assert np.linalg.norm(K1 - K2) > 1e-6

    def test_nparams_multi_material(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=10,
            ny=5,
            bounds=[[0.0, 2.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        nelems = basis.skfem_basis().mesh.nelements

        physics = CompositeLinearElasticity(
            basis=basis,
            material_map={
                "mat1": (1.0, 0.3),
                "mat2": (2.0, 0.25),
                "mat3": (0.5, 0.1),
            },
            element_materials={
                "mat1": np.arange(0, nelems // 3),
                "mat2": np.arange(nelems // 3, 2 * nelems // 3),
                "mat3": np.arange(2 * nelems // 3, nelems),
            },
            bkd=bkd,
        )
        param = create_galerkin_lame_parameterization(physics, bkd)
        assert param.nparams() == 6

    # ---- Sensitivity method tests ----

    def test_sensitivity_shape(self, numpy_bkd) -> None:
        """residual_lam/mu_sensitivity return correct shapes."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, bkd)
        n = physics.nstates()
        rng = np.random.RandomState(42)
        u = bkd.asarray(rng.randn(n))

        lam_sens = physics.residual_lam_sensitivity(u, 0)
        mu_sens = physics.residual_mu_sensitivity(u, 0)
        assert lam_sens.shape == (n,)
        assert mu_sens.shape == (n,)

    def test_sensitivity_cross_validate_parameterization(self, numpy_bkd) -> None:
        """Sensitivity methods reproduce parameterization columns via chain rule.

        For material i with params (E, nu):
          col_E  = dlam/dE * lam_sens(i) + dmu/dE * mu_sens(i)
          col_nu = dlam/dnu * lam_sens(i) + dmu/dnu * mu_sens(i)
        """
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 2.0, 0.3, bkd)
        param = create_galerkin_lame_parameterization(physics, bkd)

        rng = np.random.RandomState(42)
        u = bkd.asarray(rng.randn(physics.nstates()))
        params_1d = bkd.asarray(np.array([2.0, 0.3]))

        # Get param_jacobian via parameterization
        pjac = bkd.to_numpy(param.param_jacobian(physics, u, 0.0, params_1d))

        E, nu = 2.0, 0.3
        denom = (1.0 + nu) * (1.0 - 2.0 * nu)

        dLambda_dE = nu / denom
        dMu_dE = 1.0 / (2.0 * (1.0 + nu))
        dLambda_dnu = E * (1.0 + 2.0 * nu**2) / denom**2
        dMu_dnu = -E / (2.0 * (1.0 + nu) ** 2)

        lam_sens = bkd.to_numpy(physics.residual_lam_sensitivity(u, 0))
        mu_sens = bkd.to_numpy(physics.residual_mu_sensitivity(u, 0))

        col_E = dLambda_dE * lam_sens + dMu_dE * mu_sens
        col_nu = dLambda_dnu * lam_sens + dMu_dnu * mu_sens

        np.testing.assert_allclose(
            pjac[:, 0],
            col_E,
            rtol=1e-12,
            err_msg="Sensitivity cross-validation failed for E column",
        )
        np.testing.assert_allclose(
            pjac[:, 1],
            col_nu,
            rtol=1e-12,
            err_msg="Sensitivity cross-validation failed for nu column",
        )

    def test_sensitivity_multi_material(self, numpy_bkd) -> None:
        """Cross-validate sensitivities for multi-material setup."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=10,
            ny=5,
            bounds=[[0.0, 2.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        nelems = basis.skfem_basis().mesh.nelements

        E_vals = [1.0, 5.0]
        nu_vals = [0.3, 0.2]
        left_elems = np.arange(nelems // 2)
        right_elems = np.arange(nelems // 2, nelems)

        physics = CompositeLinearElasticity(
            basis=basis,
            material_map={
                "left": (E_vals[0], nu_vals[0]),
                "right": (E_vals[1], nu_vals[1]),
            },
            element_materials={
                "left": left_elems,
                "right": right_elems,
            },
            bkd=bkd,
        )
        param = create_galerkin_lame_parameterization(physics, bkd)

        rng = np.random.RandomState(42)
        u = bkd.asarray(rng.randn(physics.nstates()))
        params_1d = bkd.asarray(
            np.array([E_vals[0], nu_vals[0], E_vals[1], nu_vals[1]])
        )
        pjac = bkd.to_numpy(param.param_jacobian(physics, u, 0.0, params_1d))

        for i, (E, nu) in enumerate(zip(E_vals, nu_vals)):
            denom = (1.0 + nu) * (1.0 - 2.0 * nu)
            dLambda_dE = nu / denom
            dMu_dE = 1.0 / (2.0 * (1.0 + nu))
            dLambda_dnu = E * (1.0 + 2.0 * nu**2) / denom**2
            dMu_dnu = -E / (2.0 * (1.0 + nu) ** 2)

            lam_sens = bkd.to_numpy(physics.residual_lam_sensitivity(u, i))
            mu_sens = bkd.to_numpy(physics.residual_mu_sensitivity(u, i))

            col_E = dLambda_dE * lam_sens + dMu_dE * mu_sens
            col_nu = dLambda_dnu * lam_sens + dMu_dnu * mu_sens

            np.testing.assert_allclose(
                pjac[:, 2 * i],
                col_E,
                rtol=1e-12,
                err_msg=f"E column mismatch for material {i}",
            )
            np.testing.assert_allclose(
                pjac[:, 2 * i + 1],
                col_nu,
                rtol=1e-12,
                err_msg=f"nu column mismatch for material {i}",
            )

    # ---- Accessor method tests ----

    def test_accessor_methods(self, numpy_bkd) -> None:
        """Test nmaterials, material_names, material_params, element_materials."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=10,
            ny=5,
            bounds=[[0.0, 2.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        nelems = basis.skfem_basis().mesh.nelements
        left_elems = np.arange(nelems // 2)
        right_elems = np.arange(nelems // 2, nelems)

        physics = CompositeLinearElasticity(
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

        assert physics.nmaterials() == 2
        assert physics.material_names() == ["left", "right"]
        assert physics.material_params("left") == (1.0, 0.3)
        assert physics.material_params("right") == (5.0, 0.2)
        elem_mats = physics.element_materials()
        np.testing.assert_array_equal(elem_mats["left"], left_elems)
        np.testing.assert_array_equal(elem_mats["right"], right_elems)

    def test_param_jacobian_fd_check(self, numpy_bkd) -> None:
        """Finite difference check for parameterization param_jacobian."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 2.0, 0.3, bkd)
        parameterization = create_galerkin_lame_parameterization(physics, bkd)

        rng = np.random.RandomState(42)
        u0 = bkd.asarray(rng.randn(physics.nstates()))
        p0 = np.array([2.0, 0.3])
        p0_arr = bkd.asarray(p0)

        pjac = bkd.to_numpy(
            parameterization.param_jacobian(physics, u0, 0.0, p0_arr)
        )

        # FD check
        eps = 1e-7
        for j in range(2):
            p_plus = p0.copy()
            p_plus[j] += eps
            p_minus = p0.copy()
            p_minus[j] -= eps

            parameterization.apply(physics, bkd.asarray(p_plus))
            res_plus = bkd.to_numpy(physics.spatial_residual(u0, 0.0))
            parameterization.apply(physics, bkd.asarray(p_minus))
            res_minus = bkd.to_numpy(physics.spatial_residual(u0, 0.0))

            fd_col = (res_plus - res_minus) / (2 * eps)
            np.testing.assert_allclose(
                pjac[:, j],
                fd_col,
                rtol=1e-5,
                err_msg=f"param_jacobian FD check failed for param {j}",
            )

        # Restore
        parameterization.apply(physics, p0_arr)


