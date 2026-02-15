"""Tests for CompositeLinearElasticity physics."""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
    StructuredMesh3D,
)
from pyapprox.typing.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.typing.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.typing.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.typing.util.test_utils import load_tests  # noqa: F401


def _uniform_material(
    basis, E, nu, bkd, body_force=None, boundary_conditions=None,
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


class TestCompositeLinearElasticityBase(
    Generic[Array], unittest.TestCase
):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    # ---- Tests matching existing LinearElasticity tests ----

    def test_1d_stiffness_symmetric(self) -> None:
        mesh = StructuredMesh1D(
            nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)
        K = physics.stiffness_matrix()
        K_np = self.bkd_inst.to_numpy(K)
        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_1d_residual_shape(self) -> None:
        mesh = StructuredMesh1D(
            nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)
        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        self.assertEqual(res.shape, (physics.nstates(),))

    def test_1d_rigid_body_motion(self) -> None:
        mesh = StructuredMesh1D(
            nx=5, bounds=(0.0, 1.0), bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)
        u = self.bkd_inst.asarray(np.ones(physics.nstates()))
        K = physics.stiffness_matrix()
        Ku = self.bkd_inst.to_numpy(K) @ self.bkd_inst.to_numpy(u)
        self.assertLess(np.linalg.norm(Ku), 1e-10)

    def test_1d_manufactured_solution(self) -> None:
        from skfem.models.elasticity import lame_parameters as _lame
        from pyapprox.typing.pde.galerkin.boundary.implementations import (
            DirichletBC,
        )
        from pyapprox.typing.pde.galerkin.manufactured.adapter import (
            create_elasticity_manufactured_test,
        )

        E, nu = 1.0, 0.3
        lam, mu = _lame(E, nu)

        functions, nvars = create_elasticity_manufactured_test(
            bounds=[0.0, 1.0],
            sol_strs=["0.1*x*(1-x) + 0.2"],
            lambda_str=str(lam),
            mu_str=str(mu),
            bkd=self.bkd_inst,
        )

        mesh = StructuredMesh1D(
            nx=10, bounds=(0.0, 1.0), bkd=self.bkd_inst,
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
            DirichletBC(basis, "left", dirichlet_value, self.bkd_inst),
            DirichletBC(basis, "right", dirichlet_value, self.bkd_inst),
        ]
        physics = _uniform_material(
            basis, E, nu, self.bkd_inst,
            body_force=body_force,
            boundary_conditions=bc_list,
        )

        solver = SteadyStateSolver(
            physics, tol=1e-12, max_iter=5, line_search=False
        )
        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        result = solver.solve(u0)

        self.assertTrue(
            result.converged,
            f"1D did not converge: {result.residual_norm}",
        )

        u_np = self.bkd_inst.to_numpy(result.solution)
        dof_coords = self.bkd_inst.to_numpy(basis.dof_coordinates())
        sol_vals = functions["solution"](dof_coords)
        n_dofs = basis.ndofs()
        exact = np.zeros(n_dofs)
        for i in range(n_dofs):
            exact[i] = sol_vals[i, i % nvars]
        rel_error = np.linalg.norm(u_np - exact) / np.linalg.norm(exact)
        self.assertLess(
            rel_error, 1e-8,
            f"1D manufactured solution rel error: {rel_error:.2e}",
        )

    def test_2d_stiffness_symmetric(self) -> None:
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)
        K = physics.stiffness_matrix()
        K_np = self.bkd_inst.to_numpy(K)
        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_2d_mass_matrix_symmetric(self) -> None:
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)
        M = physics.mass_matrix()
        M_np = self.bkd_inst.to_numpy(M)
        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_2d_residual_shape(self) -> None:
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)
        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        self.assertEqual(res.shape, (physics.nstates(),))

    def test_2d_jacobian_shape(self) -> None:
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)
        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)
        self.assertEqual(jac.shape, (physics.nstates(), physics.nstates()))

    def test_3d_stiffness_symmetric(self) -> None:
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)
        K = physics.stiffness_matrix()
        K_np = self.bkd_inst.to_numpy(K)
        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_3d_mass_matrix_symmetric(self) -> None:
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)
        M = physics.mass_matrix()
        M_np = self.bkd_inst.to_numpy(M)
        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_3d_residual_shape(self) -> None:
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)
        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        self.assertEqual(res.shape, (physics.nstates(),))

    def test_3d_jacobian_shape(self) -> None:
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)
        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)
        self.assertEqual(jac.shape, (physics.nstates(), physics.nstates()))

    def test_2d_with_body_force(self) -> None:
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        def body_force(x, time):
            f = np.zeros_like(x)
            f[1, :] = -1.0
            return f

        physics = _uniform_material(
            basis, 1.0, 0.3, self.bkd_inst, body_force=body_force,
        )
        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = self.bkd_inst.to_numpy(res)
        self.assertTrue(np.linalg.norm(res_np) > 0)

    def test_3d_with_body_force(self) -> None:
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        def body_force(x, time):
            f = np.zeros_like(x)
            f[2, :] = -1.0
            return f

        physics = _uniform_material(
            basis, 1.0, 0.3, self.bkd_inst, body_force=body_force,
        )
        u0 = self.bkd_inst.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = self.bkd_inst.to_numpy(res)
        self.assertTrue(np.linalg.norm(res_np) > 0)

    def test_lame_parameters(self) -> None:
        mesh = StructuredMesh2D(
            nx=3, ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        E = 100.0
        nu = 0.25
        physics = _uniform_material(basis, E, nu, self.bkd_inst)
        expected_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        expected_mu = E / (2 * (1 + nu))
        self.assertAlmostEqual(
            physics.lame_lambda, expected_lambda, places=10
        )
        self.assertAlmostEqual(physics.lame_mu, expected_mu, places=10)

    def test_poisson_ratio_validation(self) -> None:
        mesh = StructuredMesh2D(
            nx=3, ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        nelems = basis.skfem_basis().mesh.nelements
        with self.assertRaises(ValueError):
            CompositeLinearElasticity(
                basis=basis,
                material_map={"m": (1.0, 0.5)},
                element_materials={"m": np.arange(nelems)},
                bkd=self.bkd_inst,
            )
        with self.assertRaises(ValueError):
            CompositeLinearElasticity(
                basis=basis,
                material_map={"m": (1.0, -1.0)},
                element_materials={"m": np.arange(nelems)},
                bkd=self.bkd_inst,
            )

    def test_rigid_body_motion_2d(self) -> None:
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)

        def u_translation(x):
            result = np.zeros_like(x)
            result[0, :] = 1.0
            result[1, :] = 2.0
            return result

        u = physics.initial_condition(u_translation)
        K = physics.stiffness_matrix()
        Ku = self.bkd_inst.to_numpy(K) @ self.bkd_inst.to_numpy(u)
        self.assertLess(np.linalg.norm(Ku), 1e-10)

    def test_rigid_body_motion_3d(self) -> None:
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.25, self.bkd_inst)

        def u_translation(x):
            result = np.zeros_like(x)
            result[0, :] = 1.0
            result[1, :] = 2.0
            result[2, :] = 3.0
            return result

        u = physics.initial_condition(u_translation)
        K = physics.stiffness_matrix()
        Ku = self.bkd_inst.to_numpy(K) @ self.bkd_inst.to_numpy(u)
        self.assertLess(np.linalg.norm(Ku), 1e-10)

    def test_stiffness_action_consistency(self) -> None:
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)

        def u_linear(x):
            return x

        u = physics.initial_condition(u_linear)
        K = physics.stiffness_matrix()
        Ku = self.bkd_inst.to_numpy(K) @ self.bkd_inst.to_numpy(u)
        self.assertEqual(Ku.shape, (physics.nstates(),))
        self.assertTrue(np.linalg.norm(Ku) > 0)

    def test_stiffness_action_consistency_3d(self) -> None:
        mesh = StructuredMesh3D(
            nx=3, ny=3, nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.25, self.bkd_inst)

        def u_linear(x):
            return x

        u = physics.initial_condition(u_linear)
        K = physics.stiffness_matrix()
        Ku = self.bkd_inst.to_numpy(K) @ self.bkd_inst.to_numpy(u)
        self.assertEqual(Ku.shape, (physics.nstates(),))
        self.assertTrue(np.linalg.norm(Ku) > 0)

    # ---- Multi-material specific tests ----

    def test_two_material_differs_from_uniform(self) -> None:
        """Two-material stiffness differs from uniform."""
        mesh = StructuredMesh2D(
            nx=10, ny=5,
            bounds=[[0.0, 2.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        nelems = basis.skfem_basis().mesh.nelements

        # Split into left and right halves
        left_elems = np.arange(nelems // 2)
        right_elems = np.arange(nelems // 2, nelems)

        physics_uniform = _uniform_material(
            basis, 1.0, 0.3, self.bkd_inst,
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
            bkd=self.bkd_inst,
        )

        K_uni = self.bkd_inst.to_numpy(physics_uniform.stiffness_matrix())
        K_comp = self.bkd_inst.to_numpy(
            physics_composite.stiffness_matrix()
        )
        # Should be different
        self.assertTrue(np.linalg.norm(K_uni - K_comp) > 1e-6)
        # But both should be symmetric
        np.testing.assert_allclose(K_comp, K_comp.T, atol=1e-12)

    def test_set_param_invalidates_cache(self) -> None:
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 1.0, 0.3, self.bkd_inst)

        K1 = self.bkd_inst.to_numpy(physics.stiffness_matrix())
        physics.set_param(
            self.bkd_inst.asarray(np.array([2.0, 0.25]))
        )
        K2 = self.bkd_inst.to_numpy(physics.stiffness_matrix())
        self.assertTrue(np.linalg.norm(K1 - K2) > 1e-6)

    def test_nparams_multi_material(self) -> None:
        mesh = StructuredMesh2D(
            nx=10, ny=5,
            bounds=[[0.0, 2.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
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
            bkd=self.bkd_inst,
        )
        self.assertEqual(physics.nparams(), 6)

    def test_param_jacobian_fd_check(self) -> None:
        """Finite difference check for param_jacobian."""
        mesh = StructuredMesh2D(
            nx=5, ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=self.bkd_inst,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = _uniform_material(basis, 2.0, 0.3, self.bkd_inst)

        rng = np.random.RandomState(42)
        u0 = self.bkd_inst.asarray(rng.randn(physics.nstates()))
        p0 = np.array([2.0, 0.3])

        pjac = self.bkd_inst.to_numpy(physics.param_jacobian(u0, 0.0))

        # FD check
        eps = 1e-7
        for j in range(2):
            p_plus = p0.copy()
            p_plus[j] += eps
            p_minus = p0.copy()
            p_minus[j] -= eps

            physics.set_param(self.bkd_inst.asarray(p_plus))
            res_plus = self.bkd_inst.to_numpy(
                physics.spatial_residual(u0, 0.0)
            )
            physics.set_param(self.bkd_inst.asarray(p_minus))
            res_minus = self.bkd_inst.to_numpy(
                physics.spatial_residual(u0, 0.0)
            )

            fd_col = (res_plus - res_minus) / (2 * eps)
            np.testing.assert_allclose(
                pjac[:, j], fd_col, rtol=1e-5,
                err_msg=f"param_jacobian FD check failed for param {j}",
            )

        # Restore
        physics.set_param(self.bkd_inst.asarray(p0))


class TestCompositeLinearElasticityNumpy(
    TestCompositeLinearElasticityBase[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


try:
    import torch
    from pyapprox.typing.util.backends.torch import TorchBkd

    class TestCompositeLinearElasticityTorch(
        TestCompositeLinearElasticityBase[torch.Tensor]
    ):
        __test__ = True

        def setUp(self) -> None:
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
