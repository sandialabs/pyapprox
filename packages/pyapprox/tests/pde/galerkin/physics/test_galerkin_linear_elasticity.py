"""Tests for LinearElasticity physics."""

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


def _to_dense(mat, bkd):
    """Convert a matrix (possibly sparse) to a dense numpy array."""
    if issparse(mat):
        return mat.toarray()
    return bkd.to_numpy(mat)


# Alias for backward compatibility in tests
LinearElasticity = CompositeLinearElasticity


class TestLinearElasticityBase:
    """Base test class for LinearElasticity."""
    def test_1d_stiffness_symmetric(self, numpy_bkd) -> None:
        """Test stiffness matrix is symmetric in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(
            nx=5,
            bounds=(0.0, 1.0),
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_1d_residual_shape(self, numpy_bkd) -> None:
        """Test residual has correct shape in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(
            nx=5,
            bounds=(0.0, 1.0),
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        assert res.shape == (physics.nstates(),)

    def test_1d_rigid_body_motion(self, numpy_bkd) -> None:
        """Test that constant displacement gives zero stiffness action in 1D."""
        bkd = numpy_bkd
        mesh = StructuredMesh1D(
            nx=5,
            bounds=(0.0, 1.0),
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )
        u = bkd.asarray(np.ones(physics.nstates()))
        K = physics.stiffness_matrix()
        Ku = _to_dense(K, bkd) @ bkd.to_numpy(u)
        assert np.linalg.norm(Ku) < 1e-10

    def test_1d_manufactured_solution(self, numpy_bkd) -> None:
        """1D solve recovers manufactured solution with non-zero BCs.

        Uses create_elasticity_manufactured_test for MMS forcing.
        u(x) = 0.1*x*(1-x) + 0.2 (degree-2, non-zero at boundaries).
        With degree-2 elements the FEM solution is exact.

        Exercises VectorLagrangeBasis.get_dofs() in 1D — regression test
        for a bug where 1D DOFs were returned empty.
        """
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
            vals = forcing_func(x)  # (npts, 1)
            return vals.T  # (1, npts) = (ndim, npts)

        def dirichlet_value(coords, time=0.0):
            vals = sol_func(coords)  # (nbndry_dofs, 1)
            nbndry_dofs = coords.shape[1]
            result = np.zeros(nbndry_dofs)
            for j in range(nbndry_dofs):
                result[j] = vals[j, j % nvars]
            return result

        bc_list = [
            DirichletBC(basis, "left", dirichlet_value, bkd),
            DirichletBC(basis, "right", dirichlet_value, bkd),
        ]
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            body_force=body_force,
            boundary_conditions=bc_list,
            bkd=bkd,
        )

        solver = SteadyStateSolver(physics, tol=1e-12, max_iter=5, line_search=False)
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        result = solver.solve(u0)

        assert result.converged, f"1D did not converge: {result.residual_norm}"

        u_np = bkd.to_numpy(result.solution)
        dof_coords = bkd.to_numpy(basis.dof_coordinates())
        sol_vals = functions["solution"](dof_coords)  # (ndofs, 1)
        n_dofs = basis.ndofs()
        exact = np.zeros(n_dofs)
        for i in range(n_dofs):
            exact[i] = sol_vals[i, i % nvars]
        rel_error = np.linalg.norm(u_np - exact) / np.linalg.norm(exact)
        assert rel_error < 1e-8

    def test_2d_stiffness_symmetric(self, numpy_bkd) -> None:
        """Test stiffness matrix is symmetric in 2D."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)

        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_2d_mass_matrix_symmetric(self, numpy_bkd) -> None:
        """Test mass matrix is symmetric in 2D."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        M = physics.mass_matrix()
        M_np = _to_dense(M, bkd)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_2d_residual_shape(self, numpy_bkd) -> None:
        """Test residual has correct shape in 2D."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        u0 = bkd.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)

        assert res.shape == (physics.nstates(),)

    def test_2d_jacobian_shape(self, numpy_bkd) -> None:
        """Test Jacobian has correct shape in 2D."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        u0 = bkd.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)

        assert jac.shape == (physics.nstates(), physics.nstates())

    def test_3d_stiffness_symmetric(self, numpy_bkd) -> None:
        """Test stiffness matrix is symmetric in 3D."""
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)

        np.testing.assert_array_almost_equal(K_np, K_np.T)

    def test_3d_mass_matrix_symmetric(self, numpy_bkd) -> None:
        """Test mass matrix is symmetric in 3D."""
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        M = physics.mass_matrix()
        M_np = _to_dense(M, bkd)

        np.testing.assert_array_almost_equal(M_np, M_np.T)

    def test_3d_residual_shape(self, numpy_bkd) -> None:
        """Test residual has correct shape in 3D."""
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        u0 = bkd.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)

        assert res.shape == (physics.nstates(),)

    def test_3d_jacobian_shape(self, numpy_bkd) -> None:
        """Test Jacobian has correct shape in 3D."""
        bkd = numpy_bkd
        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            bkd=bkd,
        )

        u0 = bkd.asarray(np.zeros(physics.nstates()))
        jac = physics.jacobian(u0, 0.0)

        assert jac.shape == (physics.nstates(), physics.nstates())

    def test_2d_with_body_force(self, numpy_bkd) -> None:
        """Test 2D elasticity with body force."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        def body_force(x, time):
            # Constant gravity-like force in y direction
            f = np.zeros_like(x)
            f[1, :] = -1.0
            return f

        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            body_force=body_force,
            bkd=bkd,
        )

        u0 = bkd.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = bkd.to_numpy(res)

        # With body force and u=0, residual should be non-zero
        assert np.linalg.norm(res_np) > 0

    def test_3d_with_body_force(self, numpy_bkd) -> None:
        """Test 3D elasticity with body force."""
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
            # Constant gravity-like force in z direction
            f = np.zeros_like(x)
            f[2, :] = -1.0
            return f

        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            body_force=body_force,
            bkd=bkd,
        )

        u0 = bkd.asarray(np.zeros(physics.nstates()))
        res = physics.residual(u0, 0.0)
        res_np = bkd.to_numpy(res)

        # With body force and u=0, residual should be non-zero
        assert np.linalg.norm(res_np) > 0

    def test_lame_parameters(self, numpy_bkd) -> None:
        """Test Lame parameter computation."""
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

        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        # Expected Lame parameters
        # lambda = E * nu / ((1 + nu) * (1 - 2*nu))
        # mu = E / (2 * (1 + nu))
        expected_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        expected_mu = E / (2 * (1 + nu))

        assert abs(physics.lame_lambda() - expected_lambda) < 10**(-10)
        assert abs(physics.lame_mu() - expected_mu) < 10**(-10)

    def test_poisson_ratio_validation(self, numpy_bkd) -> None:
        """Test that invalid Poisson ratios raise errors."""
        bkd = numpy_bkd
        mesh = StructuredMesh2D(
            nx=3,
            ny=3,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        # nu = 0.5 is invalid (incompressible limit)
        with pytest.raises(ValueError):
            LinearElasticity.from_uniform(
                basis=basis,
                youngs_modulus=1.0,
                poisson_ratio=0.5,
                bkd=bkd,
            )

        # nu = -1 is invalid
        with pytest.raises(ValueError):
            LinearElasticity.from_uniform(
                basis=basis,
                youngs_modulus=1.0,
                poisson_ratio=-1.0,
                bkd=bkd,
            )

    def test_stiffness_action_consistency(self, numpy_bkd) -> None:
        """Test that K*u gives expected result for constant strain field.

        For u = [ax, ay] (uniform scaling), K*u should represent the
        boundary traction integrals since -div(sigma) = 0 in the interior.
        """
        bkd = numpy_bkd
        E = 1.0  # Young's modulus
        nu = 0.3  # Poisson's ratio

        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        # Linear displacement u = [x, y] -> constant strain
        def u_linear(x):
            return x

        u = physics.initial_condition(u_linear)
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        u_np = bkd.to_numpy(u)

        # K*u should be consistent (non-zero due to boundary tractions)
        Ku = K_np @ u_np
        assert Ku.shape == (physics.nstates(),)
        # For uniform strain, the response should be consistent
        assert np.linalg.norm(Ku) > 0

    def test_stiffness_action_consistency_3d(self, numpy_bkd) -> None:
        """Test that K*u gives expected result for constant strain field in 3D."""
        bkd = numpy_bkd
        E = 1.0  # Young's modulus
        nu = 0.25  # Poisson's ratio

        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        # Linear displacement u = [x, y, z] -> constant strain
        def u_linear(x):
            return x

        u = physics.initial_condition(u_linear)
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        u_np = bkd.to_numpy(u)

        # K*u should be consistent (non-zero due to boundary tractions)
        Ku = K_np @ u_np
        assert Ku.shape == (physics.nstates(),)
        # For uniform strain, the response should be consistent
        assert np.linalg.norm(Ku) > 0

    def test_rigid_body_motion_2d(self, numpy_bkd) -> None:
        """Test that rigid body motion gives zero strain energy.

        Pure translation u = [c, c] should give zero K*u.
        """
        bkd = numpy_bkd
        E = 1.0
        nu = 0.3

        mesh = StructuredMesh2D(
            nx=5,
            ny=5,
            bounds=[[0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        # Constant translation
        def u_translation(x):
            result = np.zeros_like(x)
            result[0, :] = 1.0  # Translate in x
            result[1, :] = 2.0  # Translate in y
            return result

        u = physics.initial_condition(u_translation)
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        u_np = bkd.to_numpy(u)

        # K*u should be zero for rigid body motion
        Ku = K_np @ u_np
        assert np.linalg.norm(Ku) < 1e-10

    def test_rigid_body_motion_3d(self, numpy_bkd) -> None:
        """Test that rigid body motion gives zero strain energy in 3D."""
        bkd = numpy_bkd
        E = 1.0
        nu = 0.25

        mesh = StructuredMesh3D(
            nx=3,
            ny=3,
            nz=3,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=E,
            poisson_ratio=nu,
            bkd=bkd,
        )

        # Constant translation
        def u_translation(x):
            result = np.zeros_like(x)
            result[0, :] = 1.0
            result[1, :] = 2.0
            result[2, :] = 3.0
            return result

        u = physics.initial_condition(u_translation)
        K = physics.stiffness_matrix()
        K_np = _to_dense(K, bkd)
        u_np = bkd.to_numpy(u)

        # K*u should be zero for rigid body motion
        Ku = K_np @ u_np
        assert np.linalg.norm(Ku) < 1e-10




class TestLinearElasticity3DPatch:
    """3D patch tests: linear displacement => exact recovery.

    A linear displacement field u = A x + b produces constant strain and
    stress, so with constant coefficients the forcing is zero and the FEM
    solution is exact for any degree. Nonzero Dirichlet data on every face
    exercises VectorLagrangeBasis.get_dofs in 3D — the regression that
    duplicated boundary DOFs was invisible to vanishing-boundary tests.
    """

    @pytest.mark.parametrize("element_type", ["hex", "tet"])
    @pytest.mark.parametrize("degree", [1, 2])
    def test_3d_patch(self, numpy_bkd, element_type, degree) -> None:
        bkd = numpy_bkd
        A = np.array(
            [
                [0.10, 0.02, -0.03],
                [0.04, -0.08, 0.01],
                [-0.02, 0.05, 0.06],
            ]
        )
        b = np.array([0.2, -0.1, 0.3])

        def exact_displacement(x):
            # (3, npts): also the DirichletBC vector value convention
            return A @ x + b[:, None]

        mesh = StructuredMesh3D(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
            element_type=element_type,
        )
        basis = VectorLagrangeBasis(mesh, degree=degree)

        from pyapprox.pde.galerkin.boundary.implementations import (
            DirichletBC,
        )

        def dirichlet_value(coords, time=0.0):
            return exact_displacement(coords)

        bc_list = [
            DirichletBC(basis, name, dirichlet_value, bkd)
            for name in ["left", "right", "bottom", "top", "front", "back"]
        ]
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.3,
            boundary_conditions=bc_list,
            bkd=bkd,
        )

        solver = SteadyStateSolver(
            physics, tol=1e-12, max_iter=5, line_search=False
        )
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        result = solver.solve(u0)
        assert result.converged

        u_np = bkd.to_numpy(result.solution)
        dof_coords = bkd.to_numpy(basis.dof_coordinates())
        vals = exact_displacement(dof_coords)  # (3, ndofs)
        ndofs = basis.ndofs()
        exact = vals[np.arange(ndofs) % 3, np.arange(ndofs)]
        rel_error = np.linalg.norm(u_np - exact) / np.linalg.norm(exact)
        assert rel_error < 1e-10

    def test_3d_component_dirichlet_roller(self, numpy_bkd) -> None:
        """Roller/symmetry BCs constraining single components.

        With nu=0 the field u = (0, 0, alpha*z) satisfies traction-free
        lateral faces exactly, so rollers (u_x=0 on left, u_y=0 on
        bottom, u_z=0 on front) plus u_z=alpha on back recover it.
        """
        bkd = numpy_bkd
        alpha = 0.1

        mesh = StructuredMesh3D(
            nx=2,
            ny=2,
            nz=2,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
        )
        basis = VectorLagrangeBasis(mesh, degree=1)

        from pyapprox.pde.galerkin.boundary.implementations import (
            DirichletBC,
        )

        bc_list = [
            DirichletBC(basis, "left", 0.0, bkd, components=(0,)),
            DirichletBC(basis, "bottom", 0.0, bkd, components=(1,)),
            DirichletBC(basis, "front", 0.0, bkd, components=(2,)),
            DirichletBC(basis, "back", alpha, bkd, components=(2,)),
        ]
        # each BC constrains only its own component's DOFs
        for bc, comp in zip(bc_list, [0, 1, 2, 2]):
            dofs = bkd.to_numpy(bc.boundary_dofs())
            assert np.all(dofs % 3 == comp)

        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,
            poisson_ratio=0.0,
            boundary_conditions=bc_list,
            bkd=bkd,
        )
        solver = SteadyStateSolver(
            physics, tol=1e-12, max_iter=5, line_search=False
        )
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        result = solver.solve(u0)
        assert result.converged

        u_np = bkd.to_numpy(result.solution)
        dof_coords = bkd.to_numpy(basis.dof_coordinates())
        ndofs = basis.ndofs()
        exact = np.zeros(ndofs)
        zcomp = np.arange(ndofs) % 3 == 2
        exact[zcomp] = alpha * dof_coords[2, zcomp]
        rel_error = np.linalg.norm(u_np - exact) / np.linalg.norm(exact)
        assert rel_error < 1e-10


class TestLinearElasticity3DManufactured:
    """3D manufactured-solution recovery for linear elasticity."""

    @staticmethod
    def _solve_3d_mms(
        bkd,
        element_type,
        degree,
        nx,
        sol_strs,
        lambda_str,
        mu_str,
        per_quad_lame=False,
    ):
        """Solve an all-Dirichlet 3D MMS problem; return (u, exact, physics)."""
        from pyapprox.pde.galerkin.boundary.implementations import (
            DirichletBC,
        )
        from pyapprox.pde.galerkin.manufactured.adapter import (
            create_elasticity_manufactured_test,
        )

        functions, nvars = create_elasticity_manufactured_test(
            bounds=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            sol_strs=sol_strs,
            lambda_str=lambda_str,
            mu_str=mu_str,
            bkd=bkd,
        )

        mesh = StructuredMesh3D(
            nx=nx,
            ny=nx,
            nz=nx,
            bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            bkd=bkd,
            element_type=element_type,
        )
        basis = VectorLagrangeBasis(mesh, degree=degree)

        sol_func = functions["solution"]
        forcing_func = functions["forcing"]

        def body_force(x, time):
            return forcing_func(x).T  # (ndim, npts)

        def dirichlet_value(coords, time=0.0):
            return sol_func(coords).T  # (ndim, npts) vector convention

        bc_list = [
            DirichletBC(basis, name, dirichlet_value, bkd)
            for name in ["left", "right", "bottom", "top", "front", "back"]
        ]
        physics = LinearElasticity.from_uniform(
            basis=basis,
            youngs_modulus=1.0,  # overwritten below via set_lame_parameters
            poisson_ratio=0.3,
            body_force=body_force,
            boundary_conditions=bc_list,
            bkd=bkd,
        )
        # impose the manufactured (possibly varying) Lame fields exactly
        # at the quadrature points
        skfem_basis = basis.skfem_basis()
        qcoords = np.asarray(skfem_basis.global_coordinates())
        nelems, nquad = qcoords.shape[1], qcoords.shape[2]
        pts = qcoords.reshape(3, nelems * nquad)
        lam_q = np.asarray(functions["lambda"](pts)).reshape(nelems, nquad)
        mu_q = np.asarray(functions["mu"](pts)).reshape(nelems, nquad)
        physics.set_lame_parameters(lam_q, mu_q)

        solver = SteadyStateSolver(
            physics, tol=1e-12, max_iter=5, line_search=False
        )
        u0 = bkd.asarray(np.zeros(physics.nstates()))
        result = solver.solve(u0)
        assert result.converged

        u_np = bkd.to_numpy(result.solution)
        dof_coords = bkd.to_numpy(basis.dof_coordinates())
        vals = sol_func(dof_coords)  # (ndofs, 3)
        ndofs = basis.ndofs()
        exact = vals[np.arange(ndofs), np.arange(ndofs) % nvars]
        return u_np, exact, physics

    _QUADRATIC_SOLS = [
        "0.1*x*y + 0.05*z**2 + 0.2",
        "0.05*x**2 - 0.1*y*z + 0.1",
        "0.02*x*z + 0.08*y**2 - 0.2",
    ]

    @pytest.mark.parametrize("element_type", ["hex", "tet"])
    def test_3d_manufactured_solution(self, numpy_bkd, element_type) -> None:
        """Quadratic MMS with degree-2 elements => exact recovery."""
        u_np, exact, _ = self._solve_3d_mms(
            numpy_bkd,
            element_type,
            degree=2,
            nx=2,
            sol_strs=self._QUADRATIC_SOLS,
            lambda_str="1.5",
            mu_str="0.8",
        )
        rel_error = np.linalg.norm(u_np - exact) / np.linalg.norm(exact)
        assert rel_error < 1e-8

    def test_3d_variable_lame(self, numpy_bkd) -> None:
        """Linear MMS with spatially varying lam(x), mu(y) => exact
        recovery when the Lame fields are imposed at quadrature points."""
        u_np, exact, _ = self._solve_3d_mms(
            numpy_bkd,
            "hex",
            degree=1,
            nx=3,
            sol_strs=[
                "0.1*x + 0.05*y + 0.2",
                "0.05*y - 0.02*z - 0.1",
                "0.02*z + 0.04*x + 0.3",
            ],
            lambda_str="1.0 + 0.5*x",
            mu_str="1.0 + 0.25*y",
        )
        rel_error = np.linalg.norm(u_np - exact) / np.linalg.norm(exact)
        assert rel_error < 1e-10

    def test_3d_h_convergence(self, numpy_bkd) -> None:
        """P1 hex L2 convergence at the optimal O(h^2) rate.

        The solution must be genuinely coupled and non-polynomial:
        single-variable cubics plus bilinear cross-terms decouple into
        tensor-product 1D problems on a uniform hex grid, where P1 is
        nodally exact, so they cannot measure a convergence rate.
        """
        bkd = numpy_bkd
        sols = [
            "0.1*sin(x + 0.5*y + 0.2*z)",
            "0.05*cos(0.3*x + y - 0.4*z)",
            "0.08*sin(0.7*x - 0.2*y + z)",
        ]
        errors = []
        for nx in [2, 4, 8]:
            u_np, exact, physics = self._solve_3d_mms(
                bkd,
                "hex",
                degree=1,
                nx=nx,
                sol_strs=sols,
                lambda_str="1.0",
                mu_str="1.0",
            )
            err = u_np - exact
            M = physics.mass_matrix()
            if issparse(M):
                l2 = float(np.sqrt(err @ (M @ err)))
            else:
                l2 = float(np.sqrt(err @ (bkd.to_numpy(M) @ err)))
            errors.append(l2)
        rates = [
            np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)
        ]
        assert rates[-1] > 1.85, f"L2 rates {rates}, errors {errors}"
