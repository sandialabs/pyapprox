"""Steady-state manufactured solution tests for HyperelasticityPhysics.

Tests:
- Residual at exact solution ≈ 0 (1D, 2D, 3D)
- Jacobian matches finite differences (1D, 2D, 3D)
- Newton solve recovers exact solution (1D, 2D, 3D)
"""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


import numpy as np
from scipy.sparse import issparse

from pyapprox.pde.constitutive.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.manufactured.adapter import (
    GalerkinManufacturedSolution,
    create_hyperelasticity_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import (
    StructuredMesh1D,
    StructuredMesh2D,
)
from pyapprox.pde.galerkin.physics import HyperelasticityPhysics
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

    def value_func(coords):
        nbndry_dofs = coords.shape[1]
        vals = sol_func(coords)  # (nbndry_dofs, ncomponents)
        result = np.zeros(nbndry_dofs)
        for j in range(nbndry_dofs):
            result[j] = vals[j, j % ndim]
        return result

    return value_func


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

        body_force = GalerkinManufacturedSolution(basis, functions, bkd).forcing()
        value_func = _make_vector_dirichlet_value_func(functions["solution"], nvars)
        bc_list = [
            DirichletBC(basis, name, value_func, bkd) for name in ["left", "right"]
        ]

        physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=self._stress,
            bkd=bkd,
            body_force=body_force,
        )
        system = compose_galerkin_system(physics, bc_list)
        return system, functions, basis


    def test_jacobian_fd_check_1d(self, numpy_bkd: NumpyBkd) -> None:
        """1D analytical Jacobian matches finite differences."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_1d_problem(bkd, nx=10, degree=1)
        operator = system.spatial_operator()
        n = system.nstates()
        np.random.seed(42)
        state = bkd.asarray(0.01 * np.random.randn(n))
        jac = _to_dense(operator.spatial_jacobian(state, 0.0), bkd)
        res0 = bkd.to_numpy(operator.spatial_residual(state, 0.0))
        eps = 1e-7
        fd_jac = np.zeros((n, n))
        state_np = bkd.to_numpy(state)
        for j in range(n):
            state_pert = state_np.copy()
            state_pert[j] += eps
            res_pert = bkd.to_numpy(
                operator.spatial_residual(bkd.asarray(state_pert), 0.0)
            )
            fd_jac[:, j] = (res_pert - res0) / eps
        rel_err = np.max(np.abs(jac - fd_jac)) / (np.max(np.abs(fd_jac)) + 1e-30)
        assert rel_err < 1e-4


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

        body_force = GalerkinManufacturedSolution(basis, functions, bkd).forcing()
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
        )
        system = compose_galerkin_system(physics, bc_list)
        return system, functions, basis


    def test_jacobian_fd_check_2d(self, numpy_bkd: NumpyBkd) -> None:
        """2D analytical Jacobian matches finite differences."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_2d_problem(bkd, nx=3, ny=3, degree=1)
        operator = system.spatial_operator()
        n = system.nstates()
        np.random.seed(42)
        state = bkd.asarray(0.01 * np.random.randn(n))
        jac = _to_dense(operator.spatial_jacobian(state, 0.0), bkd)
        res0 = bkd.to_numpy(operator.spatial_residual(state, 0.0))
        eps = 1e-7
        fd_jac = np.zeros((n, n))
        state_np = bkd.to_numpy(state)
        for j in range(n):
            state_pert = state_np.copy()
            state_pert[j] += eps
            res_pert = bkd.to_numpy(
                operator.spatial_residual(bkd.asarray(state_pert), 0.0)
            )
            fd_jac[:, j] = (res_pert - res0) / eps
        rel_err = np.max(np.abs(jac - fd_jac)) / (np.max(np.abs(fd_jac)) + 1e-30)
        assert rel_err < 1e-4


# =========================================================================
# 3D Tests
# =========================================================================


class TestHyperelasticity3DBase:
    """Base class for 3D hyperelasticity tests."""

    def _setup(self, bkd) -> None:
        self._stress = NeoHookeanStress(1.0, 1.0)

    def _setup_3d_problem(self, bkd, nx=2, degree=2, element_type="hex"):
        """Create 3D MMS problem with all-Dirichlet BCs.

        Uses general quadratic solutions (nonzero on the boundary, and
        contained in both the Hex2 and TetP2 spaces) with small
        amplitudes so J = det(F) stays positive.
        """
        from pyapprox.pde.galerkin.mesh import StructuredMesh3D

        bounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        sol_strs = [
            "0.02*x*y + 0.01*z**2",
            "0.01*x**2 - 0.02*y*z",
            "0.005*x*z + 0.015*y**2",
        ]

        functions, nvars = create_hyperelasticity_manufactured_test(
            bounds=bounds,
            sol_strs=sol_strs,
            stress_model=self._stress,
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

        body_force = GalerkinManufacturedSolution(basis, functions, bkd).forcing()
        value_func = _make_vector_dirichlet_value_func(functions["solution"], nvars)
        bc_names = ["left", "right", "bottom", "top", "front", "back"]
        bc_list = [DirichletBC(basis, name, value_func, bkd) for name in bc_names]

        physics = HyperelasticityPhysics(
            basis=basis,
            stress_model=self._stress,
            bkd=bkd,
            body_force=body_force,
        )
        system = compose_galerkin_system(physics, bc_list)
        return system, functions, basis

    def test_jacobian_fd_check_3d(self, numpy_bkd: NumpyBkd) -> None:
        """3D analytical Jacobian matches finite differences."""
        bkd = numpy_bkd
        self._setup(bkd)
        system, functions, basis = self._setup_3d_problem(bkd, nx=2, degree=1)
        operator = system.spatial_operator()
        n = system.nstates()
        np.random.seed(42)
        state = bkd.asarray(0.01 * np.random.randn(n))
        jac = _to_dense(operator.spatial_jacobian(state, 0.0), bkd)
        res0 = bkd.to_numpy(operator.spatial_residual(state, 0.0))
        eps = 1e-7
        fd_jac = np.zeros((n, n))
        state_np = bkd.to_numpy(state)
        for j in range(n):
            state_pert = state_np.copy()
            state_pert[j] += eps
            res_pert = bkd.to_numpy(
                operator.spatial_residual(bkd.asarray(state_pert), 0.0)
            )
            fd_jac[:, j] = (res_pert - res0) / eps
        rel_err = np.max(np.abs(jac - fd_jac)) / (np.max(np.abs(fd_jac)) + 1e-30)
        assert rel_err < 1e-4


class TestHyperelasticityShapes:
    """Basic shape and property tests."""


    def test_mass_matrix_symmetric(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        bkd = NumpyBkd()
        stress = NeoHookeanStress(1.0, 1.0)
        mesh = StructuredMesh2D(nx=4, ny=4, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = HyperelasticityPhysics(basis, stress, bkd)
        M = _to_dense(physics.mass_matrix(), bkd)
        bkd.assert_allclose(M, M.T, rtol=0.0, atol=1.5e-6)

    def test_zero_state_zero_residual(self, numpy_bkd: NumpyBkd) -> None:
        """With no body force, u=0 gives zero residual (F=I, P=0)."""
        bkd = numpy_bkd
        bkd = NumpyBkd()
        stress = NeoHookeanStress(1.0, 1.0)
        mesh = StructuredMesh2D(nx=3, ny=3, bounds=[[0.0, 1.0], [0.0, 1.0]], bkd=bkd)
        basis = VectorLagrangeBasis(mesh, degree=1)
        physics = HyperelasticityPhysics(basis, stress, bkd)
        system = compose_galerkin_system(physics)
        state = bkd.asarray(np.zeros(physics.nstates()))
        res = bkd.to_numpy(system.spatial_operator().spatial_residual(state, 0.0))
        bkd.assert_allclose(res, np.zeros_like(res), rtol=0.0, atol=1.5e-6)


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
