"""Tests for GalerkinLameParameterization."""

import unittest
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import issparse

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.galerkin.basis import VectorLagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics.composite_linear_elasticity import (
    CompositeLinearElasticity,
)
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.parameterizations.galerkin_lame import (
    create_galerkin_lame_parameterization,
)
from pyapprox.pde.parameterizations.protocol import (
    ParameterizationProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests  # noqa: F401


def _to_dense(mat):
    """Convert sparse matrix to dense numpy array if needed."""
    if issparse(mat):
        return mat.toarray()
    return np.asarray(mat)


def _make_physics(bkd, E=1.0, nu=0.3, with_bcs=True):
    """Create a 2D uniform CompositeLinearElasticity."""
    mesh = StructuredMesh2D(
        nx=5,
        ny=5,
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        bkd=bkd,
    )
    basis = VectorLagrangeBasis(mesh, degree=1)

    def body_force(x, time):
        f = np.zeros_like(x)
        f[0, :] = 1.0
        f[1, :] = -2.0
        return f

    if with_bcs:
        bc_list = [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
            DirichletBC(basis, "bottom", 0.0, bkd),
            DirichletBC(basis, "top", 0.0, bkd),
        ]
    else:
        bc_list = []

    return CompositeLinearElasticity.from_uniform(
        basis=basis,
        youngs_modulus=E,
        poisson_ratio=nu,
        body_force=body_force,
        boundary_conditions=bc_list,
        bkd=bkd,
    )


def _make_multi_material_physics(bkd, with_bcs=True):
    """Create a 2-material CompositeLinearElasticity."""
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

    def body_force(x, time):
        f = np.zeros_like(x)
        f[0, :] = 1.0
        f[1, :] = -2.0
        return f

    if with_bcs:
        bc_list = [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 0.0, bkd),
            DirichletBC(basis, "bottom", 0.0, bkd),
            DirichletBC(basis, "top", 0.0, bkd),
        ]
    else:
        bc_list = []

    return CompositeLinearElasticity(
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
        body_force=body_force,
        boundary_conditions=bc_list,
    )


class TestGalerkinLameParameterizationBase(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self.bkd_inst = self.bkd()

    def test_protocol_conformance(self) -> None:
        """GalerkinLameParameterization satisfies ParameterizationProtocol."""
        physics = _make_physics(self.bkd_inst)
        param = create_galerkin_lame_parameterization(physics, self.bkd_inst)
        self.assertIsInstance(param, ParameterizationProtocol)

    def test_nparams(self) -> None:
        """nparams returns 2 * nmaterials."""
        physics = _make_physics(self.bkd_inst)
        param = create_galerkin_lame_parameterization(physics, self.bkd_inst)
        self.assertEqual(param.nparams(), 2)

        physics_multi = _make_multi_material_physics(self.bkd_inst)
        param_multi = create_galerkin_lame_parameterization(
            physics_multi, self.bkd_inst
        )
        self.assertEqual(param_multi.nparams(), 4)

    def test_apply_changes_stiffness(self) -> None:
        """apply() modifies the physics stiffness matrix."""
        physics = _make_physics(self.bkd_inst, E=1.0, nu=0.3)
        param = create_galerkin_lame_parameterization(physics, self.bkd_inst)

        K1 = _to_dense(physics.stiffness_matrix()).copy()

        param.apply(physics, self.bkd_inst.asarray(np.array([2.0, 0.25])))
        K2 = _to_dense(physics.stiffness_matrix())

        diff = np.linalg.norm(K2 - K1)
        self.assertGreater(diff, 1e-10)

    def test_apply_poisson_validation(self) -> None:
        """apply() raises for invalid Poisson ratio."""
        physics = _make_physics(self.bkd_inst)
        param = create_galerkin_lame_parameterization(physics, self.bkd_inst)
        with self.assertRaises(ValueError):
            param.apply(physics, self.bkd_inst.asarray(np.array([1.0, 0.5])))

    def test_param_jacobian_fd_validation(self) -> None:
        """FD validation of param_jacobian via DerivativeChecker."""
        bkd = self.bkd_inst
        E0, nu0 = 2.0, 0.3
        physics = _make_physics(bkd, E=E0, nu=nu0, with_bcs=False)
        param = create_galerkin_lame_parameterization(physics, bkd)
        nstates = physics.nstates()

        rng = np.random.RandomState(42)
        u = bkd.asarray(rng.randn(nstates))

        def residual_of_params(params: Array) -> Array:
            nsamples = params.shape[1]
            results = []
            for ii in range(nsamples):
                p = params[:, ii]
                param.apply(physics, p)
                res = physics.spatial_residual(u, 0.0)
                results.append(bkd.reshape(res, (nstates, 1)))
            param.apply(physics, bkd.asarray(np.array([E0, nu0])))
            return bkd.hstack(results)

        def jacobian_of_params(params: Array) -> Array:
            p = params[:, 0]
            param.apply(physics, p)
            pj = param.param_jacobian(physics, u, 0.0, p)
            param.apply(physics, bkd.asarray(np.array([E0, nu0])))
            return pj

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=2,
            fun=residual_of_params,
            jacobian=jacobian_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray(np.array([[E0], [nu0]]))
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        self.assertLessEqual(ratio, 1e-6)

    def test_param_jacobian_multi_material_fd(self) -> None:
        """FD validation for multi-material param_jacobian."""
        bkd = self.bkd_inst
        physics = _make_multi_material_physics(bkd, with_bcs=False)
        param = create_galerkin_lame_parameterization(physics, bkd)
        nstates = physics.nstates()

        rng = np.random.RandomState(42)
        u = bkd.asarray(rng.randn(nstates))
        p0 = np.array([1.0, 0.3, 5.0, 0.2])

        def residual_of_params(params: Array) -> Array:
            nsamples = params.shape[1]
            results = []
            for ii in range(nsamples):
                p = params[:, ii]
                param.apply(physics, p)
                res = physics.spatial_residual(u, 0.0)
                results.append(bkd.reshape(res, (nstates, 1)))
            param.apply(physics, bkd.asarray(p0))
            return bkd.hstack(results)

        def jacobian_of_params(params: Array) -> Array:
            p = params[:, 0]
            param.apply(physics, p)
            pj = param.param_jacobian(physics, u, 0.0, p)
            param.apply(physics, bkd.asarray(p0))
            return pj

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=nstates,
            nvars=4,
            fun=residual_of_params,
            jacobian=jacobian_of_params,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray(p0.reshape(-1, 1))
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        self.assertLessEqual(ratio, 1e-6)

    def test_initial_param_jacobian_is_zero(self) -> None:
        """initial_param_jacobian returns all zeros."""
        bkd = self.bkd_inst
        physics = _make_physics(bkd)
        param = create_galerkin_lame_parameterization(physics, bkd)
        p = bkd.asarray(np.array([1.0, 0.3]))
        ipj = param.initial_param_jacobian(physics, p)
        ipj_np = bkd.to_numpy(ipj)
        np.testing.assert_array_equal(ipj_np, 0.0)

    def test_adjoint_gradient_steady_integration(self) -> None:
        """Full steady-state solve + adjoint gradient through parameterization.

        QoI: Q(u(p)) = c^T u(p) where u(p) solves K(p)*u = b.
        Adjoint gradient: dQ/dp = (dF/dp)^T lambda, with J^T lambda = -c.
        """
        bkd = self.bkd_inst
        E0, nu0 = 1.0, 0.3
        physics = _make_physics(bkd, E=E0, nu=nu0, with_bcs=True)
        param = create_galerkin_lame_parameterization(physics, bkd)

        np.random.seed(42)
        n = physics.nstates()
        c_np = np.random.randn(n)

        def qoi_of_params(params: Array) -> Array:
            nsamples = params.shape[1]
            results = []
            for ii in range(nsamples):
                p = params[:, ii]
                param.apply(physics, p)
                r = SteadyStateSolver(
                    physics,
                    tol=1e-12,
                    max_iter=5,
                    line_search=False,
                ).solve_linear()
                u_np = bkd.to_numpy(r.solution)
                Q = c_np @ u_np
                results.append(Q)
            param.apply(physics, bkd.asarray(np.array([E0, nu0])))
            return bkd.reshape(bkd.asarray(np.array(results)), (1, nsamples))

        def adjoint_gradient(params: Array) -> Array:
            p = params[:, 0]
            param.apply(physics, p)
            r = SteadyStateSolver(
                physics,
                tol=1e-12,
                max_iter=5,
                line_search=False,
            ).solve_linear()
            u_sol = r.solution

            # Solve adjoint: J^T lambda = -c
            J_np = _to_dense(physics.jacobian(u_sol, 0.0))
            lam_np = np.linalg.solve(J_np.T, -c_np)

            # Use parameterization param_jacobian (raw, no BC enforcement)
            dF_dp_raw = bkd.to_numpy(param.param_jacobian(physics, u_sol, 0.0, p))

            # Apply BC enforcement for steady-state
            dF_dp = bkd.to_numpy(
                physics._apply_dirichlet_to_param_jacobian(
                    bkd.asarray(dF_dp_raw), u_sol, 0.0
                )
            )

            grad = dF_dp.T @ lam_np
            param.apply(physics, bkd.asarray(np.array([E0, nu0])))
            return bkd.reshape(bkd.asarray(grad), (1, 2))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=2,
            fun=qoi_of_params,
            jacobian=adjoint_gradient,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray(np.array([[E0], [nu0]]))
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        self.assertLessEqual(ratio, 1e-6)


class TestGalerkinLameParameterizationNumpy(
    TestGalerkinLameParameterizationBase[NDArray[Any]]
):
    __test__ = True

    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


try:
    import torch

    from pyapprox.util.backends.torch import TorchBkd

    class TestGalerkinLameParameterizationTorch(
        TestGalerkinLameParameterizationBase[torch.Tensor]
    ):
        __test__ = True

        def setUp(self) -> None:
            torch.set_default_dtype(torch.float64)
            self._bkd = TorchBkd()
            super().setUp()

        def bkd(self) -> Backend[torch.Tensor]:
            return self._bkd

        @unittest.skip("sparse solve not available on CPU with TorchBkd")
        def test_adjoint_gradient_steady_integration(self) -> None:
            pass

        @unittest.skip("sparse @ torch returns numpy — pre-existing issue")
        def test_param_jacobian_fd_validation(self) -> None:
            pass

        @unittest.skip("sparse @ torch returns numpy — pre-existing issue")
        def test_param_jacobian_multi_material_fd(self) -> None:
            pass

except ImportError:
    pass


if __name__ == "__main__":
    unittest.main()
