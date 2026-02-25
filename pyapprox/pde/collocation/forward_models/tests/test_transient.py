"""Tests for transient forward model classes."""

import math
import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import (
    create_uniform_mesh_1d,
    TransformedMesh1D,
)
from pyapprox.pde.collocation.boundary import (
    DirichletBC,
    zero_dirichlet_bc,
    gradient_robin_bc,
)
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.field_maps.basis_expansion import (
    BasisExpansion,
)
from pyapprox.pde.parameterizations.diffusion import (
    create_diffusion_parameterization,
)
from pyapprox.pde.collocation.time_integration.collocation_model import (
    CollocationModel,
)
from pyapprox.pde.time.config import TimeIntegrationConfig
from pyapprox.pde.time.functionals.endpoint import EndpointFunctional
from pyapprox.pde.collocation.forward_models.transient import (
    TransientForwardModel,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.protocols import (
    FunctionProtocol,
    FunctionWithJacobianProtocol,
)


def _create_parameterized_transient_diffusion_problem(bkd, npts=15):
    """Create a parameterized transient diffusion problem for testing.

    Uses base AdvectionDiffusionReaction + DiffusionParameterization.

    Problem: du/dt = div(D(x) * grad(u)) with u(-1) = 0, u(1) = 0
    D(x) = D_base + p0 * phi0(x) + p1 * phi1(x)
    phi0(x) = 1 (constant), phi1(x) = x (linear)
    IC: u(x, 0) = sin(pi*x)

    Returns
    -------
    physics, parameterization, init_state, time_config
    """
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    mesh_obj = create_uniform_mesh_1d(npts, (-1.0, 1.0), bkd)
    nodes = basis.nodes()

    phi0 = bkd.ones((npts,))
    phi1 = nodes

    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=2.0,
    )

    left_idx = mesh_obj.boundary_indices(0)
    right_idx = mesh_obj.boundary_indices(1)
    bc_left = zero_dirichlet_bc(bkd, left_idx)
    bc_right = zero_dirichlet_bc(bkd, right_idx)
    physics.set_boundary_conditions([bc_left, bc_right])

    fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
    param = create_diffusion_parameterization(bkd, basis, fm)

    init_state = bkd.sin(math.pi * nodes)

    time_config = TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.02,
        newton_tol=1e-10,
        newton_maxiter=20,
    )

    return physics, param, init_state, time_config


class TestTransientForwardModel(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_call_matches_direct_solve(self):
        """__call__ with default functional matches solve_transient result."""
        bkd = self._bkd
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )

        forward_model = TransientForwardModel(
            physics, bkd, init_state, time_config, parameterization=param
        )

        param_1d = bkd.array([0.3, 0.1])
        samples = param_1d[:, None]

        # Direct solve
        param.apply(physics, param_1d)
        model = CollocationModel(physics, bkd, parameterization=param)
        solutions, times = model.solve_transient(init_state, time_config)

        # Forward model
        qoi = forward_model(samples)

        # Default functional returns all states at final time
        bkd.assert_allclose(qoi[:, 0], solutions[:, -1], rtol=1e-10)

    def test_call_multiple_samples(self):
        """__call__ handles multiple parameter samples correctly."""
        bkd = self._bkd
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )

        forward_model = TransientForwardModel(
            physics, bkd, init_state, time_config, parameterization=param
        )

        np.random.seed(42)
        nsamples = 3
        param_array = np.random.uniform(-0.1, 0.3, (2, nsamples))
        samples = bkd.asarray(param_array)

        result = forward_model(samples)
        self.assertEqual(result.shape[0], forward_model.nqoi())
        self.assertEqual(result.shape[1], nsamples)

        # Verify each sample individually
        for ii in range(nsamples):
            single_result = forward_model(samples[:, ii:ii+1])
            bkd.assert_allclose(
                result[:, ii:ii+1], single_result, rtol=1e-10
            )

    def test_nvars_nqoi(self):
        """nvars and nqoi are correct for default functional."""
        bkd = self._bkd
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )

        forward_model = TransientForwardModel(
            physics, bkd, init_state, time_config, parameterization=param
        )

        self.assertEqual(forward_model.nvars(), 2)
        self.assertEqual(forward_model.nqoi(), physics.nstates())

    def test_jacobian_vector_qoi_derivative_checker(self):
        """Jacobian with default functional (vector QoI) passes DerivativeChecker."""
        bkd = self._bkd
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )

        forward_model = TransientForwardModel(
            physics, bkd, init_state, time_config, parameterization=param
        )

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            param_2d, direction=None, relative=True
        )[0]
        self.assertLessEqual(
            float(bkd.min(errors) / bkd.max(errors)), 1e-5
        )

    def test_jacobian_scalar_qoi_endpoint(self):
        """Jacobian with EndpointFunctional (scalar QoI) passes DerivativeChecker."""
        bkd = self._bkd
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )
        nstates = physics.nstates()
        nparams = param.nparams()

        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)

        forward_model = TransientForwardModel(
            physics, bkd, init_state, time_config,
            functional=functional, parameterization=param,
        )
        self.assertEqual(forward_model.nqoi(), 1)

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            param_2d, direction=None, relative=True
        )[0]
        self.assertLessEqual(
            float(bkd.min(errors) / bkd.max(errors)), 1e-5
        )

    def test_scalar_qoi_matches_vector_qoi_row(self):
        """Scalar QoI Jacobian matches corresponding row of vector QoI Jacobian."""
        bkd = self._bkd
        physics, param, init_state, time_config = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )
        nstates = physics.nstates()
        nparams = param.nparams()

        param_2d = bkd.array([0.3, 0.1])[:, None]

        # Vector QoI (default: all states at final time)
        fwd_vector = TransientForwardModel(
            physics, bkd, init_state, time_config, parameterization=param
        )
        jac_vector = fwd_vector.jacobian(param_2d)

        # Scalar QoI for a specific state
        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)
        fwd_scalar = TransientForwardModel(
            physics, bkd, init_state, time_config,
            functional=functional, parameterization=param,
        )
        jac_scalar = fwd_scalar.jacobian(param_2d)

        # Row state_idx of vector Jacobian should match scalar Jacobian
        bkd.assert_allclose(
            jac_vector[state_idx:state_idx+1, :], jac_scalar, rtol=1e-8
        )

    def test_protocol_isinstance(self):
        """Protocol isinstance checks with parameterization path."""
        bkd = self._bkd
        physics, param, init_state, tc = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )
        fwd = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )
        self.assertTrue(isinstance(fwd, FunctionProtocol))
        self.assertTrue(isinstance(fwd, FunctionWithJacobianProtocol))


def _create_robin_transient_problem(bkd, npts=15):
    """Create a parameterized transient diffusion problem with Robin BCs.

    Uses base AdvectionDiffusionReaction + DiffusionParameterization.

    Same PDE as the Dirichlet test but with gradient Robin BCs on both
    boundaries:
      left:  1*u + 1*grad(u).n = 0
      right: 2*u + 1*grad(u).n = 0

    Asymmetric alpha tests that Robin coupling is handled correctly.

    Returns
    -------
    physics, parameterization, init_state, time_config
    """
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = basis.nodes()

    phi0 = bkd.ones((npts,))
    phi1 = nodes

    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=2.0,
    )

    left_idx = mesh.boundary_indices(0)
    right_idx = mesh.boundary_indices(1)
    left_normals = mesh.boundary_normals(0)
    right_normals = mesh.boundary_normals(1)
    D = basis.derivative_matrix()

    bc_left = gradient_robin_bc(
        bkd, left_idx, left_normals, [D], 1.0, 1.0, 0.0,
    )
    bc_right = gradient_robin_bc(
        bkd, right_idx, right_normals, [D], 2.0, 1.0, 0.0,
    )
    physics.set_boundary_conditions([bc_left, bc_right])

    fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
    param = create_diffusion_parameterization(bkd, basis, fm)

    init_state = bkd.sin(math.pi * nodes)

    time_config = TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.02,
        newton_tol=1e-10,
        newton_maxiter=20,
    )

    return physics, param, init_state, time_config


def _create_mixed_bc_transient_problem(bkd, npts=15):
    """Create a parameterized transient diffusion problem with mixed BCs.

    Uses base AdvectionDiffusionReaction + DiffusionParameterization.

    Left: Robin (alpha=1, beta=1, g=0)
    Right: Dirichlet (u = 0)

    Returns
    -------
    physics, parameterization, init_state, time_config
    """
    mesh = TransformedMesh1D(npts, bkd)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = basis.nodes()

    phi0 = bkd.ones((npts,))
    phi1 = nodes

    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=2.0,
    )

    left_idx = mesh.boundary_indices(0)
    right_idx = mesh.boundary_indices(1)
    left_normals = mesh.boundary_normals(0)
    D = basis.derivative_matrix()

    bc_left = gradient_robin_bc(
        bkd, left_idx, left_normals, [D], 1.0, 1.0, 0.0,
    )
    bc_right = zero_dirichlet_bc(bkd, right_idx)
    physics.set_boundary_conditions([bc_left, bc_right])

    fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
    param = create_diffusion_parameterization(bkd, basis, fm)

    init_state = bkd.sin(math.pi * nodes)

    time_config = TimeIntegrationConfig(
        method="backward_euler",
        init_time=0.0,
        final_time=0.1,
        deltat=0.02,
        newton_tol=1e-10,
        newton_maxiter=20,
    )

    return physics, param, init_state, time_config


class TestTransientRobinBC(Generic[Array], unittest.TestCase):
    """Tests for transient forward model with Robin BCs on both boundaries."""
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_vector_qoi_derivative_checker(self):
        """Forward sensitivity Jacobian passes DerivativeChecker with Robin BCs."""
        bkd = self._bkd
        physics, param, init_state, tc = _create_robin_transient_problem(bkd)

        forward_model = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            param_2d, direction=None, relative=True
        )[0]
        self.assertLessEqual(
            float(bkd.min(errors) / bkd.max(errors)), 1e-5
        )

    def test_scalar_qoi_adjoint_derivative_checker(self):
        """Adjoint gradient passes DerivativeChecker with Robin BCs."""
        bkd = self._bkd
        physics, param, init_state, tc = _create_robin_transient_problem(bkd)
        nstates = physics.nstates()
        nparams = param.nparams()

        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)

        forward_model = TransientForwardModel(
            physics, bkd, init_state, tc,
            functional=functional, parameterization=param,
        )
        self.assertEqual(forward_model.nqoi(), 1)

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            param_2d, direction=None, relative=True
        )[0]
        self.assertLessEqual(
            float(bkd.min(errors) / bkd.max(errors)), 1e-5
        )

    def test_scalar_matches_vector_row(self):
        """Scalar QoI adjoint Jacobian matches vector QoI forward sensitivity row."""
        bkd = self._bkd
        physics, param, init_state, tc = _create_robin_transient_problem(bkd)
        nstates = physics.nstates()
        nparams = param.nparams()

        param_2d = bkd.array([0.3, 0.1])[:, None]

        fwd_vector = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )
        jac_vector = fwd_vector.jacobian(param_2d)

        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)
        fwd_scalar = TransientForwardModel(
            physics, bkd, init_state, tc,
            functional=functional, parameterization=param,
        )
        jac_scalar = fwd_scalar.jacobian(param_2d)

        bkd.assert_allclose(
            jac_vector[state_idx:state_idx+1, :], jac_scalar, rtol=1e-8
        )


class TestTransientMixedBC(Generic[Array], unittest.TestCase):
    """Tests for transient forward model with mixed Robin + Dirichlet BCs."""
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_vector_qoi_derivative_checker(self):
        """Forward sensitivity Jacobian passes DerivativeChecker with mixed BCs."""
        bkd = self._bkd
        physics, param, init_state, tc = (
            _create_mixed_bc_transient_problem(bkd)
        )

        forward_model = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            param_2d, direction=None, relative=True
        )[0]
        self.assertLessEqual(
            float(bkd.min(errors) / bkd.max(errors)), 1e-5
        )

    def test_scalar_qoi_adjoint_derivative_checker(self):
        """Adjoint gradient passes DerivativeChecker with mixed BCs."""
        bkd = self._bkd
        physics, param, init_state, tc = (
            _create_mixed_bc_transient_problem(bkd)
        )
        nstates = physics.nstates()
        nparams = param.nparams()

        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)

        forward_model = TransientForwardModel(
            physics, bkd, init_state, tc,
            functional=functional, parameterization=param,
        )
        self.assertEqual(forward_model.nqoi(), 1)

        param_2d = bkd.array([0.3, 0.1])[:, None]

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=forward_model.nqoi(),
            nvars=forward_model.nvars(),
            fun=forward_model,
            jacobian=forward_model.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            param_2d, direction=None, relative=True
        )[0]
        self.assertLessEqual(
            float(bkd.min(errors) / bkd.max(errors)), 1e-5
        )

    def test_scalar_matches_vector_row(self):
        """Scalar QoI adjoint Jacobian matches vector QoI forward sensitivity row."""
        bkd = self._bkd
        physics, param, init_state, tc = (
            _create_mixed_bc_transient_problem(bkd)
        )
        nstates = physics.nstates()
        nparams = param.nparams()

        param_2d = bkd.array([0.3, 0.1])[:, None]

        fwd_vector = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )
        jac_vector = fwd_vector.jacobian(param_2d)

        state_idx = nstates // 2
        functional = EndpointFunctional(state_idx, nstates, nparams, bkd)
        fwd_scalar = TransientForwardModel(
            physics, bkd, init_state, tc,
            functional=functional, parameterization=param,
        )
        jac_scalar = fwd_scalar.jacobian(param_2d)

        bkd.assert_allclose(
            jac_vector[state_idx:state_idx+1, :], jac_scalar, rtol=1e-8
        )


class TestTransientForwardModelNumpy(
    TestTransientForwardModel[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTransientForwardModelTorch(
    TestTransientForwardModel[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

    def test_torch_autograd_jacobian(self):
        """Torch autograd.functional.jacobian matches fwd.jacobian."""
        bkd = self._bkd
        physics, param, init_state, tc = (
            _create_parameterized_transient_diffusion_problem(bkd)
        )
        fwd = TransientForwardModel(
            physics, bkd, init_state, tc, parameterization=param
        )
        sample = torch.tensor([0.3, 0.1], dtype=torch.float64)

        def fwd_call(p):
            return fwd(p[:, None])[:, 0]

        autograd_jac = torch.autograd.functional.jacobian(fwd_call, sample)
        analytical_jac = fwd.jacobian(sample[:, None])
        bkd.assert_allclose(analytical_jac, autograd_jac, rtol=1e-6)


class TestTransientRobinBCNumpy(
    TestTransientRobinBC[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTransientRobinBCTorch(
    TestTransientRobinBC[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestTransientMixedBCNumpy(
    TestTransientMixedBC[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTransientMixedBCTorch(
    TestTransientMixedBC[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    test_suite = unittest.TestSuite()
    for test_class in [
        TestTransientForwardModelNumpy,
        TestTransientForwardModelTorch,
        TestTransientRobinBCNumpy,
        TestTransientRobinBCTorch,
        TestTransientMixedBCNumpy,
        TestTransientMixedBCTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
