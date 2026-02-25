"""Tests for param_jacobian with non-Dirichlet BCs (flux Neumann, Robin).

These tests verify that the BC loop in CollocationStateEquationAdapter
correctly handles physical sensitivities for coefficient-dependent BCs.
"""

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
    AffineTransform1D,
    TransformedMesh1D,
)
from pyapprox.pde.collocation.boundary import (
    flux_neumann_bc,
    gradient_robin_bc,
    zero_dirichlet_bc,
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
from pyapprox.pde.collocation.forward_models.steady import (
    CollocationStateEquationAdapter,
    SteadyForwardModel,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)


def _create_flux_neumann_problem(bkd, npts=20):
    """Create 1D ADR with left Dirichlet + right flux Neumann, KLE-param D.

    Problem: -d/dx(D(x)*du/dx) = f(x) on [0, 1]
    BCs: u(0) = 0 (Dirichlet), -D(x)*du/dx|_{x=1} = 0 (flux Neumann)
    D(x) = D_base + p0*phi0(x) + p1*phi1(x)
    """
    transform = AffineTransform1D((0.0, 1.0), bkd)
    mesh = TransformedMesh1D(npts, bkd, transform)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = mesh.points()[0, :]  # (npts,)

    forcing = lambda t: (math.pi ** 2) * bkd.sin(math.pi * nodes)

    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=2.0, forcing=forcing,
    )

    left_idx = mesh.boundary_indices(0)
    bc_left = zero_dirichlet_bc(bkd, left_idx)

    right_idx = mesh.boundary_indices(1)
    right_normals = mesh.boundary_normals(1)
    bc_right = flux_neumann_bc(bkd, right_idx, right_normals, physics, 0.0)

    physics.set_boundary_conditions([bc_left, bc_right])

    # BasisExpansion field map with D_base=2.0
    phi0 = bkd.ones((npts,))
    phi1 = nodes
    fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
    param = create_diffusion_parameterization(bkd, basis, fm)

    init_state = bkd.zeros((npts,))
    return physics, param, init_state


def _create_gradient_robin_problem(bkd, npts=20):
    """Create 1D ADR with gradient Robin on both sides, KLE-param D.

    Problem: -d/dx(D(x)*du/dx) = f(x) on [0, 1]
    BCs: u + 0.5*grad(u).n = 0 on both ends (gradient Robin)
    D(x) = D_base + p0*phi0(x) + p1*phi1(x)

    GradientNormalOperator has_coefficient_dependence() = False,
    so BC rows should be zeroed correctly.
    Uses D_base=2.0 to stay positive under FD perturbation.
    """
    transform = AffineTransform1D((0.0, 1.0), bkd)
    mesh = TransformedMesh1D(npts, bkd, transform)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = mesh.points()[0, :]

    forcing = lambda t: bkd.ones((npts,))

    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=2.0, forcing=forcing,
    )

    left_idx = mesh.boundary_indices(0)
    left_normals = mesh.boundary_normals(0)
    D_matrices = [basis.derivative_matrix(1, 0)]
    bc_left = gradient_robin_bc(
        bkd, left_idx, left_normals, D_matrices, 1.0, 0.5, 0.0,
    )

    right_idx = mesh.boundary_indices(1)
    right_normals = mesh.boundary_normals(1)
    bc_right = gradient_robin_bc(
        bkd, right_idx, right_normals, D_matrices, 1.0, 0.5, 0.0,
    )

    physics.set_boundary_conditions([bc_left, bc_right])

    phi0 = bkd.ones((npts,))
    phi1 = nodes
    fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
    param = create_diffusion_parameterization(bkd, basis, fm)

    init_state = bkd.zeros((npts,))
    return physics, param, init_state


def _create_all_dirichlet_problem(bkd, npts=20):
    """Create 1D ADR with Dirichlet on both sides, KLE-param D.

    Regression test: same as existing test_steady.py setup.
    Uses D_base=2.0 to stay positive under FD perturbation.
    """
    transform = AffineTransform1D((-1.0, 1.0), bkd)
    mesh = TransformedMesh1D(npts, bkd, transform)
    basis = ChebyshevBasis1D(mesh, bkd)
    nodes = mesh.points()[0, :]

    forcing = lambda t: (math.pi ** 2) * bkd.sin(math.pi * nodes)

    physics = AdvectionDiffusionReaction(
        basis, bkd, diffusion=2.0, forcing=forcing,
    )

    left_idx = mesh.boundary_indices(0)
    right_idx = mesh.boundary_indices(1)
    bc_left = zero_dirichlet_bc(bkd, left_idx)
    bc_right = zero_dirichlet_bc(bkd, right_idx)
    physics.set_boundary_conditions([bc_left, bc_right])

    phi0 = bkd.ones((npts,))
    phi1 = nodes
    fm = BasisExpansion(bkd, 2.0, [phi0, phi1])
    param = create_diffusion_parameterization(bkd, basis, fm)

    init_state = bkd.zeros((npts,))
    return physics, param, init_state


class TestSteadyBCParamJacobian(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def test_flux_neumann_param_jacobian(self):
        """DerivativeChecker validates param_jacobian with flux Neumann BC."""
        bkd = self._bkd
        physics, param, init_state = _create_flux_neumann_problem(bkd)

        fwd = SteadyForwardModel(
            physics, bkd, init_state, parameterization=param,
        )

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=fwd.nqoi(),
            nvars=fwd.nvars(),
            fun=fwd,
            jacobian=fwd.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.array([0.3, 0.1])[:, None]
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        self.assertLessEqual(ratio, 1e-5)

    def test_flux_neumann_adapter_param_jacobian(self):
        """DerivativeChecker validates adapter param_jacobian with flux Neumann."""
        bkd = self._bkd
        physics, param, init_state = _create_flux_neumann_problem(bkd)

        model = CollocationModel(physics, bkd, parameterization=param)
        adapter = CollocationStateEquationAdapter(
            model, bkd, parameterization=param,
        )

        param_2d = bkd.array([0.3, 0.1])[:, None]
        sol = adapter.solve(init_state[:, None], param_2d)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=adapter.nstates(),
            nvars=adapter.nparams(),
            fun=lambda p: adapter(sol, p),
            jacobian=lambda p: adapter.param_jacobian(sol, p),
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            param_2d, direction=None, relative=True,
        )[0]
        self.assertLessEqual(
            float(bkd.min(errors) / bkd.max(errors)), 1e-5
        )

    def test_gradient_robin_param_jacobian(self):
        """DerivativeChecker validates param_jacobian with gradient Robin BC.

        GradientNormalOperator.has_coefficient_dependence() returns False,
        so BC rows should be zeroed.
        """
        bkd = self._bkd
        physics, param, init_state = _create_gradient_robin_problem(bkd)

        fwd = SteadyForwardModel(
            physics, bkd, init_state, parameterization=param,
        )

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=fwd.nqoi(),
            nvars=fwd.nvars(),
            fun=fwd,
            jacobian=fwd.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.array([0.3, 0.1])[:, None]
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        self.assertLessEqual(ratio, 1e-5)

    def test_all_dirichlet_no_regression(self):
        """All-Dirichlet BCs still pass DerivativeChecker (no regression)."""
        bkd = self._bkd
        physics, param, init_state = _create_all_dirichlet_problem(bkd)

        fwd = SteadyForwardModel(
            physics, bkd, init_state, parameterization=param,
        )

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=fwd.nqoi(),
            nvars=fwd.nvars(),
            fun=fwd,
            jacobian=fwd.jacobian,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = bkd.array([0.3, 0.1])[:, None]
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.min(errors) / bkd.max(errors))
        self.assertLessEqual(ratio, 1e-5)


class TestSteadyBCParamJacobianNumpy(
    TestSteadyBCParamJacobian[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSteadyBCParamJacobianTorch(
    TestSteadyBCParamJacobian[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    test_suite = unittest.TestSuite()
    for test_class in [
        TestSteadyBCParamJacobianNumpy,
        TestSteadyBCParamJacobianTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
