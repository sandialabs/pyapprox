"""Tests for HyperelasticTractionNormalOperator.

Verifies via DerivativeChecker:
1. Traction operator Jacobian w.r.t. state
2. Traction is state-dependent (nonlinear)
3. Zero displacement gives zero traction (P(I) = 0 for NeoHookean)
4. Factory creates RobinBC with correct state indices
5. Small-displacement limit matches linear traction
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.boundary.hyperelastic_traction import (
    HyperelasticTractionNormalOperator,
    hyperelastic_traction_neumann_bc,
)
from pyapprox.pde.collocation.boundary.normal_operators import (
    TractionNormalOperator,
)
from pyapprox.pde.collocation.boundary.robin import RobinBC
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class _TractionOfState(Generic[Array]):
    """Wraps traction operator as function of state for DerivativeChecker.

    Input: state (2*npts, 1). Output: traction (nboundary, 1).
    Jacobian: operator.jacobian(state).
    """

    def __init__(self, operator, bkd):
        self._operator = operator
        self._bkd = bkd
        self._nstates = operator._npts * 2
        self._nboundary = operator._nboundary

    def bkd(self):
        return self._bkd

    def nvars(self):
        return self._nstates

    def nqoi(self):
        return self._nboundary

    def __call__(self, samples):
        bkd = self._bkd
        if samples.ndim == 2:
            return bkd.stack(
                [self._operator(samples[:, i]) for i in range(samples.shape[1])],
                axis=1,
            )
        return self._operator(samples).reshape(-1, 1)

    def jacobian(self, sample):
        if sample.ndim == 2:
            sample = sample[:, 0]
        return self._operator.jacobian(sample)


class TestHyperelasticTraction(Generic[Array], unittest.TestCase):
    """Test HyperelasticTractionNormalOperator."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()

    def _setup_operator(self, npts_1d=6, lamda=2.0, mu=1.5, component=0):
        """Create traction operator on [0,1]^2 left boundary."""
        bkd = self._bkd
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)

        stress_model = NeoHookeanStress(lamda, mu)

        # Left boundary (boundary_id=0)
        left_idx = mesh.boundary_indices(0)
        normals = mesh.boundary_normals(0)

        op = HyperelasticTractionNormalOperator(
            bkd,
            left_idx,
            normals,
            [Dx, Dy],
            stress_model,
            npts,
            component,
        )
        return op, mesh, basis, stress_model

    def test_traction_jacobian_component_0(self):
        """DerivativeChecker validates d(t_x)/d(state)."""
        bkd = self._bkd
        op, mesh, basis, _ = self._setup_operator(component=0)
        npts = basis.npts()

        np.random.seed(42)
        state = bkd.asarray(np.random.randn(2 * npts) * 0.01)

        wrapper = _TractionOfState(op, bkd)
        checker = DerivativeChecker(wrapper)
        sample = state[:, None]
        errors = checker.check_derivatives(sample, verbosity=0)
        self.assertLessEqual(float(checker.error_ratio(errors[0])), 1e-5)

    def test_traction_jacobian_component_1(self):
        """DerivativeChecker validates d(t_y)/d(state)."""
        bkd = self._bkd
        op, mesh, basis, _ = self._setup_operator(component=1)
        npts = basis.npts()

        np.random.seed(42)
        state = bkd.asarray(np.random.randn(2 * npts) * 0.01)

        wrapper = _TractionOfState(op, bkd)
        checker = DerivativeChecker(wrapper)
        sample = state[:, None]
        errors = checker.check_derivatives(sample, verbosity=0)
        self.assertLessEqual(float(checker.error_ratio(errors[0])), 1e-5)

    def test_traction_is_state_dependent(self):
        """Jacobian at two different states should differ (nonlinear)."""
        bkd = self._bkd
        op, mesh, basis, _ = self._setup_operator()
        npts = basis.npts()

        np.random.seed(42)
        state1 = bkd.asarray(np.random.randn(2 * npts) * 0.02)
        np.random.seed(99)
        state2 = bkd.asarray(np.random.randn(2 * npts) * 0.02)

        jac1 = op.jacobian(state1)
        jac2 = op.jacobian(state2)

        diff = bkd.norm(jac1 - jac2)
        self.assertGreater(float(diff), 1e-10)

    def test_zero_displacement_zero_traction(self):
        """At zero displacement, F=I so P=0, hence traction is zero."""
        bkd = self._bkd
        for comp in [0, 1]:
            op, mesh, basis, _ = self._setup_operator(component=comp)
            npts = basis.npts()
            state = bkd.zeros((2 * npts,))
            traction = op(state)
            bkd.assert_allclose(traction, bkd.zeros(traction.shape), atol=1e-14)

    def test_factory_creates_robin_bc(self):
        """hyperelastic_traction_neumann_bc returns RobinBC with correct indices."""
        bkd = self._bkd
        npts_1d = 6
        lamda, mu = 2.0, 1.5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()
        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)
        stress_model = NeoHookeanStress(lamda, mu)

        left_idx = mesh.boundary_indices(0)
        normals = mesh.boundary_normals(0)

        for comp in [0, 1]:
            bc = hyperelastic_traction_neumann_bc(
                bkd,
                left_idx,
                normals,
                [Dx, Dy],
                stress_model,
                npts,
                comp,
            )
            self.assertIsInstance(bc, RobinBC)
            expected_indices = left_idx + comp * npts
            bkd.assert_allclose(
                bc.boundary_indices(),
                expected_indices,
                atol=0,
            )

    def test_linear_limit(self):
        """Small displacement: hyperelastic traction ≈ linear traction."""
        bkd = self._bkd
        npts_1d = 6
        lamda, mu = 2.0, 1.5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()
        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)
        stress_model = NeoHookeanStress(lamda, mu)

        left_idx = mesh.boundary_indices(0)
        normals = mesh.boundary_normals(0)

        np.random.seed(42)
        eps = 1e-4
        state = bkd.asarray(np.random.randn(2 * npts) * eps)

        for comp in [0, 1]:
            hyper_op = HyperelasticTractionNormalOperator(
                bkd,
                left_idx,
                normals,
                [Dx, Dy],
                stress_model,
                npts,
                comp,
            )
            linear_op = TractionNormalOperator(
                bkd,
                left_idx,
                normals,
                [Dx, Dy],
                lamda,
                mu,
                comp,
                npts,
            )

            t_hyper = hyper_op(state)
            t_linear = linear_op(state)
            bkd.assert_allclose(t_hyper, t_linear, atol=1e-4)

    def test_has_coefficient_dependence(self):
        """Hyperelastic traction reports coefficient dependence."""
        op, _, _, _ = self._setup_operator()
        self.assertTrue(op.has_coefficient_dependence())


class TestHyperelasticTractionNumpy(TestHyperelasticTraction[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestHyperelasticTractionTorch(TestHyperelasticTraction[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()
