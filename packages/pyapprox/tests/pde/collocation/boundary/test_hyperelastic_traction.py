"""Tests for HyperelasticTractionNormalOperator.

Verifies via DerivativeChecker:
1. Traction operator Jacobian w.r.t. state
2. Traction is state-dependent (nonlinear)
3. Zero displacement gives zero traction (P(I) = 0 for NeoHookean)
4. Factory creates RobinBC with correct state indices
5. Small-displacement limit matches linear traction
"""

from typing import Generic, List

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
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
from pyapprox.pde.collocation.physics.hyperelasticity import (
    HyperelasticityPhysics,
)
from pyapprox.pde.collocation.physics.linear_elasticity import (
    LinearElasticityPhysics,
)
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.pde.constitutive.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.util.backends.protocols import Array, Backend


class _TractionOfState(Generic[Array]):
    """Wraps traction operator as function of state for DerivativeChecker.

    Input: state (2*npts, 1). Output: traction (nboundary, 1).
    Jacobian: operator.jacobian(state, 0.0).
    """

    def __init__(self, operator, bkd, nstates):
        self._operator = operator
        self._bkd = bkd
        self._nstates = nstates
        self._nboundary = operator.normals().shape[0]

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
                [
                    self._operator(samples[:, i], 0.0)
                    for i in range(samples.shape[1])
                ],
                axis=1,
            )
        return self._operator(samples, 0.0).reshape(-1, 1)

    def derivatives(self):
        return Derivatives.first_order(jacobian=self.jacobian)

    def jacobian(self, sample):
        if sample.ndim == 2:
            sample = sample[:, 0]
        return self._operator.jacobian(sample, 0.0)


class TestHyperelasticTraction:
    """Test HyperelasticTractionNormalOperator."""
    def _setup_operator(self, bkd, npts_1d=6, lamda=2.0, mu=1.5, component=0) :
        """Create traction operator on [0,1]^2 left boundary."""
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)

        stress_model = NeoHookeanStress(lamda, mu)
        physics = HyperelasticityPhysics(
            basis, bkd, NeoHookeanLaw(bkd), mu=mu, lamda=lamda
        )

        # Left boundary (boundary_id=0)
        left_idx = mesh.boundary_indices(0)
        normals = mesh.boundary_normals(0)

        op = HyperelasticTractionNormalOperator(
            bkd, left_idx, normals, physics, component
        )
        return op, mesh, basis, stress_model

    def test_traction_jacobian_component_0(self, bkd):
        """DerivativeChecker validates d(t_x)/d(state)."""
        op, mesh, basis, _ = self._setup_operator(bkd, component=0)
        npts = basis.npts()

        np.random.seed(42)
        state = bkd.asarray(np.random.randn(2 * npts) * 0.01)

        wrapper = _TractionOfState(op, bkd, 2 * npts)
        checker = DerivativeChecker(wrapper)
        sample = state[:, None]
        errors = checker.check_derivatives(sample, verbosity=0)
        assert float(checker.error_ratio(errors[0])) <= 1e-5

    def test_traction_jacobian_component_1(self, bkd):
        """DerivativeChecker validates d(t_y)/d(state)."""
        op, mesh, basis, _ = self._setup_operator(bkd, component=1)
        npts = basis.npts()

        np.random.seed(42)
        state = bkd.asarray(np.random.randn(2 * npts) * 0.01)

        wrapper = _TractionOfState(op, bkd, 2 * npts)
        checker = DerivativeChecker(wrapper)
        sample = state[:, None]
        errors = checker.check_derivatives(sample, verbosity=0)
        assert float(checker.error_ratio(errors[0])) <= 1e-5

    def test_traction_is_state_dependent(self, bkd):
        """Jacobian at two different states should differ (nonlinear)."""
        op, mesh, basis, _ = self._setup_operator(bkd)
        npts = basis.npts()

        np.random.seed(42)
        state1 = bkd.asarray(np.random.randn(2 * npts) * 0.02)
        np.random.seed(99)
        state2 = bkd.asarray(np.random.randn(2 * npts) * 0.02)

        jac1 = op.jacobian(state1, 0.0)
        jac2 = op.jacobian(state2, 0.0)

        diff = bkd.norm(jac1 - jac2)
        assert float(diff) > 1e-10

    def test_zero_displacement_zero_traction(self, bkd):
        """At zero displacement, F=I so P=0, hence traction is zero."""
        for comp in [0, 1]:
            op, mesh, basis, _ = self._setup_operator(bkd, component=comp)
            npts = basis.npts()
            state = bkd.zeros((2 * npts,))
            traction = op(state, 0.0)
            bkd.assert_allclose(traction, bkd.zeros(traction.shape), atol=1e-14)

    def test_factory_creates_robin_bc(self, bkd):
        """hyperelastic_traction_neumann_bc returns RobinBC with correct indices."""
        npts_1d = 6
        lamda, mu = 2.0, 1.5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()
        physics = HyperelasticityPhysics(
            basis, bkd, NeoHookeanLaw(bkd), mu=mu, lamda=lamda
        )

        left_idx = mesh.boundary_indices(0)
        normals = mesh.boundary_normals(0)

        for comp in [0, 1]:
            bc = hyperelastic_traction_neumann_bc(
                bkd, left_idx, normals, physics, npts, comp
            )
            assert isinstance(bc, RobinBC)
            expected_indices = left_idx + comp * npts
            bkd.assert_allclose(
                bc.boundary_indices(),
                expected_indices,
                atol=0,
            )

    def test_linear_limit(self, bkd):
        """Small displacement: hyperelastic traction ≈ linear traction."""
        npts_1d = 6
        lamda, mu = 2.0, 1.5
        mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
        basis = ChebyshevBasis2D(mesh, bkd)
        npts = basis.npts()
        physics = HyperelasticityPhysics(
            basis, bkd, NeoHookeanLaw(bkd), mu=mu, lamda=lamda
        )
        linear_physics = LinearElasticityPhysics(basis, bkd, lamda=lamda, mu=mu)

        left_idx = mesh.boundary_indices(0)
        normals = mesh.boundary_normals(0)

        np.random.seed(42)
        eps = 1e-4
        state = bkd.asarray(np.random.randn(2 * npts) * eps)

        for comp in [0, 1]:
            hyper_op = HyperelasticTractionNormalOperator(
                bkd, left_idx, normals, physics, comp
            )
            linear_op = TractionNormalOperator(
                bkd, left_idx, normals, linear_physics, comp
            )

            t_hyper = hyper_op(state, 0.0)
            t_linear = linear_op(state, 0.0)
            bkd.assert_allclose(t_hyper, t_linear, atol=1e-4)

    def test_has_coefficient_dependence(self, bkd):
        """Hyperelastic traction reports coefficient dependence."""
        op, _, _, _ = self._setup_operator(bkd)
        assert op.has_coefficient_dependence()


class _AffineStressSource(Generic[Array]):
    """A stress source with P_iJ = (i + 1) * (J + 2) * state[:npts] and
    the matching Jacobian, so the traction is known exactly."""

    def __init__(self, bkd: Backend[Array], npts: int) -> None:
        self._bkd = bkd
        self._npts = npts

    def pk1_stress(self, state: Array) -> List[List[Array]]:
        u = state[: self._npts]
        return [[(i + 1.0) * (J + 2.0) * u for J in range(2)] for i in range(2)]

    def pk1_stress_jacobian(self, state: Array) -> List[List[Array]]:
        npts = self._npts
        select = self._bkd.concatenate(
            [self._bkd.eye(npts), self._bkd.zeros((npts, npts))], axis=1
        )
        return [[(i + 1.0) * (J + 2.0) * select for J in range(2)] for i in range(2)]


class TestTractionFromStressSource:
    def test_traction_is_stress_times_normal(self, bkd: Backend[Array]) -> None:
        """t_i = sum_J P_iJ n_J at the boundary rows, and its Jacobian is
        sum_J n_J dP_iJ there, for whatever stress the source supplies."""
        npts = 6
        idx = bkd.asarray([1, 4], dtype=int)
        normals = bkd.asarray([[0.6, 0.8], [1.0, 0.0]])
        state = bkd.asarray(np.arange(2 * npts, dtype=float))
        source = _AffineStressSource(bkd, npts)
        for comp in (0, 1):
            op = HyperelasticTractionNormalOperator(bkd, idx, normals, source, comp)
            weight = (comp + 1.0) * (2.0 * normals[:, 0] + 3.0 * normals[:, 1])
            bkd.assert_allclose(op(state, 0.0), weight * state[idx], rtol=1e-14)
            expected = bkd.zeros((2, 2 * npts))
            expected = bkd.copy(expected)
            for row in range(2):
                expected[row, bkd.to_int(idx[row])] = weight[row]
            bkd.assert_allclose(op.jacobian(state, 0.0), expected, atol=1e-14)

    def test_non_source_refused(self, numpy_bkd: Backend[Array]) -> None:
        with pytest.raises(TypeError, match="PK1StressSourceProtocol"):
            HyperelasticTractionNormalOperator(
                numpy_bkd,
                numpy_bkd.asarray([0], dtype=int),
                numpy_bkd.asarray([[1.0, 0.0]]),
                object(),  # type: ignore[arg-type]
                0,
            )
