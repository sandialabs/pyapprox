"""Tests for HyperelasticityPhysics 2D extensions.

Verifies via DerivativeChecker:
1. residual_mu_jacobian / residual_lamda_jacobian assemblies in 2D
2. pk1_stress matches residual structure
3. pk1_stress_jacobian in 2D
"""

from typing import Generic

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics import HyperelasticityPhysics
from pyapprox.pde.collocation.physics.neo_hookean_law import NeoHookeanLaw
from pyapprox.util.backends.protocols import Array


def _setup_2d_hyperelastic(bkd, npts_1d=6, lamda=1.0, mu=1.0):
    """Create 2D hyperelastic physics on [0,1]^2 with uniform Lame
    parameters held as rewritable fields."""
    mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    physics = HyperelasticityPhysics(
        basis, bkd, NeoHookeanLaw(bkd), mu=mu, lamda=lamda
    )
    return physics, basis


class _ResidualOfState(Generic[Array]):
    """Wraps physics.residual(state) for DerivativeChecker.

    Input: state (nstates,1). Output: residual (nstates,1).
    Jacobian: physics.jacobian(state).
    """

    def __init__(self, physics, bkd, time=0.0):
        self._physics = physics
        self._bkd = bkd
        self._time = time

    def bkd(self):
        return self._bkd

    def nvars(self):
        return self._physics.nstates()

    def nqoi(self):
        return self._physics.nstates()

    def __call__(self, samples):
        bkd = self._bkd
        if samples.ndim == 2:
            return bkd.stack(
                [
                    self._physics.residual(samples[:, i], self._time)
                    for i in range(samples.shape[1])
                ],
                axis=1,
            )
        return self._physics.residual(samples, self._time).reshape(-1, 1)

    def jacobian(self, sample):
        if sample.ndim == 2:
            sample = sample[:, 0]
        return self._physics.jacobian(sample, self._time)


class _ResidualOfField(Generic[Array]):
    """Wraps residual(state; c) as a function of one Lame field c, set
    through the physics' NodalField, for DerivativeChecker.

    Input: c (npts,1). Output: residual (nstates,1).
    Jacobian: the given assembly, e.g. residual_mu_jacobian.
    """

    def __init__(self, physics, field, assembly, state, base, bkd, time=0.0):
        self._physics = physics
        self._field = field
        self._assembly = assembly
        self._state = state
        self._base = base
        self._bkd = bkd
        self._time = time

    def bkd(self):
        return self._bkd

    def nvars(self):
        return self._base.shape[0]

    def nqoi(self):
        return self._physics.nstates()

    def _residual_at(self, values):
        self._field.set_values(values)
        res = self._physics.residual(self._state, self._time)
        self._field.set_values(self._base)
        return res

    def __call__(self, samples):
        return self._bkd.stack(
            [self._residual_at(samples[:, i]) for i in range(samples.shape[1])],
            axis=1,
        )

    def derivatives(self):
        return Derivatives.first_order(jacobian=self.jacobian)

    def jacobian(self, sample):
        self._field.set_values(sample[:, 0])
        jac = self._assembly(self._state)
        self._field.set_values(self._base)
        return jac


class _FluxComponentOfState(Generic[Array]):
    """Wraps one PK1 stress component P_iJ(state) for DerivativeChecker.

    Input: state (nstates,1). Output: P_iJ (npts,1).
    Jacobian: pk1_stress_jacobian[i][j].
    """

    def __init__(self, physics, bkd, row, col):
        self._physics = physics
        self._bkd = bkd
        self._row = row
        self._col = col

    def bkd(self):
        return self._bkd

    def nvars(self):
        return self._physics.nstates()

    def nqoi(self):
        return self._physics.npts()

    def __call__(self, samples):
        bkd = self._bkd
        if samples.ndim == 2:
            cols = []
            for i in range(samples.shape[1]):
                flux = self._physics.pk1_stress(samples[:, i])
                cols.append(flux[self._row][self._col])
            return bkd.stack(cols, axis=1)
        flux = self._physics.pk1_stress(samples)
        return flux[self._row][self._col].reshape(-1, 1)

    def derivatives(self):
        return Derivatives.first_order(jacobian=self.jacobian)

    def jacobian(self, sample):
        if sample.ndim == 2:
            sample = sample[:, 0]
        flux_jac = self._physics.pk1_stress_jacobian(sample)
        return flux_jac[self._row][self._col]


class TestHyperelasticSensitivity2D:
    """Test 2D residual sensitivities and flux for HyperelasticityPhysics."""
    def test_residual_mu_jacobian_2d(self, bkd):
        """DerivativeChecker validates the mu assembly in 2D."""
        lamda, mu = 2.0, 1.5
        physics, basis = _setup_2d_hyperelastic(
            bkd,
            npts_1d=6,
            lamda=lamda,
            mu=mu,
        )
        npts = basis.npts()
        nstates = physics.nstates()

        # Variable mu (positive)
        np.random.seed(42)
        mu0 = bkd.asarray(mu + 0.2 * np.abs(np.random.randn(npts)))
        physics.mu_function().set_values(mu0)

        # Fixed state (small deformation, J > 0)
        np.random.seed(43)
        state = bkd.asarray(np.random.randn(nstates) * 0.01)

        wrapper = _ResidualOfField(
            physics,
            physics.mu_function(),
            physics.residual_mu_jacobian,
            state,
            mu0,
            bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = mu0[:, None]
        errors = checker.check_derivatives(sample, verbosity=0)
        assert float(checker.error_ratio(errors[0])) <= 1e-6

    def test_residual_lamda_jacobian_2d(self, bkd):
        """DerivativeChecker validates the lambda assembly in 2D."""
        lamda, mu = 2.0, 1.5
        physics, basis = _setup_2d_hyperelastic(
            bkd,
            npts_1d=6,
            lamda=lamda,
            mu=mu,
        )
        npts = basis.npts()
        nstates = physics.nstates()

        # Variable lambda (positive)
        np.random.seed(42)
        lam0 = bkd.asarray(lamda + 0.2 * np.abs(np.random.randn(npts)))
        physics.lamda_function().set_values(lam0)

        # Fixed state
        np.random.seed(43)
        state = bkd.asarray(np.random.randn(nstates) * 0.01)

        wrapper = _ResidualOfField(
            physics,
            physics.lamda_function(),
            physics.residual_lamda_jacobian,
            state,
            lam0,
            bkd,
        )
        checker = DerivativeChecker(wrapper)
        sample = lam0[:, None]
        errors = checker.check_derivatives(sample, verbosity=0)
        assert float(checker.error_ratio(errors[0])) <= 1e-6

    def test_pk1_stress_2d_matches_residual(self, bkd):
        """Verify div(P) from pk1_stress matches residual (no forcing)."""
        physics, basis = _setup_2d_hyperelastic(bkd, npts_1d=6)
        nstates = physics.nstates()

        np.random.seed(42)
        state = bkd.asarray(np.random.randn(nstates) * 0.01)

        flux = physics.pk1_stress(state)
        P11, P12 = flux[0]
        P21, P22 = flux[1]

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)
        div_P_x = Dx @ P11 + Dy @ P12
        div_P_y = Dx @ P21 + Dy @ P22
        div_P = bkd.concatenate([div_P_x, div_P_y])

        residual = physics.residual(state, 0.0)
        bkd.assert_allclose(div_P, residual, rtol=1e-12)

    def test_pk1_stress_jacobian_2d(self, bkd):
        """DerivativeChecker validates dP_iJ/d(state) for all 4 components."""
        physics, basis = _setup_2d_hyperelastic(bkd, npts_1d=6)
        nstates = physics.nstates()

        np.random.seed(42)
        state = bkd.asarray(np.random.randn(nstates) * 0.01)
        sample = state[:, None]

        for i in range(2):
            for jj in range(2):
                wrapper = _FluxComponentOfState(physics, bkd, i, jj)
                checker = DerivativeChecker(wrapper)
                errors = checker.check_derivatives(sample, verbosity=0)
                assert float(checker.error_ratio(errors[0])) <= 1e-5
