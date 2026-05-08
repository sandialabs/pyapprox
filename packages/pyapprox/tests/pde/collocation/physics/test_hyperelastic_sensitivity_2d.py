"""Tests for HyperelasticityPhysics 2D extensions.

Verifies via DerivativeChecker:
1. residual_mu_sensitivity / residual_lamda_sensitivity in 2D
2. compute_flux matches residual structure
3. compute_flux_jacobian in 2D
"""

from typing import Generic

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.physics import HyperelasticityPhysics
from pyapprox.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.util.backends.protocols import Array


def _setup_2d_hyperelastic(bkd, npts_1d=6, lamda=1.0, mu=1.0):
    """Create 2D hyperelastic physics on [0,1]^2."""
    mesh = TransformedMesh2D(npts_1d, npts_1d, bkd)
    basis = ChebyshevBasis2D(mesh, bkd)
    stress_model = NeoHookeanStress(lamda, mu)
    physics = HyperelasticityPhysics(basis, bkd, stress_model)
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


class _ResidualOfMu(Generic[Array]):
    """Wraps residual(state; mu) as function of mu for DerivativeChecker.

    Input: mu (npts,1). Output: residual (nstates,1).
    Jacobian via residual_mu_sensitivity applied column-by-column.
    """

    def __init__(self, physics, state, mu_base, bkd, time=0.0):
        self._physics = physics
        self._state = state
        self._mu_base = mu_base
        self._bkd = bkd
        self._time = time

    def bkd(self):
        return self._bkd

    def nvars(self):
        return self._mu_base.shape[0]

    def nqoi(self):
        return self._physics.nstates()

    def __call__(self, samples):
        bkd = self._bkd
        if samples.ndim == 2:
            cols = []
            for i in range(samples.shape[1]):
                self._physics.set_mu(samples[:, i])
                cols.append(self._physics.residual(self._state, self._time))
            self._physics.set_mu(self._mu_base)
            return bkd.stack(cols, axis=1)
        self._physics.set_mu(samples)
        res = self._physics.residual(self._state, self._time).reshape(-1, 1)
        self._physics.set_mu(self._mu_base)
        return res

    def jacobian(self, sample):
        if sample.ndim == 2:
            sample = sample[:, 0]
        bkd = self._bkd
        self._physics.set_mu(sample)
        npts = self.nvars()
        nstates = self.nqoi()
        jac = bkd.zeros((nstates, npts))
        jac = bkd.copy(jac)
        for j in range(npts):
            delta = bkd.zeros((npts,))
            delta = bkd.copy(delta)
            delta[j] = 1.0
            col = self._physics.residual_mu_sensitivity(
                self._state,
                self._time,
                delta,
            )
            for k in range(nstates):
                jac[k, j] = col[k]
        self._physics.set_mu(self._mu_base)
        return jac


class _ResidualOfLamda(Generic[Array]):
    """Wraps residual(state; lam) as function of lambda for DerivativeChecker."""

    def __init__(self, physics, state, lam_base, bkd, time=0.0):
        self._physics = physics
        self._state = state
        self._lam_base = lam_base
        self._bkd = bkd
        self._time = time

    def bkd(self):
        return self._bkd

    def nvars(self):
        return self._lam_base.shape[0]

    def nqoi(self):
        return self._physics.nstates()

    def __call__(self, samples):
        bkd = self._bkd
        if samples.ndim == 2:
            cols = []
            for i in range(samples.shape[1]):
                self._physics.set_lamda(samples[:, i])
                cols.append(self._physics.residual(self._state, self._time))
            self._physics.set_lamda(self._lam_base)
            return bkd.stack(cols, axis=1)
        self._physics.set_lamda(samples)
        res = self._physics.residual(self._state, self._time).reshape(-1, 1)
        self._physics.set_lamda(self._lam_base)
        return res

    def jacobian(self, sample):
        if sample.ndim == 2:
            sample = sample[:, 0]
        bkd = self._bkd
        self._physics.set_lamda(sample)
        npts = self.nvars()
        nstates = self.nqoi()
        jac = bkd.zeros((nstates, npts))
        jac = bkd.copy(jac)
        for j in range(npts):
            delta = bkd.zeros((npts,))
            delta = bkd.copy(delta)
            delta[j] = 1.0
            col = self._physics.residual_lamda_sensitivity(
                self._state,
                self._time,
                delta,
            )
            for k in range(nstates):
                jac[k, j] = col[k]
        self._physics.set_lamda(self._lam_base)
        return jac


class _FluxComponentOfState(Generic[Array]):
    """Wraps one PK1 stress component P_iJ(state) for DerivativeChecker.

    Input: state (nstates,1). Output: P_iJ (npts,1).
    Jacobian: compute_flux_jacobian[i][j].
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
                flux = self._physics.compute_flux(samples[:, i])
                cols.append(flux[self._row][self._col])
            return bkd.stack(cols, axis=1)
        flux = self._physics.compute_flux(samples)
        return flux[self._row][self._col].reshape(-1, 1)

    def jacobian(self, sample):
        if sample.ndim == 2:
            sample = sample[:, 0]
        flux_jac = self._physics.compute_flux_jacobian(sample)
        return flux_jac[self._row][self._col]


class TestHyperelasticSensitivity2D:
    """Test 2D residual sensitivities and flux for HyperelasticityPhysics."""
    def test_residual_mu_sensitivity_2d(self, bkd):
        """DerivativeChecker validates dR/dmu in 2D."""
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
        physics.set_mu(mu0)

        # Fixed state (small deformation, J > 0)
        np.random.seed(43)
        state = bkd.asarray(np.random.randn(nstates) * 0.01)

        wrapper = _ResidualOfMu(physics, state, mu0, bkd)
        checker = DerivativeChecker(wrapper)
        sample = mu0[:, None]
        errors = checker.check_derivatives(sample, verbosity=0)
        assert float(checker.error_ratio(errors[0])) <= 1e-6

    def test_residual_lamda_sensitivity_2d(self, bkd):
        """DerivativeChecker validates dR/dlam in 2D."""
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
        physics.set_lamda(lam0)

        # Fixed state
        np.random.seed(43)
        state = bkd.asarray(np.random.randn(nstates) * 0.01)

        wrapper = _ResidualOfLamda(physics, state, lam0, bkd)
        checker = DerivativeChecker(wrapper)
        sample = lam0[:, None]
        errors = checker.check_derivatives(sample, verbosity=0)
        assert float(checker.error_ratio(errors[0])) <= 1e-6

    def test_compute_flux_2d_matches_residual(self, bkd):
        """Verify div(P) from compute_flux matches residual (no forcing)."""
        physics, basis = _setup_2d_hyperelastic(bkd, npts_1d=6)
        nstates = physics.nstates()

        np.random.seed(42)
        state = bkd.asarray(np.random.randn(nstates) * 0.01)

        flux = physics.compute_flux(state)
        P11, P12 = flux[0]
        P21, P22 = flux[1]

        Dx = basis.derivative_matrix(1, 0)
        Dy = basis.derivative_matrix(1, 1)
        div_P_x = Dx @ P11 + Dy @ P12
        div_P_y = Dx @ P21 + Dy @ P22
        div_P = bkd.concatenate([div_P_x, div_P_y])

        residual = physics.residual(state, 0.0)
        bkd.assert_allclose(div_P, residual, rtol=1e-12)

    def test_compute_flux_jacobian_2d(self, bkd):
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
