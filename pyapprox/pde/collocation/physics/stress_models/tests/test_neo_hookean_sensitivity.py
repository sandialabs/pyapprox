"""Tests for NeoHookean 2D material parameter sensitivities.

Verifies dP/dmu and dP/dlambda in 2D via:
1. DerivativeChecker FD validation
2. Identity deformation (F=I) gives zero sensitivity
3. Consistency with 1D methods for diagonal F
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.pde.collocation.physics.stress_models.neo_hookean import (
    NeoHookeanStress,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


class _StressOfMu(Generic[Array]):
    """Wraps P(F; mu) as function of scalar mu for DerivativeChecker.

    Input: mu (1,1). Output: [P11, P12, P21, P22] stacked (4*npts, 1).
    Jacobian: [dP11/dmu, dP12/dmu, dP21/dmu, dP22/dmu] stacked (4*npts, 1).
    """

    def __init__(self, lamda, F11, F12, F21, F22, bkd):
        self._lamda = lamda
        self._F11 = F11
        self._F12 = F12
        self._F21 = F21
        self._F22 = F22
        self._bkd = bkd
        self._npts = F11.shape[0]

    def bkd(self):
        return self._bkd

    def nvars(self):
        return 1

    def nqoi(self):
        return 4 * self._npts

    def __call__(self, samples):
        bkd = self._bkd
        if samples.ndim == 2:
            cols = []
            for i in range(samples.shape[1]):
                mu_val = float(samples[0, i])
                model = NeoHookeanStress(self._lamda, mu_val)
                P = model.compute_stress_2d(
                    self._F11, self._F12, self._F21, self._F22, bkd,
                )
                cols.append(bkd.concatenate(list(P)))
            return bkd.stack(cols, axis=1)
        mu_val = float(samples[0])
        model = NeoHookeanStress(self._lamda, mu_val)
        P = model.compute_stress_2d(
            self._F11, self._F12, self._F21, self._F22, bkd,
        )
        return bkd.concatenate(list(P)).reshape(-1, 1)

    def jacobian(self, sample):
        bkd = self._bkd
        if sample.ndim == 2:
            mu_val = float(sample[0, 0])
        else:
            mu_val = float(sample[0])
        model = NeoHookeanStress(self._lamda, mu_val)
        dP = model.stress_sensitivity_mu_2d(
            self._F11, self._F12, self._F21, self._F22, bkd,
        )
        return bkd.concatenate(list(dP)).reshape(-1, 1)


class _StressOfLamda(Generic[Array]):
    """Wraps P(F; lamda) as function of scalar lamda for DerivativeChecker."""

    def __init__(self, mu, F11, F12, F21, F22, bkd):
        self._mu = mu
        self._F11 = F11
        self._F12 = F12
        self._F21 = F21
        self._F22 = F22
        self._bkd = bkd
        self._npts = F11.shape[0]

    def bkd(self):
        return self._bkd

    def nvars(self):
        return 1

    def nqoi(self):
        return 4 * self._npts

    def __call__(self, samples):
        bkd = self._bkd
        if samples.ndim == 2:
            cols = []
            for i in range(samples.shape[1]):
                lam_val = float(samples[0, i])
                model = NeoHookeanStress(lam_val, self._mu)
                P = model.compute_stress_2d(
                    self._F11, self._F12, self._F21, self._F22, bkd,
                )
                cols.append(bkd.concatenate(list(P)))
            return bkd.stack(cols, axis=1)
        lam_val = float(samples[0])
        model = NeoHookeanStress(lam_val, self._mu)
        P = model.compute_stress_2d(
            self._F11, self._F12, self._F21, self._F22, bkd,
        )
        return bkd.concatenate(list(P)).reshape(-1, 1)

    def jacobian(self, sample):
        bkd = self._bkd
        if sample.ndim == 2:
            lam_val = float(sample[0, 0])
        else:
            lam_val = float(sample[0])
        model = NeoHookeanStress(lam_val, self._mu)
        dP = model.stress_sensitivity_lamda_2d(
            self._F11, self._F12, self._F21, self._F22, bkd,
        )
        return bkd.concatenate(list(dP)).reshape(-1, 1)


class TestNeoHookeanSensitivity2D(Generic[Array], unittest.TestCase):
    """Test NeoHookean 2D material parameter sensitivities."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._lamda = 2.0
        self._mu = 1.5
        self._stress_model = NeoHookeanStress(self._lamda, self._mu)

    def _random_F_near_identity(self, npts, seed=42):
        """Generate F components near identity (small deformation)."""
        bkd = self._bkd
        np.random.seed(seed)
        eps = 0.1
        F11 = bkd.asarray(1.0 + eps * np.random.randn(npts))
        F12 = bkd.asarray(eps * np.random.randn(npts))
        F21 = bkd.asarray(eps * np.random.randn(npts))
        F22 = bkd.asarray(1.0 + eps * np.random.randn(npts))
        return F11, F12, F21, F22

    def test_stress_sensitivity_mu_2d(self):
        """DerivativeChecker validates dP/dmu for all 4 stress components."""
        bkd = self._bkd
        npts = 10
        F11, F12, F21, F22 = self._random_F_near_identity(npts)

        wrapper = _StressOfMu(self._lamda, F11, F12, F21, F22, bkd)
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray([[self._mu]])
        errors = checker.check_derivatives(sample, verbosity=0)
        self.assertLessEqual(float(checker.error_ratio(errors[0])), 1e-6)

    def test_stress_sensitivity_lamda_2d(self):
        """DerivativeChecker validates dP/dlambda for all 4 stress components."""
        bkd = self._bkd
        npts = 10
        F11, F12, F21, F22 = self._random_F_near_identity(npts)

        wrapper = _StressOfLamda(self._mu, F11, F12, F21, F22, bkd)
        checker = DerivativeChecker(wrapper)
        sample = bkd.asarray([[self._lamda]])
        errors = checker.check_derivatives(sample, verbosity=0)
        self.assertLessEqual(float(checker.error_ratio(errors[0])), 1e-6)

    def test_sensitivity_mu_2d_identity(self):
        """At F=I (no deformation), dP/dmu = F - F^{-T} = I - I = 0."""
        bkd = self._bkd
        npts = 5
        F11 = bkd.ones((npts,))
        F12 = bkd.zeros((npts,))
        F21 = bkd.zeros((npts,))
        F22 = bkd.ones((npts,))

        dP11, dP12, dP21, dP22 = self._stress_model.stress_sensitivity_mu_2d(
            F11, F12, F21, F22, bkd
        )

        zeros = bkd.zeros((npts,))
        bkd.assert_allclose(dP11, zeros, atol=1e-14)
        bkd.assert_allclose(dP12, zeros, atol=1e-14)
        bkd.assert_allclose(dP21, zeros, atol=1e-14)
        bkd.assert_allclose(dP22, zeros, atol=1e-14)

    def test_sensitivity_lamda_2d_identity(self):
        """At F=I, J=1 so ln(J)=0, thus dP/dlam = 0."""
        bkd = self._bkd
        npts = 5
        F11 = bkd.ones((npts,))
        F12 = bkd.zeros((npts,))
        F21 = bkd.zeros((npts,))
        F22 = bkd.ones((npts,))

        dP11, dP12, dP21, dP22 = (
            self._stress_model.stress_sensitivity_lamda_2d(
                F11, F12, F21, F22, bkd
            )
        )

        zeros = bkd.zeros((npts,))
        bkd.assert_allclose(dP11, zeros, atol=1e-14)
        bkd.assert_allclose(dP12, zeros, atol=1e-14)
        bkd.assert_allclose(dP21, zeros, atol=1e-14)
        bkd.assert_allclose(dP22, zeros, atol=1e-14)

    def test_consistency_1d_2d(self):
        """Diagonal F (F12=F21=0) matches 1D sensitivity for (1,1) component."""
        bkd = self._bkd
        npts = 8
        np.random.seed(99)
        F_vals = bkd.asarray(1.0 + 0.2 * np.random.randn(npts))

        # 1D
        dP_dmu_1d = self._stress_model.stress_sensitivity_mu_1d(F_vals, bkd)
        dP_dlam_1d = self._stress_model.stress_sensitivity_lamda_1d(
            F_vals, bkd
        )

        # 2D with diagonal F: F11=F, F22=1, F12=F21=0
        F11 = F_vals
        F12 = bkd.zeros((npts,))
        F21 = bkd.zeros((npts,))
        F22 = bkd.ones((npts,))

        dP_mu = self._stress_model.stress_sensitivity_mu_2d(
            F11, F12, F21, F22, bkd
        )
        dP_lam = self._stress_model.stress_sensitivity_lamda_2d(
            F11, F12, F21, F22, bkd
        )

        bkd.assert_allclose(dP_mu[0], dP_dmu_1d, rtol=1e-12)
        bkd.assert_allclose(dP_lam[0], dP_dlam_1d, rtol=1e-12)


class TestNeoHookeanSensitivity2DNumpy(
    TestNeoHookeanSensitivity2D[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestNeoHookeanSensitivity2DTorch(
    TestNeoHookeanSensitivity2D[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        super().setUp()
