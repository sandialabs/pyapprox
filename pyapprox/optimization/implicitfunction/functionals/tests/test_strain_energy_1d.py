"""Tests for StrainEnergyFunctional1D and factory functions."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
)
from pyapprox.optimization.implicitfunction.functionals.strain_energy_1d import (
    StrainEnergyFunctional1D,
    create_linear_strain_energy_1d,
    create_neo_hookean_strain_energy_1d,
)
from pyapprox.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


class TestStrainEnergyFunctional1D(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._npts = 15
        self._length = 2.0
        self._nparams = 3
        self._E = 3.0
        self._nu = 0.3
        transform = AffineTransform1D((0.0, self._length), self._bkd)
        mesh = TransformedMesh1D(self._npts, self._bkd, transform)
        self._basis = ChebyshevBasis1D(mesh, self._bkd)
        self._phys_pts = mesh.points()[0, :]  # shape (npts,)

    def _lame_params(self, E: float, nu: float):
        mu = E / (2.0 * (1.0 + nu))
        lamda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lamda, mu

    # ------------------------------------------------------------------
    # Linear elastic tests
    # ------------------------------------------------------------------

    def test_constant_strain_linear(self) -> None:
        """u(x) = eps*x gives W = (1/2)*E*eps^2*L."""
        bkd = self._bkd
        eps = 0.05
        func = create_linear_strain_energy_1d(
            self._basis, self._nparams, bkd, self._E,
        )
        state = bkd.reshape(eps * self._phys_pts, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[0.5 * self._E * eps ** 2 * self._length]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_zero_displacement_zero_energy(self) -> None:
        """Zero displacement gives zero strain energy."""
        bkd = self._bkd
        func = create_linear_strain_energy_1d(
            self._basis, self._nparams, bkd, self._E,
        )
        state = bkd.zeros((self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[0.0]])
        bkd.assert_allclose(result, expected, atol=1e-15)

    def test_quadratic_displacement_linear(self) -> None:
        """u(x) = x^2: eps = 2x, W = integral (1/2)*E*(2x)^2 dx = 2*E*L^3/3."""
        bkd = self._bkd
        func = create_linear_strain_energy_1d(
            self._basis, self._nparams, bkd, self._E,
        )
        state = bkd.reshape(self._phys_pts ** 2, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        # integral_0^L (1/2)*E*(2x)^2 dx = 2*E*L^3/3
        expected = bkd.asarray([[2.0 * self._E * self._length ** 3 / 3.0]])
        bkd.assert_allclose(result, expected, atol=1e-11)

    def test_linear_jacobian_derivative_checker(self) -> None:
        """DerivativeChecker validates linear elastic state Jacobian."""
        bkd = self._bkd
        func = create_linear_strain_energy_1d(
            self._basis, self._nparams, bkd, self._E,
        )
        param = bkd.zeros((self._nparams, 1))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=self._npts,
            fun=lambda s: func(s, param),
            jacobian=lambda s: func.state_jacobian(s, param),
            bkd=bkd,
        )
        # Use a polynomial displacement field
        np.random.seed(42)
        state = bkd.reshape(
            bkd.array(0.1 * np.random.randn(self._npts)),
            (self._npts, 1),
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            state, direction=None, relative=True
        )[0]
        self.assertLessEqual(checker.error_ratio(errors), 1e-6)

    # ------------------------------------------------------------------
    # Neo-Hookean tests
    # ------------------------------------------------------------------

    def test_neo_hookean_zero_displacement(self) -> None:
        """Zero displacement (F=1): Neo-Hookean strain energy is zero."""
        bkd = self._bkd
        lamda, mu = self._lame_params(self._E, self._nu)
        func = create_neo_hookean_strain_energy_1d(
            self._basis, self._nparams, bkd, lamda, mu,
        )
        state = bkd.zeros((self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[0.0]])
        bkd.assert_allclose(result, expected, atol=1e-14)

    def test_neo_hookean_constant_deformation(self) -> None:
        """u(x) = eps*x: F = 1+eps everywhere. Verify energy analytically."""
        bkd = self._bkd
        lamda, mu = self._lame_params(self._E, self._nu)
        func = create_neo_hookean_strain_energy_1d(
            self._basis, self._nparams, bkd, lamda, mu,
        )
        eps = 0.05
        state = bkd.reshape(eps * self._phys_pts, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)

        # psi(F) = mu/2*(F^2-1-2*ln(F)) + lam/2*(ln(F))^2
        F = 1.0 + eps
        ln_F = np.log(F)
        psi = mu / 2.0 * (F ** 2 - 1.0 - 2.0 * ln_F) + lamda / 2.0 * ln_F ** 2
        expected = bkd.asarray([[psi * self._length]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_neo_hookean_jacobian_derivative_checker(self) -> None:
        """DerivativeChecker validates Neo-Hookean state Jacobian."""
        bkd = self._bkd
        lamda, mu = self._lame_params(self._E, self._nu)
        func = create_neo_hookean_strain_energy_1d(
            self._basis, self._nparams, bkd, lamda, mu,
        )
        param = bkd.zeros((self._nparams, 1))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=self._npts,
            fun=lambda s: func(s, param),
            jacobian=lambda s: func.state_jacobian(s, param),
            bkd=bkd,
        )
        # Use smooth displacement u = 0.02*x*(L-x) to keep F well positive
        state = bkd.reshape(
            0.02 * self._phys_pts * (self._length - self._phys_pts),
            (self._npts, 1),
        )
        # Cap max FD eps at 1e-2 to avoid driving F negative under perturbation
        fd_eps = bkd.flip(bkd.logspace(-13, -2, 12))
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            state, fd_eps=fd_eps, direction=None, relative=True
        )[0]
        self.assertLessEqual(checker.error_ratio(errors), 1e-6)

    def test_neo_hookean_small_strain_matches_linear(self) -> None:
        """For small strains, Neo-Hookean energy ~ linear elastic energy."""
        bkd = self._bkd
        lamda, mu = self._lame_params(self._E, self._nu)
        eps = 0.001  # very small strain

        func_nh = create_neo_hookean_strain_energy_1d(
            self._basis, self._nparams, bkd, lamda, mu,
        )
        # Linear elastic with effective modulus = lamda + 2*mu (plane strain 1D)
        # Actually for 1D: linearized Neo-Hookean gives
        # P ~ (lamda + 2*mu)*eps at small eps, so psi ~ (lamda+2mu)/2 * eps^2
        E_eff = lamda + 2.0 * mu
        func_lin = create_linear_strain_energy_1d(
            self._basis, self._nparams, bkd, E_eff,
        )

        state = bkd.reshape(eps * self._phys_pts, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))

        W_nh = func_nh(state, param)
        W_lin = func_lin(state, param)
        bkd.assert_allclose(W_nh, W_lin, rtol=1e-2)

    # ------------------------------------------------------------------
    # Subdomain and factory tests
    # ------------------------------------------------------------------

    def test_subdomain_strain_energy(self) -> None:
        """Strain energy over half domain."""
        bkd = self._bkd
        eps = 0.05
        half_L = self._length / 2.0
        func_half = create_linear_strain_energy_1d(
            self._basis, self._nparams, bkd, self._E,
            a_sub=0.0, b_sub=half_L,
        )
        state = bkd.reshape(eps * self._phys_pts, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func_half(state, param)
        expected = bkd.asarray([[0.5 * self._E * eps ** 2 * half_L]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_factory_linear(self) -> None:
        """Linear factory creates a valid functional."""
        bkd = self._bkd
        func = create_linear_strain_energy_1d(
            self._basis, self._nparams, bkd, self._E,
        )
        self.assertEqual(func.nqoi(), 1)
        self.assertEqual(func.nstates(), self._npts)
        self.assertEqual(func.nparams(), self._nparams)
        self.assertEqual(func.nunique_params(), 0)

    def test_factory_neo_hookean(self) -> None:
        """Neo-Hookean factory creates a valid functional."""
        bkd = self._bkd
        lamda, mu = self._lame_params(self._E, self._nu)
        func = create_neo_hookean_strain_energy_1d(
            self._basis, self._nparams, bkd, lamda, mu,
        )
        self.assertEqual(func.nqoi(), 1)
        self.assertEqual(func.nstates(), self._npts)
        self.assertEqual(func.nparams(), self._nparams)

    def test_protocol_compliance_linear(self) -> None:
        """Linear strain energy satisfies the protocol."""
        bkd = self._bkd
        func = create_linear_strain_energy_1d(
            self._basis, self._nparams, bkd, self._E,
        )
        self.assertIsInstance(func, ParameterizedFunctionalWithJacobianProtocol)

    def test_protocol_compliance_neo_hookean(self) -> None:
        """Neo-Hookean strain energy satisfies the protocol."""
        bkd = self._bkd
        lamda, mu = self._lame_params(self._E, self._nu)
        func = create_neo_hookean_strain_energy_1d(
            self._basis, self._nparams, bkd, lamda, mu,
        )
        self.assertIsInstance(func, ParameterizedFunctionalWithJacobianProtocol)

    def test_param_jacobian_is_zero(self) -> None:
        """Strain energy functional has no parameter dependence."""
        bkd = self._bkd
        func = create_linear_strain_energy_1d(
            self._basis, self._nparams, bkd, self._E,
        )
        state = bkd.ones((self._npts, 1))
        param = bkd.ones((self._nparams, 1))
        jac = func.param_jacobian(state, param)
        expected = bkd.zeros((1, self._nparams))
        bkd.assert_allclose(jac, expected)


class TestStrainEnergyFunctional1DNumpy(
    TestStrainEnergyFunctional1D[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestStrainEnergyFunctional1DTorch(
    TestStrainEnergyFunctional1D[torch.Tensor]
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()
