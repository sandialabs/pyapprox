"""Tests for SubdomainIntegralFunctional."""

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
from pyapprox.optimization.implicitfunction.functionals.subdomain_integral import (
    SubdomainIntegralFunctional,
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


class TestSubdomainIntegralFunctional(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        self._npts = 15
        self._length = 2.0
        self._nparams = 3
        transform = AffineTransform1D((0.0, self._length), self._bkd)
        mesh = TransformedMesh1D(self._npts, self._bkd, transform)
        self._basis = ChebyshevBasis1D(mesh, self._bkd)
        self._phys_pts = mesh.points()[0, :]  # shape (npts,)

    def test_constant_function(self) -> None:
        """Integral of c=1, u=1 over [0, L] gives L."""
        bkd = self._bkd
        coeff = bkd.ones((self._npts,))
        func = SubdomainIntegralFunctional(
            self._basis, self._nparams, bkd, coefficient=coeff,
        )
        state = bkd.ones((self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[self._length]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_polynomial_exactness(self) -> None:
        """Integral of x^2 over [0, L] = L^3/3."""
        bkd = self._bkd
        coeff = bkd.ones((self._npts,))
        func = SubdomainIntegralFunctional(
            self._basis, self._nparams, bkd, coefficient=coeff,
        )
        state = bkd.reshape(self._phys_pts ** 2, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[self._length ** 3 / 3.0]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_weighted_polynomial(self) -> None:
        """Integral of c(x)*u(x) with c=x, u=x gives integral x^2 = L^3/3."""
        bkd = self._bkd
        coeff = self._phys_pts  # c(x) = x
        func = SubdomainIntegralFunctional(
            self._basis, self._nparams, bkd, coefficient=coeff,
        )
        state = bkd.reshape(self._phys_pts, (self._npts, 1))  # u = x
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[self._length ** 3 / 3.0]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_linear_jacobian_constant(self) -> None:
        """For linear integrand, state Jacobian is independent of state."""
        bkd = self._bkd
        coeff = bkd.ones((self._npts,))
        func = SubdomainIntegralFunctional(
            self._basis, self._nparams, bkd, coefficient=coeff,
        )
        param = bkd.zeros((self._nparams, 1))
        state1 = bkd.zeros((self._npts, 1))
        state2 = bkd.ones((self._npts, 1))
        jac1 = func.state_jacobian(state1, param)
        jac2 = func.state_jacobian(state2, param)
        bkd.assert_allclose(jac1, jac2, atol=1e-15)

    def test_subdomain_integration(self) -> None:
        """Integral of x^2 over [0.5, 1.5] = (1.5^3 - 0.5^3)/3."""
        bkd = self._bkd
        coeff = bkd.ones((self._npts,))
        func = SubdomainIntegralFunctional(
            self._basis, self._nparams, bkd,
            a_sub=0.5, b_sub=1.5, coefficient=coeff,
        )
        state = bkd.reshape(self._phys_pts ** 2, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[(1.5 ** 3 - 0.5 ** 3) / 3.0]])
        bkd.assert_allclose(result, expected, atol=1e-11)

    def test_full_domain_default(self) -> None:
        """None bounds give same result as explicit full domain bounds."""
        bkd = self._bkd
        coeff = bkd.ones((self._npts,))
        func_default = SubdomainIntegralFunctional(
            self._basis, self._nparams, bkd, coefficient=coeff,
        )
        func_explicit = SubdomainIntegralFunctional(
            self._basis, self._nparams, bkd,
            a_sub=0.0, b_sub=self._length, coefficient=coeff,
        )
        state = bkd.reshape(self._phys_pts ** 2, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        r1 = func_default(state, param)
        r2 = func_explicit(state, param)
        bkd.assert_allclose(r1, r2, atol=1e-12)

    def test_nonlinear_integrand(self) -> None:
        """Nonlinear integrand g(u) = u^2 verified with DerivativeChecker."""
        bkd = self._bkd

        def integrand(u, bkd):
            return u ** 2, 2.0 * u

        func = SubdomainIntegralFunctional(
            self._basis, self._nparams, bkd, integrand=integrand,
        )
        param = bkd.zeros((self._nparams, 1))

        # Wrap for DerivativeChecker
        def fun(sample):
            return func(sample, param)

        def jac(sample):
            return func.state_jacobian(sample, param)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1, nvars=self._npts, fun=fun, jacobian=jac, bkd=bkd,
        )
        np.random.seed(42)
        state = bkd.reshape(
            bkd.array(np.random.rand(self._npts) + 0.5),
            (self._npts, 1),
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            state, direction=None, relative=True
        )[0]
        self.assertLessEqual(checker.error_ratio(errors), 1e-6)

    def test_nonlinear_value(self) -> None:
        """Integral of u^2 where u(x)=x gives integral x^4 = L^5/5."""
        bkd = self._bkd

        def integrand(u, bkd):
            return u ** 2, 2.0 * u

        func = SubdomainIntegralFunctional(
            self._basis, self._nparams, bkd, integrand=integrand,
        )
        # u(x) = x^2, so g(u) = u^2 = x^4
        state = bkd.reshape(self._phys_pts ** 2, (self._npts, 1))
        param = bkd.zeros((self._nparams, 1))
        result = func(state, param)
        # integral x^4 dx from 0 to L = L^5/5
        expected = bkd.asarray([[self._length ** 5 / 5.0]])
        bkd.assert_allclose(result, expected, atol=1e-11)

    def test_protocol_compliance(self) -> None:
        """Satisfies ParameterizedFunctionalWithJacobianProtocol."""
        bkd = self._bkd
        coeff = bkd.ones((self._npts,))
        func = SubdomainIntegralFunctional(
            self._basis, self._nparams, bkd, coefficient=coeff,
        )
        self.assertIsInstance(func, ParameterizedFunctionalWithJacobianProtocol)

    def test_validation_neither_provided(self) -> None:
        """Raises ValueError if neither coefficient nor integrand given."""
        bkd = self._bkd
        with self.assertRaises(ValueError):
            SubdomainIntegralFunctional(
                self._basis, self._nparams, bkd,
            )

    def test_validation_both_provided(self) -> None:
        """Raises ValueError if both coefficient and integrand given."""
        bkd = self._bkd
        with self.assertRaises(ValueError):
            SubdomainIntegralFunctional(
                self._basis, self._nparams, bkd,
                coefficient=bkd.ones((self._npts,)),
                integrand=lambda u, bkd: (u, bkd.ones_like(u)),
            )

    def test_param_jacobian_is_zero(self) -> None:
        """Functional has no parameter dependence."""
        bkd = self._bkd
        coeff = bkd.ones((self._npts,))
        func = SubdomainIntegralFunctional(
            self._basis, self._nparams, bkd, coefficient=coeff,
        )
        state = bkd.ones((self._npts, 1))
        param = bkd.ones((self._nparams, 1))
        jac = func.param_jacobian(state, param)
        expected = bkd.zeros((1, self._nparams))
        bkd.assert_allclose(jac, expected)


class TestSubdomainIntegralFunctionalNumpy(
    TestSubdomainIntegralFunctional[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSubdomainIntegralFunctionalTorch(
    TestSubdomainIntegralFunctional[torch.Tensor]
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()
