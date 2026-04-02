"""Tests for SubdomainIntegralFunctional."""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)
from pyapprox.optimization.implicitfunction.functionals.subdomain_integral import (
    SubdomainIntegralFunctional,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
)

# TODO: this is specific to collocation, should it go in
# collocation module or in benchmark module


class TestSubdomainIntegralFunctional:

    def _make_basis(self, bkd):
        npts = 15
        length = 2.0
        nparams = 3
        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        basis = ChebyshevBasis1D(mesh, bkd)
        phys_pts = mesh.points()[0, :]  # shape (npts,)
        return basis, phys_pts, npts, length, nparams

    def test_constant_function(self, bkd) -> None:
        """Integral of c=1, u=1 over [0, L] gives L."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        coeff = bkd.ones((npts,))
        func = SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            coefficient=coeff,
        )
        state = bkd.ones((npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[length]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_polynomial_exactness(self, bkd) -> None:
        """Integral of x^2 over [0, L] = L^3/3."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        coeff = bkd.ones((npts,))
        func = SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            coefficient=coeff,
        )
        state = bkd.reshape(phys_pts**2, (npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[length**3 / 3.0]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_weighted_polynomial(self, bkd) -> None:
        """Integral of c(x)*u(x) with c=x, u=x gives integral x^2 = L^3/3."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        coeff = phys_pts  # c(x) = x
        func = SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            coefficient=coeff,
        )
        state = bkd.reshape(phys_pts, (npts, 1))  # u = x
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[length**3 / 3.0]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_linear_jacobian_constant(self, bkd) -> None:
        """For linear integrand, state Jacobian is independent of state."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        coeff = bkd.ones((npts,))
        func = SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            coefficient=coeff,
        )
        param = bkd.zeros((nparams, 1))
        state1 = bkd.zeros((npts, 1))
        state2 = bkd.ones((npts, 1))
        jac1 = func.state_jacobian(state1, param)
        jac2 = func.state_jacobian(state2, param)
        bkd.assert_allclose(jac1, jac2, atol=1e-15)

    def test_subdomain_integration(self, bkd) -> None:
        """Integral of x^2 over [0.5, 1.5] = (1.5^3 - 0.5^3)/3."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        coeff = bkd.ones((npts,))
        func = SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            a_sub=0.5,
            b_sub=1.5,
            coefficient=coeff,
        )
        state = bkd.reshape(phys_pts**2, (npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[(1.5**3 - 0.5**3) / 3.0]])
        bkd.assert_allclose(result, expected, atol=1e-11)

    def test_full_domain_default(self, bkd) -> None:
        """None bounds give same result as explicit full domain bounds."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        coeff = bkd.ones((npts,))
        func_default = SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            coefficient=coeff,
        )
        func_explicit = SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            a_sub=0.0,
            b_sub=length,
            coefficient=coeff,
        )
        state = bkd.reshape(phys_pts**2, (npts, 1))
        param = bkd.zeros((nparams, 1))
        r1 = func_default(state, param)
        r2 = func_explicit(state, param)
        bkd.assert_allclose(r1, r2, atol=1e-12)

    def test_nonlinear_integrand(self, bkd) -> None:
        """Nonlinear integrand g(u) = u^2 verified with DerivativeChecker."""

        def integrand(u, bkd):
            return u**2, 2.0 * u

        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        func = SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            integrand=integrand,
        )
        param = bkd.zeros((nparams, 1))

        # Wrap for DerivativeChecker
        def fun(sample):
            return func(sample, param)

        def jac(sample):
            return func.state_jacobian(sample, param)

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=npts,
            fun=fun,
            jacobian=jac,
            bkd=bkd,
        )
        np.random.seed(42)
        state = bkd.reshape(
            bkd.array(np.random.rand(npts) + 0.5),
            (npts, 1),
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state, direction=None, relative=True)[0]
        assert checker.error_ratio(errors) <= 2e-6

    def test_nonlinear_value(self, bkd) -> None:
        """Integral of u^2 where u(x)=x gives integral x^4 = L^5/5."""

        def integrand(u, bkd):
            return u**2, 2.0 * u

        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        func = SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            integrand=integrand,
        )
        # u(x) = x^2, so g(u) = u^2 = x^4
        state = bkd.reshape(phys_pts**2, (npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        # integral x^4 dx from 0 to L = L^5/5
        expected = bkd.asarray([[length**5 / 5.0]])
        bkd.assert_allclose(result, expected, atol=1e-11)

    def test_protocol_compliance(self, bkd) -> None:
        """Satisfies ParameterizedFunctionalWithJacobianProtocol."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        coeff = bkd.ones((npts,))
        func = SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            coefficient=coeff,
        )
        assert isinstance(func, ParameterizedFunctionalWithJacobianProtocol)

    def test_validation_neither_provided(self, bkd) -> None:
        """Raises ValueError if neither coefficient nor integrand given."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        with pytest.raises(ValueError):
            SubdomainIntegralFunctional(
                basis,
                nparams,
                bkd,
            )

    def test_validation_both_provided(self, bkd) -> None:
        """Raises ValueError if both coefficient and integrand given."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        with pytest.raises(ValueError):
            SubdomainIntegralFunctional(
                basis,
                nparams,
                bkd,
                coefficient=bkd.ones((npts,)),
                integrand=lambda u, bkd: (u, bkd.ones_like(u)),
            )

    def test_param_jacobian_is_zero(self, bkd) -> None:
        """Functional has no parameter dependence."""
        basis, phys_pts, npts, length, nparams = self._make_basis(bkd)
        coeff = bkd.ones((npts,))
        func = SubdomainIntegralFunctional(
            basis,
            nparams,
            bkd,
            coefficient=coeff,
        )
        state = bkd.ones((npts, 1))
        param = bkd.ones((nparams, 1))
        jac = func.param_jacobian(state, param)
        expected = bkd.zeros((1, nparams))
        bkd.assert_allclose(jac, expected)
