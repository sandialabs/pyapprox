"""Tests for StrainEnergyFunctional1D and factory functions."""

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.optimization.implicitfunction.functionals.protocols import (
    ParameterizedFunctionalWithJacobianProtocol,
)
from pyapprox.optimization.implicitfunction.functionals.strain_energy_1d import (
    create_linear_strain_energy_1d,
    create_neo_hookean_strain_energy_1d,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
)

# TODO: this is specific to collocation, should it go in collocation module or in benchmark module



class TestStrainEnergyFunctional1D:

    def _make_basis(self, bkd):
        npts = 15
        length = 2.0
        nparams = 3
        E = 3.0
        nu = 0.3
        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        basis = ChebyshevBasis1D(mesh, bkd)
        phys_pts = mesh.points()[0, :]  # shape (npts,)
        return basis, phys_pts, npts, length, nparams, E, nu

    def _lame_params(self, E: float, nu: float):
        mu = E / (2.0 * (1.0 + nu))
        lamda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lamda, mu

    # ------------------------------------------------------------------
    # Linear elastic tests
    # ------------------------------------------------------------------

    def test_constant_strain_linear(self, bkd) -> None:
        """u(x) = eps*x gives W = (1/2)*E*eps^2*L."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        eps = 0.05
        func = create_linear_strain_energy_1d(
            basis,
            nparams,
            bkd,
            E,
        )
        state = bkd.reshape(eps * phys_pts, (npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[0.5 * E * eps**2 * length]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_zero_displacement_zero_energy(self, bkd) -> None:
        """Zero displacement gives zero strain energy."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        func = create_linear_strain_energy_1d(
            basis,
            nparams,
            bkd,
            E,
        )
        state = bkd.zeros((npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[0.0]])
        bkd.assert_allclose(result, expected, atol=1e-15)

    def test_quadratic_displacement_linear(self, bkd) -> None:
        """u(x) = x^2: eps = 2x, W = integral (1/2)*E*(2x)^2 dx = 2*E*L^3/3."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        func = create_linear_strain_energy_1d(
            basis,
            nparams,
            bkd,
            E,
        )
        state = bkd.reshape(phys_pts**2, (npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        # integral_0^L (1/2)*E*(2x)^2 dx = 2*E*L^3/3
        expected = bkd.asarray([[2.0 * E * length**3 / 3.0]])
        bkd.assert_allclose(result, expected, atol=1e-11)

    def test_linear_jacobian_derivative_checker(self, bkd) -> None:
        """DerivativeChecker validates linear elastic state Jacobian."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        func = create_linear_strain_energy_1d(
            basis,
            nparams,
            bkd,
            E,
        )
        param = bkd.zeros((nparams, 1))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=npts,
            fun=lambda s: func(s, param),
            jacobian=lambda s: func.state_jacobian(s, param),
            bkd=bkd,
        )
        # Use a polynomial displacement field
        np.random.seed(42)
        state = bkd.reshape(
            bkd.array(0.1 * np.random.randn(npts)),
            (npts, 1),
        )
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(state, direction=None, relative=True)[0]
        assert checker.error_ratio(errors) <= 1e-6

    # ------------------------------------------------------------------
    # Neo-Hookean tests
    # ------------------------------------------------------------------

    def test_neo_hookean_zero_displacement(self, bkd) -> None:
        """Zero displacement (F=1): Neo-Hookean strain energy is zero."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        lamda, mu = self._lame_params(E, nu)
        func = create_neo_hookean_strain_energy_1d(
            basis,
            nparams,
            bkd,
            lamda,
            mu,
        )
        state = bkd.zeros((npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)
        expected = bkd.asarray([[0.0]])
        bkd.assert_allclose(result, expected, atol=1e-14)

    def test_neo_hookean_constant_deformation(self, bkd) -> None:
        """u(x) = eps*x: F = 1+eps everywhere. Verify energy analytically."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        lamda, mu = self._lame_params(E, nu)
        func = create_neo_hookean_strain_energy_1d(
            basis,
            nparams,
            bkd,
            lamda,
            mu,
        )
        eps = 0.05
        state = bkd.reshape(eps * phys_pts, (npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func(state, param)

        # psi(F) = mu/2*(F^2-1-2*ln(F)) + lam/2*(ln(F))^2
        F = 1.0 + eps
        ln_F = np.log(F)
        psi = mu / 2.0 * (F**2 - 1.0 - 2.0 * ln_F) + lamda / 2.0 * ln_F**2
        expected = bkd.asarray([[psi * length]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_neo_hookean_jacobian_derivative_checker(self, bkd) -> None:
        """DerivativeChecker validates Neo-Hookean state Jacobian."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        lamda, mu = self._lame_params(E, nu)
        func = create_neo_hookean_strain_energy_1d(
            basis,
            nparams,
            bkd,
            lamda,
            mu,
        )
        param = bkd.zeros((nparams, 1))

        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=npts,
            fun=lambda s: func(s, param),
            jacobian=lambda s: func.state_jacobian(s, param),
            bkd=bkd,
        )
        # Use smooth displacement u = 0.02*x*(L-x) to keep F well positive
        state = bkd.reshape(
            0.02 * phys_pts * (length - phys_pts),
            (npts, 1),
        )
        # Cap max FD eps at 1e-2 to avoid driving F negative under perturbation
        fd_eps = bkd.flip(bkd.logspace(-13, -2, 12))
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            state, fd_eps=fd_eps, direction=None, relative=True
        )[0]
        assert checker.error_ratio(errors) <= 1e-6

    def test_neo_hookean_small_strain_matches_linear(self, bkd) -> None:
        """For small strains, Neo-Hookean energy ~ linear elastic energy."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        lamda, mu = self._lame_params(E, nu)
        eps = 0.001  # very small strain

        func_nh = create_neo_hookean_strain_energy_1d(
            basis,
            nparams,
            bkd,
            lamda,
            mu,
        )
        # Linear elastic with effective modulus = lamda + 2*mu (plane strain 1D)
        # Actually for 1D: linearized Neo-Hookean gives
        # P ~ (lamda + 2*mu)*eps at small eps, so psi ~ (lamda+2mu)/2 * eps^2
        E_eff = lamda + 2.0 * mu
        func_lin = create_linear_strain_energy_1d(
            basis,
            nparams,
            bkd,
            E_eff,
        )

        state = bkd.reshape(eps * phys_pts, (npts, 1))
        param = bkd.zeros((nparams, 1))

        W_nh = func_nh(state, param)
        W_lin = func_lin(state, param)
        bkd.assert_allclose(W_nh, W_lin, rtol=1e-2)

    # ------------------------------------------------------------------
    # Subdomain and factory tests
    # ------------------------------------------------------------------

    def test_subdomain_strain_energy(self, bkd) -> None:
        """Strain energy over half domain."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        eps = 0.05
        half_L = length / 2.0
        func_half = create_linear_strain_energy_1d(
            basis,
            nparams,
            bkd,
            E,
            a_sub=0.0,
            b_sub=half_L,
        )
        state = bkd.reshape(eps * phys_pts, (npts, 1))
        param = bkd.zeros((nparams, 1))
        result = func_half(state, param)
        expected = bkd.asarray([[0.5 * E * eps**2 * half_L]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_factory_linear(self, bkd) -> None:
        """Linear factory creates a valid functional."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        func = create_linear_strain_energy_1d(
            basis,
            nparams,
            bkd,
            E,
        )
        assert func.nqoi() == 1
        assert func.nstates() == npts
        assert func.nparams() == nparams
        assert func.nunique_params() == 0

    def test_factory_neo_hookean(self, bkd) -> None:
        """Neo-Hookean factory creates a valid functional."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        lamda, mu = self._lame_params(E, nu)
        func = create_neo_hookean_strain_energy_1d(
            basis,
            nparams,
            bkd,
            lamda,
            mu,
        )
        assert func.nqoi() == 1
        assert func.nstates() == npts
        assert func.nparams() == nparams

    def test_protocol_compliance_linear(self, bkd) -> None:
        """Linear strain energy satisfies the protocol."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        func = create_linear_strain_energy_1d(
            basis,
            nparams,
            bkd,
            E,
        )
        assert isinstance(func, ParameterizedFunctionalWithJacobianProtocol)

    def test_protocol_compliance_neo_hookean(self, bkd) -> None:
        """Neo-Hookean strain energy satisfies the protocol."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        lamda, mu = self._lame_params(E, nu)
        func = create_neo_hookean_strain_energy_1d(
            basis,
            nparams,
            bkd,
            lamda,
            mu,
        )
        assert isinstance(func, ParameterizedFunctionalWithJacobianProtocol)

    def test_param_jacobian_is_zero(self, bkd) -> None:
        """Strain energy functional has no parameter dependence."""
        basis, phys_pts, npts, length, nparams, E, nu = self._make_basis(bkd)
        func = create_linear_strain_energy_1d(
            basis,
            nparams,
            bkd,
            E,
        )
        state = bkd.ones((npts, 1))
        param = bkd.ones((nparams, 1))
        jac = func.param_jacobian(state, param)
        expected = bkd.zeros((1, nparams))
        bkd.assert_allclose(jac, expected)
