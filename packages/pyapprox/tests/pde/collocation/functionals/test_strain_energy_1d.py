"""Tests for StrainEnergyFunctional1D, its densities and factory functions."""

import numpy as np
import pytest

from pyapprox.implicitfunction.functionals.protocols import (
    SteadyFunctionalProtocol,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.functionals.strain_energy_1d import (
    LinearStrainEnergyDensity1D,
    LinearStressDensity1D,
    NeoHookeanStrainEnergyDensity1D,
    NeoHookeanStressDensity1D,
    StrainEnergyFunctional1D,
    create_linear_strain_energy_1d,
    create_neo_hookean_strain_energy_1d,
)
from pyapprox.pde.collocation.mesh import (
    AffineTransform1D,
    TransformedMesh1D,
)

# TODO: this is specific to collocation, should it go in
# collocation module or in benchmark module

_NPARAMS = 3


class _AffineMap:
    """m(p) = base + modes @ p: the material at p = 0 is ``base``."""

    def __init__(self, bkd, base, seed=0):
        self._bkd = bkd
        self._base = base
        rng = np.random.default_rng(seed)
        self._modes = bkd.asarray(
            0.05 * rng.normal(size=(base.shape[0], _NPARAMS))
        ) * bkd.reshape(base, (-1, 1))

    def nvars(self):
        return _NPARAMS

    def __call__(self, params_1d):
        return self._base + self._modes @ params_1d

    def jacobian(self, params_1d):
        return self._modes


class TestStrainEnergyFunctional1D:
    def _make_basis(self, bkd):
        npts = 15
        length = 2.0
        E = 3.0
        nu = 0.3
        transform = AffineTransform1D((0.0, length), bkd)
        mesh = TransformedMesh1D(npts, bkd, transform)
        basis = ChebyshevBasis1D(mesh, bkd)
        phys_pts = mesh.points()[0, :]  # shape (npts,)
        return basis, phys_pts, npts, length, E, nu

    def _lame_params(self, E: float, nu: float):
        mu = E / (2.0 * (1.0 + nu))
        lamda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lamda, mu

    def _modulus_map(self, bkd, npts, E):
        return _AffineMap(bkd, bkd.full((npts,), E))

    def _lame_map(self, bkd, npts, E, nu):
        lamda, mu = self._lame_params(E, nu)
        return _AffineMap(
            bkd, bkd.concatenate([bkd.full((npts,), mu), bkd.full((npts,), lamda)])
        )

    def _linear(self, bkd, **kw):
        basis, phys_pts, npts, length, E, nu = self._make_basis(bkd)
        func = create_linear_strain_energy_1d(
            basis, self._modulus_map(bkd, npts, E), bkd, **kw
        )
        return func, phys_pts, npts, length, E

    def _neo_hookean(self, bkd):
        basis, phys_pts, npts, length, E, nu = self._make_basis(bkd)
        func = create_neo_hookean_strain_energy_1d(
            basis, self._lame_map(bkd, npts, E, nu), bkd
        )
        lamda, mu = self._lame_params(E, nu)
        return func, phys_pts, npts, length, lamda, mu

    # ------------------------------------------------------------------
    # Linear elastic tests
    # ------------------------------------------------------------------

    def test_constant_strain_linear(self, bkd) -> None:
        """u(x) = eps*x gives W = (1/2)*E*eps^2*L."""
        func, phys_pts, npts, length, E = self._linear(bkd)
        eps = 0.05
        state = bkd.reshape(eps * phys_pts, (npts, 1))
        param = bkd.zeros((_NPARAMS, 1))
        result = func(state, param)
        expected = bkd.asarray([[0.5 * E * eps**2 * length]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_zero_displacement_zero_energy(self, bkd) -> None:
        """Zero displacement gives zero strain energy."""
        func, phys_pts, npts, length, E = self._linear(bkd)
        state = bkd.zeros((npts, 1))
        param = bkd.zeros((_NPARAMS, 1))
        result = func(state, param)
        expected = bkd.asarray([[0.0]])
        bkd.assert_allclose(result, expected, atol=1e-15)

    def test_quadratic_displacement_linear(self, bkd) -> None:
        """u(x) = x^2: eps = 2x, W = integral (1/2)*E*(2x)^2 dx = 2*E*L^3/3."""
        func, phys_pts, npts, length, E = self._linear(bkd)
        state = bkd.reshape(phys_pts**2, (npts, 1))
        param = bkd.zeros((_NPARAMS, 1))
        result = func(state, param)
        # integral_0^L (1/2)*E*(2x)^2 dx = 2*E*L^3/3
        expected = bkd.asarray([[2.0 * E * length**3 / 3.0]])
        bkd.assert_allclose(result, expected, atol=1e-11)

    def test_linear_jacobian_derivative_checker(self, bkd) -> None:
        """DerivativeChecker validates linear elastic state Jacobian."""
        func, phys_pts, npts, length, E = self._linear(bkd)
        param = bkd.zeros((_NPARAMS, 1))

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

    def test_linear_param_jacobian_exact(self, bkd) -> None:
        """Linear in E: dW/dp = (w * eps^2 / 2) @ dE/dp exactly, and
        W(p) - W(0) = dW/dp @ p."""
        func, phys_pts, npts, length, E = self._linear(bkd)
        state = bkd.reshape(0.05 * phys_pts**2, (npts, 1))
        param = bkd.asarray([[0.4], [-0.2], [0.7]])
        jac = func.param_jacobian(state, param)
        assert jac.shape == (1, _NPARAMS)
        q0 = func(state, bkd.zeros((_NPARAMS, 1)))
        bkd.assert_allclose(func(state, param) - q0, jac @ param, rtol=1e-12)

    # ------------------------------------------------------------------
    # Neo-Hookean tests
    # ------------------------------------------------------------------

    def test_neo_hookean_zero_displacement(self, bkd) -> None:
        """Zero displacement (F=1): Neo-Hookean strain energy is zero."""
        func, phys_pts, npts, length, lamda, mu = self._neo_hookean(bkd)
        state = bkd.zeros((npts, 1))
        param = bkd.zeros((_NPARAMS, 1))
        result = func(state, param)
        expected = bkd.asarray([[0.0]])
        bkd.assert_allclose(result, expected, atol=1e-14)

    def test_neo_hookean_constant_deformation(self, bkd) -> None:
        """u(x) = eps*x: F = 1+eps everywhere. Verify energy analytically."""
        func, phys_pts, npts, length, lamda, mu = self._neo_hookean(bkd)
        eps = 0.05
        state = bkd.reshape(eps * phys_pts, (npts, 1))
        param = bkd.zeros((_NPARAMS, 1))
        result = func(state, param)

        # psi(F) = mu/2*(F^2-1-2*ln(F)) + lam/2*(ln(F))^2
        F = 1.0 + eps
        ln_F = np.log(F)
        psi = mu / 2.0 * (F**2 - 1.0 - 2.0 * ln_F) + lamda / 2.0 * ln_F**2
        expected = bkd.asarray([[psi * length]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_neo_hookean_jacobian_derivative_checker(self, bkd) -> None:
        """DerivativeChecker validates Neo-Hookean state Jacobian."""
        func, phys_pts, npts, length, lamda, mu = self._neo_hookean(bkd)
        param = bkd.zeros((_NPARAMS, 1))

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
        basis, phys_pts, npts, length, E, nu = self._make_basis(bkd)
        lamda, mu = self._lame_params(E, nu)
        eps = 0.001  # very small strain

        func_nh = create_neo_hookean_strain_energy_1d(
            basis, self._lame_map(bkd, npts, E, nu), bkd
        )
        # Linear elastic with effective modulus = lamda + 2*mu (plane strain 1D)
        # Actually for 1D: linearized Neo-Hookean gives
        # P ~ (lamda + 2*mu)*eps at small eps, so psi ~ (lamda+2mu)/2 * eps^2
        E_eff = lamda + 2.0 * mu
        func_lin = create_linear_strain_energy_1d(
            basis, self._modulus_map(bkd, npts, E_eff), bkd
        )

        state = bkd.reshape(eps * phys_pts, (npts, 1))
        param = bkd.zeros((_NPARAMS, 1))

        W_nh = func_nh(state, param)
        W_lin = func_lin(state, param)
        bkd.assert_allclose(W_nh, W_lin, rtol=1e-2)

    # ------------------------------------------------------------------
    # Material dependence, for every density
    # ------------------------------------------------------------------

    def _density_functional(self, bkd, density):
        basis, phys_pts, npts, length, E, nu = self._make_basis(bkd)
        material_map = (
            self._modulus_map(bkd, npts, E)
            if density.nfields() == 1
            else self._lame_map(bkd, npts, E, nu)
        )
        func = StrainEnergyFunctional1D(
            basis, density, material_map, bkd, scale=1.0 / length
        )
        state = bkd.reshape(0.02 * phys_pts * (length - phys_pts), (npts, 1))
        return func, state, npts

    @pytest.mark.parametrize(
        "density_cls",
        [
            LinearStrainEnergyDensity1D,
            LinearStressDensity1D,
            NeoHookeanStrainEnergyDensity1D,
            NeoHookeanStressDensity1D,
        ],
    )
    def test_param_jacobian_derivative_checker(self, bkd, density_cls) -> None:
        """dQ/dp through the material matches finite differences."""
        func, state, npts = self._density_functional(bkd, density_cls(bkd))
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=_NPARAMS,
            fun=lambda p: func(state, p),
            jacobian=lambda p: func.param_jacobian(state, p),
            bkd=bkd,
        )
        param = bkd.asarray([[0.3], [-0.5], [0.2]])
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(param, direction=None, relative=True)[0]
        assert checker.error_ratio(errors) <= 1e-6

    @pytest.mark.parametrize(
        "density_cls", [LinearStressDensity1D, NeoHookeanStressDensity1D]
    )
    def test_stress_state_jacobian_derivative_checker(self, bkd, density_cls) -> None:
        """The stress densities' derivative in the gradient (the tangent)
        matches finite differences."""
        func, state, npts = self._density_functional(bkd, density_cls(bkd))
        param = bkd.asarray([[0.3], [-0.5], [0.2]])
        wrapper = FunctionWithJacobianFromCallable(
            nqoi=1,
            nvars=npts,
            fun=lambda s: func(s, param),
            jacobian=lambda s: func.state_jacobian(s, param),
            bkd=bkd,
        )
        fd_eps = bkd.flip(bkd.logspace(-13, -2, 12))
        checker = DerivativeChecker(wrapper)
        errors = checker.check_derivatives(
            state, fd_eps=fd_eps, direction=None, relative=True
        )[0]
        assert checker.error_ratio(errors) <= 1e-6

    def test_neo_hookean_stress_is_energy_derivative(self, bkd) -> None:
        """The stress density is d(psi)/dF of the energy density, and
        its F-derivative the energy's second derivative (central
        differences in F)."""
        energy = NeoHookeanStrainEnergyDensity1D(bkd)
        stress = NeoHookeanStressDensity1D(bkd)
        F = bkd.asarray([0.8, 1.0, 1.3])
        material = bkd.asarray([1.2, 0.7, 2.0, 0.5, 1.5, 0.9])
        h = 1e-6
        bkd.assert_allclose(
            stress.value(F, material),
            (energy.value(F + h, material) - energy.value(F - h, material)) / (2.0 * h),
            rtol=1e-8,
            atol=1e-9,
        )
        bkd.assert_allclose(
            stress.arg_derivative(F, material),
            (stress.value(F + h, material) - stress.value(F - h, material)) / (2.0 * h),
            rtol=1e-8,
            atol=1e-9,
        )

    # ------------------------------------------------------------------
    # Subdomain, factory and validation tests
    # ------------------------------------------------------------------

    def test_subdomain_strain_energy(self, bkd) -> None:
        """Strain energy over half domain."""
        basis, phys_pts, npts, length, E, nu = self._make_basis(bkd)
        eps = 0.05
        half_L = length / 2.0
        func_half = create_linear_strain_energy_1d(
            basis,
            self._modulus_map(bkd, npts, E),
            bkd,
            a_sub=0.0,
            b_sub=half_L,
        )
        state = bkd.reshape(eps * phys_pts, (npts, 1))
        param = bkd.zeros((_NPARAMS, 1))
        result = func_half(state, param)
        expected = bkd.asarray([[0.5 * E * eps**2 * half_L]])
        bkd.assert_allclose(result, expected, atol=1e-12)

    def test_factory_linear(self, bkd) -> None:
        """Linear factory creates a valid functional."""
        func, phys_pts, npts, length, E = self._linear(bkd)
        assert func.nqoi() == 1
        assert func.nstates() == npts
        assert func.nparams() == _NPARAMS
        assert func.nunique_params() == 0
        assert isinstance(func, SteadyFunctionalProtocol)

    def test_factory_neo_hookean(self, bkd) -> None:
        """Neo-Hookean factory creates a valid functional."""
        func, phys_pts, npts, length, lamda, mu = self._neo_hookean(bkd)
        assert func.nqoi() == 1
        assert func.nstates() == npts
        assert func.nparams() == _NPARAMS
        assert isinstance(func, SteadyFunctionalProtocol)

    def test_wrong_material_size_raises(self, numpy_bkd) -> None:
        """A one-field map handed to a two-field density is refused."""
        bkd = numpy_bkd
        basis, phys_pts, npts, length, E, nu = self._make_basis(bkd)
        func = create_neo_hookean_strain_energy_1d(
            basis, self._modulus_map(bkd, npts, E), bkd
        )
        with pytest.raises(ValueError, match="nfields"):
            func(bkd.zeros((npts, 1)), bkd.zeros((_NPARAMS, 1)))

    def test_non_protocol_arguments_refused(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        basis, phys_pts, npts, length, E, nu = self._make_basis(bkd)
        material_map = self._modulus_map(bkd, npts, E)
        with pytest.raises(TypeError, match="GradientDensity1DProtocol"):
            StrainEnergyFunctional1D(basis, object(), material_map, bkd)
        with pytest.raises(TypeError, match="MaterialMapProtocol"):
            StrainEnergyFunctional1D(
                basis, LinearStrainEnergyDensity1D(bkd), object(), bkd
            )
