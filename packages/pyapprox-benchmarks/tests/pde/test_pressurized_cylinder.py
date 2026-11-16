"""Integration tests for the 2D pressurized cylinder forward UQ problems."""

import math

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.pde.collocation.basis import ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh2D
from pyapprox.pde.collocation.mesh.transforms import PolarTransform
from pyapprox.pde.field_maps.lame import FixedPoissonRatioLameMap
from pyapprox.pde.zoo.hyperelastic_cylinder_2d import (
    create_hyperelastic_pressurized_cylinder_2d,
)
from pyapprox.pde.zoo.pressurized_cylinder_2d import (
    create_linear_pressurized_cylinder_2d,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox_benchmarks.pde.pressurized_cylinder import (
    _make_functional,
    _make_hyperelastic_functional,
    build_hyperelastic_pressurized_cylinder_2d,
    build_pressurized_cylinder_2d,
)
from tests._helpers.markers import slow_test, slower_test, slowest_test


def _make_problem(
    bkd, qoi="outer_radial_displacement", npts_r=10, npts_theta=10, num_kle_terms=2
):
    """Helper for creating problems with small defaults for fast tests."""
    return build_pressurized_cylinder_2d(
        bkd,
        qoi=qoi,
        npts_r=npts_r,
        npts_theta=npts_theta,
        r_inner=1.0,
        r_outer=2.0,
        E_mean=1.0,
        poisson_ratio=0.3,
        inner_pressure=1.0,
        num_kle_terms=num_kle_terms,
        sigma=0.3,
        weld_r_fraction=0.25,
    )


def _check_jacobian(bkd, fwd, num_kle_terms=2):
    """Helper: run DerivativeChecker on a forward model."""
    # Read the capability from the bundle rather than reaching for a
    # method. Absence is None there, so a model without an analytical
    # jacobian reports that instead of raising AttributeError.
    jacobian = fwd.derivatives().jacobian
    assert jacobian is not None
    wrapper = FunctionWithJacobianFromCallable(
        nqoi=fwd.nqoi(),
        nvars=fwd.nvars(),
        fun=fwd,
        jacobian=jacobian,
        bkd=bkd,
    )
    checker = DerivativeChecker(wrapper)
    np.random.seed(42)
    sample = bkd.array([0.1, -0.1][:num_kle_terms])[:, None]
    errors = checker.check_derivatives(sample, relative=True)[0]
    ratio = float(bkd.min(errors) / bkd.max(errors))
    assert ratio <= 1e-5


@slow_test
class TestPressurizedCylinder2D:
    @classmethod
    def setup_class(cls):
        cls._class_bkd = NumpyBkd()
        bkd = cls._class_bkd
        cls._cached_probs = {}
        for qoi in [
            "outer_radial_displacement",
            "average_hoop_stress",
            "strain_energy",
        ]:
            cls._cached_probs[qoi] = _make_problem(bkd, qoi)

    # --- Evaluation tests ---

    def test_outer_displacement_evaluate(self):
        """Outer radial displacement: shape (1,1), positive value."""
        bkd = self._class_bkd
        prob = self._cached_probs["outer_radial_displacement"]
        fwd = prob.function()
        sample = bkd.zeros((2, 1))
        result = fwd(sample)
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    def test_average_hoop_stress_evaluate(self):
        """Average hoop stress: shape (1,1), positive value."""
        bkd = self._class_bkd
        prob = self._cached_probs["average_hoop_stress"]
        fwd = prob.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    def test_strain_energy_evaluate(self):
        """Strain energy: shape (1,1), positive value."""
        bkd = self._class_bkd
        prob = self._cached_probs["strain_energy"]
        fwd = prob.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    # --- Jacobian tests ---

    def test_outer_displacement_jacobian(self):
        """DerivativeChecker on outer radial displacement."""
        bkd = self._class_bkd
        prob = self._cached_probs["outer_radial_displacement"]
        _check_jacobian(bkd, prob.function())

    def test_average_hoop_stress_jacobian(self):
        """DerivativeChecker on average hoop stress."""
        bkd = self._class_bkd
        prob = self._cached_probs["average_hoop_stress"]
        _check_jacobian(bkd, prob.function())

    def test_strain_energy_jacobian(self):
        """DerivativeChecker on strain energy."""
        bkd = self._class_bkd
        prob = self._cached_probs["strain_energy"]
        _check_jacobian(bkd, prob.function())

    # --- All QoIs produce scalar ---

    def test_all_qois_produce_scalar(self):
        """All 3 QoIs return nqoi=1 and shape (1,1)."""
        bkd = self._class_bkd
        sample = bkd.zeros((2, 1))
        for qoi in [
            "outer_radial_displacement",
            "average_hoop_stress",
            "strain_energy",
        ]:
            prob = self._cached_probs[qoi]
            fwd = prob.function()
            assert fwd.nqoi() == 1, f"Failed for qoi={qoi}"
            result = fwd(sample)
            assert result.shape == (1, 1), f"Failed for qoi={qoi}"

    # --- Prior ---

    def test_prior_samples_shape(self):
        """Prior generates samples of correct shape."""
        prob = self._cached_probs["outer_radial_displacement"]
        np.random.seed(42)
        samples = prob.prior().rvs(5)
        assert samples.shape == (2, 5)

    # --- Protocol compliance ---

    def test_function_protocol_compliance(self):
        """Forward model is Function-shaped with a jacobian."""
        prob = self._cached_probs["outer_radial_displacement"]
        fwd = prob.function()
        assert isinstance(fwd, FunctionProtocol)
        assert callable(fwd.derivatives().jacobian)

    # --- Convergence (non-default params, builds fresh) ---

    def test_convergence(self):
        """Coarse and fine mesh outer displacement agree."""
        bkd = self._class_bkd
        sample = bkd.zeros((2, 1))
        prob_coarse = _make_problem(
            bkd,
            "outer_radial_displacement",
            npts_r=8,
            npts_theta=8,
        )
        prob_fine = _make_problem(
            bkd,
            "outer_radial_displacement",
            npts_r=14,
            npts_theta=14,
        )
        val_coarse = prob_coarse.function()(sample)
        val_fine = prob_fine.function()(sample)
        bkd.assert_allclose(val_coarse, val_fine, rtol=1e-2)

    # --- Error handling ---

    def test_invalid_qoi_raises(self):
        """Invalid QoI string raises ValueError."""
        bkd = self._class_bkd
        with pytest.raises(ValueError):
            build_pressurized_cylinder_2d(bkd, qoi="max_stress")

    # --- Multi-sample evaluation ---

    def test_batch_evaluation(self):
        """Forward model handles batch evaluation."""
        bkd = self._class_bkd
        prob = self._cached_probs["outer_radial_displacement"]
        fwd = prob.function()
        samples = bkd.array([[0.1, -0.1, 0.2], [0.05, 0.0, -0.15]])
        result = fwd(samples)
        assert result.shape == (1, 3)


# ======================================================================
# Hyperelastic tests
# ======================================================================


def _make_hyperelastic_problem(
    bkd,
    qoi="outer_radial_displacement",
    npts_r=10,
    npts_theta=10,
    num_kle_terms=2,
    inner_pressure=1.0,
):
    """Helper for creating hyperelastic problems with small defaults."""
    return build_hyperelastic_pressurized_cylinder_2d(
        bkd,
        qoi=qoi,
        npts_r=npts_r,
        npts_theta=npts_theta,
        r_inner=1.0,
        r_outer=2.0,
        E_mean=1.0,
        poisson_ratio=0.3,
        inner_pressure=inner_pressure,
        num_kle_terms=num_kle_terms,
        sigma=0.3,
        weld_r_fraction=0.25,
    )


def _check_hyperelastic_jacobian(bkd, fwd, num_kle_terms=2):
    """Helper: run DerivativeChecker on a hyperelastic forward model."""
    jacobian = fwd.derivatives().jacobian
    assert jacobian is not None
    wrapper = FunctionWithJacobianFromCallable(
        nqoi=fwd.nqoi(),
        nvars=fwd.nvars(),
        fun=fwd,
        jacobian=jacobian,
        bkd=bkd,
    )
    checker = DerivativeChecker(wrapper)
    np.random.seed(42)
    sample = bkd.array([0.1, -0.1][:num_kle_terms])[:, None]
    errors = checker.check_derivatives(sample, relative=True)[0]
    ratio = float(bkd.min(errors) / bkd.max(errors))
    assert ratio <= 2e-5


class TestHyperelasticPressurizedCylinder2D:
    @classmethod
    def setup_class(cls):
        cls._class_bkd = NumpyBkd()
        bkd = cls._class_bkd
        cls._cached_probs = {}
        for qoi in [
            "outer_radial_displacement",
            "average_hoop_stress",
            "strain_energy",
        ]:
            cls._cached_probs[qoi] = _make_hyperelastic_problem(bkd, qoi)

    # --- Evaluation ---

    @slow_test
    def test_hyperelastic_evaluate(self):
        """Hyperelastic: evaluate at theta=0, check shape (1,1)."""
        bkd = self._class_bkd
        prob = self._cached_probs["outer_radial_displacement"]
        fwd = prob.function()
        sample = bkd.zeros((2, 1))
        result = fwd(sample)
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    # --- Jacobian ---

    @slower_test
    def test_hyperelastic_jacobian(self):
        """DerivativeChecker on hyperelastic outer radial displacement."""
        bkd = self._class_bkd
        prob = self._cached_probs["outer_radial_displacement"]
        _check_hyperelastic_jacobian(bkd, prob.function())

    # --- Linear vs hyperelastic at low pressure (non-default params) ---

    @slow_test
    def test_linear_vs_hyperelastic(self):
        """At low pressure, linear and hyperelastic produce similar QoI."""
        bkd = self._class_bkd
        sample = bkd.zeros((2, 1))
        prob_hyper = _make_hyperelastic_problem(
            bkd,
            "outer_radial_displacement",
            npts_r=10,
            npts_theta=10,
            inner_pressure=1e-3,
        )
        prob_linear_low = build_pressurized_cylinder_2d(
            bkd,
            qoi="outer_radial_displacement",
            npts_r=10,
            npts_theta=10,
            r_inner=1.0,
            r_outer=2.0,
            E_mean=1.0,
            poisson_ratio=0.3,
            inner_pressure=1e-3,
            num_kle_terms=2,
            sigma=0.3,
        )
        val_linear = prob_linear_low.function()(sample)
        val_hyper = prob_hyper.function()(sample)
        bkd.assert_allclose(val_hyper, val_linear, rtol=1e-2)

    # --- All three QoIs ---

    @slowest_test
    def test_all_three_qois_hyperelastic(self):
        """Each QoI produces scalar (1,1) with working Jacobian."""
        bkd = self._class_bkd
        sample = bkd.zeros((2, 1))
        for qoi in [
            "outer_radial_displacement",
            "average_hoop_stress",
            "strain_energy",
        ]:
            prob = self._cached_probs[qoi]
            fwd = prob.function()
            assert fwd.nqoi() == 1, f"Failed for qoi={qoi}"
            result = fwd(sample)
            assert result.shape == (1, 1), f"Failed for qoi={qoi}"
            _check_hyperelastic_jacobian(bkd, fwd)

    # --- Convergence (non-default params) ---

    @slow_test
    def test_convergence_reference_hyperelastic(self):
        """Finer mesh resolution produces closer values (convergence)."""
        bkd = self._class_bkd
        sample = bkd.zeros((2, 1))
        prob_coarse = _make_hyperelastic_problem(
            bkd,
            "outer_radial_displacement",
            npts_r=8,
            npts_theta=8,
        )
        prob_fine = _make_hyperelastic_problem(
            bkd,
            "outer_radial_displacement",
            npts_r=14,
            npts_theta=14,
        )
        val_coarse = prob_coarse.function()(sample)
        val_fine = prob_fine.function()(sample)
        bkd.assert_allclose(val_coarse, val_fine, rtol=1e-2)


# ======================================================================
# Stress QoIs read the material the state was solved with
# ======================================================================


class _UniformLogModulus:
    """A one-parameter field map: E = exp(p_0) at every node."""

    def __init__(self, bkd, npts):
        self._bkd = bkd
        self._npts = npts

    def nvars(self):
        return 1

    def __call__(self, params_1d):
        return self._bkd.exp(params_1d[0]) * self._bkd.ones((self._npts,))

    def jacobian(self, params_1d):
        return self(params_1d)[:, None]


def _uniform_modulus_model(bkd, qoi, hyperelastic, pressure, npts=8):
    """The benchmark's model and QoI, with a uniform modulus exp(p_0)."""
    transform = PolarTransform((1.0, 2.0), (0.0, math.pi / 2.0), bkd)
    mesh = TransformedMesh2D(npts, npts, bkd, transform)
    basis = ChebyshevBasis2D(mesh, bkd)
    field_map = _UniformLogModulus(bkd, basis.npts())
    lame_map = FixedPoissonRatioLameMap(field_map, 0.3, basis.npts(), bkd)
    make_functional = (
        _make_hyperelastic_functional if hyperelastic else _make_functional
    )
    functional = make_functional(bkd, qoi, lame_map, basis, mesh, transform, 1, 0.25)
    factory = (
        create_hyperelastic_pressurized_cylinder_2d
        if hyperelastic
        else create_linear_pressurized_cylinder_2d
    )
    return factory(
        bkd=bkd,
        npts_r=npts,
        npts_theta=npts,
        r_inner=1.0,
        r_outer=2.0,
        E_mean=1.0,
        poisson_ratio=0.3,
        inner_pressure=pressure,
        field_map=field_map,
        functional=functional,
    )


class TestStressQoIMaterial:
    """Scaling laws that hold only if a stress QoI reads the modulus the
    state was solved with. A QoI reading a fixed material breaks each:
    its stress then scales as 1/E."""

    def test_linear_hoop_stress_independent_of_modulus(self):
        """Traction loading fixes the linear stress whatever the uniform
        modulus: Q(E=3) = Q(E=1), and dQ/dp_0 = 0."""
        bkd = NumpyBkd()
        fwd = _uniform_modulus_model(bkd, "average_hoop_stress", False, 1.0)
        q1 = fwd(bkd.zeros((1, 1)))
        q3 = fwd(bkd.full((1, 1), math.log(3.0)))
        bkd.assert_allclose(q3, q1, rtol=1e-8)
        grad = fwd.derivatives().jacobian(bkd.full((1, 1), math.log(3.0)))
        assert abs(float(grad[0, 0])) <= 1e-8 * abs(float(q1[0, 0]))

    def test_linear_strain_energy_inverse_in_modulus(self):
        """Displacement scales as 1/E at fixed traction, so the strain
        energy does too: Q(E=3) = Q(E=1)/3, and dQ/dp_0 = -Q."""
        bkd = NumpyBkd()
        fwd = _uniform_modulus_model(bkd, "strain_energy", False, 1.0)
        p3 = bkd.full((1, 1), math.log(3.0))
        q1 = fwd(bkd.zeros((1, 1)))
        q3 = fwd(p3)
        bkd.assert_allclose(q3, q1 / 3.0, rtol=1e-8)
        bkd.assert_allclose(fwd.derivatives().jacobian(p3), -q3, rtol=1e-6)

    @pytest.mark.parametrize("qoi", ["average_hoop_stress", "strain_energy"])
    def test_neo_hookean_scaling(self, qoi):
        """Neo-Hookean stress and energy are linear in (mu, lamda) at
        fixed F, so scaling the modulus and the pressure by c leaves the
        displacement unchanged and scales the QoI by c."""
        bkd = NumpyBkd()
        c = 3.0
        q_base = _uniform_modulus_model(bkd, qoi, True, 0.2)(bkd.zeros((1, 1)))
        q_scaled = _uniform_modulus_model(bkd, qoi, True, c * 0.2)(
            bkd.full((1, 1), math.log(c))
        )
        bkd.assert_allclose(q_scaled, c * q_base, rtol=1e-8)
