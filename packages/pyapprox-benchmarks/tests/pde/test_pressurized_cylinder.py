"""Integration tests for the 2D pressurized cylinder forward UQ problems."""

import numpy as np
import pytest

from pyapprox_benchmarks.pde.pressurized_cylinder import (
    build_hyperelastic_pressurized_cylinder_2d,
    build_pressurized_cylinder_2d,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.interface.functions.protocols import (
    FunctionProtocol,
    FunctionWithJacobianProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd
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
    wrapper = FunctionWithJacobianFromCallable(
        nqoi=fwd.nqoi(),
        nvars=fwd.nvars(),
        fun=fwd,
        jacobian=fwd.jacobian,
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
        """Forward model satisfies FunctionWithJacobianProtocol."""
        prob = self._cached_probs["outer_radial_displacement"]
        fwd = prob.function()
        assert isinstance(fwd, FunctionProtocol)
        assert isinstance(fwd, FunctionWithJacobianProtocol)

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
    wrapper = FunctionWithJacobianFromCallable(
        nqoi=fwd.nqoi(),
        nvars=fwd.nvars(),
        fun=fwd,
        jacobian=fwd.jacobian,
        bkd=bkd,
    )
    checker = DerivativeChecker(wrapper)
    np.random.seed(42)
    sample = bkd.array([0.1, -0.1][:num_kle_terms])[:, None]
    errors = checker.check_derivatives(sample, relative=True)[0]
    ratio = float(bkd.min(errors) / bkd.max(errors))
    assert ratio <= 1e-5


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
