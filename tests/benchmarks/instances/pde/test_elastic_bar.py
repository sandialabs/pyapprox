"""Integration tests for the 1D elastic bar benchmark instances."""

import numpy as np
import pytest

from pyapprox.benchmarks.instances.pde.elastic_bar import elastic_bar_1d
from pyapprox.benchmarks.protocols import BenchmarkWithPriorProtocol
from pyapprox.benchmarks.registry import BenchmarkRegistry
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


def _make_benchmark(
    bkd, constitutive="linear", qoi="tip_displacement", npts=20, num_kle_terms=2
):
    """Helper for creating benchmarks with small defaults for fast tests."""
    return elastic_bar_1d(
        bkd,
        constitutive=constitutive,
        qoi=qoi,
        npts=npts,
        length=1.0,
        E_mean=1.0,
        poisson_ratio=0.3,
        traction=1.0,
        num_kle_terms=num_kle_terms,
        sigma=0.3,
        correlation_length=0.3,
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
    assert ratio <= 2e-5


class TestElasticBar1DBenchmark:
    # --- Evaluation tests ---

    def test_linear_tip_displacement_evaluate(self, bkd):
        """Linear bar, tip displacement: shape (1,1), positive value."""
        bm = _make_benchmark(bkd, "linear", "tip_displacement")
        fwd = bm.function()
        sample = bkd.zeros((2, 1))
        result = fwd(sample)
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    def test_linear_average_displacement_evaluate(self, bkd):
        """Linear bar, average displacement: shape (1,1), positive value."""
        bm = _make_benchmark(bkd, "linear", "average_displacement")
        fwd = bm.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    def test_linear_average_stress_evaluate(self, bkd):
        """Linear bar, average stress: shape (1,1), positive value."""
        bm = _make_benchmark(bkd, "linear", "average_stress")
        fwd = bm.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    def test_linear_strain_energy_evaluate(self, bkd):
        """Linear bar, strain energy: shape (1,1), positive value."""
        bm = _make_benchmark(bkd, "linear", "strain_energy")
        fwd = bm.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    def test_hyperelastic_tip_displacement_evaluate(self, bkd):
        """Hyperelastic bar, tip displacement: shape (1,1), positive value."""
        bm = _make_benchmark(bkd, "hyperelastic", "tip_displacement")
        fwd = bm.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    def test_hyperelastic_strain_energy_evaluate(self, bkd):
        """Hyperelastic bar, strain energy: shape (1,1), positive value."""
        bm = _make_benchmark(bkd, "hyperelastic", "strain_energy")
        fwd = bm.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)
        assert float(result[0, 0]) > 0.0

    # --- Jacobian tests ---

    def test_linear_tip_jacobian(self, bkd):
        """DerivativeChecker on linear bar, tip displacement."""
        bm = _make_benchmark(bkd, "linear", "tip_displacement")
        _check_jacobian(bkd, bm.function())

    def test_hyperelastic_tip_jacobian(self, bkd):
        """DerivativeChecker on hyperelastic bar, tip displacement."""
        bm = _make_benchmark(bkd, "hyperelastic", "tip_displacement")
        _check_jacobian(bkd, bm.function())

    def test_linear_strain_energy_jacobian(self, bkd):
        """DerivativeChecker on linear bar, strain energy."""
        bm = _make_benchmark(bkd, "linear", "strain_energy")
        _check_jacobian(bkd, bm.function())

    def test_hyperelastic_strain_energy_jacobian(self, bkd):
        """DerivativeChecker on hyperelastic bar, strain energy."""
        bm = _make_benchmark(bkd, "hyperelastic", "strain_energy")
        _check_jacobian(bkd, bm.function())

    def test_linear_average_stress_jacobian(self, bkd):
        """DerivativeChecker on linear bar, average stress."""
        bm = _make_benchmark(bkd, "linear", "average_stress")
        _check_jacobian(bkd, bm.function())

    def test_hyperelastic_average_stress_jacobian(self, bkd):
        """DerivativeChecker on hyperelastic bar, average stress."""
        bm = _make_benchmark(bkd, "hyperelastic", "average_stress")
        _check_jacobian(bkd, bm.function())

    # --- All QoIs produce scalar ---

    def test_all_qois_produce_scalar(self, bkd):
        """All 4 QoIs return nqoi=1 and shape (1,1)."""
        sample = bkd.zeros((2, 1))
        for qoi in [
            "tip_displacement",
            "average_displacement",
            "average_stress",
            "strain_energy",
        ]:
            bm = _make_benchmark(bkd, "linear", qoi)
            fwd = bm.function()
            assert fwd.nqoi() == 1, f"Failed for qoi={qoi}"
            result = fwd(sample)
            assert result.shape == (1, 1), f"Failed for qoi={qoi}"

    # --- Prior and domain ---

    def test_prior_samples_shape(self, bkd):
        """Prior generates samples of correct shape."""
        bm = _make_benchmark(bkd, "linear", "tip_displacement")
        np.random.seed(42)
        samples = bm.prior().rvs(5)
        assert samples.shape == (2, 5)

    def test_domain_bounds(self, bkd):
        """Domain bounds are [-4, 4]^2 for 2 KLE terms."""
        bm = _make_benchmark(bkd, "linear", "tip_displacement")
        bounds = bm.domain().bounds()
        assert bounds.shape == (2, 2)
        bkd.assert_allclose(
            bounds,
            bkd.array([[-4.0, 4.0], [-4.0, 4.0]]),
        )

    # --- Protocol compliance ---

    def test_benchmark_protocol_compliance(self, bkd):
        """Benchmark satisfies BenchmarkWithPriorProtocol."""
        bm = _make_benchmark(bkd, "linear", "tip_displacement")
        assert isinstance(bm, BenchmarkWithPriorProtocol)

    def test_function_protocol_compliance(self, bkd):
        """Forward model satisfies FunctionWithJacobianProtocol."""
        bm = _make_benchmark(bkd, "linear", "tip_displacement")
        fwd = bm.function()
        assert isinstance(fwd, FunctionProtocol)
        assert isinstance(fwd, FunctionWithJacobianProtocol)

    # --- Convergence tests ---

    def test_convergence_linear(self, bkd):
        """Linear bar: coarse and fine mesh tip displacement agree."""
        sample = bkd.zeros((2, 1))
        bm_coarse = _make_benchmark(bkd, "linear", "tip_displacement", npts=15)
        bm_fine = _make_benchmark(bkd, "linear", "tip_displacement", npts=40)
        val_coarse = bm_coarse.function()(sample)
        val_fine = bm_fine.function()(sample)
        bkd.assert_allclose(val_coarse, val_fine, rtol=1e-3)

    def test_convergence_hyperelastic(self, bkd):
        """Hyperelastic bar: coarse and fine mesh tip displacement agree."""
        sample = bkd.zeros((2, 1))
        bm_coarse = _make_benchmark(
            bkd,
            "hyperelastic",
            "tip_displacement",
            npts=15,
        )
        bm_fine = _make_benchmark(
            bkd,
            "hyperelastic",
            "tip_displacement",
            npts=40,
        )
        val_coarse = bm_coarse.function()(sample)
        val_fine = bm_fine.function()(sample)
        bkd.assert_allclose(val_coarse, val_fine, rtol=1e-3)

    # --- Registry access ---

    def test_registry_access_linear(self, bkd):
        """BenchmarkRegistry.get works for linear bar."""
        bm = BenchmarkRegistry.get("elastic_bar_1d_linear", bkd)
        fwd = bm.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)

    def test_registry_access_hyperelastic(self, bkd):
        """BenchmarkRegistry.get works for hyperelastic bar."""
        bm = BenchmarkRegistry.get("elastic_bar_1d_hyperelastic", bkd)
        fwd = bm.function()
        result = fwd(bkd.zeros((2, 1)))
        assert result.shape == (1, 1)

    # --- Error handling ---

    def test_invalid_constitutive_raises(self, bkd):
        """Invalid constitutive model raises ValueError."""
        with pytest.raises(ValueError):
            elastic_bar_1d(bkd, constitutive="rubber")

    def test_invalid_qoi_raises(self, bkd):
        """Invalid QoI string raises ValueError."""
        with pytest.raises(ValueError):
            elastic_bar_1d(bkd, qoi="max_stress")

    # --- Multi-sample evaluation ---

    def test_batch_evaluation(self, bkd):
        """Forward model handles batch evaluation."""
        bm = _make_benchmark(bkd, "linear", "tip_displacement")
        fwd = bm.function()
        np.random.seed(42)
        samples = bkd.array([[0.1, -0.1, 0.2], [0.05, 0.0, -0.15]])
        result = fwd(samples)
        assert result.shape == (1, 3)
