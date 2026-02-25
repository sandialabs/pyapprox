"""Integration tests for the 2D pressurized cylinder benchmark instances."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.benchmarks.instances.pde.pressurized_cylinder import (
    hyperelastic_pressurized_cylinder_2d,
    pressurized_cylinder_2d,
)
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
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import (
    load_tests,  # noqa: F401
    slow_test,
    slower_test,
    slowest_test,
)


def _make_benchmark(
    bkd, qoi="outer_radial_displacement", npts_r=10, npts_theta=10, num_kle_terms=2
):
    """Helper for creating benchmarks with small defaults for fast tests."""
    return pressurized_cylinder_2d(
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


def _check_jacobian(test_case, bkd, fwd, num_kle_terms=2):
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
    test_case.assertLessEqual(ratio, 1e-5)


@slow_test
class TestPressurizedCylinder2DBenchmark(Generic[Array], unittest.TestCase):
    __test__ = False

    @classmethod
    def _get_bkd(cls) -> Backend[Array]:
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        bkd = cls._get_bkd()
        cls._class_bkd = bkd
        cls._cached_bms = {}
        for qoi in [
            "outer_radial_displacement",
            "average_hoop_stress",
            "strain_energy",
        ]:
            cls._cached_bms[qoi] = _make_benchmark(bkd, qoi)

    def setUp(self):
        self._bkd = self._class_bkd

    # --- Evaluation tests ---

    def test_outer_displacement_evaluate(self):
        """Outer radial displacement: shape (1,1), positive value."""
        bkd = self._bkd
        bm = self._cached_bms["outer_radial_displacement"]
        fwd = bm.function()
        sample = bkd.zeros((2, 1))
        result = fwd(sample)
        self.assertEqual(result.shape, (1, 1))
        self.assertGreater(float(result[0, 0]), 0.0)

    def test_average_hoop_stress_evaluate(self):
        """Average hoop stress: shape (1,1), positive value."""
        bkd = self._bkd
        bm = self._cached_bms["average_hoop_stress"]
        fwd = bm.function()
        result = fwd(bkd.zeros((2, 1)))
        self.assertEqual(result.shape, (1, 1))
        self.assertGreater(float(result[0, 0]), 0.0)

    def test_strain_energy_evaluate(self):
        """Strain energy: shape (1,1), positive value."""
        bkd = self._bkd
        bm = self._cached_bms["strain_energy"]
        fwd = bm.function()
        result = fwd(bkd.zeros((2, 1)))
        self.assertEqual(result.shape, (1, 1))
        self.assertGreater(float(result[0, 0]), 0.0)

    # --- Jacobian tests ---

    def test_outer_displacement_jacobian(self):
        """DerivativeChecker on outer radial displacement."""
        bkd = self._bkd
        bm = self._cached_bms["outer_radial_displacement"]
        _check_jacobian(self, bkd, bm.function())

    def test_average_hoop_stress_jacobian(self):
        """DerivativeChecker on average hoop stress."""
        bkd = self._bkd
        bm = self._cached_bms["average_hoop_stress"]
        _check_jacobian(self, bkd, bm.function())

    def test_strain_energy_jacobian(self):
        """DerivativeChecker on strain energy."""
        bkd = self._bkd
        bm = self._cached_bms["strain_energy"]
        _check_jacobian(self, bkd, bm.function())

    # --- All QoIs produce scalar ---

    def test_all_qois_produce_scalar(self):
        """All 3 QoIs return nqoi=1 and shape (1,1)."""
        bkd = self._bkd
        sample = bkd.zeros((2, 1))
        for qoi in [
            "outer_radial_displacement",
            "average_hoop_stress",
            "strain_energy",
        ]:
            bm = self._cached_bms[qoi]
            fwd = bm.function()
            self.assertEqual(fwd.nqoi(), 1, f"Failed for qoi={qoi}")
            result = fwd(sample)
            self.assertEqual(result.shape, (1, 1), f"Failed for qoi={qoi}")

    # --- Prior and domain ---

    def test_prior_samples_shape(self):
        """Prior generates samples of correct shape."""
        bm = self._cached_bms["outer_radial_displacement"]
        np.random.seed(42)
        samples = bm.prior().rvs(5)
        self.assertEqual(samples.shape, (2, 5))

    def test_domain_bounds(self):
        """Domain bounds are [-4, 4]^2 for 2 KLE terms."""
        bkd = self._bkd
        bm = self._cached_bms["outer_radial_displacement"]
        bounds = bm.domain().bounds()
        self.assertEqual(bounds.shape, (2, 2))
        bkd.assert_allclose(
            bounds,
            bkd.array([[-4.0, 4.0], [-4.0, 4.0]]),
        )

    # --- Protocol compliance ---

    def test_benchmark_protocol_compliance(self):
        """Benchmark satisfies BenchmarkWithPriorProtocol."""
        bm = self._cached_bms["outer_radial_displacement"]
        self.assertTrue(isinstance(bm, BenchmarkWithPriorProtocol))

    def test_function_protocol_compliance(self):
        """Forward model satisfies FunctionWithJacobianProtocol."""
        bm = self._cached_bms["outer_radial_displacement"]
        fwd = bm.function()
        self.assertTrue(isinstance(fwd, FunctionProtocol))
        self.assertTrue(isinstance(fwd, FunctionWithJacobianProtocol))

    # --- Convergence (non-default params, builds fresh) ---

    def test_convergence(self):
        """Coarse and fine mesh outer displacement agree."""
        bkd = self._bkd
        sample = bkd.zeros((2, 1))
        bm_coarse = _make_benchmark(
            bkd,
            "outer_radial_displacement",
            npts_r=8,
            npts_theta=8,
        )
        bm_fine = _make_benchmark(
            bkd,
            "outer_radial_displacement",
            npts_r=14,
            npts_theta=14,
        )
        val_coarse = bm_coarse.function()(sample)
        val_fine = bm_fine.function()(sample)
        bkd.assert_allclose(val_coarse, val_fine, rtol=1e-2)

    # --- Registry access (builds fresh via registry) ---

    def test_registry_access(self):
        """BenchmarkRegistry.get works for pressurized cylinder."""
        bkd = self._bkd
        bm = BenchmarkRegistry.get("pressurized_cylinder_2d_linear", bkd)
        fwd = bm.function()
        result = fwd(bkd.zeros((2, 1)))
        self.assertEqual(result.shape, (1, 1))

    # --- Error handling ---

    def test_invalid_qoi_raises(self):
        """Invalid QoI string raises ValueError."""
        bkd = self._bkd
        with self.assertRaises(ValueError):
            pressurized_cylinder_2d(bkd, qoi="max_stress")

    # --- Multi-sample evaluation ---

    def test_batch_evaluation(self):
        """Forward model handles batch evaluation."""
        bkd = self._bkd
        bm = self._cached_bms["outer_radial_displacement"]
        fwd = bm.function()
        samples = bkd.array([[0.1, -0.1, 0.2], [0.05, 0.0, -0.15]])
        result = fwd(samples)
        self.assertEqual(result.shape, (1, 3))


class TestPressurizedCylinder2DBenchmarkNumpy(
    TestPressurizedCylinder2DBenchmark[NDArray[Any]],
):
    @classmethod
    def _get_bkd(cls) -> NumpyBkd:
        return NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._class_bkd


class TestPressurizedCylinder2DBenchmarkTorch(
    TestPressurizedCylinder2DBenchmark[torch.Tensor],
):
    @classmethod
    def _get_bkd(cls) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._class_bkd


# ======================================================================
# Hyperelastic benchmark tests
# ======================================================================


def _make_hyperelastic_benchmark(
    bkd,
    qoi="outer_radial_displacement",
    npts_r=10,
    npts_theta=10,
    num_kle_terms=2,
    inner_pressure=1.0,
):
    """Helper for creating hyperelastic benchmarks with small defaults."""
    return hyperelastic_pressurized_cylinder_2d(
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


def _check_hyperelastic_jacobian(test_case, bkd, fwd, num_kle_terms=2):
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
    test_case.assertLessEqual(ratio, 1e-5)


class TestHyperelasticPressurizedCylinder2D(
    Generic[Array],
    unittest.TestCase,
):
    __test__ = False

    @classmethod
    def _get_bkd(cls) -> Backend[Array]:
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        bkd = cls._get_bkd()
        cls._class_bkd = bkd
        cls._cached_bms = {}
        for qoi in [
            "outer_radial_displacement",
            "average_hoop_stress",
            "strain_energy",
        ]:
            cls._cached_bms[qoi] = _make_hyperelastic_benchmark(bkd, qoi)

    def setUp(self):
        self._bkd = self._class_bkd

    # --- Evaluation ---

    @slow_test
    def test_benchmark_hyperelastic_evaluate(self):
        """Hyperelastic benchmark: evaluate at theta=0, check shape (1,1)."""
        bkd = self._bkd
        bm = self._cached_bms["outer_radial_displacement"]
        fwd = bm.function()
        sample = bkd.zeros((2, 1))
        result = fwd(sample)
        self.assertEqual(result.shape, (1, 1))
        self.assertGreater(float(result[0, 0]), 0.0)

    # --- Jacobian ---

    @slower_test
    def test_benchmark_hyperelastic_jacobian(self):
        """DerivativeChecker on hyperelastic outer radial displacement."""
        bkd = self._bkd
        bm = self._cached_bms["outer_radial_displacement"]
        _check_hyperelastic_jacobian(self, bkd, bm.function())

    # --- Linear vs hyperelastic at low pressure (non-default params) ---

    @slow_test
    def test_linear_vs_hyperelastic(self):
        """At low pressure, linear and hyperelastic produce similar QoI."""
        bkd = self._bkd
        sample = bkd.zeros((2, 1))
        bm_hyper = _make_hyperelastic_benchmark(
            bkd,
            "outer_radial_displacement",
            npts_r=10,
            npts_theta=10,
            inner_pressure=1e-3,
        )
        bm_linear_low = pressurized_cylinder_2d(
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
        val_linear = bm_linear_low.function()(sample)
        val_hyper = bm_hyper.function()(sample)
        bkd.assert_allclose(val_hyper, val_linear, rtol=1e-2)

    # --- All three QoIs ---

    @slowest_test
    def test_all_three_qois_hyperelastic(self):
        """Each QoI produces scalar (1,1) with working Jacobian."""
        bkd = self._bkd
        sample = bkd.zeros((2, 1))
        for qoi in [
            "outer_radial_displacement",
            "average_hoop_stress",
            "strain_energy",
        ]:
            bm = self._cached_bms[qoi]
            fwd = bm.function()
            self.assertEqual(fwd.nqoi(), 1, f"Failed for qoi={qoi}")
            result = fwd(sample)
            self.assertEqual(result.shape, (1, 1), f"Failed for qoi={qoi}")
            _check_hyperelastic_jacobian(self, bkd, fwd)

    # --- Convergence (non-default params) ---

    @slow_test
    def test_convergence_reference_hyperelastic(self):
        """Finer mesh resolution produces closer values (convergence)."""
        bkd = self._bkd
        sample = bkd.zeros((2, 1))
        bm_coarse = _make_hyperelastic_benchmark(
            bkd,
            "outer_radial_displacement",
            npts_r=8,
            npts_theta=8,
        )
        bm_fine = _make_hyperelastic_benchmark(
            bkd,
            "outer_radial_displacement",
            npts_r=14,
            npts_theta=14,
        )
        val_coarse = bm_coarse.function()(sample)
        val_fine = bm_fine.function()(sample)
        bkd.assert_allclose(val_coarse, val_fine, rtol=1e-2)

    # --- Registry access (builds fresh) ---

    @slow_test
    def test_registry_hyperelastic(self):
        """BenchmarkRegistry.get works for hyperelastic cylinder."""
        bkd = self._bkd
        bm = BenchmarkRegistry.get(
            "pressurized_cylinder_2d_hyperelastic",
            bkd,
        )
        fwd = bm.function()
        result = fwd(bkd.zeros((2, 1)))
        self.assertEqual(result.shape, (1, 1))


class TestHyperelasticPressurizedCylinder2DNumpy(
    TestHyperelasticPressurizedCylinder2D[NDArray[Any]],
):
    @classmethod
    def _get_bkd(cls) -> NumpyBkd:
        return NumpyBkd()

    def bkd(self) -> NumpyBkd:
        return self._class_bkd


class TestHyperelasticPressurizedCylinder2DTorch(
    TestHyperelasticPressurizedCylinder2D[torch.Tensor],
):
    @classmethod
    def _get_bkd(cls) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()

    def bkd(self) -> TorchBkd:
        return self._class_bkd
