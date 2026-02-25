"""Tests for the analytical 2D cantilever beam model."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.benchmarks.functions.algebraic.cantilever_beam_2d import (
    CantileverBeam2DAnalytical,
    CantileverBeam2DConstraints,
    CantileverBeam2DObjective,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class TestCantileverBeam2DAnalytical(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._model = CantileverBeam2DAnalytical(length=100.0, bkd=self._bkd)

    def _nominal_sample(self) -> Array:
        """Return a nominal sample (X, Y, E, R, w, t)."""
        return self._bkd.array(
            [
                [500.0],
                [1000.0],
                [2.9e7],
                [40000.0],
                [2.5],
                [3.0],
            ]
        )

    def test_output_shape(self):
        """Verify output shape for single and batch evaluations."""
        bkd = self._bkd
        model = self._model

        # Single sample
        sample = self._nominal_sample()
        result = model(sample)
        bkd.assert_allclose(
            bkd.asarray([result.shape[0]]),
            bkd.asarray([2]),
        )
        bkd.assert_allclose(
            bkd.asarray([result.shape[1]]),
            bkd.asarray([1]),
        )

        # Batch of 5
        samples = bkd.concatenate([sample] * 5, axis=1)
        result_batch = model(samples)
        bkd.assert_allclose(
            bkd.asarray([result_batch.shape[0]]),
            bkd.asarray([2]),
        )
        bkd.assert_allclose(
            bkd.asarray([result_batch.shape[1]]),
            bkd.asarray([5]),
        )

    def test_jacobian_derivative_checker(self):
        """Validate analytical Jacobian via DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        bkd = self._bkd
        model = self._model
        sample = self._nominal_sample()

        # Verify Jacobian shape
        jac = model.jacobian(sample)
        self.assertEqual(jac.shape, (2, 6))

        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        self.assertLessEqual(ratio, 1e-5)

    def test_stress_positive(self):
        """Stress is positive for positive loads and dimensions."""
        bkd = self._bkd
        result = self._model(self._nominal_sample())
        stress = result[0, 0]
        self.assertGreater(float(bkd.to_numpy(bkd.asarray([stress]))[0]), 0.0)

    def test_displacement_positive(self):
        """Displacement is positive for positive loads and dimensions."""
        bkd = self._bkd
        result = self._model(self._nominal_sample())
        disp = result[1, 0]
        self.assertGreater(float(bkd.to_numpy(bkd.asarray([disp]))[0]), 0.0)

    def test_legacy_consistency(self):
        """Verify raw QoIs recover legacy constraint ratios."""
        bkd = self._bkd
        sample = self._nominal_sample()
        result = self._model(sample)
        stress = result[0, 0]
        disp = result[1, 0]

        R = sample[3, 0]
        D0 = 2.2535

        # Legacy constraint: 1 - stress/R
        stress_constraint = 1.0 - stress / R
        # Legacy constraint: 1 - disp/D0
        disp_constraint = 1.0 - disp / D0

        # Compute legacy formulas directly
        X, Y, E = sample[0, 0], sample[1, 0], sample[2, 0]
        w, t = sample[4, 0], sample[5, 0]
        L = 100.0

        legacy_stress_constraint = 1.0 - 6.0 * L / (w * t) * (X / w + Y / t) / R
        legacy_disp_constraint = (
            1.0
            - 4.0
            * L**3
            / (E * w * t)
            * bkd.sqrt(bkd.asarray([X**2 / w**4 + Y**2 / t**4]))[0]
            / D0
        )

        bkd.assert_allclose(
            bkd.asarray([stress_constraint]),
            bkd.asarray([legacy_stress_constraint]),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            bkd.asarray([disp_constraint]),
            bkd.asarray([legacy_disp_constraint]),
            rtol=1e-12,
        )

    def test_nvars_nqoi(self):
        """Check dimension methods."""
        self.assertEqual(self._model.nvars(), 6)
        self.assertEqual(self._model.nqoi(), 2)


class TestCantileverBeam2DAnalyticalNumpy(
    TestCantileverBeam2DAnalytical[NDArray[Any]],
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCantileverBeam2DAnalyticalTorch(
    TestCantileverBeam2DAnalytical[torch.Tensor],
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestCantileverBeam2DConstraints(Generic[Array], unittest.TestCase):
    """Tests for constraint wrapper."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        beam = CantileverBeam2DAnalytical(length=100.0, bkd=self._bkd)
        self._R = 40000.0
        self._D0 = 2.2535
        self._model = CantileverBeam2DConstraints(beam, self._R, self._D0)

    def _nominal_sample(self) -> Array:
        return self._bkd.array(
            [
                [500.0],
                [1000.0],
                [2.9e7],
                [40000.0],
                [2.5],
                [3.0],
            ]
        )

    def test_output_shape(self):
        """Verify output shape."""
        result = self._model(self._nominal_sample())
        self.assertEqual(result.shape, (2, 1))

    def test_batch_shape(self):
        """Verify batch output shape."""
        sample = self._nominal_sample()
        samples = self._bkd.concatenate([sample] * 5, axis=1)
        result = self._model(samples)
        self.assertEqual(result.shape, (2, 5))

    def test_values_are_safety_margins(self):
        """Constraints equal 1 - stress/R and 1 - disp/D0."""
        bkd = self._bkd
        sample = self._nominal_sample()
        beam = CantileverBeam2DAnalytical(length=100.0, bkd=bkd)
        raw = beam(sample)  # (2, 1)

        expected_c1 = 1.0 - raw[0:1, :] / self._R
        expected_c2 = 1.0 - raw[1:2, :] / self._D0
        expected = bkd.concatenate([expected_c1, expected_c2], axis=0)

        result = self._model(sample)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_nvars_nqoi(self):
        self.assertEqual(self._model.nvars(), 6)
        self.assertEqual(self._model.nqoi(), 2)

    def test_jacobian_derivative_checker(self):
        """Validate Jacobian via DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        sample = self._nominal_sample()
        jac = self._model.jacobian(sample)
        self.assertEqual(jac.shape, (2, 6))

        checker = DerivativeChecker(self._model)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(self._bkd.to_numpy(checker.error_ratio(errors)))
        self.assertLessEqual(ratio, 1e-5)


class TestCantileverBeam2DConstraintsNumpy(
    TestCantileverBeam2DConstraints[NDArray[Any]],
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCantileverBeam2DConstraintsTorch(
    TestCantileverBeam2DConstraints[torch.Tensor],
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestCantileverBeam2DObjective(Generic[Array], unittest.TestCase):
    """Tests for objective wrapper."""

    __test__ = False

    def bkd(self):
        raise NotImplementedError

    def setUp(self):
        self._bkd = self.bkd()
        self._model = CantileverBeam2DObjective(self._bkd)

    def _nominal_sample(self) -> Array:
        return self._bkd.array(
            [
                [500.0],
                [1000.0],
                [2.9e7],
                [40000.0],
                [2.5],
                [3.0],
            ]
        )

    def test_output_shape(self):
        result = self._model(self._nominal_sample())
        self.assertEqual(result.shape, (1, 1))

    def test_batch_shape(self):
        sample = self._nominal_sample()
        samples = self._bkd.concatenate([sample] * 5, axis=1)
        result = self._model(samples)
        self.assertEqual(result.shape, (1, 5))

    def test_value_is_area(self):
        """Objective equals w * t."""
        bkd = self._bkd
        sample = self._nominal_sample()
        result = self._model(sample)
        expected = sample[4:5, :] * sample[5:6, :]  # w * t
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_nvars_nqoi(self):
        self.assertEqual(self._model.nvars(), 6)
        self.assertEqual(self._model.nqoi(), 1)

    def test_jacobian_derivative_checker(self):
        """Validate Jacobian via DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        sample = self._nominal_sample()
        jac = self._model.jacobian(sample)
        self.assertEqual(jac.shape, (1, 6))

        checker = DerivativeChecker(self._model)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(self._bkd.to_numpy(checker.error_ratio(errors)))
        self.assertLessEqual(ratio, 1e-5)


class TestCantileverBeam2DObjectiveNumpy(
    TestCantileverBeam2DObjective[NDArray[Any]],
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCantileverBeam2DObjectiveTorch(
    TestCantileverBeam2DObjective[torch.Tensor],
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestCantileverBeam2DRegistry(unittest.TestCase):
    def test_registry_access(self):
        """Verify benchmark is accessible from registry."""
        # Trigger registration
        import pyapprox.benchmarks.instances.analytic.cantilever_beam_2d  # noqa: F401
        from pyapprox.benchmarks.registry import BenchmarkRegistry
        from pyapprox.util.backends.numpy import NumpyBkd

        bkd = NumpyBkd()
        bm = BenchmarkRegistry.get(
            "cantilever_beam_2d_analytical",
            bkd=bkd,
        )
        func = bm.function()
        self.assertEqual(func.nvars(), 6)
        self.assertEqual(func.nqoi(), 2)
