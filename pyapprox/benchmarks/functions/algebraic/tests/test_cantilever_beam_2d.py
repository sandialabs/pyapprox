"""Tests for the analytical 2D cantilever beam model."""

import pytest

from pyapprox.benchmarks.functions.algebraic.cantilever_beam_2d import (
    CantileverBeam2DAnalytical,
    CantileverBeam2DConstraints,
    CantileverBeam2DObjective,
)
from pyapprox.util.backends.numpy import NumpyBkd


class TestCantileverBeam2DAnalytical:
    def _make_model(self, bkd):
        return CantileverBeam2DAnalytical(length=100.0, bkd=bkd)

    def _nominal_sample(self, bkd):
        """Return a nominal sample (X, Y, E, R, w, t)."""
        return bkd.array(
            [
                [500.0],
                [1000.0],
                [2.9e7],
                [40000.0],
                [2.5],
                [3.0],
            ]
        )

    def test_output_shape(self, bkd):
        """Verify output shape for single and batch evaluations."""
        model = self._make_model(bkd)

        # Single sample
        sample = self._nominal_sample(bkd)
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

    def test_jacobian_derivative_checker(self, bkd):
        """Validate analytical Jacobian via DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        model = self._make_model(bkd)
        sample = self._nominal_sample(bkd)

        # Verify Jacobian shape
        jac = model.jacobian(sample)
        assert jac.shape == (2, 6)

        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5

    def test_stress_positive(self, bkd):
        """Stress is positive for positive loads and dimensions."""
        model = self._make_model(bkd)
        result = model(self._nominal_sample(bkd))
        stress = result[0, 0]
        assert float(bkd.to_numpy(bkd.asarray([stress]))[0]) > 0.0

    def test_displacement_positive(self, bkd):
        """Displacement is positive for positive loads and dimensions."""
        model = self._make_model(bkd)
        result = model(self._nominal_sample(bkd))
        disp = result[1, 0]
        assert float(bkd.to_numpy(bkd.asarray([disp]))[0]) > 0.0

    def test_legacy_consistency(self, bkd):
        """Verify raw QoIs recover legacy constraint ratios."""
        model = self._make_model(bkd)
        sample = self._nominal_sample(bkd)
        result = model(sample)
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

    def test_nvars_nqoi(self, bkd):
        """Check dimension methods."""
        model = self._make_model(bkd)
        assert model.nvars() == 6
        assert model.nqoi() == 2


class TestCantileverBeam2DConstraints:
    """Tests for constraint wrapper."""

    def _make_model(self, bkd):
        beam = CantileverBeam2DAnalytical(length=100.0, bkd=bkd)
        R = 40000.0
        D0 = 2.2535
        return CantileverBeam2DConstraints(beam, R, D0), R, D0

    def _nominal_sample(self, bkd):
        return bkd.array(
            [
                [500.0],
                [1000.0],
                [2.9e7],
                [40000.0],
                [2.5],
                [3.0],
            ]
        )

    def test_output_shape(self, bkd):
        """Verify output shape."""
        model, _, _ = self._make_model(bkd)
        result = model(self._nominal_sample(bkd))
        assert result.shape == (2, 1)

    def test_batch_shape(self, bkd):
        """Verify batch output shape."""
        model, _, _ = self._make_model(bkd)
        sample = self._nominal_sample(bkd)
        samples = bkd.concatenate([sample] * 5, axis=1)
        result = model(samples)
        assert result.shape == (2, 5)

    def test_values_are_safety_margins(self, bkd):
        """Constraints equal 1 - stress/R and 1 - disp/D0."""
        model, R, D0 = self._make_model(bkd)
        sample = self._nominal_sample(bkd)
        beam = CantileverBeam2DAnalytical(length=100.0, bkd=bkd)
        raw = beam(sample)  # (2, 1)

        expected_c1 = 1.0 - raw[0:1, :] / R
        expected_c2 = 1.0 - raw[1:2, :] / D0
        expected = bkd.concatenate([expected_c1, expected_c2], axis=0)

        result = model(sample)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_nvars_nqoi(self, bkd):
        model, _, _ = self._make_model(bkd)
        assert model.nvars() == 6
        assert model.nqoi() == 2

    def test_jacobian_derivative_checker(self, bkd):
        """Validate Jacobian via DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        model, _, _ = self._make_model(bkd)
        sample = self._nominal_sample(bkd)
        jac = model.jacobian(sample)
        assert jac.shape == (2, 6)

        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5


class TestCantileverBeam2DObjective:
    """Tests for objective wrapper."""

    def _make_model(self, bkd):
        return CantileverBeam2DObjective(bkd)

    def _nominal_sample(self, bkd):
        return bkd.array(
            [
                [500.0],
                [1000.0],
                [2.9e7],
                [40000.0],
                [2.5],
                [3.0],
            ]
        )

    def test_output_shape(self, bkd):
        model = self._make_model(bkd)
        result = model(self._nominal_sample(bkd))
        assert result.shape == (1, 1)

    def test_batch_shape(self, bkd):
        model = self._make_model(bkd)
        sample = self._nominal_sample(bkd)
        samples = bkd.concatenate([sample] * 5, axis=1)
        result = model(samples)
        assert result.shape == (1, 5)

    def test_value_is_area(self, bkd):
        """Objective equals w * t."""
        model = self._make_model(bkd)
        sample = self._nominal_sample(bkd)
        result = model(sample)
        expected = sample[4:5, :] * sample[5:6, :]  # w * t
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_nvars_nqoi(self, bkd):
        model = self._make_model(bkd)
        assert model.nvars() == 6
        assert model.nqoi() == 1

    def test_jacobian_derivative_checker(self, bkd):
        """Validate Jacobian via DerivativeChecker."""
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        model = self._make_model(bkd)
        sample = self._nominal_sample(bkd)
        jac = model.jacobian(sample)
        assert jac.shape == (1, 6)

        checker = DerivativeChecker(model)
        errors = checker.check_derivatives(sample, relative=True)[0]
        ratio = float(bkd.to_numpy(checker.error_ratio(errors)))
        assert ratio <= 1e-5


class TestCantileverBeam2DRegistry:
    def test_registry_access(self):
        """Verify benchmark is accessible from registry."""
        # Trigger registration
        import pyapprox.benchmarks.instances.analytic.cantilever_beam_2d  # noqa: F401
        from pyapprox.benchmarks.registry import BenchmarkRegistry

        bkd = NumpyBkd()
        bm = BenchmarkRegistry.get(
            "cantilever_beam_2d_analytical",
            bkd=bkd,
        )
        func = bm.function()
        assert func.nvars() == 6
        assert func.nqoi() == 2
