"""Tests for SobolGFunction."""

# TODO: this test class should be where function is defined
# not at this level which is for integration tests.

import pytest

from pyapprox_benchmarks.functions.algebraic.sobol_g import (
    SobolGFunction,
    SobolGSensitivityIndices,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols.function import (
    FunctionProtocol,
)
from pyapprox.interface.functions.protocols.hessian import (
    FunctionWithJacobianAndHVPProtocol,
)
from pyapprox.util.backends.numpy import NumpyBkd


class TestSobolGFunction:
    """Base tests for SobolGFunction."""

    def test_protocol_compliance_function(self, bkd) -> None:
        """Test that SobolGFunction satisfies FunctionProtocol."""
        func = SobolGFunction(bkd, a=[0, 1, 4.5])
        assert isinstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self, bkd) -> None:
        """Test that SobolGFunction satisfies FunctionWithJacobianAndHVPProtocol."""
        func = SobolGFunction(bkd, a=[0, 1, 4.5])
        assert isinstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars(self, bkd) -> None:
        """Test nvars returns correct count."""
        func = SobolGFunction(bkd, a=[0, 1, 4.5, 9])
        assert func.nvars() == 4

    def test_nqoi(self, bkd) -> None:
        """Test nqoi returns 1."""
        func = SobolGFunction(bkd, a=[0, 1])
        assert func.nqoi() == 1

    def test_empty_a_raises(self, bkd) -> None:
        """Test that empty a raises ValueError."""
        with pytest.raises(ValueError):
            SobolGFunction(bkd, a=[])

    def test_evaluation_at_center(self, bkd) -> None:
        """Test evaluation at center point (0.5, 0.5, ...)."""
        func = SobolGFunction(bkd, a=[0, 1, 4.5])
        # At x = 0.5: |4*0.5 - 2| = 0, so g_i = a_i / (1 + a_i)
        sample = bkd.array([[0.5], [0.5], [0.5]])
        result = func(sample)
        # g_0 = 0/1 = 0, so product = 0
        expected = bkd.array([[0.0]])
        bkd.assert_allclose(result, expected, atol=1e-14)

    def test_evaluation_at_corner(self, bkd) -> None:
        """Test evaluation at corner point (0, 0, ...)."""
        func = SobolGFunction(bkd, a=[0, 1, 4.5])
        # At x = 0: |4*0 - 2| = 2, so g_i = (2 + a_i) / (1 + a_i)
        sample = bkd.array([[0.0], [0.0], [0.0]])
        result = func(sample)
        # g_0 = 2/1 = 2, g_1 = 3/2 = 1.5, g_2 = 6.5/5.5
        g0 = 2.0 / 1.0
        g1 = 3.0 / 2.0
        g2 = 6.5 / 5.5
        expected = bkd.array([[g0 * g1 * g2]])
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_evaluation_batch(self, bkd) -> None:
        """Test evaluation at multiple samples."""
        func = SobolGFunction(bkd, a=[0, 1])
        samples = bkd.array(
            [
                [0.0, 0.5, 1.0],
                [0.0, 0.5, 1.0],
            ]
        )
        result = func(samples)
        assert result.shape == (1, 3)

    def test_jacobian_shape(self, bkd) -> None:
        """Test Jacobian has correct shape."""
        func = SobolGFunction(bkd, a=[0, 1, 4.5])
        sample = bkd.array([[0.3], [0.7], [0.1]])
        jac = func.jacobian(sample)
        assert jac.shape == (1, 3)

    def test_jacobian_invalid_shape(self, bkd) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = SobolGFunction(bkd, a=[0, 1])
        sample = bkd.array([[0.5, 0.1], [0.3, 0.2]])
        with pytest.raises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self, bkd) -> None:
        """Test HVP has correct shape."""
        func = SobolGFunction(bkd, a=[0, 1, 4.5])
        sample = bkd.array([[0.3], [0.7], [0.1]])
        vec = bkd.array([[1.0], [0.0], [0.0]])
        hvp = func.hvp(sample, vec)
        assert hvp.shape == (3, 1)

    def test_hvp_invalid_sample_shape(self, bkd) -> None:
        """Test HVP raises for invalid sample shape."""
        func = SobolGFunction(bkd, a=[0, 1])
        sample = bkd.array([[0.5, 0.1], [0.3, 0.2]])
        vec = bkd.array([[1.0], [0.0]])
        with pytest.raises(ValueError):
            func.hvp(sample, vec)

    def test_derivative_checker_jacobian(self, bkd) -> None:
        """Test Jacobian passes derivative checker."""
        func = SobolGFunction(bkd, a=[0, 1, 4.5])
        checker = DerivativeChecker(func)
        # Avoid x=0.5 where function is not differentiable
        sample = bkd.array([[0.3], [0.7], [0.1]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        assert error_ratio < 1e-6

    def test_derivative_checker_hvp(self, bkd) -> None:
        """Test HVP passes derivative checker."""
        func = SobolGFunction(bkd, a=[0, 1, 4.5])
        checker = DerivativeChecker(func)
        # Avoid x=0.5 where function is not differentiable
        sample = bkd.array([[0.3], [0.7], [0.1]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        assert error_ratio < 1e-6


class TestSobolGSensitivityIndices:
    """Tests for SobolGSensitivityIndices class."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._bkd = NumpyBkd()

    def test_standard_6d_indices(self) -> None:
        """Test analytical indices for standard 6D configuration."""
        a = [0, 1, 4.5, 9, 99, 99]
        indices = SobolGSensitivityIndices(self._bkd, a)
        main = indices.main_effects()
        total = indices.total_effects()
        var = indices.variance()

        # main and total have shape (nvars, 1)
        assert main.shape == (6, 1)
        assert total.shape == (6, 1)

        # First variable (a=0) should be most important
        assert self._bkd.to_int(self._bkd.argmax(main[:, 0])) == 0
        assert self._bkd.to_int(self._bkd.argmax(total[:, 0])) == 0

        # Main effects should sum to less than 1 (interactions exist)
        assert self._bkd.to_float(self._bkd.sum(main)) < 1.0

        # All indices should be non-negative
        assert self._bkd.all_bool(main >= 0)
        assert self._bkd.all_bool(total >= 0)

        # Check variance is positive
        assert float(var[0]) > 0

    def test_single_important_variable(self) -> None:
        """Test with one important variable."""
        # a=0 means most important, a=99 means almost irrelevant
        a = [0, 99, 99]
        indices = SobolGSensitivityIndices(self._bkd, a)
        main = indices.main_effects()

        # main has shape (nvars, 1)
        assert main.shape == (3, 1)

        # First variable should dominate
        assert float(main[0, 0]) > 0.9
        assert float(main[1, 0]) < 0.01
        assert float(main[2, 0]) < 0.01

    def test_mean_is_one(self) -> None:
        """Test that mean is always 1 for Sobol G function."""
        a = [0, 1, 4.5]
        indices = SobolGSensitivityIndices(self._bkd, a)
        self._bkd.assert_allclose(indices.mean(), self._bkd.asarray([1.0]))

    def test_sobol_indices_shape(self) -> None:
        """Test that sobol_indices has correct shape (main + pairwise)."""
        a = [0, 1, 4.5]  # 3 variables
        indices = SobolGSensitivityIndices(self._bkd, a)
        sobol = indices.sobol_indices()
        # 3 main + 3 pairwise = 6
        assert sobol.shape == (6, 1)

    def test_interaction_indices_shape(self) -> None:
        """Test that interaction indices matrix has correct shape."""
        a = [0, 1, 4.5]  # 3 variables
        indices = SobolGSensitivityIndices(self._bkd, a)
        interaction = indices.sobol_interaction_indices()
        # Shape: (nvars, nindices) = (3, 6)
        assert interaction.shape == (3, 6)
