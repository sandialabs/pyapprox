"""Tests for SobolGFunction."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.benchmarks.functions.algebraic.sobol_g import (
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
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestSobolGFunction(Generic[Array], unittest.TestCase):
    """Base tests for SobolGFunction."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_protocol_compliance_function(self) -> None:
        """Test that SobolGFunction satisfies FunctionProtocol."""
        func = SobolGFunction(self._bkd, a=[0, 1, 4.5])
        self.assertIsInstance(func, FunctionProtocol)

    def test_protocol_compliance_jacobian_hvp(self) -> None:
        """Test that SobolGFunction satisfies FunctionWithJacobianAndHVPProtocol."""
        func = SobolGFunction(self._bkd, a=[0, 1, 4.5])
        self.assertIsInstance(func, FunctionWithJacobianAndHVPProtocol)

    def test_nvars(self) -> None:
        """Test nvars returns correct count."""
        func = SobolGFunction(self._bkd, a=[0, 1, 4.5, 9])
        self.assertEqual(func.nvars(), 4)

    def test_nqoi(self) -> None:
        """Test nqoi returns 1."""
        func = SobolGFunction(self._bkd, a=[0, 1])
        self.assertEqual(func.nqoi(), 1)

    def test_empty_a_raises(self) -> None:
        """Test that empty a raises ValueError."""
        with self.assertRaises(ValueError):
            SobolGFunction(self._bkd, a=[])

    def test_evaluation_at_center(self) -> None:
        """Test evaluation at center point (0.5, 0.5, ...)."""
        func = SobolGFunction(self._bkd, a=[0, 1, 4.5])
        # At x = 0.5: |4*0.5 - 2| = 0, so g_i = a_i / (1 + a_i)
        sample = self._bkd.array([[0.5], [0.5], [0.5]])
        result = func(sample)
        # g_0 = 0/1 = 0, so product = 0
        expected = self._bkd.array([[0.0]])
        self._bkd.assert_allclose(result, expected, atol=1e-14)

    def test_evaluation_at_corner(self) -> None:
        """Test evaluation at corner point (0, 0, ...)."""
        func = SobolGFunction(self._bkd, a=[0, 1, 4.5])
        # At x = 0: |4*0 - 2| = 2, so g_i = (2 + a_i) / (1 + a_i)
        sample = self._bkd.array([[0.0], [0.0], [0.0]])
        result = func(sample)
        # g_0 = 2/1 = 2, g_1 = 3/2 = 1.5, g_2 = 6.5/5.5
        g0 = 2.0 / 1.0
        g1 = 3.0 / 2.0
        g2 = 6.5 / 5.5
        expected = self._bkd.array([[g0 * g1 * g2]])
        self._bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_evaluation_batch(self) -> None:
        """Test evaluation at multiple samples."""
        func = SobolGFunction(self._bkd, a=[0, 1])
        samples = self._bkd.array(
            [
                [0.0, 0.5, 1.0],
                [0.0, 0.5, 1.0],
            ]
        )
        result = func(samples)
        self.assertEqual(result.shape, (1, 3))

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        func = SobolGFunction(self._bkd, a=[0, 1, 4.5])
        sample = self._bkd.array([[0.3], [0.7], [0.1]])
        jac = func.jacobian(sample)
        self.assertEqual(jac.shape, (1, 3))

    def test_jacobian_invalid_shape(self) -> None:
        """Test Jacobian raises for invalid input shape."""
        func = SobolGFunction(self._bkd, a=[0, 1])
        sample = self._bkd.array([[0.5, 0.1], [0.3, 0.2]])
        with self.assertRaises(ValueError):
            func.jacobian(sample)

    def test_hvp_shape(self) -> None:
        """Test HVP has correct shape."""
        func = SobolGFunction(self._bkd, a=[0, 1, 4.5])
        sample = self._bkd.array([[0.3], [0.7], [0.1]])
        vec = self._bkd.array([[1.0], [0.0], [0.0]])
        hvp = func.hvp(sample, vec)
        self.assertEqual(hvp.shape, (3, 1))

    def test_hvp_invalid_sample_shape(self) -> None:
        """Test HVP raises for invalid sample shape."""
        func = SobolGFunction(self._bkd, a=[0, 1])
        sample = self._bkd.array([[0.5, 0.1], [0.3, 0.2]])
        vec = self._bkd.array([[1.0], [0.0]])
        with self.assertRaises(ValueError):
            func.hvp(sample, vec)

    def test_derivative_checker_jacobian(self) -> None:
        """Test Jacobian passes derivative checker."""
        func = SobolGFunction(self._bkd, a=[0, 1, 4.5])
        checker = DerivativeChecker(func)
        # Avoid x=0.5 where function is not differentiable
        sample = self._bkd.array([[0.3], [0.7], [0.1]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[0])
        self.assertLess(error_ratio, 1e-6)

    def test_derivative_checker_hvp(self) -> None:
        """Test HVP passes derivative checker."""
        func = SobolGFunction(self._bkd, a=[0, 1, 4.5])
        checker = DerivativeChecker(func)
        # Avoid x=0.5 where function is not differentiable
        sample = self._bkd.array([[0.3], [0.7], [0.1]])
        errors = checker.check_derivatives(sample, verbosity=0)
        error_ratio = checker.error_ratio(errors[1])
        self.assertLess(error_ratio, 1e-6)


class TestSobolGFunctionNumpy(TestSobolGFunction[NDArray[Any]]):
    """NumPy backend tests for SobolGFunction."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSobolGFunctionTorch(TestSobolGFunction[torch.Tensor]):
    """PyTorch backend tests for SobolGFunction."""

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestSobolGSensitivityIndices(unittest.TestCase):
    """Tests for SobolGSensitivityIndices class."""

    def setUp(self) -> None:
        self._bkd = NumpyBkd()

    def test_standard_6d_indices(self) -> None:
        """Test analytical indices for standard 6D configuration."""
        a = [0, 1, 4.5, 9, 99, 99]
        indices = SobolGSensitivityIndices(self._bkd, a)
        main = indices.main_effects()
        total = indices.total_effects()
        var = indices.variance()

        # main and total have shape (nvars, 1)
        self.assertEqual(main.shape, (6, 1))
        self.assertEqual(total.shape, (6, 1))

        # First variable (a=0) should be most important
        self.assertEqual(int(self._bkd.argmax(main[:, 0])), 0)
        self.assertEqual(int(self._bkd.argmax(total[:, 0])), 0)

        # Main effects should sum to less than 1 (interactions exist)
        self.assertLess(float(self._bkd.sum(main)), 1.0)

        # All indices should be non-negative
        self.assertTrue(self._bkd.all_bool(main >= 0))
        self.assertTrue(self._bkd.all_bool(total >= 0))

        # Check variance is positive
        self.assertGreater(float(var[0]), 0)

    def test_single_important_variable(self) -> None:
        """Test with one important variable."""
        # a=0 means most important, a=99 means almost irrelevant
        a = [0, 99, 99]
        indices = SobolGSensitivityIndices(self._bkd, a)
        main = indices.main_effects()

        # main has shape (nvars, 1)
        self.assertEqual(main.shape, (3, 1))

        # First variable should dominate
        self.assertGreater(float(main[0, 0]), 0.9)
        self.assertLess(float(main[1, 0]), 0.01)
        self.assertLess(float(main[2, 0]), 0.01)

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
        self.assertEqual(sobol.shape, (6, 1))

    def test_interaction_indices_shape(self) -> None:
        """Test that interaction indices matrix has correct shape."""
        a = [0, 1, 4.5]  # 3 variables
        indices = SobolGSensitivityIndices(self._bkd, a)
        interaction = indices.sobol_interaction_indices()
        # Shape: (nvars, nindices) = (3, 6)
        self.assertEqual(interaction.shape, (3, 6))


if __name__ == "__main__":
    unittest.main()
