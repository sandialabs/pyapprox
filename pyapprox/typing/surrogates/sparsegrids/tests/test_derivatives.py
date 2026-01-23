"""Dual-backend tests for sparse grid derivatives using DerivativeChecker.

Tests run on both NumPy and PyTorch backends using the base class pattern.
"""

import unittest
from typing import Any, Generic, List

import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests

from pyapprox.typing.surrogates.sparsegrids import (
    IsotropicCombinationSparseGrid,
    SparseGridFunction,
)
from pyapprox.typing.surrogates.sparsegrids.basis_factory import (
    BasisFactoryProtocol,
    GaussLagrangeFactory,
)
from pyapprox.typing.probability import UniformMarginal
from pyapprox.typing.surrogates.affine.indices import LinearGrowthRule
from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)


class TestSparseGridDerivatives(Generic[Array], unittest.TestCase):
    """Tests for sparse grid derivatives using DerivativeChecker."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_jacobian_linear_function(self) -> None:
        """Test Jacobian of linear function is constant."""
        nvars = 2
        level = 2

        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories: List[BasisFactoryProtocol[Array]] = [
            GaussLagrangeFactory(marginal, self._bkd) for _ in range(nvars)
        ]
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # f(x, y) = x + 2*y
        samples = grid.get_samples()
        values = self._bkd.reshape(
            samples[0, :] + 2 * samples[1, :], (1, -1)
        )
        grid.set_values(values)

        # Wrap for derivative checker
        sg_func = SparseGridFunction(grid)

        # Check derivatives
        test_pt = self._bkd.asarray([[0.3], [0.4]])
        checker = DerivativeChecker(sg_func)
        errors = checker.check_derivatives(test_pt, verbosity=0)

        # Jacobian should be [1, 2] - shape is (nqoi, nvars) = (1, 2)
        jac = sg_func.jacobian(test_pt)
        expected_jac = self._bkd.asarray([[1.0, 2.0]])

        self.assertTrue(self._bkd.allclose(jac, expected_jac, rtol=1e-6))

        # Error ratio should be small for linear function
        jac_error = float(checker.error_ratio(errors[0]).item())
        self.assertLess(jac_error, 1e-6)

    def test_jacobian_quadratic_function(self) -> None:
        """Test Jacobian of quadratic function."""
        nvars = 2
        level = 3

        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories: List[BasisFactoryProtocol[Array]] = [
            GaussLagrangeFactory(marginal, self._bkd) for _ in range(nvars)
        ]
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # f(x, y) = x^2 + x*y
        samples = grid.get_samples()
        x, y = samples[0, :], samples[1, :]
        values = self._bkd.reshape(x ** 2 + x * y, (1, -1))
        grid.set_values(values)

        # Wrap for derivative checker
        sg_func = SparseGridFunction(grid)

        # Check derivatives
        test_pt = self._bkd.asarray([[0.3], [0.4]])
        checker = DerivativeChecker(sg_func)
        errors = checker.check_derivatives(test_pt, verbosity=0)

        # Jacobian at (0.3, 0.4) should be [2*0.3 + 0.4, 0.3] = [1.0, 0.3]
        # Shape is (nqoi, nvars) = (1, 2)
        jac = sg_func.jacobian(test_pt)
        expected_jac = self._bkd.asarray([[1.0, 0.3]])

        self.assertTrue(self._bkd.allclose(jac, expected_jac, rtol=1e-6))

        # Error ratio should be small
        jac_error = float(checker.error_ratio(errors[0]).item())
        self.assertLess(jac_error, 1e-6)

    def test_jacobian_3d_function(self) -> None:
        """Test Jacobian of 3D function."""
        nvars = 3
        level = 2

        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories: List[BasisFactoryProtocol[Array]] = [
            GaussLagrangeFactory(marginal, self._bkd) for _ in range(nvars)
        ]
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # f(x, y, z) = x + y + z
        samples = grid.get_samples()
        values = self._bkd.reshape(
            samples[0, :] + samples[1, :] + samples[2, :], (1, -1)
        )
        grid.set_values(values)

        # Wrap for derivative checker
        sg_func = SparseGridFunction(grid)

        # Check derivatives
        test_pt = self._bkd.asarray([[0.1], [0.2], [0.3]])
        checker = DerivativeChecker(sg_func)
        errors = checker.check_derivatives(test_pt, verbosity=0)

        # Jacobian should be [1, 1, 1] - shape is (nqoi, nvars) = (1, 3)
        jac = sg_func.jacobian(test_pt)
        expected_jac = self._bkd.asarray([[1.0, 1.0, 1.0]])

        self.assertTrue(self._bkd.allclose(jac, expected_jac, rtol=1e-6))

        # Error ratio should be small
        jac_error = float(checker.error_ratio(errors[0]).item())
        self.assertLess(jac_error, 1e-6)

    def test_derivative_checker_passes(self) -> None:
        """Test that DerivativeChecker passes for sparse grid function."""
        nvars = 2
        level = 3

        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories: List[BasisFactoryProtocol[Array]] = [
            GaussLagrangeFactory(marginal, self._bkd) for _ in range(nvars)
        ]
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # f(x, y) = x^2 + y^2
        samples = grid.get_samples()
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )
        grid.set_values(values)

        # Wrap for derivative checker
        sg_func = SparseGridFunction(grid)

        # Check derivatives at multiple points
        for x_val, y_val in [(0.0, 0.0), (0.3, 0.4), (-0.5, 0.2)]:
            test_pt = self._bkd.asarray([[x_val], [y_val]])
            checker = DerivativeChecker(sg_func)
            errors = checker.check_derivatives(test_pt, verbosity=0)

            jac_error = float(checker.error_ratio(errors[0]).item())
            self.assertLess(jac_error, 1e-6)

    def test_hessian_vector_product(self) -> None:
        """Test HVP computation."""
        nvars = 2
        level = 3

        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories: List[BasisFactoryProtocol[Array]] = [
            GaussLagrangeFactory(marginal, self._bkd) for _ in range(nvars)
        ]
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # f(x, y) = x^2 + y^2
        samples = grid.get_samples()
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )
        grid.set_values(values)

        # Wrap for derivative checker
        sg_func = SparseGridFunction(grid)

        # Check HVP
        test_pt = self._bkd.asarray([[0.3], [0.4]])
        vec = self._bkd.asarray([[1.0], [0.0]])

        hvp = sg_func.hvp(test_pt, vec)

        # Hessian is [[2, 0], [0, 2]]
        # HVP with [1, 0] should give [2, 0]
        expected_hvp = self._bkd.asarray([[2.0], [0.0]])

        self.assertTrue(self._bkd.allclose(hvp, expected_hvp, rtol=1e-6))

    def test_weighted_hvp(self) -> None:
        """Test weighted HVP computation."""
        nvars = 2
        level = 3

        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories: List[BasisFactoryProtocol[Array]] = [
            GaussLagrangeFactory(marginal, self._bkd) for _ in range(nvars)
        ]
        growth = LinearGrowthRule(scale=1, shift=1)

        grid = IsotropicCombinationSparseGrid(
            self._bkd, factories, growth, level
        )

        # f(x, y) = x^2 + y^2
        samples = grid.get_samples()
        values = self._bkd.reshape(
            samples[0, :] ** 2 + samples[1, :] ** 2, (1, -1)
        )
        grid.set_values(values)

        # Wrap for derivative checker
        sg_func = SparseGridFunction(grid)

        # Check WHVP
        test_pt = self._bkd.asarray([[0.3], [0.4]])
        vec = self._bkd.asarray([[1.0], [1.0]])
        weights = self._bkd.asarray([[0.5]])  # Shape (1, nqoi)

        whvp = sg_func.whvp(test_pt, vec, weights)

        # Hessian is [[2, 0], [0, 2]]
        # WHVP with [1, 1] and weight 0.5 should give 0.5 * [2, 2] = [1, 1]
        expected_whvp = self._bkd.asarray([[1.0], [1.0]])

        self.assertTrue(self._bkd.allclose(whvp, expected_whvp, rtol=1e-6))


# NumPy backend tests
class TestSparseGridDerivativesNumpy(TestSparseGridDerivatives[NDArray[Any]]):
    """NumPy backend tests for sparse grid derivatives."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestSparseGridDerivativesTorch(TestSparseGridDerivatives[torch.Tensor]):
    """PyTorch backend tests for sparse grid derivatives."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    def bkd(self) -> TorchBkd:
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
