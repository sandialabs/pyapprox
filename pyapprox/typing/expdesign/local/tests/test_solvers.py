"""
Tests for local OED solvers.

Tests cover:
- ScipyLocalOEDSolver for scalar criteria (A-optimal)
- MinimaxLocalOEDSolver for G-optimal designs
- AVaRLocalOEDSolver for R-optimal designs
- Dual-backend support (NumPy and PyTorch)
"""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.test_utils import load_tests
from pyapprox.typing.expdesign.local.design_matrices import (
    LeastSquaresDesignMatrices,
)
from pyapprox.typing.expdesign.local.criteria import (
    DOptimalCriterion,
    AOptimalCriterion,
    GOptimalCriterion,
    ROptimalCriterion,
)
from pyapprox.typing.expdesign.local.solver import (
    ScipyLocalOEDSolver,
    MinimaxLocalOEDSolver,
    AVaRLocalOEDSolver,
)


class TestScipyLocalOEDSolver(Generic[Array], unittest.TestCase):
    """Base test class for SciPy local OED solver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

        # Create design factors
        self._ndesign_pts = 7
        self._ndesign_vars = 3
        self._design_factors = self._bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )

    def test_d_optimal_solver(self) -> None:
        """Test solver with D-optimal criterion (no HVP)."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        # D-optimal does not have HVP - optimizer should work without it
        crit = DOptimalCriterion(dm, self._bkd)
        solver = ScipyLocalOEDSolver(crit, self._bkd, verbosity=0)

        optimal_weights = solver.construct()

        # Check shape
        self.assertEqual(optimal_weights.shape, (self._ndesign_pts, 1))

        # Check simplex constraints
        weight_sum = self._bkd.sum(optimal_weights)
        self.assertTrue(
            self._bkd.allclose(weight_sum, self._bkd.asarray(1.0), atol=1e-6)
        )
        self.assertTrue(
            self._bkd.all_bool(optimal_weights >= -1e-8)
        )

    def test_a_optimal_solver(self) -> None:
        """Test solver with A-optimal criterion (has HVP)."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        # A-optimal creates adjoints internally using unit vectors
        crit = AOptimalCriterion(dm, self._bkd)
        solver = ScipyLocalOEDSolver(crit, self._bkd, verbosity=0)

        optimal_weights = solver.construct()

        # Check shape
        self.assertEqual(optimal_weights.shape, (self._ndesign_pts, 1))

        # Check simplex constraints
        weight_sum = self._bkd.sum(optimal_weights)
        self.assertTrue(
            self._bkd.allclose(weight_sum, self._bkd.asarray(1.0), atol=1e-6)
        )
        self.assertTrue(
            self._bkd.all_bool(optimal_weights >= -1e-8)
        )

    def test_raises_for_vector_criterion(self) -> None:
        """Test that solver raises error for vector criteria."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)

        # Create G-optimal criterion (vector output)
        npred_pts = 3
        pred_factors = self._bkd.asarray(
            np.random.randn(npred_pts, self._ndesign_vars)
        )

        crit = GOptimalCriterion(dm, pred_factors, self._bkd)

        # Should raise ValueError because nqoi > 1
        with self.assertRaises(ValueError):
            ScipyLocalOEDSolver(crit, self._bkd)

    def test_get_result(self) -> None:
        """Test that get_result returns optimization info."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        crit = AOptimalCriterion(dm, self._bkd)
        solver = ScipyLocalOEDSolver(crit, self._bkd, verbosity=0)

        # Should raise before construct()
        with self.assertRaises(AttributeError):
            solver.get_result()

        solver.construct()
        result = solver.get_result()

        # Check result has expected methods
        self.assertTrue(hasattr(result, 'success'))
        self.assertTrue(hasattr(result, 'optima'))


class TestMinimaxLocalOEDSolver(Generic[Array], unittest.TestCase):
    """Base test class for Minimax local OED solver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

        self._ndesign_pts = 7
        self._ndesign_vars = 3
        self._design_factors = self._bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._npred_pts = 4
        self._pred_factors = self._bkd.asarray(
            np.random.randn(self._npred_pts, self._ndesign_vars)
        )

    def _create_g_optimal_criterion(self):
        """Helper to create G-optimal criterion."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        # G-optimal takes pred_factors directly
        return GOptimalCriterion(dm, self._pred_factors, self._bkd)

    def test_g_optimal_solver(self) -> None:
        """Test solver with G-optimal criterion."""
        crit = self._create_g_optimal_criterion()
        solver = MinimaxLocalOEDSolver(crit, self._bkd, verbosity=0)

        optimal_weights = solver.construct()

        # Check shape
        self.assertEqual(optimal_weights.shape, (self._ndesign_pts, 1))

        # Check simplex constraints
        weight_sum = self._bkd.sum(optimal_weights)
        self.assertTrue(
            self._bkd.allclose(weight_sum, self._bkd.asarray(1.0), atol=1e-6)
        )
        self.assertTrue(
            self._bkd.all_bool(optimal_weights >= -1e-8)
        )

    def test_get_minimax_value(self) -> None:
        """Test that minimax value can be retrieved."""
        crit = self._create_g_optimal_criterion()
        solver = MinimaxLocalOEDSolver(crit, self._bkd, verbosity=0)

        # Should raise before construct()
        with self.assertRaises(AttributeError):
            solver.get_minimax_value()

        solver.construct()
        minimax_val = solver.get_minimax_value()

        # Check shape and positivity
        self.assertEqual(minimax_val.shape, (1, 1))
        self.assertTrue(minimax_val[0, 0] > 0)


class TestAVaRLocalOEDSolver(Generic[Array], unittest.TestCase):
    """Base test class for AVaR local OED solver."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

        self._ndesign_pts = 7
        self._ndesign_vars = 3
        self._design_factors = self._bkd.asarray(
            np.random.randn(self._ndesign_pts, self._ndesign_vars)
        )
        self._npred_pts = 4
        self._pred_factors = self._bkd.asarray(
            np.random.randn(self._npred_pts, self._ndesign_vars)
        )

    def _create_r_optimal_criterion(self):
        """Helper to create R-optimal criterion."""
        dm = LeastSquaresDesignMatrices(self._design_factors, self._bkd)
        # R-optimal inherits from G-optimal, takes pred_factors directly
        return ROptimalCriterion(dm, self._pred_factors, self._bkd)

    def test_r_optimal_solver(self) -> None:
        """Test solver with R-optimal criterion."""
        crit = self._create_r_optimal_criterion()
        solver = AVaRLocalOEDSolver(crit, alpha=0.5, bkd=self._bkd, verbosity=0)

        optimal_weights = solver.construct()

        # Check shape
        self.assertEqual(optimal_weights.shape, (self._ndesign_pts, 1))

        # Check simplex constraints
        weight_sum = self._bkd.sum(optimal_weights)
        self.assertTrue(
            self._bkd.allclose(weight_sum, self._bkd.asarray(1.0), atol=1e-6)
        )

    def test_alpha_validation(self) -> None:
        """Test that invalid alpha raises error."""
        crit = self._create_r_optimal_criterion()

        with self.assertRaises(ValueError):
            AVaRLocalOEDSolver(crit, alpha=-0.1, bkd=self._bkd)

        with self.assertRaises(ValueError):
            AVaRLocalOEDSolver(crit, alpha=1.0, bkd=self._bkd)

    def test_get_avar_value(self) -> None:
        """Test that AVaR value can be retrieved."""
        crit = self._create_r_optimal_criterion()
        solver = AVaRLocalOEDSolver(crit, alpha=0.5, bkd=self._bkd, verbosity=0)

        # Should raise before construct()
        with self.assertRaises(AttributeError):
            solver.get_avar_value()

        solver.construct()
        avar_val = solver.get_avar_value()

        # Check shape and positivity
        self.assertEqual(avar_val.shape, (1, 1))
        self.assertTrue(avar_val[0, 0] > 0)

    def test_alpha_zero_approaches_mean(self) -> None:
        """Test that alpha=0 gives mean-like behavior."""
        crit = self._create_r_optimal_criterion()
        solver = AVaRLocalOEDSolver(crit, alpha=0.0, bkd=self._bkd, verbosity=0)

        optimal_weights = solver.construct()
        avar_val = solver.get_avar_value()

        # Check that it's a valid design
        self.assertEqual(optimal_weights.shape, (self._ndesign_pts, 1))
        self.assertTrue(avar_val[0, 0] > 0)


# NumPy backend tests
class TestScipyLocalOEDSolverNumpy(TestScipyLocalOEDSolver[NDArray[Any]]):
    """NumPy backend tests for SciPy solver."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMinimaxLocalOEDSolverNumpy(TestMinimaxLocalOEDSolver[NDArray[Any]]):
    """NumPy backend tests for Minimax solver."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAVaRLocalOEDSolverNumpy(TestAVaRLocalOEDSolver[NDArray[Any]]):
    """NumPy backend tests for AVaR solver."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestScipyLocalOEDSolverTorch(TestScipyLocalOEDSolver[torch.Tensor]):
    """PyTorch backend tests for SciPy solver."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestMinimaxLocalOEDSolverTorch(TestMinimaxLocalOEDSolver[torch.Tensor]):
    """PyTorch backend tests for Minimax solver."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestAVaRLocalOEDSolverTorch(TestAVaRLocalOEDSolver[torch.Tensor]):
    """PyTorch backend tests for AVaR solver."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
