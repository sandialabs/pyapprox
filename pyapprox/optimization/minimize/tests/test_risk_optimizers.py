"""
Tests for minimax and AVaR optimizers.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.test_utils import load_tests

from pyapprox.optimization.minimize.minimax import (
    MinimaxOptimizer,
    MinimaxObjective,
    MinimaxConstraint,
)
from pyapprox.optimization.minimize.avar import (
    AVaROptimizer,
    AVaRObjective,
    AVaRConstraint,
)


class SimpleMultiQoI(Generic[Array]):
    """Simple multi-output objective for testing: f_i(x) = (x - c_i)^2."""

    def __init__(
        self,
        centers: Array,
        bkd: Backend[Array],
    ) -> None:
        self._centers = centers  # Shape: (nqoi, 1)
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 1

    def nqoi(self) -> int:
        return self._centers.shape[0]

    def __call__(self, sample: Array) -> Array:
        """Evaluate (x - c_i)^2 for all centers."""
        x = sample[0, 0]
        return (x - self._centers) ** 2

    def jacobian(self, sample: Array) -> Array:
        """Jacobian: 2(x - c_i)."""
        x = sample[0, 0]
        return 2 * (x - self._centers)


class TestMinimaxOptimizer(Generic[Array], unittest.TestCase):
    """Base class for minimax optimizer tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_minimax_symmetric_objectives(self) -> None:
        """Minimax of symmetric objectives should be at center."""
        # f_1 = (x-1)^2, f_2 = (x+1)^2
        # Minimax occurs at x=0 with t=1
        centers = self._bkd.asarray([[-1.0], [1.0]])
        model = SimpleMultiQoI(centers, self._bkd)
        bounds = self._bkd.asarray([[-3.0, 3.0]])

        optimizer = MinimaxOptimizer(model, bounds, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        x_opt = optimizer.extract_original_variables(optima)
        t_opt = optimizer.get_minimax_value(optima)

        self.assertTrue(result.success())
        self.assertTrue(self._bkd.allclose(x_opt, self._bkd.zeros((1, 1)), atol=1e-3))
        self.assertTrue(self._bkd.allclose(t_opt, self._bkd.ones((1, 1)), atol=1e-3))

    def test_minimax_asymmetric_objectives(self) -> None:
        """Minimax of asymmetric objectives."""
        # f_1 = (x-0)^2, f_2 = (x-2)^2
        # Minimax occurs at x=1 with t=1
        centers = self._bkd.asarray([[0.0], [2.0]])
        model = SimpleMultiQoI(centers, self._bkd)
        bounds = self._bkd.asarray([[-3.0, 5.0]])

        optimizer = MinimaxOptimizer(model, bounds, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        x_opt = optimizer.extract_original_variables(optima)
        t_opt = optimizer.get_minimax_value(optima)

        self.assertTrue(result.success())
        self.assertTrue(self._bkd.allclose(x_opt, self._bkd.ones((1, 1)), atol=1e-3))
        self.assertTrue(self._bkd.allclose(t_opt, self._bkd.ones((1, 1)), atol=1e-3))

    def test_minimax_three_objectives(self) -> None:
        """Minimax with three objectives."""
        # f_1 = (x+1)^2, f_2 = x^2, f_3 = (x-1)^2
        # At x=0: f = [1, 0, 1], max = 1
        centers = self._bkd.asarray([[-1.0], [0.0], [1.0]])
        model = SimpleMultiQoI(centers, self._bkd)
        bounds = self._bkd.asarray([[-2.0, 2.0]])

        optimizer = MinimaxOptimizer(model, bounds, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        x_opt = optimizer.extract_original_variables(optima)
        t_opt = optimizer.get_minimax_value(optima)

        self.assertTrue(result.success())
        self.assertTrue(self._bkd.allclose(x_opt, self._bkd.zeros((1, 1)), atol=1e-3))
        self.assertTrue(self._bkd.allclose(t_opt, self._bkd.ones((1, 1)), atol=1e-3))


class TestMinimaxObjective(Generic[Array], unittest.TestCase):
    """Base class for minimax objective tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_objective_value(self) -> None:
        """Objective should return t value."""
        obj = MinimaxObjective(nmodel_vars=2, bkd=self._bkd)

        # [t, x1, x2] = [5.0, 1.0, 2.0]
        sample = self._bkd.asarray([[5.0], [1.0], [2.0]])
        result = obj(sample)

        expected = self._bkd.asarray([[5.0]])
        self.assertTrue(self._bkd.allclose(result, expected))

    def test_objective_jacobian(self) -> None:
        """Jacobian should be [1, 0, 0, ...]."""
        obj = MinimaxObjective(nmodel_vars=2, bkd=self._bkd)

        sample = self._bkd.asarray([[5.0], [1.0], [2.0]])
        jac = obj.jacobian(sample)

        expected = self._bkd.asarray([[1.0, 0.0, 0.0]])
        self.assertTrue(self._bkd.allclose(jac, expected))


class TestMinimaxConstraint(Generic[Array], unittest.TestCase):
    """Base class for minimax constraint tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        centers = self._bkd.asarray([[0.0], [1.0]])
        self._model = SimpleMultiQoI(centers, self._bkd)
        self._constraint = MinimaxConstraint(self._model)

    def test_constraint_values(self) -> None:
        """Constraint g_i = t - f_i(x) should be computed correctly."""
        # t=2, x=0: f=[0, 1], g=[2, 1]
        sample = self._bkd.asarray([[2.0], [0.0]])
        result = self._constraint(sample)

        expected = self._bkd.asarray([[2.0], [1.0]])
        self.assertTrue(self._bkd.allclose(result, expected))

    def test_constraint_jacobian(self) -> None:
        """Jacobian should be [1, -df/dx]."""
        sample = self._bkd.asarray([[2.0], [0.0]])
        jac = self._constraint.jacobian(sample)

        # At x=0: df/dx = [0, -2] (for centers [0, 1])
        # Jacobian: [1, 0], [1, 2]
        expected = self._bkd.asarray([[1.0, 0.0], [1.0, 2.0]])
        self.assertTrue(self._bkd.allclose(jac, expected))


class TestAVaROptimizer(Generic[Array], unittest.TestCase):
    """Base class for AVaR optimizer tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_avar_alpha_zero_equals_mean(self) -> None:
        """AVaR with alpha=0 should equal the mean."""
        centers = self._bkd.asarray([[-1.0], [0.0], [1.0]])
        model = SimpleMultiQoI(centers, self._bkd)
        bounds = self._bkd.asarray([[-2.0, 2.0]])

        optimizer = AVaROptimizer(model, bounds, alpha=0.0, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        avar_val = optimizer.get_avar_value(optima)

        # At x=0: f = [1, 0, 1], mean = 2/3
        expected_mean = self._bkd.asarray([[2.0 / 3.0]])
        self.assertTrue(result.success())
        self.assertTrue(self._bkd.allclose(avar_val, expected_mean, atol=1e-3))

    def test_avar_alpha_half(self) -> None:
        """AVaR with alpha=0.5 averages worst 50%."""
        centers = self._bkd.asarray([[-1.0], [0.0], [1.0]])
        model = SimpleMultiQoI(centers, self._bkd)
        bounds = self._bkd.asarray([[-2.0, 2.0]])

        optimizer = AVaROptimizer(model, bounds, alpha=0.5, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        x_opt = optimizer.extract_original_variables(optima)
        avar_val = optimizer.get_avar_value(optima)

        # At x=0: f = [1, 0, 1], sorted=[1, 1, 0], top 50% avg = 1
        self.assertTrue(result.success())
        self.assertTrue(self._bkd.allclose(x_opt, self._bkd.zeros((1, 1)), atol=1e-3))
        self.assertTrue(self._bkd.allclose(avar_val, self._bkd.ones((1, 1)), atol=1e-2))

    def test_avar_approaches_minimax_for_high_alpha(self) -> None:
        """AVaR with high alpha should approach minimax."""
        centers = self._bkd.asarray([[-1.0], [0.0], [1.0]])
        model = SimpleMultiQoI(centers, self._bkd)
        bounds = self._bkd.asarray([[-2.0, 2.0]])

        # High alpha (near 1)
        optimizer = AVaROptimizer(model, bounds, alpha=0.9, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        x_opt = optimizer.extract_original_variables(optima)
        avar_val = optimizer.get_avar_value(optima)

        # Should be close to minimax: x=0, value=1
        self.assertTrue(result.success())
        self.assertTrue(self._bkd.allclose(x_opt, self._bkd.zeros((1, 1)), atol=1e-2))
        self.assertTrue(self._bkd.allclose(avar_val, self._bkd.ones((1, 1)), atol=0.1))


class TestAVaRObjective(Generic[Array], unittest.TestCase):
    """Base class for AVaR objective tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_objective_value_alpha_half(self) -> None:
        """Test AVaR objective computation."""
        obj = AVaRObjective(nmodel_vars=1, nscenarios=3, alpha=0.5, bkd=self._bkd)

        # [t, s1, s2, s3, x] = [1.0, 0.5, 0.0, 0.5, 0.0]
        sample = self._bkd.asarray([[1.0], [0.5], [0.0], [0.5], [0.0]])
        result = obj(sample)

        # AVaR = t + (1/(n*(1-alpha))) * sum(s) = 1 + (1/(3*0.5)) * 1 = 1.667
        expected = self._bkd.asarray([[1.0 + 2.0 / 3.0]])
        self.assertTrue(self._bkd.allclose(result, expected))

    def test_objective_jacobian(self) -> None:
        """Test AVaR objective Jacobian."""
        obj = AVaRObjective(nmodel_vars=1, nscenarios=3, alpha=0.5, bkd=self._bkd)

        sample = self._bkd.asarray([[1.0], [0.5], [0.0], [0.5], [0.0]])
        jac = obj.jacobian(sample)

        # Jacobian: [1, c, c, c, 0] where c = 1/(3*0.5) = 2/3
        c = 2.0 / 3.0
        expected = self._bkd.asarray([[1.0, c, c, c, 0.0]])
        self.assertTrue(self._bkd.allclose(jac, expected))


class TestAVaRConstraint(Generic[Array], unittest.TestCase):
    """Base class for AVaR constraint tests."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        centers = self._bkd.asarray([[-1.0], [0.0], [1.0]])
        self._model = SimpleMultiQoI(centers, self._bkd)
        self._constraint = AVaRConstraint(self._model)

    def test_constraint_values(self) -> None:
        """Test constraint g_i = t + s_i - f_i(x)."""
        # At x=0: f = [1, 0, 1]
        # With t=1, s=[0, 0, 0]: g = [0, 1, 0]
        sample = self._bkd.asarray([[1.0], [0.0], [0.0], [0.0], [0.0]])
        result = self._constraint(sample)

        expected = self._bkd.asarray([[0.0], [1.0], [0.0]])
        self.assertTrue(self._bkd.allclose(result, expected))

    def test_constraint_jacobian_shape(self) -> None:
        """Test constraint Jacobian shape."""
        sample = self._bkd.asarray([[1.0], [0.0], [0.0], [0.0], [0.0]])
        jac = self._constraint.jacobian(sample)

        # Shape: (nqoi, nvars) = (3, 5)
        self.assertEqual(jac.shape, (3, 5))


# NumPy backend tests
class TestMinimaxOptimizerNumpy(TestMinimaxOptimizer[NDArray[Any]]):
    """NumPy backend tests for MinimaxOptimizer."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMinimaxObjectiveNumpy(TestMinimaxObjective[NDArray[Any]]):
    """NumPy backend tests for MinimaxObjective."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMinimaxConstraintNumpy(TestMinimaxConstraint[NDArray[Any]]):
    """NumPy backend tests for MinimaxConstraint."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAVaROptimizerNumpy(TestAVaROptimizer[NDArray[Any]]):
    """NumPy backend tests for AVaROptimizer."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAVaRObjectiveNumpy(TestAVaRObjective[NDArray[Any]]):
    """NumPy backend tests for AVaRObjective."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAVaRConstraintNumpy(TestAVaRConstraint[NDArray[Any]]):
    """NumPy backend tests for AVaRConstraint."""

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


# PyTorch backend tests
class TestMinimaxOptimizerTorch(TestMinimaxOptimizer[torch.Tensor]):
    """PyTorch backend tests for MinimaxOptimizer."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestMinimaxObjectiveTorch(TestMinimaxObjective[torch.Tensor]):
    """PyTorch backend tests for MinimaxObjective."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestMinimaxConstraintTorch(TestMinimaxConstraint[torch.Tensor]):
    """PyTorch backend tests for MinimaxConstraint."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestAVaROptimizerTorch(TestAVaROptimizer[torch.Tensor]):
    """PyTorch backend tests for AVaROptimizer."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestAVaRObjectiveTorch(TestAVaRObjective[torch.Tensor]):
    """PyTorch backend tests for AVaRObjective."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


class TestAVaRConstraintTorch(TestAVaRConstraint[torch.Tensor]):
    """PyTorch backend tests for AVaRConstraint."""

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
