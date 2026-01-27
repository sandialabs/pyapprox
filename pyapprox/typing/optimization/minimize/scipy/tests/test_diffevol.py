import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.optimization.minimize.scipy.diffevol import (
    ScipyDifferentialEvolutionOptimizer,
)
from pyapprox.typing.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)


class TestScipyDifferentialEvolutionOptimizer(
    Generic[Array], unittest.TestCase
):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_optimizer_with_quadratic_objective(self) -> None:
        """
        Test the ScipyDifferentialEvolutionOptimizer class with a simple quadratic objective.
        """
        bkd = self.bkd()

        # Define the quadratic objective function
        def value_function(x: Array) -> Array:
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=0)

        # Wrap the function using FunctionFromCallable
        objective = FunctionFromCallable(
            nqoi=1,
            nvars=2,
            fun=value_function,
            bkd=bkd,
        )

        # Define bounds for the optimization variables
        bounds = bkd.asarray([[-5, 5], [-5, 5]])

        # Initialize the optimizer
        optimizer = ScipyDifferentialEvolutionOptimizer(
            objective=objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False,
        )

        # Perform optimization
        result = optimizer.minimize(bkd.asarray([[0, 0]]))

        # Assert that the optimization was successful
        self.assertTrue(result.success())

        # Derive the analytical solution
        expected_optima = bkd.array([0.0, 0.0])[:, None]

        # Assert that the result matches the expected values
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

        # Assert that the objective value matches the expected value
        expected_fun = value_function(expected_optima)
        self.assertAlmostEqual(result.fun(), expected_fun, places=6)

    def test_deferred_binding(self) -> None:
        """Test optimizer constructed without objective/bounds."""
        bkd = self.bkd()

        # Define a simple quadratic objective
        def value_function(x: Array) -> Array:
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=0)

        objective = FunctionFromCallable(
            nqoi=1, nvars=2, fun=value_function, bkd=bkd
        )

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[1.0, 1.0]])

        # Create optimizer without objective/bounds
        optimizer = ScipyDifferentialEvolutionOptimizer(
            maxiter=100, seed=42, tol=1e-6
        )
        self.assertFalse(optimizer.is_bound())

        # Should raise RuntimeError if minimizing without binding
        with self.assertRaises(RuntimeError):
            optimizer.minimize(init_guess)

        # Bind and optimize
        optimizer.bind(objective, bounds)
        self.assertTrue(optimizer.is_bound())
        result = optimizer.minimize(init_guess)

        # Check result
        self.assertTrue(result.success())
        expected_optima = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_copy(self) -> None:
        """Test copy() returns unbound optimizer with same options."""
        bkd = self.bkd()

        # Create objective
        def value_function(x: Array) -> Array:
            return bkd.stack([x[0] ** 2], axis=0)

        obj = FunctionFromCallable(nqoi=1, nvars=1, fun=value_function, bkd=bkd)
        bounds = bkd.array([[-5.0, 5.0]])

        # Create and bind optimizer
        optimizer = ScipyDifferentialEvolutionOptimizer(
            maxiter=500, seed=42, tol=1e-8
        )
        optimizer.bind(obj, bounds)
        self.assertTrue(optimizer.is_bound())

        # Copy should be unbound with same options
        copy_opt = optimizer.copy()
        self.assertFalse(copy_opt.is_bound())
        self.assertEqual(copy_opt._maxiter, 500)
        self.assertEqual(copy_opt._seed, 42)

        # Copy can be bound and used independently
        copy_opt.bind(obj, bounds)
        result = copy_opt.minimize(bkd.asarray([[2.0]]))
        bkd.assert_allclose(result.optima(), bkd.array([[0.0]]), atol=1e-5)

    def test_backward_compatibility(self) -> None:
        """Test existing API with objective/bounds in constructor still works."""
        bkd = self.bkd()

        def value_function(x: Array) -> Array:
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=0)

        objective = FunctionFromCallable(
            nqoi=1, nvars=2, fun=value_function, bkd=bkd
        )

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[1.0, 1.0]])

        # Create optimizer with objective/bounds in constructor (old API)
        optimizer = ScipyDifferentialEvolutionOptimizer(
            objective=objective, bounds=bounds, maxiter=100, seed=42, tol=1e-6
        )

        # Should be bound immediately
        self.assertTrue(optimizer.is_bound())

        # Should work without explicit bind()
        result = optimizer.minimize(init_guess)
        self.assertTrue(result.success())
        expected_optima = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_invalid_objective_raises_typeerror(self) -> None:
        """Test that binding an invalid objective raises TypeError."""
        bkd = self.bkd()
        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ScipyDifferentialEvolutionOptimizer(maxiter=100)

        # Plain function is not a valid objective (missing bkd, nvars, nqoi)
        invalid_objective = lambda x: x**2  # noqa: E731

        with self.assertRaises(TypeError) as context:
            optimizer.bind(invalid_objective, bounds)

        # Error message should mention missing methods
        self.assertIn("ObjectiveProtocol", str(context.exception))


class TestScipyDifferentialEvolutionOptimizerNumpy(
    TestScipyDifferentialEvolutionOptimizer[NDArray[Any]]
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestScipyDifferentialEvolutionOptimizerTorch(
    TestScipyDifferentialEvolutionOptimizer[torch.Tensor]
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


# Custom test loader to exclude the base class
def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    """
    Custom test loader to exclude the base class
    ContinuousScipyRandomVariable1D.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestScipyDifferentialEvolutionOptimizerNumpy,
        TestScipyDifferentialEvolutionOptimizerTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
