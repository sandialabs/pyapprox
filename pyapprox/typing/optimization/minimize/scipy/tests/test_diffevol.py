import unittest
from typing import Generic, Any

from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backend import Array, Backend
from pyapprox.typing.util.numpy import NumpyBkd
from pyapprox.typing.util.torch import TorchBkd
from pyapprox.typing.util.abstracttestcase import AbstractTestCase
from pyapprox.typing.optimization.minimize.scipy.diffevol import (
    ScipyDifferentialEvolutionOptimizer,
)
from pyapprox.typing.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)


class TestScipyDifferentialEvolutionOptimizer(
    Generic[Array], AbstractTestCase
):
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
        result = optimizer.minimize()

        # Assert that the optimization was successful
        self.assertTrue(result.success())

        # Derive the analytical solution
        expected_optima = bkd.array([0.0, 0.0])[:, None]

        # Assert that the result matches the expected values
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

        # Assert that the objective value matches the expected value
        expected_fun = value_function(expected_optima)
        self.assertAlmostEqual(result.fun(), expected_fun, places=6)


class TestScipyDifferentialEvolutionOptimizerNumpy(
    TestScipyDifferentialEvolutionOptimizer[NDArray[Any]], unittest.TestCase
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestScipyDifferentialEvolutionOptimizerTorch(
    TestScipyDifferentialEvolutionOptimizer[torch.Tensor], unittest.TestCase
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> Backend[torch.Tensor]:
        return self._bkd


if __name__ == "__main__":
    unittest.main()
