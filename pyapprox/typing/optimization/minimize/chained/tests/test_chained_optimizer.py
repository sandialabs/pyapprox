import unittest
from typing import Generic, Any

import numpy as np
from numpy.typing import NDArray
import torch

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.typing.optimization.minimize.scipy.diffevol import (
    ScipyDifferentialEvolutionOptimizer,
)
from pyapprox.typing.optimization.minimize.chained.chained_optimizer import (
    ChainedOptimizer,
)
from pyapprox.typing.optimization.minimize.benchmarks.evutushenko import (
    EvtushenkoObjective,
    EvtushenkoNonLinearConstraint,
)
from pyapprox.typing.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)


class TestChainedOptimizer(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the specific
        backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_chained_optimizer_with_evtushenko_objective_and_constraints(
        self,
    ) -> None:
        """
        Test the ChainedOptimizer class with the Evtushenko objective,
        nonlinear constraints, and a linear constraint.
        """
        bkd = self.bkd()

        # Define the Evtushenko objective
        objective = EvtushenkoObjective(backend=bkd)

        # Define the Evtushenko nonlinear constraint
        nonlinear_constraint = EvtushenkoNonLinearConstraint(backend=bkd)

        # Define the linear constraint
        linear_con = PyApproxLinearConstraint(
            bkd.ones((1, 3)), bkd.asarray([1.0]), bkd.asarray([1.0]), bkd
        )

        # Define bounds for the optimization variables
        bounds = bkd.array([[0, np.inf], [0, np.inf], [0, np.inf]])

        # Define finite bounds required by DifferentialEvolution
        truncated_bounds = bkd.array([[0, 2.0], [0, 2.0], [0, 2.0]])

        # Initialize the global optimizer (Differential Evolution)
        global_optimizer = ScipyDifferentialEvolutionOptimizer(
            objective=objective,
            bounds=truncated_bounds,
            strategy="best1bin",
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
        )

        # Initialize the local optimizer (Trust Constr)
        local_optimizer = ScipyTrustConstrOptimizer(
            objective=objective,
            bounds=bounds,
            constraints=[nonlinear_constraint, linear_con],
            verbosity=0,
            maxiter=100,
            gtol=1e-15,
        )

        # Chain the optimizers
        chained_optimizer = ChainedOptimizer(global_optimizer, local_optimizer)

        # Define initial guess for the global optimizer
        init_guess = bkd.asarray([[0.1], [0.7], [0.2]])

        # Perform optimization
        result = chained_optimizer.minimize(init_guess)

        # Assert that the optimization was successful
        self.assertTrue(result.success())

        # Assert that the constraints are satisfied
        nonlinear_constraint_value = nonlinear_constraint(result.optima())
        self.assertTrue(
            bkd.all_bool(
                nonlinear_constraint_value >= nonlinear_constraint.lb()
            )
            and bkd.all_bool(
                nonlinear_constraint_value <= nonlinear_constraint.ub()
            )
        )

        linear_constraint_value = bkd.ones((1, 3)) @ result.optima()
        self.assertAlmostEqual(linear_constraint_value[0, 0], 1.0, places=8)

        # Assert that the result matches the expected optima
        expected_optima = bkd.array([0.0, 0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-5)

        # Assert that the objective value matches the expected value
        expected_fun = objective(expected_optima)
        self.bkd().assert_allclose(
            result.fun(), float(expected_fun[0, 0]), atol=1e-8
        )

    def test_deferred_binding(self) -> None:
        """Test ChainedOptimizer with deferred binding."""
        bkd = self.bkd()

        # Define the Evtushenko objective
        objective = EvtushenkoObjective(backend=bkd)

        # Define the Evtushenko nonlinear constraint
        nonlinear_constraint = EvtushenkoNonLinearConstraint(backend=bkd)

        # Define the linear constraint
        linear_con = PyApproxLinearConstraint(
            bkd.ones((1, 3)), bkd.asarray([1.0]), bkd.asarray([1.0]), bkd
        )

        # Define bounds for the optimization variables
        bounds = bkd.array([[0, np.inf], [0, np.inf], [0, np.inf]])

        # Define finite bounds required by DifferentialEvolution
        truncated_bounds = bkd.array([[0, 2.0], [0, 2.0], [0, 2.0]])

        # Create unbound optimizers
        global_optimizer = ScipyDifferentialEvolutionOptimizer(
            strategy="best1bin",
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
        )

        local_optimizer = ScipyTrustConstrOptimizer(
            verbosity=0,
            maxiter=100,
            gtol=1e-15,
        )

        # Chain the unbound optimizers
        chained_optimizer = ChainedOptimizer(global_optimizer, local_optimizer)

        # Should not be bound
        self.assertFalse(chained_optimizer.is_bound())

        # Should raise RuntimeError if minimizing without binding
        init_guess = bkd.asarray([[0.1], [0.7], [0.2]])
        with self.assertRaises(RuntimeError):
            chained_optimizer.minimize(init_guess)

        # Bind global with truncated bounds (required for diffevol)
        global_optimizer.bind(objective, truncated_bounds)
        # Bind local with constraints
        local_optimizer.bind(
            objective, bounds, constraints=[nonlinear_constraint, linear_con]
        )

        # Now should be bound
        self.assertTrue(chained_optimizer.is_bound())

        # Perform optimization
        result = chained_optimizer.minimize(init_guess)

        # Assert that the optimization was successful
        self.assertTrue(result.success())

        # Assert that the result matches the expected optima
        expected_optima = bkd.array([0.0, 0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-5)

    def test_copy(self) -> None:
        """Test copy() returns unbound ChainedOptimizer with same options."""
        bkd = self.bkd()

        # Create objective
        objective = EvtushenkoObjective(backend=bkd)
        bounds = bkd.array([[0, 2.0], [0, 2.0], [0, 2.0]])

        # Create and bind optimizers
        global_optimizer = ScipyDifferentialEvolutionOptimizer(
            maxiter=500, seed=42
        )
        local_optimizer = ScipyTrustConstrOptimizer(maxiter=200)

        chained_optimizer = ChainedOptimizer(global_optimizer, local_optimizer)
        chained_optimizer.bind(objective, bounds)

        self.assertTrue(chained_optimizer.is_bound())

        # Copy should be unbound with same options
        copy_opt = chained_optimizer.copy()
        self.assertFalse(copy_opt.is_bound())

        # Underlying optimizers should also be copied
        self.assertFalse(copy_opt._global_optimizer.is_bound())
        self.assertFalse(copy_opt._local_optimizer.is_bound())

    def test_invalid_optimizer_raises_typeerror(self) -> None:
        """Test that passing an invalid optimizer raises TypeError."""
        bkd = self.bkd()

        global_optimizer = ScipyDifferentialEvolutionOptimizer(maxiter=100)

        # Plain object is not a valid optimizer
        invalid_optimizer = "not an optimizer"

        with self.assertRaises(TypeError) as context:
            ChainedOptimizer(global_optimizer, invalid_optimizer)  # type: ignore

        self.assertIn("BindableOptimizerProtocol", str(context.exception))


class TestChainedOptimizerNumpy(TestChainedOptimizer[NDArray[Any]]):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


class TestChainedOptimizerTorch(TestChainedOptimizer[torch.Tensor]):
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
        TestChainedOptimizerNumpy,
        TestChainedOptimizerTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


# Main block to explicitly run tests using the custom loader
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
