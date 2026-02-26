import pytest

from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.optimization.minimize.scipy.diffevol import (
    ScipyDifferentialEvolutionOptimizer,
)


class TestScipyDifferentialEvolutionOptimizer:

    def test_optimizer_with_quadratic_objective(self, bkd) -> None:
        """
        Test the ScipyDifferentialEvolutionOptimizer class with a simple quadratic
        objective.
        """
        # Define the quadratic objective function
        def value_function(x):
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
        assert result.success()

        # Derive the analytical solution
        expected_optima = bkd.array([0.0, 0.0])[:, None]

        # Assert that the result matches the expected values
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

        # Assert that the objective value matches the expected value
        expected_fun = value_function(expected_optima)
        assert result.fun() == pytest.approx(expected_fun, abs=1e-6)

    def test_deferred_binding(self, bkd) -> None:
        """Test optimizer constructed without objective/bounds."""
        # Define a simple quadratic objective
        def value_function(x):
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=0)

        objective = FunctionFromCallable(nqoi=1, nvars=2, fun=value_function, bkd=bkd)

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[1.0, 1.0]])

        # Create optimizer without objective/bounds
        optimizer = ScipyDifferentialEvolutionOptimizer(maxiter=100, seed=42, tol=1e-6)
        assert not optimizer.is_bound()

        # Should raise RuntimeError if minimizing without binding
        with pytest.raises(RuntimeError):
            optimizer.minimize(init_guess)

        # Bind and optimize
        optimizer.bind(objective, bounds)
        assert optimizer.is_bound()
        result = optimizer.minimize(init_guess)

        # Check result
        assert result.success()
        expected_optima = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_copy(self, bkd) -> None:
        """Test copy() returns unbound optimizer with same options."""
        # Create objective
        def value_function(x):
            return bkd.stack([x[0] ** 2], axis=0)

        obj = FunctionFromCallable(nqoi=1, nvars=1, fun=value_function, bkd=bkd)
        bounds = bkd.array([[-5.0, 5.0]])

        # Create and bind optimizer
        optimizer = ScipyDifferentialEvolutionOptimizer(maxiter=500, seed=42, tol=1e-8)
        optimizer.bind(obj, bounds)
        assert optimizer.is_bound()

        # Copy should be unbound with same options
        copy_opt = optimizer.copy()
        assert not copy_opt.is_bound()
        assert copy_opt._maxiter == 500
        assert copy_opt._seed == 42

        # Copy can be bound and used independently
        copy_opt.bind(obj, bounds)
        result = copy_opt.minimize(bkd.asarray([[2.0]]))
        bkd.assert_allclose(result.optima(), bkd.array([[0.0]]), atol=1e-5)

    def test_backward_compatibility(self, bkd) -> None:
        """Test existing API with objective/bounds in constructor still works."""
        def value_function(x):
            return bkd.stack([x[0] ** 2 + x[1] ** 2], axis=0)

        objective = FunctionFromCallable(nqoi=1, nvars=2, fun=value_function, bkd=bkd)

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[1.0, 1.0]])

        # Create optimizer with objective/bounds in constructor (old API)
        optimizer = ScipyDifferentialEvolutionOptimizer(
            objective=objective, bounds=bounds, maxiter=100, seed=42, tol=1e-6
        )

        # Should be bound immediately
        assert optimizer.is_bound()

        # Should work without explicit bind()
        result = optimizer.minimize(init_guess)
        assert result.success()
        expected_optima = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-6)

    def test_invalid_objective_raises_typeerror(self, bkd) -> None:
        """Test that binding an invalid objective raises TypeError."""
        bounds = bkd.array([[-5.0, 5.0]])

        optimizer = ScipyDifferentialEvolutionOptimizer(maxiter=100)

        # Plain function is not a valid objective (missing bkd, nvars, nqoi)
        invalid_objective = lambda x: x**2  # noqa: E731

        with pytest.raises(TypeError) as context:
            optimizer.bind(invalid_objective, bounds)

        # Error message should mention missing methods
        assert "ObjectiveProtocol" in str(context.value)
