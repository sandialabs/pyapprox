import numpy as np
import pytest

from pyapprox.benchmarks.functions.algebraic.evutushenko import (
    EvtushenkoNonLinearConstraint,
    EvtushenkoObjective,
)
from pyapprox.optimization.minimize.chained.chained_optimizer import (
    ChainedOptimizer,
)
from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.optimization.minimize.scipy.diffevol import (
    ScipyDifferentialEvolutionOptimizer,
)
from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)


class TestChainedOptimizer:

    def test_chained_optimizer_with_evtushenko_objective_and_constraints(
        self, bkd,
    ) -> None:
        """
        Test the ChainedOptimizer class with the Evtushenko objective,
        nonlinear constraints, and a linear constraint.
        """
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
            constraints=[nonlinear_constraint, linear_con],
            strategy="best1bin",
            maxiter=100,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            raise_on_failure=False,
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
        assert result.success()

        # Assert that the constraints are satisfied
        nonlinear_constraint_value = nonlinear_constraint(result.optima())
        assert (
            bkd.all_bool(nonlinear_constraint_value >= nonlinear_constraint.lb())
            and bkd.all_bool(nonlinear_constraint_value <= nonlinear_constraint.ub())
        )

        linear_constraint_value = bkd.ones((1, 3)) @ result.optima()
        assert linear_constraint_value[0, 0] == pytest.approx(1.0, abs=1e-8)

        # Assert that the result matches the expected optima
        expected_optima = bkd.array([0.0, 0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-5)

        # Assert that the objective value matches the expected value
        expected_fun = objective(expected_optima)
        bkd.assert_allclose(result.fun(), float(expected_fun[0, 0]), atol=1e-8)

    def test_deferred_binding(self, bkd) -> None:
        """Test ChainedOptimizer with deferred binding."""
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
            raise_on_failure=False,
        )

        local_optimizer = ScipyTrustConstrOptimizer(
            verbosity=0,
            maxiter=100,
            gtol=1e-15,
        )

        # Chain the unbound optimizers
        chained_optimizer = ChainedOptimizer(global_optimizer, local_optimizer)

        # Should not be bound
        assert not chained_optimizer.is_bound()

        # Should raise RuntimeError if minimizing without binding
        init_guess = bkd.asarray([[0.1], [0.7], [0.2]])
        with pytest.raises(RuntimeError):
            chained_optimizer.minimize(init_guess)

        # Bind global with truncated bounds and constraints
        global_optimizer.bind(
            objective, truncated_bounds, constraints=[nonlinear_constraint, linear_con]
        )
        # Bind local with constraints
        local_optimizer.bind(
            objective, bounds, constraints=[nonlinear_constraint, linear_con]
        )

        # Now should be bound
        assert chained_optimizer.is_bound()

        # Perform optimization
        result = chained_optimizer.minimize(init_guess)

        # Assert that the optimization was successful
        assert result.success()

        # Assert that the result matches the expected optima
        expected_optima = bkd.array([0.0, 0.0, 1.0])[:, None]
        bkd.assert_allclose(result.optima(), expected_optima, atol=1e-5)

    def test_copy(self, bkd) -> None:
        """Test copy() returns unbound ChainedOptimizer with same options."""
        # Create objective
        objective = EvtushenkoObjective(backend=bkd)
        bounds = bkd.array([[0, 2.0], [0, 2.0], [0, 2.0]])

        # Create and bind optimizers
        global_optimizer = ScipyDifferentialEvolutionOptimizer(maxiter=500, seed=42)
        local_optimizer = ScipyTrustConstrOptimizer(maxiter=200)

        chained_optimizer = ChainedOptimizer(global_optimizer, local_optimizer)
        chained_optimizer.bind(objective, bounds)

        assert chained_optimizer.is_bound()

        # Copy should be unbound with same options
        copy_opt = chained_optimizer.copy()
        assert not copy_opt.is_bound()

        # Underlying optimizers should also be copied
        assert not copy_opt._global_optimizer.is_bound()
        assert not copy_opt._local_optimizer.is_bound()

    def test_invalid_optimizer_raises_typeerror(self, bkd) -> None:
        """Test that passing an invalid optimizer raises TypeError."""
        global_optimizer = ScipyDifferentialEvolutionOptimizer(maxiter=100)

        # Plain object is not a valid optimizer
        invalid_optimizer = "not an optimizer"

        with pytest.raises(TypeError) as context:
            ChainedOptimizer(global_optimizer, invalid_optimizer)  # type: ignore

        assert "BindableOptimizerProtocol" in str(context.value)
