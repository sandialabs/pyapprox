"""End-to-end tests for ROL with inexact gradients."""

import numpy as np
import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("pyrol"):
    pytest.skip("pyrol not installed", allow_module_level=True)

from pyapprox.expdesign.statistics import SampleAverageMean
from pyapprox.optimization.minimize.inexact.fixed import (
    FixedSampleStrategy,
)
from pyapprox.optimization.minimize.inexact.monte_carlo import (
    MonteCarloSAAStrategy,
)
from pyapprox.optimization.minimize.inexact.protocols import (
    InexactDifferentiable,
    InexactEvaluable,
)
from pyapprox.optimization.minimize.inexact.wrapper import (
    InexactWrapper,
)
from pyapprox.optimization.minimize.rol.rol_optimizer import (
    ROLOptimizer,
)


class _QuadraticObjectiveModel:
    """f(z, x) = z^2 + (x - 1)^2. z random, x design. nqoi=1.

    E_z[f] = E[z^2] + (x - 1)^2 = sigma^2 + (x - 1)^2
    Minimizer: x* = 1.
    """

    def __init__(self, bkd):
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def nvars(self):
        return 2

    def nqoi(self):
        return 1

    def __call__(self, samples):
        z = samples[0:1, :]
        x = samples[1:2, :]
        return z**2 + (x - 1.0) ** 2

    def jacobian(self, sample):
        z = sample[0, 0]
        x = sample[1, 0]
        zero = 0.0 * z
        return self._bkd.asarray([[2.0 * z + zero, 2.0 * (x - 1.0) + zero]])


class TestROLInexactObjective:
    """Test ROL optimization with InexactWrapper as objective."""

    def test_exact_via_fixed_strategy(self, bkd) -> None:
        """InexactWrapper + FixedSampleStrategy through ROL gives correct
        optimum (same as exact optimization)."""
        model = _QuadraticObjectiveModel(bkd)
        stat = SampleAverageMean(bkd)

        # 3-point quadrature for z
        quad_samples = bkd.asarray([[-1.0, 0.0, 1.0]])
        quad_weights = bkd.asarray([1.0 / 6, 2.0 / 3, 1.0 / 6])
        strategy = FixedSampleStrategy(quad_samples, quad_weights, bkd)

        wrapper = InexactWrapper(
            model=model,
            stat=stat,
            strategy=strategy,
            design_indices=[1],
            bkd=bkd,
        )

        assert isinstance(wrapper, InexactEvaluable)
        assert isinstance(wrapper, InexactDifferentiable)

        bounds = bkd.array([[-5.0, 5.0]])
        init_guess = bkd.asarray([[3.0]])

        optimizer = ROLOptimizer(
            objective=wrapper,
            bounds=bounds,
            verbosity=0,
        )
        result = optimizer.minimize(init_guess)

        assert result.success()
        bkd.assert_allclose(result.optima(), bkd.array([[1.0]]), atol=1e-5)

    def test_inexact_gradient_parameters(self, bkd) -> None:
        """ROL with inexact gradient parameters converges to correct optimum."""
        model = _QuadraticObjectiveModel(bkd)
        stat = SampleAverageMean(bkd)

        # Large MC sample set for SAA
        np.random.seed(42)
        base_np = np.random.randn(1, 5000)
        base_samples = bkd.array(base_np)
        strategy = MonteCarloSAAStrategy(
            base_samples, bkd, scale_factor=1.0,
        )

        wrapper = InexactWrapper(
            model=model,
            stat=stat,
            strategy=strategy,
            design_indices=[1],
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])
        init_guess = bkd.asarray([[3.0]])

        params = ROLOptimizer.inexact_gradient_parameters(tol_scaling=0.1)
        optimizer = ROLOptimizer(
            objective=wrapper,
            bounds=bounds,
            verbosity=0,
            parameters=params,
        )
        result = optimizer.minimize(init_guess)

        assert result.success()
        # x* = 1.0; allow tolerance for MC approximation
        bkd.assert_allclose(result.optima(), bkd.array([[1.0]]), atol=0.05)

    def test_default_params_still_works(self, bkd) -> None:
        """InexactWrapper with default (non-inexact) params still optimizes
        correctly — ROL passes tol ≈ 1.5e-8 so strategy returns max samples."""
        model = _QuadraticObjectiveModel(bkd)
        stat = SampleAverageMean(bkd)

        np.random.seed(42)
        base_np = np.random.randn(1, 1000)
        base_samples = bkd.array(base_np)
        strategy = MonteCarloSAAStrategy(
            base_samples, bkd, scale_factor=1.0,
        )

        wrapper = InexactWrapper(
            model=model,
            stat=stat,
            strategy=strategy,
            design_indices=[1],
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])
        init_guess = bkd.asarray([[3.0]])

        # Default parameters — no inexact gradient flag
        optimizer = ROLOptimizer(
            objective=wrapper,
            bounds=bounds,
            verbosity=0,
        )
        result = optimizer.minimize(init_guess)

        assert result.success()
        bkd.assert_allclose(result.optima(), bkd.array([[1.0]]), atol=0.05)


class _ConstrainedModel:
    """f(z, x1, x2) = [z^2 + x1, z + x2]. z random, (x1,x2) design. nqoi=2.

    E[f1] = sigma^2 + x1, E[f2] = x2.
    """

    def __init__(self, bkd):
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def nvars(self):
        return 3

    def nqoi(self):
        return 2

    def __call__(self, samples):
        z = samples[0:1, :]
        x1 = samples[1:2, :]
        x2 = samples[2:3, :]
        return self._bkd.concatenate([z**2 + x1, z + x2], axis=0)

    def jacobian(self, sample):
        z = sample[0, 0]
        x1 = sample[1, 0]
        zero = 0.0 * z
        return self._bkd.asarray([
            [2.0 * z + zero, 1.0 + zero, zero],
            [1.0 + zero, zero, 1.0 + zero],
        ])


class TestROLInexactConstraint:
    """Test ROL optimization with InexactWrapper as constraint."""

    def _make_quadratic_objective(self, bkd):
        from pyapprox.interface.functions.fromcallable.hessian import (
            FunctionWithJacobianAndHVPFromCallable,
        )

        return FunctionWithJacobianAndHVPFromCallable(
            nvars=2,
            fun=lambda x: bkd.stack([x[0] ** 2 + x[1] ** 2], axis=1),
            jacobian=lambda x: bkd.stack([2 * x[0], 2 * x[1]], axis=1),
            hvp=lambda x, v: bkd.stack([2 * v[0], 2 * v[1]], axis=0),
            bkd=bkd,
        )

    def test_inexact_constraint_fixed_strategy(self, bkd) -> None:
        """Optimize a simple quadratic objective with an inexact constraint
        using FixedSampleStrategy (exact quadrature)."""
        objective = self._make_quadratic_objective(bkd)

        # Constraint: E[z^2 + x1] >= 0.5 (lower bound on first QoI)
        #             E[z + x2] >= 0 (lower bound on second QoI)
        con_model = _ConstrainedModel(bkd)
        stat = SampleAverageMean(bkd)

        quad_samples = bkd.asarray([[-1.0, 0.0, 1.0]])
        quad_weights = bkd.asarray([1.0 / 6, 2.0 / 3, 1.0 / 6])
        strategy = FixedSampleStrategy(quad_samples, quad_weights, bkd)

        constraint = InexactWrapper(
            model=con_model,
            stat=stat,
            strategy=strategy,
            design_indices=[1, 2],
            bkd=bkd,
            constraint_lb=bkd.asarray([0.5, 0.0]),
            constraint_ub=bkd.asarray([float("inf"), float("inf")]),
        )

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[1.0], [1.0]])

        optimizer = ROLOptimizer(
            objective=objective,
            bounds=bounds,
            constraints=[constraint],
            verbosity=0,
        )
        result = optimizer.minimize(init_guess)

        assert result.success()

        # Verify constraint satisfied at optimum
        con_val = constraint(result.optima())
        lb = constraint.lb()
        assert float(bkd.to_numpy(con_val[0, 0])) >= float(
            bkd.to_numpy(lb[0])
        ) - 1e-6
        assert float(bkd.to_numpy(con_val[1, 0])) >= float(
            bkd.to_numpy(lb[1])
        ) - 1e-6

    def test_inexact_constraint_mc_strategy(self, bkd) -> None:
        """Inexact constraint with MonteCarloSAAStrategy converges correctly."""
        objective = self._make_quadratic_objective(bkd)

        con_model = _ConstrainedModel(bkd)
        stat = SampleAverageMean(bkd)

        np.random.seed(42)
        base_np = np.random.randn(1, 5000)
        base_samples = bkd.array(base_np)
        strategy = MonteCarloSAAStrategy(base_samples, bkd, scale_factor=1.0)

        constraint = InexactWrapper(
            model=con_model,
            stat=stat,
            strategy=strategy,
            design_indices=[1, 2],
            bkd=bkd,
            constraint_lb=bkd.asarray([0.5, 0.0]),
            constraint_ub=bkd.asarray([float("inf"), float("inf")]),
        )

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[1.0], [1.0]])

        optimizer = ROLOptimizer(
            objective=objective,
            bounds=bounds,
            constraints=[constraint],
            verbosity=0,
        )
        result = optimizer.minimize(init_guess)

        assert result.success()

        con_val = constraint(result.optima())
        lb = constraint.lb()
        assert float(bkd.to_numpy(con_val[0, 0])) >= float(
            bkd.to_numpy(lb[0])
        ) - 0.05
        assert float(bkd.to_numpy(con_val[1, 0])) >= float(
            bkd.to_numpy(lb[1])
        ) - 0.05

    def test_inexact_objective_and_constraint(self, bkd) -> None:
        """Both objective and constraint are InexactWrapper with MC strategy."""
        # Objective: E[z^2 + (x-1)^2] — minimizer x*=1 unconstrained
        obj_model = _QuadraticObjectiveModel(bkd)
        stat = SampleAverageMean(bkd)

        np.random.seed(42)
        obj_base = bkd.array(np.random.randn(1, 5000))
        obj_strategy = MonteCarloSAAStrategy(obj_base, bkd, scale_factor=1.0)

        objective = InexactWrapper(
            model=obj_model,
            stat=stat,
            strategy=obj_strategy,
            design_indices=[1],
            bkd=bkd,
        )

        # Constraint model: f(z, x) = [z + x]. z random, x design. nqoi=1.
        # E[f] = x. Constraint: x >= 2.0.
        # With this constraint, optimum moves from x*=1 to x*=2.
        class _ShiftConstraintModel:
            def __init__(self, b):
                self._bkd = b

            def bkd(self):
                return self._bkd

            def nvars(self):
                return 2

            def nqoi(self):
                return 1

            def __call__(self, samples):
                z = samples[0:1, :]
                x = samples[1:2, :]
                return z + x

            def jacobian(self, sample):
                z = sample[0, 0]
                zero = 0.0 * z
                return self._bkd.asarray([[1.0 + zero, 1.0 + zero]])

        con_model = _ShiftConstraintModel(bkd)
        np.random.seed(123)
        con_base = bkd.array(np.random.randn(1, 5000))
        con_strategy = MonteCarloSAAStrategy(con_base, bkd, scale_factor=1.0)

        constraint = InexactWrapper(
            model=con_model,
            stat=stat,
            strategy=con_strategy,
            design_indices=[1],
            bkd=bkd,
            constraint_lb=bkd.asarray([2.0]),
            constraint_ub=bkd.asarray([float("inf")]),
        )

        bounds = bkd.array([[-5.0, 5.0]])
        init_guess = bkd.asarray([[3.0]])

        optimizer = ROLOptimizer(
            objective=objective,
            bounds=bounds,
            constraints=[constraint],
            verbosity=0,
        )
        result = optimizer.minimize(init_guess)

        assert result.success()
        # Constrained optimum: x* = 2.0
        bkd.assert_allclose(result.optima(), bkd.array([[2.0]]), atol=0.1)

    def test_linear_and_inexact_nonlinear_constraints(self, bkd) -> None:
        """Mix linear constraint + inexact nonlinear constraint."""
        from pyapprox.optimization.minimize.constraints.linear import (
            PyApproxLinearConstraint,
        )

        objective = self._make_quadratic_objective(bkd)

        # Linear constraint: x1 + x2 == 1 (equality)
        A = bkd.asarray([[1.0, 1.0]])
        b = bkd.asarray([1.0])
        linear_con = PyApproxLinearConstraint(A, b, b, bkd)

        # Nonlinear inexact constraint: E[z^2 + x1] >= 0.5
        #   With fixed quadrature: E[z^2] = 1/3, so x1 >= 1/6
        con_model = _ConstrainedModel(bkd)
        stat = SampleAverageMean(bkd)

        quad_samples = bkd.asarray([[-1.0, 0.0, 1.0]])
        quad_weights = bkd.asarray([1.0 / 6, 2.0 / 3, 1.0 / 6])
        strategy = FixedSampleStrategy(quad_samples, quad_weights, bkd)

        nonlinear_con = InexactWrapper(
            model=con_model,
            stat=stat,
            strategy=strategy,
            design_indices=[1, 2],
            bkd=bkd,
            constraint_lb=bkd.asarray([0.5, -float("inf")]),
            constraint_ub=bkd.asarray([float("inf"), float("inf")]),
        )

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[0.5], [0.5]])

        optimizer = ROLOptimizer(
            objective=objective,
            bounds=bounds,
            constraints=[linear_con, nonlinear_con],
            verbosity=0,
        )
        result = optimizer.minimize(init_guess)

        assert result.success()

        # Linear constraint satisfied: x1 + x2 = 1
        lin_residual = A @ result.optima()[:, 0] - b
        bkd.assert_allclose(lin_residual, bkd.zeros(b.shape), atol=1e-5)

        # Nonlinear constraint satisfied: E[z^2] + x1 >= 0.5
        con_val = nonlinear_con(result.optima())
        assert float(bkd.to_numpy(con_val[0, 0])) >= 0.5 - 1e-5


class TestROLExistingTestsRegression:
    """Verify existing ROL tests still pass with modified wrappers."""

    def test_non_inexact_objective_unchanged(self, bkd) -> None:
        """Standard objective (no inexact methods) works as before."""
        from pyapprox.interface.functions.fromcallable.hessian import (
            FunctionWithJacobianAndHVPFromCallable,
        )

        function = FunctionWithJacobianAndHVPFromCallable(
            nvars=2,
            fun=lambda x: bkd.stack(
                [x[0] ** 2 + x[1] ** 2], axis=1
            ),
            jacobian=lambda x: bkd.stack([2 * x[0], 2 * x[1]], axis=1),
            hvp=lambda x, v: bkd.stack([2 * v[0], 2 * v[1]], axis=0),
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0], [-5.0, 5.0]])
        init_guess = bkd.asarray([[1.0], [1.0]])

        optimizer = ROLOptimizer(
            objective=function, bounds=bounds, verbosity=0,
        )
        result = optimizer.minimize(init_guess)

        assert result.success()
        expected = bkd.array([0.0, 0.0])[:, None]
        bkd.assert_allclose(result.optima(), expected, atol=1e-6)
