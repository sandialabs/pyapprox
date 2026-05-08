"""End-to-end tests for ROL with inexact gradients."""

import numpy as np
import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("pyrol"):
    pytest.skip("pyrol not installed", allow_module_level=True)

from pyapprox.risk import SampleAverageMean
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

    def jacobian_batch(self, samples):
        # samples: (nvars, K) -> return (K, nqoi, nvars) = (K, 1, 2)
        bkd = self._bkd
        z = samples[0, :]  # (K,)
        x = samples[1, :]  # (K,)
        col0 = 2.0 * z  # (K,)
        col1 = 2.0 * (x - 1.0)  # (K,)
        # Stack to (K, 2) then reshape to (K, 1, 2)
        jac_2d = bkd.stack([col0, col1], axis=1)  # (K, 2)
        return bkd.reshape(jac_2d, (samples.shape[1], 1, 2))


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
        _x1 = sample[1, 0]
        zero = 0.0 * z
        return self._bkd.asarray([
            [2.0 * z + zero, 1.0 + zero, zero],
            [1.0 + zero, zero, 1.0 + zero],
        ])

    def jacobian_batch(self, samples):
        # samples: (3, K) -> return (K, 2, 3)
        bkd = self._bkd
        K = samples.shape[1]
        z = samples[0, :]  # (K,)
        zeros = 0.0 * z
        ones = zeros + 1.0
        # Row 0: [2z, 1, 0], Row 1: [1, 0, 1]
        row0 = bkd.stack([2.0 * z, ones, zeros], axis=1)  # (K, 3)
        row1 = bkd.stack([ones, zeros, ones], axis=1)  # (K, 3)
        return bkd.stack([row0, row1], axis=1)  # (K, 2, 3)


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

            def jacobian_batch(self, samples):
                # (K, 1, 2) — constant jacobian [1, 1] for all samples
                bkd = self._bkd
                K = samples.shape[1]
                ones = bkd.ones((K, 2))
                return bkd.reshape(ones, (K, 1, 2))

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


class _ExpObjectiveModel:
    """f(z, x) = exp(z) + (x - 1)^2. z random on U(-1,1), x design. nqoi=1.

    E_z[f] = sinh(1) + (x - 1)^2 on U(-1,1). Minimizer: x* = 1.
    exp(z) is not polynomial, so it requires higher quadrature levels.
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
        return self._bkd.exp(z) + (x - 1.0) ** 2

    def jacobian(self, sample):
        z = sample[0, 0]
        x = sample[1, 0]
        return self._bkd.asarray([[self._bkd.exp(z), 2.0 * (x - 1.0) + 0.0 * z]])


def _make_tp_rule(bkd):
    """Create TP quadrature rule for 1D Uniform(-1,1)."""
    from pyapprox.probability.univariate import UniformMarginal
    from pyapprox.surrogates.affine.indices import LinearGrowthRule
    from pyapprox.surrogates.quadrature import gauss_quadrature_rule
    from pyapprox.surrogates.quadrature.tensor_product import (
        ParameterizedTensorProductQuadratureRule,
    )

    marginal = UniformMarginal(-1.0, 1.0, bkd)
    growth = LinearGrowthRule(scale=1, shift=1)

    def univar_rule(npts):
        return gauss_quadrature_rule(marginal, npts, bkd)

    return ParameterizedTensorProductQuadratureRule(
        bkd, [univar_rule], growth,
    )


def _make_sg_rule(bkd):
    """Create sparse grid quadrature rule for 1D Uniform(-1,1)."""
    from pyapprox.probability.univariate import UniformMarginal
    from pyapprox.surrogates.affine.indices import LinearGrowthRule
    from pyapprox.surrogates.sparsegrids import (
        GaussLagrangeFactory,
        ParameterizedIsotropicSparseGridQuadratureRule,
        TensorProductSubspaceFactory,
    )

    marginal = UniformMarginal(-1.0, 1.0, bkd)
    factories = [GaussLagrangeFactory(marginal, bkd)]
    growth = LinearGrowthRule(scale=1, shift=1)
    tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
    return ParameterizedIsotropicSparseGridQuadratureRule(bkd, tp_factory)


class TestROLInexactQuadrature:
    """Test ROL optimization with QuadratureStrategy (TP and SG)."""

    def _solve_with_strategy(self, bkd, model, strategy, inexact=True):
        """Solve optimization with a given strategy."""

        stat = SampleAverageMean(bkd)
        wrapper = InexactWrapper(
            model=model,
            stat=stat,
            strategy=strategy,
            design_indices=[1],
            bkd=bkd,
        )

        bounds = bkd.array([[-5.0, 5.0]])
        init_guess = bkd.asarray([[3.0]])

        if inexact:
            params = ROLOptimizer.inexact_gradient_parameters(tol_scaling=0.1)
        else:
            params = None

        optimizer = ROLOptimizer(
            objective=wrapper,
            bounds=bounds,
            verbosity=0,
            parameters=params,
        )
        return optimizer.minimize(init_guess)

    def _solve_exact_baseline(self, bkd, model, rule, level=5):
        """Solve with fixed high-level quadrature (non-inexact baseline)."""
        samples, weights = rule(level)
        strategy = FixedSampleStrategy(samples, weights, bkd)
        return self._solve_with_strategy(bkd, model, strategy, inexact=False)

    def _solve_inexact(self, bkd, model, rule, min_level=1, max_level=5):
        """Solve with QuadratureStrategy (inexact)."""
        from pyapprox.optimization.minimize.inexact.quadrature import (
            QuadratureStrategy,
        )

        strategy = QuadratureStrategy(
            rule, bkd, min_level=min_level, max_level=max_level,
        )
        return self._solve_with_strategy(bkd, model, strategy, inexact=True)

    def test_tp_quadrature_strategy_easy(self, bkd) -> None:
        """TP QuadratureStrategy solves easy (quadratic) problem correctly."""
        model = _QuadraticObjectiveModel(bkd)
        tp_rule = _make_tp_rule(bkd)

        # Exact baseline
        result_exact = self._solve_exact_baseline(bkd, model, tp_rule)
        assert result_exact.success()
        bkd.assert_allclose(
            result_exact.optima(), bkd.array([[1.0]]), atol=1e-4,
        )

        # Inexact
        result_inexact = self._solve_inexact(bkd, model, tp_rule)
        assert result_inexact.success()
        bkd.assert_allclose(
            result_inexact.optima(), bkd.array([[1.0]]), atol=1e-4,
        )

        # Agreement
        bkd.assert_allclose(
            result_inexact.optima(), result_exact.optima(), atol=1e-3,
        )

    def test_sg_quadrature_strategy_easy(self, bkd) -> None:
        """SG QuadratureStrategy solves easy (quadratic) problem correctly."""
        model = _QuadraticObjectiveModel(bkd)
        sg_rule = _make_sg_rule(bkd)

        # Exact baseline
        result_exact = self._solve_exact_baseline(bkd, model, sg_rule)
        assert result_exact.success()
        bkd.assert_allclose(
            result_exact.optima(), bkd.array([[1.0]]), atol=1e-4,
        )

        # Inexact
        result_inexact = self._solve_inexact(bkd, model, sg_rule)
        assert result_inexact.success()
        bkd.assert_allclose(
            result_inexact.optima(), bkd.array([[1.0]]), atol=1e-4,
        )

    def test_sg_vs_tp_same_result_easy(self, bkd) -> None:
        """SG and TP strategies produce same optimum on easy problem."""
        model = _QuadraticObjectiveModel(bkd)
        tp_rule = _make_tp_rule(bkd)
        sg_rule = _make_sg_rule(bkd)

        result_tp = self._solve_inexact(bkd, model, tp_rule)
        result_sg = self._solve_inexact(bkd, model, sg_rule)

        assert result_tp.success()
        assert result_sg.success()
        bkd.assert_allclose(
            result_tp.optima(), result_sg.optima(), atol=1e-3,
        )
        # Both match x*=1
        bkd.assert_allclose(
            result_tp.optima(), bkd.array([[1.0]]), atol=1e-4,
        )

    def test_tp_convergence_hard(self, bkd) -> None:
        """TP QuadratureStrategy converges on hard (exp) problem."""
        model = _ExpObjectiveModel(bkd)
        tp_rule = _make_tp_rule(bkd)

        # Exact baseline at high level
        result_exact = self._solve_exact_baseline(bkd, model, tp_rule, level=5)
        assert result_exact.success()
        _x_exact = float(bkd.to_numpy(result_exact.optima())[0, 0])

        # Convergence: collect errors at increasing max_level
        errors = []
        for max_level in [2, 3, 4, 5]:
            result = self._solve_inexact(
                bkd, model, tp_rule, min_level=1, max_level=max_level,
            )
            assert result.success()
            x_val = float(bkd.to_numpy(result.optima())[0, 0])
            errors.append(abs(x_val - 1.0))

        # Final level should be accurate
        assert errors[-1] < 1e-4
        # Error should decrease overall (final < initial)
        assert errors[-1] < 0.1 * errors[0] + 1e-10

    def test_sg_convergence_hard(self, bkd) -> None:
        """SG QuadratureStrategy converges on hard (exp) problem."""
        model = _ExpObjectiveModel(bkd)
        sg_rule = _make_sg_rule(bkd)

        # Exact baseline
        result_exact = self._solve_exact_baseline(bkd, model, sg_rule, level=5)
        assert result_exact.success()

        # Convergence at increasing max_level
        errors = []
        for max_level in [2, 3, 4, 5]:
            result = self._solve_inexact(
                bkd, model, sg_rule, min_level=1, max_level=max_level,
            )
            assert result.success()
            x_val = float(bkd.to_numpy(result.optima())[0, 0])
            errors.append(abs(x_val - 1.0))

        # Final level should be accurate
        assert errors[-1] < 1e-4
        # Error should decrease overall
        assert errors[-1] < 0.1 * errors[0] + 1e-10


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
