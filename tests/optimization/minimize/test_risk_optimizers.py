"""
Tests for minimax and AVaR optimizers.
"""

from pyapprox.optimization.minimize.avar import (
    AVaRConstraint,
    AVaRObjective,
    AVaROptimizer,
)
from pyapprox.optimization.minimize.minimax import (
    MinimaxConstraint,
    MinimaxObjective,
    MinimaxOptimizer,
)
from pyapprox.util.backends.protocols import Backend


class SimpleMultiQoI:
    """Simple multi-output objective for testing: f_i(x) = (x - c_i)^2."""

    def __init__(
        self,
        centers,
        bkd: Backend,
    ) -> None:
        self._centers = centers  # Shape: (nqoi, 1)
        self._bkd = bkd

    def bkd(self) -> Backend:
        return self._bkd

    def nvars(self) -> int:
        return 1

    def nqoi(self) -> int:
        return self._centers.shape[0]

    def __call__(self, sample):
        """Evaluate (x - c_i)^2 for all centers."""
        x = sample[0, 0]
        return (x - self._centers) ** 2

    def jacobian(self, sample):
        """Jacobian: 2(x - c_i)."""
        x = sample[0, 0]
        return 2 * (x - self._centers)


class TestMinimaxOptimizer:
    """Base class for minimax optimizer tests."""

    def test_minimax_symmetric_objectives(self, bkd) -> None:
        """Minimax of symmetric objectives should be at center."""
        # f_1 = (x-1)^2, f_2 = (x+1)^2
        # Minimax occurs at x=0 with t=1
        centers = bkd.asarray([[-1.0], [1.0]])
        model = SimpleMultiQoI(centers, bkd)
        bounds = bkd.asarray([[-3.0, 3.0]])

        optimizer = MinimaxOptimizer(model, bounds, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        x_opt = optimizer.extract_original_variables(optima)
        t_opt = optimizer.get_minimax_value(optima)

        assert result.success()
        assert bkd.allclose(x_opt, bkd.zeros((1, 1)), atol=1e-3)
        assert bkd.allclose(t_opt, bkd.ones((1, 1)), atol=1e-3)

    def test_minimax_asymmetric_objectives(self, bkd) -> None:
        """Minimax of asymmetric objectives."""
        # f_1 = (x-0)^2, f_2 = (x-2)^2
        # Minimax occurs at x=1 with t=1
        centers = bkd.asarray([[0.0], [2.0]])
        model = SimpleMultiQoI(centers, bkd)
        bounds = bkd.asarray([[-3.0, 5.0]])

        optimizer = MinimaxOptimizer(model, bounds, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        x_opt = optimizer.extract_original_variables(optima)
        t_opt = optimizer.get_minimax_value(optima)

        assert result.success()
        assert bkd.allclose(x_opt, bkd.ones((1, 1)), atol=1e-3)
        assert bkd.allclose(t_opt, bkd.ones((1, 1)), atol=1e-3)

    def test_minimax_three_objectives(self, bkd) -> None:
        """Minimax with three objectives."""
        # f_1 = (x+1)^2, f_2 = x^2, f_3 = (x-1)^2
        # At x=0: f = [1, 0, 1], max = 1
        centers = bkd.asarray([[-1.0], [0.0], [1.0]])
        model = SimpleMultiQoI(centers, bkd)
        bounds = bkd.asarray([[-2.0, 2.0]])

        optimizer = MinimaxOptimizer(model, bounds, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        x_opt = optimizer.extract_original_variables(optima)
        t_opt = optimizer.get_minimax_value(optima)

        assert result.success()
        assert bkd.allclose(x_opt, bkd.zeros((1, 1)), atol=1e-3)
        assert bkd.allclose(t_opt, bkd.ones((1, 1)), atol=1e-3)


class TestMinimaxObjective:
    """Base class for minimax objective tests."""

    def test_objective_value(self, bkd) -> None:
        """Objective should return t value."""
        obj = MinimaxObjective(nmodel_vars=2, bkd=bkd)

        # [t, x1, x2] = [5.0, 1.0, 2.0]
        sample = bkd.asarray([[5.0], [1.0], [2.0]])
        result = obj(sample)

        expected = bkd.asarray([[5.0]])
        assert bkd.allclose(result, expected)

    def test_objective_jacobian(self, bkd) -> None:
        """Jacobian should be [1, 0, 0, ...]."""
        obj = MinimaxObjective(nmodel_vars=2, bkd=bkd)

        sample = bkd.asarray([[5.0], [1.0], [2.0]])
        jac = obj.jacobian(sample)

        expected = bkd.asarray([[1.0, 0.0, 0.0]])
        assert bkd.allclose(jac, expected)


class TestMinimaxConstraint:
    """Base class for minimax constraint tests."""

    def _make_constraint(self, bkd):
        centers = bkd.asarray([[0.0], [1.0]])
        model = SimpleMultiQoI(centers, bkd)
        constraint = MinimaxConstraint(model)
        return constraint

    def test_constraint_values(self, bkd) -> None:
        """Constraint g_i = t - f_i(x) should be computed correctly."""
        constraint = self._make_constraint(bkd)
        # t=2, x=0: f=[0, 1], g=[2, 1]
        sample = bkd.asarray([[2.0], [0.0]])
        result = constraint(sample)

        expected = bkd.asarray([[2.0], [1.0]])
        assert bkd.allclose(result, expected)

    def test_constraint_jacobian(self, bkd) -> None:
        """Jacobian should be [1, -df/dx]."""
        constraint = self._make_constraint(bkd)
        sample = bkd.asarray([[2.0], [0.0]])
        jac = constraint.jacobian(sample)

        # At x=0: df/dx = [0, -2] (for centers [0, 1])
        # Jacobian: [1, 0], [1, 2]
        expected = bkd.asarray([[1.0, 0.0], [1.0, 2.0]])
        assert bkd.allclose(jac, expected)


class TestAVaROptimizer:
    """Base class for AVaR optimizer tests."""

    def test_avar_alpha_zero_equals_mean(self, bkd) -> None:
        """AVaR with alpha=0 should equal the mean."""
        centers = bkd.asarray([[-1.0], [0.0], [1.0]])
        model = SimpleMultiQoI(centers, bkd)
        bounds = bkd.asarray([[-2.0, 2.0]])

        optimizer = AVaROptimizer(model, bounds, alpha=0.0, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        avar_val = optimizer.get_avar_value(optima)

        # At x=0: f = [1, 0, 1], mean = 2/3
        expected_mean = bkd.asarray([[2.0 / 3.0]])
        assert result.success()
        assert bkd.allclose(avar_val, expected_mean, atol=1e-3)

    def test_avar_alpha_half(self, bkd) -> None:
        """AVaR with alpha=0.5 averages worst 50%."""
        centers = bkd.asarray([[-1.0], [0.0], [1.0]])
        model = SimpleMultiQoI(centers, bkd)
        bounds = bkd.asarray([[-2.0, 2.0]])

        optimizer = AVaROptimizer(model, bounds, alpha=0.5, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        x_opt = optimizer.extract_original_variables(optima)
        avar_val = optimizer.get_avar_value(optima)

        # At x=0: f = [1, 0, 1], sorted=[1, 1, 0], top 50% avg = 1
        assert result.success()
        assert bkd.allclose(x_opt, bkd.zeros((1, 1)), atol=1e-3)
        assert bkd.allclose(avar_val, bkd.ones((1, 1)), atol=1e-2)

    def test_avar_approaches_minimax_for_high_alpha(self, bkd) -> None:
        """AVaR with high alpha should approach minimax."""
        centers = bkd.asarray([[-1.0], [0.0], [1.0]])
        model = SimpleMultiQoI(centers, bkd)
        bounds = bkd.asarray([[-2.0, 2.0]])

        # High alpha (near 1)
        optimizer = AVaROptimizer(model, bounds, alpha=0.9, verbosity=0)
        result = optimizer.minimize()
        optima = result.optima()

        x_opt = optimizer.extract_original_variables(optima)
        avar_val = optimizer.get_avar_value(optima)

        # Should be close to minimax: x=0, value=1
        assert result.success()
        assert bkd.allclose(x_opt, bkd.zeros((1, 1)), atol=1e-2)
        assert bkd.allclose(avar_val, bkd.ones((1, 1)), atol=0.1)


class TestAVaRObjective:
    """Base class for AVaR objective tests."""

    def test_objective_value_alpha_half(self, bkd) -> None:
        """Test AVaR objective computation."""
        obj = AVaRObjective(nmodel_vars=1, nscenarios=3, alpha=0.5, bkd=bkd)

        # [t, s1, s2, s3, x] = [1.0, 0.5, 0.0, 0.5, 0.0]
        sample = bkd.asarray([[1.0], [0.5], [0.0], [0.5], [0.0]])
        result = obj(sample)

        # AVaR = t + (1/(n*(1-alpha))) * sum(s) = 1 + (1/(3*0.5)) * 1 = 1.667
        expected = bkd.asarray([[1.0 + 2.0 / 3.0]])
        assert bkd.allclose(result, expected)

    def test_objective_jacobian(self, bkd) -> None:
        """Test AVaR objective Jacobian."""
        obj = AVaRObjective(nmodel_vars=1, nscenarios=3, alpha=0.5, bkd=bkd)

        sample = bkd.asarray([[1.0], [0.5], [0.0], [0.5], [0.0]])
        jac = obj.jacobian(sample)

        # Jacobian: [1, c, c, c, 0] where c = 1/(3*0.5) = 2/3
        c = 2.0 / 3.0
        expected = bkd.asarray([[1.0, c, c, c, 0.0]])
        assert bkd.allclose(jac, expected)


class TestAVaRConstraint:
    """Base class for AVaR constraint tests."""

    def _make_constraint(self, bkd):
        centers = bkd.asarray([[-1.0], [0.0], [1.0]])
        model = SimpleMultiQoI(centers, bkd)
        return AVaRConstraint(model)

    def test_constraint_values(self, bkd) -> None:
        """Test constraint g_i = t + s_i - f_i(x)."""
        constraint = self._make_constraint(bkd)
        # At x=0: f = [1, 0, 1]
        # With t=1, s=[0, 0, 0]: g = [0, 1, 0]
        sample = bkd.asarray([[1.0], [0.0], [0.0], [0.0], [0.0]])
        result = constraint(sample)

        expected = bkd.asarray([[0.0], [1.0], [0.0]])
        assert bkd.allclose(result, expected)

    def test_constraint_jacobian_shape(self, bkd) -> None:
        """Test constraint Jacobian shape."""
        constraint = self._make_constraint(bkd)
        sample = bkd.asarray([[1.0], [0.0], [0.0], [0.0], [0.0]])
        jac = constraint.jacobian(sample)

        # Shape: (nqoi, nvars) = (3, 5)
        assert jac.shape == (3, 5)
