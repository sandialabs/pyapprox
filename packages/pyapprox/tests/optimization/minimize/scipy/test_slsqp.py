"""Regression tests for ScipySLSQPOptimizer against the public
bind/minimize API, using legacy-style producers. Written BEFORE the
Derivatives-bundle consumer rewrite so the rewrite is protected."""

from typing import Generic, List

import numpy as np
import pytest

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.optimization.minimize.scipy.slsqp import (
    ScipySLSQPOptimizer,
    _convert_constraints_for_slsqp,
)
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.optimizer_fixtures import (
    QuadraticNoDerivatives,
    QuadraticWithJacobian,
    SumConstraint,
    SumConstraintWithJacobian,
)


class TestScipySLSQPOptimizer:
    def test_converges_with_analytic_jacobian(self, bkd):
        objective = QuadraticWithJacobian(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipySLSQPOptimizer(ftol=1e-12).bind(objective, bounds)
        result = optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [-0.5]]), atol=1e-6
        )
        assert objective.njacobian_calls > 0

    def test_converges_without_jacobian(self, bkd):
        objective = QuadraticNoDerivatives(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipySLSQPOptimizer().bind(objective, bounds)
        result = optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [-0.5]]), atol=1e-5
        )

    def test_inequality_constraint_with_jacobian(self, bkd):
        # minimize sum(x^2) s.t. x0 + x1 >= 1 -> optimum (0.5, 0.5)
        objective = QuadraticWithJacobian(bkd, [0.0, 0.0])
        constraint = SumConstraintWithJacobian(bkd, 2, 1.0, float("inf"))
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipySLSQPOptimizer().bind(
            objective, bounds, [constraint]
        )
        result = optimizer.minimize(bkd.asarray([[2.0], [2.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[0.5], [0.5]]), atol=1e-6
        )
        assert constraint.njacobian_calls > 0

    def test_inequality_constraint_without_jacobian(self, bkd):
        objective = QuadraticWithJacobian(bkd, [0.0, 0.0])
        constraint = SumConstraint(bkd, 2, 1.0, float("inf"))
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipySLSQPOptimizer().bind(
            objective, bounds, [constraint]
        )
        result = optimizer.minimize(bkd.asarray([[2.0], [2.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[0.5], [0.5]]), atol=1e-5
        )

    def test_equality_constraint(self, bkd):
        # minimize sum(x^2) s.t. x0 + x1 == 2 -> optimum (1, 1)
        objective = QuadraticWithJacobian(bkd, [0.0, 0.0])
        constraint = SumConstraintWithJacobian(bkd, 2, 2.0, 2.0)
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipySLSQPOptimizer().bind(
            objective, bounds, [constraint]
        )
        result = optimizer.minimize(bkd.asarray([[2.0], [0.5]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [1.0]]), atol=1e-6
        )


class _SumAndDifference(Generic[Array]):
    """Rows ``x0 + x1`` and ``x0 - x1``, each with its own bounds.

    One function with two rows, computed together, as a constraint whose
    rows share work is; counts how often it is evaluated.
    """

    def __init__(self, bkd: Backend[Array], lb: List[float], ub: List[float]):
        self._bkd = bkd
        self._lb = bkd.asarray(lb)
        self._ub = bkd.asarray(ub)
        self.ncalls = 0
        self._derivs = Derivatives.first_order(jacobian=self.jacobian)

    def derivatives(self) -> Derivatives[Array]:
        return self._derivs

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 2

    def lb(self) -> Array:
        return self._lb

    def ub(self) -> Array:
        return self._ub

    def __call__(self, samples: Array) -> Array:
        self.ncalls += 1
        return self._bkd.stack(
            (samples[0] + samples[1], samples[0] - samples[1])
        )

    def jacobian(self, sample: Array) -> Array:
        return self._bkd.asarray([[1.0, 1.0], [1.0, -1.0]])


INF = float("inf")


class TestRowWiseBounds:
    """Every row's finite bound is held, whatever the other rows' are.

    Bounds used to be all or nothing across a constraint's rows: one row
    without an upper bound dropped every row's upper bound, silently.
    """

    def test_a_finite_upper_bound_beside_an_infinite_one_is_held(
        self, bkd: Backend[Array]
    ) -> None:
        # min |x - (3, 3)|^2 s.t. x0 + x1 <= 2 and x0 - x1 >= -10:
        # the first row is active, at (1, 1).
        constraint = _SumAndDifference(bkd, [-INF, -10.0], [2.0, INF])
        optimizer = ScipySLSQPOptimizer(ftol=1e-12).bind(
            QuadraticWithJacobian(bkd, [3.0, 3.0]),
            bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]]),
            [constraint],
        )
        result = optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [1.0]]), atol=1e-6
        )

    def test_an_equality_row_beside_an_inequality_row_is_held(
        self, bkd: Backend[Array]
    ) -> None:
        # min |x|^2 s.t. x0 + x1 == 2 and x0 - x1 >= 1: both active, at
        # (1.5, 0.5).
        constraint = _SumAndDifference(bkd, [2.0, 1.0], [2.0, INF])
        optimizer = ScipySLSQPOptimizer(ftol=1e-12).bind(
            QuadraticWithJacobian(bkd, [0.0, 0.0]),
            bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]]),
            [constraint],
        )
        result = optimizer.minimize(bkd.asarray([[2.0], [0.0]]))
        assert result.success()
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.5], [0.5]]), atol=1e-6
        )

    def test_one_constraint_is_one_evaluation_per_point(
        self, bkd: Backend[Array]
    ) -> None:
        """SciPy calls each entry separately, so one entry per constraint."""
        constraint = _SumAndDifference(bkd, [2.0, 1.0], [2.0, INF])
        entries = _convert_constraints_for_slsqp([constraint])
        assert len(entries) == 1
        values = entries[0]["fun"](np.array([1.5, 0.5]))
        assert constraint.ncalls == 1
        # Rows bounded below, then rows bounded above: the equality row
        # appears in both, held as two opposing inequalities.
        np.testing.assert_allclose(values, [0.0, 0.0, 0.0])

    def test_an_all_equality_constraint_stays_an_equality(
        self, bkd: Backend[Array]
    ) -> None:
        constraint = _SumAndDifference(bkd, [2.0, 0.0], [2.0, 0.0])
        (entry,) = _convert_constraints_for_slsqp([constraint])
        assert entry["type"] == "eq"

    def test_a_constraint_with_no_finite_bound_is_refused(
        self, bkd: Backend[Array]
    ) -> None:
        constraint = _SumAndDifference(bkd, [-INF, -INF], [INF, INF])
        with pytest.raises(ValueError, match="no finite bound"):
            _convert_constraints_for_slsqp([constraint])
