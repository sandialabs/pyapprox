"""Tests for ScipyDirectOptimizer against the public bind/minimize API."""

import pytest

from pyapprox.optimization.minimize.scipy.direct import ScipyDirectOptimizer
from pyapprox.util.backends.protocols import Array, Backend
from tests._helpers.optimizer_fixtures import (
    QuadraticNoDerivatives,
    SumConstraint,
)


class TestScipyDirectOptimizer:
    def test_finds_the_minimum_in_the_box(self, bkd: Backend[Array]) -> None:
        objective = QuadraticNoDerivatives(bkd, [1.0, -0.5])
        bounds = bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]])
        optimizer = ScipyDirectOptimizer(maxfun=4000).bind(objective, bounds)
        result = optimizer.minimize(bkd.asarray([[0.0], [0.0]]))
        bkd.assert_allclose(
            result.optima(), bkd.asarray([[1.0], [-0.5]]), atol=1e-2
        )

    def test_constraints_are_refused_not_ignored(
        self, bkd: Backend[Array]
    ) -> None:
        """Accepting a constraint it cannot hold would return an
        unconstrained optimum with nothing to say so."""
        constraint = SumConstraint(bkd, 2, 1.0, float("inf"))
        with pytest.raises(NotImplementedError, match="box bounds only"):
            ScipyDirectOptimizer().bind(
                QuadraticNoDerivatives(bkd, [0.0, 0.0]),
                bkd.asarray([[-5.0, 5.0], [-5.0, 5.0]]),
                [constraint],
            )
