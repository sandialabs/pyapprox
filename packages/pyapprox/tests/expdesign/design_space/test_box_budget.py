"""Tests for BoxBudgetDesignSpace."""

import pytest
from pyapprox.expdesign.design_space import BoxBudgetDesignSpace
from pyapprox.expdesign.protocols import DesignSpaceProtocol
from pyapprox.optimization.minimize.constraints.protocols import (
    LinearConstraintProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestBoxBudgetDesignSpace:
    """Weights in [lower, upper] summing to a budget."""

    def test_satisfies_protocol(self, bkd: Backend[Array]) -> None:
        space = BoxBudgetDesignSpace(5, 2.0, bkd)
        assert isinstance(space, DesignSpaceProtocol)

    def test_bounds(self, bkd: Backend[Array]) -> None:
        space = BoxBudgetDesignSpace(3, 1.0, bkd, lower=0.1, upper=0.8)
        bkd.assert_allclose(
            space.bounds(), bkd.asarray([[0.1, 0.8], [0.1, 0.8], [0.1, 0.8]])
        )

    def test_initial_is_feasible(self, bkd: Backend[Array]) -> None:
        space = BoxBudgetDesignSpace(4, 3.0, bkd)
        initial = space.initial()
        assert initial.shape == (4, 1)
        bkd.assert_allclose(bkd.sum(initial, axis=0), bkd.asarray([3.0]))
        bounds = space.bounds()
        assert bkd.all_bool(initial[:, 0] >= bounds[:, 0])
        assert bkd.all_bool(initial[:, 0] <= bounds[:, 1])

    def test_constraint_is_budget(self, bkd: Backend[Array]) -> None:
        """The single constraint is sum(w) = budget."""
        space = BoxBudgetDesignSpace(4, 2.5, bkd)
        constraints = space.constraints()
        assert len(constraints) == 1
        constraint = constraints[0]
        assert isinstance(constraint, LinearConstraintProtocol)
        bkd.assert_allclose(constraint.A(), bkd.ones((1, 4)), rtol=1e-12)
        bkd.assert_allclose(constraint.lb(), bkd.asarray([2.5]), rtol=1e-12)
        bkd.assert_allclose(constraint.ub(), bkd.asarray([2.5]), rtol=1e-12)

    def test_accessors(self, bkd: Backend[Array]) -> None:
        space = BoxBudgetDesignSpace(6, 2.0, bkd, lower=0.05, upper=0.9)
        assert space.nvars() == 6
        assert space.budget() == 2.0
        assert space.lower() == 0.05
        assert space.upper() == 0.9

    @pytest.mark.parametrize("budget", [-0.1, 4.1])
    def test_rejects_infeasible_budget(
        self, bkd: Backend[Array], budget: float
    ) -> None:
        with pytest.raises(ValueError):
            BoxBudgetDesignSpace(4, budget, bkd)

    def test_rejects_empty_box(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError):
            BoxBudgetDesignSpace(4, 1.0, bkd, lower=0.5, upper=0.5)

    def test_rejects_no_weights(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError):
            BoxBudgetDesignSpace(0, 0.0, bkd)
