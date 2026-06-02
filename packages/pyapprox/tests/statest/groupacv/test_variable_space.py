"""Unit tests for variable_space module: wrappers, strategies, config."""

import numpy as np
import pytest
from pyapprox.statest.groupacv import GroupACVEstimatorIS
from pyapprox.statest.groupacv.optimization import (
    GroupACVCostConstraint,
    GroupACVTraceObjective,
)
from pyapprox.statest.groupacv.variable_space import (
    AllocationProblemConfig,
    ConstraintScaledSpace,
    EqualityBudget,
    FullCostSpace,
    IdentitySpace,
    InequalityBudget,
    _NormalizedConstraint,
    _RescaledConstraint,
    _RescaledObjective,
)
from pyapprox.statest.statistics import (
    MultiOutputMean,
    MultiOutputVariance,
)


def _make_estimator(bkd, nmodels=3, nqoi=1):
    """Create a simple IS estimator for testing."""
    np.random.seed(1)
    cov_size = nmodels * nqoi
    cov = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
    cov = cov.T @ cov
    costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    return GroupACVEstimatorIS(stat, costs)


def _make_objective_and_constraint(bkd, nmodels=3, target_cost=100.0):
    """Create a bound objective and constraint ready for wrapping."""
    est = _make_estimator(bkd, nmodels)
    obj = GroupACVTraceObjective(bkd)
    obj.set_estimator(est)
    con = GroupACVCostConstraint(bkd)
    con.set_estimator(est)
    con.set_budget(target_cost, min_nhf_samples=1)
    iterate = est._init_guess(target_cost)
    return est, obj, con, iterate


class TestRescaledObjective:
    """Tests for _RescaledObjective chain-rule transforms."""

    def test_value_unchanged(self, bkd):
        """Rescaled objective at m = n*scale equals original at n."""
        est, obj, _, iterate = _make_objective_and_constraint(bkd)
        partition_costs = bkd.einsum(
            "m,mp->p", est._costs, est._partitions_per_model
        )
        scale = partition_costs / bkd.min(partition_costs)
        wrapped = _RescaledObjective(obj, scale)
        m_iterate = iterate * scale[:, None]
        bkd.assert_allclose(wrapped(m_iterate), obj(iterate), rtol=1e-12)

    def test_jacobian_chain_rule(self, bkd):
        """Wrapped Jacobian equals J_n / scale (chain rule)."""
        est, obj, _, iterate = _make_objective_and_constraint(bkd)
        partition_costs = bkd.einsum(
            "m,mp->p", est._costs, est._partitions_per_model
        )
        scale = partition_costs / bkd.min(partition_costs)
        wrapped = _RescaledObjective(obj, scale)
        m_iterate = iterate * scale[:, None]
        J_wrapped = wrapped.jacobian(m_iterate)
        J_original = obj.jacobian(iterate)
        bkd.assert_allclose(J_wrapped, J_original / scale[None, :], rtol=1e-10)

    def test_hasattr_propagation(self, bkd):
        """Wrapped objective has jacobian/hessian/hvp iff inner does."""
        _, obj, _, _ = _make_objective_and_constraint(bkd)
        scale = bkd.array([1.0, 2.0, 3.0])
        wrapped = _RescaledObjective(obj, scale)
        assert hasattr(wrapped, "jacobian") == hasattr(obj, "jacobian")
        assert hasattr(wrapped, "hessian") == hasattr(obj, "hessian")
        assert hasattr(wrapped, "hvp") == hasattr(obj, "hvp")


class TestRescaledConstraint:
    """Tests for _RescaledConstraint chain-rule transforms."""

    def test_value_unchanged(self, bkd):
        """Rescaled constraint at m = n*scale equals original at n."""
        est, _, con, iterate = _make_objective_and_constraint(bkd)
        partition_costs = bkd.einsum(
            "m,mp->p", est._costs, est._partitions_per_model
        )
        scale = partition_costs / bkd.min(partition_costs)
        wrapped = _RescaledConstraint(con, scale)
        m_iterate = iterate * scale[:, None]
        bkd.assert_allclose(wrapped(m_iterate), con(iterate), rtol=1e-12)

    def test_jacobian_chain_rule(self, bkd):
        """Wrapped Jacobian equals J_n / scale."""
        est, _, con, iterate = _make_objective_and_constraint(bkd)
        partition_costs = bkd.einsum(
            "m,mp->p", est._costs, est._partitions_per_model
        )
        scale = partition_costs / bkd.min(partition_costs)
        wrapped = _RescaledConstraint(con, scale)
        m_iterate = iterate * scale[:, None]
        J_wrapped = wrapped.jacobian(m_iterate)
        J_original = con.jacobian(iterate)
        bkd.assert_allclose(J_wrapped, J_original / scale[None, :], rtol=1e-12)

    def test_lb_ub_passthrough(self, bkd):
        """lb and ub are passed through unchanged."""
        _, _, con, _ = _make_objective_and_constraint(bkd)
        scale = bkd.array([1.0, 2.0, 3.0])
        wrapped = _RescaledConstraint(con, scale)
        bkd.assert_allclose(wrapped.lb(), con.lb())
        bkd.assert_allclose(wrapped.ub(), con.ub())


class TestNormalizedConstraint:
    """Tests for _NormalizedConstraint normalization."""

    def test_value_scaled(self, bkd):
        """Output divided by normalization factors."""
        _, _, con, iterate = _make_objective_and_constraint(bkd)
        norm = bkd.array([100.0, 1.0])
        wrapped = _NormalizedConstraint(con, norm)
        bkd.assert_allclose(
            wrapped(iterate), con(iterate) / norm[:, None], rtol=1e-12
        )

    def test_lb_ub_scaled(self, bkd):
        """lb and ub divided by normalization factors."""
        _, _, con, _ = _make_objective_and_constraint(bkd)
        norm = bkd.array([100.0, 1.0])
        wrapped = _NormalizedConstraint(con, norm)
        bkd.assert_allclose(wrapped.lb(), con.lb() / norm, rtol=1e-12)
        bkd.assert_allclose(wrapped.ub(), con.ub() / norm, rtol=1e-12)

    def test_jacobian_scaled(self, bkd):
        """Jacobian divided by normalization factors."""
        _, _, con, iterate = _make_objective_and_constraint(bkd)
        norm = bkd.array([100.0, 1.0])
        wrapped = _NormalizedConstraint(con, norm)
        J_wrapped = wrapped.jacobian(iterate)
        J_original = con.jacobian(iterate)
        bkd.assert_allclose(J_wrapped, J_original / norm[:, None], rtol=1e-12)

    def test_positive_normalization_required(self, bkd):
        """Raises ValueError if normalization factors <= 0."""
        _, _, con, _ = _make_objective_and_constraint(bkd)
        with pytest.raises(ValueError, match="positive"):
            _NormalizedConstraint(con, bkd.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="positive"):
            _NormalizedConstraint(con, bkd.array([-1.0, 1.0]))

    def test_composition_with_rescaled(self, bkd):
        """Full composition: _RescaledConstraint -> _NormalizedConstraint."""
        est, _, con, iterate = _make_objective_and_constraint(bkd)
        partition_costs = bkd.einsum(
            "m,mp->p", est._costs, est._partitions_per_model
        )
        scale = partition_costs / bkd.min(partition_costs)
        norm = bkd.array([100.0, 1.0])
        rescaled = _RescaledConstraint(con, scale)
        composed = _NormalizedConstraint(rescaled, norm)

        m_iterate = iterate * scale[:, None]
        expected = con(iterate) / norm[:, None]
        bkd.assert_allclose(composed(m_iterate), expected, rtol=1e-12)


class TestIdentitySpace:
    """Tests for IdentitySpace — all transforms are identity."""

    def test_scale_ones(self, bkd):
        costs = bkd.array([3.0, 2.0, 1.0])
        space = IdentitySpace()
        scale = space.compute_scale(costs, bkd)
        bkd.assert_allclose(scale, bkd.ones((3,)))

    def test_roundtrip(self, bkd):
        space = IdentitySpace()
        costs = bkd.array([3.0, 2.0, 1.0])
        scale = space.compute_scale(costs, bkd)
        n = bkd.array([[10.0], [20.0], [30.0]])
        bkd.assert_allclose(
            space.transform_from_optimizer(
                space.transform_init_guess(n, scale)[:, 0], scale
            ),
            n[:, 0],
        )

    def test_wrap_passthrough(self, bkd):
        """wrap_objective and wrap_constraint return originals."""
        _, obj, con, _ = _make_objective_and_constraint(bkd)
        space = IdentitySpace()
        scale = bkd.ones((3,))
        assert space.wrap_objective(obj, scale) is obj
        assert space.wrap_constraint(con, scale) is con


class TestFullCostSpace:
    """Tests for FullCostSpace — cost-based rescaling."""

    def test_scale_computation(self, bkd):
        costs = bkd.array([100.0, 10.0, 1.0])
        space = FullCostSpace()
        scale = space.compute_scale(costs, bkd)
        bkd.assert_allclose(scale, bkd.array([100.0, 10.0, 1.0]))

    def test_roundtrip(self, bkd):
        space = FullCostSpace()
        costs = bkd.array([100.0, 10.0, 1.0])
        scale = space.compute_scale(costs, bkd)
        n = bkd.array([[5.0], [50.0], [500.0]])
        m = space.transform_init_guess(n, scale)
        n_back = space.transform_from_optimizer(m[:, 0], scale)
        bkd.assert_allclose(n_back, n[:, 0], rtol=1e-12)

    def test_bounds_uniform_upper(self, bkd):
        """Upper bounds become uniform target_cost/c_ref in m-space."""
        space = FullCostSpace()
        costs = bkd.array([100.0, 10.0, 1.0])
        scale = space.compute_scale(costs, bkd)
        target_cost = 1000.0
        n_bounds = bkd.array([
            [1e-8, target_cost / 100.0],
            [1e-8, target_cost / 10.0],
            [1e-8, target_cost / 1.0],
        ])
        m_bounds = space.transform_bounds(n_bounds, scale, bkd)
        uppers = m_bounds[:, 1]
        bkd.assert_allclose(
            uppers,
            bkd.full((3,), target_cost, dtype=bkd.double_dtype()),
            rtol=1e-12,
        )


class TestConstraintScaledSpace:
    """Tests for ConstraintScaledSpace — constraint normalization only."""

    def test_scale_ones(self, bkd):
        costs = bkd.array([3.0, 2.0, 1.0])
        space = ConstraintScaledSpace()
        scale = space.compute_scale(costs, bkd)
        bkd.assert_allclose(scale, bkd.ones((3,)))

    def test_wrap_constraint_normalizes(self, bkd):
        """Wrapped constraint has normalized output."""
        est, _, con, iterate = _make_objective_and_constraint(
            bkd, target_cost=200.0
        )
        space = ConstraintScaledSpace()
        scale = bkd.ones((est.npartitions(),))
        wrapped = space.wrap_constraint(con, scale)
        raw_val = con(iterate)
        wrapped_val = wrapped(iterate)
        bkd.assert_allclose(
            wrapped_val[0], raw_val[0] / 200.0, rtol=1e-12
        )
        bkd.assert_allclose(wrapped_val[1], raw_val[1], rtol=1e-12)


class TestBudgetConstraintForm:
    """Tests for InequalityBudget and EqualityBudget."""

    def test_inequality_resets(self, bkd):
        """InequalityBudget resets lb/ub to default inequality form."""
        _, _, con, _ = _make_objective_and_constraint(bkd)
        con._set_cost_equality()
        InequalityBudget().adjust_bounds(con)
        bkd.assert_allclose(con.lb(), bkd.zeros((2,)))
        bkd.assert_allclose(
            con.ub(), bkd.full((2,), float("inf"), dtype=bkd.double_dtype())
        )

    def test_equality_sets_cost_row(self, bkd):
        """EqualityBudget sets cost row to equality (ub=0 for budget slack)."""
        _, _, con, _ = _make_objective_and_constraint(bkd)
        EqualityBudget().adjust_bounds(con)
        bkd.assert_allclose(con.lb(), bkd.array([0.0, 0.0]))
        bkd.assert_allclose(con.ub(), bkd.array([0.0, float("inf")]))


class TestAllocationProblemConfig:
    """Tests for AllocationProblemConfig dataclass."""

    def test_default_config(self):
        config = AllocationProblemConfig()
        assert config.variable_scaling == "none"
        assert config.budget_constraint_form == "inequality"
        assert config.bounds_lb == "dead_threshold"

    def test_build_variable_space_identity(self):
        config = AllocationProblemConfig(variable_scaling="none")
        assert isinstance(config.build_variable_space(), IdentitySpace)

    def test_build_variable_space_full(self):
        config = AllocationProblemConfig(variable_scaling="full")
        assert isinstance(config.build_variable_space(), FullCostSpace)

    def test_build_budget_form_inequality(self):
        config = AllocationProblemConfig(budget_constraint_form="inequality")
        assert isinstance(config.build_budget_form(), InequalityBudget)

    def test_build_budget_form_equality(self):
        config = AllocationProblemConfig(budget_constraint_form="equality")
        assert isinstance(config.build_budget_form(), EqualityBudget)

    def test_resolve_bounds_lb_dead_threshold_mean(self, bkd):
        """Mean stat has threshold 0.0 → resolves to 1e-8."""
        stat = MultiOutputMean(1, bkd)
        config = AllocationProblemConfig(bounds_lb="dead_threshold")
        assert config.resolve_bounds_lb(stat) == pytest.approx(1e-8)

    def test_resolve_bounds_lb_dead_threshold_variance(self, bkd):
        """Variance stat has threshold 2.0 → resolves to 2.0."""
        np.random.seed(1)
        nmodels = 3
        cov = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        cov = cov.T @ cov
        W = bkd.array(np.random.normal(0, 1, (nmodels, nmodels)))
        W = W.T @ W
        stat = MultiOutputVariance(1, bkd)
        stat.set_pilot_quantities(cov, W)
        config = AllocationProblemConfig(bounds_lb="dead_threshold")
        assert config.resolve_bounds_lb(stat) == pytest.approx(2.0)

    def test_resolve_bounds_lb_explicit_float(self, bkd):
        """Explicit float is used directly."""
        stat = MultiOutputMean(1, bkd)
        config = AllocationProblemConfig(bounds_lb=0.5)
        assert config.resolve_bounds_lb(stat) == pytest.approx(0.5)

    def test_frozen(self):
        """Config is immutable."""
        config = AllocationProblemConfig()
        with pytest.raises(AttributeError):
            config.variable_scaling = "full"
