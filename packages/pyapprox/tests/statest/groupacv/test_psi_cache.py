"""Tests for the Psi cache shared by GroupACV objective derivatives."""

import numpy as np
import pytest
from pyapprox.interface.functions.autograd import WithAutogradJacobian
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.statest.groupacv import GroupACVEstimatorIS
from pyapprox.statest.groupacv.optimization import (
    GroupACVLogDetObjective,
    GroupACVTraceObjective,
)
from pyapprox.statest.statistics import MultiOutputMean


def _make_estimator(bkd, nmodels=3, nqoi=1, seed=1):
    np.random.seed(seed)
    cov_size = nmodels * nqoi
    cov = bkd.array(np.random.normal(0, 1, (cov_size, cov_size)))
    cov = cov.T @ cov
    costs = bkd.arange(nmodels, 0, -1, dtype=bkd.double_dtype())
    stat = MultiOutputMean(nqoi, bkd)
    stat.set_pilot_quantities(cov)
    return GroupACVEstimatorIS(stat, costs)


def _count_psi_builds(objective, monkeypatch):
    """Count how often Psi is actually recomputed rather than reused."""
    builds = []
    cls = type(objective)
    original = cls._compute_psi_and_derivs

    def counting(self, npartition_samples_1d):
        before = self._cache_val
        result = original(self, npartition_samples_1d)
        if result is not before:
            builds.append(1)
        return result

    monkeypatch.setattr(cls, "_compute_psi_and_derivs", counting)
    return builds


class TestPsiCacheReuse:
    """The cache exists so one iterate costs one Psi, not three."""

    @pytest.mark.parametrize(
        "objective_cls", [GroupACVTraceObjective, GroupACVLogDetObjective]
    )
    def test_one_build_serves_value_jacobian_and_hessian(
        self, bkd, objective_cls, monkeypatch
    ) -> None:
        """Asserted as a count, not a duration: a timing threshold would
        measure the machine as much as the cache."""
        est = _make_estimator(bkd)
        obj = objective_cls(bkd)
        obj.set_estimator(est)
        iterate = est._init_guess(200.0)
        builds = _count_psi_builds(obj, monkeypatch)
        obj(iterate)
        obj.derivatives().jacobian(iterate)
        obj.derivatives().hessian(iterate)
        assert sum(builds) == 1

    def test_a_second_iterate_rebuilds(self, bkd, monkeypatch) -> None:
        """Moving to a new point must not reuse the previous Psi."""
        est = _make_estimator(bkd)
        obj = GroupACVTraceObjective(bkd)
        obj.set_estimator(est)
        builds = _count_psi_builds(obj, monkeypatch)
        obj(est._init_guess(200.0))
        obj(est._init_guess(400.0))
        assert sum(builds) == 2

    def test_rebinding_the_estimator_clears_the_cache(self, bkd) -> None:
        """Psi depends on estimator internals the cache key omits.

        Reusing an objective across estimators would otherwise answer
        the second with the first estimator's Psi.
        """
        first = _make_estimator(bkd, seed=1)
        second = _make_estimator(bkd, seed=7)
        obj = GroupACVTraceObjective(bkd)
        obj.set_estimator(first)
        iterate = first._init_guess(200.0)
        obj(iterate)
        obj.set_estimator(second)
        rebound = obj(iterate)
        fresh_obj = GroupACVTraceObjective(bkd)
        fresh_obj.set_estimator(second)
        bkd.assert_allclose(rebound, fresh_obj(iterate), rtol=1e-12)


class TestPsiCacheAutogradSafety:
    """Cached results must never be served to a differentiated input.

    The cached arrays are expressions built from the array that produced
    them. Handing them to a different array of equal values leaves the
    new input with no path through the graph, so the derivative comes
    back zero with nothing to indicate anything went wrong.
    """

    def test_autograd_matches_analytical_on_a_warm_cache(
        self, torch_bkd
    ) -> None:
        est = _make_estimator(torch_bkd)
        obj = GroupACVLogDetObjective(torch_bkd)
        obj.set_estimator(est)
        iterate = est._init_guess(200.0)
        # Warm the cache with a detached array, exactly as an optimizer
        # asking for the analytical jacobian would.
        analytical = obj.derivatives().jacobian(iterate)
        composed = WithAutogradJacobian(obj, torch_bkd)
        torch_bkd.assert_allclose(
            composed.derivatives().jacobian(iterate),
            analytical,
            rtol=1e-8,
        )

    def test_composed_jacobian_is_nonzero(self, torch_bkd) -> None:
        """The failure this guards against is a zero gradient, which an
        accuracy check alone would not distinguish from a bad one."""
        est = _make_estimator(torch_bkd)
        obj = GroupACVLogDetObjective(torch_bkd)
        obj.set_estimator(est)
        iterate = est._init_guess(200.0)
        obj.derivatives().jacobian(iterate)
        composed = WithAutogradJacobian(obj, torch_bkd)
        jacobian = composed.derivatives().jacobian(iterate)
        assert float(torch_bkd.max(torch_bkd.abs(jacobian))) > 1e-8

    def test_derivative_checker_accepts_the_composed_jacobian(
        self, torch_bkd
    ) -> None:
        est = _make_estimator(torch_bkd)
        obj = GroupACVLogDetObjective(torch_bkd)
        obj.set_estimator(est)
        iterate = est._init_guess(200.0)
        obj.derivatives().jacobian(iterate)
        checker = DerivativeChecker(WithAutogradJacobian(obj, torch_bkd))
        errors = checker.check_derivatives(iterate, verbosity=0)
        assert float(checker.error_ratio(errors[0])) <= 1e-6

    def test_gradient_carrying_input_is_not_cached(
        self, torch_bkd
    ) -> None:
        """A differentiated input must also leave the cache untouched,
        so a later detached call is not served a graph-bound result."""
        est = _make_estimator(torch_bkd)
        obj = GroupACVLogDetObjective(torch_bkd)
        obj.set_estimator(est)
        iterate = est._init_guess(200.0)
        WithAutogradJacobian(obj, torch_bkd).derivatives().jacobian(iterate)
        assert obj._cache_key is None
