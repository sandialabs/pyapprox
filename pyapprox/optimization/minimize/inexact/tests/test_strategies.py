"""Dual-backend tests for inexact gradient strategies."""

import math

import numpy as np
import pytest

from pyapprox.optimization.minimize.inexact.fixed import (
    FixedSampleStrategy,
)
from pyapprox.optimization.minimize.inexact.monte_carlo import (
    MonteCarloSAAStrategy,
)
from pyapprox.optimization.minimize.inexact.protocols import (
    InexactGradientStrategyProtocol,
)
from pyapprox.optimization.minimize.inexact.qmc import (
    QMCSAAStrategy,
)
from pyapprox.optimization.minimize.inexact.quadrature import (
    QuadratureStrategy,
)


class TestFixedSampleStrategy:
    def test_protocol_satisfaction(self, bkd) -> None:
        samples = bkd.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weights = bkd.array([0.5, 0.3, 0.2])
        strategy = FixedSampleStrategy(samples, weights, bkd)
        assert isinstance(strategy, InexactGradientStrategyProtocol)

    def test_shapes(self, bkd) -> None:
        nvars, npts = 3, 10
        samples = bkd.ones((nvars, npts))
        weights = bkd.ones((npts,)) / npts
        strategy = FixedSampleStrategy(samples, weights, bkd)
        s, w = strategy.samples_and_weights(0.1)
        assert s.shape == (nvars, npts)
        assert w.shape == (npts,)

    def test_nvars(self, bkd) -> None:
        samples = bkd.ones((5, 20))
        weights = bkd.ones((20,)) / 20
        strategy = FixedSampleStrategy(samples, weights, bkd)
        assert strategy.nvars() == 5

    def test_ignores_tol(self, bkd) -> None:
        samples = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        weights = bkd.array([0.5, 0.5])
        strategy = FixedSampleStrategy(samples, weights, bkd)
        for tol in [1.0, 0.1, 0.001, 0.0, -1.0]:
            s, w = strategy.samples_and_weights(tol)
            bkd.assert_allclose(s, samples)
            bkd.assert_allclose(w, weights)

    def test_bkd(self, bkd) -> None:
        samples = bkd.ones((2, 3))
        weights = bkd.ones((3,)) / 3
        strategy = FixedSampleStrategy(samples, weights, bkd)
        assert strategy.bkd() is bkd


class TestMonteCarloSAAStrategy:
    def test_protocol_satisfaction(self, bkd) -> None:
        base = bkd.ones((2, 100))
        strategy = MonteCarloSAAStrategy(base, bkd)
        assert isinstance(strategy, InexactGradientStrategyProtocol)

    def test_shapes(self, bkd) -> None:
        nvars, n_max = 3, 50
        base = bkd.ones((nvars, n_max))
        strategy = MonteCarloSAAStrategy(base, bkd)
        s, w = strategy.samples_and_weights(0.5)
        assert s.shape[0] == nvars
        assert s.shape[1] >= 1
        assert w.shape[0] == s.shape[1]

    def test_nvars(self, bkd) -> None:
        base = bkd.ones((4, 100))
        strategy = MonteCarloSAAStrategy(base, bkd)
        assert strategy.nvars() == 4

    def test_tol_zero_returns_all(self, bkd) -> None:
        n_max = 50
        base = bkd.ones((2, n_max))
        strategy = MonteCarloSAAStrategy(base, bkd)
        s, w = strategy.samples_and_weights(0.0)
        assert s.shape[1] == n_max

    def test_tol_negative_returns_all(self, bkd) -> None:
        n_max = 50
        base = bkd.ones((2, n_max))
        strategy = MonteCarloSAAStrategy(base, bkd)
        s, w = strategy.samples_and_weights(-1.0)
        assert s.shape[1] == n_max

    def test_monotonicity(self, bkd) -> None:
        base = bkd.ones((2, 10000))
        strategy = MonteCarloSAAStrategy(base, bkd, scale_factor=1.0)
        tols = [1.0, 0.5, 0.1, 0.01]
        counts = []
        for tol in tols:
            s, _ = strategy.samples_and_weights(tol)
            counts.append(s.shape[1])
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1]

    def test_prefix_consistency(self, bkd) -> None:
        np.random.seed(42)
        base_np = np.random.randn(2, 100)
        base = bkd.array(base_np)
        strategy = MonteCarloSAAStrategy(base, bkd, scale_factor=1.0)
        s_large, _ = strategy.samples_and_weights(0.1)
        s_small, _ = strategy.samples_and_weights(0.5)
        n_small = s_small.shape[1]
        bkd.assert_allclose(s_small, s_large[:, :n_small])

    def test_weight_normalization(self, bkd) -> None:
        base = bkd.ones((2, 100))
        strategy = MonteCarloSAAStrategy(base, bkd)
        for tol in [1.0, 0.1, 0.01]:
            _, w = strategy.samples_and_weights(tol)
            bkd.assert_allclose(
                bkd.sum(w), bkd.array(1.0), rtol=1e-12
            )

    def test_nsamples_formula(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        n_max = 1000
        base = bkd.ones((2, n_max))
        scale = 4.0
        strategy = MonteCarloSAAStrategy(base, bkd, scale_factor=scale)
        tol = 0.1
        expected_n = min(n_max, math.ceil(scale / (tol * tol)))
        s, _ = strategy.samples_and_weights(tol)
        assert s.shape[1] == expected_n

    def test_cap_at_n_max(self, bkd) -> None:
        n_max = 10
        base = bkd.ones((2, n_max))
        strategy = MonteCarloSAAStrategy(base, bkd, scale_factor=1e6)
        s, _ = strategy.samples_and_weights(0.001)
        assert s.shape[1] == n_max

    def test_invalid_scale_factor(self, bkd) -> None:
        base = bkd.ones((2, 10))
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            MonteCarloSAAStrategy(base, bkd, scale_factor=0.0)
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            MonteCarloSAAStrategy(base, bkd, scale_factor=-1.0)

    def test_from_variance_bound(self, bkd) -> None:
        base = bkd.ones((2, 1000))
        var_bound = 4.0
        strategy = MonteCarloSAAStrategy.from_variance_bound(
            base, bkd, gradient_variance_bound=var_bound
        )
        tol = 0.5
        s, _ = strategy.samples_and_weights(tol)
        expected_n = min(1000, math.ceil(var_bound / (tol * tol)))
        assert s.shape[1] == expected_n

    def test_very_small_tol_hits_cap(self, bkd) -> None:
        n_max = 50
        base = bkd.ones((2, n_max))
        strategy = MonteCarloSAAStrategy(base, bkd)
        s, _ = strategy.samples_and_weights(1.5e-8)
        assert s.shape[1] == n_max


class TestQMCSAAStrategy:
    def test_protocol_satisfaction(self, bkd) -> None:
        base = bkd.ones((2, 100))
        strategy = QMCSAAStrategy(base, bkd)
        assert isinstance(strategy, InexactGradientStrategyProtocol)

    def test_shapes(self, bkd) -> None:
        nvars, n_max = 3, 50
        base = bkd.ones((nvars, n_max))
        strategy = QMCSAAStrategy(base, bkd)
        s, w = strategy.samples_and_weights(0.5)
        assert s.shape[0] == nvars
        assert s.shape[1] >= 1
        assert w.shape[0] == s.shape[1]

    def test_nvars(self, bkd) -> None:
        base = bkd.ones((4, 100))
        strategy = QMCSAAStrategy(base, bkd)
        assert strategy.nvars() == 4

    def test_tol_zero_returns_all(self, bkd) -> None:
        n_max = 50
        base = bkd.ones((2, n_max))
        strategy = QMCSAAStrategy(base, bkd)
        s, _ = strategy.samples_and_weights(0.0)
        assert s.shape[1] == n_max

    def test_tol_negative_returns_all(self, bkd) -> None:
        n_max = 50
        base = bkd.ones((2, n_max))
        strategy = QMCSAAStrategy(base, bkd)
        s, _ = strategy.samples_and_weights(-1.0)
        assert s.shape[1] == n_max

    def test_monotonicity(self, bkd) -> None:
        base = bkd.ones((2, 10000))
        strategy = QMCSAAStrategy(base, bkd, scale_factor=1.0)
        tols = [1.0, 0.5, 0.1, 0.01]
        counts = []
        for tol in tols:
            s, _ = strategy.samples_and_weights(tol)
            counts.append(s.shape[1])
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1]

    def test_prefix_consistency(self, bkd) -> None:
        np.random.seed(42)
        base_np = np.random.randn(2, 100)
        base = bkd.array(base_np)
        strategy = QMCSAAStrategy(base, bkd, scale_factor=1.0)
        s_large, _ = strategy.samples_and_weights(0.1)
        s_small, _ = strategy.samples_and_weights(0.5)
        n_small = s_small.shape[1]
        bkd.assert_allclose(s_small, s_large[:, :n_small])

    def test_weight_normalization(self, bkd) -> None:
        base = bkd.ones((2, 100))
        strategy = QMCSAAStrategy(base, bkd)
        for tol in [1.0, 0.1, 0.01]:
            _, w = strategy.samples_and_weights(tol)
            bkd.assert_allclose(
                bkd.sum(w), bkd.array(1.0), rtol=1e-12
            )

    def test_exponent_comparison(self, numpy_bkd) -> None:
        """QMC with exponent=1 uses fewer samples than MC-like exponent=2."""
        bkd = numpy_bkd
        n_max = 100000
        base = bkd.ones((2, n_max))
        qmc_strategy = QMCSAAStrategy(
            base, bkd, scale_factor=1.0, tol_exponent=1.0
        )
        mc_like_strategy = QMCSAAStrategy(
            base, bkd, scale_factor=1.0, tol_exponent=2.0
        )
        tol = 0.1
        s_qmc, _ = qmc_strategy.samples_and_weights(tol)
        s_mc, _ = mc_like_strategy.samples_and_weights(tol)
        assert s_qmc.shape[1] < s_mc.shape[1]

    def test_nsamples_formula(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        n_max = 10000
        base = bkd.ones((2, n_max))
        scale = 4.0
        exponent = 1.5
        strategy = QMCSAAStrategy(
            base, bkd, scale_factor=scale, tol_exponent=exponent
        )
        tol = 0.2
        expected_n = min(n_max, math.ceil(scale / (tol**exponent)))
        s, _ = strategy.samples_and_weights(tol)
        assert s.shape[1] == expected_n

    def test_invalid_scale_factor(self, bkd) -> None:
        base = bkd.ones((2, 10))
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            QMCSAAStrategy(base, bkd, scale_factor=-1.0)

    def test_invalid_tol_exponent(self, bkd) -> None:
        base = bkd.ones((2, 10))
        with pytest.raises(ValueError, match="tol_exponent must be positive"):
            QMCSAAStrategy(base, bkd, tol_exponent=0.0)

    def test_from_variance_bound(self, bkd) -> None:
        base = bkd.ones((2, 1000))
        var_bound = 4.0
        strategy = QMCSAAStrategy.from_variance_bound(
            base, bkd,
            gradient_variance_bound=var_bound,
            tol_exponent=1.5,
        )
        tol = 0.5
        s, _ = strategy.samples_and_weights(tol)
        expected_n = min(1000, math.ceil(var_bound / (tol**1.5)))
        assert s.shape[1] == expected_n

    def test_very_small_tol_hits_cap(self, bkd) -> None:
        n_max = 50
        base = bkd.ones((2, n_max))
        strategy = QMCSAAStrategy(base, bkd)
        s, _ = strategy.samples_and_weights(1.5e-8)
        assert s.shape[1] == n_max


class _MockQuadratureRule:
    """Mock parameterized quadrature rule for testing."""

    def __init__(self, nvars: int, bkd: object) -> None:
        self._nvars = nvars
        self._bkd = bkd
        self._call_count = 0

    def nvars(self) -> int:
        return self._nvars

    def bkd(self) -> object:
        return self._bkd

    def __call__(self, level: int):  # type: ignore[no-untyped-def]
        self._call_count += 1
        npts = 2**level
        samples = self._bkd.ones((self._nvars, npts)) * level
        weights = self._bkd.ones((npts,)) / npts
        return samples, weights


class TestQuadratureStrategy:
    def test_protocol_satisfaction(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        strategy = QuadratureStrategy(rule, bkd)
        assert isinstance(strategy, InexactGradientStrategyProtocol)

    def test_shapes(self, bkd) -> None:
        nvars = 3
        rule = _MockQuadratureRule(nvars, bkd)
        strategy = QuadratureStrategy(rule, bkd, min_level=1, max_level=5)
        s, w = strategy.samples_and_weights(0.1)
        assert s.shape[0] == nvars
        assert w.shape[0] == s.shape[1]

    def test_nvars(self, bkd) -> None:
        rule = _MockQuadratureRule(5, bkd)
        strategy = QuadratureStrategy(rule, bkd)
        assert strategy.nvars() == 5

    def test_tol_zero_returns_max_level(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        max_level = 5
        strategy = QuadratureStrategy(
            rule, bkd, min_level=1, max_level=max_level
        )
        s, _ = strategy.samples_and_weights(0.0)
        assert s.shape[1] == 2**max_level

    def test_tol_negative_returns_max_level(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        max_level = 5
        strategy = QuadratureStrategy(
            rule, bkd, min_level=1, max_level=max_level
        )
        s, _ = strategy.samples_and_weights(-1.0)
        assert s.shape[1] == 2**max_level

    def test_monotonicity(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        strategy = QuadratureStrategy(
            rule, bkd, min_level=1, max_level=10, rate=2.0
        )
        tols = [1.0, 0.5, 0.1, 0.01, 0.001]
        counts = []
        for tol in tols:
            s, _ = strategy.samples_and_weights(tol)
            counts.append(s.shape[1])
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1]

    def test_level_formula(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        rule = _MockQuadratureRule(2, bkd)
        rate = 2.0
        min_level, max_level = 1, 20
        strategy = QuadratureStrategy(
            rule, bkd, min_level=min_level, max_level=max_level, rate=rate
        )
        tol = 0.1
        expected_level = max(
            min_level,
            min(max_level, math.ceil(-math.log(tol) / math.log(rate))),
        )
        s, _ = strategy.samples_and_weights(tol)
        assert s.shape[1] == 2**expected_level

    def test_min_level_enforced(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        min_level = 3
        strategy = QuadratureStrategy(
            rule, bkd, min_level=min_level, max_level=10
        )
        # Large tol would give level < min_level without clamping
        s, _ = strategy.samples_and_weights(100.0)
        assert s.shape[1] == 2**min_level

    def test_max_level_enforced(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        max_level = 4
        strategy = QuadratureStrategy(
            rule, bkd, min_level=1, max_level=max_level
        )
        # Very small tol would give level > max_level without clamping
        s, _ = strategy.samples_and_weights(1e-20)
        assert s.shape[1] == 2**max_level

    def test_cache_avoids_redundant_calls(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        strategy = QuadratureStrategy(
            rule, bkd, min_level=1, max_level=10
        )
        tol = 0.1
        strategy.samples_and_weights(tol)
        count_after_first = rule._call_count
        strategy.samples_and_weights(tol)
        assert rule._call_count == count_after_first

    def test_cache_eviction(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        strategy = QuadratureStrategy(
            rule, bkd, min_level=1, max_level=10
        )
        # Fill cache with 3 different levels
        strategy.samples_and_weights(0.5)   # level ~1
        strategy.samples_and_weights(0.1)   # level ~4
        strategy.samples_and_weights(0.01)  # level ~7
        count_3 = rule._call_count
        assert count_3 == 3
        # Access a 4th level → evicts oldest (level ~1)
        strategy.samples_and_weights(0.001)  # level ~10
        assert rule._call_count == 4
        # Accessing evicted level requires recomputation
        strategy.samples_and_weights(0.5)
        assert rule._call_count == 5

    def test_cache_with_alternating_tols(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        strategy = QuadratureStrategy(
            rule, bkd, min_level=1, max_level=10
        )
        # Alternate between two tols
        tol_a, tol_b = 0.1, 0.01
        strategy.samples_and_weights(tol_a)
        strategy.samples_and_weights(tol_b)
        count = rule._call_count
        # Repeating should hit cache
        for _ in range(5):
            strategy.samples_and_weights(tol_a)
            strategy.samples_and_weights(tol_b)
        assert rule._call_count == count

    def test_weight_normalization(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        strategy = QuadratureStrategy(rule, bkd, min_level=1, max_level=5)
        for tol in [1.0, 0.1, 0.01]:
            _, w = strategy.samples_and_weights(tol)
            bkd.assert_allclose(
                bkd.sum(w), bkd.array(1.0), rtol=1e-12
            )

    def test_invalid_min_level(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        with pytest.raises(ValueError, match="min_level must be non-negative"):
            QuadratureStrategy(rule, bkd, min_level=-1)

    def test_invalid_max_level(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        with pytest.raises(ValueError, match="max_level.*must be >= min_level"):
            QuadratureStrategy(rule, bkd, min_level=5, max_level=3)

    def test_invalid_rate(self, bkd) -> None:
        rule = _MockQuadratureRule(2, bkd)
        with pytest.raises(ValueError, match="rate must be > 1.0"):
            QuadratureStrategy(rule, bkd, rate=1.0)
        with pytest.raises(ValueError, match="rate must be > 1.0"):
            QuadratureStrategy(rule, bkd, rate=0.5)
