"""Tests for per-subspace statistics, the cache, and box evaluation.

These are the pieces candidate scoring is built from: a statistic per
subspace, memoized against the subspace, and summed with signs over a
backward box. The box sums must reproduce what building two surrogates
and differencing them would give.
"""

import gc
from typing import List, Tuple

import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import LinearGrowthRule
from pyapprox.surrogates.sparsegrids import create_basis_factories
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.smolyak import (
    backward_box,
    compute_smolyak_coefficients,
    evaluate_box,
)
from pyapprox.surrogates.sparsegrids.statistics.cache import (
    SubspaceCache,
    box_sum,
)
from pyapprox.surrogates.sparsegrids.statistics.subspace_moments import (
    subspace_mean,
    subspace_raw_moment,
    subspace_variance,
    variance_delta,
    variance_from_raw_moments,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)


def _make_factory(bkd, nvars: int = 2, basis_type: str = "leja"):
    marginals = [UniformMarginal(0.0, 1.0, bkd) for _ in range(nvars)]
    factories = create_basis_factories(marginals, bkd, basis_type)
    return TensorProductSubspaceFactory(
        bkd, factories, LinearGrowthRule(scale=1, shift=1)
    )


def _valued_subspace(bkd, factory, key, fun):
    """Build a subspace at ``key`` and set values from ``fun``."""
    idx = bkd.asarray(list(key), dtype=bkd.int64_dtype())
    subspace = factory(idx)
    subspace.set_values(fun(subspace.get_samples()))
    return subspace


def _quadratic(bkd):
    """f(x, y) = x^2 + y^2, a single QoI."""

    def fun(samples):
        return bkd.reshape(samples[0] ** 2 + samples[1] ** 2, (1, -1))

    return fun


def _two_qoi(bkd):
    """A two-QoI target, so shapes are exercised beyond nqoi=1."""

    def fun(samples):
        first = samples[0] ** 2 + samples[1]
        second = samples[0] * samples[1]
        return bkd.stack([first, second], axis=0)

    return fun


class TestSubspaceMoments:
    """The free functions agree with direct quadrature."""

    def test_mean_is_weighted_sum(self, bkd) -> None:
        factory = _make_factory(bkd)
        subspace = _valued_subspace(bkd, factory, (2, 2), _quadratic(bkd))
        values = subspace.get_values()
        weights = subspace.get_quadrature_weights()
        bkd.assert_allclose(
            subspace_mean(subspace), values @ weights, rtol=1e-12
        )

    def test_variance_matches_one_pass_form(self, bkd) -> None:
        """Two-pass and one-pass agree when cancellation is not an issue."""
        factory = _make_factory(bkd)
        subspace = _valued_subspace(bkd, factory, (2, 2), _quadratic(bkd))
        values = subspace.get_values()
        weights = subspace.get_quadrature_weights()
        one_pass = (values**2) @ weights - (values @ weights) ** 2
        bkd.assert_allclose(
            subspace_variance(subspace), one_pass, rtol=1e-8
        )

    def test_variance_is_stable_under_large_offset(self, bkd) -> None:
        """Adding 1e6 to f shifts the mean but must not move the variance.

        The one-pass form differences two numbers near 1e12 to recover a
        variance of order 1e-2, losing most of its significant digits.
        """
        factory = _make_factory(bkd)
        plain = _valued_subspace(bkd, factory, (2, 2), _quadratic(bkd))
        base_variance = subspace_variance(plain)

        offset_fun = _quadratic(bkd)

        def shifted(samples):
            return offset_fun(samples) + 1e6

        offset = _valued_subspace(bkd, factory, (2, 2), shifted)
        bkd.assert_allclose(
            subspace_variance(offset), base_variance, rtol=1e-8
        )

    def test_raw_moment_order_one_is_the_mean(self, bkd) -> None:
        factory = _make_factory(bkd)
        subspace = _valued_subspace(bkd, factory, (2, 2), _quadratic(bkd))
        bkd.assert_allclose(
            subspace_raw_moment(subspace, 1),
            subspace_mean(subspace),
            rtol=1e-12,
        )

    def test_raw_moment_order_two(self, bkd) -> None:
        factory = _make_factory(bkd)
        subspace = _valued_subspace(bkd, factory, (2, 2), _quadratic(bkd))
        values = subspace.get_values()
        weights = subspace.get_quadrature_weights()
        bkd.assert_allclose(
            subspace_raw_moment(subspace, 2),
            (values**2) @ weights,
            rtol=1e-12,
        )

    def test_multi_qoi_shapes(self, bkd) -> None:
        factory = _make_factory(bkd)
        subspace = _valued_subspace(bkd, factory, (2, 2), _two_qoi(bkd))
        assert subspace_mean(subspace).shape == (2,)
        assert subspace_variance(subspace).shape == (2,)
        assert subspace_raw_moment(subspace, 2).shape == (2,)

    def test_raises_without_values(self, bkd) -> None:
        factory = _make_factory(bkd)
        idx = bkd.asarray([1, 1], dtype=bkd.int64_dtype())
        subspace = factory(idx)
        with pytest.raises(ValueError, match="no values"):
            subspace_mean(subspace)

    def test_raises_on_bad_order(self, bkd) -> None:
        factory = _make_factory(bkd)
        subspace = _valued_subspace(bkd, factory, (1, 1), _quadratic(bkd))
        with pytest.raises(ValueError, match="at least 1"):
            subspace_raw_moment(subspace, 0)


class TestVarianceFormulas:
    """The two raw-moment formulas, and the shapes they require."""

    def test_variance_from_raw_moments(self, bkd) -> None:
        mean = bkd.asarray([2.0, -1.0])
        second = bkd.asarray([5.0, 4.0])
        bkd.assert_allclose(
            variance_from_raw_moments(mean, second),
            bkd.asarray([1.0, 3.0]),
            rtol=1e-12,
        )

    def test_variance_delta_equals_the_difference_it_avoids(
        self, bkd
    ) -> None:
        """Algebraically V_new - V_old, without forming either."""
        mean = bkd.asarray([2.0, -1.0])
        second = bkd.asarray([5.0, 4.0])
        delta_mean = bkd.asarray([0.25, -0.5])
        delta_second = bkd.asarray([1.5, 0.75])

        old = variance_from_raw_moments(mean, second)
        new = variance_from_raw_moments(
            mean + delta_mean, second + delta_second
        )
        bkd.assert_allclose(
            variance_delta(mean, delta_mean, delta_second),
            new - old,
            rtol=1e-12,
        )

    def test_variance_delta_is_stable_when_converged(self, bkd) -> None:
        """The direct difference loses digits; this form does not.

        With a mean near 1e6 and a change near 1e-9, V_new and V_old
        agree to about 15 digits, so subtracting them keeps almost none
        of the answer.
        """
        mean = bkd.asarray([1.0e6])
        second = bkd.asarray([1.0e12 + 4.0])
        delta_mean = bkd.asarray([1.0e-9])
        delta_second = bkd.asarray([3.0e-3])

        got = bkd.to_float(
            variance_delta(mean, delta_mean, delta_second)[0]
        )
        # dM2 - dm(2m + dm) = 3e-3 - 1e-9*(2e6 + 1e-9) = 1e-3
        assert abs(got - 1.0e-3) < 1e-12

        # The form this avoids: V_new and V_old are both near 4, and
        # differencing them loses the 1e-3 answer to rounding.
        old = variance_from_raw_moments(mean, second)
        new = variance_from_raw_moments(
            mean + delta_mean, second + delta_second
        )
        differenced = bkd.to_float((new - old)[0])
        assert abs(differenced - 1.0e-3) > abs(got - 1.0e-3)

    @pytest.mark.parametrize(
        "bad", [(2, 1), (1, 2), (2, 2)]
    )
    def test_rejects_non_1d(self, bkd, bad) -> None:
        """A (nqoi, 1) would broadcast to (nqoi, nqoi) rather than fail."""
        with pytest.raises(ValueError, match="must be 1D"):
            variance_from_raw_moments(bkd.zeros(bad), bkd.zeros(bad))

    def test_rejects_mismatched_nqoi(self, bkd) -> None:
        with pytest.raises(ValueError, match="nqoi"):
            variance_from_raw_moments(
                bkd.zeros((2,)), bkd.zeros((3,))
            )

    def test_delta_rejects_mismatched_nqoi(self, bkd) -> None:
        with pytest.raises(ValueError, match="nqoi"):
            variance_delta(
                bkd.zeros((2,)), bkd.zeros((2,)), bkd.zeros((3,))
            )


class TestSubspaceCache:
    """Memoization is per subspace object and does not retain them."""

    def test_statistic_computed_once_per_subspace(self, bkd) -> None:
        factory = _make_factory(bkd)
        calls: List[int] = []

        def counting(subspace):
            calls.append(1)
            return subspace_mean(subspace)

        cache = SubspaceCache(counting)
        first = _valued_subspace(bkd, factory, (1, 0), _quadratic(bkd))
        second = _valued_subspace(bkd, factory, (0, 1), _quadratic(bkd))

        for _ in range(3):
            cache.get(first)
            cache.get(second)
        assert len(calls) == 2
        assert cache.nentries() == 2

    def test_equal_indices_do_not_collide(self, bkd) -> None:
        """Two subspaces with the same index stay separate entries.

        Different fidelities of one grid, or two grids entirely, can
        hold distinct subspaces carrying equal multi-indices.
        """
        factory = _make_factory(bkd)
        cache = SubspaceCache(subspace_mean)

        first = _valued_subspace(bkd, factory, (2, 2), _quadratic(bkd))

        def offset(samples):
            return _quadratic(bkd)(samples) + 5.0

        second = _valued_subspace(bkd, factory, (2, 2), offset)

        bkd.assert_allclose(
            cache.get(second) - cache.get(first),
            bkd.asarray([5.0]),
            rtol=1e-10,
        )
        assert cache.nentries() == 2

    def test_entries_released_after_collection(self, bkd) -> None:
        """A dropped subspace is not kept alive by the cache."""
        factory = _make_factory(bkd)
        cache = SubspaceCache(subspace_mean)
        subspace = _valued_subspace(bkd, factory, (1, 1), _quadratic(bkd))
        cache.get(subspace)
        assert cache.nentries() == 1
        del subspace
        gc.collect()
        assert cache.nentries() == 0

    def test_get_without_values_raises(self, bkd) -> None:
        factory = _make_factory(bkd)
        idx = bkd.asarray([1, 1], dtype=bkd.int64_dtype())
        cache = SubspaceCache(subspace_mean)
        with pytest.raises(ValueError, match="no values"):
            cache.get(factory(idx))


class TestBoxSum:
    """Signed sums over a backward box."""

    def test_matches_explicit_signed_sum(self, bkd) -> None:
        factory = _make_factory(bkd)
        fun = _quadratic(bkd)
        subspaces = {
            key: _valued_subspace(bkd, factory, key, fun)
            for key in [(0, 0), (1, 0), (0, 1), (1, 1)]
        }
        terms = [
            (sign, subspaces[key]) for sign, key in backward_box((1, 1))
        ]
        cache = SubspaceCache(subspace_mean)

        expected = bkd.zeros((1,))
        for sign, subspace in terms:
            expected = expected + sign * subspace_mean(subspace)
        bkd.assert_allclose(box_sum(cache, terms), expected, rtol=1e-12)

    def test_empty_box_raises(self, bkd) -> None:
        cache = SubspaceCache(subspace_mean)
        with pytest.raises(ValueError, match="empty box"):
            box_sum(cache, [])


class TestEvaluateBox:
    """Box evaluation reproduces a surrogate difference."""

    @pytest.mark.parametrize(
        "candidate", [(1, 1), (2, 0), (2, 1), (0, 2)]
    )
    def test_equals_surrogate_difference(
        self, bkd, candidate: Tuple[int, int]
    ) -> None:
        """sum_e (-1)^|e| I_{k-e}(x) equals I_{K+k}(x) - I_K(x).

        The two surrogates are built from compute_smolyak_coefficients,
        so this checks the box identity against the definition rather
        than against another incremental computation.
        """
        factory = _make_factory(bkd)
        fun = _quadratic(bkd)
        # A downward-closed set that admits every candidate above.
        selected = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
        selected = [k for k in selected if k != candidate]
        subspaces = {
            key: _valued_subspace(bkd, factory, key, fun)
            for key in set(selected) | {candidate}
        }

        def surrogate(keys):
            indices = bkd.asarray(
                [[k[d] for k in keys] for d in range(2)],
                dtype=bkd.int64_dtype(),
            )
            coefs = compute_smolyak_coefficients(indices, bkd)
            return CombinationSurrogate(
                bkd,
                2,
                [subspaces[k] for k in keys],
                coefs,
                1,
                indices=indices,
            )

        samples = bkd.asarray(
            [[0.1, 0.35, 0.6, 0.9], [0.2, 0.45, 0.75, 0.05]]
        )
        before = surrogate(selected)(samples)
        after = surrogate(selected + [candidate])(samples)

        terms = [
            (sign, subspaces[key])
            for sign, key in backward_box(candidate)
        ]
        bkd.assert_allclose(
            evaluate_box(terms, samples),
            after - before,
            rtol=1e-12,
            atol=1e-14,
        )

    def test_multi_qoi(self, bkd) -> None:
        factory = _make_factory(bkd)
        fun = _two_qoi(bkd)
        subspaces = {
            key: _valued_subspace(bkd, factory, key, fun)
            for key in [(0, 0), (1, 0), (0, 1), (1, 1)]
        }
        terms = [
            (sign, subspaces[key]) for sign, key in backward_box((1, 1))
        ]
        samples = bkd.asarray([[0.3, 0.7], [0.4, 0.8]])
        assert evaluate_box(terms, samples).shape == (2, 2)

    def test_empty_box_raises(self, bkd) -> None:
        samples = bkd.asarray([[0.5], [0.5]])
        with pytest.raises(ValueError, match="empty box"):
            evaluate_box([], samples)
