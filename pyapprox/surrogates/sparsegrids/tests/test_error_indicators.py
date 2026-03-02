"""Tests for error indicators used in adaptive sparse grid refinement.

Tests verify mathematical correctness of L2SurrogateDifferenceIndicator,
L2NewSamplesIndicator, VarianceChangeIndicator, and CostWeightedIndicator.

Tests run on both NumPy and PyTorch backends.
"""

from typing import Tuple

import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    LinearGrowthRule,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    GaussLagrangeFactory,
)
from pyapprox.surrogates.sparsegrids.candidate_info import (
    CandidateInfo,
)
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.error_indicators import (
    CostWeightedIndicator,
    L2NewSamplesIndicator,
    L2SurrogateDifferenceIndicator,
    VarianceChangeIndicator,
)
from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.sample_tracker import (
    SampleTracker,
)
from pyapprox.surrogates.sparsegrids.smolyak import (
    compute_smolyak_coefficients,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.util.test_utils import (
    slower_test,
)

# =============================================================================
# Helper: build CandidateInfo from a selected index set + candidate
# =============================================================================


def _build_candidate_info(
    bkd,
    nvars: int,
    selected_level: int,
    candidate_index_tuple: Tuple[int, ...],
    target_fn,
    nqoi: int = 1,
    subspace_cost: float = None,
):
    """Build a CandidateInfo for testing.

    Creates a sparse grid at selected_level, fits it, then constructs a
    CandidateInfo for the given candidate index.

    Parameters
    ----------
    bkd : Backend
        Backend.
    nvars : int
        Number of variables.
    selected_level : int
        Level for the "selected" isotropic sparse grid.
    candidate_index_tuple : tuple
        Multi-index for the candidate subspace.
    target_fn : callable
        Function samples -> values, shape (nqoi, nsamples).
    nqoi : int
        Number of QoIs.
    subspace_cost : float or None
        If not None, set subspace_cost on CandidateInfo.
    """
    marginal = UniformMarginal(-1.0, 1.0, bkd)
    factories = [GaussLagrangeFactory(marginal, bkd)] * nvars
    growth = LinearGrowthRule(scale=1, shift=1)
    tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)

    # Build selected surrogate
    fitter = IsotropicSparseGridFitter(bkd, tp_factory, selected_level)
    sel_samples = fitter.get_samples()
    sel_values = target_fn(sel_samples)
    sel_result = fitter.fit(sel_values)
    selected_surrogate = sel_result.surrogate
    sel_indices = sel_result.indices

    # Create candidate subspace
    candidate_index = bkd.asarray(list(candidate_index_tuple), dtype=bkd.int64_dtype())
    candidate_subspace = tp_factory(candidate_index)

    # Build tracker to get new samples
    tracker = SampleTracker(bkd, tp_factory)

    # Register all selected subspaces first
    for j in range(sel_indices.shape[1]):
        idx = sel_indices[:, j]
        sub = tp_factory(idx)
        tracker.register(idx, sub)

    # Register candidate
    pos = tracker.register(candidate_index, candidate_subspace)
    new_local_indices = tracker.get_unique_local_indices(pos)

    # Collect all samples and new samples
    all_samples = tracker.collect_unique_samples()
    cand_samples = candidate_subspace.get_samples()
    new_samples = cand_samples[:, new_local_indices]

    # Provide values for all samples
    all_values = target_fn(all_samples)
    tracker.append_new_values(all_values)
    tracker.distribute_values_to_subspaces()

    # Build sel+candidate surrogate
    combined_indices = bkd.hstack((sel_indices, bkd.reshape(candidate_index, (-1, 1))))
    combined_coefs = compute_smolyak_coefficients(combined_indices, bkd)

    # Collect all subspaces (selected + candidate)
    all_subspaces = list(selected_surrogate.subspaces()) + [candidate_subspace]

    sel_plus_surrogate = CombinationSurrogate(
        bkd,
        nvars,
        all_subspaces,
        combined_coefs,
        nqoi,
        indices=combined_indices,
    )

    return CandidateInfo(
        candidate_index=candidate_index,
        candidate_subspace=candidate_subspace,
        all_samples=all_samples,
        new_samples=new_samples,
        new_sample_local_indices=new_local_indices,
        selected_surrogate=selected_surrogate,
        sel_plus_candidate_surrogate=sel_plus_surrogate,
        subspace_cost=subspace_cost,
    )


# =============================================================================
# L2SurrogateDifferenceIndicator tests
# =============================================================================


class TestL2SurrogateDifference:
    """Tests for L2SurrogateDifferenceIndicator."""

    def test_zero_for_exactly_represented_function(self, bkd) -> None:
        """If sel+candidate exactly represents the function, and selected
        also exactly represents it, the L2 difference should be zero."""

        def target_fn(samples):
            # Linear: exactly captured at level 1
            return bkd.reshape(samples[0, :] + samples[1, :], (1, -1))

        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
        )
        indicator = L2SurrogateDifferenceIndicator(bkd)
        priority, error = indicator(info)

        # Both surrogates exactly represent the linear function,
        # so the difference should be near zero
        assert error < 1e-10

    @pytest.mark.slow_on("TorchBkd")
    def test_nonzero_for_underresolved_function(self, bkd) -> None:
        """L2 difference should be positive when selected cannot represent
        the function but sel+candidate improves it."""

        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**4, (1, -1))

        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
        )
        indicator = L2SurrogateDifferenceIndicator(bkd)
        priority, error = indicator(info)
        assert error > 0

    def test_priority_equals_error(self, bkd) -> None:
        """Without cost weighting, priority should equal error."""

        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**2 + y**2, (1, -1))

        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
        )
        indicator = L2SurrogateDifferenceIndicator(bkd)
        priority, error = indicator(info)
        bkd.assert_allclose(
            bkd.asarray([priority]),
            bkd.asarray([error]),
        )


# =============================================================================
# L2NewSamplesIndicator tests
# =============================================================================


class TestL2NewSamples:
    """Tests for L2NewSamplesIndicator."""

    def test_zero_for_exactly_represented_function(self, bkd) -> None:
        """L2 on new samples should be zero when function is already exact."""

        def target_fn(samples):
            return bkd.reshape(samples[0, :] + samples[1, :], (1, -1))

        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
        )
        indicator = L2NewSamplesIndicator(bkd)
        priority, error = indicator(info)
        assert error < 1e-10

    def test_nonzero_for_underresolved_function(self, bkd) -> None:
        """L2 on new samples should be positive for underresolved function."""

        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**4, (1, -1))

        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
        )
        indicator = L2NewSamplesIndicator(bkd)
        priority, error = indicator(info)
        assert error > 0


# =============================================================================
# VarianceChangeIndicator tests
# =============================================================================


class TestVarianceChange:
    """Tests for VarianceChangeIndicator."""

    def test_zero_for_constant(self, bkd) -> None:
        """Variance change should be zero for a constant function.

        Both selected and sel+candidate will have variance = 0.
        Tolerance is 1e-7 due to Smolyak combination floating-point error.
        """

        def target_fn(samples):
            return bkd.full((1, samples.shape[1]), 5.0)

        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
        )
        indicator = VarianceChangeIndicator(bkd)
        priority, error = indicator(info)
        assert error < 1e-6

    def test_zero_for_already_resolved_variance(self, bkd) -> None:
        """Variance change should be zero when selected already captures
        all variance (linear function, level >= 1).

        Tolerance is 1e-7 due to Smolyak combination floating-point error.
        """

        def target_fn(samples):
            return bkd.reshape(samples[0, :], (1, -1))

        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
        )
        indicator = VarianceChangeIndicator(bkd)
        priority, error = indicator(info)
        assert error < 1e-6

    def test_nonzero_for_underresolved_variance(self, bkd) -> None:
        """Variance change should be positive when selected does not
        capture all variance."""

        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**4, (1, -1))

        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
        )
        indicator = VarianceChangeIndicator(bkd)
        priority, error = indicator(info)
        assert error > 0

    def test_variance_of_linear_x_is_one_third(self, bkd) -> None:
        """After adding enough subspaces, variance of x should be 1/3.

        Start from level 0 (only (0,0) subspace, 1 point -> variance 0),
        add candidate (1,0) which introduces the linear basis in x.
        The sel+candidate surrogate should have variance = 1/3.
        """

        def target_fn(samples):
            return bkd.reshape(samples[0, :], (1, -1))

        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=0,
            candidate_index_tuple=(1, 0),
            target_fn=target_fn,
        )

        # Verify the variance of sel+candidate surrogate
        var_new = info.sel_plus_candidate_surrogate.variance()
        bkd.assert_allclose(var_new, bkd.asarray([1.0 / 3.0]), rtol=1e-10)

    def test_variance_of_x_squared_is_4_over_45(self, bkd) -> None:
        """Variance of x^2 on [-1,1]^2 should be 4/45.

        Build enough subspaces to exactly represent x^2 and compute
        its variance.
        """

        def target_fn(samples):
            return bkd.reshape(samples[0, :] ** 2, (1, -1))

        # Level 1 has {(0,0),(1,0),(0,1)} -- captures x up to degree 2
        # Add (2,0) to capture x^2 exactly
        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
        )

        # The sel+candidate should have the correct variance
        var_new = info.sel_plus_candidate_surrogate.variance()
        expected = 4.0 / 45.0
        bkd.assert_allclose(var_new, bkd.asarray([expected]), rtol=1e-10)


# =============================================================================
# CostWeightedIndicator tests
# =============================================================================


class TestCostWeighted:
    """Tests for CostWeightedIndicator."""

    @slower_test
    def test_cost_divides_priority(self, bkd) -> None:
        """Cost weighting divides priority by subspace_cost."""

        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**2 + y**2, (1, -1))

        cost = 10.0
        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
            subspace_cost=cost,
        )

        base = L2SurrogateDifferenceIndicator(bkd)
        weighted = CostWeightedIndicator(bkd, base)

        base_priority, base_error = base(info)
        w_priority, w_error = weighted(info)

        # Error should be unchanged
        bkd.assert_allclose(
            bkd.asarray([w_error]),
            bkd.asarray([base_error]),
        )

        # Priority should be divided by cost
        bkd.assert_allclose(
            bkd.asarray([w_priority]),
            bkd.asarray([base_priority / cost]),
            rtol=1e-12,
        )

    def test_no_cost_leaves_priority_unchanged(self, bkd) -> None:
        """When subspace_cost is None, priority is unchanged."""

        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**2 + y**2, (1, -1))

        info = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
            subspace_cost=None,
        )

        base = L2SurrogateDifferenceIndicator(bkd)
        weighted = CostWeightedIndicator(bkd, base)

        base_priority, _ = base(info)
        w_priority, _ = weighted(info)

        bkd.assert_allclose(
            bkd.asarray([w_priority]),
            bkd.asarray([base_priority]),
        )

    def test_higher_cost_lower_priority(self, bkd) -> None:
        """Higher cost should result in lower priority."""

        def target_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**4, (1, -1))

        info_low_cost = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
            subspace_cost=1.0,
        )
        info_high_cost = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=target_fn,
            subspace_cost=100.0,
        )

        base = L2SurrogateDifferenceIndicator(bkd)
        weighted = CostWeightedIndicator(bkd, base)

        p_low, _ = weighted(info_low_cost)
        p_high, _ = weighted(info_high_cost)

        assert p_low > p_high

    def test_higher_error_refined_first(self, bkd) -> None:
        """Candidate with higher error should have higher priority.

        Tests that with equal cost, the subspace with larger surrogate
        difference gets higher priority.
        """

        # Low-degree function: nearly exact at level 1
        def easy_fn(samples):
            return bkd.reshape(samples[0, :] + samples[1, :], (1, -1))

        # High-degree function: large error at level 1
        def hard_fn(samples):
            x, y = samples[0, :], samples[1, :]
            return bkd.reshape(x**4 + y**4, (1, -1))

        info_easy = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=easy_fn,
            subspace_cost=1.0,
        )
        info_hard = _build_candidate_info(
            bkd,
            nvars=2,
            selected_level=1,
            candidate_index_tuple=(2, 0),
            target_fn=hard_fn,
            subspace_cost=1.0,
        )

        indicator = L2SurrogateDifferenceIndicator(bkd)
        p_easy, _ = indicator(info_easy)
        p_hard, _ = indicator(info_hard)

        assert p_hard > p_easy
