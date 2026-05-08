"""Tests for sample/value accessors on SampleTracker and adaptive fitters.

Verifies:
- SampleTracker filtered collection (Phase 1)
- MultiFidelity/SingleFidelity adaptive fitter accessors (Phase 2-3)
- IsotropicSparseGridFitter.get_values() (Phase 4)

All tests run on both NumPy and PyTorch backends.
"""


from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    LinearGrowthRule,
    MaxLevelCriteria,
)
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    MultiFidelityAdaptiveSparseGridFitter,
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    GaussLagrangeFactory,
)
from pyapprox.surrogates.sparsegrids.cost_model import (
    ConstantCostModel,
)
from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.sample_tracker import (
    SampleTracker,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)

# =============================================================================
# SampleTracker filtered collection tests
# =============================================================================


class TestSampleTrackerFiltered:
    """Unit tests for SampleTracker filtered collection methods."""

    def _make_tracker_with_subspaces(self, bkd, nvars: int = 2, level: int = 2):
        """Build a tracker with several registered subspaces."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factories = [GaussLagrangeFactory(marginal, bkd)] * nvars
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        tracker = SampleTracker(bkd, tp_factory)

        # Register a few subspaces manually
        indices_list = []
        positions = []
        for i in range(level + 1):
            idx = bkd.zeros((nvars,), dtype=bkd.int64_dtype())
            idx = bkd.copy(idx)
            # set first dim to i
            idx_np = bkd.to_numpy(idx).copy()
            idx_np[0] = i
            idx = bkd.asarray(idx_np, dtype=bkd.int64_dtype())
            subspace = tp_factory(idx)
            pos = tracker.register(idx, subspace)
            indices_list.append(idx)
            positions.append(pos)

        return tracker, tp_factory, indices_list, positions

    def test_positions_none_matches_collect_all(self, bkd) -> None:
        """positions=None gives same result as collect_unique_samples()."""
        tracker, _, _, _ = self._make_tracker_with_subspaces(bkd)
        all_samples = tracker.collect_unique_samples()
        filtered_all = tracker.collect_filtered_unique_samples(None)
        bkd.assert_allclose(filtered_all, all_samples)

    def test_positions_none_count_matches(self, bkd) -> None:
        """n_filtered_unique_samples(None) == n_unique_samples()."""
        tracker, _, _, _ = self._make_tracker_with_subspaces(bkd)
        assert (
            tracker.n_filtered_unique_samples(None)
            == tracker.n_unique_samples()
        )

    def test_empty_filter_returns_empty(self, bkd) -> None:
        """Empty positions set returns zero-column array."""
        tracker, _, _, _ = self._make_tracker_with_subspaces(bkd)
        empty = tracker.collect_filtered_unique_samples(set())
        assert empty.shape[1] == 0
        assert tracker.n_filtered_unique_samples(set()) == 0

    def test_filter_subset_gives_subset(self, bkd) -> None:
        """Filtering to subset of positions gives fewer samples."""
        tracker, _, _, positions = self._make_tracker_with_subspaces(bkd, level=2)
        # Filter to just the first subspace
        subset = {positions[0]}
        n_subset = tracker.n_filtered_unique_samples(subset)
        n_all = tracker.n_unique_samples()
        assert n_all > n_subset
        assert n_subset > 0

        filtered = tracker.collect_filtered_unique_samples(subset)
        assert filtered.shape[1] == n_subset

    def test_values_alignment_with_samples(self, bkd) -> None:
        """Values from filtered collection align with filtered samples."""
        tracker, tp_factory, indices_list, positions = (
            self._make_tracker_with_subspaces(bkd, nvars=1, level=2)
        )

        # Define a simple function: f(x) = x^2
        def func(samples):
            return bkd.reshape(samples[0] ** 2, (1, -1))

        # Add values for each subspace's unique samples
        for pos, idx in zip(positions, indices_list):
            new_samples = tracker.get_new_samples(pos, tracker._registered[pos])
            if new_samples is not None:
                tracker.append_new_values(func(new_samples))

        # Check all positions
        all_samples = tracker.collect_filtered_unique_samples(None)
        all_values = tracker.collect_filtered_unique_values(None)
        assert all_values is not None
        expected_values = func(all_samples)
        bkd.assert_allclose(all_values, expected_values, rtol=1e-12)

        # Check a subset
        subset = {positions[0], positions[1]}
        sub_samples = tracker.collect_filtered_unique_samples(subset)
        sub_values = tracker.collect_filtered_unique_values(subset)
        assert sub_values is not None
        expected_sub = func(sub_samples)
        bkd.assert_allclose(sub_values, expected_sub, rtol=1e-12)

    def test_values_none_when_no_values(self, bkd) -> None:
        """collect_filtered_unique_values returns None before values set."""
        tracker, _, _, _ = self._make_tracker_with_subspaces(bkd)
        result = tracker.collect_filtered_unique_values(None)
        assert result is None


# =============================================================================
# Adaptive fitter accessor tests
# =============================================================================


class TestAdaptiveFitterAccessors:
    """Tests for get_samples/get_values/get_indices on adaptive fitters."""

    def _make_sf_fitter_and_run(self, bkd, nsteps: int = 5):
        """Create a SF fitter, run a few steps, return fitter and func."""
        nvars = 2
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factories = [GaussLagrangeFactory(marginal, bkd)] * nvars
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=5, pnorm=1.0, bkd=bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(bkd, tp_factory, admis)

        def func(s):
            return bkd.reshape(s[0] ** 2 + 0.5 * s[1], (1, -1))

        for _ in range(nsteps):
            samples = fitter.step_samples()
            if samples is None:
                break
            fitter.step_values(func(samples))

        return fitter, func

    def test_all_equals_union_of_selected_and_candidate(self, bkd) -> None:
        """get_samples('all') == selected + candidate samples."""
        fitter, _ = self._make_sf_fitter_and_run(bkd)
        all_s = fitter.get_samples("all")
        sel_s = fitter.get_samples("selected")
        cand_s = fitter.get_samples("candidate")

        n_all = all_s.shape[1]
        n_sel = sel_s.shape[1]
        n_cand = cand_s.shape[1]
        assert n_all == n_sel + n_cand

    def test_samples_values_alignment(self, bkd) -> None:
        """f(get_samples('selected')) == get_values('selected')."""
        fitter, func = self._make_sf_fitter_and_run(bkd)
        sel_samples = fitter.get_samples("selected")
        sel_values = fitter.get_values("selected")
        assert sel_values is not None
        expected = func(sel_samples)
        bkd.assert_allclose(sel_values, expected, rtol=1e-12)

    def test_all_samples_values_alignment(self, bkd) -> None:
        """f(get_samples('all')) == get_values('all')."""
        fitter, func = self._make_sf_fitter_and_run(bkd)
        all_samples = fitter.get_samples("all")
        all_values = fitter.get_values("all")
        assert all_values is not None
        expected = func(all_samples)
        bkd.assert_allclose(all_values, expected, rtol=1e-12)

    def test_selected_indices_matches_result(self, bkd) -> None:
        """get_selected_indices() matches result().indices."""
        fitter, _ = self._make_sf_fitter_and_run(bkd)
        sel_indices = fitter.get_selected_indices()
        result_indices = fitter.result().indices
        bkd.assert_allclose(sel_indices, result_indices)

    def test_candidate_indices_not_none(self, bkd) -> None:
        """get_candidate_indices() is not None when candidates exist."""
        fitter, _ = self._make_sf_fitter_and_run(bkd, nsteps=2)
        cand = fitter.get_candidate_indices()
        # After a couple of steps there should be candidates
        assert cand is not None
        assert cand.shape[1] > 0

    def test_cumulative_cost_with_constant_model(self, bkd) -> None:
        """cumulative_cost with ConstantCostModel == total unique samples."""
        fitter, _ = self._make_sf_fitter_and_run(bkd)
        cost = fitter.cumulative_cost(ConstantCostModel())
        all_samples = fitter.get_samples("all")
        assert cost == float(all_samples.shape[1])

    def test_cumulative_cost_default(self, bkd) -> None:
        """cumulative_cost() with no arg uses fitter's cost model."""
        fitter, _ = self._make_sf_fitter_and_run(bkd)
        cost = fitter.cumulative_cost()
        all_samples = fitter.get_samples("all")
        # Default is ConstantCostModel (unit cost)
        assert cost == float(all_samples.shape[1])

    def test_nselected_ncandidates(self, bkd) -> None:
        """nselected() and ncandidates() return correct counts."""
        fitter, _ = self._make_sf_fitter_and_run(bkd, nsteps=3)
        sel_idx = fitter.get_selected_indices()
        cand_idx = fitter.get_candidate_indices()
        assert fitter.nselected() == sel_idx.shape[1]
        if cand_idx is not None:
            assert fitter.ncandidates() == cand_idx.shape[1]


# =============================================================================
# MF fitter accessor tests
# =============================================================================


class TestMFAdaptiveFitterAccessors:
    """Tests for MF fitter accessors returning Dict[ConfigIdx, Array]."""

    def _make_mf_fitter_and_run(self, bkd, nsteps: int = 5):
        """Create a 1D physical + 1 config MF fitter, run a few steps."""
        from pyapprox.surrogates.affine.indices import (
            CompositeCriteria,
            Max1DLevelsCriteria,
        )
        from pyapprox.surrogates.sparsegrids import DictModelFactory

        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factories = [GaussLagrangeFactory(marginal, bkd)]
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)

        max_1d = bkd.asarray([6, 1], dtype=bkd.int64_dtype())
        admis = CompositeCriteria(
            MaxLevelCriteria(max_level=6, pnorm=1.0, bkd=bkd),
            Max1DLevelsCriteria(max_1d, bkd),
        )

        fitter = MultiFidelityAdaptiveSparseGridFitter(
            bkd,
            tp_factory,
            admis,
            nconfig_vars=1,
        )

        def model_0(s):
            return bkd.reshape(bkd.cos(s[0] + 0.5), (1, -1))

        def model_1(s):
            return bkd.reshape(bkd.cos(s[0]), (1, -1))

        model_factory = DictModelFactory({(0,): model_0, (1,): model_1})

        for _ in range(nsteps):
            samples = fitter.step_samples()
            if samples is None:
                break
            values = {
                cfg: model_factory.get_model(cfg)(s) for cfg, s in samples.items()
            }
            fitter.step_values(values)

        return fitter, model_factory

    def test_get_samples_returns_dict(self, bkd) -> None:
        """get_samples returns dict for MF fitter."""
        fitter, _ = self._make_mf_fitter_and_run(bkd)
        samples = fitter.get_samples("all")
        assert isinstance(samples, dict)
        for cfg, s in samples.items():
            assert isinstance(cfg, tuple)
            assert s.shape[0] == 1  # 1D physical

    def test_mf_selected_candidate_partition(self, bkd) -> None:
        """Selected + candidate samples == all samples for each config."""
        fitter, _ = self._make_mf_fitter_and_run(bkd)
        all_s = fitter.get_samples("all")
        sel_s = fitter.get_samples("selected")
        cand_s = fitter.get_samples("candidate")

        for cfg in all_s:
            n_all = all_s[cfg].shape[1]
            n_sel = sel_s.get(cfg, bkd.zeros((1, 0))).shape[1]
            n_cand = cand_s.get(cfg, bkd.zeros((1, 0))).shape[1]
            assert n_all == n_sel + n_cand

    def test_mf_cumulative_cost(self, bkd) -> None:
        """cumulative_cost with unit cost == total unique samples."""
        fitter, _ = self._make_mf_fitter_and_run(bkd)
        cost = fitter.cumulative_cost(ConstantCostModel())
        all_s = fitter.get_samples("all")
        total = sum(s.shape[1] for s in all_s.values())
        assert cost == float(total)


# =============================================================================
# Isotropic fitter get_values tests
# =============================================================================


class TestIsotropicFitterGetValues:
    """Tests for IsotropicSparseGridFitter.get_values()."""

    def test_get_values_none_before_fit(self, bkd) -> None:
        """get_values() returns None before fit() is called."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factories = [GaussLagrangeFactory(marginal, bkd)] * 2
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        fitter = IsotropicSparseGridFitter(bkd, tp_factory, level=2)
        assert fitter.get_values() is None

    def test_get_values_after_fit(self, bkd) -> None:
        """get_values() returns correct values after fit()."""
        nvars = 2
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factories = [GaussLagrangeFactory(marginal, bkd)] * nvars
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        fitter = IsotropicSparseGridFitter(bkd, tp_factory, level=2)

        def func(s):
            return bkd.reshape(s[0] ** 2 + s[1], (1, -1))

        samples = fitter.get_samples()
        values = func(samples)
        fitter.fit(values)

        got_values = fitter.get_values()
        assert got_values is not None
        # Values should align with samples
        expected = func(samples)
        bkd.assert_allclose(got_values, expected, rtol=1e-12)
