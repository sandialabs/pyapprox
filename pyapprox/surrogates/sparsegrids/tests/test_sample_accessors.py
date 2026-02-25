"""Tests for sample/value accessors on SampleTracker and adaptive fitters.

Verifies:
- SampleTracker filtered collection (Phase 1)
- MultiFidelity/SingleFidelity adaptive fitter accessors (Phase 2-3)
- IsotropicSparseGridFitter.get_values() (Phase 4)

All tests run on both NumPy and PyTorch backends.
"""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

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
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401

# =============================================================================
# SampleTracker filtered collection tests
# =============================================================================


class TestSampleTrackerFiltered(Generic[Array], unittest.TestCase):
    """Unit tests for SampleTracker filtered collection methods."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _make_tracker_with_subspaces(self, nvars: int = 2, level: int = 2):
        """Build a tracker with several registered subspaces."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * nvars
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(self._bkd, factories, growth)
        tracker = SampleTracker(self._bkd, tp_factory)

        # Register a few subspaces manually
        indices_list = []
        positions = []
        for i in range(level + 1):
            idx = self._bkd.zeros((nvars,), dtype=self._bkd.int64_dtype())
            idx = self._bkd.copy(idx)
            # set first dim to i
            idx_np = self._bkd.to_numpy(idx).copy()
            idx_np[0] = i
            idx = self._bkd.asarray(idx_np, dtype=self._bkd.int64_dtype())
            subspace = tp_factory(idx)
            pos = tracker.register(idx, subspace)
            indices_list.append(idx)
            positions.append(pos)

        return tracker, tp_factory, indices_list, positions

    def test_positions_none_matches_collect_all(self) -> None:
        """positions=None gives same result as collect_unique_samples()."""
        tracker, _, _, _ = self._make_tracker_with_subspaces()
        all_samples = tracker.collect_unique_samples()
        filtered_all = tracker.collect_filtered_unique_samples(None)
        self._bkd.assert_allclose(filtered_all, all_samples)

    def test_positions_none_count_matches(self) -> None:
        """n_filtered_unique_samples(None) == n_unique_samples()."""
        tracker, _, _, _ = self._make_tracker_with_subspaces()
        self.assertEqual(
            tracker.n_filtered_unique_samples(None),
            tracker.n_unique_samples(),
        )

    def test_empty_filter_returns_empty(self) -> None:
        """Empty positions set returns zero-column array."""
        tracker, _, _, _ = self._make_tracker_with_subspaces()
        empty = tracker.collect_filtered_unique_samples(set())
        self.assertEqual(empty.shape[1], 0)
        self.assertEqual(tracker.n_filtered_unique_samples(set()), 0)

    def test_filter_subset_gives_subset(self) -> None:
        """Filtering to subset of positions gives fewer samples."""
        tracker, _, _, positions = self._make_tracker_with_subspaces(level=2)
        # Filter to just the first subspace
        subset = {positions[0]}
        n_subset = tracker.n_filtered_unique_samples(subset)
        n_all = tracker.n_unique_samples()
        self.assertGreater(n_all, n_subset)
        self.assertGreater(n_subset, 0)

        filtered = tracker.collect_filtered_unique_samples(subset)
        self.assertEqual(filtered.shape[1], n_subset)

    def test_values_alignment_with_samples(self) -> None:
        """Values from filtered collection align with filtered samples."""
        tracker, tp_factory, indices_list, positions = (
            self._make_tracker_with_subspaces(nvars=1, level=2)
        )

        # Define a simple function: f(x) = x^2
        def func(samples: Array) -> Array:
            return self._bkd.reshape(samples[0] ** 2, (1, -1))

        # Add values for each subspace's unique samples
        for pos, idx in zip(positions, indices_list):
            new_samples = tracker.get_new_samples(pos, tracker._registered[pos])
            if new_samples is not None:
                tracker.append_new_values(func(new_samples))

        # Check all positions
        all_samples = tracker.collect_filtered_unique_samples(None)
        all_values = tracker.collect_filtered_unique_values(None)
        self.assertIsNotNone(all_values)
        expected_values = func(all_samples)
        self._bkd.assert_allclose(all_values, expected_values, rtol=1e-12)

        # Check a subset
        subset = {positions[0], positions[1]}
        sub_samples = tracker.collect_filtered_unique_samples(subset)
        sub_values = tracker.collect_filtered_unique_values(subset)
        self.assertIsNotNone(sub_values)
        expected_sub = func(sub_samples)
        self._bkd.assert_allclose(sub_values, expected_sub, rtol=1e-12)

    def test_values_none_when_no_values(self) -> None:
        """collect_filtered_unique_values returns None before values set."""
        tracker, _, _, _ = self._make_tracker_with_subspaces()
        result = tracker.collect_filtered_unique_values(None)
        self.assertIsNone(result)


class TestSampleTrackerFilteredNumpy(TestSampleTrackerFiltered[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSampleTrackerFilteredTorch(TestSampleTrackerFiltered[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Adaptive fitter accessor tests
# =============================================================================


class TestAdaptiveFitterAccessors(Generic[Array], unittest.TestCase):
    """Tests for get_samples/get_values/get_indices on adaptive fitters."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _make_sf_fitter_and_run(self, nsteps: int = 5):
        """Create a SF fitter, run a few steps, return fitter and func."""
        nvars = 2
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * nvars
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(self._bkd, factories, growth)
        admis = MaxLevelCriteria(max_level=5, pnorm=1.0, bkd=self._bkd)
        fitter = SingleFidelityAdaptiveSparseGridFitter(self._bkd, tp_factory, admis)

        def func(s: Array) -> Array:
            return self._bkd.reshape(s[0] ** 2 + 0.5 * s[1], (1, -1))

        for _ in range(nsteps):
            samples = fitter.step_samples()
            if samples is None:
                break
            fitter.step_values(func(samples))

        return fitter, func

    def test_all_equals_union_of_selected_and_candidate(self) -> None:
        """get_samples('all') == selected + candidate samples."""
        fitter, _ = self._make_sf_fitter_and_run()
        all_s = fitter.get_samples("all")
        sel_s = fitter.get_samples("selected")
        cand_s = fitter.get_samples("candidate")

        n_all = all_s.shape[1]
        n_sel = sel_s.shape[1]
        n_cand = cand_s.shape[1]
        self.assertEqual(n_all, n_sel + n_cand)

    def test_samples_values_alignment(self) -> None:
        """f(get_samples('selected')) == get_values('selected')."""
        fitter, func = self._make_sf_fitter_and_run()
        sel_samples = fitter.get_samples("selected")
        sel_values = fitter.get_values("selected")
        self.assertIsNotNone(sel_values)
        expected = func(sel_samples)
        self._bkd.assert_allclose(sel_values, expected, rtol=1e-12)

    def test_all_samples_values_alignment(self) -> None:
        """f(get_samples('all')) == get_values('all')."""
        fitter, func = self._make_sf_fitter_and_run()
        all_samples = fitter.get_samples("all")
        all_values = fitter.get_values("all")
        self.assertIsNotNone(all_values)
        expected = func(all_samples)
        self._bkd.assert_allclose(all_values, expected, rtol=1e-12)

    def test_selected_indices_matches_result(self) -> None:
        """get_selected_indices() matches result().indices."""
        fitter, _ = self._make_sf_fitter_and_run()
        sel_indices = fitter.get_selected_indices()
        result_indices = fitter.result().indices
        self._bkd.assert_allclose(sel_indices, result_indices)

    def test_candidate_indices_not_none(self) -> None:
        """get_candidate_indices() is not None when candidates exist."""
        fitter, _ = self._make_sf_fitter_and_run(nsteps=2)
        cand = fitter.get_candidate_indices()
        # After a couple of steps there should be candidates
        self.assertIsNotNone(cand)
        self.assertGreater(cand.shape[1], 0)

    def test_cumulative_cost_with_constant_model(self) -> None:
        """cumulative_cost with ConstantCostModel == total unique samples."""
        fitter, _ = self._make_sf_fitter_and_run()
        cost = fitter.cumulative_cost(ConstantCostModel())
        all_samples = fitter.get_samples("all")
        self.assertEqual(cost, float(all_samples.shape[1]))

    def test_cumulative_cost_default(self) -> None:
        """cumulative_cost() with no arg uses fitter's cost model."""
        fitter, _ = self._make_sf_fitter_and_run()
        cost = fitter.cumulative_cost()
        all_samples = fitter.get_samples("all")
        # Default is ConstantCostModel (unit cost)
        self.assertEqual(cost, float(all_samples.shape[1]))

    def test_nselected_ncandidates(self) -> None:
        """nselected() and ncandidates() return correct counts."""
        fitter, _ = self._make_sf_fitter_and_run(nsteps=3)
        sel_idx = fitter.get_selected_indices()
        cand_idx = fitter.get_candidate_indices()
        self.assertEqual(fitter.nselected(), sel_idx.shape[1])
        if cand_idx is not None:
            self.assertEqual(fitter.ncandidates(), cand_idx.shape[1])


class TestAdaptiveFitterAccessorsNumpy(TestAdaptiveFitterAccessors[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAdaptiveFitterAccessorsTorch(TestAdaptiveFitterAccessors[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# MF fitter accessor tests
# =============================================================================


class TestMFAdaptiveFitterAccessors(Generic[Array], unittest.TestCase):
    """Tests for MF fitter accessors returning Dict[ConfigIdx, Array]."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _make_mf_fitter_and_run(self, nsteps: int = 5):
        """Create a 1D physical + 1 config MF fitter, run a few steps."""
        from pyapprox.surrogates.affine.indices import (
            CompositeCriteria,
            Max1DLevelsCriteria,
        )
        from pyapprox.surrogates.sparsegrids import DictModelFactory

        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)]
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(self._bkd, factories, growth)

        max_1d = self._bkd.asarray([6, 1], dtype=self._bkd.int64_dtype())
        admis = CompositeCriteria(
            MaxLevelCriteria(max_level=6, pnorm=1.0, bkd=self._bkd),
            Max1DLevelsCriteria(max_1d, self._bkd),
        )

        fitter = MultiFidelityAdaptiveSparseGridFitter(
            self._bkd,
            tp_factory,
            admis,
            nconfig_vars=1,
        )

        def model_0(s: Array) -> Array:
            return self._bkd.reshape(self._bkd.cos(s[0] + 0.5), (1, -1))

        def model_1(s: Array) -> Array:
            return self._bkd.reshape(self._bkd.cos(s[0]), (1, -1))

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

    def test_get_samples_returns_dict(self) -> None:
        """get_samples returns dict for MF fitter."""
        fitter, _ = self._make_mf_fitter_and_run()
        samples = fitter.get_samples("all")
        self.assertIsInstance(samples, dict)
        for cfg, s in samples.items():
            self.assertIsInstance(cfg, tuple)
            self.assertEqual(s.shape[0], 1)  # 1D physical

    def test_mf_selected_candidate_partition(self) -> None:
        """Selected + candidate samples == all samples for each config."""
        fitter, _ = self._make_mf_fitter_and_run()
        all_s = fitter.get_samples("all")
        sel_s = fitter.get_samples("selected")
        cand_s = fitter.get_samples("candidate")

        for cfg in all_s:
            n_all = all_s[cfg].shape[1]
            n_sel = sel_s.get(cfg, self._bkd.zeros((1, 0))).shape[1]
            n_cand = cand_s.get(cfg, self._bkd.zeros((1, 0))).shape[1]
            self.assertEqual(n_all, n_sel + n_cand)

    def test_mf_cumulative_cost(self) -> None:
        """cumulative_cost with unit cost == total unique samples."""
        fitter, _ = self._make_mf_fitter_and_run()
        cost = fitter.cumulative_cost(ConstantCostModel())
        all_s = fitter.get_samples("all")
        total = sum(s.shape[1] for s in all_s.values())
        self.assertEqual(cost, float(total))


class TestMFAdaptiveFitterAccessorsNumpy(TestMFAdaptiveFitterAccessors[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMFAdaptiveFitterAccessorsTorch(TestMFAdaptiveFitterAccessors[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Isotropic fitter get_values tests
# =============================================================================


class TestIsotropicFitterGetValues(Generic[Array], unittest.TestCase):
    """Tests for IsotropicSparseGridFitter.get_values()."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_get_values_none_before_fit(self) -> None:
        """get_values() returns None before fit() is called."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * 2
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(self._bkd, factories, growth)
        fitter = IsotropicSparseGridFitter(self._bkd, tp_factory, level=2)
        self.assertIsNone(fitter.get_values())

    def test_get_values_after_fit(self) -> None:
        """get_values() returns correct values after fit()."""
        nvars = 2
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)] * nvars
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(self._bkd, factories, growth)
        fitter = IsotropicSparseGridFitter(self._bkd, tp_factory, level=2)

        def func(s: Array) -> Array:
            return self._bkd.reshape(s[0] ** 2 + s[1], (1, -1))

        samples = fitter.get_samples()
        values = func(samples)
        fitter.fit(values)

        got_values = fitter.get_values()
        self.assertIsNotNone(got_values)
        # Values should align with samples
        expected = func(samples)
        self._bkd.assert_allclose(got_values, expected, rtol=1e-12)


class TestIsotropicFitterGetValuesNumpy(TestIsotropicFitterGetValues[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIsotropicFitterGetValuesTorch(TestIsotropicFitterGetValues[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
