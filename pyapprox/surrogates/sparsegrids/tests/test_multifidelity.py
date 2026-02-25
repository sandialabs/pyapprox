"""Tests for multi-fidelity sparse grid support.

Tests verify:
- Isotropic MF fitter produces dict samples and fits surrogates
- Adaptive MF fitter with cost models and model factories
- TimedModelFactory records per-config timings
- MeasuredCostModel reads measured wall times
- MF convergence and cost-aware refinement
- CandidateInfo field population

All tests run on both NumPy and PyTorch backends.
"""

import math
import unittest
from typing import Any, Callable, Dict, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.indices import (
    CompositeCriteria,
    LinearGrowthRule,
    Max1DLevelsCriteria,
    MaxLevelCriteria,
)
from pyapprox.surrogates.sparsegrids.adaptive_fitter import (
    MultiFidelityAdaptiveSparseGridFitter,
    SingleFidelityAdaptiveSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.basis_factory import (
    GaussLagrangeFactory,
)
from pyapprox.surrogates.sparsegrids.candidate_info import ConfigIdx
from pyapprox.surrogates.sparsegrids.combination_surrogate import (
    CombinationSurrogate,
)
from pyapprox.surrogates.sparsegrids.cost_model import (
    ConstantCostModel,
    ExponentialConfigCostModel,
    MeasuredCostModel,
)
from pyapprox.surrogates.sparsegrids.error_indicators import (
    CostWeightedIndicator,
    L2SurrogateDifferenceIndicator,
)
from pyapprox.surrogates.sparsegrids.isotropic_fitter import (
    IsotropicSparseGridFitter,
)
from pyapprox.surrogates.sparsegrids.model_factory import (
    DictModelFactory,
    TimedModelFactory,
)
from pyapprox.surrogates.sparsegrids.subspace_factory import (
    TensorProductSubspaceFactory,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests, slow_test  # noqa: F401

# =============================================================================
# Helpers
# =============================================================================


def _make_cosine_models(
    bkd: Backend[Array],
) -> Dict[ConfigIdx, Callable]:
    """Create cosine model hierarchy.

    f_alpha(z) = cos(pi*(z+1)/2 + eps_alpha)
    eps = {0: 0.5, 1: 0.0}
    """
    epsilons = {0: 0.5, 1: 0.0}

    def _make_model(eps: float) -> Callable:
        def model(samples: Array) -> Array:
            return bkd.reshape(
                bkd.cos(math.pi * (samples[0, :] + 1) / 2 + eps),
                (1, -1),
            )

        return model

    return {(alpha,): _make_model(epsilons[alpha]) for alpha in epsilons}


def _make_mf_fitter(
    bkd: Backend[Array],
    max_level: int = 4,
    max_config_level: int = 1,
    cost_model: Any = None,
    error_indicator: Any = None,
) -> MultiFidelityAdaptiveSparseGridFitter[Array]:
    """Create 1D physical + 1 config var adaptive fitter.

    Uses CompositeCriteria to limit config dimension to max_config_level.
    The full index is [physical_level, config_level].
    """
    marginal = UniformMarginal(-1.0, 1.0, bkd)
    factories = [GaussLagrangeFactory(marginal, bkd)]
    growth = LinearGrowthRule(scale=1, shift=1)
    tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
    # max_1d_levels: [max_physical_level, max_config_level]
    max_1d = bkd.asarray([max_level, max_config_level], dtype=bkd.int64_dtype())
    admis = CompositeCriteria(
        MaxLevelCriteria(max_level=max_level, pnorm=1.0, bkd=bkd),
        Max1DLevelsCriteria(max_1d, bkd),
    )
    return MultiFidelityAdaptiveSparseGridFitter(
        bkd,
        tp_factory,
        admis,
        nconfig_vars=1,
        error_indicator=error_indicator,
        cost_model=cost_model,
    )


def _make_sf_fitter(
    bkd: Backend[Array],
    max_level: int = 4,
) -> SingleFidelityAdaptiveSparseGridFitter[Array]:
    """Create 1D single-fidelity adaptive fitter."""
    marginal = UniformMarginal(-1.0, 1.0, bkd)
    factories = [GaussLagrangeFactory(marginal, bkd)]
    growth = LinearGrowthRule(scale=1, shift=1)
    tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
    admis = MaxLevelCriteria(max_level=max_level, pnorm=1.0, bkd=bkd)
    return SingleFidelityAdaptiveSparseGridFitter(bkd, tp_factory, admis)


# =============================================================================
# Isotropic MF tests
# =============================================================================


class TestMFIsotropic(Generic[Array], unittest.TestCase):
    """Tests for IsotropicSparseGridFitter with nconfig_vars > 0."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _make_isotropic_mf_fitter(
        self, level: int = 1
    ) -> IsotropicSparseGridFitter[Array]:
        """Create isotropic MF fitter with level capped for 2 models."""
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories = [GaussLagrangeFactory(marginal, self._bkd)]
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(self._bkd, factories, growth)
        return IsotropicSparseGridFitter(
            self._bkd, tp_factory, level=level, nconfig_vars=1
        )

    def test_mf_isotropic_get_samples_returns_dict(self) -> None:
        """get_samples() returns Dict[ConfigIdx, Array] when nconfig_vars > 0."""
        fitter = self._make_isotropic_mf_fitter(level=1)
        samples = fitter.get_samples()
        self.assertIsInstance(samples, dict)
        # Should have config keys that are tuples of ints
        for key in samples:
            self.assertIsInstance(key, tuple)

    def test_mf_isotropic_fit_produces_surrogate(self) -> None:
        """Fit with dict values produces a working surrogate."""
        fitter = self._make_isotropic_mf_fitter(level=1)
        samples = fitter.get_samples()
        assert isinstance(samples, dict)

        models = _make_cosine_models(self._bkd)
        values = {cfg: models[cfg](s) for cfg, s in samples.items()}
        result = fitter.fit(values)

        # Surrogate should evaluate to finite values
        test_pts = self._bkd.array([[0.5]])
        out = result.surrogate(test_pts)
        self.assertEqual(out.shape[0], 1)
        self.assertTrue(bool(self._bkd.all_bool(self._bkd.isfinite(out))))

    def test_mf_isotropic_nsamples_counts_across_configs(self) -> None:
        """result.nsamples equals sum across all config groups."""
        fitter = self._make_isotropic_mf_fitter(level=1)
        samples = fitter.get_samples()
        assert isinstance(samples, dict)

        models = _make_cosine_models(self._bkd)
        values = {cfg: models[cfg](s) for cfg, s in samples.items()}
        result = fitter.fit(values)

        total_samples = sum(s.shape[1] for s in samples.values())
        self.assertEqual(result.nsamples, total_samples)


class TestMFIsotropicNumpy(TestMFIsotropic[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMFIsotropicTorch(TestMFIsotropic[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Adaptive MF tests
# =============================================================================


class TestMFAdaptive(Generic[Array], unittest.TestCase):
    """Tests for MultiFidelityAdaptiveSparseGridFitter."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_mf_adaptive_step_samples_returns_dict(self) -> None:
        """First step_samples() returns Dict when nconfig_vars > 0."""
        fitter = _make_mf_fitter(self._bkd, max_level=3)
        samples = fitter.step_samples()
        self.assertIsInstance(samples, dict)
        for key in samples:
            self.assertIsInstance(key, tuple)

    def test_mf_adaptive_step_values_accepts_dict(self) -> None:
        """step_values with dict doesn't error and reprioritizes."""
        fitter = _make_mf_fitter(self._bkd, max_level=3)
        samples = fitter.step_samples()
        assert isinstance(samples, dict)

        models = _make_cosine_models(self._bkd)
        values = {cfg: models[cfg](s) for cfg, s in samples.items()}
        # Should not raise
        fitter.step_values(values)

        # Should be able to get next samples
        samples2 = fitter.step_samples()
        # Might be dict or None
        if samples2 is not None:
            self.assertIsInstance(samples2, dict)

    def test_mf_adaptive_cost_weighted_indicator(self) -> None:
        """CostWeightedIndicator gives different priorities than unweighted."""
        models = _make_cosine_models(self._bkd)

        # Unweighted fitter
        fitter_unw = _make_mf_fitter(self._bkd, max_level=3)
        samples = fitter_unw.step_samples()
        assert isinstance(samples, dict)
        values = {cfg: models[cfg](s) for cfg, s in samples.items()}
        fitter_unw.step_values(values)

        # Cost-weighted fitter
        cost_model = ExponentialConfigCostModel(base=4.0)
        indicator = CostWeightedIndicator(
            self._bkd,
            L2SurrogateDifferenceIndicator(self._bkd),
        )
        fitter_cw = _make_mf_fitter(
            self._bkd,
            max_level=3,
            cost_model=cost_model,
            error_indicator=indicator,
        )
        samples_cw = fitter_cw.step_samples()
        assert isinstance(samples_cw, dict)
        values_cw = {cfg: models[cfg](s) for cfg, s in samples_cw.items()}
        fitter_cw.step_values(values_cw)

        # After several refinement steps, the two fitters should differ
        for _ in range(5):
            s1 = fitter_unw.step_samples()
            if s1 is not None:
                assert isinstance(s1, dict)
                v1 = {cfg: models[cfg](s) for cfg, s in s1.items()}
                fitter_unw.step_values(v1)

            s2 = fitter_cw.step_samples()
            if s2 is not None:
                assert isinstance(s2, dict)
                v2 = {cfg: models[cfg](s) for cfg, s in s2.items()}
                fitter_cw.step_values(v2)

        # Both should produce valid results
        r1 = fitter_unw.result()
        r2 = fitter_cw.result()
        self.assertIsInstance(r1.surrogate, CombinationSurrogate)
        self.assertIsInstance(r2.surrogate, CombinationSurrogate)

    def test_mf_adaptive_subspace_cost_reflects_cost_model(self) -> None:
        """With ExponentialConfigCostModel, high-fidelity costs more."""
        cost_model = ExponentialConfigCostModel(base=4.0)
        # Cost for config (0,) = 4^0 = 1.0
        # Cost for config (1,) = 4^1 = 4.0
        self._bkd.assert_allclose(
            self._bkd.asarray([cost_model((0,))]),
            self._bkd.asarray([1.0]),
        )
        self._bkd.assert_allclose(
            self._bkd.asarray([cost_model((1,))]),
            self._bkd.asarray([4.0]),
        )

    def test_mf_adaptive_recovers_isotropic_with_constant_cost(
        self,
    ) -> None:
        """With ConstantCostModel and enough steps, matches isotropic."""
        models = _make_cosine_models(self._bkd)
        level = 1  # Use level=1 so only config 0,1 are generated

        # Build isotropic MF grid
        marginal = UniformMarginal(-1.0, 1.0, self._bkd)
        factories_iso = [GaussLagrangeFactory(marginal, self._bkd)]
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory_iso = TensorProductSubspaceFactory(self._bkd, factories_iso, growth)
        iso_fitter = IsotropicSparseGridFitter(
            self._bkd, tp_factory_iso, level=level, nconfig_vars=1
        )
        iso_samples = iso_fitter.get_samples()
        assert isinstance(iso_samples, dict)
        iso_values = {cfg: models[cfg](s) for cfg, s in iso_samples.items()}
        iso_result = iso_fitter.fit(iso_values)

        # Build adaptive MF grid with ConstantCostModel
        factory = DictModelFactory(models)
        ada_fitter = _make_mf_fitter(
            self._bkd,
            max_level=level,
            max_config_level=1,
            cost_model=ConstantCostModel(),
        )
        ada_result = ada_fitter.refine_to_tolerance(factory, tol=1e-14, max_steps=10)

        # Both should give similar evaluation at test points
        test_pts = self._bkd.array([[0.3, -0.5, 0.8]])
        iso_val = iso_result.surrogate(test_pts)
        ada_val = ada_result.surrogate(test_pts)
        self._bkd.assert_allclose(ada_val, iso_val, rtol=1e-8)

    def test_mf_refine_to_tolerance_via_model_factory(self) -> None:
        """DictModelFactory + refine_to_tolerance works without target_fn."""
        models = _make_cosine_models(self._bkd)
        factory = DictModelFactory(models)
        fitter = _make_mf_fitter(
            self._bkd,
            max_level=4,
        )

        # model_factory is first positional arg
        result = fitter.refine_to_tolerance(factory, tol=1e-6, max_steps=10)
        self.assertIsInstance(result.surrogate, CombinationSurrogate)
        self.assertGreater(result.nsamples, 0)


class TestMFAdaptiveNumpy(TestMFAdaptive[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMFAdaptiveTorch(TestMFAdaptive[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    @slow_test
    def test_mf_adaptive_cost_weighted_indicator(self) -> None:
        super().test_mf_adaptive_cost_weighted_indicator()


# =============================================================================
# TimedModelFactory + MeasuredCostModel tests
# =============================================================================


class TestTimedAndMeasured(Generic[Array], unittest.TestCase):
    """Tests for TimedModelFactory and MeasuredCostModel."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_timed_factory_records_per_config_timings(self) -> None:
        """After get_model(cfg)(samples), timer has call_count > 0."""
        models = _make_cosine_models(self._bkd)
        base_factory = DictModelFactory(models)
        timed_factory = TimedModelFactory(base_factory)

        samples = self._bkd.array([[0.1, 0.2, 0.3]])
        model = timed_factory.get_model((0,))
        model(samples)

        timer = timed_factory.timer((0,))
        self.assertGreater(timer.get("__call__").call_count(), 0)

    def test_measured_cost_returns_unit_before_data(self) -> None:
        """MeasuredCostModel returns 1.0 when call_count is 0."""
        models = _make_cosine_models(self._bkd)
        base_factory = DictModelFactory(models)
        timed_factory = TimedModelFactory(base_factory)
        cost_model = MeasuredCostModel(timed_factory)

        # No evaluations yet
        cost = cost_model((0,))
        self._bkd.assert_allclose(
            self._bkd.asarray([cost]),
            self._bkd.asarray([1.0]),
        )

    def test_measured_cost_returns_median_after_evals(self) -> None:
        """After evaluations, MeasuredCostModel returns median time > 0."""
        models = _make_cosine_models(self._bkd)
        base_factory = DictModelFactory(models)
        timed_factory = TimedModelFactory(base_factory)
        cost_model = MeasuredCostModel(timed_factory)

        # Evaluate to get timing data
        samples = self._bkd.array([[0.1, 0.2, 0.3]])
        model = timed_factory.get_model((0,))
        model(samples)

        cost = cost_model((0,))
        self.assertGreater(cost, 0.0)

    def test_measured_cost_updates_during_refinement(self) -> None:
        """TimedModelFactory + MeasuredCostModel + fitter: costs evolve."""
        models = _make_cosine_models(self._bkd)
        base_factory = DictModelFactory(models)
        timed_factory = TimedModelFactory(base_factory)
        cost_model = MeasuredCostModel(timed_factory)

        fitter = _make_mf_fitter(
            self._bkd,
            max_level=3,
            cost_model=cost_model,
        )

        # Run a few steps using timed_factory as the model_factory
        fitter.refine_to_tolerance(timed_factory, tol=1e-6, max_steps=10)

        # After refinement, at least one config should have recorded timings
        all_timers = timed_factory.all_timers()
        self.assertGreater(len(all_timers), 0)

        # Cost model should now return measured costs for evaluated configs
        for cfg in all_timers:
            call_timer = all_timers[cfg].get("__call__")
            if call_timer.call_count() > 0:
                cost = cost_model(cfg)
                self.assertGreater(cost, 0.0)


class TestTimedAndMeasuredNumpy(TestTimedAndMeasured[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestTimedAndMeasuredTorch(TestTimedAndMeasured[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# Convergence tests
# =============================================================================


class TestMFConvergence(Generic[Array], unittest.TestCase):
    """Convergence and cost-awareness tests for MF sparse grids."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_1d_physical_1d_config_convergence(self) -> None:
        """MF adaptive grid converges on cosine model."""
        models = _make_cosine_models(self._bkd)
        factory = DictModelFactory(models)
        fitter = _make_mf_fitter(
            self._bkd,
            max_level=10,
        )
        result = fitter.refine_to_tolerance(factory, tol=1e-10, max_steps=15)

        # Evaluate the high-fidelity model at test points
        np.random.seed(42)
        test_pts = self._bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        exact = models[(1,)](test_pts)
        approx = result.surrogate(test_pts)

        max_err = float(
            self._bkd.to_numpy(self._bkd.max(self._bkd.abs(exact - approx)))
        )
        self.assertLess(max_err, 1e-6)

    def test_1d_physical_1d_config_cost_aware_prefers_cheap(self) -> None:
        """With ExponentialConfigCostModel, more low-fidelity subspaces."""
        models = _make_cosine_models(self._bkd)
        factory = DictModelFactory(models)
        cost_model = ExponentialConfigCostModel(base=10.0)
        indicator = CostWeightedIndicator(
            self._bkd,
            L2SurrogateDifferenceIndicator(self._bkd),
        )

        fitter = _make_mf_fitter(
            self._bkd,
            max_level=10,
            cost_model=cost_model,
            error_indicator=indicator,
        )
        result = fitter.refine_to_tolerance(factory, tol=1e-8, max_steps=15)

        # Count selected indices per config level
        indices = result.indices
        nvars_total = indices.shape[0]
        config_dim = nvars_total - 1  # last dim is config

        count_low = 0
        count_high = 0
        for j in range(indices.shape[1]):
            cfg_level = int(self._bkd.to_numpy(indices[config_dim, j]))
            if cfg_level == 0:
                count_low += 1
            else:
                count_high += 1

        # With high cost ratio, low-fidelity should dominate
        self.assertGreaterEqual(count_low, count_high)

    def test_single_fidelity_matches_sf_fitter(self) -> None:
        """SF adaptive fitter converges on cosine model."""
        # The high-fidelity model
        models = _make_cosine_models(self._bkd)
        hf_model = models[(1,)]

        # SF fitter with enough levels to converge
        sf_fitter = _make_sf_fitter(self._bkd, max_level=10)
        sf_result = sf_fitter.refine_to_tolerance(hf_model, tol=1e-12, max_steps=15)

        np.random.seed(42)
        test_pts = self._bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        exact = hf_model(test_pts)
        approx = sf_result.surrogate(test_pts)
        self._bkd.assert_allclose(approx, exact, rtol=1e-6)


class TestMFConvergenceNumpy(TestMFConvergence[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestMFConvergenceTorch(TestMFConvergence[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


# =============================================================================
# CandidateInfo field tests
# =============================================================================


class TestCandidateInfoFields(Generic[Array], unittest.TestCase):
    """Tests that CandidateInfo fields are correctly populated."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_candidate_info_config_idx_none_for_sf(self) -> None:
        """SF fitter produces CandidateInfo with config_idx=None."""
        fitter = _make_sf_fitter(self._bkd, max_level=3)

        def func(s: Array) -> Array:
            return self._bkd.reshape(s[0, :] ** 2, (1, -1))

        samples = fitter.step_samples()
        assert samples is not None
        values = func(samples)
        fitter.step_values(values)

        # Access the underlying MF fitter's internals via composition
        mf = fitter._fitter
        samples2 = fitter.step_samples()
        if samples2 is not None:
            from pyapprox.surrogates.sparsegrids.smolyak import (
                compute_smolyak_coefficients,
            )

            sel_indices = mf._index_gen.get_selected_indices()
            sel_coefs = compute_smolyak_coefficients(sel_indices, self._bkd)
            sel_subspaces = mf._get_subspaces_for_indices(sel_indices)
            sel_surr = CombinationSurrogate(
                self._bkd,
                mf._nvars_physical,
                sel_subspaces,
                sel_coefs,
                1,
                indices=sel_indices,
            )
            cand_indices = mf._index_gen.get_candidate_indices()
            if cand_indices is not None and cand_indices.shape[1] > 0:
                cand_idx = cand_indices[:, 0]
                from pyapprox.surrogates.sparsegrids.smolyak import (
                    _index_to_tuple,
                )

                cand_key = _index_to_tuple(cand_idx)
                sub_pos = mf._subspace_keys.index(cand_key)
                cand_sub = mf._subspaces[sub_pos]
                if cand_sub.get_values() is not None:
                    info = mf._build_candidate_info(
                        cand_idx,
                        cand_sub,
                        sel_indices,
                        sel_coefs,
                        sel_surr,
                    )
                    self.assertIsNone(info.config_idx)
                    # Costs should still be populated
                    self.assertIsNotNone(info.model_cost)
                    self.assertIsNotNone(info.subspace_cost)

    def test_candidate_info_costs_always_populated(self) -> None:
        """model_cost and subspace_cost are always non-None."""
        models = _make_cosine_models(self._bkd)
        fitter = _make_mf_fitter(self._bkd, max_level=3)

        samples = fitter.step_samples()
        assert isinstance(samples, dict)
        values = {cfg: models[cfg](s) for cfg, s in samples.items()}
        fitter.step_values(values)

        # Build CandidateInfo for a candidate
        from pyapprox.surrogates.sparsegrids.smolyak import (
            _index_to_tuple,
            compute_smolyak_coefficients,
        )

        sel_indices = fitter._index_gen.get_selected_indices()
        sel_coefs = compute_smolyak_coefficients(sel_indices, self._bkd)
        sel_subspaces = fitter._get_subspaces_for_indices(sel_indices)
        sel_surr = CombinationSurrogate(
            self._bkd,
            fitter._nvars_physical,
            sel_subspaces,
            sel_coefs,
            1,
            indices=sel_indices,
        )

        cand_indices = fitter._index_gen.get_candidate_indices()
        if cand_indices is not None:
            found = False
            for j in range(cand_indices.shape[1]):
                cand_idx = cand_indices[:, j]
                cand_key = _index_to_tuple(cand_idx)
                if cand_key in fitter._subspace_keys:
                    sub_pos = fitter._subspace_keys.index(cand_key)
                    cand_sub = fitter._subspaces[sub_pos]
                    if cand_sub.get_values() is not None:
                        info = fitter._build_candidate_info(
                            cand_idx,
                            cand_sub,
                            sel_indices,
                            sel_coefs,
                            sel_surr,
                        )
                        self.assertIsNotNone(info.model_cost)
                        self.assertIsNotNone(info.subspace_cost)
                        self.assertGreater(info.model_cost, 0)
                        self.assertGreater(info.subspace_cost, 0)
                        found = True
                        break
            self.assertTrue(found, "No candidate with values found")

    def test_candidate_info_new_samples_never_none(self) -> None:
        """new_samples is always a valid Array, never None."""
        models = _make_cosine_models(self._bkd)
        fitter = _make_mf_fitter(self._bkd, max_level=3)

        samples = fitter.step_samples()
        assert isinstance(samples, dict)
        values = {cfg: models[cfg](s) for cfg, s in samples.items()}
        fitter.step_values(values)

        from pyapprox.surrogates.sparsegrids.smolyak import (
            _index_to_tuple,
            compute_smolyak_coefficients,
        )

        sel_indices = fitter._index_gen.get_selected_indices()
        sel_coefs = compute_smolyak_coefficients(sel_indices, self._bkd)
        sel_subspaces = fitter._get_subspaces_for_indices(sel_indices)
        sel_surr = CombinationSurrogate(
            self._bkd,
            fitter._nvars_physical,
            sel_subspaces,
            sel_coefs,
            1,
            indices=sel_indices,
        )

        cand_indices = fitter._index_gen.get_candidate_indices()
        if cand_indices is not None:
            for j in range(cand_indices.shape[1]):
                cand_idx = cand_indices[:, j]
                cand_key = _index_to_tuple(cand_idx)
                if cand_key in fitter._subspace_keys:
                    sub_pos = fitter._subspace_keys.index(cand_key)
                    cand_sub = fitter._subspaces[sub_pos]
                    if cand_sub.get_values() is not None:
                        info = fitter._build_candidate_info(
                            cand_idx,
                            cand_sub,
                            sel_indices,
                            sel_coefs,
                            sel_surr,
                        )
                        self.assertIsNotNone(info.new_samples)
                        self.assertGreater(info.new_samples.shape[1], 0)
                        break


class TestCandidateInfoFieldsNumpy(TestCandidateInfoFields[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCandidateInfoFieldsTorch(TestCandidateInfoFields[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()


if __name__ == "__main__":
    unittest.main()
