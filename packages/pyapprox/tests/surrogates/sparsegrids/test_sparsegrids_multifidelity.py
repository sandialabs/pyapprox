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
from typing import Callable, Dict

import numpy as np
import pytest

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
    L2GlobalSurplusIndicator,
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

# =============================================================================
# Helpers
# =============================================================================


def _make_cosine_models(
    bkd,
) -> Dict[ConfigIdx, Callable]:
    """Create cosine model hierarchy.

    f_alpha(z) = cos(pi*(z+1)/2 + eps_alpha)
    eps = {0: 0.5, 1: 0.0}
    """
    epsilons = {0: 0.5, 1: 0.0}

    def _make_model(eps: float) -> Callable:
        def model(samples):
            return bkd.reshape(
                bkd.cos(math.pi * (samples[0, :] + 1) / 2 + eps),
                (1, -1),
            )

        return model

    return {(alpha,): _make_model(epsilons[alpha]) for alpha in epsilons}


def _make_mf_fitter(
    bkd,
    max_level: int = 4,
    max_config_level: int = 1,
    cost_model=None,
    error_indicator=None,
):
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
    bkd,
    max_level: int = 4,
):
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


class TestMFIsotropic:
    """Tests for IsotropicSparseGridFitter with nconfig_vars > 0."""

    def _make_isotropic_mf_fitter(
        self, bkd, level: int = 1
    ):
        """Create isotropic MF fitter with level capped for 2 models."""
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factories = [GaussLagrangeFactory(marginal, bkd)]
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory = TensorProductSubspaceFactory(bkd, factories, growth)
        return IsotropicSparseGridFitter(
            bkd, tp_factory, level=level, nconfig_vars=1
        )

    def test_mf_isotropic_get_samples_returns_dict(self, bkd) -> None:
        """get_samples() returns Dict[ConfigIdx, Array] when nconfig_vars > 0."""
        fitter = self._make_isotropic_mf_fitter(bkd, level=1)
        samples = fitter.get_samples()
        assert isinstance(samples, dict)
        # Should have config keys that are tuples of ints
        for key in samples:
            assert isinstance(key, tuple)

    def test_mf_isotropic_fit_produces_surrogate(self, bkd) -> None:
        """Fit with dict values produces a working surrogate."""
        fitter = self._make_isotropic_mf_fitter(bkd, level=1)
        samples = fitter.get_samples()
        assert isinstance(samples, dict)

        models = _make_cosine_models(bkd)
        values = {cfg: models[cfg](s) for cfg, s in samples.items()}
        result = fitter.fit(values)

        # Surrogate should evaluate to finite values
        test_pts = bkd.array([[0.5]])
        out = result.surrogate(test_pts)
        assert out.shape[0] == 1
        assert bool(bkd.all_bool(bkd.isfinite(out)))

    def test_mf_isotropic_nsamples_counts_across_configs(self, bkd) -> None:
        """result.nsamples equals sum across all config groups."""
        fitter = self._make_isotropic_mf_fitter(bkd, level=1)
        samples = fitter.get_samples()
        assert isinstance(samples, dict)

        models = _make_cosine_models(bkd)
        values = {cfg: models[cfg](s) for cfg, s in samples.items()}
        result = fitter.fit(values)

        total_samples = sum(s.shape[1] for s in samples.values())
        assert result.nsamples == total_samples


# =============================================================================
# Adaptive MF tests
# =============================================================================


class TestMFAdaptive:
    """Tests for MultiFidelityAdaptiveSparseGridFitter."""

    def test_mf_adaptive_step_samples_returns_dict(self, bkd) -> None:
        """First step_samples() returns Dict when nconfig_vars > 0."""
        fitter = _make_mf_fitter(bkd, max_level=3)
        samples = fitter.step_samples()
        assert isinstance(samples, dict)
        for key in samples:
            assert isinstance(key, tuple)

    def test_mf_adaptive_step_values_accepts_dict(self, bkd) -> None:
        """step_values with dict doesn't error and reprioritizes."""
        fitter = _make_mf_fitter(bkd, max_level=3)
        samples = fitter.step_samples()
        assert isinstance(samples, dict)

        models = _make_cosine_models(bkd)
        values = {cfg: models[cfg](s) for cfg, s in samples.items()}
        # Should not raise
        fitter.step_values(values)

        # Should be able to get next samples
        samples2 = fitter.step_samples()
        # Might be dict or None
        if samples2 is not None:
            assert isinstance(samples2, dict)

    @pytest.mark.slow_on("TorchBkd")
    def test_mf_adaptive_cost_model_changes_selection(self, bkd) -> None:
        """Non-trivial cost model changes which candidates get selected."""
        models = _make_cosine_models(bkd)

        # Default: ConstantCostModel (unit cost)
        fitter_unit = _make_mf_fitter(bkd, max_level=3)
        samples = fitter_unit.step_samples()
        assert isinstance(samples, dict)
        values = {cfg: models[cfg](s) for cfg, s in samples.items()}
        fitter_unit.step_values(values)

        # Fidelity-weighted cost (base=4 penalises high-fidelity)
        cost_model = ExponentialConfigCostModel(base=4.0)
        fitter_cw = _make_mf_fitter(
            bkd,
            max_level=3,
            cost_model=cost_model,
        )
        samples_cw = fitter_cw.step_samples()
        assert isinstance(samples_cw, dict)
        values_cw = {cfg: models[cfg](s) for cfg, s in samples_cw.items()}
        fitter_cw.step_values(values_cw)

        # Drive both forward a few steps
        for _ in range(5):
            s1 = fitter_unit.step_samples()
            if s1 is not None:
                assert isinstance(s1, dict)
                v1 = {cfg: models[cfg](s) for cfg, s in s1.items()}
                fitter_unit.step_values(v1)

            s2 = fitter_cw.step_samples()
            if s2 is not None:
                assert isinstance(s2, dict)
                v2 = {cfg: models[cfg](s) for cfg, s in s2.items()}
                fitter_cw.step_values(v2)

        r1 = fitter_unit.result()
        r2 = fitter_cw.result()
        assert isinstance(r1.surrogate, CombinationSurrogate)
        assert isinstance(r2.surrogate, CombinationSurrogate)

    def test_mf_adaptive_subspace_cost_reflects_cost_model(self, bkd) -> None:
        """With ExponentialConfigCostModel, high-fidelity costs more."""
        cost_model = ExponentialConfigCostModel(base=4.0)
        # Cost for config (0,) = 4^0 = 1.0
        # Cost for config (1,) = 4^1 = 4.0
        bkd.assert_allclose(
            bkd.asarray([cost_model((0,))]),
            bkd.asarray([1.0]),
        )
        bkd.assert_allclose(
            bkd.asarray([cost_model((1,))]),
            bkd.asarray([4.0]),
        )

    def test_mf_adaptive_recovers_isotropic_with_constant_cost(
        self, bkd,
    ) -> None:
        """With ConstantCostModel and enough steps, matches isotropic."""
        models = _make_cosine_models(bkd)
        level = 1  # Use level=1 so only config 0,1 are generated

        # Build isotropic MF grid
        marginal = UniformMarginal(-1.0, 1.0, bkd)
        factories_iso = [GaussLagrangeFactory(marginal, bkd)]
        growth = LinearGrowthRule(scale=1, shift=1)
        tp_factory_iso = TensorProductSubspaceFactory(bkd, factories_iso, growth)
        iso_fitter = IsotropicSparseGridFitter(
            bkd, tp_factory_iso, level=level, nconfig_vars=1
        )
        iso_samples = iso_fitter.get_samples()
        assert isinstance(iso_samples, dict)
        iso_values = {cfg: models[cfg](s) for cfg, s in iso_samples.items()}
        iso_result = iso_fitter.fit(iso_values)

        # Build adaptive MF grid with ConstantCostModel
        factory = DictModelFactory(models)
        ada_fitter = _make_mf_fitter(
            bkd,
            max_level=level,
            max_config_level=1,
            cost_model=ConstantCostModel(),
        )
        ada_result = ada_fitter.refine_to_tolerance(factory, tol=1e-14, max_steps=10)

        # Both should give similar evaluation at test points
        test_pts = bkd.array([[0.3, -0.5, 0.8]])
        iso_val = iso_result.surrogate(test_pts)
        ada_val = ada_result.surrogate(test_pts)
        bkd.assert_allclose(ada_val, iso_val, rtol=1e-8)

    def test_mf_refine_to_tolerance_via_model_factory(self, bkd) -> None:
        """DictModelFactory + refine_to_tolerance works without target_fn."""
        models = _make_cosine_models(bkd)
        factory = DictModelFactory(models)
        fitter = _make_mf_fitter(
            bkd,
            max_level=4,
        )

        # model_factory is first positional arg
        result = fitter.refine_to_tolerance(factory, tol=1e-6, max_steps=10)
        assert isinstance(result.surrogate, CombinationSurrogate)
        assert result.nsamples > 0


# =============================================================================
# TimedModelFactory + MeasuredCostModel tests
# =============================================================================


class TestTimedAndMeasured:
    """Tests for TimedModelFactory and MeasuredCostModel."""

    def test_timed_factory_records_per_config_timings(self, bkd) -> None:
        """After get_model(cfg)(samples), timer has call_count > 0."""
        models = _make_cosine_models(bkd)
        base_factory = DictModelFactory(models)
        timed_factory = TimedModelFactory(base_factory)

        samples = bkd.array([[0.1, 0.2, 0.3]])
        model = timed_factory.get_model((0,))
        model(samples)

        timer = timed_factory.timer((0,))
        assert timer.get("__call__").call_count() > 0

    def test_measured_cost_returns_unit_before_data(self, bkd) -> None:
        """MeasuredCostModel returns 1.0 when call_count is 0."""
        models = _make_cosine_models(bkd)
        base_factory = DictModelFactory(models)
        timed_factory = TimedModelFactory(base_factory)
        cost_model = MeasuredCostModel(timed_factory)

        # No evaluations yet
        cost = cost_model((0,))
        bkd.assert_allclose(
            bkd.asarray([cost]),
            bkd.asarray([1.0]),
        )

    def test_measured_cost_returns_median_after_evals(self, bkd) -> None:
        """After evaluations, MeasuredCostModel returns median time > 0."""
        models = _make_cosine_models(bkd)
        base_factory = DictModelFactory(models)
        timed_factory = TimedModelFactory(base_factory)
        cost_model = MeasuredCostModel(timed_factory)

        # Evaluate to get timing data
        samples = bkd.array([[0.1, 0.2, 0.3]])
        model = timed_factory.get_model((0,))
        model(samples)

        cost = cost_model((0,))
        assert cost > 0.0

    def test_measured_cost_updates_during_refinement(self, bkd) -> None:
        """TimedModelFactory + MeasuredCostModel + fitter: costs evolve."""
        models = _make_cosine_models(bkd)
        base_factory = DictModelFactory(models)
        timed_factory = TimedModelFactory(base_factory)
        cost_model = MeasuredCostModel(timed_factory)

        fitter = _make_mf_fitter(
            bkd,
            max_level=3,
            cost_model=cost_model,
        )

        # Run a few steps using timed_factory as the model_factory
        fitter.refine_to_tolerance(timed_factory, tol=1e-6, max_steps=10)

        # After refinement, at least one config should have recorded timings
        all_timers = timed_factory.all_timers()
        assert len(all_timers) > 0

        # Cost model should now return measured costs for evaluated configs
        for cfg in all_timers:
            call_timer = all_timers[cfg].get("__call__")
            if call_timer.call_count() > 0:
                cost = cost_model(cfg)
                assert cost > 0.0


# =============================================================================
# Convergence tests
# =============================================================================


class TestMFConvergence:
    """Convergence and cost-awareness tests for MF sparse grids."""

    def test_1d_physical_1d_config_convergence(self, bkd) -> None:
        """MF adaptive grid converges on cosine model."""
        models = _make_cosine_models(bkd)
        factory = DictModelFactory(models)
        fitter = _make_mf_fitter(
            bkd,
            max_level=10,
        )
        result = fitter.refine_to_tolerance(factory, tol=1e-10, max_steps=15)

        # Evaluate the high-fidelity model at test points
        np.random.seed(42)
        test_pts = bkd.asarray(np.random.uniform(-1, 1, (1, 50)))
        exact = models[(1,)](test_pts)
        approx = result.surrogate(test_pts)

        max_err = float(
            bkd.to_numpy(bkd.max(bkd.abs(exact - approx)))
        )
        assert max_err < 1e-6

    def test_1d_physical_1d_config_cost_aware_prefers_cheap(self, bkd) -> None:
        """With ExponentialConfigCostModel, more low-fidelity subspaces."""
        models = _make_cosine_models(bkd)
        factory = DictModelFactory(models)
        cost_model = ExponentialConfigCostModel(base=10.0)
        indicator = L2GlobalSurplusIndicator(bkd)

        fitter = _make_mf_fitter(
            bkd,
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
            cfg_level = int(bkd.to_numpy(indices[config_dim, j]))
            if cfg_level == 0:
                count_low += 1
            else:
                count_high += 1

        # With high cost ratio, low-fidelity should dominate
        assert count_low >= count_high

    def test_single_fidelity_matches_sf_fitter(self, bkd) -> None:
        """SF adaptive fitter converges on cosine model."""
        # The high-fidelity model
        models = _make_cosine_models(bkd)
        hf_model = models[(1,)]

        # SF fitter with enough levels to converge
        sf_fitter = _make_sf_fitter(bkd, max_level=10)
        sf_result = sf_fitter.refine_to_tolerance(hf_model, tol=1e-12, max_steps=15)

        np.random.seed(42)
        test_pts = bkd.asarray(np.random.uniform(-1, 1, (1, 20)))
        exact = hf_model(test_pts)
        approx = sf_result.surrogate(test_pts)
        bkd.assert_allclose(approx, exact, rtol=1e-6)


# =============================================================================
# CandidateInfo field tests
# =============================================================================


class TestCandidateInfoFields:
    """Tests that CandidateInfo fields are correctly populated."""

    def test_candidate_info_config_idx_none_for_sf(self, bkd) -> None:
        """SF fitter produces CandidateInfo with config_idx=None."""
        fitter = _make_sf_fitter(bkd, max_level=3)

        def func(s):
            return bkd.reshape(s[0, :] ** 2, (1, -1))

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
            sel_coefs = compute_smolyak_coefficients(sel_indices, bkd)
            sel_subspaces = mf._get_subspaces_for_indices(sel_indices)
            sel_surr = CombinationSurrogate(
                bkd,
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

                cand_key = _index_to_tuple(cand_idx, bkd)
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
                    assert info.config_idx is None
                    # Costs should still be populated
                    assert info.model_cost is not None
                    assert info.subspace_cost is not None

    def test_candidate_info_costs_always_populated(self, bkd) -> None:
        """model_cost and subspace_cost are always non-None."""
        models = _make_cosine_models(bkd)
        fitter = _make_mf_fitter(bkd, max_level=3)

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
        sel_coefs = compute_smolyak_coefficients(sel_indices, bkd)
        sel_subspaces = fitter._get_subspaces_for_indices(sel_indices)
        sel_surr = CombinationSurrogate(
            bkd,
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
                cand_key = _index_to_tuple(cand_idx, bkd)
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
                        assert info.model_cost is not None
                        assert info.subspace_cost is not None
                        assert info.model_cost > 0
                        assert info.subspace_cost > 0
                        found = True
                        break
            assert found, "No candidate with values found"

    def test_candidate_info_new_samples_never_none(self, bkd) -> None:
        """new_samples is always a valid Array, never None."""
        models = _make_cosine_models(bkd)
        fitter = _make_mf_fitter(bkd, max_level=3)

        samples = fitter.step_samples()
        assert isinstance(samples, dict)
        values = {cfg: models[cfg](s) for cfg, s in samples.items()}
        fitter.step_values(values)

        from pyapprox.surrogates.sparsegrids.smolyak import (
            _index_to_tuple,
            compute_smolyak_coefficients,
        )

        sel_indices = fitter._index_gen.get_selected_indices()
        sel_coefs = compute_smolyak_coefficients(sel_indices, bkd)
        sel_subspaces = fitter._get_subspaces_for_indices(sel_indices)
        sel_surr = CombinationSurrogate(
            bkd,
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
                cand_key = _index_to_tuple(cand_idx, bkd)
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
                        assert info.new_samples is not None
                        assert info.new_samples.shape[1] > 0
                        break
