"""Tests for cached FunctionTrain compute functions.

Tests validate:
- cache_basis_matrices deduplication and correctness
- core_eval_cached matches FunctionTrainCore.__call__
- ft_eval_cached matches FunctionTrain.__call__
- ft_jacobian_wrt_params_cached matches FunctionTrain.jacobian_wrt_params
- Additive FT with ConstantExpansion handled correctly
- Convenience methods (eval_cached, jacobian_wrt_params_cached) on classes
- with_params + cached pipeline matches original pipeline
"""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.functiontrain import (
    create_additive_functiontrain,
    create_pce_functiontrain,
)
from pyapprox.surrogates.functiontrain.compute import (
    cache_basis_matrices,
    core_eval_cached,
    ft_eval_cached,
    ft_jacobian_wrt_params_cached,
)


class TestCacheBasisMatrices:
    """Tests for cache_basis_matrices deduplication."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_uniform_pce_deduplication(self, bkd) -> None:
        """Shared basis objects produce one cache entry per variable."""
        nvars = 4
        rank = 3
        max_level = 4
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        ft = create_pce_functiontrain(marginals, max_level, [rank] * (nvars - 1), bkd)
        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        cache = cache_basis_matrices(ft.cores(), samples, bkd)

        # All expansions in core k share one basis -> nvars unique entries
        bkd.assert_allclose(
            bkd.asarray([float(len(cache))]),
            bkd.asarray([float(nvars)]),
        )

    def test_additive_ft_skips_constants(self, bkd) -> None:
        """ConstantExpansion has no get_basis, so it's skipped in cache."""
        nvars = 3
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        univariate_bases = []
        for k in range(nvars):
            indices = compute_hyperbolic_indices(1, 3, 1.0, bkd)
            basis = OrthonormalPolynomialBasis([bases_1d[k]], bkd, indices)
            exp = BasisExpansion(basis, bkd, nqoi=1)
            exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))
            univariate_bases.append(exp)
        ft = create_additive_functiontrain(univariate_bases, bkd)

        nsamples = 10
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        cache = cache_basis_matrices(ft.cores(), samples, bkd)

        # Additive FT: each core has one BasisExpansion and several
        # ConstantExpansion. Only the BasisExpansion entries are cached.
        # Number of unique basis objects = nvars
        bkd.assert_allclose(
            bkd.asarray([float(len(cache))]),
            bkd.asarray([float(nvars)]),
        )

    def test_cache_values_match_basis_eval(self, bkd) -> None:
        """Cached basis matrices match direct basis(sample_1d)."""
        nvars = 3
        max_level = 3
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        ft = create_pce_functiontrain(marginals, max_level, [2, 2], bkd)
        nsamples = 15
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        cache = cache_basis_matrices(ft.cores(), samples, bkd)

        # Verify each cache entry
        for kk, core in enumerate(ft.cores()):
            sample_1d = samples[kk : kk + 1]
            r_left, r_right = core.ranks()
            for ii in range(r_left):
                for jj in range(r_right):
                    bexp = core.get_basisexp(ii, jj)
                    if not hasattr(bexp, "get_basis"):
                        continue
                    basis = bexp.get_basis()
                    expected = basis(sample_1d)
                    cached = cache[id(basis)]
                    bkd.assert_allclose(cached, expected, rtol=1e-14)


class TestCoreEvalCached:
    """Tests for core_eval_cached correctness."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_pce_core_eval_matches_original(self, bkd) -> None:
        """Cached core eval matches FunctionTrainCore.__call__."""
        nvars = 3
        max_level = 4
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        ft = create_pce_functiontrain(marginals, max_level, [3, 3], bkd, init_scale=0.5)
        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        cache = cache_basis_matrices(ft.cores(), samples, bkd)

        for kk, core in enumerate(ft.cores()):
            sample_1d = samples[kk : kk + 1]
            reference = core(sample_1d)
            cached = core_eval_cached(core, sample_1d, cache, bkd)
            bkd.assert_allclose(cached, reference, rtol=1e-13)

    def test_additive_core_eval_matches_original(self, bkd) -> None:
        """Cached core eval matches original for additive FT with constants."""
        nvars = 3
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        univariate_bases = []
        for k in range(nvars):
            indices = compute_hyperbolic_indices(1, 3, 1.0, bkd)
            basis = OrthonormalPolynomialBasis([bases_1d[k]], bkd, indices)
            exp = BasisExpansion(basis, bkd, nqoi=1)
            exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))
            univariate_bases.append(exp)
        ft = create_additive_functiontrain(univariate_bases, bkd)
        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        cache = cache_basis_matrices(ft.cores(), samples, bkd)

        for kk, core in enumerate(ft.cores()):
            sample_1d = samples[kk : kk + 1]
            reference = core(sample_1d)
            cached = core_eval_cached(core, sample_1d, cache, bkd)
            bkd.assert_allclose(cached, reference, rtol=1e-13)


class TestFTEvalCached:
    """Tests for ft_eval_cached correctness."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_uniform_pce_eval(self, bkd) -> None:
        """Cached FT eval matches FT.__call__ for uniform PCE."""
        nvars = 5
        max_level = 5
        rank = 4
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        ft = create_pce_functiontrain(
            marginals, max_level, [rank] * (nvars - 1), bkd, init_scale=0.3
        )
        nsamples = 200
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        cache = cache_basis_matrices(ft.cores(), samples, bkd)
        reference = ft(samples)
        cached = ft_eval_cached(ft.cores(), samples, cache, bkd)

        bkd.assert_allclose(cached, reference, rtol=1e-12)

    def test_additive_ft_eval(self, bkd) -> None:
        """Cached FT eval matches FT.__call__ for additive FT."""
        nvars = 3
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        univariate_bases = []
        for k in range(nvars):
            indices = compute_hyperbolic_indices(1, 3, 1.0, bkd)
            basis = OrthonormalPolynomialBasis([bases_1d[k]], bkd, indices)
            exp = BasisExpansion(basis, bkd, nqoi=1)
            exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))
            univariate_bases.append(exp)
        ft = create_additive_functiontrain(univariate_bases, bkd)
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        cache = cache_basis_matrices(ft.cores(), samples, bkd)
        reference = ft(samples)
        cached = ft_eval_cached(ft.cores(), samples, cache, bkd)

        bkd.assert_allclose(cached, reference, rtol=1e-12)

    def test_rank_2_nvars_2(self, bkd) -> None:
        """Edge case: 2 variables, rank 2 (only first and last core)."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        ft = create_pce_functiontrain(marginals, 3, [2], bkd, init_scale=0.5)
        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))

        cache = cache_basis_matrices(ft.cores(), samples, bkd)
        reference = ft(samples)
        cached = ft_eval_cached(ft.cores(), samples, cache, bkd)

        bkd.assert_allclose(cached, reference, rtol=1e-12)

    def test_convenience_method(self, bkd) -> None:
        """FunctionTrain.eval_cached matches FT.__call__."""
        nvars = 3
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        ft = create_pce_functiontrain(marginals, 4, [2, 2], bkd, init_scale=0.3)
        nsamples = 40
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        cache = cache_basis_matrices(ft.cores(), samples, bkd)
        reference = ft(samples)
        cached = ft.eval_cached(samples, cache)

        bkd.assert_allclose(cached, reference, rtol=1e-12)


class TestFTJacobianCached:
    """Tests for ft_jacobian_wrt_params_cached correctness."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_uniform_pce_jacobian(self, bkd) -> None:
        """Cached jacobian matches FT.jacobian_wrt_params for uniform PCE."""
        nvars = 5
        max_level = 5
        rank = 4
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        ft = create_pce_functiontrain(
            marginals, max_level, [rank] * (nvars - 1), bkd, init_scale=0.3
        )
        nsamples = 100
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        cache = cache_basis_matrices(ft.cores(), samples, bkd)
        reference = ft.jacobian_wrt_params(samples)
        cached = ft_jacobian_wrt_params_cached(ft.cores(), samples, cache, bkd)

        bkd.assert_allclose(cached, reference, rtol=1e-12)

    def test_additive_ft_jacobian(self, bkd) -> None:
        """Cached jacobian matches original for additive FT."""
        nvars = 3
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        univariate_bases = []
        for k in range(nvars):
            indices = compute_hyperbolic_indices(1, 3, 1.0, bkd)
            basis = OrthonormalPolynomialBasis([bases_1d[k]], bkd, indices)
            exp = BasisExpansion(basis, bkd, nqoi=1)
            exp.set_coefficients(bkd.asarray(np.random.randn(exp.nterms(), 1)))
            univariate_bases.append(exp)
        ft = create_additive_functiontrain(univariate_bases, bkd)
        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        cache = cache_basis_matrices(ft.cores(), samples, bkd)
        reference = ft.jacobian_wrt_params(samples)
        cached = ft_jacobian_wrt_params_cached(ft.cores(), samples, cache, bkd)

        bkd.assert_allclose(cached, reference, rtol=1e-12)

    def test_rank_2_nvars_2_jacobian(self, bkd) -> None:
        """Edge case: 2 variables, rank 2."""
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(2)]
        ft = create_pce_functiontrain(marginals, 3, [2], bkd, init_scale=0.5)
        nsamples = 20
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))

        cache = cache_basis_matrices(ft.cores(), samples, bkd)
        reference = ft.jacobian_wrt_params(samples)
        cached = ft_jacobian_wrt_params_cached(ft.cores(), samples, cache, bkd)

        bkd.assert_allclose(cached, reference, rtol=1e-12)

    def test_convenience_method(self, bkd) -> None:
        """FunctionTrain.jacobian_wrt_params_cached matches original."""
        nvars = 3
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        ft = create_pce_functiontrain(marginals, 4, [2, 2], bkd, init_scale=0.3)
        nsamples = 40
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        cache = cache_basis_matrices(ft.cores(), samples, bkd)
        reference = ft.jacobian_wrt_params(samples)
        cached = ft.jacobian_wrt_params_cached(samples, cache)

        bkd.assert_allclose(cached, reference, rtol=1e-12)

    def test_with_params_then_cached(self, bkd) -> None:
        """with_params + cached pipeline matches original after param change."""
        nvars = 3
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        ft = create_pce_functiontrain(marginals, 4, [3, 3], bkd, init_scale=0.3)
        nsamples = 50
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        # Change params
        new_params = bkd.asarray(np.random.randn(ft.nparams()))
        ft_new = ft.with_params(new_params)

        # Cache is built from original FT cores (same basis objects)
        cache = cache_basis_matrices(ft.cores(), samples, bkd)

        # Cached eval/jac with new FT should match direct
        ref_eval = ft_new(samples)
        cached_eval = ft_eval_cached(ft_new.cores(), samples, cache, bkd)
        bkd.assert_allclose(cached_eval, ref_eval, rtol=1e-12)

        ref_jac = ft_new.jacobian_wrt_params(samples)
        cached_jac = ft_jacobian_wrt_params_cached(ft_new.cores(), samples, cache, bkd)
        bkd.assert_allclose(cached_jac, ref_jac, rtol=1e-12)

    def test_varying_ranks(self, bkd) -> None:
        """Non-uniform ranks across cores."""
        nvars = 4
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        ranks = [2, 4, 3]
        ft = create_pce_functiontrain(marginals, 3, ranks, bkd, init_scale=0.3)
        nsamples = 30
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))

        cache = cache_basis_matrices(ft.cores(), samples, bkd)
        ref_eval = ft(samples)
        cached_eval = ft_eval_cached(ft.cores(), samples, cache, bkd)
        bkd.assert_allclose(cached_eval, ref_eval, rtol=1e-12)

        ref_jac = ft.jacobian_wrt_params(samples)
        cached_jac = ft_jacobian_wrt_params_cached(ft.cores(), samples, cache, bkd)
        bkd.assert_allclose(cached_jac, ref_jac, rtol=1e-12)


class TestCoreConvenienceMethod:
    """Tests for FunctionTrainCore.eval_cached convenience method."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def test_core_eval_cached_matches(self, bkd) -> None:
        """Core.eval_cached matches Core.__call__."""
        nvars = 3
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        ft = create_pce_functiontrain(marginals, 4, [2, 2], bkd, init_scale=0.5)
        nsamples = 25
        samples = bkd.asarray(np.random.uniform(-1, 1, (nvars, nsamples)))
        cache = cache_basis_matrices(ft.cores(), samples, bkd)

        for kk, core in enumerate(ft.cores()):
            sample_1d = samples[kk : kk + 1]
            reference = core(sample_1d)
            cached = core.eval_cached(sample_1d, cache)
            bkd.assert_allclose(cached, reference, rtol=1e-13)
