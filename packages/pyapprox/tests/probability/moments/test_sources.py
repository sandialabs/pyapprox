"""Tests for moment sources and rule adapters.

A linear map of independent Gaussian inputs has exact blocks, and any rule
exact for degree-2 polynomials reproduces them. A two-point-per-dimension
tensor Gauss-Hermite rule is such a rule.
"""

from typing import Tuple

import numpy as np
import pytest
from pyapprox.interface.functions.fromcallable.function import (
    FunctionFromCallable,
)
from pyapprox.interface.functions.joint import (
    InputTarget,
    SeparateFunctions,
    SplitFunction,
)
from pyapprox.probability.moments import (
    AtLevel,
    CachedMoments,
    MomentSourceProtocol,
    QuadratureMoments,
    SampledRule,
    UnbiasedMCAccumulator,
    WeightedRuleProtocol,
)
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.surrogates.affine.indices import LinearGrowthRule
from pyapprox.surrogates.quadrature import (
    ParameterizedTensorProductQuadratureRule,
    TensorProductQuadratureRule,
    gauss_quadrature_rule,
)
from pyapprox.util.backends.protocols import Array, Backend


class _GaussianSampler:
    """Independent Gaussian sampler with its own generator."""

    def __init__(
        self, mean: np.ndarray, std: np.ndarray, bkd: Backend[Array], seed: int
    ) -> None:
        self._mean, self._std, self._bkd = mean, std, bkd
        self._rng = np.random.default_rng(seed)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._mean.shape[0]

    def sample(self, nsamples: int) -> Tuple[Array, Array]:
        draws = self._rng.standard_normal((self.nvars(), nsamples))
        points = self._mean[:, None] + self._std[:, None] * draws
        return self._bkd.asarray(points), self._bkd.full((nsamples,), 1.0 / nsamples)

    def reset(self) -> None:
        pass


class _Counting:
    """A linear-then-sine function of the inputs that counts its calls."""

    def __init__(self, mat: np.ndarray, bkd: Backend[Array]) -> None:
        self.ncalls = 0
        self._mat, self._bkd = bkd.asarray(mat), bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return int(self._mat.shape[1])

    def nqoi(self) -> int:
        return int(self._mat.shape[0])

    def __call__(self, samples: Array, /) -> Array:
        self.ncalls += 1
        return self._bkd.dot(self._mat, samples)


class TestQuadratureMoments:
    """Inputs N(mu, diag(std^2)) in 3 dims; G = A x; targets x[0:2], B x."""

    def _setup(self, bkd: Backend[Array]) -> None:
        rng = np.random.default_rng(5)
        self._mu = np.array([0.3, -0.2, 0.5])
        self._std = np.array([0.5, 1.0, 0.8])
        self._amat = rng.normal(size=(4, 3))
        self._bmat = rng.normal(size=(2, 3))
        self._bkd = bkd

    def _evaluator(self, bkd: Backend[Array]) -> SeparateFunctions[Array]:
        amat, bmat = bkd.asarray(self._amat), bkd.asarray(self._bmat)
        return SeparateFunctions(
            FunctionFromCallable(4, 3, lambda x: bkd.dot(amat, x), bkd),
            [
                InputTarget(3, bkd, rows=[0, 1]),
                FunctionFromCallable(2, 3, lambda x: bkd.dot(bmat, x), bkd),
            ],
        )

    def _univariate_rules(self, bkd: Backend[Array]) -> list:
        marginals = [GaussianMarginal(m, s, bkd) for m, s in zip(self._mu, self._std)]
        return [
            (lambda n, marginal=marginal: gauss_quadrature_rule(marginal, n, bkd))
            for marginal in marginals
        ]

    def _tensor_rule(self, bkd: Backend[Array]) -> TensorProductQuadratureRule[Array]:
        return TensorProductQuadratureRule(bkd, self._univariate_rules(bkd), [2, 2, 2])

    def _exact(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mean and covariance of (x[0:2], B x, A x)."""
        stacked_map = np.vstack([np.eye(3)[:2], self._bmat, self._amat])
        cov = stacked_map @ np.diag(self._std**2) @ stacked_map.T
        return stacked_map @ self._mu[:, None], cov

    def test_degree_two_tensor_rule_is_exact(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        blocks = QuadratureMoments(
            self._tensor_rule(bkd), self._evaluator(bkd)
        ).blocks()
        mean, cov = self._exact()
        bkd.assert_allclose(blocks.mean(), bkd.asarray(mean), rtol=1e-12, atol=1e-14)
        bkd.assert_allclose(
            blocks.covariance(), bkd.asarray(cov), rtol=1e-12, atol=1e-14
        )
        bkd.assert_allclose(
            blocks.obs_covariance(),
            bkd.asarray(self._amat @ np.diag(self._std**2) @ self._amat.T),
            rtol=1e-12,
        )

    def test_at_level_matches_fixed_rule(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        parameterized = ParameterizedTensorProductQuadratureRule(
            bkd, self._univariate_rules(bkd), LinearGrowthRule(scale=1, shift=1)
        )
        rule = AtLevel(parameterized, 1)
        assert isinstance(rule, WeightedRuleProtocol)
        a = QuadratureMoments(rule, self._evaluator(bkd)).blocks()
        b = QuadratureMoments(self._tensor_rule(bkd), self._evaluator(bkd)).blocks()
        bkd.assert_allclose(a.covariance(), b.covariance(), rtol=1e-12, atol=1e-14)

    def test_batches_equal_one_pass(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        rule = self._tensor_rule(bkd)
        whole = QuadratureMoments(rule, self._evaluator(bkd)).blocks()
        batched = QuadratureMoments(rule, self._evaluator(bkd), batch_size=3).blocks()
        bkd.assert_allclose(batched.mean(), whole.mean(), rtol=1e-12, atol=1e-14)
        bkd.assert_allclose(
            batched.covariance(), whole.covariance(), rtol=1e-12, atol=1e-14
        )

    def test_split_function_runs_once_per_batch(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        shared = _Counting(np.vstack([self._amat, self._bmat]), bkd)
        split = SplitFunction(shared, [0, 1, 2, 3], [[4, 5]])
        source = QuadratureMoments(self._tensor_rule(bkd), split, batch_size=3)
        source.blocks()
        assert shared.ncalls == 3  # 8 points in batches of 3
        source.blocks()
        assert shared.ncalls == 3  # the blocks are cached

    def test_cached_matches_quadrature(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        rule = self._tensor_rule(bkd)
        points, weights = rule()
        outputs = self._evaluator(bkd).evaluate(points)
        cached = CachedMoments(outputs, bkd.reshape(weights, (1, -1)), bkd)
        assert isinstance(cached, MomentSourceProtocol)
        direct = QuadratureMoments(rule, self._evaluator(bkd)).blocks()
        bkd.assert_allclose(
            cached.blocks().covariance(), direct.covariance(), rtol=1e-12, atol=1e-14
        )

    def test_custom_accumulator(self, bkd: Backend[Array]) -> None:
        """An unbiased accumulator scales the covariance by 1/(1 - sum w^2)."""
        self._setup(bkd)
        rule = SampledRule(_GaussianSampler(self._mu, self._std, bkd, 0), 50)
        plain = QuadratureMoments(rule, self._evaluator(bkd)).blocks()
        unbiased = QuadratureMoments(
            rule, self._evaluator(bkd), lambda: UnbiasedMCAccumulator(bkd)
        ).blocks()
        bkd.assert_allclose(
            unbiased.covariance(), plain.covariance() * 50.0 / 49.0, rtol=1e-12
        )

    def test_sampled_rule_is_fixed(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        rule = SampledRule(_GaussianSampler(self._mu, self._std, bkd, 0), 10)
        first, second = rule(), rule()
        bkd.assert_allclose(first[0], second[0])
        assert first[1].shape == (10,)

    def test_monte_carlo_convergence(self, bkd: Backend[Array]) -> None:
        """Sampled blocks converge to the exact ones at the N^(-1/2) rate."""
        self._setup(bkd)
        _, cov = self._exact()
        sizes, nreps, rmse = [1000, 10000, 100000], 50, []
        for nsamples in sizes:
            errors = []
            for rep in range(nreps):
                sampler = _GaussianSampler(self._mu, self._std, bkd, 1000 * rep + 7)
                blocks = QuadratureMoments(
                    SampledRule(sampler, nsamples), self._evaluator(bkd)
                ).blocks()
                errors.append(bkd.to_numpy(blocks.covariance()) - cov)
            rmse.append(np.sqrt(np.mean(np.array(errors) ** 2)))
        slope = np.polyfit(np.log(sizes), np.log(rmse), 1)[0]
        assert abs(slope + 0.5) < 0.1, f"MC convergence slope {slope:.3f}"
        assert rmse[-1] < 1e-2 * np.max(np.abs(cov))

    def test_rejects_bad_inputs(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        rule, evaluator = self._tensor_rule(bkd), self._evaluator(bkd)
        with pytest.raises(TypeError):
            QuadratureMoments(object(), evaluator)
        with pytest.raises(TypeError):
            QuadratureMoments(rule, object())
        with pytest.raises(ValueError):
            QuadratureMoments(rule, evaluator, batch_size=0)
        wide = SeparateFunctions(InputTarget(4, bkd), [InputTarget(4, bkd)])
        with pytest.raises(ValueError):
            QuadratureMoments(rule, wide)
        with pytest.raises(TypeError):
            QuadratureMoments(rule, evaluator, lambda: object()).blocks()
        with pytest.raises(TypeError):
            AtLevel(object(), 1)
        with pytest.raises(ValueError):
            SampledRule(_GaussianSampler(self._mu, self._std, bkd, 0), 0)
