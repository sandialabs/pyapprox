"""Moment sources on the linear-Gaussian benchmark with a lognormal QoI.

The benchmark's QoI is exp(F xi), so its blocks are not polynomial in the
inputs and no finite rule is exact. They are known in closed form,
which makes them a reference for both a high-order tensor Gauss-Hermite
rule and Monte Carlo.
"""

from typing import Tuple

import numpy as np

from pyapprox.interface.functions.joint import SeparateFunctions
from pyapprox.probability.moments import (
    CovarianceBlocksProtocol,
    QuadratureMoments,
    SampledRule,
)
from pyapprox.probability.univariate.gaussian import GaussianMarginal
from pyapprox.surrogates.quadrature import (
    TensorProductQuadratureRule,
    gauss_quadrature_rule,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox_benchmarks.expdesign import (
    LinearGaussianNuisanceLognormalBenchmark,
    build_linear_gaussian_nuisance_lognormal_benchmark,
)


class _GaussianSampler:
    """Zero-mean independent Gaussian sampler with its own generator."""

    def __init__(self, std: np.ndarray, bkd: Backend[Array], seed: int) -> None:
        self._std, self._bkd = std, bkd
        self._rng = np.random.default_rng(seed)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._std.shape[0]

    def sample(self, nsamples: int) -> Tuple[Array, Array]:
        points = self._std[:, None] * self._rng.standard_normal(
            (self.nvars(), nsamples)
        )
        return self._bkd.asarray(points), self._bkd.full((nsamples,), 1.0 / nsamples)

    def reset(self) -> None:
        pass


class TestMomentsOnLognormalBenchmark:
    """Two parameters, one observation and one prediction nuisance."""

    def _benchmark(
        self, bkd: Backend[Array]
    ) -> LinearGaussianNuisanceLognormalBenchmark[Array]:
        return build_linear_gaussian_nuisance_lognormal_benchmark(
            nobs=4, nparams=2, nobs_nuisance=1, npred_nuisance=1, nqoi=2, bkd=bkd
        )

    def _evaluator(
        self, bench: LinearGaussianNuisanceLognormalBenchmark[Array]
    ) -> SeparateFunctions[Array]:
        return SeparateFunctions(bench.obs_map(), [bench.qoi_map()])

    def _std(
        self, bench: LinearGaussianNuisanceLognormalBenchmark[Array]
    ) -> np.ndarray:
        cov = bench.bkd().to_numpy(bench.prior_covariance())
        return np.sqrt(np.diag(cov))

    def _flat(
        self, bkd: Backend[Array], blocks: CovarianceBlocksProtocol[Array]
    ) -> np.ndarray:
        return np.concatenate(
            [
                bkd.to_numpy(blocks.target_mean(0)).ravel(),
                bkd.to_numpy(blocks.target_covariance(0)).ravel(),
                bkd.to_numpy(blocks.target_obs_covariance(0)).ravel(),
                bkd.to_numpy(blocks.obs_covariance()).ravel(),
            ]
        )

    def _exact(
        self,
        bkd: Backend[Array],
        bench: LinearGaussianNuisanceLognormalBenchmark[Array],
    ) -> np.ndarray:
        exact = bench.exact_mg_blocks()
        return np.concatenate(
            [
                bkd.to_numpy(exact.qoi_mean).ravel(),
                bkd.to_numpy(exact.qoi_cov).ravel(),
                bkd.to_numpy(exact.qoi_obs_cov).ravel(),
                bkd.to_numpy(exact.obs_cov).ravel(),
            ]
        )

    def test_tensor_gauss_hermite_matches_exact(self, bkd: Backend[Array]) -> None:
        bench = self._benchmark(bkd)
        marginals = [GaussianMarginal(0.0, s, bkd) for s in self._std(bench)]
        rules = [
            (lambda n, m=marginal: gauss_quadrature_rule(m, n, bkd))
            for marginal in marginals
        ]
        # 16 points per dimension converge the exponential moments to
        # rounding; the rule's first point is its extreme corner, which
        # also guards the accumulator against centring on an outlier.
        rule = TensorProductQuadratureRule(bkd, rules, [16] * len(rules))
        for batch_size in (None, 97):
            blocks = QuadratureMoments(
                rule, self._evaluator(bench), batch_size=batch_size
            ).blocks()
            bkd.assert_allclose(
                bkd.asarray(self._flat(bkd, blocks)),
                bkd.asarray(self._exact(bkd, bench)),
                rtol=1e-12,
                atol=1e-14,
            )

    def test_monte_carlo_converges_to_exact(self, bkd: Backend[Array]) -> None:
        bench = self._benchmark(bkd)
        exact, std = self._exact(bkd, bench), self._std(bench)
        sizes, nreps, rmse = [1000, 10000, 100000], 50, []
        for nsamples in sizes:
            errors = []
            for rep in range(nreps):
                rule = SampledRule(_GaussianSampler(std, bkd, 1000 * rep + 3), nsamples)
                blocks = QuadratureMoments(rule, self._evaluator(bench)).blocks()
                errors.append(self._flat(bkd, blocks) - exact)
            rmse.append(np.sqrt(np.mean(np.array(errors) ** 2)))
        slope = np.polyfit(np.log(sizes), np.log(rmse), 1)[0]
        assert abs(slope + 0.5) < 0.1, f"MC convergence slope {slope:.3f}"
        assert rmse[-1] < 1e-2 * np.max(np.abs(exact))
