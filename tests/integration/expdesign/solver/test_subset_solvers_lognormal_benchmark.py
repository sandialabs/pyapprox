"""Subset searches on the linear-Gaussian benchmark with a lognormal QoI.

The benchmark gives exact values at any 0/1 design, so exhaustive search
over them is the true optimum. Two targets:

- the parameter ``m``, which is linear-Gaussian, so the moment-Gaussian
  objective is exact and must choose the same subsets with the same
  values;
- the QoI ``q = exp(F xi)``, where the moment-Gaussian A-criterion is a
  conservative bound on the true expected posterior variance; its
  choices are scored by their true values.
"""

from typing import Callable, Generic, Sequence

import pytest
from pyapprox.expdesign.design_space import (
    BinaryDesignSubsetObjective,
    ReevaluatingIncremental,
)
from pyapprox.expdesign.gaussian import AOptimal, BlendedObservation, DesignObjective
from pyapprox.expdesign.solver import (
    ExchangeSubsetSolver,
    ExhaustiveSubsetSolver,
    GreedySubsetSolver,
)
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox_benchmarks.expdesign import (
    LinearGaussianNuisanceLognormalBenchmark,
    build_linear_gaussian_nuisance_lognormal_benchmark,
)


class _ExactSubsetObjective(Generic[Array]):
    """A benchmark's exact value at the 0/1 design of a subset."""

    def __init__(
        self, exact: Callable[[Array], float], nobs: int, bkd: Backend[Array]
    ) -> None:
        self._exact = exact
        self._nobs = nobs
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def ncandidates(self) -> int:
        return self._nobs

    def value(self, subset: Sequence[int]) -> float:
        weights = self._bkd.zeros((self._nobs, 1))
        for ii in subset:
            weights[ii, 0] = 1.0
        return self._exact(weights)


class TestSubsetSolversOnLognormalBenchmark:
    """Eight candidate observations, two parameters, both nuisances."""

    def _benchmark(
        self, bkd: Backend[Array]
    ) -> LinearGaussianNuisanceLognormalBenchmark[Array]:
        return build_linear_gaussian_nuisance_lognormal_benchmark(
            nobs=8, nparams=2, nobs_nuisance=1, npred_nuisance=1, nqoi=2, bkd=bkd
        )

    def _mg_objective(
        self,
        bench: LinearGaussianNuisanceLognormalBenchmark[Array],
        blocks: DenseBlocks[Array],
    ) -> BinaryDesignSubsetObjective[Array]:
        bkd = bench.bkd()
        noise = DenseCholeskyCovarianceOperator(bench.noise_covariance(), bkd)
        objective = DesignObjective(
            JointGaussian(blocks, noise),
            BlendedObservation.from_noise(noise),
            AOptimal(),
            0,
        )
        return BinaryDesignSubsetObjective(objective)

    def _param_blocks(
        self, bench: LinearGaussianNuisanceLognormalBenchmark[Array]
    ) -> DenseBlocks[Array]:
        bkd = bench.bkd()
        nvars = bench.obs_matrix().shape[1]
        return DenseBlocks.from_linear_model(
            bench.obs_matrix(),
            bench.prior_mean(),
            bench.prior_covariance(),
            [bkd.eye(nvars)[: bench.nparams()]],
            bkd,
        )

    def _goal_blocks(
        self, bench: LinearGaussianNuisanceLognormalBenchmark[Array]
    ) -> DenseBlocks[Array]:
        bkd = bench.bkd()
        mg = bench.exact_mg_blocks()
        mean = bkd.vstack([mg.qoi_mean, mg.obs_mean])
        cov = bkd.vstack(
            [
                bkd.hstack([mg.qoi_cov, mg.qoi_obs_cov]),
                bkd.hstack([mg.qoi_obs_cov.T, mg.obs_cov]),
            ]
        )
        nqoi = mg.qoi_mean.shape[0]
        return DenseBlocks(mean, cov, (nqoi,), mg.obs_mean.shape[0], bkd)

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_parameter_target_is_exact(self, bkd: Backend[Array], k: int) -> None:
        bench = self._benchmark(bkd)
        mg = self._mg_objective(bench, self._param_blocks(bench))
        exact = _ExactSubsetObjective(
            lambda w: bkd.to_float(
                bkd.trace(bench.exact_param_posterior_covariance(w))
            ),
            8,
            bkd,
        )
        mg_result = ExhaustiveSubsetSolver(mg).solve(k)
        exact_result = ExhaustiveSubsetSolver(exact).solve(k)
        assert mg_result.subset == exact_result.subset
        bkd.assert_allclose(
            bkd.asarray([mg_result.value]),
            bkd.asarray([exact_result.value]),
            rtol=1e-10,
        )

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_goal_target_scored_by_truth(self, bkd: Backend[Array], k: int) -> None:
        bench = self._benchmark(bkd)
        mg = self._mg_objective(bench, self._goal_blocks(bench))
        truth = _ExactSubsetObjective(
            lambda w: bkd.to_float(
                bkd.sum(bench.exact_goal_expected_posterior_variance(w))
            ),
            8,
            bkd,
        )
        best = ExhaustiveSubsetSolver(truth).solve(k)
        mg_best = ExhaustiveSubsetSolver(mg).solve(k)
        incremental = ReevaluatingIncremental(mg)
        greedy = GreedySubsetSolver(incremental).solve(k)
        exchanged = ExchangeSubsetSolver(incremental).solve(greedy.subset)
        # The criterion bounds the truth: a linear estimator cannot beat
        # the conditional mean.
        assert mg_best.value >= truth.value(mg_best.subset)
        assert exchanged.value <= greedy.value
        assert exchanged.value >= mg_best.value
        # Efficiency of each moment-Gaussian design under the truth.
        for subset in (mg_best.subset, greedy.subset, exchanged.subset):
            efficiency = best.value / truth.value(subset)
            assert 0.95 <= efficiency <= 1.0 + 1e-12
