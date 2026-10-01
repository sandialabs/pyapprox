"""Tests for the exact moment-Gaussian blocks of a lognormal QoI."""

import numpy as np
import pytest
from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance,
    LogNormalMGBlocks,
    lognormal_goal_mg_blocks,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestLogNormalMGBlocks:
    """Blocks of (exp(F x), H x) for Gaussian x, nuisances stacked in x."""

    def _setup(self, prior_scale: float = 1.0) -> None:
        rng = np.random.default_rng(7)
        nm, na, nb, nobs, nqoi = 3, 2, 1, 5, 2
        nx = nm + na + nb
        variances = np.concatenate(
            [
                rng.uniform(0.1, 0.3, nm),
                rng.uniform(0.1, 0.3, na),
                rng.uniform(0.05, 0.1, nb),
            ]
        )
        self._prior_cov = np.diag(variances) * prior_scale
        self._prior_mean = rng.normal(size=(nx, 1)) * 0.1
        amat = rng.normal(size=(nobs, nm))
        ba = rng.normal(size=(nobs, na))
        self._obs_mat = np.hstack([amat, ba, np.zeros((nobs, nb))])
        rmat = rng.normal(size=(nqoi, nm)) * 0.8
        sa = rng.normal(size=(nqoi, na)) * 0.5
        sb = rng.normal(size=(nqoi, nb)) * 0.5
        self._qoi_mat = np.hstack([rmat, sa, sb])
        self._noise_cov = 0.05 * np.eye(nobs)

    def _blocks(self, bkd: Backend[Array]) -> LogNormalMGBlocks[Array]:
        return lognormal_goal_mg_blocks(
            bkd.asarray(self._obs_mat),
            bkd.asarray(self._qoi_mat),
            bkd.asarray(self._prior_mean),
            bkd.asarray(self._prior_cov),
            bkd,
        )

    def _mg_goal_a(self, bkd: Backend[Array]) -> float:
        """MG trace of the QoI covariance given all the data."""
        blocks = self._blocks(bkd)
        syy = blocks.obs_cov + bkd.asarray(self._noise_cov)
        cond = blocks.qoi_cov - bkd.dot(
            blocks.qoi_obs_cov, bkd.solve(syy, blocks.qoi_obs_cov.T)
        )
        return float(bkd.trace(cond))

    def _true_goal_a(self, bkd: Backend[Array]) -> float:
        """Sum over QoIs of the exact expected posterior variance."""
        total = 0.0
        for ii in range(self._qoi_mat.shape[0]):
            utility = ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance(
                bkd.asarray(self._prior_mean),
                bkd.asarray(self._prior_cov),
                bkd.asarray(self._qoi_mat[ii : ii + 1]),
                bkd,
            )
            utility.set_observation_matrix(bkd.asarray(self._obs_mat))
            utility.set_noise_covariance(bkd.asarray(self._noise_cov))
            total += float(utility.value())
        return total

    def _mc_blocks(self, nsamples: int, seed: int) -> np.ndarray:
        """MC estimate of all blocks, flattened: E[q], Cov(q), Cov(q, g)."""
        rng = np.random.default_rng(seed)
        x = rng.multivariate_normal(
            self._prior_mean[:, 0], self._prior_cov, size=nsamples
        )
        q = np.exp(x @ self._qoi_mat.T)
        g = x @ self._obs_mat.T
        nqoi = q.shape[1]
        joint = np.cov(np.hstack([q, g]).T)
        return np.concatenate(
            [q.mean(0), joint[:nqoi, :nqoi].ravel(), joint[:nqoi, nqoi:].ravel()]
        )

    def test_blocks_monte_carlo_convergence(self, bkd: Backend[Array]) -> None:
        """MC estimates converge to the blocks at the N^(-1/2) rate."""
        self._setup()
        blocks = self._blocks(bkd)
        exact = np.concatenate(
            [
                bkd.to_numpy(blocks.qoi_mean).ravel(),
                bkd.to_numpy(blocks.qoi_cov).ravel(),
                bkd.to_numpy(blocks.qoi_obs_cov).ravel(),
            ]
        )
        sizes = [1000, 10000, 100000]
        nreps = 50
        rmse = []
        for nsamples in sizes:
            errors = np.array(
                [
                    self._mc_blocks(nsamples, 1000 * rep + nsamples) - exact
                    for rep in range(nreps)
                ]
            )
            rmse.append(np.sqrt(np.mean(errors**2)))
        slope = np.polyfit(np.log(sizes), np.log(rmse), 1)[0]
        # With 50 replications the fitted slope has a spread of about 0.03
        # across seeds, so 0.1 is a tight but robust band around -1/2.
        assert abs(slope + 0.5) < 0.1, f"MC convergence slope {slope:.3f}"
        # The error at the largest N is small relative to the block scale,
        # so a constant bias in the formulas would show up here.
        assert rmse[-1] < 1e-2 * np.max(np.abs(exact))

    def test_obs_blocks_are_exact_linear_gaussian(self, bkd: Backend[Array]) -> None:
        self._setup()
        blocks = self._blocks(bkd)
        bkd.assert_allclose(
            blocks.obs_cov,
            bkd.asarray(self._obs_mat @ self._prior_cov @ self._obs_mat.T),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            blocks.obs_mean,
            bkd.asarray(self._obs_mat @ self._prior_mean),
            rtol=1e-12,
        )

    def test_a_bound_holds(self, bkd: Backend[Array]) -> None:
        """MG goal-A bounds the exact expected posterior variance."""
        self._setup()
        assert self._mg_goal_a(bkd) >= self._true_goal_a(bkd)

    def test_small_variance_limit(self, bkd: Backend[Array]) -> None:
        """For a small prior the QoI is nearly Gaussian and the gap vanishes."""
        self._setup(prior_scale=1e-4)
        mg = self._mg_goal_a(bkd)
        true = self._true_goal_a(bkd)
        bkd.assert_allclose(bkd.asarray([mg]), bkd.asarray([true]), rtol=1e-3)

    def test_rejects_1d_mean(self, bkd: Backend[Array]) -> None:
        self._setup()
        with pytest.raises(ValueError):
            lognormal_goal_mg_blocks(
                bkd.asarray(self._obs_mat),
                bkd.asarray(self._qoi_mat),
                bkd.asarray(self._prior_mean[:, 0]),
                bkd.asarray(self._prior_cov),
                bkd,
            )

    def test_rejects_mismatched_qoi_mat(self, bkd: Backend[Array]) -> None:
        self._setup()
        with pytest.raises(ValueError):
            lognormal_goal_mg_blocks(
                bkd.asarray(self._obs_mat),
                bkd.asarray(self._qoi_mat[:, :-1]),
                bkd.asarray(self._prior_mean),
                bkd.asarray(self._prior_cov),
                bkd,
            )
