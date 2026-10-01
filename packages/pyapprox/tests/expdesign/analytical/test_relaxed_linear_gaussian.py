"""Tests for the relaxed-weight linear-Gaussian references.

The existing conjugate classes are special cases: they take a noise
covariance and invert it, so they cover w = 1, binary designs (selected
rows) and, for diagonal noise, any 0 < w <= 1 through the noise
diag(sigma^2 / w). Zero weights with all rows kept and correlated noise at
interior weights are checked against the explicit square-root construction
of the blended observation instead.
"""

import numpy as np
import pytest
from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDExpectedPushforwardKLDivergence,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance,
    relaxed_linear_target_covariance,
    relaxed_linear_target_eig,
    relaxed_lognormal_expected_variance,
)
from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.util.backends.protocols import Array, Backend


class TestRelaxedLinearGaussian:
    """Stacked inputs x = (m, a, b); observations H x + e; QoI exp(F x)."""

    def _setup(self, correlated: bool = False) -> None:
        rng = np.random.default_rng(7)
        nm, na, nb, nobs, nqoi = 3, 2, 1, 5, 2
        self._nm = nm
        nx = nm + na + nb
        self._prior_cov = np.diag(
            np.concatenate(
                [
                    rng.uniform(0.1, 0.3, nm),
                    rng.uniform(0.1, 0.3, na),
                    rng.uniform(0.05, 0.1, nb),
                ]
            )
        )
        self._prior_mean = rng.normal(size=(nx, 1)) * 0.1
        self._obs_mat = np.hstack(
            [
                rng.normal(size=(nobs, nm)),
                rng.normal(size=(nobs, na)),
                np.zeros((nobs, nb)),
            ]
        )
        self._qoi_mat = np.hstack(
            [
                rng.normal(size=(nqoi, nm)) * 0.8,
                rng.normal(size=(nqoi, na)) * 0.5,
                rng.normal(size=(nqoi, nb)) * 0.5,
            ]
        )
        self._sig2 = 0.05 * np.ones(nobs)
        if correlated:
            ymat = rng.normal(size=(nobs, nobs))
            self._noise_cov = ymat @ ymat.T / nobs * 0.1 + 0.05 * np.eye(nobs)
        else:
            self._noise_cov = np.diag(self._sig2)

    def _param_mat(self) -> np.ndarray:
        nx = self._prior_cov.shape[0]
        return np.eye(nx)[: self._nm]

    def _sqrt_form_cov(self, target: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Independent reference: explicit blended observation with sqrt(w)."""
        d = np.diag(np.sqrt(w))
        s2 = np.diag(self._noise_cov)
        syy = self._obs_mat @ self._prior_cov @ self._obs_mat.T + self._noise_cov
        szz = d @ syy @ d + np.diag((1.0 - w) * s2)
        ctz = target @ self._prior_cov @ self._obs_mat.T @ d
        gain = np.linalg.solve(szz, ctz.T)
        cond: np.ndarray = target @ self._prior_cov @ target.T - ctz @ gain
        return cond

    def _cov(self, bkd: Backend[Array], target: np.ndarray, w: np.ndarray) -> Array:
        return relaxed_linear_target_covariance(
            bkd.asarray(target),
            bkd.asarray(self._obs_mat),
            bkd.asarray(self._prior_cov),
            bkd.asarray(self._noise_cov),
            bkd.asarray(w[:, None]),
            bkd,
        )

    def _eig(self, bkd: Backend[Array], w: np.ndarray) -> float:
        value = relaxed_linear_target_eig(
            bkd.asarray(self._qoi_mat),
            bkd.asarray(self._obs_mat),
            bkd.asarray(self._prior_cov),
            bkd.asarray(self._noise_cov),
            bkd.asarray(w[:, None]),
            bkd,
        )
        return float(bkd.to_numpy(value)[0, 0])

    def _evar(self, bkd: Backend[Array], w: np.ndarray) -> np.ndarray:
        value = relaxed_lognormal_expected_variance(
            bkd.asarray(self._qoi_mat),
            bkd.asarray(self._obs_mat),
            bkd.asarray(self._prior_mean),
            bkd.asarray(self._prior_cov),
            bkd.asarray(self._noise_cov),
            bkd.asarray(w[:, None]),
            bkd,
        )
        return bkd.to_numpy(value)[:, 0]

    def _existing_eig(
        self, bkd: Backend[Array], obs_mat: np.ndarray, noise_cov: np.ndarray
    ) -> float:
        kl = ConjugateGaussianOEDExpectedPushforwardKLDivergence(
            bkd.asarray(self._prior_mean),
            bkd.asarray(self._prior_cov),
            bkd.asarray(self._qoi_mat),
            bkd,
        )
        kl.set_observation_matrix(bkd.asarray(obs_mat))
        kl.set_noise_covariance(bkd.asarray(noise_cov))
        return float(kl.value())

    def _existing_evar(
        self, bkd: Backend[Array], obs_mat: np.ndarray, noise_cov: np.ndarray
    ) -> np.ndarray:
        values = []
        for ii in range(self._qoi_mat.shape[0]):
            util = ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance(
                bkd.asarray(self._prior_mean),
                bkd.asarray(self._prior_cov),
                bkd.asarray(self._qoi_mat[ii : ii + 1]),
                bkd,
            )
            util.set_observation_matrix(bkd.asarray(obs_mat))
            util.set_noise_covariance(bkd.asarray(noise_cov))
            values.append(float(util.value()))
        return np.array(values)

    def _existing_param_cov(
        self, bkd: Backend[Array], obs_mat: np.ndarray, noise_cov: np.ndarray
    ) -> np.ndarray:
        post = DenseGaussianConjugatePosterior(
            bkd.asarray(obs_mat),
            bkd.asarray(self._prior_mean),
            bkd.asarray(self._prior_cov),
            bkd.asarray(noise_cov),
            bkd,
        )
        post.compute(bkd.ones((obs_mat.shape[0], 1)))
        cov = bkd.to_numpy(post.posterior_covariance())
        return cov[: self._nm, : self._nm]

    @pytest.mark.parametrize("case", ["one", "interior", "interior_small"])
    def test_matches_existing_classes_diagonal_noise(
        self, bkd: Backend[Array], case: str
    ) -> None:
        """w = 1 and interior w via noise sigma^2 / w (diagonal noise)."""
        self._setup()
        w = {
            "one": np.ones(5),
            "interior": np.array([0.68, 0.77, 0.18, 0.59, 0.56]),
            "interior_small": np.array([1.0, 0.3, 0.05, 1.0, 0.6]),
        }[case]
        noise = np.diag(self._sig2 / w)
        bkd.assert_allclose(
            bkd.asarray([self._eig(bkd, w)]),
            bkd.asarray([self._existing_eig(bkd, self._obs_mat, noise)]),
            rtol=1e-10,
        )
        bkd.assert_allclose(
            bkd.asarray(self._evar(bkd, w)),
            bkd.asarray(self._existing_evar(bkd, self._obs_mat, noise)),
            rtol=1e-10,
        )
        bkd.assert_allclose(
            self._cov(bkd, self._param_mat(), w),
            bkd.asarray(self._existing_param_cov(bkd, self._obs_mat, noise)),
            rtol=1e-10,
        )

    @pytest.mark.parametrize("correlated", [False, True])
    def test_binary_equals_selected_rows(
        self, bkd: Backend[Array], correlated: bool
    ) -> None:
        """Zero weights drop rows exactly, also with correlated noise."""
        self._setup(correlated=correlated)
        w = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        sel = [0, 2, 4]
        noise = self._noise_cov[np.ix_(sel, sel)]
        obs = self._obs_mat[sel]
        bkd.assert_allclose(
            bkd.asarray([self._eig(bkd, w)]),
            bkd.asarray([self._existing_eig(bkd, obs, noise)]),
            rtol=1e-10,
        )
        bkd.assert_allclose(
            bkd.asarray(self._evar(bkd, w)),
            bkd.asarray(self._existing_evar(bkd, obs, noise)),
            rtol=1e-10,
        )
        bkd.assert_allclose(
            self._cov(bkd, self._param_mat(), w),
            bkd.asarray(self._existing_param_cov(bkd, obs, noise)),
            rtol=1e-10,
        )

    def test_correlated_interior_matches_sqrt_form(self, bkd: Backend[Array]) -> None:
        """Correlated noise at interior and zero weights: sqrt-form reference."""
        self._setup(correlated=True)
        for w in [np.array([0.68, 0.0, 0.18, 0.59, 1.0]), np.full(5, 0.4)]:
            for target in [self._param_mat(), self._qoi_mat]:
                bkd.assert_allclose(
                    self._cov(bkd, target, w),
                    bkd.asarray(self._sqrt_form_cov(target, w)),
                    rtol=1e-10,
                )

    def test_zero_design_returns_prior(self, bkd: Backend[Array]) -> None:
        self._setup(correlated=True)
        w = np.zeros(5)
        target = self._param_mat()
        bkd.assert_allclose(
            self._cov(bkd, target, w),
            bkd.asarray(target @ self._prior_cov @ target.T),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            bkd.asarray([self._eig(bkd, w)]), bkd.asarray([0.0]), atol=1e-12
        )

    def test_monotone_in_weights(self, bkd: Backend[Array]) -> None:
        """More weight never increases uncertainty (Loewner order)."""
        self._setup(correlated=True)
        w_low = np.array([0.2, 0.0, 0.5, 0.3, 0.6])
        w_high = np.array([0.6, 0.4, 0.5, 0.9, 1.0])
        target = self._param_mat()
        diff = bkd.to_numpy(
            self._cov(bkd, target, w_low) - self._cov(bkd, target, w_high)
        )
        assert np.linalg.eigvalsh((diff + diff.T) / 2).min() > -1e-12
        assert self._eig(bkd, w_high) >= self._eig(bkd, w_low)

    def _mc_estimates(self, w: np.ndarray, nsamples: int, seed: int) -> np.ndarray:
        """Sample-based estimates from the simulated blended observation.

        Returns the flattened log-QoI conditional covariance, the goal EIG
        and the expected posterior variance of each QoI, all computed from
        samples of (x, e, eps) without the closed forms under test.
        """
        rng = np.random.default_rng(seed)
        nobs = self._obs_mat.shape[0]
        x = rng.multivariate_normal(
            self._prior_mean[:, 0], self._prior_cov, size=nsamples
        )
        e = rng.multivariate_normal(np.zeros(nobs), self._noise_cov, size=nsamples)
        eps = rng.standard_normal((nsamples, nobs))
        s2 = np.diag(self._noise_cov)
        z = np.sqrt(w) * (x @ self._obs_mat.T + e) + np.sqrt((1.0 - w) * s2) * eps
        log_q = x @ self._qoi_mat.T
        nqoi = log_q.shape[1]
        joint = np.cov(np.hstack([log_q, z]).T)
        c_ll, c_lz, c_zz = joint[:nqoi, :nqoi], joint[:nqoi, nqoi:], joint[nqoi:, nqoi:]
        cond = c_ll - c_lz @ np.linalg.solve(c_zz, c_lz.T)
        eig = 0.5 * (np.linalg.slogdet(c_ll)[1] - np.linalg.slogdet(cond)[1])
        # E[q | z] = exp(mu(z) + v / 2), with mu(z) the linear regression of
        # log q on z; average the squared error of q about it.
        gain = np.linalg.solve(c_zz, c_lz.T)
        mu = log_q.mean(0) + (z - z.mean(0)) @ gain
        cond_mean_q = np.exp(mu + 0.5 * np.diag(cond))
        evar = np.mean((np.exp(log_q) - cond_mean_q) ** 2, axis=0)
        return np.concatenate([cond.ravel(), [eig], evar])

    def test_monte_carlo_convergence_correlated(self, bkd: Backend[Array]) -> None:
        """Sample estimates converge to the oracle at the N^(-1/2) rate.

        Correlated noise and a zero weight: the regime where no existing
        class applies.
        """
        self._setup(correlated=True)
        w = np.array([0.68, 0.0, 0.18, 0.59, 1.0])
        exact = np.concatenate(
            [
                bkd.to_numpy(self._cov(bkd, self._qoi_mat, w)).ravel(),
                [self._eig(bkd, w)],
                self._evar(bkd, w),
            ]
        )
        sizes = [1000, 10000, 100000]
        nreps = 50
        rmse = []
        for nsamples in sizes:
            errors = np.array(
                [
                    self._mc_estimates(w, nsamples, 1000 * rep + nsamples) - exact
                    for rep in range(nreps)
                ]
            )
            rmse.append(np.sqrt(np.mean(errors**2)))
        slope = np.polyfit(np.log(sizes), np.log(rmse), 1)[0]
        assert abs(slope + 0.5) < 0.1, f"MC convergence slope {slope:.3f}"
        assert rmse[-1] < 1e-2 * np.max(np.abs(exact))

    def test_rejects_1d_weights(self, bkd: Backend[Array]) -> None:
        self._setup()
        with pytest.raises(ValueError):
            relaxed_linear_target_covariance(
                bkd.asarray(self._param_mat()),
                bkd.asarray(self._obs_mat),
                bkd.asarray(self._prior_cov),
                bkd.asarray(self._noise_cov),
                bkd.asarray(np.ones(5)),
                bkd,
            )
