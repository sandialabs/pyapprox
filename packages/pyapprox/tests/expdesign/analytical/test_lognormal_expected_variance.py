"""Tests for the expected posterior variance of a lognormal QoI.

The model is linear-Gaussian in the stacked inputs x = (m, a, b): a
parameter m, an observation nuisance a and a prediction nuisance b. The
observations are y = H x + e with H = [A, B_a, 0], and the QoI is
q = exp(F x) with F = [R, S_a, S_b]. Stacking the nuisances into the
parameter vector marginalizes them exactly.
"""

import numpy as np
from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
    ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance,
)
from pyapprox.util.backends.protocols import Backend


class TestLogNormalExpectedVariance:
    """Expected posterior variance of q = exp(F x) with nuisances."""

    def _setup(self, nuisance_scale: float = 1.0) -> None:
        rng = np.random.default_rng(7)
        nm, na, nb, nobs = 3, 2, 1, 5
        self._nm, self._na, self._nb = nm, na, nb
        pm = np.diag(rng.uniform(0.1, 0.3, nm))
        pa = np.diag(rng.uniform(0.1, 0.3, na)) * nuisance_scale
        pb = np.diag(rng.uniform(0.05, 0.1, nb)) * nuisance_scale
        nx = nm + na + nb
        self._prior_cov = np.zeros((nx, nx))
        self._prior_cov[:nm, :nm] = pm
        self._prior_cov[nm : nm + na, nm : nm + na] = pa
        self._prior_cov[nm + na :, nm + na :] = pb
        self._prior_mean = np.zeros((nx, 1))
        amat = rng.normal(size=(nobs, nm))
        ba = rng.normal(size=(nobs, na))
        self._obs_mat = np.hstack([amat, ba, np.zeros((nobs, nb))])
        rmat = rng.normal(size=(1, nm)) * 0.8
        sa = rng.normal(size=(1, na)) * 0.5
        sb = rng.normal(size=(1, nb)) * 0.5
        self._qoi_mat = np.hstack([rmat, sa, sb])
        self._noise_cov = 0.05 * np.eye(nobs)

    def _utility(self, cls: type, bkd: Backend, prior_cov: np.ndarray) -> float:
        utility = cls(
            bkd.asarray(self._prior_mean),
            bkd.asarray(prior_cov),
            bkd.asarray(self._qoi_mat),
            bkd,
        )
        utility.set_observation_matrix(bkd.asarray(self._obs_mat))
        utility.set_noise_covariance(bkd.asarray(self._noise_cov))
        return float(utility.value())

    def _reference(self, prior_cov: np.ndarray) -> float:
        """Closed form from the joint Gaussian of (L, y), L = log q."""
        syy = self._obs_mat @ prior_cov @ self._obs_mat.T + self._noise_cov
        c_l = float((self._qoi_mat @ prior_cov @ self._qoi_mat.T)[0, 0])
        c_ly = self._qoi_mat @ prior_cov @ self._obs_mat.T
        v = c_l - float((c_ly @ np.linalg.solve(syy, c_ly.T))[0, 0])
        mean_l = float((self._qoi_mat @ self._prior_mean)[0, 0])
        return (np.exp(v) - 1.0) * np.exp(2.0 * mean_l + 2.0 * c_l - v)

    def test_matches_closed_form_with_nuisances(self, bkd: Backend) -> None:
        self._setup()
        value = self._utility(
            ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance,
            bkd,
            self._prior_cov,
        )
        bkd.assert_allclose(
            bkd.asarray([value]),
            bkd.asarray([self._reference(self._prior_cov)]),
            rtol=1e-12,
        )

    def test_no_nuisance_limit(self, bkd: Backend) -> None:
        """Vanishing nuisance variance recovers the no-nuisance model."""
        self._setup(nuisance_scale=1e-12)
        value = self._utility(
            ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance,
            bkd,
            self._prior_cov,
        )
        nm = self._nm
        no_nuisance = np.zeros_like(self._prior_cov)
        no_nuisance[:nm, :nm] = self._prior_cov[:nm, :nm]
        bkd.assert_allclose(
            bkd.asarray([value]),
            bkd.asarray([self._reference(no_nuisance)]),
            rtol=1e-8,
        )

    def test_marginalizing_differs_from_ignoring(self, bkd: Backend) -> None:
        """The nuisance-free model understates the expected variance.

        Ignoring the nuisances means the model y = A m + e, q = exp(R m);
        the class inverts the prior covariance, so the nuisance-free model
        is built on m alone rather than with zero nuisance blocks.
        """
        self._setup()
        nm = self._nm
        marginalized = self._utility(
            ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance,
            bkd,
            self._prior_cov,
        )
        ignoring = ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance(
            bkd.asarray(self._prior_mean[:nm]),
            bkd.asarray(self._prior_cov[:nm, :nm]),
            bkd.asarray(self._qoi_mat[:, :nm]),
            bkd,
        )
        ignoring.set_observation_matrix(bkd.asarray(self._obs_mat[:, :nm]))
        ignoring.set_noise_covariance(bkd.asarray(self._noise_cov))
        ignored = float(ignoring.value())
        nm_only = np.zeros_like(self._prior_cov)
        nm_only[:nm, :nm] = self._prior_cov[:nm, :nm]
        bkd.assert_allclose(
            bkd.asarray([ignored]),
            bkd.asarray([self._reference(nm_only)]),
            rtol=1e-12,
        )
        assert marginalized > ignored

    def test_jensen_bound_against_expected_stdev(self, bkd: Backend) -> None:
        """E[Var(q|y)] >= (E[Std(q|y)])^2."""
        self._setup()
        variance = self._utility(
            ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance,
            bkd,
            self._prior_cov,
        )
        stdev = self._utility(
            ConjugateGaussianOEDForLogNormalDataMeanQoIMeanStdDev,
            bkd,
            self._prior_cov,
        )
        assert variance >= stdev**2

    def _mc_expected_variance(self, nsamples: int, seed: int) -> float:
        """MC over the data of the exact lognormal posterior variance."""
        rng = np.random.default_rng(seed)
        x = rng.multivariate_normal(
            self._prior_mean[:, 0], self._prior_cov, size=nsamples
        )
        e = rng.multivariate_normal(
            np.zeros(self._noise_cov.shape[0]), self._noise_cov, size=nsamples
        )
        y = x @ self._obs_mat.T + e
        syy = self._obs_mat @ self._prior_cov @ self._obs_mat.T + self._noise_cov
        c_l = float((self._qoi_mat @ self._prior_cov @ self._qoi_mat.T)[0, 0])
        c_ly = self._qoi_mat @ self._prior_cov @ self._obs_mat.T
        gain = np.linalg.solve(syy, c_ly.T)[:, 0]
        v = c_l - float((c_ly @ gain)[0])
        mu = y @ gain
        return float(np.mean((np.exp(v) - 1.0) * np.exp(2.0 * mu + v)))

    def test_monte_carlo_convergence(self, bkd: Backend) -> None:
        """MC over (x, e) converges to the closed form at the N^(-1/2) rate."""
        self._setup()
        value = self._utility(
            ConjugateGaussianOEDForLogNormalDataMeanQoIMeanVariance,
            bkd,
            self._prior_cov,
        )
        sizes = [1000, 10000, 100000]
        nreps = 50
        rmse = []
        for nsamples in sizes:
            errors = np.array(
                [
                    self._mc_expected_variance(nsamples, 1000 * rep + nsamples)
                    - value
                    for rep in range(nreps)
                ]
            )
            rmse.append(np.sqrt(np.mean(errors**2)))
        slope = np.polyfit(np.log(sizes), np.log(rmse), 1)[0]
        # With 50 replications the fitted slope has a spread of about 0.03
        # across seeds, so 0.1 is a tight but robust band around -1/2.
        assert abs(slope + 0.5) < 0.1, f"MC convergence slope {slope:.3f}"
        # A constant bias in the formula would leave a floor in the error.
        assert rmse[-1] < 1e-2 * value
