"""Tests for JointGaussian.observe and LinearGaussianObservation.

The square-root-free quantities are checked against the explicit
construction of the blended observation, z = sqrt(w) y + sqrt(nu) eps, and
against the relaxed-weight oracles of the analytical module.
"""

import numpy as np
import pytest

from pyapprox.expdesign.analytical import (
    relaxed_linear_target_covariance,
    relaxed_linear_target_eig,
)
from pyapprox.inverse.joint_gaussian import JointGaussian, LinearGaussianObservation
from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import DenseBlocks
from pyapprox.util.backends.protocols import Array, Backend


class TestObservation:
    """Input in R^3, prediction target B (2x3), 4 observations, correlated noise.

    Weights include a zero and a one; nu = (1 - w) s^2 with s^2 the noise
    variances, as the blended relaxation sets it.
    """

    def _setup(self, bkd: Backend[Array]) -> None:
        rng = np.random.default_rng(12)
        root = rng.normal(size=(3, 3))
        self._prior_cov = root @ root.T + 0.2 * np.eye(3)
        self._prior_mean = rng.normal(size=(3, 1))
        self._amat = rng.normal(size=(4, 3))
        self._bmat = rng.normal(size=(2, 3))
        noise_root = rng.normal(size=(4, 4))
        self._noise_cov = 0.05 * noise_root @ noise_root.T + 0.1 * np.eye(4)
        self._w = np.array([[0.7], [0.0], [1.0], [0.3]])
        self._nu = (1.0 - self._w) * np.diag(self._noise_cov)[:, None]
        self._data = rng.normal(size=(4, 1))

    def _joint(self, bkd: Backend[Array]) -> JointGaussian[Array]:
        blocks = DenseBlocks.from_linear_model(
            bkd.asarray(self._amat),
            bkd.asarray(self._prior_mean),
            bkd.asarray(self._prior_cov),
            [bkd.asarray(self._bmat)],
            bkd,
        )
        noise = DenseCholeskyCovarianceOperator(bkd.asarray(self._noise_cov), bkd)
        return JointGaussian(blocks, noise)

    def _explicit(self) -> dict[str, np.ndarray]:
        """The sqrt(w) construction: Gamma_zz = D Gamma_yy D + Lambda."""
        d = np.diag(np.sqrt(self._w[:, 0]))
        syy = self._amat @ self._prior_cov @ self._amat.T + self._noise_cov
        szz = d @ syy @ d + np.diag(self._nu[:, 0])
        cty = self._bmat @ self._prior_cov @ self._amat.T
        ctz = cty @ d
        mu_t = self._bmat @ self._prior_mean
        mu_y = self._amat @ self._prior_mean
        gain = np.linalg.solve(szz, ctz.T).T
        return {
            "logdet": np.array([np.linalg.slogdet(szz)[1]]),
            "mean": mu_t + gain @ d @ (self._data - mu_y),
            "cov": self._bmat @ self._prior_cov @ self._bmat.T - gain @ ctz.T,
        }

    def _observe(self, bkd: Backend[Array]) -> LinearGaussianObservation[Array]:
        return self._joint(bkd).observe(bkd.asarray(self._w), bkd.asarray(self._nu), 0)

    def test_matches_explicit_construction(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        obs = self._observe(bkd)
        ref = self._explicit()
        bkd.assert_allclose(obs.covariance(), bkd.asarray(ref["cov"]), rtol=1e-10)
        bkd.assert_allclose(obs.logdet_zz(), bkd.asarray(ref["logdet"]), rtol=1e-10)
        bkd.assert_allclose(
            obs.mean(bkd.asarray(self._data)), bkd.asarray(ref["mean"]), rtol=1e-10
        )

    def test_matches_relaxed_oracles(self, bkd: Backend[Array]) -> None:
        """Covariance and EIG = (logdet_zz - logdet_zz_given_t) / 2."""
        self._setup(bkd)
        obs = self._observe(bkd)
        args = (
            bkd.asarray(self._bmat),
            bkd.asarray(self._amat),
            bkd.asarray(self._prior_cov),
            bkd.asarray(self._noise_cov),
            bkd.asarray(self._w),
            bkd,
        )
        bkd.assert_allclose(
            obs.covariance(), relaxed_linear_target_covariance(*args), rtol=1e-10
        )
        eig = 0.5 * (obs.logdet_zz() - obs.logdet_zz_given_t())
        bkd.assert_allclose(eig, relaxed_linear_target_eig(*args)[0], rtol=1e-10)

    def test_full_weights_equal_condition(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        joint = self._joint(bkd)
        obs = joint.observe(bkd.ones((4, 1)), bkd.zeros((4, 1)), 0)
        mean, cov = joint.condition(bkd.asarray(self._data), 0)
        bkd.assert_allclose(obs.covariance(), cov, rtol=1e-10)
        bkd.assert_allclose(obs.mean(bkd.asarray(self._data)), mean, rtol=1e-10)

    def test_zero_weight_data_is_ignored(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        obs = self._observe(bkd)
        changed = self._data.copy()
        changed[1] = 1e6  # the sensor with w = 0
        bkd.assert_allclose(
            obs.mean(bkd.asarray(changed)),
            obs.mean(bkd.asarray(self._data)),
            rtol=1e-12,
        )

    def test_rejects_bad_inputs(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        joint = self._joint(bkd)
        w, nu = bkd.asarray(self._w), bkd.asarray(self._nu)
        with pytest.raises(ValueError):
            joint.observe(w[:3], nu, 0)
        with pytest.raises(ValueError):
            joint.observe(-w, nu, 0)
        with pytest.raises(ValueError):
            joint.observe(w, -nu - 1.0, 0)
        with pytest.raises(ValueError, match="singular"):
            joint.observe(w, bkd.zeros((4, 1)), 0)
        with pytest.raises(ValueError):
            joint.observe(w, nu, 1)
        with pytest.raises(ValueError):
            self._observe(bkd).mean(bkd.ones((3, 1)))
