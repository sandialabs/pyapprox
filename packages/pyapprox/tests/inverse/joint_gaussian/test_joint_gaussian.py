"""Tests for JointGaussian conditioning and observation selection.

Linear maps of a correlated Gaussian input give exact blocks, so the
results are checked against independent references: the conjugate
posterior for the parameter, Gaussian-factor conditioning in canonical
form for a prediction, and the relaxed-weight oracle at binary weights for
a selected subset.
"""

import numpy as np
import pytest

from pyapprox.expdesign.analytical import relaxed_linear_target_covariance
from pyapprox.inverse.bayesnet import GaussianFactor
from pyapprox.inverse.conjugate.gaussian import DenseGaussianConjugatePosterior
from pyapprox.inverse.joint_gaussian import JointGaussian
from pyapprox.probability.covariance import (
    DenseCholeskyCovarianceOperator,
    DiagonalCovarianceOperator,
)
from pyapprox.probability.moments import DenseBlocks, EigenClip
from pyapprox.util.backends.protocols import Array, Backend


class TestJointGaussian:
    """Input xi in R^3; targets xi itself and B xi; data A xi + e, A 4x3."""

    def _setup(self, bkd: Backend[Array]) -> None:
        rng = np.random.default_rng(10)
        root = rng.normal(size=(3, 3))
        self._prior_cov = root @ root.T + 0.2 * np.eye(3)
        self._prior_mean = rng.normal(size=(3, 1))
        self._amat = rng.normal(size=(4, 3))
        self._bmat = rng.normal(size=(2, 3))
        noise_root = rng.normal(size=(4, 4))
        self._noise_cov = 0.05 * noise_root @ noise_root.T + 0.1 * np.eye(4)
        self._data = rng.normal(size=(4, 1))

    def _joint(
        self, bkd: Backend[Array], noise_cov: np.ndarray
    ) -> JointGaussian[Array]:
        blocks = DenseBlocks.from_linear_model(
            bkd.asarray(self._amat),
            bkd.asarray(self._prior_mean),
            bkd.asarray(self._prior_cov),
            [bkd.eye(3), bkd.asarray(self._bmat)],
            bkd,
        )
        return JointGaussian(
            blocks, DenseCholeskyCovarianceOperator(bkd.asarray(noise_cov), bkd)
        )

    def _posterior(
        self, bkd: Backend[Array], rows: list[int]
    ) -> DenseGaussianConjugatePosterior[Array]:
        post = DenseGaussianConjugatePosterior(
            bkd.asarray(self._amat[rows]),
            bkd.asarray(self._prior_mean),
            bkd.asarray(self._prior_cov),
            bkd.asarray(self._noise_cov[np.ix_(rows, rows)]),
            bkd,
        )
        post.compute(bkd.asarray(self._data[rows]))
        return post

    def test_parameter_matches_conjugate_posterior(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        mean, cov = self._joint(bkd, self._noise_cov).condition(
            bkd.asarray(self._data), 0
        )
        post = self._posterior(bkd, [0, 1, 2, 3])
        bkd.assert_allclose(mean, post.posterior_mean(), rtol=1e-10)
        bkd.assert_allclose(cov, post.posterior_covariance(), rtol=1e-10)

    def test_prediction_matches_canonical_form(self, bkd: Backend[Array]) -> None:
        """Condition (B xi, y) in precision form and compare."""
        self._setup(bkd)
        a, b, p = self._amat, self._bmat, self._prior_cov
        mu = self._prior_mean
        joint_mean = np.vstack([b @ mu, a @ mu])[:, 0]
        joint_cov = np.block(
            [[b @ p @ b.T, b @ p @ a.T], [a @ p @ b.T, a @ p @ a.T + self._noise_cov]]
        )
        factor = GaussianFactor.from_moments(
            bkd.asarray(joint_mean),
            bkd.asarray(joint_cov),
            var_ids=[0, 1],
            nvars_per_var=[2, 4],
            bkd=bkd,
        )
        ref_mean, ref_cov = factor.condition_vars(
            [1], bkd.asarray(self._data[:, 0])
        ).to_moments()
        mean, cov = self._joint(bkd, self._noise_cov).condition(
            bkd.asarray(self._data), 1
        )
        bkd.assert_allclose(mean[:, 0], bkd.reshape(ref_mean, (-1,)), rtol=1e-10)
        bkd.assert_allclose(cov, ref_cov, rtol=1e-10)

    def test_select_matches_posterior_on_rows(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        rows = [3, 0]
        selected = self._joint(bkd, self._noise_cov).select(rows)
        assert selected.nobs() == 2
        mean, cov = selected.condition(bkd.asarray(self._data[rows]), 0)
        post = self._posterior(bkd, rows)
        bkd.assert_allclose(mean, post.posterior_mean(), rtol=1e-10)
        bkd.assert_allclose(cov, post.posterior_covariance(), rtol=1e-10)

    def test_select_matches_relaxed_oracle(self, bkd: Backend[Array]) -> None:
        """Selecting rows equals the relaxed-weight oracle at binary weights."""
        self._setup(bkd)
        noise_cov = np.diag(np.diag(self._noise_cov))
        rows = [1, 2]
        _, cov = (
            self._joint(bkd, noise_cov)
            .select(rows)
            .condition(bkd.asarray(self._data[rows]), 1)
        )
        weights = np.zeros((4, 1))
        weights[rows] = 1.0
        oracle = relaxed_linear_target_covariance(
            bkd.asarray(self._bmat),
            bkd.asarray(self._amat),
            bkd.asarray(self._prior_cov),
            bkd.asarray(noise_cov),
            bkd.asarray(weights),
            bkd,
        )
        bkd.assert_allclose(cov, oracle, rtol=1e-10)

    def test_default_refuses_indefinite_blocks(self, bkd: Backend[Array]) -> None:
        """Repair happens only when asked for."""
        rng = np.random.default_rng(11)
        rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        cov = bkd.asarray(rot @ np.diag([2.0, 1.0, -0.5]) @ rot.T)
        blocks = DenseBlocks(bkd.zeros((3, 1)), cov, (1,), 2, bkd)
        noise = DiagonalCovarianceOperator(bkd.full((2,), 0.1), bkd)
        with pytest.raises(ValueError, match="indefinite"):
            JointGaussian(blocks, noise)
        repaired = JointGaussian(blocks, noise, EigenClip())
        eigvals = bkd.eigvalsh(repaired.blocks().covariance())
        assert bkd.to_float(bkd.min(eigvals)) > -1e-12

    def test_rejects_bad_inputs(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        joint = self._joint(bkd, self._noise_cov)
        with pytest.raises(ValueError):
            joint.condition(bkd.ones((3, 1)), 0)
        with pytest.raises(ValueError):
            joint.condition(bkd.asarray(self._data), 2)
        for rows in ([], [0, 0], [4]):
            with pytest.raises(ValueError):
                joint.select(rows)
        noise = DiagonalCovarianceOperator(bkd.full((3,), 0.1), bkd)
        with pytest.raises(ValueError):
            JointGaussian(joint.blocks(), noise)
        with pytest.raises(TypeError):
            JointGaussian(joint.blocks(), bkd.eye(4))
        with pytest.raises(TypeError):
            JointGaussian(joint.blocks().covariance(), joint.noise())
