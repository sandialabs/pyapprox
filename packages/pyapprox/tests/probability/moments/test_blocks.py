"""Tests for DenseBlocks constructors and covariance repairs."""

import numpy as np
import pytest
from pyapprox.probability.moments import (
    CovarianceRepairProtocol,
    DenseBlocks,
    EigenClip,
    NoRepair,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestDenseBlocks:
    """Correlated 3-dimensional input, targets of sizes (2, 1), 4 observations."""

    def _setup(self, bkd: Backend[Array]) -> None:
        rng = np.random.default_rng(8)
        root = rng.normal(size=(3, 3))
        self._cov = root @ root.T + 0.1 * np.eye(3)
        self._mean = rng.normal(size=(3, 1))
        self._amat = rng.normal(size=(4, 3))
        self._tmats = [rng.normal(size=(2, 3)), rng.normal(size=(1, 3))]

    def _blocks(self, bkd: Backend[Array]) -> DenseBlocks[Array]:
        return DenseBlocks.from_linear_model(
            bkd.asarray(self._amat),
            bkd.asarray(self._mean),
            bkd.asarray(self._cov),
            [bkd.asarray(mat) for mat in self._tmats],
            bkd,
        )

    def test_from_linear_model(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        blocks = self._blocks(bkd)
        a, t0, t1 = self._amat, self._tmats[0], self._tmats[1]
        bkd.assert_allclose(
            blocks.obs_covariance(), bkd.asarray(a @ self._cov @ a.T), rtol=1e-12
        )
        bkd.assert_allclose(
            blocks.target_obs_covariance(0),
            bkd.asarray(t0 @ self._cov @ a.T),
            rtol=1e-12,
        )
        bkd.assert_allclose(
            blocks.target_covariance(1), bkd.asarray(t1 @ self._cov @ t1.T), rtol=1e-12
        )
        bkd.assert_allclose(blocks.obs_mean(), bkd.asarray(a @ self._mean), rtol=1e-12)
        assert blocks.target_sizes() == (2, 1)
        assert blocks.nobs() == 4
        assert blocks.nsamples() is None

    def test_from_linear_model_rejects_mismatch(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        with pytest.raises(ValueError):
            DenseBlocks.from_linear_model(
                bkd.asarray(self._amat[:, :2]),
                bkd.asarray(self._mean),
                bkd.asarray(self._cov),
                [bkd.asarray(self._tmats[0])],
                bkd,
            )

    def test_with_known_targets(self, bkd: Backend[Array]) -> None:
        """Only the chosen target's own moments change."""
        self._setup(bkd)
        blocks = self._blocks(bkd)
        known_mean, known_cov = bkd.ones((2, 1)), 2.0 * bkd.eye(2)
        updated = blocks.with_known_targets({0: (known_mean, known_cov)})
        bkd.assert_allclose(updated.target_mean(0), known_mean)
        bkd.assert_allclose(updated.target_covariance(0), known_cov)
        bkd.assert_allclose(
            updated.target_obs_covariance(0), blocks.target_obs_covariance(0)
        )
        bkd.assert_allclose(updated.target_covariance(1), blocks.target_covariance(1))
        bkd.assert_allclose(updated.obs_covariance(), blocks.obs_covariance())
        # The original is untouched.
        assert not np.allclose(
            bkd.to_numpy(blocks.target_covariance(0)), 2.0 * np.eye(2)
        )

    def test_with_known_targets_rejects_bad_input(self, bkd: Backend[Array]) -> None:
        self._setup(bkd)
        blocks = self._blocks(bkd)
        with pytest.raises(ValueError):
            blocks.with_known_targets({2: (bkd.ones((1, 1)), bkd.eye(1))})
        with pytest.raises(ValueError):
            blocks.with_known_targets({0: (bkd.ones((3, 1)), bkd.eye(3))})


class TestRepair:
    """Stacked covariance diag(3, 1, -0.5) after a random rotation."""

    def _indefinite(self, bkd: Backend[Array], smallest: float) -> DenseBlocks[Array]:
        rng = np.random.default_rng(9)
        rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        cov = rot @ np.diag([3.0, 1.0, smallest]) @ rot.T
        return DenseBlocks(bkd.zeros((3, 1)), bkd.asarray(cov), (1,), 2, bkd)

    def _eigvals(self, bkd: Backend[Array], blocks: DenseBlocks[Array]) -> np.ndarray:
        return np.sort(bkd.to_numpy(bkd.eigvalsh(blocks.covariance())))

    def test_satisfy_protocol(self, bkd: Backend[Array]) -> None:
        assert isinstance(NoRepair(), CovarianceRepairProtocol)
        assert isinstance(EigenClip(), CovarianceRepairProtocol)

    def test_no_repair_raises_on_indefinite(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="indefinite"):
            NoRepair().repair(self._indefinite(bkd, -0.5))

    def test_no_repair_accepts_rounding_negatives(self, bkd: Backend[Array]) -> None:
        blocks = self._indefinite(bkd, -1e-14)
        assert NoRepair().repair(blocks) is blocks

    def test_eigen_clip_zero_floor(self, bkd: Backend[Array]) -> None:
        """The negative eigenvalue becomes zero; the others are unchanged."""
        repaired = EigenClip().repair(self._indefinite(bkd, -0.5))
        bkd.assert_allclose(
            bkd.asarray(self._eigvals(bkd, repaired)),
            bkd.asarray([0.0, 1.0, 3.0]),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_eigen_clip_positive_floor(self, bkd: Backend[Array]) -> None:
        repaired = EigenClip(rel_floor=0.01).repair(self._indefinite(bkd, -0.5))
        bkd.assert_allclose(
            bkd.asarray(self._eigvals(bkd, repaired)),
            bkd.asarray([0.03, 1.0, 3.0]),
            rtol=1e-12,
        )

    def test_eigen_clip_leaves_psd_unchanged(self, bkd: Backend[Array]) -> None:
        blocks = self._indefinite(bkd, 0.2)
        repaired = EigenClip().repair(blocks)
        bkd.assert_allclose(repaired.covariance(), blocks.covariance(), rtol=1e-12)

    def test_reject_negative_settings(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError):
            NoRepair(tol=-1.0)
        with pytest.raises(ValueError):
            EigenClip(rel_floor=-1.0)
