"""Tests for the weighted moment accumulators and DenseBlocks."""

import numpy as np
import pytest
from pyapprox.interface.functions.joint import JointOutputs
from pyapprox.probability.moments import (
    CovarianceBlocksProtocol,
    DenseBlocks,
    MomentAccumulatorProtocol,
    UnbiasedMCAccumulator,
    WeightedAccumulator,
)
from pyapprox.util.backends.protocols import Array, Backend


class TestAccumulators:
    """Targets of sizes (2, 1) and 3 observations at 20 weighted samples."""

    _sizes, _nobs, _n = (2, 1), 3, 20

    def _data(
        self, bkd: Backend[Array], offset: float = 0.0
    ) -> tuple[JointOutputs[Array], Array, Array]:
        rng = np.random.default_rng(4)
        stacked = rng.normal(size=(sum(self._sizes) + self._nobs, self._n)) + offset
        weights = rng.uniform(0.5, 1.5, size=(1, self._n))
        weights /= weights.sum()
        outputs = JointOutputs(
            targets=(bkd.asarray(stacked[:2]), bkd.asarray(stacked[2:3])),
            observations=bkd.asarray(stacked[3:]),
        )
        return outputs, bkd.asarray(weights), bkd.asarray(stacked)

    def _reference(
        self, bkd: Backend[Array], stacked: Array, weights: Array
    ) -> tuple[Array, Array]:
        mean = bkd.dot(stacked, weights.T)
        dev = stacked - mean
        return mean, bkd.dot(dev * weights, dev.T)

    def _batch(
        self, outputs: JointOutputs[Array], weights: Array, cols: slice
    ) -> tuple[Array, JointOutputs[Array]]:
        return weights[:, cols], JointOutputs(
            targets=tuple(t[:, cols] for t in outputs.targets),
            observations=outputs.observations[:, cols],
        )

    def test_matches_weighted_formula(self, bkd: Backend[Array]) -> None:
        outputs, weights, stacked = self._data(bkd)
        acc = WeightedAccumulator(bkd)
        acc.update(weights, outputs)
        blocks = acc.finalize()
        mean, cov = self._reference(bkd, stacked, weights)
        bkd.assert_allclose(blocks.mean(), mean, rtol=1e-12)
        bkd.assert_allclose(blocks.covariance(), cov, rtol=1e-12)
        assert blocks.nsamples() == self._n

    def test_streaming_equals_one_batch(self, bkd: Backend[Array]) -> None:
        outputs, weights, _ = self._data(bkd)
        whole = WeightedAccumulator(bkd)
        whole.update(weights, outputs)
        streamed = WeightedAccumulator(bkd)
        for cols in (slice(0, 5), slice(5, 12), slice(12, 13), slice(13, 20)):
            streamed.update(*self._batch(outputs, weights, cols))
        a, b = whole.finalize(), streamed.finalize()
        bkd.assert_allclose(b.mean(), a.mean(), rtol=1e-12)
        bkd.assert_allclose(b.covariance(), a.covariance(), rtol=1e-12, atol=1e-15)
        assert b.nsamples() == self._n

    def test_large_mean_does_not_cancel(self, bkd: Backend[Array]) -> None:
        """A mean of 1e8 with unit spread keeps the covariance accurate."""
        outputs, weights, stacked = self._data(bkd, offset=1e8)
        acc = WeightedAccumulator(bkd)
        for cols in (slice(0, 10), slice(10, 20)):
            acc.update(*self._batch(outputs, weights, cols))
        _, cov = self._reference(bkd, stacked - 1e8, weights)
        bkd.assert_allclose(acc.finalize().covariance(), cov, rtol=1e-6, atol=1e-8)

    def test_unbiased_matches_sample_covariance(self, bkd: Backend[Array]) -> None:
        outputs, _, stacked = self._data(bkd)
        equal = bkd.full((1, self._n), 1.0 / self._n)
        acc = UnbiasedMCAccumulator(bkd)
        acc.update(equal, outputs)
        dev = stacked - bkd.dot(stacked, equal.T)
        expected = bkd.dot(dev, dev.T) / (self._n - 1)
        bkd.assert_allclose(acc.finalize().covariance(), expected, rtol=1e-12)

    def test_negative_weights_report_indefinite(self, bkd: Backend[Array]) -> None:
        """Weights (-1/2, 2, -1/2) at (-1, 0, 1) give variance -1."""
        values = bkd.asarray([[-1.0, 0.0, 1.0]])
        outputs = JointOutputs(targets=(values,), observations=values)
        acc = WeightedAccumulator(bkd)
        acc.update(bkd.asarray([[-0.5, 2.0, -0.5]]), outputs)
        blocks = acc.finalize()
        bkd.assert_allclose(blocks.obs_covariance(), bkd.asarray([[-1.0]]))
        assert bkd.to_float(blocks.min_eigenvalue()[0]) < 0.0

    def test_block_slices(self, bkd: Backend[Array]) -> None:
        outputs, weights, stacked = self._data(bkd)
        acc = WeightedAccumulator(bkd)
        acc.update(weights, outputs)
        blocks = acc.finalize()
        mean, cov = self._reference(bkd, stacked, weights)
        assert blocks.target_sizes() == self._sizes
        assert blocks.nobs() == self._nobs
        bkd.assert_allclose(blocks.target_mean(0), mean[:2], rtol=1e-12)
        bkd.assert_allclose(blocks.target_mean(1), mean[2:3], rtol=1e-12)
        bkd.assert_allclose(blocks.obs_mean(), mean[3:], rtol=1e-12)
        bkd.assert_allclose(blocks.target_covariance(0), cov[:2, :2], rtol=1e-12)
        bkd.assert_allclose(blocks.target_covariance(1), cov[2:3, 2:3], rtol=1e-12)
        bkd.assert_allclose(blocks.target_obs_covariance(0), cov[:2, 3:], rtol=1e-12)
        bkd.assert_allclose(blocks.target_obs_covariance(1), cov[2:3, 3:], rtol=1e-12)
        bkd.assert_allclose(blocks.obs_covariance(), cov[3:, 3:], rtol=1e-12)

    def test_satisfy_protocols(self, bkd: Backend[Array]) -> None:
        outputs, weights, _ = self._data(bkd)
        for acc in (WeightedAccumulator(bkd), UnbiasedMCAccumulator(bkd)):
            assert isinstance(acc, MomentAccumulatorProtocol)
            acc.update(weights, outputs)
            assert isinstance(acc.finalize(), CovarianceBlocksProtocol)

    def test_rejects_1d_weights(self, bkd: Backend[Array]) -> None:
        outputs, weights, _ = self._data(bkd)
        with pytest.raises(ValueError):
            WeightedAccumulator(bkd).update(weights[0], outputs)

    def test_rejects_unnormalized_weights(self, bkd: Backend[Array]) -> None:
        outputs, weights, _ = self._data(bkd)
        acc = WeightedAccumulator(bkd)
        acc.update(2.0 * weights, outputs)
        with pytest.raises(ValueError):
            acc.finalize()

    def test_rejects_changed_block_sizes(self, bkd: Backend[Array]) -> None:
        outputs, weights, _ = self._data(bkd)
        acc = WeightedAccumulator(bkd)
        acc.update(*self._batch(outputs, weights, slice(0, 10)))
        merged = JointOutputs(
            targets=(outputs.targets[0][:, 10:],),
            observations=outputs.observations[:, 10:],
        )
        with pytest.raises(ValueError):
            acc.update(weights[:, 10:], merged)

    def test_rejects_empty_finalize(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError):
            WeightedAccumulator(bkd).finalize()

    def test_unbiased_rejects_single_sample(self, bkd: Backend[Array]) -> None:
        values = bkd.ones((1, 1))
        acc = UnbiasedMCAccumulator(bkd)
        single = JointOutputs(targets=(values,), observations=values)
        acc.update(bkd.ones((1, 1)), single)
        with pytest.raises(ValueError):
            acc.finalize()

    def test_dense_blocks_rejects_bad_shapes(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError):
            DenseBlocks(bkd.zeros((4, 1)), bkd.eye(4), (2,), 3, bkd)
        with pytest.raises(ValueError):
            DenseBlocks(bkd.zeros((5, 1)), bkd.eye(4), (2,), 3, bkd)
        with pytest.raises(ValueError):
            DenseBlocks(bkd.zeros((3, 1)), bkd.eye(3), (0,), 3, bkd)
