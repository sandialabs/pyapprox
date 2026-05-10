"""Tests for DGP quadrature rules."""

import numpy as np
import pytest

from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
    FullEnumerationRule,
    IndexBatchRule,
    MonteCarloRule,
    PropagationRule,
    SobolRule,
    TensorProductGHRule,
)


class TestMonteCarloRule:
    def test_shape_and_weights(self, numpy_bkd):
        rule = MonteCarloRule(rng=np.random.default_rng(0))
        nodes, weights = rule(n_samples=100, n_layers=3, bkd=numpy_bkd)
        assert nodes.shape == (100, 3)
        assert weights.shape == (100,)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([np.sum(numpy_bkd.to_numpy(weights))]),
            numpy_bkd.ones((1,)),
            rtol=1e-12,
        )

    def test_first_two_moments(self, numpy_bkd):
        rule = MonteCarloRule(rng=np.random.default_rng(0))
        nodes, _ = rule(n_samples=100000, n_layers=2, bkd=numpy_bkd)
        nodes_np = numpy_bkd.to_numpy(nodes)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(np.mean(nodes_np, axis=0)),
            numpy_bkd.zeros((2,)),
            atol=0.05,
        )
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray(np.var(nodes_np, axis=0)),
            numpy_bkd.ones((2,)),
            atol=0.05,
        )

    def test_satisfies_protocol(self):
        assert isinstance(MonteCarloRule(), PropagationRule)


class TestSobolRule:
    def test_shape_and_weights(self, numpy_bkd):
        rule = SobolRule(seed=42)
        nodes, weights = rule(n_samples=64, n_layers=3, bkd=numpy_bkd)
        assert nodes.shape == (64, 3)
        assert weights.shape == (64,)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([np.sum(numpy_bkd.to_numpy(weights))]),
            numpy_bkd.ones((1,)),
            rtol=1e-12,
        )

    def test_low_discrepancy(self, numpy_bkd):
        """Sobol estimate of E[x0*x1] should be close to 0."""
        rule = SobolRule(seed=42)
        nodes, _ = rule(n_samples=1024, n_layers=2, bkd=numpy_bkd)
        nodes_np = numpy_bkd.to_numpy(nodes)
        est = np.mean(nodes_np[:, 0] * nodes_np[:, 1])
        assert abs(est) < 0.05

    def test_satisfies_protocol(self):
        assert isinstance(SobolRule(), PropagationRule)


class TestTensorProductGHRule:
    def test_polynomial_exactness(self, numpy_bkd):
        """GH at order 5 is exact for x^2*y^2 under N(0,I)."""
        rule = TensorProductGHRule(order=5)
        nodes, weights = rule(n_samples=25, n_layers=2, bkd=numpy_bkd)
        assert nodes.shape == (25, 2)
        nodes_np = numpy_bkd.to_numpy(nodes)
        weights_np = numpy_bkd.to_numpy(weights)
        f = nodes_np[:, 0] ** 2 * nodes_np[:, 1] ** 2
        est = np.sum(weights_np * f)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([est]),
            numpy_bkd.ones((1,)),
            rtol=1e-10,
        )

    def test_1d_moments(self, numpy_bkd):
        """1D GH: E[x^{2p}] = (2p-1)!! for p < order."""
        rule = TensorProductGHRule(order=10)
        nodes, weights = rule(n_samples=10, n_layers=1, bkd=numpy_bkd)
        nodes_np = numpy_bkd.to_numpy(nodes)
        weights_np = numpy_bkd.to_numpy(weights)

        for p in range(1, 10):
            expected = 1.0
            for k in range(1, 2 * p, 2):
                expected *= k
            integrand = nodes_np[:, 0] ** (2 * p)
            result = np.sum(weights_np * integrand)
            numpy_bkd.assert_allclose(
                numpy_bkd.asarray([result]),
                numpy_bkd.asarray([expected]),
                rtol=1e-10,
            )

    def test_weights_sum_to_one(self, numpy_bkd):
        rule = TensorProductGHRule(order=7)
        _, weights = rule(n_samples=7, n_layers=1, bkd=numpy_bkd)
        numpy_bkd.assert_allclose(
            numpy_bkd.asarray([np.sum(numpy_bkd.to_numpy(weights))]),
            numpy_bkd.ones((1,)),
            rtol=1e-12,
        )

    def test_wrong_n_samples_raises(self, numpy_bkd):
        rule = TensorProductGHRule(order=3)
        with pytest.raises(ValueError, match="produces exactly 9 nodes"):
            rule(n_samples=10, n_layers=2, bkd=numpy_bkd)

    def test_satisfies_protocol(self):
        assert isinstance(TensorProductGHRule(order=3), PropagationRule)


class TestIndexBatchRule:
    def test_shape_and_weights(self, bkd):
        ibr = IndexBatchRule(bkd)
        N, B = 100, 10
        indices, weights = ibr(N, B, seed=0)
        assert indices.shape == (B,)
        assert weights.shape == (B,)
        expected_w = float(N) / float(B)
        bkd.assert_allclose(
            weights, bkd.full((B,), expected_w), rtol=1e-12,
        )

    def test_no_duplicates(self, numpy_bkd):
        ibr = IndexBatchRule(numpy_bkd)
        indices, _ = ibr(50, 10, seed=42)
        idx_np = numpy_bkd.to_numpy(indices)
        assert len(set(idx_np.tolist())) == 10


class TestFullEnumerationRule:
    def test_all_indices(self, bkd):
        fer = FullEnumerationRule(bkd)
        indices, weights = fer(20)
        assert indices.shape == (20,)
        bkd.assert_allclose(weights, bkd.ones((20,)), rtol=1e-12)
