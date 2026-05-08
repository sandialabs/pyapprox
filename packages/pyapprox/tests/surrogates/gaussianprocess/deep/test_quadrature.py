"""Tests for DGP quadrature rules."""

import numpy as np

from pyapprox.surrogates.gaussianprocess.deep.quadrature import (
    FullEnumerationRule,
    GaussHermiteRule,
    IndexBatchRule,
    MonteCarloRule,
    UnscentedRule,
)


class TestGaussHermiteRule:
    def test_polynomial_exactness(self, bkd):
        """Tier 0: GH integrates x^{2p} N(x|0,1) = (2p-1)!! for p < k."""
        gh = GaussHermiteRule(bkd)
        order = 10
        nodes, weights = gh(order, dim=1)

        for p in range(1, order):
            # E[x^{2p}] = (2p-1)!! = 1*3*5*...*(2p-1)
            expected = 1.0
            for k in range(1, 2 * p, 2):
                expected *= k

            integrand = nodes[:, 0] ** (2 * p)
            result = bkd.sum(weights * integrand)
            bkd.assert_allclose(
                bkd.asarray([result]), bkd.asarray([expected]), rtol=1e-10,
            )

    def test_2d_mean_zero(self, bkd):
        """GH in 2D: E[x] = 0 for each dimension."""
        gh = GaussHermiteRule(bkd)
        nodes, weights = gh(5, dim=2)
        for d in range(2):
            result = bkd.sum(weights * nodes[:, d])
            bkd.assert_allclose(
                bkd.asarray([result]), bkd.zeros((1,)), atol=1e-12,
            )

    def test_weights_sum_to_one(self, bkd):
        gh = GaussHermiteRule(bkd)
        _, weights = gh(7, dim=1)
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]), bkd.ones((1,)), rtol=1e-12,
        )


class TestMonteCarloRule:
    def test_convergence_rate(self, bkd):
        """Tier 0: MC mean/variance convergence rate O(N^{-1/2})."""
        mc = MonteCarloRule(bkd)

        errors = []
        ns = [1000, 4000, 16000]
        for n in ns:
            nodes, weights = mc(n, dim=1, seed=42)
            # E[x^2] = 1 for N(0,1)
            est = bkd.sum(weights * nodes[:, 0] ** 2)
            err = abs(float(bkd.to_numpy(bkd.asarray([est]))[0]) - 1.0)
            errors.append(err)

        # Error should decrease roughly as 1/sqrt(N)
        # ratio of errors for 4x samples should be ~2x reduction
        ratio = errors[0] / max(errors[2], 1e-15)
        assert ratio > 1.0

    def test_shape(self, bkd):
        mc = MonteCarloRule(bkd)
        nodes, weights = mc(100, dim=3, seed=0)
        assert nodes.shape == (100, 3)
        assert weights.shape == (100,)

    def test_weights_sum_to_one(self, bkd):
        mc = MonteCarloRule(bkd)
        _, weights = mc(50, dim=2, seed=0)
        bkd.assert_allclose(
            bkd.asarray([bkd.sum(weights)]), bkd.ones((1,)), rtol=1e-12,
        )


class TestUnscentedRule:
    def test_captures_moments(self, bkd):
        """Unscented captures first two moments of N(0, I)."""
        ut = UnscentedRule(bkd, kappa=1.0)
        dim = 3
        nodes, weights = ut(dim)
        assert nodes.shape == (2 * dim + 1, dim)

        # E[x] = 0
        for d in range(dim):
            mean_d = bkd.sum(weights * nodes[:, d])
            bkd.assert_allclose(
                bkd.asarray([mean_d]), bkd.zeros((1,)), atol=1e-12,
            )

        # E[x_i * x_j] = delta_ij
        for i in range(dim):
            for j in range(dim):
                cov_ij = bkd.sum(weights * nodes[:, i] * nodes[:, j])
                expected = 1.0 if i == j else 0.0
                bkd.assert_allclose(
                    bkd.asarray([cov_ij]),
                    bkd.asarray([expected]),
                    atol=1e-12,
                )

    def test_point_count(self, bkd):
        dim = 4
        ut = UnscentedRule(bkd)
        nodes, weights = ut(dim)
        assert nodes.shape[0] == 2 * dim + 1


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
