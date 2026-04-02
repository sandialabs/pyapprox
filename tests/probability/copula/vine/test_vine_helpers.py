"""
Tests for D-vine helper functions.
"""

import numpy as np

from pyapprox.probability.copula.vine.helpers import (
    compute_dvine_partial_correlations,
    correlation_from_partial_correlations,
    precision_bandwidth,
)


class TestHelpers:
    """Base tests for vine helper functions."""

    def test_precision_bandwidth_diagonal(self, bkd) -> None:
        """Diagonal precision matrix has bandwidth 0."""
        P = bkd.asarray(np.diag([2.0, 3.0, 1.0, 4.0]))
        assert precision_bandwidth(P, bkd) == 0

    def test_precision_bandwidth_tridiagonal(self, bkd) -> None:
        """Tridiagonal precision matrix has bandwidth 1."""
        P_np = np.array(
            [
                [2.0, -0.5, 0.0, 0.0],
                [-0.5, 2.0, -0.3, 0.0],
                [0.0, -0.3, 2.0, -0.4],
                [0.0, 0.0, -0.4, 2.0],
            ]
        )
        P = bkd.asarray(P_np)
        assert precision_bandwidth(P, bkd) == 1

    def test_precision_bandwidth_pentadiagonal(self, bkd) -> None:
        """Pentadiagonal precision matrix has bandwidth 2."""
        P_np = np.array(
            [
                [3.0, -0.5, 0.2, 0.0, 0.0],
                [-0.5, 3.0, -0.3, 0.1, 0.0],
                [0.2, -0.3, 3.0, -0.4, 0.2],
                [0.0, 0.1, -0.4, 3.0, -0.3],
                [0.0, 0.0, 0.2, -0.3, 3.0],
            ]
        )
        P = bkd.asarray(P_np)
        assert precision_bandwidth(P, bkd) == 2

    def test_partial_correlations_tree1(self, bkd) -> None:
        """Tree-1 partial correlations equal pairwise correlations."""
        R_np = np.array(
            [
                [1.0, 0.6, 0.3, 0.1],
                [0.6, 1.0, 0.5, 0.2],
                [0.3, 0.5, 1.0, 0.7],
                [0.1, 0.2, 0.7, 1.0],
            ]
        )
        R = bkd.asarray(R_np)
        pc = compute_dvine_partial_correlations(R, 1, bkd)

        assert len(pc[1]) == 3
        bkd.assert_allclose(
            bkd.asarray(pc[1]),
            bkd.asarray([0.6, 0.5, 0.7]),
            rtol=1e-12,
        )

    def test_partial_correlations_tree2(self, bkd) -> None:
        """Tree-2 partial correlations from known correlation matrix."""
        # Build a known positive definite correlation matrix
        R_np = np.array(
            [
                [1.0, 0.6, 0.3, 0.1],
                [0.6, 1.0, 0.5, 0.2],
                [0.3, 0.5, 1.0, 0.7],
                [0.1, 0.2, 0.7, 1.0],
            ]
        )
        R = bkd.asarray(R_np)
        pc = compute_dvine_partial_correlations(R, 2, bkd)

        # Verify tree 2 partial correlations via submatrix inversion
        # Edge (0,2|1): invert R[0:3, 0:3]
        sub = R_np[0:3, 0:3]
        P = np.linalg.inv(sub)
        expected_02_1 = -P[0, 2] / np.sqrt(P[0, 0] * P[2, 2])

        # Edge (1,3|2): invert R[1:4, 1:4]
        sub = R_np[1:4, 1:4]
        P = np.linalg.inv(sub)
        expected_13_2 = -P[0, 2] / np.sqrt(P[0, 0] * P[2, 2])

        assert len(pc[2]) == 2
        bkd.assert_allclose(
            bkd.asarray([pc[2][0]]),
            bkd.asarray([expected_02_1]),
            rtol=1e-10,
            atol=1e-14,
        )
        bkd.assert_allclose(
            bkd.asarray([pc[2][1]]),
            bkd.asarray([expected_13_2]),
            rtol=1e-10,
            atol=1e-14,
        )

    def test_partial_correlations_near_singular(self, bkd) -> None:
        """High correlations (0.95) don't cause numerical issues."""
        # Build correlation matrix with high correlations
        n = 4
        R_np = np.eye(n)
        for i in range(n - 1):
            R_np[i, i + 1] = 0.95
            R_np[i + 1, i] = 0.95
        # Fill in remaining entries to make it pos def
        # Use a Gaussian copula structure: product of adjacent correlations
        for i in range(n):
            for j in range(i + 2, n):
                prod = 1.0
                for k in range(i, j):
                    prod *= R_np[k, k + 1]
                R_np[i, j] = prod
                R_np[j, i] = prod

        R = bkd.asarray(R_np)
        pc = compute_dvine_partial_correlations(R, n - 1, bkd)

        # Should not raise and partial correlations should be finite
        for t in pc:
            for val in pc[t]:
                assert np.isfinite(val), f"Non-finite partial corr at tree {t}"
                assert val > -1.0
                assert val < 1.0

    def test_correlation_roundtrip(self, bkd) -> None:
        """Roundtrip: R -> partial correlations -> R."""
        R_np = np.array(
            [
                [1.0, 0.6, 0.3, 0.1],
                [0.6, 1.0, 0.5, 0.2],
                [0.3, 0.5, 1.0, 0.7],
                [0.1, 0.2, 0.7, 1.0],
            ]
        )
        R = bkd.asarray(R_np)
        n = 4

        pc = compute_dvine_partial_correlations(R, n - 1, bkd)
        R_recovered = correlation_from_partial_correlations(pc, n, bkd)

        bkd.assert_allclose(R_recovered, R, rtol=1e-10)

    def test_correlation_roundtrip_5d(self, bkd) -> None:
        """Roundtrip for 5x5 matrix."""
        # Build positive definite correlation matrix
        np.random.seed(42)
        A = np.random.randn(5, 5) * 0.3
        Sigma = A @ A.T + np.eye(5)
        D = np.sqrt(np.diag(Sigma))
        R_np = Sigma / np.outer(D, D)

        R = bkd.asarray(R_np)
        n = 5

        pc = compute_dvine_partial_correlations(R, n - 1, bkd)
        R_recovered = correlation_from_partial_correlations(pc, n, bkd)

        bkd.assert_allclose(R_recovered, R, rtol=1e-8)
