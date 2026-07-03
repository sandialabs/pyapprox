"""Tests for the randomized SVD classes.

TwoPassRandomizedSVD is the standard Halko-Martinsson-Tropp algorithm for
general rectangular operators; SymmetricRandomizedSVD requires a symmetric
operator.  These tests pin accuracy against the exact SVD, seed
reproducibility (without perturbing the global RNG), and the symmetric-only
constraint.
"""

import numpy as np
import pytest

from pyapprox.util.linalg import (
    DenseMatVecOperator,
    DenseSymmetricMatVecOperator,
    SymmetricRandomizedSVD,
    TwoPassRandomizedSVD,
)


def _low_rank_rectangular(m, n, rank, seed):
    rng = np.random.RandomState(seed)
    U, _ = np.linalg.qr(rng.normal(size=(m, rank)))
    V, _ = np.linalg.qr(rng.normal(size=(n, rank)))
    svals = np.geomspace(1.0, 1e-3, rank)
    return (U * svals) @ V.T


class TestTwoPassRandomizedSVD:
    """General rectangular operators."""

    def test_matches_exact_svd(self, bkd):
        A_np = _low_rank_rectangular(40, 60, 12, seed=0)
        svd = TwoPassRandomizedSVD(
            DenseMatVecOperator(bkd.asarray(A_np), bkd), seed=0,
        )
        U, S, Vh = svd.compute(12)
        s_exact = np.linalg.svd(A_np, compute_uv=False)[:12]
        np.testing.assert_allclose(
            np.asarray(bkd.to_numpy(S), float), s_exact, rtol=1e-8,
        )
        # Rank-12 reconstruction recovers the (exactly rank-12) matrix.
        recon = bkd.to_numpy(U) @ np.diag(bkd.to_numpy(S)) @ bkd.to_numpy(Vh)
        np.testing.assert_allclose(recon, A_np, atol=1e-8)

    def test_seed_reproducible(self, bkd):
        A = bkd.asarray(_low_rank_rectangular(30, 45, 8, seed=1))
        out = []
        for _ in range(2):
            svd = TwoPassRandomizedSVD(
                DenseMatVecOperator(A, bkd), seed=13,
            )
            out.append([np.asarray(bkd.to_numpy(x), float)
                        for x in svd.compute(8)])
        for a, b in zip(out[0], out[1]):
            np.testing.assert_array_equal(a, b)

    def test_seed_does_not_touch_global_rng(self, bkd):
        A = bkd.asarray(_low_rank_rectangular(20, 30, 5, seed=2))
        np.random.seed(0)
        expected = np.random.rand()
        np.random.seed(0)
        TwoPassRandomizedSVD(
            DenseMatVecOperator(A, bkd), seed=99,
        ).compute(5)
        assert np.random.rand() == expected


class TestSymmetricRandomizedSVD:
    """Symmetric operators only."""

    def test_matches_exact_on_symmetric(self, bkd):
        rng = np.random.RandomState(3)
        B = rng.normal(size=(30, 10))
        A_np = B @ B.T  # PSD, rank 10
        svd = SymmetricRandomizedSVD(
            DenseSymmetricMatVecOperator(bkd.asarray(A_np), bkd), seed=0,
        )
        _U, S, _Vh = svd.compute(10)
        s_exact = np.linalg.svd(A_np, compute_uv=False)[:10]
        np.testing.assert_allclose(
            np.asarray(bkd.to_numpy(S), float), s_exact, rtol=1e-8,
        )

    def test_rejects_general_operator(self, bkd):
        A = bkd.asarray(np.random.RandomState(4).normal(size=(20, 30)))
        with pytest.raises(ValueError, match="SymmetricMatVecOperator"):
            SymmetricRandomizedSVD(DenseMatVecOperator(A, bkd))
