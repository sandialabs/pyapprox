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


def _fast_decaying_symmetric(n, seed=0):
    """Symmetric PSD matrix whose spectrum falls off sharply.

    A gently decaying spectrum -- the ``geomspace(1, 1e-3)`` used above
    -- does not exercise the power-iteration path: the sketch stays well
    conditioned no matter how many iterations run. Six decades is what
    makes the subdominant directions vulnerable.
    """
    pts = np.linspace(0.0, 1.0, n)
    return np.exp(-0.5 * ((pts[:, None] - pts[None, :]) / 0.3) ** 2)


class TestPowerIterationStability:
    """Power iterations must improve accuracy, never destroy it.

    Each application of the operator scales direction i by lambda_i, so
    without re-orthonormalizing between iterations the sketch collapses
    onto the dominant eigenvector and the subdominant directions fall
    below round-off. The QR that prevents this is the difference between
    Algorithms 4.3 and 4.4 of Halko, Martinsson and Tropp.

    Nothing here covered ``npower_iters`` before: the existing tests use
    the default and a gently decaying spectrum, so the collapse never
    showed and the defect passed a green suite.
    """

    @pytest.mark.parametrize("npower_iters", [0, 1, 2, 4])
    def test_accuracy_does_not_degrade_with_power_iterations(
        self, bkd, npower_iters
    ):
        """The property the bug violated, stated directly.

        Measured before the fix, error *grew* with iteration count --
        8.6e-05 at two iterations against 5.7e-03 at four -- because
        each pass discarded more of the subspace. Asserting an absolute
        tolerance at every count is a stronger statement than comparing
        counts against each other, and it holds for any kernel.
        """
        A_np = _fast_decaying_symmetric(120)
        svd = SymmetricRandomizedSVD(
            DenseSymmetricMatVecOperator(bkd.asarray(A_np), bkd),
            seed=0,
            npower_iters=npower_iters,
        )
        _, S, _ = svd.compute(8)
        s_exact = np.linalg.svd(A_np, compute_uv=False)[:8]
        np.testing.assert_allclose(
            np.asarray(bkd.to_numpy(S), float), s_exact, rtol=1e-8,
        )

    def _sketch_rank(self, bkd, matrix, npower_iters, noversampling=10):
        svd = SymmetricRandomizedSVD(
            DenseSymmetricMatVecOperator(bkd.asarray(matrix), bkd),
            seed=0,
            npower_iters=npower_iters,
            noversampling=noversampling,
        )
        sketch = bkd.to_numpy(svd._sample_column_space(8))
        return int(
            np.linalg.matrix_rank(sketch, tol=1e-12 * np.abs(sketch).max())
        ), sketch.shape[1]

    def test_sketch_reaches_full_rank_when_reachable(self, numpy_bkd):
        """When the operator can supply every column, the sketch must.

        A full-rank operator places no ceiling below the column count,
        so any shortfall is the algorithm's doing. This random SPD
        matrix is rank 60 with a spectrum spanning only 1e-04, which is
        the *undemanding* half of the pair: it pins the correctness
        property but is insensitive to the collapse itself, because
        there is too little dynamic range for directions to be lost.
        """
        bkd = numpy_bkd
        rng = np.random.RandomState(0)
        square = rng.normal(size=(60, 60))
        full_rank = square @ square.T / 60
        for npower_iters in (0, 1, 2, 4):
            rank, ncolumns = self._sketch_rank(
                bkd, full_rank, npower_iters
            )
            assert rank == ncolumns, (
                f"sketch reached rank {rank} of {ncolumns} columns at "
                f"npower_iters={npower_iters} on a full-rank operator"
            )

    def test_sketch_spans_the_leading_eigenvectors_when_unreachable(
        self, numpy_bkd
    ):
        """The discriminating case, checked against a dense eigensolve.

        ``np.linalg.eigh`` is deterministic given identical input, so it
        supplies a fixed reference with no randomness and no tolerance
        to choose. Requiring the sketch to span its leading
        eigenvectors states what the algorithm actually promises.

        Counting rank was tried first and does not work here. The
        operator's singular values decay smoothly to 1e-17 with no
        cliff, so any rank is a thresholding artifact: traced through
        one iteration the count reads 14, then 18 once the QR
        manufactures orthonormal directions for dimensions the operator
        had annihilated, then 14 again when the operator maps them back
        to zero. A test asserting a stable rank fails on correct code.

        What the bug destroyed was the span. With the sketch pulled onto
        the dominant eigenvector the leading directions were no longer
        represented, and eigenvalue errors reached 5.7e-03. The residual
        below is the part of each reference eigenvector that the sketch
        cannot represent, which is invariant to how the sketch is
        scaled or where a rank threshold would sit.
        """
        bkd = numpy_bkd
        matrix = _fast_decaying_symmetric(120)
        reference = np.linalg.eigh(matrix)[1][:, ::-1][:, :8]
        for npower_iters in (0, 1, 2, 4):
            svd = SymmetricRandomizedSVD(
                DenseSymmetricMatVecOperator(bkd.asarray(matrix), bkd),
                seed=0,
                npower_iters=npower_iters,
                noversampling=10,
            )
            sketch = bkd.to_numpy(svd._sample_column_space(8))
            basis, _ = np.linalg.qr(sketch)
            residual = np.abs(
                reference - basis @ (basis.T @ reference)
            ).max()
            assert residual < 1e-6, (
                "sketch failed to span the leading eigenvectors at "
                f"npower_iters={npower_iters}, residual {residual:.2e}. "
                "The re-orthonormalization inside the power-iteration "
                "loop is missing or ineffective, so the sketch is "
                "collapsing onto the dominant eigenvector."
            )
