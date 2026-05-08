"""Tests for pivoted Cholesky factorization.

Tests replicate legacy tests from pyapprox/util/tests/test_linalg.py
using the new typing module implementation.
"""

import numpy as np
import pytest

from pyapprox.surrogates.kernels.matern import (
    Matern32Kernel,
    SquaredExponentialKernel,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.pivoted_cholesky import (
    KernelColumnOperator,
    PivotedCholeskyFactorizer,
    _HAS_NUMBA,
)


class TestPivotedCholesky:
    """Base tests for PivotedCholeskyFactorizer.

    Replicates legacy test_pivoted_cholesky_decomposition and
    test_update_pivoted_cholesky from pyapprox/util/tests/test_linalg.py.
    """

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(1)

    def test_full_factorization(self, bkd) -> None:
        """Full pivoted Cholesky recovers A = L @ L.T."""
        nrows = 4
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, nrows)))
        A = A.T @ A
        fact = PivotedCholeskyFactorizer(A, bkd)
        fact.factorize(nrows)
        assert fact.success()
        L = fact.factor()
        bkd.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_low_rank_factorization(self, bkd) -> None:
        """Partial factorization of rank-deficient matrix."""
        nrows, npivots = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (npivots, nrows)))
        A = A.T @ A
        fact = PivotedCholeskyFactorizer(A, bkd)
        fact.factorize(npivots)
        L = fact.factor()
        assert L.shape == (nrows, npivots)
        assert fact.pivots().shape[0] == npivots
        assert fact.npivots() == npivots
        bkd.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_init_pivots_enforced(self, bkd) -> None:
        """init_pivots forces specific pivot ordering."""
        nrows, npivots = 4, 3
        A = bkd.asarray(np.random.normal(0.0, 1.0, (npivots, nrows)))
        A = A.T @ A

        # Get natural pivot order
        fact1 = PivotedCholeskyFactorizer(A, bkd)
        fact1.factorize(npivots)
        pivots1 = fact1.pivots()

        # Force second natural pivot to go first
        fact2 = PivotedCholeskyFactorizer(A, bkd)
        fact2.factorize(npivots, init_pivots=pivots1[1:2])
        pivots2 = fact2.pivots()
        pivots1_np = bkd.to_numpy(pivots1)
        bkd.assert_allclose(
            pivots2,
            bkd.asarray(pivots1_np[[1, 0, 2]], dtype=bkd.int64_dtype()),
        )

    def test_known_matrix(self, bkd) -> None:
        """Factorize a known 3x3 matrix and compare with numpy cholesky."""
        A = bkd.asarray([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])
        fact = PivotedCholeskyFactorizer(A, bkd)
        fact.factorize(A.shape[0])
        L = fact.factor()

        # Reorder A so cholesky needs no pivoting
        true_pivots = np.array([2, 1, 0])
        A_no_pivots = A[true_pivots, :][:, true_pivots]
        L_np = bkd.cholesky(A_no_pivots)
        bkd.assert_allclose(L[fact.pivots(), :], L_np, rtol=1e-10)

    def test_known_matrix_permuted(self, bkd) -> None:
        """Factorize permuted known matrix."""
        A_orig = bkd.asarray(
            [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
        )
        true_pivots = np.array([1, 0, 2])
        A = A_orig[true_pivots, :][:, true_pivots]

        fact = PivotedCholeskyFactorizer(A, bkd)
        fact.factorize(A.shape[0])
        L = fact.factor()

        orig_pivots = np.array([2, 1, 0])
        A_no_pivots = A_orig[orig_pivots, :][:, orig_pivots]
        L_np = bkd.cholesky(A_no_pivots)
        bkd.assert_allclose(L[fact.pivots(), :], L_np, rtol=1e-10)

    def test_econ_false(self, bkd) -> None:
        """Full Schur complement pivot selection mode."""
        nrows = 4
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, nrows)))
        A = A.T @ A
        fact = PivotedCholeskyFactorizer(A, bkd, econ=False)
        fact.factorize(nrows)
        assert fact.success()
        L = fact.factor()
        bkd.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_econ_false_rank_deficient(self, bkd) -> None:
        """econ=False on rank-deficient matrix raises ValueError."""
        nrows, rank = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, rank)))
        A = A.T @ A
        fact = PivotedCholeskyFactorizer(A, bkd, econ=False)
        with pytest.raises(ValueError):
            fact.factorize(nrows)

    def test_econ_true_rank_deficient(self, bkd) -> None:
        """econ=True on rank-deficient matrix recovers A."""
        nrows, rank = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, rank)))
        A = A @ A.T
        fact = PivotedCholeskyFactorizer(A, bkd, econ=True)
        fact.factorize(nrows)
        L = fact.factor()
        bkd.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_econ_false_rank_deficient_wide(self, bkd) -> None:
        """econ=False on rank-deficient wide matrix recovers A."""
        nrows, rank = 4, 2
        A = bkd.asarray(np.random.normal(0.0, 1.0, (nrows, rank)))
        A = A @ A.T
        fact = PivotedCholeskyFactorizer(A, bkd, econ=False)
        fact.factorize(nrows)
        L = fact.factor()
        bkd.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_update_continues_factorization(self, bkd) -> None:
        """update() continues from partial factorization and matches full.

        Replicates legacy test_update_pivoted_cholesky.
        """
        nrows = 10
        A = bkd.asarray(np.random.normal(0, 1, (nrows, nrows)))
        A = A.T @ A
        pivot_weights = bkd.asarray(np.random.uniform(1, 2, nrows))

        # Full factorization
        fact1 = PivotedCholeskyFactorizer(A, bkd)
        fact1.factorize(nrows, pivot_weights=pivot_weights)
        L1 = fact1.factor()

        # Partial then update
        npivots_partial = nrows - 2
        fact2 = PivotedCholeskyFactorizer(A, bkd)
        fact2.factorize(npivots_partial, pivot_weights=pivot_weights)
        assert fact2.npivots() == npivots_partial
        fact2.update(nrows)
        L2 = fact2.factor()

        bkd.assert_allclose(L2, L1, rtol=1e-10)
        bkd.assert_allclose(fact2.pivots(), fact1.pivots())

    def test_pivot_weights(self, bkd) -> None:
        """Pivot weights influence selection order."""
        nrows = 5
        A = bkd.asarray(np.random.normal(0, 1, (nrows, nrows)))
        A = A.T @ A

        # Without weights
        fact1 = PivotedCholeskyFactorizer(A, bkd)
        fact1.factorize(nrows)
        fact1.pivots()

        # With weights heavily favoring index 0
        weights = bkd.asarray([100.0, 1.0, 1.0, 1.0, 1.0])
        fact2 = PivotedCholeskyFactorizer(A, bkd)
        fact2.factorize(nrows, pivot_weights=weights)
        pivots2 = fact2.pivots()

        # First pivot should be 0 with heavy weight
        assert int(bkd.to_numpy(pivots2[0:1])[0]) == 0

    def test_too_many_pivots_raises(self, bkd) -> None:
        """Requesting more pivots than rows raises ValueError."""
        A = bkd.eye(3)
        fact = PivotedCholeskyFactorizer(A, bkd)
        with pytest.raises(ValueError):
            fact.factorize(4)

    def test_factorize_before_update_raises(self, bkd) -> None:
        """Calling update() before factorize() raises."""
        A = bkd.eye(3)
        fact = PivotedCholeskyFactorizer(A, bkd)
        with pytest.raises(RuntimeError):
            fact.update(3)


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not available")
class TestKernelColumnOperatorFused:
    """Verify fused kernel pchol matches dense kernel pchol."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(7)

    def _make_kernel_and_points(self, numpy_bkd, n, lenscale=0.3):
        bkd = numpy_bkd
        kernel = Matern32Kernel(
            bkd.asarray([lenscale]), (1e-6, 1e6),
            nvars=1, bkd=bkd, fixed=True,
        )
        x = bkd.asarray(
            np.sort(np.random.uniform(-3.0, 6.0, n)).reshape(1, -1)
        )
        return kernel, x

    def test_fused_matches_dense(self, numpy_bkd) -> None:
        """Fused kernel pchol produces same pivots and factor as dense."""
        bkd = numpy_bkd
        n, npivots = 200, 30
        kernel, X = self._make_kernel_and_points(bkd, n)
        K = bkd.to_numpy(kernel(X, X))

        fact_dense = PivotedCholeskyFactorizer(bkd.asarray(K), bkd)
        fact_dense.factorize(npivots)

        op = KernelColumnOperator(kernel, X, bkd)
        fact_fused = PivotedCholeskyFactorizer(op, bkd)
        fact_fused.factorize(npivots)

        bkd.assert_allclose(fact_fused.pivots(), fact_dense.pivots())
        bkd.assert_allclose(
            fact_fused.factor(), fact_dense.factor(), rtol=1e-10,
            atol=1e-14,
        )

    def test_fused_with_weights(self, numpy_bkd) -> None:
        """Fused kernel pchol with pivot weights matches dense."""
        bkd = numpy_bkd
        n, npivots = 100, 20
        kernel, X = self._make_kernel_and_points(bkd, n)
        K = bkd.to_numpy(kernel(X, X))
        weights = bkd.asarray(np.random.uniform(0.5, 5.0, n))

        fact_dense = PivotedCholeskyFactorizer(bkd.asarray(K), bkd)
        fact_dense.factorize(npivots, pivot_weights=weights)

        op = KernelColumnOperator(kernel, X, bkd)
        fact_fused = PivotedCholeskyFactorizer(op, bkd)
        fact_fused.factorize(npivots, pivot_weights=weights)

        bkd.assert_allclose(fact_fused.pivots(), fact_dense.pivots())
        bkd.assert_allclose(
            fact_fused.factor(), fact_dense.factor(), rtol=1e-10,
        )

    def test_fused_with_init_pivots(self, numpy_bkd) -> None:
        """Fused kernel pchol with init pivots matches dense."""
        bkd = numpy_bkd
        n, npivots = 100, 20
        kernel, X = self._make_kernel_and_points(bkd, n)
        K = bkd.to_numpy(kernel(X, X))
        init_pivots = bkd.asarray(np.array([5, 50, 90], dtype=np.int64))

        fact_dense = PivotedCholeskyFactorizer(bkd.asarray(K), bkd)
        fact_dense.factorize(npivots, init_pivots=init_pivots)

        op = KernelColumnOperator(kernel, X, bkd)
        fact_fused = PivotedCholeskyFactorizer(op, bkd)
        fact_fused.factorize(npivots, init_pivots=init_pivots)

        bkd.assert_allclose(fact_fused.pivots(), fact_dense.pivots())
        bkd.assert_allclose(
            fact_fused.factor(), fact_dense.factor(), rtol=1e-10,
        )

    def test_fused_recovers_kernel_matrix(self, numpy_bkd) -> None:
        """L @ L.T from fused path approximates kernel matrix."""
        bkd = numpy_bkd
        n, npivots = 50, 50
        kernel, X = self._make_kernel_and_points(bkd, n, lenscale=1.0)
        K = kernel(X, X)

        op = KernelColumnOperator(kernel, X, bkd)
        fact = PivotedCholeskyFactorizer(op, bkd)
        fact.factorize(npivots)
        L = fact.factor()
        bkd.assert_allclose(L @ L.T, K, rtol=1e-10)

    def test_generic_fallback_for_kernel_op(self, numpy_bkd) -> None:
        """KernelColumnOperator with numba disabled uses generic path."""
        bkd = numpy_bkd
        n, npivots = 50, 15
        kernel, X = self._make_kernel_and_points(bkd, n)
        K = bkd.to_numpy(kernel(X, X))

        fact_dense = PivotedCholeskyFactorizer(bkd.asarray(K), bkd)
        fact_dense.factorize(npivots)

        op = KernelColumnOperator(kernel, X, bkd)
        fact_gen = PivotedCholeskyFactorizer(op, bkd)
        fact_gen._use_numba = False
        fact_gen.factorize(npivots)

        bkd.assert_allclose(fact_gen.pivots(), fact_dense.pivots())
        bkd.assert_allclose(
            fact_gen.factor(), fact_dense.factor(), rtol=1e-10,
        )

    def test_unrecognized_kernel_falls_back(self, numpy_bkd) -> None:
        """KernelColumnOperator with kernel lacking numba protocol uses generic."""
        bkd = numpy_bkd
        n, npivots = 80, 20

        class _OpaqueKernel:
            """Kernel wrapper that hides numba protocol methods."""
            def __init__(self, inner):
                self._inner = inner
            def __call__(self, X1, X2=None):
                return self._inner(X1, X2)
            def diag(self, X1):
                return self._inner.diag(X1)

        inner = Matern32Kernel(
            bkd.asarray([0.5]), (1e-6, 1e6),
            nvars=1, bkd=bkd, fixed=True,
        )
        kernel = _OpaqueKernel(inner)
        X = bkd.asarray(
            np.sort(np.random.uniform(-3.0, 6.0, n)).reshape(1, -1)
        )
        K = inner(X, X)

        fact_dense = PivotedCholeskyFactorizer(bkd.asarray(bkd.to_numpy(K)), bkd)
        fact_dense.factorize(npivots)

        op = KernelColumnOperator(kernel, X, bkd)
        fact_op = PivotedCholeskyFactorizer(op, bkd)
        fact_op.factorize(npivots)

        bkd.assert_allclose(fact_op.pivots(), fact_dense.pivots())
        bkd.assert_allclose(
            fact_op.factor(), fact_dense.factor(), rtol=1e-10,
        )

    def test_fused_sqexp_matches_dense(self, numpy_bkd) -> None:
        """Fused path works for SquaredExponentialKernel via protocol."""
        bkd = numpy_bkd
        n, npivots = 100, 20
        lenscale = 0.5
        kernel = SquaredExponentialKernel(
            bkd.asarray([lenscale]), (1e-6, 1e6),
            nvars=1, bkd=bkd, fixed=True,
        )
        X = bkd.asarray(
            np.sort(np.random.uniform(-3.0, 6.0, n)).reshape(1, -1)
        )
        K = bkd.to_numpy(kernel(X, X))

        fact_dense = PivotedCholeskyFactorizer(bkd.asarray(K), bkd)
        fact_dense.factorize(npivots)

        op = KernelColumnOperator(kernel, X, bkd)
        fact_fused = PivotedCholeskyFactorizer(op, bkd)
        fact_fused.factorize(npivots)

        bkd.assert_allclose(fact_fused.pivots(), fact_dense.pivots())
        bkd.assert_allclose(
            fact_fused.factor(), fact_dense.factor(), rtol=1e-10,
            atol=1e-14,
        )

    def test_fused_multidimensional(self, numpy_bkd) -> None:
        """Fused path works for multi-dimensional kernel inputs."""
        bkd = numpy_bkd
        nvars, n, npivots = 3, 100, 20
        lenscales = [0.3, 0.5, 0.8]
        kernel = Matern32Kernel(
            bkd.asarray(lenscales), (1e-6, 1e6),
            nvars=nvars, bkd=bkd, fixed=True,
        )
        X = bkd.asarray(np.random.uniform(-2.0, 2.0, (nvars, n)))
        K = bkd.to_numpy(kernel(X, X))

        fact_dense = PivotedCholeskyFactorizer(bkd.asarray(K), bkd)
        fact_dense.factorize(npivots)

        op = KernelColumnOperator(kernel, X, bkd)
        fact_fused = PivotedCholeskyFactorizer(op, bkd)
        fact_fused.factorize(npivots)

        bkd.assert_allclose(fact_fused.pivots(), fact_dense.pivots())
        bkd.assert_allclose(
            fact_fused.factor(), fact_dense.factor(), rtol=1e-10,
            atol=1e-14,
        )


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not available")
class TestNumbaGenericAgreement:
    """Verify numba and generic paths produce identical pivots and factors."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _run_both(self, numpy_bkd, A_np, npivots,
                  init_pivots=None, pivot_weights=None):
        bkd = numpy_bkd
        A = bkd.asarray(A_np)

        fact_nb = PivotedCholeskyFactorizer(A, bkd, econ=True)
        assert fact_nb._use_numba
        fact_nb.factorize(npivots, init_pivots=init_pivots,
                          pivot_weights=pivot_weights)

        fact_gen = PivotedCholeskyFactorizer(A, bkd, econ=True)
        fact_gen._use_numba = False
        fact_gen.factorize(npivots, init_pivots=init_pivots,
                           pivot_weights=pivot_weights)

        bkd.assert_allclose(fact_nb.pivots(), fact_gen.pivots())
        bkd.assert_allclose(fact_nb.factor(), fact_gen.factor(), rtol=1e-12)
        return fact_nb, fact_gen

    def test_no_weights(self, numpy_bkd) -> None:
        n = 20
        A = np.random.randn(n, n)
        A = A.T @ A
        self._run_both(numpy_bkd, A, n)

    def test_low_rank(self, numpy_bkd) -> None:
        n, rank = 50, 10
        B = np.random.randn(n, rank)
        A = B @ B.T + 1e-10 * np.eye(n)
        self._run_both(numpy_bkd, A, rank)

    def test_with_weights(self, numpy_bkd) -> None:
        n = 20
        A = np.random.randn(n, n)
        A = A.T @ A
        weights = numpy_bkd.asarray(np.random.uniform(0.5, 5.0, n))
        self._run_both(numpy_bkd, A, n, pivot_weights=weights)

    def test_with_init_pivots(self, numpy_bkd) -> None:
        n = 15
        A = np.random.randn(n, n)
        A = A.T @ A
        init_pivots = numpy_bkd.asarray(np.array([3, 7], dtype=np.int64))
        self._run_both(numpy_bkd, A, n, init_pivots=init_pivots)

    def test_with_weights_and_init_pivots(self, numpy_bkd) -> None:
        n = 15
        A = np.random.randn(n, n)
        A = A.T @ A
        weights = numpy_bkd.asarray(np.random.uniform(1.0, 3.0, n))
        init_pivots = numpy_bkd.asarray(np.array([2, 5], dtype=np.int64))
        self._run_both(numpy_bkd, A, n, init_pivots=init_pivots,
                       pivot_weights=weights)
