"""Tests for matrix-free kernel matrix-vector products."""

import numpy as np
import pytest
from pyapprox.surrogates.kernels.matern import ExponentialKernel
from pyapprox.util.linalg.kernel_operators import KernelMatVecOperator
from pyapprox.util.linalg.randomized import SymmetricMatVecOperator


def _setup(bkd, npoints=37, nvars_in=2, lenscale=0.4):
    """Points and a kernel, with npoints deliberately not round."""
    np.random.seed(0)
    X = bkd.array(np.random.uniform(0.0, 1.0, (nvars_in, npoints)))
    lenscale_arr = bkd.full((nvars_in,), lenscale)
    kernel = ExponentialKernel(lenscale_arr, (0.01, 100.0), nvars_in, bkd)
    return X, kernel


class TestMatchesDense:
    """The operator must agree with an assembled K exactly."""

    @pytest.mark.parametrize("block_size", [1, 7, 37, 100])
    def test_unweighted_matches_dense(self, bkd, block_size) -> None:
        """Blocking is an implementation detail, not an approximation.

        Sizes chosen so that some divide npoints=37 and some do not; an
        off-by-one in the final partial block is the likely bug and only
        shows up when the size does not divide evenly.
        """
        X, kernel = _setup(bkd)
        op = KernelMatVecOperator(kernel, X, bkd, block_size=block_size)
        vecs = bkd.array(np.random.normal(0.0, 1.0, (37, 3)))
        bkd.assert_allclose(op.apply(vecs), kernel(X, X) @ vecs, rtol=1e-12)

    def test_weighted_matches_dense_symmetrized(self, bkd) -> None:
        """The weighted form is W^{1/2} K W^{1/2}, applied without W.

        Folding the weights into the operator is what keeps the weighted
        case matrix-free, so it has to reproduce the assembled
        symmetrization rather than merely something similar.
        """
        X, kernel = _setup(bkd)
        weights = bkd.array(np.random.uniform(0.5, 1.5, 37))
        sqrt_w = bkd.sqrt(weights)
        op = KernelMatVecOperator(kernel, X, bkd, sqrt_weights=sqrt_w)
        vecs = bkd.array(np.random.normal(0.0, 1.0, (37, 3)))
        dense = (sqrt_w[:, None] * kernel(X, X)) * sqrt_w[None, :]
        bkd.assert_allclose(op.apply(vecs), dense @ vecs, rtol=1e-12)

    def test_unit_weights_reduce_to_unweighted(self, bkd) -> None:
        """W = I must be indistinguishable from passing no weights."""
        X, kernel = _setup(bkd)
        vecs = bkd.array(np.random.normal(0.0, 1.0, (37, 2)))
        plain = KernelMatVecOperator(kernel, X, bkd)
        unit = KernelMatVecOperator(
            kernel, X, bkd, sqrt_weights=bkd.full((37,), 1.0)
        )
        bkd.assert_allclose(unit.apply(vecs), plain.apply(vecs), rtol=1e-12)


class TestSpectrum:
    """The operator must carry the right eigenvalues, not just the right
    products.

    Agreement on ``K @ V`` for a few random ``V`` is necessary but does
    not pin the spectrum, and the spectrum is what every downstream
    eigensolver consumes. Applying the operator to a full identity
    recovers the matrix column by column, so its eigenvalues can be
    compared against a dense reference directly -- at full rank, with no
    truncation anywhere, so this is an equality rather than a tolerance
    on an approximation.
    """

    def test_recovers_exact_eigenvalues_unweighted(self, bkd) -> None:
        X, kernel = _setup(bkd)
        op = KernelMatVecOperator(kernel, X, bkd, block_size=8)
        recovered = op.apply(bkd.eye(37))
        bkd.assert_allclose(
            bkd.eigvalsh(recovered), bkd.eigvalsh(kernel(X, X)), rtol=1e-10
        )

    def test_recovers_exact_eigenvalues_weighted(self, bkd) -> None:
        """The weighted operator's spectrum is that of W^{1/2} K W^{1/2}.

        This is the spectrum the KLE eigensolvers are defined against,
        so an error in how the weights fold into the operator would
        silently shift every eigenvalue downstream.
        """
        X, kernel = _setup(bkd)
        np.random.seed(1)
        weights = bkd.array(np.random.uniform(0.5, 1.5, 37))
        sqrt_w = bkd.sqrt(weights)
        op = KernelMatVecOperator(
            kernel, X, bkd, sqrt_weights=sqrt_w, block_size=8
        )
        recovered = op.apply(bkd.eye(37))
        dense = (sqrt_w[:, None] * kernel(X, X)) * sqrt_w[None, :]
        bkd.assert_allclose(
            bkd.eigvalsh(recovered), bkd.eigvalsh(dense), rtol=1e-10
        )

    def test_recovered_matrix_is_symmetric(self, bkd) -> None:
        """Blockwise assembly must not break symmetry.

        A transposed index in the block loop would produce a matrix that
        is wrong but still gives plausible products, and non-symmetry is
        the cheapest way to detect it.
        """
        X, kernel = _setup(bkd)
        op = KernelMatVecOperator(kernel, X, bkd, block_size=8)
        recovered = op.apply(bkd.eye(37))
        bkd.assert_allclose(recovered, recovered.T, rtol=1e-12)


class TestSymmetricOperatorContract:
    """What the base class supplies must actually hold."""

    def test_satisfies_symmetric_operator(self, bkd) -> None:
        X, kernel = _setup(bkd)
        op = KernelMatVecOperator(kernel, X, bkd)
        assert isinstance(op, SymmetricMatVecOperator)

    def test_shape_accessors(self, bkd) -> None:
        X, kernel = _setup(bkd)
        op = KernelMatVecOperator(kernel, X, bkd)
        assert op.nvars() == 37
        assert op.nrows() == 37
        assert op.ncols() == 37

    def test_apply_transpose_equals_apply(self, bkd) -> None:
        """Inherited from the base; worth pinning that it is true here.

        It holds because a kernel covariance matrix is symmetric, and
        stays true under the weighted form since W^{1/2} K W^{1/2} is
        symmetric whenever K is.
        """
        X, kernel = _setup(bkd)
        weights = bkd.array(np.random.uniform(0.5, 1.5, 37))
        op = KernelMatVecOperator(
            kernel, X, bkd, sqrt_weights=bkd.sqrt(weights)
        )
        vecs = bkd.array(np.random.normal(0.0, 1.0, (37, 2)))
        bkd.assert_allclose(
            op.apply_transpose(vecs), op.apply(vecs), rtol=1e-12
        )


class TestMemoryBehaviour:
    """Blocking must actually bound the work per pass."""

    def test_never_evaluates_the_whole_matrix(self, numpy_bkd) -> None:
        """The point of the class: no single kernel call sees n x n.

        Without this, a refactor could assemble K and slice it, keeping
        every numerical test green while losing the entire benefit.
        """
        bkd = numpy_bkd
        X, kernel = _setup(bkd)
        largest = 0

        class _Watched:
            def __call__(self, X1, X2=None):
                nonlocal largest
                out = kernel(X1, X2)
                largest = max(largest, int(out.shape[0] * out.shape[1]))
                return out

            def diag(self, X):
                return kernel.diag(X)

        op = KernelMatVecOperator(_Watched(), X, bkd, block_size=8)
        op.apply(bkd.array(np.random.normal(0.0, 1.0, (37, 2))))
        assert largest <= 8 * 37


class TestValidation:
    """Bad input fails at construction or call, not silently."""

    def test_rejects_nonpositive_block_size(self, bkd) -> None:
        X, kernel = _setup(bkd)
        with pytest.raises(ValueError, match="block_size"):
            KernelMatVecOperator(kernel, X, bkd, block_size=0)

    def test_rejects_mismatched_weights(self, bkd) -> None:
        X, kernel = _setup(bkd)
        with pytest.raises(ValueError, match="but X has"):
            KernelMatVecOperator(
                kernel, X, bkd, sqrt_weights=bkd.full((10,), 1.0)
            )

    def test_rejects_2d_weights(self, bkd) -> None:
        X, kernel = _setup(bkd)
        with pytest.raises(ValueError, match="1D"):
            KernelMatVecOperator(
                kernel, X, bkd, sqrt_weights=bkd.full((37, 1), 1.0)
            )

    def test_rejects_1d_vecs(self, bkd) -> None:
        """Single vectors are (n, 1), per the shape conventions."""
        X, kernel = _setup(bkd)
        op = KernelMatVecOperator(kernel, X, bkd)
        with pytest.raises(ValueError, match="2D"):
            op.apply(bkd.full((37,), 1.0))

    def test_rejects_wrong_row_count(self, bkd) -> None:
        X, kernel = _setup(bkd)
        op = KernelMatVecOperator(kernel, X, bkd)
        with pytest.raises(ValueError, match="rows"):
            op.apply(bkd.full((10, 2), 1.0))
