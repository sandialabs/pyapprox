"""Tests for KLE eigensolvers.

``DenseEigenSolver`` wraps behaviour that already existed, so the tests
that matter here pin the *convention* every solver must satisfy --
un-weighted, non-negative, descending, deterministically signed -- since
that is what the matrix-free solvers will be checked against next.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kernels.matern import ExponentialKernel
from pyapprox.surrogates.kle.eigensolvers import (
    DenseEigenSolver,
    KLEEigenSolverProtocol,
    finalize_eigenpairs,
)
from pyapprox.surrogates.kle.utils import (
    eigendecomposition_unweighted,
    eigendecomposition_weighted,
)


def _setup(bkd, npoints=40, lenscale=0.3):
    coords = bkd.array(np.linspace(0.0, 1.0, npoints)[None, :])
    kernel = ExponentialKernel(
        bkd.full((1,), lenscale), (0.01, 100.0), 1, bkd
    )
    return coords, kernel


def _weights(bkd, npoints=40, seed=0):
    np.random.seed(seed)
    return bkd.array(np.random.uniform(0.5, 1.5, npoints))


class TestBehaviourPreserved:
    """The dense solver must reproduce the path it replaces.

    This is the evidence that introducing the solver abstraction changed
    no numbers. Once the matrix-free solvers arrive they are checked
    against this one, so an unnoticed shift here would propagate.
    """

    @pytest.mark.parametrize("nterms", [5, 12])
    def test_matches_unweighted_library_path(self, bkd, nterms) -> None:
        coords, kernel = _setup(bkd)
        expected_vals, expected_vecs = eigendecomposition_unweighted(
            kernel(coords, coords), nterms, bkd
        )
        expected_vals = bkd.maximum(expected_vals, bkd.asarray([0.0]))
        vals, vecs = DenseEigenSolver(bkd).solve(kernel, coords, nterms)
        bkd.assert_allclose(vals, expected_vals, atol=1e-12, rtol=1e-8)
        bkd.assert_allclose(vecs, expected_vecs, atol=1e-12, rtol=1e-8)

    @pytest.mark.parametrize("nterms", [5, 12])
    def test_matches_weighted_library_path(self, bkd, nterms) -> None:
        coords, kernel = _setup(bkd)
        weights = _weights(bkd)
        expected_vals, expected_vecs = eigendecomposition_weighted(
            kernel(coords, coords), weights, nterms, bkd
        )
        expected_vals = bkd.maximum(expected_vals, bkd.asarray([0.0]))
        vals, vecs = DenseEigenSolver(bkd).solve(
            kernel, coords, nterms, quad_weights=weights
        )
        bkd.assert_allclose(vals, expected_vals, atol=1e-12, rtol=1e-8)
        bkd.assert_allclose(vecs, expected_vecs, atol=1e-12, rtol=1e-8)


class TestConvention:
    """What every solver must return, whatever its algorithm."""

    def test_satisfies_protocol(self, bkd) -> None:
        assert isinstance(DenseEigenSolver(bkd), KLEEigenSolverProtocol)

    def test_eigenvalues_descending_and_nonnegative(self, bkd) -> None:
        coords, kernel = _setup(bkd)
        vals, _ = DenseEigenSolver(bkd).solve(kernel, coords, 12)
        vals_np = bkd.to_numpy(vals)
        assert np.all(np.diff(vals_np) <= 1e-14)
        assert np.all(vals_np >= 0.0)

    def test_unweighted_eigenvectors_are_orthonormal(self, bkd) -> None:
        """Without weights the basis is orthonormal in the usual sense."""
        coords, kernel = _setup(bkd)
        _, vecs = DenseEigenSolver(bkd).solve(kernel, coords, 8)
        bkd.assert_allclose(vecs.T @ vecs, bkd.eye(8), atol=1e-10, rtol=0.0)

    def test_weighted_eigenvectors_orthonormal_under_weights(
        self, bkd
    ) -> None:
        r"""The weighted convention is :math:`\Phi^T W \Phi = I`.

        This is the property the un-weighting step in
        ``finalize_eigenpairs`` exists to establish, and it fails
        silently if that step is skipped: the eigenvectors would come
        back orthonormal in the symmetrized convention instead, which is
        the right shape and the wrong basis.
        """
        coords, kernel = _setup(bkd)
        weights = _weights(bkd)
        _, vecs = DenseEigenSolver(bkd).solve(
            kernel, coords, 8, quad_weights=weights
        )
        gram = vecs.T @ (weights[:, None] * vecs)
        bkd.assert_allclose(gram, bkd.eye(8), atol=1e-10, rtol=0.0)

    def test_all_terms_is_bit_reproducible(self, bkd) -> None:
        """Requesting every term must reproduce exactly.

        ``nterms == ncoords`` takes the full ``eigh`` path rather than
        Lanczos, and ``eigh`` is deterministic given identical input --
        measured at exactly 0.0 run-to-run. Pinning it isolates the
        shared postcondition: since the partial path below *is*
        nondeterministic while this one is not, the cause is the solver
        rather than ``finalize_eigenpairs`` or the sign convention.
        """
        coords, kernel = _setup(bkd, npoints=20)
        solver = DenseEigenSolver(bkd)
        first_vals, first_vecs = solver.solve(kernel, coords, 20)
        second_vals, second_vecs = solver.solve(kernel, coords, 20)
        bkd.assert_allclose(first_vals, second_vals, rtol=0.0, atol=0.0)
        bkd.assert_allclose(first_vecs, second_vecs, rtol=0.0, atol=0.0)

    def test_partial_solves_span_the_same_subspace(self, bkd) -> None:
        """A partial solve reproduces its *span*, not its columns.

        .. warning::
            This documents undesirable behaviour rather than endorsing
            it. Requesting fewer terms than points routes through
            Lanczos (``scipy eigsh``), which ARPACK seeds from an
            internal random vector that numpy's global seed does not
            reach -- so the conftest reproducibility fixture has no
            effect, and **two identical constructions of the same KLE
            return bases differing by O(1)**.

            Measured on three full-rank 40x40 matrices, eigenvectors
            differed between successive calls by 4.4e-01 to 8.3e-01
            while eigenvalues agreed to 5e-15. That includes a random
            SPD matrix with well-separated eigenvalues
            (``lambda_8 / lambda_1 = 0.61``), which disproves an earlier
            hypothesis that this was rotation within a degenerate
            subspace: there is no degeneracy there to rotate in. It is
            the random start.

            The consequence for callers is that a stored KLE basis will
            not reproduce column-wise against a recomputed one, so
            comparisons must be made at subspace level until this is
            fixed. ``test_fixed_start_vector_is_reproducible`` pins the
            fix; adopting it changes the basis for every existing
            partial-solve caller and so belongs in its own commit.

        The invariant available today is the span, checked through the
        projector, which is invariant to both rotation and sign.
        """
        coords, kernel = _setup(bkd)
        solver = DenseEigenSolver(bkd)
        _, first = solver.solve(kernel, coords, 8)
        _, second = solver.solve(kernel, coords, 8)
        bkd.assert_allclose(
            first @ first.T, second @ second.T, atol=1e-8, rtol=0.0
        )

    def test_fixed_start_vector_is_reproducible(self, numpy_bkd) -> None:
        """A deterministic ``v0`` removes the nondeterminism entirely.

        Pins the fix for the defect above so it is a passing target
        rather than a description. Measured across an exponential
        kernel, a squared-exponential kernel and a random SPD matrix at
        two truncation levels: eigenvectors reproduced to exactly 0.0,
        with eigenvalues accurate to ~1e-15 against a dense ``eigh``
        reference, and convergence in every case. The current random
        start therefore buys nothing.

        Exercised at the scipy level because the library does not yet
        pass ``v0``; that is the change this test exists to justify. A
        fixed-seed random draw is preferable to a constant vector, since
        a constant is orthogonal to any antisymmetric leading
        eigenvector and would stall there.
        """
        from scipy.sparse.linalg import eigsh

        bkd = numpy_bkd
        coords, kernel = _setup(bkd)
        kmat = bkd.to_numpy(kernel(coords, coords))
        start = np.random.RandomState(0).normal(0.0, 1.0, kmat.shape[0])
        first = eigsh(kmat, k=8, which="LM", v0=start)
        second = eigsh(kmat, k=8, which="LM", v0=start)
        np.testing.assert_array_equal(first[1], second[1])
        reference = np.sort(np.linalg.eigvalsh(kmat))[::-1][:8]
        assert np.abs(
            np.sort(first[0])[::-1] - reference
        ).max() / reference[0] < 1e-12


class TestFinalizeEigenpairs:
    """The postcondition, tested without going through a solver.

    Free-standing so it can be used by a Nystrom expansion, which does
    not fit the solve skeleton, and by external solvers implementing the
    protocol directly.
    """

    def test_clips_negative_eigenvalues(self, bkd) -> None:
        """A covariance is PSD, so negatives are rounding on true zeros.

        Left unclipped they become NaN when a KLE takes their square
        root to scale the basis, and the NaN reaches every field
        evaluation.

        .. note::
            **This test pins behaviour that is scheduled to change, and
            must be updated rather than preserved.** The clip currently
            absorbs two different things: rounding on a true zero, which
            is what this test exercises and is legitimate, and
            over-requesting more terms than the operator's numerical
            rank, which it turns into silent zero-variance modes. The
            planned replacement clips only within a machine-precision
            tolerance and raises beyond it.

            So do not read a passing test here as evidence that
            unconditional clipping is intended. Once that change lands,
            this test should assert the tolerance boundary -- clipped
            below it, raising above -- not the current
            accept-anything behaviour.
        """
        vals = bkd.array([2.0, 1.0, -1e-16])
        vecs = bkd.eye(3)
        out_vals, _ = finalize_eigenpairs(vals, vecs, None, 3, bkd)
        assert float(bkd.to_numpy(out_vals).min()) >= 0.0

    def test_sorts_descending(self, bkd) -> None:
        vals = bkd.array([0.5, 3.0, 1.0])
        vecs = bkd.eye(3)
        out_vals, _ = finalize_eigenpairs(vals, vecs, None, 3, bkd)
        bkd.assert_allclose(
            out_vals, bkd.array([3.0, 1.0, 0.5]), rtol=1e-14
        )

    def test_truncates_to_nterms(self, bkd) -> None:
        vals = bkd.array([0.5, 3.0, 1.0])
        vecs = bkd.eye(3)
        out_vals, out_vecs = finalize_eigenpairs(vals, vecs, None, 2, bkd)
        assert out_vals.shape[0] == 2
        assert out_vecs.shape[1] == 2

    def test_undoes_the_weighting(self, bkd) -> None:
        """Dividing by sqrt(w) is what returns the unweighted basis."""
        vals = bkd.array([2.0, 1.0])
        vecs = bkd.eye(2)
        sqrt_w = bkd.array([2.0, 4.0])
        _, out_vecs = finalize_eigenpairs(vals, vecs, sqrt_w, 2, bkd)
        expected = bkd.eye(2) / sqrt_w[:, None]
        # signs are fixed afterwards, so compare magnitudes
        bkd.assert_allclose(
            bkd.abs(out_vecs), bkd.abs(expected), rtol=1e-14
        )


class TestValidation:
    """Bad requests fail loudly rather than producing odd output."""

    def test_rejects_nterms_above_npoints(self, bkd) -> None:
        coords, kernel = _setup(bkd, npoints=10)
        with pytest.raises(ValueError, match="cannot exceed"):
            DenseEigenSolver(bkd).solve(kernel, coords, 11)

    def test_rejects_nonpositive_nterms(self, bkd) -> None:
        coords, kernel = _setup(bkd)
        with pytest.raises(ValueError, match="nterms"):
            DenseEigenSolver(bkd).solve(kernel, coords, 0)

    def test_rejects_mismatched_weights(self, bkd) -> None:
        coords, kernel = _setup(bkd, npoints=10)
        with pytest.raises(ValueError, match="but coords has"):
            DenseEigenSolver(bkd).solve(
                kernel, coords, 3, quad_weights=bkd.full((5,), 1.0)
            )

    def test_rejects_2d_weights(self, bkd) -> None:
        coords, kernel = _setup(bkd, npoints=10)
        with pytest.raises(ValueError, match="1D"):
            DenseEigenSolver(bkd).solve(
                kernel, coords, 3, quad_weights=bkd.full((10, 1), 1.0)
            )
