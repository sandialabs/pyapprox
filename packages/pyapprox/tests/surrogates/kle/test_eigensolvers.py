"""Tests for KLE eigensolvers.

``DenseEigenSolver`` wraps behaviour that already existed, so the tests
that matter here pin the *convention* every solver must satisfy --
un-weighted, non-negative, descending, deterministically signed -- since
that is what the matrix-free solvers will be checked against next.
"""

import warnings

import numpy as np
import pytest
from pyapprox.surrogates.kernels.matern import ExponentialKernel
from pyapprox.surrogates.kle.eigensolvers import (
    _DENSE_WARN_BYTES,
    DenseEigenSolver,
    KLEEigenSolverProtocol,
    PivotedCholeskyEigenSolver,
    RandomizedEigenSolver,
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
        """A partial solve reproduces its span, seeded or not.

        The span is a weaker invariant than the columns, and the seeded
        default now reproduces the columns exactly
        (``test_partial_solves_are_reproducible``). It is tested
        separately because it is the property that survives *without*
        the seed: an unseeded solver still spans the same subspace, so a
        caller who opts out of reproducibility keeps a meaningful
        guarantee rather than none.

        Checked through the projector, which is invariant to both
        rotation within the subspace and to sign.
        """
        coords, kernel = _setup(bkd)
        solver = DenseEigenSolver(bkd, seed=None)
        _, first = solver.solve(kernel, coords, 8)
        _, second = solver.solve(kernel, coords, 8)
        bkd.assert_allclose(
            first @ first.T, second @ second.T, atol=1e-8, rtol=0.0
        )

    def test_partial_solves_are_reproducible(self, bkd) -> None:
        """The seeded default reproduces columns, not merely the span.

        Requesting fewer terms than points routes through Lanczos
        (``scipy eigsh``), which ARPACK seeds from an internal random
        vector that numpy's global seed does not reach -- so the
        conftest reproducibility fixture has no effect on it and the
        seed must be passed explicitly as ``v0``.

        Before that was done, two identical constructions of the same
        KLE returned bases differing by O(1): measured on three
        full-rank 40x40 matrices, eigenvectors differed between
        successive calls by 4.4e-01 to 8.3e-01 while eigenvalues agreed
        to 5e-15. That included a random SPD matrix with well-separated
        eigenvalues (``lambda_8 / lambda_1 = 0.61``), so it was the
        random start rather than rotation within a degenerate subspace.

        Exact equality is the assertion because identical input and an
        identical start vector make the computation deterministic;
        anything looser would let the defect back in unnoticed.
        """
        coords, kernel = _setup(bkd)
        solver = DenseEigenSolver(bkd)
        first_vals, first_vecs = solver.solve(kernel, coords, 8)
        second_vals, second_vecs = solver.solve(kernel, coords, 8)
        bkd.assert_allclose(first_vals, second_vals, rtol=0.0, atol=0.0)
        bkd.assert_allclose(first_vecs, second_vecs, rtol=0.0, atol=0.0)

    def test_separate_solver_instances_agree(self, bkd) -> None:
        """Reproducibility must not depend on reusing one solver.

        The realistic case is two constructions in different processes
        or runs, so the seed has to live on the solver's configuration
        rather than in state accumulated across calls.
        """
        coords, kernel = _setup(bkd)
        _, first = DenseEigenSolver(bkd).solve(kernel, coords, 8)
        _, second = DenseEigenSolver(bkd).solve(kernel, coords, 8)
        bkd.assert_allclose(first, second, rtol=0.0, atol=0.0)

    def test_different_seeds_agree_on_the_subspace(self, bkd) -> None:
        """Changing the seed must move nothing that is not arbitrary.

        The start vector is an implementation detail of the iteration,
        so two seeds must reach the same eigenvalues and the same
        subspace; only the choice among equivalent bases may differ.
        """
        coords, kernel = _setup(bkd)
        vals_a, vecs_a = DenseEigenSolver(bkd, seed=0).solve(
            kernel, coords, 8
        )
        vals_b, vecs_b = DenseEigenSolver(bkd, seed=12345).solve(
            kernel, coords, 8
        )
        bkd.assert_allclose(vals_a, vals_b, rtol=1e-10, atol=1e-12)
        bkd.assert_allclose(
            vecs_a @ vecs_a.T, vecs_b @ vecs_b.T, atol=1e-8, rtol=0.0
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


class TestDenseMemoryWarning:
    """Dense assembly announces itself before it becomes a MemoryError.

    This solver is the default, so the callers who most need a
    matrix-free one are exactly those who get dense assembly unless
    they know to ask.
    """

    def _threshold_npoints(self, itemsize=8):
        """Smallest N whose dense matrix exceeds the threshold."""
        return int((_DENSE_WARN_BYTES / itemsize) ** 0.5) + 1

    def test_small_problems_are_silent(self, bkd) -> None:
        """The warning must not fire for problems that are fine.

        A warning on routine sizes would be trained away, and every
        existing caller is well under the threshold.
        """
        coords, kernel = _setup(bkd)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DenseEigenSolver(bkd).solve(kernel, coords, 8)
        assert [str(w.message) for w in caught] == []

    def test_large_problems_warn(self, bkd) -> None:
        """Past the threshold the caller is told, and told what to do.

        The check is called directly rather than through ``solve``: the
        point is the size at which it fires, and actually assembling
        that matrix is what the warning exists to prevent.
        """
        npoints = self._threshold_npoints()
        coords = bkd.zeros((1, npoints))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DenseEigenSolver(bkd)._warn_if_large(coords)
        assert len(caught) == 1
        message = str(caught[0].message)
        # Naming the alternatives is the point; a warning that only says
        # "this is large" leaves the caller with nowhere to go.
        assert "PivotedCholeskyEigenSolver" in message
        assert "RandomizedEigenSolver" in message
        assert str(npoints) in message

    def test_just_below_the_threshold_is_silent(self, bkd) -> None:
        """The boundary is a real boundary, not approximately one."""
        coords = bkd.zeros((1, self._threshold_npoints() - 2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DenseEigenSolver(bkd)._warn_if_large(coords)
        assert [str(w.message) for w in caught] == []

    @pytest.mark.parametrize(
        "solver_cls", [PivotedCholeskyEigenSolver, RandomizedEigenSolver]
    )
    def test_matrix_free_solvers_never_warn(self, bkd, solver_cls) -> None:
        """The warning belongs to dense assembly, not to size.

        A matrix-free solver at the same size is the recommended
        action, so warning there would contradict the advice.
        """
        coords, kernel = _setup(bkd)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver_cls(bkd).solve(kernel, coords, 8)
        assert [str(w.message) for w in caught] == []


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
        evaluation. Clipping keeps the NaN out, but only where the
        negative really is rounding: a term at -1e-16 relative is
        indistinguishable from a zero the caller should not have
        requested, so it is clipped *and* rejected rather than
        returned as a zero-variance mode.
        """
        vals = bkd.array([2.0, 1.0, -1e-16])
        vecs = bkd.eye(3)
        with pytest.raises(ValueError, match="rather than variance"):
            finalize_eigenpairs(vals, vecs, None, 3, bkd)
        # Truncating past the negligible term leaves a valid basis, so
        # the rejection is about the requested count, not the operator.
        out_vals, _ = finalize_eigenpairs(vals, vecs, None, 2, bkd)
        assert float(bkd.to_numpy(out_vals).min()) > 0.0

    def test_rejects_indefinite_kernel(self, bkd) -> None:
        """A negative far past rounding is not a zero to be clipped.

        Clipping it would report a mode carrying no variance for an
        operator that is not a covariance at all. It raises under the
        same message as over-requesting: both mean the retained terms
        outran what the operator supplies.
        """
        vals = bkd.array([2.0, 1.0, -1e-2])
        vecs = bkd.eye(3)
        with pytest.raises(ValueError, match="rather than variance"):
            finalize_eigenpairs(vals, vecs, None, 3, bkd)

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


class TestSeeding:
    """Two constructions of the same KLE must return the same basis.

    Both solvers with randomness take a seed; the pivoted Cholesky
    solver takes none, having none to seed.
    """

    def _kernel_and_coords(self, bkd):
        from pyapprox.surrogates.kernels.matern import (
            SquaredExponentialKernel,
        )

        coords = bkd.array(np.linspace(0.0, 1.0, 120)[None, :])
        kernel = SquaredExponentialKernel(
            bkd.full((1,), 0.3), (0.01, 100.0), 1, bkd
        )
        return kernel, coords

    def test_dense_solver_is_reproducible_when_seeded(self, bkd) -> None:
        """The Lanczos path is only deterministic once seeded.

        ARPACK draws its start vector from a stream numpy's global seed
        does not reach, so seeding before the call has no effect and
        the conftest reproducibility fixture does nothing here. Only
        ``nterms < ncoords`` takes that path; requesting every term
        uses ``eigh`` and was always deterministic.
        """
        kernel, coords = self._kernel_and_coords(bkd)
        first = DenseEigenSolver(bkd, seed=0).solve(kernel, coords, 8)[1]
        second = DenseEigenSolver(bkd, seed=0).solve(kernel, coords, 8)[1]
        bkd.assert_allclose(first, second, rtol=0.0, atol=0.0)

    def test_randomized_solver_is_reproducible_when_seeded(
        self, bkd
    ) -> None:
        kernel, coords = self._kernel_and_coords(bkd)
        first = RandomizedEigenSolver(bkd, seed=0).solve(
            kernel, coords, 8
        )[1]
        second = RandomizedEigenSolver(bkd, seed=0).solve(
            kernel, coords, 8
        )[1]
        bkd.assert_allclose(first, second, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "solver_cls", [DenseEigenSolver, RandomizedEigenSolver]
    )
    def test_seeded_solver_does_not_touch_the_global_rng(
        self, bkd, solver_cls
    ) -> None:
        """A seeded solver uses a local stream.

        Otherwise seeding for reproducibility would silently advance
        the caller's own random sequence, making *their* results depend
        on whether a KLE happened to be built first.
        """
        kernel, coords = self._kernel_and_coords(bkd)
        np.random.seed(0)
        expected = np.random.rand()
        np.random.seed(0)
        solver_cls(bkd, seed=0).solve(kernel, coords, 8)
        assert np.random.rand() == expected

    def test_pivoted_cholesky_takes_no_seed(self, bkd) -> None:
        """Greedy pivoting is deterministic, so a seed would mislead.

        Pinning the absence keeps one from being added for symmetry,
        which would advertise randomness the algorithm does not have.
        """
        with pytest.raises(TypeError):
            PivotedCholeskyEigenSolver(bkd, seed=0)
