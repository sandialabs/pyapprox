"""Tests for the matrix-free KLE eigensolvers.

The central test is that every solver reproduces ``DenseEigenSolver``
on a problem small enough to solve densely. Eigenvectors are compared
as *subspaces* rather than column by column: near-degenerate
eigenvalues are routine for isotropic kernels on symmetric domains, and
within a degenerate subspace any rotation is an equally valid
eigenbasis, so a column-wise comparison is flaky for reasons unrelated
to the code.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kernels.matern import (
    ExponentialKernel,
    SquaredExponentialKernel,
)
from pyapprox.surrogates.kle.eigensolvers import (
    DenseEigenSolver,
    KLEEigenSolverProtocol,
    PivotedCholeskyEigenSolver,
    RandomizedEigenSolver,
)


def _coords(bkd, npoints=120):
    return bkd.array(np.linspace(0.0, 1.0, npoints)[None, :])


def _smooth_kernel(bkd, lenscale=0.3):
    """Squared exponential: rapidly decaying spectrum, low rank."""
    return SquaredExponentialKernel(
        bkd.full((1,), lenscale), (0.01, 100.0), 1, bkd
    )


def _rough_kernel(bkd, lenscale=0.3):
    """Exponential (Matern-1/2): slowly decaying spectrum."""
    return ExponentialKernel(bkd.full((1,), lenscale), (0.01, 100.0), 1, bkd)


def _weights(bkd, npoints=120, seed=0):
    np.random.seed(seed)
    return bkd.array(np.random.uniform(0.5, 1.5, npoints))


def _subspace_error(bkd, first, second):
    """Distance between the spans of two bases, via projectors.

    Invariant to rotation within a degenerate subspace and to sign, so
    it measures what is actually determined by the problem.
    """
    lhs = bkd.to_numpy(first @ first.T)
    rhs = bkd.to_numpy(second @ second.T)
    return float(np.abs(lhs - rhs).max())


def _relative_eigenvalue_error(bkd, vals, reference):
    ref = bkd.to_numpy(reference)
    return float(np.abs(bkd.to_numpy(vals) - ref).max() / ref.max())


class TestMatchesDenseSolver:
    """Every solver must reproduce the reference it will replace."""

    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("kernel_kind", ["smooth", "rough"])
    @pytest.mark.parametrize(
        "solver",
        [
            # each solver is given enough of its own accuracy knob to
            # converge; which knob, and how much, is algorithm specific
            lambda bkd: PivotedCholeskyEigenSolver(bkd, rank_multiplier=15.0),
            lambda bkd: RandomizedEigenSolver(bkd, npower_iters=4),
        ],
        ids=["pivoted", "randomized"],
    )
    def test_converges_to_dense(
        self, bkd, solver, kernel_kind, weighted
    ) -> None:
        """Given enough accuracy, every solver reproduces the reference.

        The assertion is convergence, not a ranking. Which solver is
        more accurate at *default* settings depends on the kernel and on
        the knob each default happens to set -- pivoted Cholesky at the
        default 4x rank leaves 1.5e-03 on a rough kernel purely because
        rank 32 truncates it -- so asserting a comparison would pin a
        default rather than a property of the algorithm.
        """
        coords = _coords(bkd)
        kernel = (
            _smooth_kernel(bkd)
            if kernel_kind == "smooth"
            else _rough_kernel(bkd)
        )
        kwargs = {"quad_weights": _weights(bkd)} if weighted else {}
        expected_vals, expected_vecs = DenseEigenSolver(bkd).solve(
            kernel, coords, 8, **kwargs
        )
        vals, vecs = solver(bkd).solve(kernel, coords, 8, **kwargs)
        assert _relative_eigenvalue_error(bkd, vals, expected_vals) < 1e-8
        assert _subspace_error(bkd, vecs, expected_vecs) < 1e-4

    def test_pivoted_cholesky_is_exact_at_full_rank(self, bkd) -> None:
        """At rank == npoints there is no truncation left to make.

        Isolates the QR-to-eigenpairs derivation from the low-rank
        approximation: if this fails the algebra is wrong, whereas a
        loose result at reduced rank only means the rank was too low for
        that kernel.
        """
        coords, kernel = _coords(bkd, 60), _rough_kernel(bkd)
        expected_vals, _ = DenseEigenSolver(bkd).solve(kernel, coords, 8)
        vals, _ = PivotedCholeskyEigenSolver(bkd, rank=60).solve(
            kernel, coords, 8
        )
        assert _relative_eigenvalue_error(bkd, vals, expected_vals) < 1e-12


class TestAccuracyImprovesWithEffort:
    """Each solver's accuracy knob must actually buy accuracy.

    Convergence is the property worth pinning. How accurate a solver is
    at its *default* setting depends on the kernel and on what that
    default happens to be, so an absolute threshold there would assert a
    parameter choice rather than anything about the algorithm.
    """

    def test_pivoted_cholesky_improves_with_rank(self, numpy_bkd) -> None:
        """More factorization rank, less truncation error.

        Exercised on a slowly decaying spectrum, where the rank
        genuinely binds: a squared exponential is resolved at any of
        these ranks and would measure round-off throughout, testing
        nothing.
        """
        bkd = numpy_bkd
        coords, kernel = _coords(bkd), _rough_kernel(bkd)
        expected_vals, _ = DenseEigenSolver(bkd).solve(kernel, coords, 8)
        errors = [
            _relative_eigenvalue_error(
                bkd,
                PivotedCholeskyEigenSolver(
                    bkd, rank_multiplier=multiplier
                ).solve(kernel, coords, 8)[0],
                expected_vals,
            )
            for multiplier in (2.0, 4.0, 8.0)
        ]
        assert errors[0] > errors[1] > errors[2], (
            f"errors {errors} did not decrease as the factorization rank "
            "grew; the low-rank approximation is not converging"
        )

    def test_randomized_improves_with_power_iterations(
        self, numpy_bkd
    ) -> None:
        """More power iterations, closer to the dense reference.

        The randomized solver's knob is iteration count rather than
        rank. Also a regression guard: before
        ``randomized_symmetric_eigendecomposition`` was corrected, the
        sketch collapsed a little further on each iteration and this
        sequence *increased*.
        """
        bkd = numpy_bkd
        coords, kernel = _coords(bkd), _rough_kernel(bkd)
        expected_vals, _ = DenseEigenSolver(bkd).solve(kernel, coords, 8)
        errors = [
            _relative_eigenvalue_error(
                bkd,
                RandomizedEigenSolver(bkd, npower_iters=npower_iters).solve(
                    kernel, coords, 8
                )[0],
                expected_vals,
            )
            for npower_iters in (0, 1, 2)
        ]
        assert errors[0] > errors[1] > errors[2], (
            f"errors {errors} did not decrease as power iterations grew"
        )

    def test_more_terms_than_rank_is_refused(self, bkd) -> None:
        """Requesting more modes than the factorization holds.

        ``rank_for`` clamps the rank to at least ``nterms``, so this
        cannot silently return fewer terms than asked for.
        """
        solver = PivotedCholeskyEigenSolver(bkd, rank=4)
        assert solver.rank_for(8, 120) == 8


class TestConvention:
    """The postcondition holds whatever the algorithm."""

    @pytest.mark.parametrize(
        "solver_cls", [PivotedCholeskyEigenSolver, RandomizedEigenSolver]
    )
    def test_satisfies_protocol(self, bkd, solver_cls) -> None:
        assert isinstance(solver_cls(bkd), KLEEigenSolverProtocol)

    @pytest.mark.parametrize(
        "solver_cls", [PivotedCholeskyEigenSolver, RandomizedEigenSolver]
    )
    def test_returns_requested_shape(self, bkd, solver_cls) -> None:
        """Oversampling is internal and must not reach the caller.

        The solvers factorize to a rank above ``nterms``; returning
        that wider result would be the right shape for nobody and, on
        the pivoted path, silently kept the *smallest* eigenvalues
        because the shared sort assumes exactly ``nterms`` columns.
        """
        coords, kernel = _coords(bkd), _smooth_kernel(bkd)
        vals, vecs = solver_cls(bkd).solve(kernel, coords, 8)
        assert vals.shape == (8,)
        assert vecs.shape == (120, 8)

    @pytest.mark.parametrize(
        "solver_cls", [PivotedCholeskyEigenSolver, RandomizedEigenSolver]
    )
    def test_eigenvalues_descending_and_nonnegative(
        self, bkd, solver_cls
    ) -> None:
        coords, kernel = _coords(bkd), _smooth_kernel(bkd)
        vals, _ = solver_cls(bkd).solve(kernel, coords, 8)
        vals_np = bkd.to_numpy(vals)
        assert np.all(np.diff(vals_np) <= 1e-12)
        assert np.all(vals_np >= 0.0)

    @pytest.mark.parametrize(
        "solver_cls", [PivotedCholeskyEigenSolver, RandomizedEigenSolver]
    )
    def test_weighted_eigenvectors_orthonormal_under_weights(
        self, bkd, solver_cls
    ) -> None:
        r"""The weighted convention is :math:`\Phi^T W \Phi = I`.

        Checks that each solver applies the weights where its own
        algorithm requires -- inside a column operator for the pivoted
        path, inside the matvec operator for the randomized one -- and
        that the shared postcondition then undoes them consistently.
        """
        coords, kernel = _coords(bkd), _smooth_kernel(bkd)
        weights = _weights(bkd)
        _, vecs = solver_cls(bkd).solve(
            kernel, coords, 6, quad_weights=weights
        )
        gram = vecs.T @ (weights[:, None] * vecs)
        bkd.assert_allclose(gram, bkd.eye(6), atol=1e-6, rtol=0.0)


class TestValidation:
    """Bad configuration fails at construction."""

    def test_rejects_rank_multiplier_below_one(self, bkd) -> None:
        """Rank below nterms cannot supply the requested modes."""
        with pytest.raises(ValueError, match="rank_multiplier"):
            PivotedCholeskyEigenSolver(bkd, rank_multiplier=0.5)

    def test_rejects_nonpositive_rank(self, bkd) -> None:
        with pytest.raises(ValueError, match="rank"):
            PivotedCholeskyEigenSolver(bkd, rank=0)

    def test_rejects_negative_oversampling(self, bkd) -> None:
        with pytest.raises(ValueError, match="noversampling"):
            RandomizedEigenSolver(bkd, noversampling=-1)

    def test_rejects_negative_power_iterations(self, bkd) -> None:
        with pytest.raises(ValueError, match="npower_iters"):
            RandomizedEigenSolver(bkd, npower_iters=-1)
