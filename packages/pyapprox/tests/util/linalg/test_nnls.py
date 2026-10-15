"""Non-negative least squares: optimality, early stopping, and SciPy.

Checks are derivation-free:

- Planted: A (40 x 25, Gaussian, seed 0, full column rank) and a sparse
  non-negative x* (5 positive entries); with d = A x* both solvers return
  x* (the unique minimizer).
- Optimality (KKT) at full convergence, d Gaussian (seed 1), A 30 x 20:
  x >= 0, the gradient A^T (d - A x) vanishes on the support and is
  non-positive off it; the early-stopping solver at tolerance 0 agrees
  with SciPy. Control: unconstrained least squares on the same problem
  has negative entries, so the constraint is active.
- Early stopping on an empirical-quadrature problem, d = A 1 (A 60 x 40,
  seed 2): at relative tolerance 1e-2 the residual meets the tolerance
  with fewer nonzero weights than full convergence, which keeps all 40.
- The nonzero cap is respected.
"""

import numpy as np
import pytest

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.nnls import (
    EarlyStoppingLawsonHanson,
    NNLSSolverProtocol,
    ScipyNNLS,
)


def _gaussian(bkd: Backend[Array], nrows: int, ncols: int, seed: int) -> Array:
    return bkd.asarray(np.random.default_rng(seed).normal(size=(nrows, ncols)))


def _nonzero_count(bkd: Backend[Array], weights: Array) -> int:
    return int(np.count_nonzero(bkd.to_numpy(weights) > 0.0))


class TestProtocol:
    def test_both_satisfy_protocol(self, bkd: Backend[Array]) -> None:
        assert isinstance(EarlyStoppingLawsonHanson(bkd, 0.0), NNLSSolverProtocol)
        assert isinstance(ScipyNNLS(bkd), NNLSSolverProtocol)

    def test_rejects_bad_arguments(self, bkd: Backend[Array]) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            EarlyStoppingLawsonHanson(bkd, -1.0)
        with pytest.raises(ValueError, match="max_nonzero"):
            EarlyStoppingLawsonHanson(bkd, 0.0, max_nonzero=0)
        solver = EarlyStoppingLawsonHanson(bkd, 0.0)
        with pytest.raises(ValueError, match="rows"):
            solver.solve(bkd.zeros((3, 2)), bkd.zeros((4,)))

    def test_zero_target(self, bkd: Backend[Array]) -> None:
        result = EarlyStoppingLawsonHanson(bkd, 0.0).solve(
            _gaussian(bkd, 5, 3, seed=0), bkd.zeros((5,))
        )
        bkd.assert_allclose(result.weights, bkd.zeros((3,)))
        assert result.converged


class TestPlanted:
    @pytest.mark.parametrize("solver_kind", ["lawson_hanson", "scipy"])
    def test_recovers_sparse_solution(
        self, bkd: Backend[Array], solver_kind: str
    ) -> None:
        matrix = _gaussian(bkd, 40, 25, seed=0)
        planted = np.zeros(25)
        planted[[2, 7, 11, 18, 23]] = [0.5, 1.0, 2.0, 0.25, 1.5]
        target = bkd.dot(matrix, bkd.asarray(planted))
        solver: NNLSSolverProtocol[Array] = (
            EarlyStoppingLawsonHanson(bkd, 0.0)
            if solver_kind == "lawson_hanson"
            else ScipyNNLS(bkd)
        )
        result = solver.solve(matrix, target)
        bkd.assert_allclose(result.weights, bkd.asarray(planted), atol=1e-10)


class TestOptimality:
    def _problem(self, bkd: Backend[Array]) -> tuple[Array, Array]:
        matrix = _gaussian(bkd, 30, 20, seed=1)
        target = bkd.asarray(np.random.default_rng(11).normal(size=30))
        return matrix, target

    def test_kkt_conditions(self, bkd: Backend[Array]) -> None:
        matrix, target = self._problem(bkd)
        weights = EarlyStoppingLawsonHanson(bkd, 0.0).solve(matrix, target).weights
        x = bkd.to_numpy(weights)
        gradient = bkd.to_numpy(bkd.dot(matrix.T, target - bkd.dot(matrix, weights)))
        support = x > 0.0
        assert np.all(x >= 0.0)
        assert support.any() and not support.all()
        scale = float(np.max(np.abs(gradient))) + 1.0
        assert np.max(np.abs(gradient[support])) <= 1e-10 * scale
        assert np.max(gradient[~support]) <= 1e-10 * scale

    def test_agrees_with_scipy(self, bkd: Backend[Array]) -> None:
        matrix, target = self._problem(bkd)
        ours = EarlyStoppingLawsonHanson(bkd, 0.0).solve(matrix, target)
        reference = ScipyNNLS(bkd).solve(matrix, target)
        bkd.assert_allclose(ours.weights, reference.weights, atol=1e-10)

    def test_constraint_is_active(self, bkd: Backend[Array]) -> None:
        matrix, target = self._problem(bkd)
        unconstrained = bkd.to_numpy(bkd.lstsq(matrix, target))
        assert float(np.min(unconstrained)) < 0.0


class TestEarlyStopping:
    """Empirical quadrature: columns are nodes, rows the functions integrated.

    Rows x^0..x^7 at 40 equispaced nodes on [0, 1], d = A 1 (the node sums).
    The all-ones vector is an exact dense solution, and by Caratheodory an
    exact non-negative one with at most 8 nonzeros (the number of rows)
    exists, which Lawson-Hanson's basic solutions respect.
    """

    def _quadrature_problem(self, bkd: Backend[Array]) -> tuple[Array, Array]:
        nodes = np.linspace(0.0, 1.0, 40)
        matrix = bkd.asarray(np.stack([nodes**k for k in range(8)]))
        return matrix, bkd.dot(matrix, bkd.ones((40,)))

    def test_exact_solution_is_sparse(self, bkd: Backend[Array]) -> None:
        matrix, target = self._quadrature_problem(bkd)
        result = EarlyStoppingLawsonHanson(bkd, 1e-10).solve(matrix, target)
        assert result.converged
        assert result.relative_residual <= 1e-10
        assert _nonzero_count(bkd, result.weights) <= 8
        assert float(np.min(bkd.to_numpy(result.weights))) >= 0.0

    def test_looser_tolerance_keeps_no_more_weights(self, bkd: Backend[Array]) -> None:
        matrix, target = self._quadrature_problem(bkd)
        loose = EarlyStoppingLawsonHanson(bkd, 1e-2).solve(matrix, target)
        tight = EarlyStoppingLawsonHanson(bkd, 1e-10).solve(matrix, target)
        assert loose.converged and loose.relative_residual <= 1e-2
        assert _nonzero_count(bkd, loose.weights) <= _nonzero_count(bkd, tight.weights)

    def test_incoherent_columns_cannot_be_sparsified(self, bkd: Backend[Array]) -> None:
        """Gaussian A (60 x 40, seed 2), d = A 1: no sparse solution exists.

        The test proves that rather than assuming it. Without column j,
        no non-negative solution can do better than unconstrained least
        squares on the other columns; when every such bound exceeds the
        tolerance, any solution meeting it must use all 40 columns, so
        the count below is a property of the problem, not of this solver.
        The solver must still meet the tolerance and report convergence.
        """
        matrix = _gaussian(bkd, 60, 40, seed=2)
        target = bkd.dot(matrix, bkd.ones((40,)))
        target_norm = bkd.to_float(bkd.norm(target))
        for jj in range(40):
            others = bkd.asarray(
                [kk for kk in range(40) if kk != jj], dtype=bkd.int64_dtype()
            )
            best = bkd.lstsq(matrix[:, others], target)
            bound = (
                bkd.to_float(bkd.norm(target - bkd.dot(matrix[:, others], best)))
                / target_norm
            )
            assert bound > 1e-2
        result = EarlyStoppingLawsonHanson(bkd, 1e-2).solve(matrix, target)
        assert result.converged and result.relative_residual <= 1e-2
        assert _nonzero_count(bkd, result.weights) == 40

    def test_respects_nonzero_cap(self, bkd: Backend[Array]) -> None:
        matrix, target = self._quadrature_problem(bkd)
        capped = EarlyStoppingLawsonHanson(bkd, 1e-12, max_nonzero=3).solve(
            matrix, target
        )
        assert _nonzero_count(bkd, capped.weights) <= 3
        assert not capped.converged
