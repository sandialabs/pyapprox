r"""Non-negative least squares, with an early-stopping variant.

Solves :math:`\min_{x \geq 0} \|A x - d\|_2`. The classic Lawson-Hanson
active-set method adds one variable at a time to the passive (free) set,
so its iterates grow from sparse to dense. Stopping as soon as the
relative residual :math:`\|A x - d\| / \|d\|` falls below a tolerance
therefore returns a *sparse* non-negative solution. That is the point
when the problem has an exact dense solution and only a sparse
approximate one is wanted, as in empirical quadrature, where the
all-ones vector reproduces the target exactly and the number of
nonzero weights is the online cost. A solver run to full convergence
returns the dense one.

Implementations satisfy :class:`NNLSSolverProtocol` and are injected, so
a greedy, linear-programming, streamed or randomized method replaces
another without changing its consumers:

- :class:`EarlyStoppingLawsonHanson`: Lawson-Hanson in backend
  operations, stopping at a relative-residual tolerance.
- :class:`ScipyNNLS`: ``scipy.optimize.nnls`` behind the same protocol,
  to full convergence (SciPy offers no residual-based stop).
"""

from dataclasses import dataclass
from typing import Generic, List, Optional, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@dataclass(frozen=True)
class NNLSResult(Generic[Array]):
    r"""The outcome of a non-negative least-squares solve.

    Attributes
    ----------
    weights : Array
        The non-negative solution. Shape: (ncols,).
    relative_residual : float
        :math:`\|A x - d\| / \|d\|` at the solution, 0 when ``d`` is 0.
    niterations : int
        Least-squares solves performed.
    converged : bool
        Whether the solver stopped for a legitimate reason: its tolerance
        was met or the solution is optimal. False when a cap (on nonzero
        weights or iterations) stopped it.
    """

    weights: Array
    relative_residual: float
    niterations: int
    converged: bool


@runtime_checkable
class NNLSSolverProtocol(Protocol, Generic[Array]):
    r"""Solves :math:`\min_{x \geq 0} \|A x - d\|_2`."""

    def solve(self, matrix: Array, target: Array) -> NNLSResult[Array]:
        """Return the solution for ``matrix`` (m, n) and ``target`` (m,)."""
        ...


def _validate(matrix: Array, target: Array) -> None:
    if matrix.ndim != 2 or target.ndim != 1:
        raise ValueError("matrix must be 2D and target 1D")
    if int(target.shape[0]) != int(matrix.shape[0]):
        raise ValueError(
            f"target has {int(target.shape[0])} entries but matrix has "
            f"{int(matrix.shape[0])} rows"
        )


class EarlyStoppingLawsonHanson(Generic[Array]):
    r"""Lawson-Hanson active-set NNLS with a relative-residual stop.

    Written in backend operations, so it runs on either backend. The
    choice of which variable enters the passive set is discrete and is
    made on host values; the weights are not differentiated through.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    relative_tolerance : float
        Stop once :math:`\|A x - d\| \leq` this times :math:`\|d\|`.
        Zero runs to the full NNLS solution.
    max_nonzero : int, optional
        Stop once this many variables are positive.
    max_iterations : int, optional
        Cap on least-squares solves; defaults to ``3 * ncols``.

    Raises
    ------
    ValueError
        If the tolerance is negative or a cap is below 1.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        relative_tolerance: float,
        max_nonzero: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        if relative_tolerance < 0.0:
            raise ValueError(
                f"relative_tolerance must be non-negative, got {relative_tolerance}"
            )
        for name, value in (
            ("max_nonzero", max_nonzero),
            ("max_iterations", max_iterations),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        self._bkd = bkd
        self._tolerance = relative_tolerance
        self._max_nonzero = max_nonzero
        self._max_iterations = max_iterations

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def solve(self, matrix: Array, target: Array) -> NNLSResult[Array]:
        """Return the non-negative solution, stopped at the tolerance.

        Raises
        ------
        ValueError
            If the shapes disagree.
        """
        _validate(matrix, target)
        bkd = self._bkd
        ncols = int(matrix.shape[1])
        weights = bkd.zeros((ncols,))
        target_norm = bkd.to_float(bkd.norm(target))
        if target_norm == 0.0:
            return NNLSResult(weights, 0.0, 0, True)
        max_iterations = (
            3 * ncols if self._max_iterations is None else self._max_iterations
        )
        # A gradient entry below this is roundoff, not a descent direction.
        gradient_floor = 1e-13 * bkd.to_float(bkd.norm(matrix)) * target_norm
        passive: List[int] = []
        niterations = 0
        # True once the solver stops for a legitimate reason: the tolerance
        # is met, or no variable can still decrease the residual (the KKT
        # conditions hold). The caps stop it without converging.
        converged = False
        while True:
            if self._relative_residual(matrix, target, weights) <= self._tolerance:
                converged = True
                break
            if self._max_nonzero is not None and len(passive) >= self._max_nonzero:
                break
            if niterations >= max_iterations:
                break
            gradient = bkd.to_numpy(
                bkd.dot(matrix.T, target - bkd.dot(matrix, weights))
            )
            candidates = [
                jj
                for jj in range(ncols)
                if jj not in passive and float(gradient[jj]) > gradient_floor
            ]
            if not candidates:
                converged = True
                break
            passive.append(max(candidates, key=lambda jj: float(gradient[jj])))
            while niterations < max_iterations:
                niterations += 1
                trial = self._passive_least_squares(matrix, target, passive, ncols)
                trial_np = bkd.to_numpy(trial)
                if all(float(trial_np[jj]) > 0.0 for jj in passive):
                    weights = trial
                    break
                # Step from the feasible iterate toward the trial until the
                # first passive variable reaches zero, then drop the zeros.
                weights_np = bkd.to_numpy(weights)
                step = min(
                    float(weights_np[jj])
                    / (float(weights_np[jj]) - float(trial_np[jj]))
                    for jj in passive
                    if float(trial_np[jj]) <= 0.0
                )
                weights = weights + step * (trial - weights)
                weights_np = bkd.to_numpy(weights)
                passive = [jj for jj in passive if float(weights_np[jj]) > 0.0]
                weights = self._zero_outside(weights, passive, ncols)
        residual = self._relative_residual(matrix, target, weights)
        return NNLSResult(weights, residual, niterations, converged)

    def _relative_residual(self, matrix: Array, target: Array, x: Array) -> float:
        bkd = self._bkd
        return bkd.to_float(bkd.norm(target - bkd.dot(matrix, x))) / bkd.to_float(
            bkd.norm(target)
        )

    def _passive_least_squares(
        self, matrix: Array, target: Array, passive: List[int], ncols: int
    ) -> Array:
        """Unconstrained least squares on the passive set, zero elsewhere."""
        bkd = self._bkd
        indices = bkd.asarray(passive, dtype=bkd.int64_dtype())
        local = bkd.lstsq(matrix[:, indices], target)
        return bkd.index_update(bkd.zeros((ncols,)), indices, local)

    def _zero_outside(self, weights: Array, passive: List[int], ncols: int) -> Array:
        mask = [1.0 if jj in passive else 0.0 for jj in range(ncols)]
        return weights * self._bkd.asarray(mask)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(relative_tolerance={self._tolerance}, "
            f"max_nonzero={self._max_nonzero})"
        )


class ScipyNNLS(Generic[Array]):
    """``scipy.optimize.nnls`` behind :class:`NNLSSolverProtocol`.

    A generic interface over a NumPy computation: the arrays are
    converted to NumPy on the way in and back to the backend on the way
    out. Not differentiable through. Runs to full convergence, since
    SciPy offers no residual-based stop, so the weights are generally
    dense; for sparse weights use :class:`EarlyStoppingLawsonHanson`.
    SciPy does not report its iteration count, so ``niterations`` is 0.

    Parameters
    ----------
    bkd : Backend
        Computational backend of the arrays passed and returned.
    max_iterations : int, optional
        Passed to SciPy as ``maxiter``; SciPy raises when it is reached.
    """

    def __init__(
        self, bkd: Backend[Array], max_iterations: Optional[int] = None
    ) -> None:
        self._bkd = bkd
        self._max_iterations = max_iterations

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def solve(self, matrix: Array, target: Array) -> NNLSResult[Array]:
        """Return the full NNLS solution.

        Raises
        ------
        ValueError
            If the shapes disagree.
        RuntimeError
            From SciPy, if ``max_iterations`` is reached.
        """
        from scipy.optimize import nnls

        _validate(matrix, target)
        bkd = self._bkd
        weights, residual_norm = nnls(
            bkd.to_numpy(matrix), bkd.to_numpy(target), maxiter=self._max_iterations
        )
        target_norm = bkd.to_float(bkd.norm(target))
        relative = 0.0 if target_norm == 0.0 else float(residual_norm) / target_norm
        return NNLSResult(bkd.asarray(weights), relative, 0, True)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(max_iterations={self._max_iterations})"
