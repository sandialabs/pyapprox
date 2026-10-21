r"""Krylov solvers generic over the stack: conjugate gradients and GMRES.

Both only apply the operator, so they pair with any format, including
:class:`~pyapprox.util.linalg.operators.formats.MatrixFree`, and run on
any stack through its inner-product space. Vectors go only through the
space; the reductions it returns (small arrays replicated on every
process) become Python floats through ``bkd``, which is where the
scalar work of the methods happens. That makes the solves **not
differentiable through**: an iterative solve stops at a tolerance, so
differentiating its iterations would give the derivative of the
algorithm, not of :math:`A^{-1} b`. Gradients come from the transposed
solve instead; a caller that needs a graph through the solve itself
injects a direct dense solver.

Convergence is decided on the true residual :math:`\|b - A x\|`,
recomputed after the iterations, against ``tol`` times :math:`\|b\|`,
so a solve whose recurrence drifts (conjugate gradients on a strongly
nonsymmetric operator, say) reports ``converged=False`` rather than a
wrong answer. A block of right-hand sides is solved one column at a
time, through the space's ``column`` and ``concatenate``.
"""

import math
from abc import abstractmethod
from typing import Generic, Optional, TypeVar

from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators.algebra import Transposed
from pyapprox.util.linalg.operators.formats import FormatProtocol
from pyapprox.util.linalg.operators.protocols import (
    ActionOperatorProtocol,
    InnerProductSpaceProtocol,
)
from pyapprox.util.linalg.operators.solvers import (
    SolveInfo,
    SolverCapabilities,
    SolverHandleBase,
    SolverHandleProtocol,
    merge_column_reports,
    require_square,
)

V = TypeVar("V")
# The small dense array type of the space's reductions, a backend array.
A = TypeVar("A", bound=ArrayProtocol)
# The format of an operator a Krylov solver applies but does not inspect.
G = TypeVar("G", bound=FormatProtocol)

_REAL_DTYPES = ("float32", "float64")


class _KrylovHandle(SolverHandleBase[V], Generic[V, A, G]):
    """What conjugate gradients and GMRES share: the checks, the initial
    guess, the true-residual test and the report. A subclass supplies the
    iterations and its transpose."""

    def __init__(
        self,
        op: ActionOperatorProtocol[V, G],
        space: InnerProductSpaceProtocol[V, A],
        bkd: Backend[A],
        tol: float,
        maxiter: int,
    ) -> None:
        super().__init__(op)
        self._op = op
        self._space = space
        self._bkd = bkd
        self._tol = tol
        self._maxiter = maxiter

    def _float(self, reduction: A) -> float:
        """A single replicated reduction as a Python float."""
        return self._bkd.to_float(self._bkd.sum(reduction))

    def _norm(self, x: V) -> float:
        return self._float(self._space.norm(x))

    def _dot(self, x: V, y: V) -> float:
        return self._float(self._space.dot(x, y))

    def _residual(self, b: V, x: V) -> V:
        """:math:`b - A x`."""
        return self._space.axpby(1.0, b, -1.0, self._op.apply(x))

    def solve(
        self, rhs: V, tol: Optional[float] = None, x0: Optional[V] = None
    ) -> tuple[V, SolveInfo]:
        r"""Solve :math:`A x = b`, each column to ``tol`` relative to its
        own :math:`\|b\|`.

        The columns are solved one after another, each from its column
        of ``x0``. The block converges when every column does; the report
        is merged by :func:`merge_column_reports`.

        Raises
        ------
        ValueError
            If ``x0`` has a different number of columns from ``rhs``.
        """
        space = self._space
        ncols = space.ncols(rhs)
        if x0 is not None and space.ncols(x0) != ncols:
            raise ValueError(
                f"x0 must have {ncols} columns like rhs, got {space.ncols(x0)}"
            )
        if ncols == 0:
            return space.zeros(0), SolveInfo(True, 0, 0.0)
        if ncols == 1:
            return self._solve_vector(rhs, tol, x0)
        results = [
            self._solve_vector(
                space.column(rhs, k),
                tol,
                None if x0 is None else space.column(x0, k),
            )
            for k in range(ncols)
        ]
        return space.concatenate([x for x, _ in results]), merge_column_reports(
            [info for _, info in results]
        )

    def _solve_vector(
        self, rhs: V, tol: Optional[float], x0: Optional[V]
    ) -> tuple[V, SolveInfo]:
        """Solve for a single right-hand side."""
        space = self._space
        tolerance = self._tol if tol is None else tol
        rhs_norm = self._norm(rhs)
        if rhs_norm == 0.0:
            return space.zeros(1), SolveInfo(True, 0, 0.0)
        x = space.zeros(1) if x0 is None else space.copy(x0)
        x, iterations = self._iterate(rhs, x, tolerance * rhs_norm)
        residual_norm = self._norm(self._residual(rhs, x))
        if not space.all_finite(x) or not math.isfinite(residual_norm):
            return x, SolveInfo(False, iterations, message="non-finite iterate")
        if residual_norm > tolerance * rhs_norm:
            return x, SolveInfo(
                False,
                iterations,
                residual_norm,
                f"relative residual {residual_norm / rhs_norm:.3e} above "
                f"{tolerance:.1e} after {iterations} iterations",
            )
        return x, SolveInfo(True, iterations, residual_norm)

    @abstractmethod
    def _iterate(self, b: V, x: V, atol: float) -> tuple[V, int]:
        """Improve ``x`` toward :math:`A x = b` until the residual norm is
        at most ``atol`` or the iterations run out; return it and the
        iterations taken."""
        ...


class _CGHandle(_KrylovHandle[V, A, G]):
    def _iterate(self, b: V, x: V, atol: float) -> tuple[V, int]:
        space = self._space
        r = self._residual(b, x)
        p = space.copy(r)
        rr = self._dot(r, r)
        for iteration in range(self._maxiter):
            if math.sqrt(rr) <= atol:
                return x, iteration
            ap = self._op.apply(p)
            curvature = self._dot(p, ap)
            if curvature == 0.0 or not math.isfinite(curvature):
                return x, iteration
            alpha = rr / curvature
            x = space.axpby(alpha, p, 1.0, x, out=x)
            r = space.axpby(-alpha, ap, 1.0, r, out=r)
            rr_next = self._dot(r, r)
            p = space.axpby(1.0, r, rr_next / rr, p, out=p)
            rr = rr_next
        return x, self._maxiter

    def transpose(self) -> "_CGHandle[V, A, G]":
        return _CGHandle(
            Transposed(self._op), self._space, self._bkd, self._tol, self._maxiter
        )


def _givens(a: float, b: float) -> tuple[float, float, float]:
    """``(c, s, r)`` with ``[[c, s], [-s, c]] @ [a, b] = [r, 0]``."""
    r = math.hypot(a, b)
    if r == 0.0:
        return 1.0, 0.0, 0.0
    return a / r, b / r, r


class _GMRESHandle(_KrylovHandle[V, A, G]):
    def __init__(
        self,
        op: ActionOperatorProtocol[V, G],
        space: InnerProductSpaceProtocol[V, A],
        bkd: Backend[A],
        tol: float,
        maxiter: int,
        restart: int,
    ) -> None:
        super().__init__(op, space, bkd, tol, maxiter)
        self._restart = restart

    def _iterate(self, b: V, x: V, atol: float) -> tuple[V, int]:
        total = 0
        while total < self._maxiter:
            x, total, done = self._cycle(b, x, atol, total)
            if done:
                break
        return x, total

    def _cycle(self, b: V, x: V, atol: float, total: int) -> tuple[V, int, bool]:
        """One restart cycle: Arnoldi with modified Gram-Schmidt, Givens
        rotations on the Hessenberg matrix, and the least-squares update.
        Returns the iterate, the iterations so far and whether to stop."""
        space = self._space
        r = self._residual(b, x)
        beta = self._norm(r)
        if beta <= atol:
            return x, total, True
        basis = [space.scale(1.0 / beta, r)]
        m = min(self._restart, self._maxiter - total)
        hessenberg = [[0.0] * m for _ in range(m + 1)]
        rotations: list[tuple[float, float]] = []
        g = [beta] + [0.0] * m
        steps, stop = 0, False
        for j in range(m):
            w = self._op.apply(basis[j])
            total += 1
            for i in range(j + 1):
                h = self._dot(basis[i], w)
                hessenberg[i][j] = h
                w = space.axpby(-h, basis[i], 1.0, w, out=w)
            next_norm = self._norm(w)
            hessenberg[j + 1][j] = next_norm
            for i, (c, s) in enumerate(rotations):
                upper, lower = hessenberg[i][j], hessenberg[i + 1][j]
                hessenberg[i][j] = c * upper + s * lower
                hessenberg[i + 1][j] = -s * upper + c * lower
            c, s, diagonal = _givens(hessenberg[j][j], hessenberg[j + 1][j])
            rotations.append((c, s))
            hessenberg[j][j], hessenberg[j + 1][j] = diagonal, 0.0
            g[j + 1], g[j] = -s * g[j], c * g[j]
            steps = j + 1
            if abs(g[j + 1]) <= atol or next_norm == 0.0 or diagonal == 0.0:
                stop = True
                break
            basis.append(space.scale(1.0 / next_norm, w))
        y = [0.0] * steps
        for i in reversed(range(steps)):
            if hessenberg[i][i] == 0.0:
                return x, total, True
            partial = sum(hessenberg[i][k] * y[k] for k in range(i + 1, steps))
            y[i] = (g[i] - partial) / hessenberg[i][i]
        for i in range(steps):
            x = space.axpby(y[i], basis[i], 1.0, x, out=x)
        return x, total, stop

    def transpose(self) -> "_GMRESHandle[V, A, G]":
        return _GMRESHandle(
            Transposed(self._op),
            self._space,
            self._bkd,
            self._tol,
            self._maxiter,
            self._restart,
        )


class _KrylovSolver(Generic[V, A]):
    """Setup shared by the Krylov solvers."""

    def __init__(
        self,
        space: InnerProductSpaceProtocol[V, A],
        bkd: Backend[A],
        tol: float,
        maxiter: int,
    ) -> None:
        if not isinstance(space, InnerProductSpaceProtocol):
            raise TypeError(
                "space must satisfy InnerProductSpaceProtocol, "
                f"got {type(space).__name__}"
            )
        if tol <= 0.0 or maxiter < 1:
            raise ValueError(f"need tol > 0 and maxiter >= 1, got {tol}, {maxiter}")
        self._space = space
        self._bkd = bkd
        self._tol = tol
        self._maxiter = maxiter

    def capabilities(self) -> SolverCapabilities:
        return SolverCapabilities(transpose=True)

    def _check(self, op: ActionOperatorProtocol[V, G]) -> None:
        name = type(self).__name__
        require_square(op, name)
        descriptor = self._space.descriptor()
        descriptor.require_same(op.domain(), f"domain of the operator given to {name}")
        descriptor.require_same(op.range(), f"range of the operator given to {name}")
        if descriptor.dtype not in _REAL_DTYPES:
            raise TypeError(f"{name} is for real scalars, got {descriptor.dtype}")


class ConjugateGradient(_KrylovSolver[V, A]):
    r"""Conjugate gradients, for symmetric positive definite operators.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.solvers.LinearSolverProtocol`
    for every format. It needs no stored matrix and keeps three vectors.
    The operator is not checked for symmetry, so a nearly symmetric one
    (a diffusion-dominated Newton matrix, a small time step) may still be
    given; one too far from symmetric fails the true-residual test and
    reports ``converged=False``. The transposed handle runs conjugate
    gradients on :math:`A^\top`.

    Parameters
    ----------
    space : InnerProductSpaceProtocol
        The operator's domain and range.
    bkd : Backend
        The backend of the space's reductions, for the scalar work.
    tol : float, default 1e-10
        Relative residual :math:`\|b - A x\| / \|b\|` to reach.
    maxiter : int, default 1000
        Iteration limit.
    """

    def __init__(
        self,
        space: InnerProductSpaceProtocol[V, A],
        bkd: Backend[A],
        tol: float = 1e-10,
        maxiter: int = 1000,
    ) -> None:
        super().__init__(space, bkd, tol, maxiter)

    def setup(self, op: ActionOperatorProtocol[V, G]) -> SolverHandleProtocol[V]:
        """Raises ``ValueError`` or ``SpaceMismatchError`` unless ``op`` is
        square on the solver's space, ``TypeError`` for complex scalars."""
        self._check(op)
        return _CGHandle(op, self._space, self._bkd, self._tol, self._maxiter)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(tol={self._tol}, maxiter={self._maxiter})"


class GMRES(_KrylovSolver[V, A]):
    r"""Restarted GMRES, for general nonsingular operators.

    Satisfies
    :class:`~pyapprox.util.linalg.operators.solvers.LinearSolverProtocol`
    for every format. Each cycle builds an orthonormal Krylov basis of up
    to ``restart`` vectors (modified Gram-Schmidt) and minimizes the
    residual over it; memory grows with ``restart``. The transposed
    handle runs GMRES on :math:`A^\top`.

    Parameters
    ----------
    space : InnerProductSpaceProtocol
        The operator's domain and range.
    bkd : Backend
        The backend of the space's reductions, for the scalar work.
    tol : float, default 1e-10
        Relative residual :math:`\|b - A x\| / \|b\|` to reach.
    maxiter : int, default 1000
        Limit on operator applications, over all cycles.
    restart : int, default 50
        Basis size per cycle.
    """

    def __init__(
        self,
        space: InnerProductSpaceProtocol[V, A],
        bkd: Backend[A],
        tol: float = 1e-10,
        maxiter: int = 1000,
        restart: int = 50,
    ) -> None:
        super().__init__(space, bkd, tol, maxiter)
        if restart < 1:
            raise ValueError(f"need restart >= 1, got {restart}")
        self._restart = restart

    def setup(self, op: ActionOperatorProtocol[V, G]) -> SolverHandleProtocol[V]:
        """Raises ``ValueError`` or ``SpaceMismatchError`` unless ``op`` is
        square on the solver's space, ``TypeError`` for complex scalars."""
        self._check(op)
        return _GMRESHandle(
            op, self._space, self._bkd, self._tol, self._maxiter, self._restart
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(tol={self._tol}, maxiter={self._maxiter}, "
            f"restart={self._restart})"
        )
