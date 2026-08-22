"""Eigensolvers for Karhunen-Loeve expansions.

A KLE needs the leading eigenpairs of a kernel covariance operator. How
those are computed -- dense assembly, pivoted Cholesky, randomized
subspace iteration -- varies independently of what a KLE *is*, so the
solver is injected rather than expressed as a KLE subclass.

Solvers take a kernel and coordinates rather than an assembled matrix.
That is deliberate: a matrix-free solver must never be handed the very
matrix it exists to avoid. :class:`DenseEigenSolver` assembles
internally.
"""

import math
from abc import ABC, abstractmethod
from typing import Generic, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from pyapprox.surrogates.kernels.protocols import KernelProtocol
from pyapprox.surrogates.kle.utils import (
    adjust_sign_eig,
    eigendecomposition_unweighted,
    sort_eigenpairs,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.kernel_operators import KernelMatVecOperator
from pyapprox.util.linalg.pivoted_cholesky import (
    ColumnOperatorProtocol,
    KernelColumnOperator,
    PivotedCholeskyFactorizer,
)
from pyapprox.util.linalg.randomized import (
    randomized_symmetric_eigendecomposition,
)

_MACHINE_EPS = float(np.finfo(float).eps)


@runtime_checkable
class KLEEigenSolverProtocol(Protocol, Generic[Array]):
    """Computes the leading eigenpairs of a kernel covariance operator.

    Implementations must return eigenpairs in the **unweighted
    convention**: eigenvectors orthonormal under the quadrature weight
    inner product, eigenvalues non-negative and in descending order,
    with a deterministic sign. :func:`finalize_eigenpairs` establishes
    all of that and is available to implementations that do not inherit
    from the shipped base class.
    """

    def solve(
        self,
        kernel: KernelProtocol[Array],
        coords: Array,
        nterms: int,
        quad_weights: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Return the leading ``nterms`` eigenpairs.

        Parameters
        ----------
        kernel : KernelProtocol[Array]
            Covariance kernel.
        coords : Array
            Collocation points, shape ``(ndim, N)``.
        nterms : int
            Number of eigenpairs to return.
        quad_weights : Array, optional
            Quadrature weights, shape ``(N,)``. When None the plain
            eigenproblem is solved.

        Returns
        -------
        eig_vals : Array
            Shape ``(nterms,)``, descending, non-negative.
        eig_vecs : Array
            Shape ``(N, nterms)``, orthonormal under the quadrature
            inner product.
        """
        ...


def finalize_eigenpairs(
    eig_vals: Array,
    eig_vecs: Array,
    sqrt_weights: Optional[Array],
    nterms: int,
    bkd: Backend[Array],
) -> Tuple[Array, Array]:
    r"""Put raw eigenpairs into the convention every caller expects.

    Given eigenpairs of the symmetrized operator
    :math:`W^{1/2} K W^{1/2}`, undo the weighting, clip negative
    eigenvalues to zero, sort descending and fix signs.

    Public and free-standing rather than a method on a base class,
    because two callers need it without fitting the base class's
    ``solve`` skeleton: a Nystrom expansion produces an extension matrix
    alongside its eigenpairs, and an external solver satisfying the
    protocol directly should not have to reimplement the convention or
    reach into a private base.

    Every step here fails *silently* when skipped. Omit the
    un-weighting and the eigenvectors come back in the symmetrized
    convention; omit the sort and "leading" terms are not the leading
    ones; omit the sign fix and results differ between platforms. None
    of those raises, and all produce output of the right shape. That is
    why the step is centralized rather than left to each solver.

    The clip needs its own explanation, since zero is not an arbitrary
    floor, and it is paired with a check that separates the two cases
    it would otherwise conflate.

    A covariance operator is positive semi-definite, so its eigenvalues
    are non-negative in exact arithmetic. Finite precision returns small
    negatives for those that should be zero -- measured between -2e-16
    and -8e-16 relative to the largest eigenvalue, across several
    problem sizes and correlation lengths, so within a small factor of
    machine epsilon. A KLE takes ``sqrt`` of each eigenvalue to scale
    its basis, so an unclipped negative becomes NaN and propagates into
    every field evaluation. Clipping *that* is correct.

    What the clip must not absorb is over-requesting. Asking for more
    terms than the operator's numerical rank yields a tail of near-zero
    eigenvalues, and clipping alone would turn them into modes that
    exist in shape but carry no variance -- zero columns in the basis,
    silently. So the clip runs first, to keep genuine rounding from
    becoming NaN, and :func:`_reject_negligible_terms` then rejects
    what survives truncation still negligible.

    Parameters
    ----------
    eig_vals : Array
        Eigenvalues of the symmetrized operator.
    eig_vecs : Array
        Corresponding eigenvectors, shape ``(N, k)``.
    sqrt_weights : Array, optional
        Square roots of the quadrature weights, shape ``(N,)``. When
        None the eigenvectors are already in the unweighted convention.
    nterms : int
        Number of eigenpairs to keep.
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    eig_vals : Array
        Shape ``(nterms,)``, descending, non-negative.
    eig_vecs : Array
        Shape ``(N, nterms)``, unweighted convention, deterministic sign.
    """
    if sqrt_weights is not None:
        eig_vecs = eig_vecs / sqrt_weights[:, None]
    eig_vals = bkd.maximum(eig_vals, bkd.asarray([0.0]))
    eig_vals, eig_vecs = sort_eigenpairs(eig_vals, eig_vecs, nterms, bkd)
    _reject_negligible_terms(eig_vals, bkd)
    return eig_vals, adjust_sign_eig(eig_vecs, bkd)


def usable_nterms(eig_vals: Array, bkd: Backend[Array]) -> int:
    """How many of ``eig_vals`` carry variance rather than rounding.

    Shares its threshold with :func:`_reject_negligible_terms`, so a
    basis truncated to this many terms is exactly one that passes that
    check.
    """
    largest = bkd.to_float(bkd.max(eig_vals))
    if largest <= 0.0:
        return 0
    nvals = int(eig_vals.shape[0])
    tolerance = largest * nvals * _MACHINE_EPS
    return int((bkd.to_numpy(eig_vals) > tolerance).sum())


def _reject_negligible_terms(
    eig_vals: Array, bkd: Backend[Array]
) -> None:
    """Refuse a basis containing modes that carry no variance.

    A KLE scales each eigenvector by ``sqrt(eigenvalue)``, so a term
    whose eigenvalue is zero to machine precision contributes a column
    of zeros: a mode the caller asked for and did not get. It arises
    from requesting more terms than the operator supplies, which is
    easy to do because smooth kernels are severely rank deficient --
    a squared exponential on 100 points has numerical rank 15 at
    lengthscale 0.3, so anything past the fifteenth term is empty.

    Note this is a check on the *retained* terms, not on the operator.
    Rank deficiency itself is normal and not an error; the covariance
    is positive semi-definite rather than positive definite, and the
    zeros further down its spectrum are simply not requested. Only a
    zero that survives truncation means the caller was handed a mode
    that does not exist.

    Eigenvalues negative by more than rounding are caught here too,
    under the same message: both mean the requested terms outran what
    the operator can supply. Genuine rounding on a true zero was
    measured between -2e-16 and -8e-16 relative across problem sizes
    200 to 900, within a small factor of machine epsilon, so the
    ``n * eps`` threshold sits four orders above it.
    """
    largest = bkd.to_float(bkd.max(eig_vals))
    if largest <= 0.0:
        # Nothing positive to scale the tolerance against, and the
        # cause differs by sign: all-zero means the operator has no
        # modes at these points, while a negative maximum means it is
        # indefinite and so is not a covariance at all.
        raise ValueError(
            f"the largest retained eigenvalue is {largest:.3e}, so the "
            "operator supplies no usable modes at these points"
            + (
                "; a negative maximum means it is indefinite rather than "
                "merely rank deficient"
                if largest < 0.0
                else ""
            )
        )
    nvals = int(eig_vals.shape[0])
    tolerance = largest * nvals * _MACHINE_EPS
    negligible = int((bkd.to_numpy(eig_vals) <= tolerance).sum())
    if negligible > 0:
        raise ValueError(
            f"{negligible} of the {nvals} requested KLE terms have "
            f"eigenvalues at or below {tolerance:.3e}, which is rounding "
            "error rather than variance. Those terms would contribute "
            "columns of zeros to the basis. Request at most "
            f"{nvals - negligible} terms, or use a kernel whose spectrum "
            "decays more slowly."
        )


class _KLEEigenSolver(Generic[Array], ABC):
    """Shared skeleton for the eigensolvers shipped with PyApprox.

    Handles weight symmetrization on the way in and
    :func:`finalize_eigenpairs` on the way out, so a subclass implements
    only :meth:`_solve_symmetrized`.

    Private, and not part of the contract: :class:`KLEEigenSolverProtocol`
    is the injection point, and a solver may satisfy it directly without
    inheriting from here so long as it calls
    :func:`finalize_eigenpairs` itself.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def solve(
        self,
        kernel: KernelProtocol[Array],
        coords: Array,
        nterms: int,
        quad_weights: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Return the leading ``nterms`` eigenpairs."""
        if nterms < 1:
            raise ValueError(f"nterms must be >= 1, got {nterms}")
        npoints = int(coords.shape[1])
        if nterms > npoints:
            raise ValueError(
                f"nterms ({nterms}) cannot exceed the number of "
                f"collocation points ({npoints})"
            )
        sqrt_weights = None
        if quad_weights is not None:
            if quad_weights.ndim != 1:
                raise ValueError(
                    "quad_weights must be 1D with shape (N,), got ndim="
                    f"{quad_weights.ndim}"
                )
            if quad_weights.shape[0] != npoints:
                raise ValueError(
                    f"quad_weights has {quad_weights.shape[0]} entries but "
                    f"coords has {npoints} points"
                )
            sqrt_weights = self._bkd.sqrt(quad_weights)
        eig_vals, eig_vecs = self._solve_symmetrized(
            kernel, coords, nterms, sqrt_weights
        )
        return finalize_eigenpairs(
            eig_vals, eig_vecs, sqrt_weights, nterms, self._bkd
        )

    @abstractmethod
    def _solve_symmetrized(
        self,
        kernel: KernelProtocol[Array],
        coords: Array,
        nterms: int,
        sqrt_weights: Optional[Array],
    ) -> Tuple[Array, Array]:
        r"""Eigenpairs of :math:`W^{1/2} K W^{1/2}`.

        Returns raw eigenpairs; :meth:`solve` applies the convention.
        When ``sqrt_weights`` is None the operator is just :math:`K`.
        """
        raise NotImplementedError


class PivotedCholeskyEigenSolver(_KLEEigenSolver[Array]):
    r"""Eigenpairs from a low-rank pivoted Cholesky factor.

    Greedy pivoting builds :math:`K \approx L L^T` with ``L`` of shape
    ``(N, r)``, ``r << N``, costing ``r * N`` kernel evaluations and
    never forming ``K``. The eigenpairs then come from a QR of ``L``.

    **Why a QR of L gives eigenpairs.** With :math:`L = QR` where ``Q``
    has orthonormal columns,

    .. math::
        L L^T = (QR)(QR)^T = Q\,(R R^T)\,Q^T

    and ``R R^T`` is a small ``(r, r)`` symmetric positive semi-definite
    matrix, cheap to eigendecompose densely as
    :math:`R R^T = V \Sigma V^T`. Substituting back,

    .. math::
        L L^T = Q V \Sigma (Q V)^T .

    That is an eigendecomposition rather than merely a factorization
    because ``QV`` is itself orthonormal:
    :math:`(QV)^T (QV) = V^T Q^T Q V = V^T I V = I`. So the eigenvalues
    are :math:`\mathrm{diag}(\Sigma)` and the eigenvectors the columns
    of ``QV``.

    **Why ``r`` eigenpairs is the whole nonzero spectrum**, not a
    truncation: ``L L^T`` has rank at most ``r``, so its remaining
    ``N - r`` eigenvalues are exactly zero. Checked on a random
    ``N=500, r=40`` case, entries ``r``, ``r+1`` and ``r+2`` of a
    brute-force spectrum came out at 9e-14, 7e-14 and 5e-14.

    Cost is ``O(N r^2)`` for the QR against the ``O(N^3)`` avoided.

    **Why not the cheaper ``L^T L`` route.** ``eigh(L.T @ L)`` followed
    by ``U = L W / sqrt(sig)`` skips the QR and looks strictly better,
    but forming ``L^T L`` squares the condition number, so
    ``cond(L) ~ 1e8`` already reaches the double-precision limit. QR
    obtains ``R`` through orthogonal transformations and never squares
    anything. Both routes are safe here only because dropping
    non-positive eigenvalues keeps ``L`` well conditioned, so anyone
    loosening that filter must re-measure both.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    rank_multiplier : float
        Factor times ``nterms`` giving the factorization rank. The
        default suits smooth kernels and is wrong for rough ones:
        measured over 60 modes, a squared exponential saturates at
        6.4e-08 by 4x while a Matern-3/2 needs 8-16x to reach 3.7e-03
        to 5.9e-04. Raise it for rough fields rather than concluding
        the solver is broken. Raising it is not unbounded -- the QR is
        ``O(N rank^2)``, so at ``N=1e5`` it overtakes the kernel
        evaluations near rank 3000.
    rank : int, optional
        Absolute rank, overriding ``rank_multiplier``.
    tol : float
        Relative trace tolerance for early termination.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        rank_multiplier: float = 4.0,
        rank: Optional[int] = None,
        tol: float = 1e-12,
    ):
        super().__init__(bkd)
        if rank_multiplier < 1.0:
            raise ValueError(
                "rank_multiplier must be >= 1, since the factorization "
                f"rank cannot be below nterms; got {rank_multiplier}"
            )
        if rank is not None and rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self._rank_multiplier = float(rank_multiplier)
        self._rank = rank
        self._tol = float(tol)

    def rank_for(self, nterms: int, npoints: int) -> int:
        """Factorization rank used for ``nterms`` modes."""
        rank = (
            self._rank
            if self._rank is not None
            else int(math.ceil(self._rank_multiplier * nterms))
        )
        return min(max(rank, nterms), npoints)

    def factorize(
        self,
        kernel: KernelProtocol[Array],
        coords: Array,
        nterms: int,
        sqrt_weights: Optional[Array] = None,
    ) -> "PivotedCholeskyFactorizer[Array]":
        """Return the completed factorizer, pivots and factor included.

        Exposed so a Nystrom expansion can reuse this factorization
        rather than repeating it: with pivoted landmarks the pivots
        *are* the landmark set, and two implementations of one
        algorithm would drift while both still passed a
        reproduce-the-dense-solver test.
        """
        operator: ColumnOperatorProtocol[Array] = (
            _WeightedColumnOperator(kernel, coords, sqrt_weights, self._bkd)
            if sqrt_weights is not None
            else KernelColumnOperator(kernel, coords, self._bkd)
        )
        factorizer = PivotedCholeskyFactorizer(
            operator, self._bkd, tol=self._tol
        )
        factorizer.factorize(self.rank_for(nterms, int(coords.shape[1])))
        return factorizer

    def _solve_symmetrized(
        self,
        kernel: KernelProtocol[Array],
        coords: Array,
        nterms: int,
        sqrt_weights: Optional[Array],
    ) -> Tuple[Array, Array]:
        bkd = self._bkd
        lfactor = self.factorize(
            kernel, coords, nterms, sqrt_weights
        ).factor()
        qmat, rmat = bkd.qr(lfactor)
        vals, vmat = bkd.eigh(rmat @ rmat.T)
        # eigh returns ascending, and the factorization rank exceeds
        # nterms whenever rank_multiplier > 1, so take the largest
        # nterms here. finalize_eigenpairs sorts, but its index runs
        # over exactly nterms columns, so handing it the full rank-r
        # decomposition would silently keep the r smallest eigenvalues.
        leading = bkd.arange(vals.shape[0] - 1, -1, -1, dtype=int)[:nterms]
        return vals[leading], (qmat @ vmat)[:, leading]


class RandomizedEigenSolver(_KLEEigenSolver[Array]):
    """Eigenpairs by randomized subspace iteration.

    Applies the kernel matrix to a block of random vectors through
    :class:`KernelMatVecOperator`, so the matrix is never formed. Costs
    ``(2 + npower_iters)`` passes of ``N^2`` kernel evaluations, far
    more than pivoted Cholesky at large ``N``, but degrades more
    gracefully when the spectrum decays slowly.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    noversampling : int
        Extra random samples beyond ``nterms``, improving accuracy.
    npower_iters : int
        Power iterations, which help when eigenvalues decay slowly.
    block_size : int
        Rows of the kernel matrix evaluated per pass.
    seed : int or None
        Seeds the random sketch, so two solves of the same problem
        return the same basis. Pass None to draw from the global RNG
        instead, which is what a caller wanting independent draws
        across repeated solves would want; a seeded solver uses a local
        stream and never perturbs global state.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        noversampling: int = 20,
        npower_iters: int = 2,
        block_size: int = 2048,
        seed: Optional[int] = 0,
    ):
        super().__init__(bkd)
        if noversampling < 0:
            raise ValueError(
                f"noversampling must be >= 0, got {noversampling}"
            )
        if npower_iters < 0:
            raise ValueError(
                f"npower_iters must be >= 0, got {npower_iters}"
            )
        self._noversampling = int(noversampling)
        self._npower_iters = int(npower_iters)
        self._block_size = int(block_size)
        self._seed = seed

    def seed(self) -> Optional[int]:
        """Return the sketch seed, or None for the global RNG."""
        return self._seed

    def _solve_symmetrized(
        self,
        kernel: KernelProtocol[Array],
        coords: Array,
        nterms: int,
        sqrt_weights: Optional[Array],
    ) -> Tuple[Array, Array]:
        npoints = int(coords.shape[1])
        operator = KernelMatVecOperator(
            kernel,
            coords,
            self._bkd,
            sqrt_weights=sqrt_weights,
            block_size=self._block_size,
        )
        # oversampling cannot exceed the problem size
        noversampling = min(self._noversampling, npoints - nterms)
        return randomized_symmetric_eigendecomposition(
            operator.apply,
            npoints,
            nterms,
            self._bkd,
            noversampling=max(noversampling, 0),
            npower_iters=self._npower_iters,
            seed=self._seed,
        )


class _WeightedColumnOperator(Generic[Array]):
    r"""Columns of :math:`W^{1/2} K W^{1/2}`, formed one at a time.

    The pivoted Cholesky factorizer consumes a column operator, so the
    symmetrization is applied per column rather than to an assembled
    matrix -- forming the matrix is exactly what the solver avoids.

    Implemented as an operator rather than as a kernel wrapper because
    the weight of a column depends on *which* column is requested, and
    only the operator is told the index. A kernel wrapper would have to
    recover the index by matching coordinates, which is quadratic and
    ambiguous when two collocation points coincide.
    """

    def __init__(
        self,
        kernel: KernelProtocol[Array],
        X: Array,
        sqrt_weights: Array,
        bkd: Backend[Array],
    ):
        self._kernel = kernel
        self._X = X
        self._sqrt_weights = sqrt_weights
        self._bkd = bkd
        self._n = int(X.shape[1])

    def column(self, j: int) -> Array:
        col = self._kernel(self._X, self._X[:, j : j + 1])
        col = self._bkd.reshape(col, (-1,))
        return col * self._sqrt_weights * self._sqrt_weights[j]

    def diagonal(self) -> Array:
        return self._kernel.diag(self._X) * self._sqrt_weights**2

    def nvars(self) -> int:
        return self._n


class DenseEigenSolver(_KLEEigenSolver[Array]):
    """Assembles the kernel matrix and eigendecomposes it.

    The default, and the reference the matrix-free solvers are tested
    against. Costs ``O(N^2)`` memory, which is what caps the usable
    problem size; a partial Lanczos solve is used when ``nterms < N``.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    seed : int or None
        Seeds the Lanczos start vector, which is used only when
        ``nterms < N``; the full ``eigh`` path taken otherwise is
        already deterministic.

        Pass None for ARPACK's own random start, which is *not*
        reproducible: it draws from a stream numpy's global seed does
        not reach, so seeding before the call has no effect. Measured
        on three full-rank matrices, successive unseeded solves agreed
        on eigenvalues to 5e-15 while individual eigenvectors differed
        by 4.4e-01 to 8.3e-01 -- not degenerate-subspace rotation, as
        one was a random SPD matrix with well-separated eigenvalues.
    """

    def __init__(self, bkd: Backend[Array], seed: Optional[int] = 0):
        super().__init__(bkd)
        self._seed = seed

    def seed(self) -> Optional[int]:
        """Return the Lanczos start seed, or None if unseeded."""
        return self._seed

    def _solve_symmetrized(
        self,
        kernel: KernelProtocol[Array],
        coords: Array,
        nterms: int,
        sqrt_weights: Optional[Array],
    ) -> Tuple[Array, Array]:
        kmat = kernel(coords, coords)
        if sqrt_weights is not None:
            kmat = (sqrt_weights[:, None] * kmat) * sqrt_weights[None, :]
        # sorts and sign-adjusts internally; finalize_eigenpairs repeats
        # both, which is idempotent, and owns the clip that this does not
        # perform.
        return eigendecomposition_unweighted(
            kmat, nterms, self._bkd, seed=self._seed
        )
