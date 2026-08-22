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

from abc import ABC, abstractmethod
from typing import Generic, Optional, Protocol, Tuple, runtime_checkable

from pyapprox.surrogates.kernels.protocols import KernelProtocol
from pyapprox.surrogates.kle.utils import (
    adjust_sign_eig,
    eigendecomposition_unweighted,
    sort_eigenpairs,
)
from pyapprox.util.backends.protocols import Array, Backend


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
    floor and the clip currently does two jobs of unequal merit.

    A covariance operator is positive semi-definite, so its eigenvalues
    are non-negative in exact arithmetic. Finite precision returns small
    negatives for those that should be zero -- measured between -2e-16
    and -8e-16 relative to the largest eigenvalue, across several
    problem sizes and correlation lengths, so within a small factor of
    machine epsilon. A KLE takes ``sqrt`` of each eigenvalue to scale
    its basis, so an unclipped negative becomes NaN and propagates into
    every field evaluation. Clipping *that* is correct.

    What the clip also absorbs, less defensibly, is over-requesting.
    Asking for more terms than the operator's numerical rank yields a
    tail of near-zero eigenvalues, and clipping turns them into modes
    that exist in shape but carry no variance -- zero columns in the
    basis, silently. A caller who asked for 200 terms from a rank-11
    operator gets 189 of those and no indication. Distinguishing
    rounding from over-requesting, and from a genuinely indefinite
    kernel at -1e-8 or worse, is deliberately left alone here: it is a
    change to established behaviour and belongs in its own commit, not
    folded into a refactor that is otherwise behaviour-preserving.

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
    return eig_vals, adjust_sign_eig(eig_vecs, bkd)


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


class DenseEigenSolver(_KLEEigenSolver[Array]):
    """Assembles the kernel matrix and eigendecomposes it.

    The default, and the reference the matrix-free solvers are tested
    against. Costs ``O(N^2)`` memory, which is what caps the usable
    problem size; a partial Lanczos solve is used when ``nterms < N``.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    """

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
        return eigendecomposition_unweighted(kmat, nterms, self._bkd)
