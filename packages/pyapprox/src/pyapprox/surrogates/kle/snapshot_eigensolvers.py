r"""How a basis is extracted from snapshot data.

The kernel-driven KLEs take an injected
:class:`~pyapprox.surrogates.kle.eigensolvers.KLEEigenSolverProtocol`,
so a caller who cannot afford a dense kernel matrix supplies a
matrix-free solver instead of editing the class. Snapshot-driven KLEs
had no such seam: the SVD was written inline, which fixed both the
algorithm and the metric it could handle.

The choice is not cosmetic. Given centered snapshots :math:`S` and a
metric :math:`M`, the basis wanted is the leading eigenvectors of
:math:`S S^T M`, and there are two ways to get them:

**Symmetrize.** Factor :math:`M = L^T L`, take the SVD of :math:`L S`,
and map back. Cheap and accurate, but it needs :math:`M^{1/2}` -- free
for a diagonal metric, and unavailable for a sparse mass matrix without
CHOLMOD or densifying.

**Method of snapshots.** Eigendecompose the :math:`n \times n` Gram
:math:`S^T M S` and set :math:`V = S Q \Lambda^{-1/2}`. This touches
:math:`M` only through :meth:`apply`, so it works for *any* SPD metric,
and when :math:`n \ll N` the Gram is far smaller than the covariance.
The cost is conditioning: forming the Gram squares the condition number,
so the smallest retained modes are computed to about half the precision
the SVD would give them.

Neither dominates, which is why both are here and why the metric decides
the default rather than the caller having to know this. The default
prefers the SVD wherever it applies, buying accuracy at a real cost in
speed: measured on random matrices, the Gram route ran 4-12x faster
(0.21s against 0.022s at 5000x200, 5.9s against 0.49s at 50000x500).
A caller who has measured their own conditioning and wants the speed
can inject :class:`MethodOfSnapshotsSolver` for a diagonal metric too.

**Why there is no randomized solver here.** Randomization exists to
avoid decomposing a matrix that cannot be afforded exactly, and a
snapshot matrix rarely is one: :math:`S` is *already* the low-rank
factor, so a thin SVD is exact at :math:`O(N n^2)` rather than an
approximation. At 50000 states and 500 snapshots -- a serious FEM set --
both solvers above finish in under six seconds. The regime where
randomization pays needs :math:`n` itself to be large, which is narrower
than it first looks. Should it arrive, it is a third axis rather than a
third entry in the list above: these two vary by *metric* and are both
exact, while a randomized solver varies by *cost* and is approximate, so
it would compose with either. ``util.linalg.randomized`` already has the
machinery.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional, Protocol, Tuple, runtime_checkable

from pyapprox.surrogates.kle.eigensolvers import finalize_eigenpairs
from pyapprox.surrogates.kle.truncation import by_numerical_rank
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import InnerProductProtocol


@runtime_checkable
class SnapshotEigenSolverProtocol(Protocol, Generic[Array]):
    r"""Computes the leading eigenpairs of an empirical covariance.

    The sibling of
    :class:`~pyapprox.surrogates.kle.eigensolvers.KLEEigenSolverProtocol`
    for the case where the covariance is not a kernel but a snapshot
    matrix. Implementations return eigenpairs in the same **unweighted
    convention**: eigenvectors orthonormal under the metric's inner
    product, eigenvalues non-negative and descending, signs
    deterministic. :func:`~pyapprox.surrogates.kle.finalize_eigenpairs`
    establishes all of that.
    """

    def solve(
        self,
        snapshots: Array,
        nterms: Optional[int] = None,
        metric: Optional[InnerProductProtocol[Array]] = None,
    ) -> Tuple[Array, Array]:
        r"""Return the leading ``nterms`` eigenpairs.

        Parameters
        ----------
        snapshots : Array
            Shape ``(nstates, nsamples)``. Centering is the caller's
            business: this is a decomposition of whatever matrix it is
            handed, and a solver cannot tell whether a mean was already
            removed.
        nterms : int, optional
            Number of eigenpairs to keep. None keeps every mode carrying
            variance, which is the numerical rank.

            Passing a count buys no computation -- both solvers form the
            whole spectrum either way, and the count only decides where
            it is cut and whether over-requesting is an error. So a
            caller who must see the spectrum before choosing, as one
            truncating by variance fraction does, should omit it and
            slice the result rather than decompose twice.
        metric : InnerProductProtocol, optional
            The inner product the eigenvectors are orthonormal in. None
            means Euclidean.

        Returns
        -------
        eig_vals : Array
            Shape ``(nterms,)``, descending, non-negative. Eigenvalues
            of ``S S^T M``, without the ``1/(n-1)`` that would make them
            a sample covariance -- that convention belongs to the KLE,
            which knows whether its input was centered.
        eig_vecs : Array
            Shape ``(nstates, nterms)``, metric-orthonormal.
        """
        ...


class _SnapshotEigenSolver(Generic[Array], ABC):
    """Shared plumbing: validation, then the convention."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def solve(
        self,
        snapshots: Array,
        nterms: Optional[int] = None,
        metric: Optional[InnerProductProtocol[Array]] = None,
    ) -> Tuple[Array, Array]:
        if snapshots.ndim != 2:
            raise ValueError(
                "snapshots must be 2D (nstates, nsamples), got "
                f"ndim={snapshots.ndim}"
            )
        nstates, nsamples = (int(s) for s in snapshots.shape)
        if nterms is not None:
            if nterms < 1:
                raise ValueError(f"nterms={nterms} must be positive")
            max_nterms = min(nstates, nsamples)
            if nterms > max_nterms:
                raise ValueError(
                    f"nterms={nterms} exceeds the rank of the snapshot "
                    f"matrix, min(nstates={nstates}, nsamples={nsamples})"
                    f"={max_nterms}"
                )
        if metric is not None and metric.nstates() != nstates:
            raise ValueError(
                f"metric is defined on {metric.nstates()} states but "
                f"snapshots have {nstates}"
            )
        return self._solve(snapshots, nterms, metric)

    @abstractmethod
    def _solve(
        self,
        snapshots: Array,
        nterms: Optional[int],
        metric: Optional[InnerProductProtocol[Array]],
    ) -> Tuple[Array, Array]:
        """Compute eigenpairs; arguments already validated."""


class SVDSnapshotSolver(_SnapshotEigenSolver[Array]):
    r"""Thin SVD of the symmetrized snapshot matrix.

    The default when the metric is diagonal, and the more accurate of
    the two: it never forms a Gram matrix, so the singular values are
    computed to full precision rather than to half of it.

    Refuses a non-diagonal metric rather than densifying it. A caller
    who reaches this error wants :class:`MethodOfSnapshotsSolver`, and
    :class:`~pyapprox.surrogates.kle.DataDrivenKLE` selects it
    automatically.
    """

    def _solve(
        self,
        snapshots: Array,
        nterms: Optional[int],
        metric: Optional[InnerProductProtocol[Array]],
    ) -> Tuple[Array, Array]:
        bkd = self._bkd
        sqrt_weights: Optional[Array] = None
        if metric is not None:
            if not metric.is_diagonal():
                raise ValueError(
                    f"{type(self).__name__} needs a diagonal metric to "
                    "symmetrize with, got a non-diagonal one. Use "
                    "MethodOfSnapshotsSolver, which reaches the same "
                    "basis through the Gram and never factorizes M."
                )
            # A diagonal metric's square root is elementwise, and
            # applying it to a vector of ones recovers the diagonal
            # without the protocol having to expose it.
            sqrt_weights = bkd.sqrt(
                metric.apply(bkd.ones((int(snapshots.shape[0]), 1)))[:, 0]
            )
            snapshots = sqrt_weights[:, None] * snapshots

        # Thin: only min(nstates, nsamples) left vectors can have a
        # nonzero singular value, and the right factor is discarded, so
        # the full form builds an (nsamples, nsamples) block for nothing.
        eig_vecs, svals, _ = bkd.svd(snapshots, full_matrices=False)
        eig_vals = svals**2
        if nterms is None:
            nterms = max(1, by_numerical_rank(eig_vals, bkd))
        return finalize_eigenpairs(
            eig_vals, eig_vecs, sqrt_weights, nterms, bkd
        )


class MethodOfSnapshotsSolver(_SnapshotEigenSolver[Array]):
    r"""Eigendecomposition of the ``(nsamples, nsamples)`` Gram.

    Works for **any** SPD metric, because it touches :math:`M` only
    through :meth:`~InnerProductProtocol.apply` -- never a square root,
    never a factorization. That is what makes it usable with an
    assembled FEM mass matrix, where symmetrizing would mean CHOLMOD or
    an :math:`O(N^2)` densification.

    Cheaper than the SVD when snapshots are few relative to states, the
    usual case for FEM data: the Gram is ``nsamples`` square while the
    covariance is ``nstates`` square.

    The tradeoff is conditioning. :math:`S^T M S` squares the condition
    number of :math:`S`, so modes whose singular values approach the
    square root of machine epsilon are lost. Prefer
    :class:`SVDSnapshotSolver` whenever the metric is diagonal.
    """

    def _solve(
        self,
        snapshots: Array,
        nterms: Optional[int],
        metric: Optional[InnerProductProtocol[Array]],
    ) -> Tuple[Array, Array]:
        bkd = self._bkd
        weighted = (
            snapshots if metric is None else metric.apply(snapshots)
        )
        gram = bkd.dot(snapshots.T, weighted)
        # Symmetrize: the two triangles differ at rounding level, and an
        # eigh of a matrix that is not exactly symmetric reads only one
        # triangle, silently discarding the discrepancy rather than
        # averaging it.
        gram = (gram + gram.T) / 2.0
        gram_vals, gram_vecs = bkd.eigh(gram)

        # eigh returns ascending; the convention here is descending, and
        # the leading terms must be selected before the basis is formed
        # so the dangerous 1/sqrt(lambda) is only ever applied to modes
        # that survive truncation.
        gram_vals = bkd.flip(gram_vals, axis=(0,))
        gram_vecs = bkd.flip(gram_vecs, axis=(1,))
        gram_vals = bkd.maximum(gram_vals, bkd.asarray([0.0]))
        if nterms is None:
            nterms = max(1, by_numerical_rank(gram_vals, bkd))

        kept_vals = gram_vals[:nterms]
        kept_vecs = gram_vecs[:, :nterms]
        if bkd.to_float(bkd.min(kept_vals)) <= 0.0:
            raise ValueError(
                f"nterms={nterms} exceeds the numerical rank of the "
                "snapshot matrix: a retained Gram eigenvalue is zero, so "
                "the corresponding mode would be a zero column."
            )

        # V = S Q / sqrt(lambda) is M-orthonormal:
        # V^T M V = diag(1/sqrt(l)) Q^T (S^T M S) Q diag(1/sqrt(l)) = I.
        basis = bkd.dot(snapshots, kept_vecs) / bkd.sqrt(kept_vals)
        return finalize_eigenpairs(kept_vals, basis, None, nterms, bkd)


def default_snapshot_eigensolver(
    bkd: Backend[Array],
    metric: Optional[InnerProductProtocol[Array]] = None,
) -> SnapshotEigenSolverProtocol[Array]:
    """Return the solver appropriate to ``metric``.

    The SVD is preferred for its accuracy and used whenever the metric
    admits a square root; the method of snapshots covers the rest. This
    is a choice about numerics rather than about which class a caller
    wants, so it is made here rather than left to the caller.
    """
    if metric is None or metric.is_diagonal():
        return SVDSnapshotSolver(bkd)
    return MethodOfSnapshotsSolver(bkd)
