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

**The randomized solver is a third axis, not a third entry.** The two
above vary by *metric* and are both exact;
:class:`RandomizedSnapshotSolver` varies by *cost* and is approximate,
so it is chosen for a different reason. Time is rarely that reason --
:math:`S` is already the low-rank factor, so a thin SVD is exact at
:math:`O(N n^2)` and finishes in under six seconds at 50000 states and
500 snapshots. Memory is: the exact solvers form a basis of the
numerical rank's width where the sketch forms one of
``nterms + noversampling``, which is the difference between an
affordable fit and an impossible one once :math:`N` is large enough.
"""

from dataclasses import dataclass
from typing import (
    Generic,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from pyapprox.surrogates.kle.eigensolvers import (
    finalize_eigenpairs_with_convention,
)
from pyapprox.surrogates.kle.snapshot_sources import (
    SnapshotSourceOperator,
    SnapshotSourceProtocol,
)
from pyapprox.surrogates.kle.truncation import by_numerical_rank
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.inner_product import InnerProductProtocol
from pyapprox.util.linalg.randomized import (
    DenseMatVecOperator,
    MatVecOperator,
    TwoPassRandomizedSVD,
)


@dataclass(frozen=True)
class SnapshotDecomposition(Generic[Array]):
    r"""What decomposing a snapshot matrix yields.

    The three pieces are one object rather than three returns because
    they are only meaningful together. An eigenvector's sign is free in
    isolation, but flipping it changes the coordinates the snapshots
    have in that direction, so a caller handed the two separately could
    hold a consistent-looking pair that no longer reconstructs the data.
    Bundling them lets the sign convention be applied to both at once.

    Attributes
    ----------
    eigenvalues : Array
        Shape ``(nterms,)``, descending and non-negative. Eigenvalues of
        ``S S^T M``, without the ``1/(n-1)`` that would make them a
        sample covariance -- that convention belongs to the KLE, which
        knows whether its input was centered.
    eigenvectors : Array
        Shape ``(nstates, nterms)``, metric-orthonormal.
    coordinates : Array
        Shape ``(nterms, nsamples)``. Each snapshot expressed in the
        basis: ``coordinates[a, n]`` is the component of snapshot ``n``
        along ``eigenvectors[:, a]``, so
        ``eigenvectors @ coordinates`` reconstructs the input to the
        retained rank. Equal to ``diag(s) Psi^T`` for the thin SVD
        ``S = Phi diag(s) Psi^T``.

        Carried rather than left to the caller to recompute as
        ``eigenvectors.T @ snapshots``, which costs an
        ``(nterms, nstates, nsamples)`` product that both solvers have
        already done the work for.
    """

    eigenvalues: Array
    eigenvectors: Array
    coordinates: Array

    def nterms(self) -> int:
        """Number of retained modes."""
        return int(self.eigenvalues.shape[0])


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
    ) -> SnapshotDecomposition[Array]:
        r"""Return the leading ``nterms`` eigenpairs and coordinates.

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

            Pass it when it is known. It buys little computation --
            the small factor each exact solver decomposes is the same
            size either way -- but the basis is the one ambient-sized
            array and every solver here sizes it by ``nterms``.
            Measured at 200000 states and 150 snapshots of numerical
            rank 60, requesting 10 terms cut peak allocation from
            412 MB to 260 MB for :class:`SVDSnapshotSolver` and from
            504 MB to 275 MB for :class:`MethodOfSnapshotsSolver`.
            :class:`RandomizedSnapshotSolver` requires it.

            A caller that must see the spectrum before choosing, as one
            truncating by variance fraction does, should omit it and
            slice the result rather than decompose twice.
        metric : InnerProductProtocol, optional
            The inner product the eigenvectors are orthonormal in. None
            means Euclidean.

        Returns
        -------
        SnapshotDecomposition
            The eigenvalues, the metric-orthonormal basis, and the
            snapshot coordinates in that basis.
        """
        ...


def validate_snapshot_arguments(
    nstates: int,
    nsamples: int,
    nterms: Optional[int] = None,
    metric: Optional[InnerProductProtocol[Array]] = None,
) -> None:
    """Check the arguments every solver here rejects the same way.

    Free-standing for the reason
    :func:`~pyapprox.surrogates.kle.eigensolvers.finalize_eigenpairs`
    is: a solver satisfying
    :class:`SnapshotEigenSolverProtocol` directly should be able to
    reuse the checks without inheriting from anything, and the
    alternative is each implementation writing its own -- which is how
    two solvers come to disagree about what they accept.

    Takes the dimensions rather than the snapshots, so a solver reading
    from a source can call it too.

    Raises
    ------
    ValueError
        If ``nterms`` is not positive, exceeds ``min(nstates,
        nsamples)`` -- which bounds the rank, so more terms than that
        cannot exist -- or if ``metric`` is defined on a different
        number of states.
    """
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


def snapshot_shape(snapshots: Array) -> Tuple[int, int]:
    """Return ``(nstates, nsamples)`` for resident snapshots.

    Raises
    ------
    TypeError
        If given a snapshot source. Only a solver whose every pass is a
        row-block accumulation can read one, so this is where the rest
        say so -- rather than failing on a missing attribute, which
        reports the symptom and not the reason.
    ValueError
        If the array is not 2D.
    """
    if isinstance(snapshots, SnapshotSourceProtocol):
        raise TypeError(
            "this solver needs the snapshots in memory and was handed "
            "a source to read them from. Symmetrizing needs the whole "
            "matrix, and a Gram over a coupled metric needs rows from "
            "outside the block, so neither streams. Use "
            "RandomizedSnapshotSolver, or materialize the source."
        )
    if snapshots.ndim != 2:
        raise ValueError(
            "snapshots must be 2D (nstates, nsamples), got "
            f"ndim={snapshots.ndim}"
        )
    return int(snapshots.shape[0]), int(snapshots.shape[1])


def bundle_decomposition(
    eig_vals: Array,
    eig_vecs: Array,
    raw_coordinates: Array,
    sqrt_weights: Optional[Array],
    nterms: int,
    bkd: Backend[Array],
) -> SnapshotDecomposition[Array]:
    """Apply the convention to both factors at once.

    The convention sorts, truncates and signs the eigenvectors. The
    coordinates are the other half of the same factorization, so they
    must be permuted and signed identically: a flip applied to one and
    not the other leaves a pair that no longer reconstructs the
    snapshots.

    The permutation and signs are taken from the function that applied
    them rather than re-derived here, which keeps the tie-breaking rule
    in one place. It also keeps only one ambient-sized array alive:
    recovering the permutation by comparing the finalized basis against
    the raw one needs both resident, and at a large ambient dimension
    that doubles the peak for information the sort already had.
    """
    vals, vecs, order, signs = finalize_eigenpairs_with_convention(
        eig_vals, eig_vecs, sqrt_weights, nterms, bkd
    )
    coordinates = raw_coordinates[order, :] * signs[:, None]
    return SnapshotDecomposition(vals, vecs, coordinates)


class SVDSnapshotSolver(Generic[Array]):
    r"""Thin SVD of the symmetrized snapshot matrix.

    The default when the metric is diagonal, and the more accurate of
    the two: it never forms a Gram matrix, so the singular values are
    computed to full precision rather than to half of it.

    Refuses a non-diagonal metric rather than densifying it. A caller
    who reaches this error wants :class:`MethodOfSnapshotsSolver`, and
    :class:`~pyapprox.surrogates.kle.DataDrivenKLE` selects it
    automatically.

    Takes resident snapshots only. Symmetrizing needs the whole matrix,
    and a dense SVD cannot be fed in pieces, so there is no row-block
    form of this algorithm to offer a
    :class:`~pyapprox.surrogates.kle.snapshot_sources.SnapshotSourceProtocol`.
    """

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
    ) -> SnapshotDecomposition[Array]:
        """Return the leading ``nterms`` eigenpairs and coordinates."""
        nstates, nsamples = snapshot_shape(snapshots)
        validate_snapshot_arguments(nstates, nsamples, nterms, metric)
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
        # nonzero singular value, so the full form would build an
        # (nsamples, nsamples) block whose extra columns pair with zero
        # singular values.
        eig_vecs, svals, right_factor = bkd.svd(
            snapshots, full_matrices=False
        )
        eig_vals = svals**2
        if nterms is None:
            nterms = max(1, by_numerical_rank(eig_vals, bkd))
        # diag(s) Psi^T: the snapshot coordinates in the basis, in the
        # solver's own (pre-convention) column order.
        raw_coordinates = svals[:, None] * right_factor
        return bundle_decomposition(
            eig_vals, eig_vecs, raw_coordinates, sqrt_weights, nterms, bkd
        )


class MethodOfSnapshotsSolver(Generic[Array]):
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

    Takes resident snapshots only, and the reason is the metric it
    exists to support. The Gram accumulates over row blocks only when
    :math:`M` has no coupling across block boundaries; a mass matrix
    couples neighbouring nodes, so a block holding rows ``i:j`` needs
    columns of :math:`S` from outside it. Accumulating the diagonal
    sub-blocks alone drops that coupling and returns a basis that is
    still orthonormal, still plausible, and wrong.
    """

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
    ) -> SnapshotDecomposition[Array]:
        """Return the leading ``nterms`` eigenpairs and coordinates."""
        nstates, nsamples = snapshot_shape(snapshots)
        validate_snapshot_arguments(nstates, nsamples, nterms, metric)
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
        # The Gram eigenvectors are the right singular factor: for
        # S = Phi diag(s) Psi^T the Gram S^T M S has eigenvectors Psi and
        # eigenvalues s^2, so the coordinates diag(s) Psi^T are
        # sqrt(lambda) Q^T -- already computed, not a second pass.
        raw_coordinates = bkd.sqrt(kept_vals)[:, None] * kept_vecs.T
        return bundle_decomposition(
            kept_vals, basis, raw_coordinates, None, nterms, bkd
        )


class RandomizedSnapshotSolver(Generic[Array]):
    r"""A sketched SVD of the snapshots, when the basis is what costs.

    Sketches :math:`S` onto ``nterms + noversampling`` random directions
    and factorizes there, so the widest array formed is
    ``(nstates, nterms + noversampling)`` rather than one of the
    numerical rank's width. Measured at 200000 states and 150 snapshots
    of numerical rank 60, extracting 10 terms: peak allocation 69 MB
    against the dense SVD's 260 MB, for a reconstruction error 1.0004
    times the dense one.

    Approximate, and how good the approximation is depends on the
    spectrum rather than on this class. A sketch resolves a subspace
    well when the spectrum decays quickly past ``nterms``; a slow decay
    leaves energy in directions the sketch can miss, and power
    iterations are the remedy. On 20000 states and 150 snapshots with a
    spectrum spanning three decades, relative reconstruction error
    against the dense SVD:

    ==============  ================  =================
    npower_iters    noversampling=5   noversampling=10
    ==============  ================  =================
    0               1.223x            1.082x
    1               1.003x            1.000x
    2               1.000x            1.000x
    ==============  ================  =================

    **Requires an explicit ``nterms``.** The other solvers discover the
    numerical rank from a spectrum they have already formed. This one
    never forms it.

    **Takes no metric.** Weighting the snapshots before sketching is
    sound for a diagonal metric, but the un-weighting afterwards
    interacts with the sketch's error in a way nothing here measures.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    noversampling : int
        Extra sketch directions beyond ``nterms``. Larger is more
        accurate and costs one ambient column each.
    npower_iters : int
        Subspace iterations. Each costs two passes over the snapshots
        and sharpens the separation between kept and discarded modes.
    seed : int, optional
        Seeds the sketch, for a reproducible basis. None uses the
        global RNG, so the basis then varies between runs -- which is
        correct for a randomized method and surprising in a test.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        noversampling: int = 10,
        npower_iters: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self._bkd = bkd
        if noversampling < 0:
            raise ValueError(
                f"noversampling={noversampling} must be non-negative"
            )
        if npower_iters < 0:
            raise ValueError(
                f"npower_iters={npower_iters} must be non-negative"
            )
        self._noversampling = int(noversampling)
        self._npower_iters = int(npower_iters)
        self._seed = seed

    def seed(self) -> Optional[int]:
        """Return the sketch seed, or None for the global RNG."""
        return self._seed

    def solve(
        self,
        snapshots: "Array | SnapshotSourceProtocol[Array]",
        nterms: Optional[int] = None,
        metric: Optional[InnerProductProtocol[Array]] = None,
    ) -> SnapshotDecomposition[Array]:
        """Return the leading ``nterms`` eigenpairs and coordinates.

        Accepts a
        :class:`~pyapprox.surrogates.kle.snapshot_sources.SnapshotSourceProtocol`
        as well as an array, and is the only solver here that does.
        Every product the sketch needs is a row-block accumulation, so
        the snapshots are read in pieces and the widest array formed is
        the ``(nstates, nterms + noversampling)`` sketch. The two exact
        solvers cannot offer this: one symmetrizes, which needs the
        whole matrix, and the other's Gram needs rows from outside the
        block whenever the metric couples them.
        """
        if isinstance(snapshots, SnapshotSourceProtocol):
            operator: MatVecOperator[Array] = SnapshotSourceOperator(
                snapshots, self._bkd
            )
            nstates, nsamples = (
                snapshots.nstates(),
                snapshots.nsamples(),
            )
        else:
            nstates, nsamples = snapshot_shape(snapshots)
            operator = DenseMatVecOperator(snapshots, self._bkd)
        validate_snapshot_arguments(nstates, nsamples, nterms, metric)
        if nterms is None:
            raise ValueError(
                f"{type(self).__name__} needs an explicit nterms: it "
                "never forms the whole spectrum, so it cannot discover "
                "the numerical rank. Pass a count, or use "
                "SVDSnapshotSolver if the rank is what you want."
            )
        if metric is not None:
            raise ValueError(
                f"{type(self).__name__} does not accept a metric. Use "
                "SVDSnapshotSolver for a diagonal metric or "
                "MethodOfSnapshotsSolver for any SPD one."
            )
        # Oversampling cannot exceed what is left to sample.
        noversampling = max(
            min(self._noversampling, nsamples - nterms), 0
        )
        svd = TwoPassRandomizedSVD(
            operator,
            noversampling=noversampling,
            npower_iters=self._npower_iters,
            seed=self._seed,
        )
        eig_vecs, svals, right_factor = svd.compute(nterms)
        # Same relationship as the dense SVD: eigenvalues of S S^T are
        # the squared singular values, and diag(s) Psi^T the coordinates.
        return bundle_decomposition(
            svals**2,
            eig_vecs,
            svals[:, None] * right_factor,
            None,
            nterms,
            self._bkd,
        )


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
