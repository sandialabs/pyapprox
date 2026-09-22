r"""Fitting the manifold correction without holding the snapshots.

The polynomial manifold decodes as :math:`\mu + V z + W h(z)`, and
:math:`W` is ``(nstates, p)`` -- ambient-sized, like the basis it
corrects. At a fine mesh it is the second object in the fit that does
not fit, and unlike the basis it is produced by a least-squares solve
rather than a decomposition.

That turns out to make it *easier* to stream, not harder. The matrix
actually inverted is

.. math:: H = h(z)\, h(z)^T + \gamma I

which is ``(p, p)`` and accumulates over *snapshots*: the ambient
dimension does not appear in it at all. Only the right-hand side
:math:`h(z)\,R^T` touches ``nstates``, and it carries that dimension as
a free axis, so

.. math:: W[B, :] = \left(H^{-1} h(z)\, R[B, :]^T\right)^T

for any set of rows :math:`B`. **Each ambient row of** :math:`W`
**depends only on the same row of the data.** There is no coupling
across rows to accumulate, no sketch, and no sign convention -- the
three things that made the basis harder.

What does need a pass is :math:`z = V^T S_c`, a contraction over rows,
and it is the same shape of accumulation
:meth:`~pyapprox.surrogates.kle.basis_operator.BasisOperatorProtocol.apply_transpose`
performs. So a fit costs two reads: one for :math:`z`, one for the rows
of :math:`W`.

**Every gamma shares those two reads.** Validation selection fits a grid
of gammas, and the expensive parts -- the features, the residual, the
Gram -- do not depend on gamma. The resident implementation keeps one
``(nstates, p)`` weight matrix per gamma, all alive at once; here the
gamma loop moves *inside* the block loop, so a grid costs
``len(grid)`` small solves per block and no ambient array at all.
"""

from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

from pyapprox.surrogates.kle.basis_operator import BasisOperatorProtocol
from pyapprox.surrogates.kle.basis_sinks import BasisSinkProtocol
from pyapprox.surrogates.kle.snapshot_sources import (
    SnapshotSourceProtocol,
)
from pyapprox.surrogates.reduction.feature_maps import FeatureMap
from pyapprox.util.backends.protocols import Array, Backend


class CenteredSource(Generic[Array]):
    """A source with the mean subtracted as each block is read.

    The fit works on centered snapshots, and the resident path takes a
    ``centered`` array the caller has already formed -- a second copy of
    the data. Subtracting per block instead means the only ambient array
    involved is the mean itself, which is ``(nstates, 1)`` and is needed
    by the decoder regardless.

    Parameters
    ----------
    source : SnapshotSourceProtocol[Array]
        The raw snapshots.
    mean : Array, shape (nstates, 1) or (nstates,)
        Subtracted from every block. A column vector, matching what the
        encoder stores.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        source: SnapshotSourceProtocol[Array],
        mean: Array,
        bkd: Backend[Array],
    ) -> None:
        if int(mean.shape[0]) != source.nstates():
            raise ValueError(
                f"mean has {mean.shape[0]} entries but the source has "
                f"{source.nstates()} states"
            )
        self._source = source
        self._mean = mean if mean.ndim == 2 else mean[:, None]
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nstates(self) -> int:
        """Return the ambient dimension."""
        return self._source.nstates()

    def nsamples(self) -> int:
        """Return the number of snapshots."""
        return self._source.nsamples()

    def row_blocks(
        self, max_bytes: Optional[int] = None
    ) -> Iterator[Tuple[slice, Array]]:
        """Yield the underlying blocks, each with the mean removed."""
        for rows, block in self._source.row_blocks(max_bytes):
            yield rows, block - self._mean[rows, :]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nstates={self.nstates()}, "
            f"nsamples={self.nsamples()})"
        )


def encode_from_source(
    source: SnapshotSourceProtocol[Array],
    basis: Array,
    bkd: Backend[Array],
    max_bytes: Optional[int] = None,
) -> Array:
    r"""Return :math:`z = V^T S_c`, shape ``(nreduced, nsamples)``.

    A sum over row blocks, so the result is small whatever the ambient
    dimension: block ``i:j`` contributes
    ``basis[i:j].T @ centered[i:j]`` and the terms add.

    The snapshots must already be centered, matching the resident path,
    which takes ``centered`` rather than centering itself. A source
    yielding raw snapshots would need the mean subtracted per block, and
    the mean is an ambient vector the caller has to own either way.
    """
    validate_basis(basis, source.nstates())
    out = bkd.zeros((int(basis.shape[1]), source.nsamples()))
    for rows, block in source.row_blocks(max_bytes):
        out = out + bkd.dot(basis[rows, :].T, block)
    return out


def fit_weights_from_source(
    source: SnapshotSourceProtocol[Array],
    basis: Array,
    feature_map: FeatureMap[Array],
    gamma: float,
    sink: BasisSinkProtocol[Array],
    bkd: Backend[Array],
    encoded: Optional[Array] = None,
    max_bytes: Optional[int] = None,
) -> BasisOperatorProtocol[Array]:
    r"""Fit :math:`W` a row block at a time, writing it to ``sink``.

    The streaming counterpart of
    :meth:`~pyapprox.surrogates.reduction.manifold_scoring.ManifoldScorer.fit_weights`,
    producing the same matrix: the regularized normal equations are
    solved once, small, and applied per block.

    Parameters
    ----------
    source : SnapshotSourceProtocol[Array]
        The *centered* snapshots, in row blocks.
    basis : Array, shape (nstates, nreduced)
        The selected orthonormal basis :math:`V`. An array rather than a
        basis operator because each block needs the rows of it matching
        that block, which the operator seam does not expose -- it is
        built for contracting a basis away, not for indexing into one.
        At the sizes where that matters the basis is itself in a sink,
        and reading a slice of it is the sink's business.
    feature_map : FeatureMap[Array]
        Maps ``(nreduced, nsamples)`` coordinates to ``(p, nsamples)``
        features.
    gamma : float
        Ridge parameter.
    sink : BasisSinkProtocol[Array]
        Where the ``(nstates, p)`` result goes.
    bkd : Backend[Array]
        Computational backend.
    encoded : Array, optional
        :math:`z`, when a caller already has it. Gamma selection does,
        and recomputing it would buy a second read of the source for
        nothing. None computes it here.
    max_bytes : int, optional
        Byte budget per row block.

    Returns
    -------
    BasisOperatorProtocol[Array]
        Whatever the sink finalized to.
    """
    validate_basis(basis, source.nstates())
    encoded = (
        encode_from_source(source, basis, bkd, max_bytes)
        if encoded is None
        else encoded
    )
    features = feature_map(encoded)
    if int(features.shape[0]) != sink.nterms():
        raise ValueError(
            f"sink is sized for {sink.nterms()} terms but the feature "
            f"map produces {features.shape[0]}"
        )
    gram = regularized_gram(features, gamma, bkd)
    for rows, block in source.row_blocks(max_bytes):
        sink.write(
            rows,
            weight_block(
                block, basis[rows, :], encoded, features, gram, bkd
            ),
        )
    return sink.finalize()


def select_gamma_from_source(
    source: SnapshotSourceProtocol[Array],
    validation: SnapshotSourceProtocol[Array],
    basis: Array,
    feature_map: FeatureMap[Array],
    gammas: Sequence[float],
    bkd: Backend[Array],
    max_bytes: Optional[int] = None,
) -> Tuple[float, Dict[str, List[float]]]:
    r"""Choose gamma by held-out error, without holding either dataset.

    The resident implementation fits every gamma, keeping a
    ``(nstates, p)`` weight matrix for each, then reconstructs the
    validation snapshots with each in turn. The list of weights and the
    per-gamma reconstruction are both ambient-sized, and the whole grid
    is alive at once.

    Here the gamma loop is inside the block loop. Both datasets span the
    same rows, so they are read in step: for each pair of blocks, every
    gamma's rows of :math:`W` are solved for, used immediately against
    the matching validation rows, and discarded. What survives a block is
    ``len(gammas)`` running scalars.

    The error accumulated is that of the whole model,
    :math:`\|P_V S + W h(z) - S\|_F^2` on the held-out data, matching
    the resident selection rather than scoring the correction alone.

    Returns
    -------
    best_gamma : float
        The gamma with the lowest held-out error.
    diagnostics : dict
        ``"gammas"`` and ``"val_err"``, the latter in grid order.
    """
    if len(gammas) == 0:
        raise ValueError("gammas must not be empty")
    validate_basis(basis, source.nstates())
    require_matching_states(source, validation)

    encoded = encode_from_source(source, basis, bkd, max_bytes)
    encoded_val = encode_from_source(
        validation, basis, bkd, max_bytes
    )
    features = feature_map(encoded)
    features_val = feature_map(encoded_val)
    # The gammas differ only in the diagonal they add, so the product
    # underneath is formed once and each gamma's matrix is that plus a
    # scaled identity. Hoisted out of the block loop as well: neither
    # part depends on the rows, since both contract over snapshots.
    base = feature_gram(features, bkd)
    eye = bkd.eye(int(features.shape[0]))
    grams = [base + gamma * eye for gamma in gammas]

    errors = [0.0] * len(gammas)
    for rows, block, block_val in _paired_blocks(
        source, validation, max_bytes, bkd
    ):
        basis_block = basis[rows, :]
        projected = bkd.dot(basis_block, encoded_val)
        # The residual and its contraction against the features are the
        # expensive part of a block and do not depend on gamma; only the
        # solve does. Computed once per block, as the resident
        # multi-gamma fit does once per dataset.
        cross = cross_block(
            block, basis_block, encoded, features, bkd
        )
        for index, gram in enumerate(grams):
            weights = bkd.solve(gram, cross).T
            difference = (
                projected + bkd.dot(weights, features_val) - block_val
            )
            errors[index] += float(
                bkd.to_float(bkd.sum(difference * difference))
            )

    best = min(range(len(gammas)), key=lambda index: errors[index])
    return gammas[best], {"gammas": list(gammas), "val_err": errors}


def weight_block(
    block: Array,
    basis_block: Array,
    encoded: Array,
    features: Array,
    gram: Array,
    bkd: Backend[Array],
) -> Array:
    r"""Rows of :math:`W` for one row block of the data.

    For several gammas over the same block, call :func:`cross_block`
    once and solve against each gamma's matrix: the cross is the part
    that touches the data, and it is the same for all of them.
    """
    return bkd.solve(
        gram, cross_block(block, basis_block, encoded, features, bkd)
    ).T


def cross_block(
    block: Array,
    basis_block: Array,
    encoded: Array,
    features: Array,
    bkd: Backend[Array],
) -> Array:
    r"""Return :math:`h R[B,:]^T`, the right-hand side for one block.

    ``block - basis_block @ encoded`` is the projection residual
    restricted to these rows, which is what the correction is fitted to.
    Formed per block rather than once for the whole dataset, so the
    second ambient-sized array the resident path keeps beside the
    snapshots never exists.
    """
    residual = block - bkd.dot(basis_block, encoded)
    return bkd.dot(features, residual.T)


def regularized_gram(
    features: Array, gamma: float, bkd: Backend[Array]
) -> Array:
    r"""Return :math:`h h^T + \gamma I`, shape ``(p, p)``.

    The matrix the fit inverts, and the reason the fit streams at all:
    it contracts over snapshots, so the ambient dimension never enters
    it and it is formed once for a whole pass rather than per block.

    A grid of gammas should use :func:`feature_gram` and add its own
    diagonals instead of calling this per gamma -- the product beneath
    is the expensive part and is the same for all of them.
    """
    return feature_gram(features, bkd) + gamma * bkd.eye(
        int(features.shape[0])
    )


def feature_gram(features: Array, bkd: Backend[Array]) -> Array:
    r"""Return :math:`h h^T`, shape ``(p, p)``, before regularization.

    Separate from the diagonal because gamma is the only thing a grid
    varies, and this is the part that costs anything to compute.
    """
    return bkd.dot(features, features.T)


def validate_basis(basis: Array, nstates: int) -> None:
    """Check the basis spans the rows the source will deliver.

    Caught here rather than at the first block, where the failure is a
    shape error inside a product naming neither argument.
    """
    if basis.ndim != 2:
        raise ValueError(
            f"basis must be 2D (nstates, nreduced), got "
            f"ndim={basis.ndim}"
        )
    if int(basis.shape[0]) != nstates:
        raise ValueError(
            f"basis has {basis.shape[0]} rows but the source has "
            f"{nstates} states"
        )


def require_matching_states(
    source: SnapshotSourceProtocol[Array],
    validation: SnapshotSourceProtocol[Array],
) -> None:
    """Both datasets must live on the same mesh to be compared."""
    if source.nstates() != validation.nstates():
        raise ValueError(
            f"training data has {source.nstates()} states and "
            f"validation has {validation.nstates()}; the held-out error "
            "compares them row by row"
        )


def _paired_blocks(
    source: SnapshotSourceProtocol[Array],
    validation: SnapshotSourceProtocol[Array],
    max_bytes: Optional[int],
    bkd: Backend[Array],
) -> Iterator[Tuple[slice, Array, Array]]:
    """Walk two sources over the same rows, yielding the overlap.

    The two span the same mesh but hold different numbers of snapshots,
    so the same byte budget buys each a different number of rows and
    their block boundaries do not line up. This advances both and yields
    the largest span both currently have buffered -- never more, which
    is what keeps the pieces small: asking for a span longer than a
    source's block forces it to join blocks back together, and joining
    enough of them reassembles the dataset the blocking existed to
    avoid.
    """
    left = _BlockCursor(source, max_bytes)
    right = _BlockCursor(validation, max_bytes)
    position = 0
    while left.advance() and right.advance():
        extent = min(left.available(), right.available())
        yield (
            slice(position, position + extent),
            left.take(extent),
            right.take(extent),
        )
        position += extent


class _BlockCursor(Generic[Array]):
    """One source's blocks, served as views and never joined.

    Holds at most one block and hands out slices of it. Deliberately
    cannot return more rows than the block it currently holds: a cursor
    that joined blocks to satisfy a longer request would undo the
    blocking, and doing so across a whole dataset would rebuild the
    array that could not be held in the first place.
    """

    def __init__(
        self,
        source: SnapshotSourceProtocol[Array],
        max_bytes: Optional[int],
    ) -> None:
        self._blocks = source.row_blocks(max_bytes)
        self._held: Optional[Array] = None
        self._offset = 0

    def advance(self) -> bool:
        """Ensure a block is buffered; False when the source is spent."""
        while self._held is None or self._offset >= int(
            self._held.shape[0]
        ):
            block = next(self._blocks, None)
            if block is None:
                return False
            _, self._held = block
            self._offset = 0
        return True

    def available(self) -> int:
        """Rows left in the currently held block."""
        if self._held is None:
            return 0
        return int(self._held.shape[0]) - self._offset

    def take(self, nrows: int) -> Array:
        """Return the next ``nrows`` rows as a view of the held block.

        ``nrows`` must not exceed :meth:`available`; the paired walk
        guarantees that by taking the smaller of the two cursors'
        availabilities.
        """
        if self._held is None or nrows > self.available():
            raise ValueError(
                f"asked for {nrows} rows but only {self.available()} "
                "are buffered; a cursor never joins blocks"
            )
        view = self._held[self._offset : self._offset + nrows, :]
        self._offset += nrows
        return view


