r"""Forming the ambient basis from the small factor that determines it.

A snapshot decomposition produces three things, and only one of them is
ambient-sized:

=================  ======================  ==========================
piece              shape                   at :math:`10^7` states
=================  ======================  ==========================
eigenvalues        ``(nterms,)``           400 bytes
coordinates        ``(nterms, nsamples)``  400 KB
eigenvectors       ``(nstates, nterms)``   **4 GB**
=================  ======================  ==========================

Every solver here computes the first two without touching the third.
The method of snapshots gets :math:`Q` and :math:`\lambda` from an
``(nsamples, nsamples)`` eigenproblem; the thin SVD gets :math:`\Psi`
and :math:`s` from an equally small one. Both then *derive* the
eigenvectors, as

.. math:: V = S\, R

for a right factor :math:`R` of shape ``(nsamples, nterms)`` -- for the
method of snapshots :math:`R = Q\,\Lambda^{-1/2}`, for the SVD
:math:`R = \Psi\,\mathrm{diag}(s)^{-1}`.

That last step is the only one that needs the ambient dimension, and it
is a contraction of the snapshots against something small on the right.
So it decomposes over row blocks exactly: block ``i:j`` of :math:`V` is
block ``i:j`` of :math:`S` times :math:`R`, with no coupling between
blocks. This module is that step, kept separate from the solvers so each
supplies only its own :math:`R` and none of them reimplements blocking.

**The sign convention is the reason there are two passes.** The
convention sets each column's sign from its largest-magnitude entry,
which is a property of the whole ambient column -- so it cannot be known
while writing the first blocks. Rather than re-reading and flipping a
4 GB result, the pivots are found in a pass that writes nothing, and the
sign is folded into :math:`R` before the write pass begins. This works
because a column scaling commutes through the lift:

.. math:: V \,\mathrm{diag}(\sigma) = (S R)\,\mathrm{diag}(\sigma)
          = S\,(R\,\mathrm{diag}(\sigma))

so flipping the ``(nsamples, nterms)`` factor and flipping the
``(nstates, nterms)`` result are the same operation, one applied where
it is cheap. The running maximum this needs is exact rather than
approximate: ``max`` is associative and each candidate is an unmodified
element rather than an accumulated sum, so no tolerance enters and the
block size cannot change the answer.
"""

from typing import Optional

from pyapprox.surrogates.kle.basis_operator import BasisOperatorProtocol
from pyapprox.surrogates.kle.basis_sinks import BasisSinkProtocol
from pyapprox.surrogates.kle.snapshot_sources import (
    SnapshotSourceProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


def lift_basis(
    source: SnapshotSourceProtocol[Array],
    right_factor: Array,
    sink: BasisSinkProtocol[Array],
    bkd: Backend[Array],
    apply_sign_convention: bool = True,
    max_bytes: Optional[int] = None,
) -> BasisOperatorProtocol[Array]:
    r"""Write :math:`V = S R` to ``sink``, a row block at a time.

    The ambient half of a snapshot decomposition. What a solver calls
    once it holds its right factor, and the reason a solver does not
    need to know that sources or sinks exist.

    Parameters
    ----------
    source : SnapshotSourceProtocol[Array]
        The snapshots, read in row blocks. Read twice when
        ``apply_sign_convention`` is True, once otherwise.
    right_factor : Array, shape (nsamples, nterms)
        :math:`R`, with any eigenvalue scaling already folded in --
        :math:`Q\,\Lambda^{-1/2}` for the method of snapshots. Taken
        pre-scaled rather than as a factor and a spectrum because the
        two are one column scaling and separating them would invite a
        caller to apply it twice.
    sink : BasisSinkProtocol[Array]
        Where the blocks go. Sized ``(source.nstates(), nterms)``.
    bkd : Backend[Array]
        Computational backend.
    apply_sign_convention : bool
        Whether to make each column's sign deterministic, matching what
        a resident decomposition produces. Costs one extra read of the
        source, which writes nothing but is a full pass over the data.

        Leaving it on is the safer default: a basis whose signs differ
        from the resident path reconstructs its own data correctly and
        still disagrees with a stored basis, a plotted mode, or a
        coefficient whose sign was assumed -- a discrepancy that shows
        up far from here. Turn it off when the sign genuinely does not
        matter downstream and the extra pass does.
    max_bytes : int, optional
        Byte budget per row block. None lets the source choose.

    Returns
    -------
    BasisOperatorProtocol[Array]
        Whatever ``sink.finalize()`` returns -- an array-backed basis
        or one that reads the blocks back, depending on the sink.
    """
    validate_lift_arguments(source, right_factor, sink, bkd)
    factor = right_factor
    if apply_sign_convention:
        factor = factor * pivot_signs(source, factor, bkd, max_bytes)
    for rows, block in source.row_blocks(max_bytes):
        sink.write(rows, bkd.dot(block, factor))
    return sink.finalize()


def pivot_signs(
    source: SnapshotSourceProtocol[Array],
    right_factor: Array,
    bkd: Backend[Array],
    max_bytes: Optional[int] = None,
) -> Array:
    r"""Return the sign the convention gives each column of :math:`S R`.

    Shape ``(nterms,)``, entries ``+1`` or ``-1``. Multiply
    ``right_factor`` by this and the lift produces a signed basis
    directly, which is cheaper than signing the result: the factor is
    ``(nsamples, nterms)`` and the result is ``(nstates, nterms)``.

    Computes a running maximum of ``|S R|`` down the rows, keeping only
    the largest entry seen per column rather than the column itself.
    Exact: the comparison is between unmodified elements, so it neither
    accumulates error nor depends on where the block boundaries fall.

    Ties go to the lowest row index, matching ``argmax`` in the resident
    path -- a strict ``>`` keeps the earlier candidate, and blocks are
    visited in increasing row order.
    """
    nterms = int(right_factor.shape[1])
    best = bkd.zeros((nterms,))
    for _, block in source.row_blocks(max_bytes):
        candidates = bkd.dot(block, right_factor)
        rows = bkd.argmax(bkd.abs(candidates), axis=0)
        values = bkd.get_diagonal(candidates[rows, :])
        best = bkd.where(
            bkd.abs(values) > bkd.abs(best), values, best
        )
    signs = bkd.sign(best)
    # A column that is zero everywhere has no orientation to
    # canonicalize, and sign() is 0 there, which would erase it.
    return bkd.where(
        bkd.equal(signs, 0.0), bkd.full(signs.shape, 1.0), signs
    )


def validate_lift_arguments(
    source: SnapshotSourceProtocol[Array],
    right_factor: Array,
    sink: BasisSinkProtocol[Array],
    bkd: Backend[Array],
) -> None:
    """Check the three objects describe one decomposition.

    Each mismatch would otherwise surface inside the block loop, naming
    shapes that have already been combined: a right factor of the wrong
    height fails on the first ``dot``, and a mis-sized sink fails on the
    first ``write`` -- both after an arbitrary amount of I/O.
    """
    if not isinstance(source, SnapshotSourceProtocol):
        raise TypeError(
            "source must satisfy SnapshotSourceProtocol, got "
            f"{type(source).__name__}"
        )
    if not isinstance(sink, BasisSinkProtocol):
        raise TypeError(
            "sink must satisfy BasisSinkProtocol, got "
            f"{type(sink).__name__}"
        )
    if right_factor.ndim != 2:
        raise ValueError(
            "right_factor must be 2D (nsamples, nterms), got "
            f"ndim={right_factor.ndim}"
        )
    if int(right_factor.shape[0]) != source.nsamples():
        raise ValueError(
            f"right_factor has {right_factor.shape[0]} rows but the "
            f"source has {source.nsamples()} snapshots; the lift "
            "contracts one against the other"
        )
    if sink.nstates() != source.nstates():
        raise ValueError(
            f"sink is sized for {sink.nstates()} states but the source "
            f"has {source.nstates()}"
        )
    if sink.nterms() != int(right_factor.shape[1]):
        raise ValueError(
            f"sink is sized for {sink.nterms()} terms but right_factor "
            f"has {right_factor.shape[1]} columns"
        )
