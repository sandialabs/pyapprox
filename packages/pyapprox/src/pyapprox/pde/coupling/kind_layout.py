r"""Where each piece's interface data and output sit in the skeleton system.

An interface carries two matching conditions, one per quantity kind. A
piece reads one kind on its ports (its
:meth:`~DeclaresKindsProtocol.receives`) and returns another (its
:meth:`~DeclaresKindsProtocol.returns`). The skeleton gives each DOF
one unknown per distinct kind its pieces receive and one residual row
per distinct kind they return:

- Where every piece receives the same kind, that kind is the DOF's only
  unknown and the rows sum the pieces' outputs. With every piece
  receiving the trace and returning the flux this is the primal
  (Dirichlet-to-Neumann) system; with every piece receiving the flux and
  returning the trace, the dual one, whose signs sit in the port
  transfers.
- Where pieces receive different kinds (a *mixed* DOF), one piece's
  input is the value another returns. The row of a returned kind
  :math:`K` adds the input :math:`u` of the piece receiving :math:`K`,
  as :math:`s_K u`, with :math:`s_K` the kind's matching sign
  (:class:`MatchingKindProtocol`). A trace-receiving piece :math:`A`
  beside a flux-receiving piece :math:`B` gives
  :math:`[q_A(\lambda) + \mu;\ t_B(\mu) - \lambda]`.

The fixed part, the entries :math:`(\text{row}, \text{unknown}, s_K)`, is
the layout's :attr:`KindLayout.coupling`. A DOF whose unknown and row
counts differ is refused: the system would not be square.

Refused until a step needs them, each needing more than one value of a
kind per DOF:

- a mixed DOF where two pieces receive the same kind (a trace there
  would need one continuity row per extra piece, a flux one multiplier
  per extra piece);
- a DOF whose only unknown is a kind with matching sign :math:`+1` (a
  flux) read by more than two pieces: a dual cross point, which needs
  one multiplier per pair.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from pyapprox.pde.coupling.responses import PortQuantity


@runtime_checkable
class DeclaresKindsProtocol(Protocol):
    """What :func:`kind_layout` asks of a piece: the kinds it receives
    and returns on its ports."""

    def receives(self) -> PortQuantity: ...

    def returns(self) -> PortQuantity: ...


@runtime_checkable
class MatchingKindProtocol(Protocol):
    """A kind whose values on a shared port match in a known form; asked
    only of kinds at a mixed DOF or read by more than two pieces."""

    def matching_sign(self) -> int:
        """The sign ``s`` for which the values ``a``, ``b`` two pieces
        hold on a shared port match when ``a + s b = 0``."""
        ...


@dataclass(frozen=True)
class CouplingEntry:
    """``sign`` times ``unknown`` added to residual ``row``."""

    row: int
    unknown: int
    sign: int


@dataclass(frozen=True)
class KindLayout:
    """The skeleton system's unknowns and rows, and each piece's place in
    it.

    Attributes
    ----------
    unknowns
        The ``(dof, kind)`` of each unknown, by unknown index.
    rows
        The ``(dof, kind)`` of each residual row, by row index.
    inputs
        Per piece, the unknown each of its port rows reads, in the
        piece's port order.
    outputs
        Per piece, the residual row each of its port rows adds to, in the
        piece's port order.
    coupling
        The fixed entries at mixed DOFs, in row order; empty when no DOF
        is mixed.
    """

    unknowns: tuple[tuple[int, PortQuantity], ...]
    rows: tuple[tuple[int, PortQuantity], ...]
    inputs: tuple[tuple[int, ...], ...]
    outputs: tuple[tuple[int, ...], ...]
    coupling: tuple[CouplingEntry, ...]

    def nunknowns(self) -> int:
        return len(self.unknowns)

    def nrows(self) -> int:
        return len(self.rows)


def _distinct(kinds: Sequence[PortQuantity]) -> list[PortQuantity]:
    """``kinds`` without repeats, in order of first appearance."""
    out: list[PortQuantity] = []
    for kind in kinds:
        if kind not in out:
            out.append(kind)
    return out


def _matching_sign(kind: PortQuantity, dof: int, why: str) -> int:
    if not isinstance(kind, MatchingKindProtocol):
        raise TypeError(
            f"DOF {dof}: {kind} must satisfy MatchingKindProtocol ({why}), "
            f"got {type(kind).__name__}"
        )
    return kind.matching_sign()


def kind_layout(
    ndofs: int,
    port_dofs: Sequence[Sequence[int]],
    pieces: Sequence[DeclaresKindsProtocol],
) -> KindLayout:
    """Number the skeleton system's unknowns and rows.

    Unknowns and rows are numbered DOF by DOF, increasing; within a DOF,
    kinds in the order pieces first declare them. Where every DOF has one
    unknown and one row, both numberings are the DOFs'.

    Parameters
    ----------
    ndofs
        The number of skeleton DOFs.
    port_dofs
        Per piece, the skeleton DOF of each port row, distinct, in the
        piece's port order.
    pieces
        Per piece, its declared kinds (a local response satisfies
        :class:`DeclaresKindsProtocol`).

    Raises
    ------
    TypeError
        If a piece does not satisfy :class:`DeclaresKindsProtocol`, or a
        kind whose matching sign is needed does not satisfy
        :class:`MatchingKindProtocol`.
    ValueError
        If ``port_dofs`` and ``pieces`` differ in length, a piece's DOFs
        repeat or lie outside ``[0, ndofs)``, a DOF is read by no piece,
        its unknown and row counts differ, or it is one of the refused
        configurations in the module docstring.
    """
    if len(port_dofs) != len(pieces):
        raise ValueError(f"{len(port_dofs)} port DOF lists for {len(pieces)} pieces")
    for i, piece in enumerate(pieces):
        if not isinstance(piece, DeclaresKindsProtocol):
            raise TypeError(
                f"piece {i} must satisfy DeclaresKindsProtocol, got "
                f"{type(piece).__name__}"
            )
        dofs = port_dofs[i]
        if len(set(dofs)) != len(dofs) or any(not 0 <= d < ndofs for d in dofs):
            raise ValueError(
                f"piece {i}'s DOFs must be distinct and lie in [0, {ndofs}), "
                f"got {tuple(dofs)}"
            )
    received = [piece.receives() for piece in pieces]
    returned = [piece.returns() for piece in pieces]
    readers: list[list[int]] = [[] for _ in range(ndofs)]
    for i, dofs in enumerate(port_dofs):
        for d in dofs:
            readers[d].append(i)

    unknowns: list[tuple[int, PortQuantity]] = []
    rows: list[tuple[int, PortQuantity]] = []
    unknown_of: dict[tuple[int, PortQuantity], int] = {}
    row_of: dict[tuple[int, PortQuantity], int] = {}
    coupling: list[CouplingEntry] = []
    for dof, at in enumerate(readers):
        if not at:
            raise ValueError(f"DOF {dof} is read by no piece")
        in_kinds = _distinct([received[i] for i in at])
        out_kinds = _distinct([returned[i] for i in at])
        if len(in_kinds) != len(out_kinds):
            raise ValueError(
                f"DOF {dof}: its pieces receive {len(in_kinds)} kinds "
                f"({', '.join(map(str, in_kinds))}) and return "
                f"{len(out_kinds)} ({', '.join(map(str, out_kinds))}); the "
                "skeleton system would not be square"
            )
        mixed = len(in_kinds) > 1
        for kind in in_kinds:
            nreaders = sum(received[i] == kind for i in at)
            if mixed and nreaders > 1:
                raise ValueError(
                    f"DOF {dof} is mixed and {nreaders} pieces receive {kind}; "
                    "a mixed DOF takes one piece per received kind"
                )
            if not mixed and nreaders > 2:
                sign = _matching_sign(kind, dof, f"{nreaders} pieces read it")
                if sign == 1:
                    raise ValueError(
                        f"DOF {dof}: {nreaders} pieces read {kind}, whose "
                        "values cancel in pairs; a dual cross point needs "
                        "one multiplier per pair"
                    )
            unknown_of[(dof, kind)] = len(unknowns)
            unknowns.append((dof, kind))
        for kind in out_kinds:
            row_of[(dof, kind)] = len(rows)
            rows.append((dof, kind))
        if mixed:
            for kind in out_kinds:
                if kind in in_kinds:
                    sign = _matching_sign(kind, dof, "it is matched at a mixed DOF")
                    coupling.append(
                        CouplingEntry(
                            row_of[(dof, kind)], unknown_of[(dof, kind)], sign
                        )
                    )
    inputs = tuple(
        tuple(unknown_of[(d, received[i])] for d in dofs)
        for i, dofs in enumerate(port_dofs)
    )
    outputs = tuple(
        tuple(row_of[(d, returned[i])] for d in dofs)
        for i, dofs in enumerate(port_dofs)
    )
    return KindLayout(tuple(unknowns), tuple(rows), inputs, outputs, tuple(coupling))
