r"""Periodic closure rows of a collocation system.

A periodic pair of sides A and B is closed by two relations: the values
match, :math:`u_A - u_B = 0`, on the rows of A, and the derivatives in the
periodic direction match, :math:`(D_A - D_B) u = 0`, on the rows of B.
Both are linear in the state, carry no data and do not depend on the
parameters.
"""

from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, Set, Tuple

from pyapprox.ode.residual_derivatives import STATED_ZERO, backend_state_space
from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureRowsDerivatives,
    NormalTermParamFn,
)
from pyapprox.pde.collocation.boundary.essential_rows import (
    StateRowsLayoutProtocol,
    check_state_layout,
    component_state_rows,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet

# Paired nodes of a tensor mesh share their other reference coordinates
# exactly; a larger difference means the lists are misaligned
_PAIRING_ATOL = 1e-12


@dataclass(frozen=True)
class PeriodicPart(Generic[Array]):
    """One periodic pair of sides for one state component.

    Attributes
    ----------
    primary_indices : Array
        Mesh node indices of side A, whose rows carry the value match.
        Shape: (nboundary,)
    partner_indices : Array
        Mesh node indices of side B, paired entry by entry with A, whose
        rows carry the derivative match. Shape: (nboundary,)
    derivative_matrix : Array
        The derivative in the periodic direction. Shape: (npts, npts)
    component : int
        The state component (0 for a scalar physics).
    """

    primary_indices: Array
    partner_indices: Array
    derivative_matrix: Array
    component: int = 0


class _PeriodicPartRows(Generic[Array]):
    def __init__(
        self,
        bkd: Backend[Array],
        part: PeriodicPart[Array],
        primary: List[int],
        partner: List[int],
        start: int,
        npts: int,
        nstates: int,
    ) -> None:
        self.primary = bkd.asarray(primary, dtype=bkd.int64_dtype())
        self.partner = bkd.asarray(partner, dtype=bkd.int64_dtype())
        identity = bkd.eye(nstates)
        # Rows of A: u_A - u_B
        self.value_block = identity[self.primary] - identity[self.partner]
        # Rows of B: (D_A - D_B) u on the part's component columns, which
        # start at row ``start`` and span the component's ``npts`` nodes
        mesh_primary = self.primary - start
        mesh_partner = self.partner - start
        derivative = (
            part.derivative_matrix[mesh_primary] - part.derivative_matrix[mesh_partner]
        )
        self.derivative_block = bkd.hstack(
            [
                bkd.zeros((len(partner), start)),
                derivative,
                bkd.zeros((len(partner), nstates - start - npts)),
            ]
        )


class PeriodicRows(Generic[Array]):
    r"""Periodic closure rows, as
    :class:`~pyapprox.pde.collocation.boundary.closure_rows.ClosureRowsProtocol`
    declares them.

    Every derivative beyond the state Jacobian is stated zero. Parts may
    share rows (a doubly periodic square's corners lie in both pairs); a
    later part wins a shared row.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    layout : StateRowsLayoutProtocol
        Where each component sits in the state, e.g.
        ``StateLayout.uniform(npts, ncomponents)``.
    reference_points : Array
        The reference coordinates of the periodic components' mesh, used to
        check each pairing. Shape: (ndim, npts)
    parts : sequence of PeriodicPart
        The pairs, in the order that settles shared rows.

    Raises
    ------
    TypeError
        If ``layout`` does not satisfy
        :class:`~pyapprox.pde.collocation.boundary.essential_rows.StateRowsLayoutProtocol`.
    ValueError
        If there are no parts, a component or index is out of range, a
        part's component does not have one node per reference point, A
        and B differ in length or share a node, the derivative matrix is
        not (npts, npts), or a pair's nodes differ in a reference
        coordinate other than one periodic direction common to all pairs
        (reversed or misaligned lists).
    """

    def __init__(
        self,
        bkd: Backend[Array],
        layout: StateRowsLayoutProtocol,
        reference_points: Array,
        parts: Sequence[PeriodicPart[Array]],
    ) -> None:
        check_state_layout(layout)
        if not parts:
            raise ValueError("a periodic declaration needs at least one part")
        self._bkd = bkd
        npts = reference_points.shape[1]
        self._nstates = layout.nstates()
        self._parts: List[_PeriodicPartRows[Array]] = []
        all_rows: List[int] = []
        for part in parts:
            primary = component_state_rows(
                bkd, part.primary_indices, part.component, layout
            )
            partner = component_state_rows(
                bkd, part.partner_indices, part.component, layout
            )
            if layout.npts(part.component) != npts:
                raise ValueError(
                    f"component {part.component} has "
                    f"{layout.npts(part.component)} nodes, but the reference "
                    f"points give {npts}"
                )
            self._check_part(part, reference_points)
            self._parts.append(
                _PeriodicPartRows(
                    bkd,
                    part,
                    primary,
                    partner,
                    layout.offset(part.component),
                    npts,
                    self._nstates,
                )
            )
            all_rows.extend(primary + partner)
        self._rows = BackendRowSet(
            backend_state_space(bkd, self._nstates).descriptor(), all_rows, bkd
        )
        self._jacobian = self._assemble_jacobian()

    def _check_part(self, part: PeriodicPart[Array], reference_points: Array) -> None:
        bkd = self._bkd
        npts = reference_points.shape[1]
        if part.derivative_matrix.shape != (npts, npts):
            raise ValueError(
                f"the derivative matrix must have shape ({npts}, {npts}), got "
                f"{tuple(part.derivative_matrix.shape)}"
            )
        primary = [bkd.to_int(i) for i in part.primary_indices]
        partner = [bkd.to_int(i) for i in part.partner_indices]
        if len(primary) != len(partner):
            raise ValueError(
                f"primary and partner sides differ in length: {len(primary)} "
                f"and {len(partner)}"
            )
        if not primary:
            raise ValueError("a periodic part needs at least one pair")
        if set(primary) & set(partner):
            raise ValueError("primary and partner sides share nodes")
        gaps = bkd.abs(
            reference_points[:, part.primary_indices]
            - reference_points[:, part.partner_indices]
        )
        differing: Set[int] = {
            dim
            for dim in range(reference_points.shape[0])
            if bkd.to_float(bkd.max(gaps[dim])) > _PAIRING_ATOL
        }
        periodic = [
            dim for dim in differing if bkd.to_float(bkd.min(gaps[dim])) > _PAIRING_ATOL
        ]
        if len(differing) != 1 or len(periodic) != 1:
            raise ValueError(
                "each primary node must pair with the partner node that "
                "differs from it only in one periodic reference coordinate, "
                f"the same for every pair; coordinates differing: "
                f"{sorted(differing)}"
            )

    def _assemble_jacobian(self) -> Array:
        bkd = self._bkd
        matrix = bkd.zeros((self._nstates, self._nstates))
        for part in self._parts:
            matrix = bkd.index_update(matrix, part.primary, part.value_block)
            matrix = bkd.index_update(matrix, part.partner, part.derivative_block)
        return matrix

    def rows(self) -> BackendRowSet[Array]:
        """The replaced rows: each part's primary and partner rows."""
        return self._rows

    def _check_state(self, state: Array) -> None:
        if state.shape != (self._nstates,):
            raise ValueError(
                f"state must have shape ({self._nstates},), got {tuple(state.shape)}"
            )

    def residual(self, state: Array, time: float) -> Array:
        r""":math:`u_A - u_B` on the primary rows and :math:`(D_A - D_B) u`
        on the partner rows, zero elsewhere. Shape: (nstates,)"""
        self._check_state(state)
        return self._jacobian @ state

    def jacobian(self, state: Array, time: float) -> Array:
        """The constant Jacobian on the rows, zero elsewhere.
        Shape: (nstates, nstates)"""
        self._check_state(state)
        return self._jacobian

    def pairs(self) -> List[Tuple[List[int], List[int]]]:
        """Each part's paired state rows: (primary rows, partner rows),
        entry by entry."""
        bkd = self._bkd
        return [
            (
                [bkd.to_int(i) for i in part.primary],
                [bkd.to_int(i) for i in part.partner],
            )
            for part in self._parts
        ]

    def derivatives(
        self, normal_term_param: Optional[NormalTermParamFn[Array]] = None
    ) -> ClosureRowsDerivatives[Array]:
        """Every derivative stated zero: the rows are linear in the state
        and do not depend on the parameters. ``normal_term_param`` is
        unused: the rows have no normal term."""
        return ClosureRowsDerivatives(
            STATED_ZERO, STATED_ZERO, STATED_ZERO, STATED_ZERO
        )

    def is_time_invariant(self) -> bool:
        """True: the rows hold no data and a fixed matrix."""
        return True

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nparts={len(self._parts)}, "
            f"nrows={self._rows.count()})"
        )
