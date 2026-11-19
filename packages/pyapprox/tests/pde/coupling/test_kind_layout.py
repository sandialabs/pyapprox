"""The kind layout: unknowns, rows and mixed-DOF coupling from the kinds
each piece receives and returns.

Checks: an all-trace partition numbers unknowns and rows as its DOFs; a
trace-receiving piece beside a flux-receiving one carries both unknowns
at their shared DOF with C = [[0, 1], [-1, 0]]; two custom kinds received
and returned by both pieces (the Stokes-Darcy shape) give one unknown
without asking either kind for a matching sign; and each refused
configuration raises.
"""

from dataclasses import dataclass

import pytest

from pyapprox.pde.coupling.kind_layout import CouplingEntry, kind_layout
from pyapprox.pde.coupling.responses import Flux, PortQuantity, Trace


@dataclass(frozen=True)
class _Piece:
    received: PortQuantity
    returned: PortQuantity

    def receives(self) -> PortQuantity:
        return self.received

    def returns(self) -> PortQuantity:
        return self.returned


@dataclass(frozen=True)
class _Pressure(PortQuantity):
    pass


@dataclass(frozen=True)
class _NormalVelocity(PortQuantity):
    pass


_DTN = _Piece(Trace(), Flux())
_NTD = _Piece(Flux(), Trace())


def test_all_trace_chain_numbers_as_dofs() -> None:
    layout = kind_layout(2, [(0,), (0, 1), (1,)], [_DTN, _DTN, _DTN])
    assert layout.unknowns == ((0, Trace()), (1, Trace()))
    assert layout.rows == ((0, Flux()), (1, Flux()))
    assert layout.inputs == ((0,), (0, 1), (1,))
    assert layout.outputs == ((0,), (0, 1), (1,))
    assert layout.coupling == ()


def test_trace_cross_point_has_one_unknown() -> None:
    # Four pieces sharing a 2D vertex: a trace is one shared value.
    layout = kind_layout(1, [(0,)] * 4, [_DTN] * 4)
    assert (layout.nunknowns(), layout.nrows()) == (1, 1)
    assert layout.inputs == ((0,),) * 4


def test_mixed_pair_couples_trace_and_flux() -> None:
    layout = kind_layout(1, [(0,), (0,)], [_DTN, _NTD])
    assert layout.unknowns == ((0, Trace()), (0, Flux()))
    assert layout.rows == ((0, Flux()), (0, Trace()))
    assert layout.inputs == ((0,), (1,))
    assert layout.outputs == ((0,), (1,))
    # [q_A + mu; t_B - lambda]: C = [[0, 1], [-1, 0]].
    assert layout.coupling == (CouplingEntry(0, 1, 1), CouplingEntry(1, 0, -1))


def test_mixed_and_dual_dofs_in_one_chain() -> None:
    # A | B | C with A trace-receiving, B and C flux-receiving: DOF 0 is
    # mixed, DOF 1 is dual with mu as its only unknown.
    layout = kind_layout(2, [(0,), (0, 1), (1,)], [_DTN, _NTD, _NTD])
    assert layout.unknowns == ((0, Trace()), (0, Flux()), (1, Flux()))
    assert layout.rows == ((0, Flux()), (0, Trace()), (1, Trace()))
    assert layout.inputs == ((0,), (1, 2), (2,))
    assert layout.outputs == ((0,), (1, 2), (2,))
    assert layout.coupling == (CouplingEntry(0, 1, 1), CouplingEntry(1, 0, -1))


def test_custom_kinds_shared_by_both_pieces_give_one_unknown() -> None:
    # Stokes-Darcy shape: both receive the pressure and return the normal
    # velocity. The custom kinds declare no matching sign and none is
    # asked for.
    piece = _Piece(_Pressure(), _NormalVelocity())
    layout = kind_layout(1, [(0,), (0,)], [piece, piece])
    assert layout.unknowns == ((0, _Pressure()),)
    assert layout.rows == ((0, _NormalVelocity()),)
    assert layout.coupling == ()


def test_refuses_non_square_dof() -> None:
    with pytest.raises(ValueError, match="not be square"):
        kind_layout(1, [(0,), (0,)], [_DTN, _Piece(Flux(), Flux())])


def test_refuses_dual_cross_point() -> None:
    with pytest.raises(ValueError, match="dual cross point"):
        kind_layout(1, [(0,)] * 3, [_NTD] * 3)


def test_refuses_two_receivers_of_one_kind_at_mixed_dof() -> None:
    with pytest.raises(ValueError, match="one piece per received kind"):
        kind_layout(1, [(0,)] * 3, [_DTN, _DTN, _NTD])


def test_refuses_kind_without_matching_sign_where_needed() -> None:
    piece = _Piece(_Pressure(), _NormalVelocity())
    with pytest.raises(TypeError, match="MatchingKindProtocol"):
        kind_layout(1, [(0,)] * 3, [piece] * 3)


def test_refuses_unread_dof() -> None:
    with pytest.raises(ValueError, match="read by no piece"):
        kind_layout(2, [(0,)], [_DTN])


@pytest.mark.parametrize("dofs", [(0, 0), (2,), (-1,)])
def test_refuses_bad_port_dofs(dofs: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="distinct"):
        kind_layout(2, [dofs], [_DTN])


def test_refuses_length_mismatch() -> None:
    with pytest.raises(ValueError, match="port DOF lists"):
        kind_layout(1, [(0,), (0,)], [_DTN])


def test_refuses_piece_without_kinds() -> None:
    with pytest.raises(TypeError, match="DeclaresKindsProtocol"):
        kind_layout(1, [(0,)], [object()])  # type: ignore[list-item]
