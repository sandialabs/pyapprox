"""Group DOFs and fixed values for nested substructuring.

Checks: a group's DOFs, outer positions, local numbers and inner
positions on a hand-made chain of children; refusals; fixed values hold
their data with parameter-independent derivatives (both backends).
"""

import pytest

from pyapprox.implicitfunction.pinned_rows import PinnedValuesProtocol
from pyapprox.implicitfunction.residual_derivatives import StatedZero
from pyapprox.pde.coupling.nesting import FixedValues, GroupDofs, group_dofs
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet, BackendVectorSpace
from pyapprox.util.linalg.operators import SpaceDescriptor

# Four children in a chain sharing one DOF each: 0-1 share 2, 1-2 share
# 5, 2-3 share 8.
_PORTS = ((0, 1, 2), (2, 3, 4, 5), (5, 6, 7, 8), (8, 9))


def test_group_dofs_of_a_chain_segment() -> None:
    group = group_dofs([_PORTS[1], _PORTS[2]], [_PORTS[0], _PORTS[3]])
    assert group.dofs == (2, 3, 4, 5, 6, 7, 8)
    # DOFs 2 and 8 are also read by children 0 and 3.
    assert group.outer == (0, 6)
    assert group.outer_parent() == (2, 8)
    assert group.inner() == (1, 2, 3, 4, 5)
    assert group.size() == 7
    assert group.local([8, 5, 2]) == (6, 3, 0)


def test_a_whole_partition_has_no_outer_dofs() -> None:
    group = group_dofs(_PORTS, [])
    assert group.dofs == tuple(range(10)) and group.outer == ()


def test_a_member_group_shows_only_its_ports() -> None:
    inner = group_dofs([_PORTS[1], _PORTS[2]], [_PORTS[0], _PORTS[3]])
    outer = group_dofs([_PORTS[0], inner.outer_parent()], [_PORTS[3]])
    # The inner group's own DOFs 3-7 are not the outer group's: no member
    # of it reads them.
    assert outer.dofs == (0, 1, 2, 8)
    assert outer.local(inner.outer_parent()) == (2, 3)
    assert outer.outer_parent() == (8,)


def test_refusals() -> None:
    with pytest.raises(ValueError, match="at least one"):
        group_dofs([], _PORTS)
    with pytest.raises(ValueError, match="no skeleton DOF"):
        group_dofs([()], [(0,)])
    with pytest.raises(ValueError, match="not the group's"):
        GroupDofs((2, 3), ()).local([4])


def test_fixed_values(bkd: Backend[Array]) -> None:
    descriptor = SpaceDescriptor("skeleton", 4)
    space = BackendVectorSpace(descriptor, bkd)
    values = bkd.asarray([[0.0], [1.5], [0.0], [-2.0]])
    rows = BackendRowSet(descriptor, [1, 3], bkd)
    fixed = FixedValues(rows, values, space)
    assert isinstance(fixed, PinnedValuesProtocol)
    values[1, 0] = 7.0
    # The data were copied.
    bkd.assert_allclose(fixed.values(), bkd.asarray([[0.0], [1.5], [0.0], [-2.0]]))
    derivs = fixed.parameter_derivatives()
    assert isinstance(derivs.values, StatedZero)
    assert isinstance(derivs.curvature, StatedZero)
    other = BackendRowSet(SpaceDescriptor("other", 4), [1], bkd)
    with pytest.raises(ValueError):
        FixedValues(other, values, space)
    with pytest.raises(ValueError, match="single column"):
        FixedValues(rows, bkd.zeros((4, 2)), space)
