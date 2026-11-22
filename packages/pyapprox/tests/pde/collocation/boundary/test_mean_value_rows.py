"""The closure row fixing a field's weighted mean.

On a layout of fields with 4 and 3 nodes, the row replaces node 1 of the
second field: its residual is the weighted sum minus the value there and
zero elsewhere, its Jacobian the weights in that field's columns, every
derivative is stated zero, and it is a closure row. Refused: a field or
node out of range, weights of the wrong length, a non-layout.
"""

import pytest

from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.pde.collocation.boundary.closure_rows import ClosureRowsProtocol
from pyapprox.pde.collocation.boundary.mean_value_rows import MeanValueRows
from pyapprox.pde.collocation.state_layout import StateLayout
from pyapprox.util.backends.protocols import Array, Backend


def test_row(bkd: Backend[Array]) -> None:
    layout = StateLayout([4, 3])
    weights = bkd.asarray([0.25, 0.5, 0.25])
    rows = MeanValueRows(bkd, layout, 1, weights, 2.0, node=1)
    assert isinstance(rows, ClosureRowsProtocol)
    assert rows.rows().indices() == (5,)
    state = bkd.asarray([9.0, 9.0, 9.0, 9.0, 1.0, 2.0, 5.0])
    expected = bkd.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
    bkd.assert_allclose(rows.residual(state, 0.0), expected)
    jacobian = rows.jacobian(state, 0.0)
    bkd.assert_allclose(jacobian[5], bkd.asarray([0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.25]))
    bkd.assert_allclose(bkd.sum(bkd.abs(jacobian)), bkd.asarray(1.0))
    derivatives = rows.derivatives()
    assert derivatives.param_jacobian is STATED_ZERO
    assert derivatives.state_curvature is STATED_ZERO
    assert rows.is_time_invariant()


def test_refusals(bkd: Backend[Array]) -> None:
    layout = StateLayout([4, 3])
    weights = bkd.ones((3,))
    with pytest.raises(ValueError, match="field 2 is not one of the layout's 2"):
        MeanValueRows(bkd, layout, 2, weights, 0.0)
    with pytest.raises(ValueError, match=r"weights must have shape \(4,\)"):
        MeanValueRows(bkd, layout, 0, weights, 0.0)
    with pytest.raises(ValueError, match="node 3 is not one of field 1's 3"):
        MeanValueRows(bkd, layout, 1, weights, 0.0, node=3)
    with pytest.raises(TypeError, match="StateRowsLayoutProtocol"):
        MeanValueRows(bkd, "layout", 1, weights, 0.0)  # type: ignore[arg-type]
    rows = MeanValueRows(bkd, layout, 1, weights, 0.0)
    with pytest.raises(ValueError, match=r"state must have shape \(7,\)"):
        rows.residual(bkd.zeros((6,)), 0.0)
