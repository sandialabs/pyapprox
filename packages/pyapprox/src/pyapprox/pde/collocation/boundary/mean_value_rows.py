r"""A closure row fixing a field's weighted mean.

A field the equations determine only up to a constant (the pressure of
incompressible flow with the velocity prescribed on the whole boundary)
needs one more condition. This row replaces one of the field's rows with

.. math::

    c(u) = \sum_i w_i u_i - m = 0,

the weights :math:`w` integrating the field's interpolant (so the
condition fixes its integral, or its mean when they sum to one). It does
not depend on the time or on any parameter.
"""

from typing import Generic, Optional

from pyapprox.ode.residual_derivatives import STATED_ZERO, backend_state_space
from pyapprox.pde.collocation.boundary.closure_rows import (
    ClosureRowsDerivatives,
    NormalTermParamFn,
)
from pyapprox.pde.collocation.boundary.essential_rows import (
    StateRowsLayoutProtocol,
    check_state_layout,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendRowSet


class MeanValueRows(Generic[Array]):
    r"""One row :math:`\sum_i w_i u_i - m` of a field, as
    :class:`~pyapprox.pde.collocation.boundary.closure_rows.ClosureRowsProtocol`
    declares closure rows.

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    layout : StateRowsLayoutProtocol
        Where each field sits in the state.
    field : int
        The field whose mean is fixed.
    weights : Array
        :math:`w`, one per node of the field. Shape: (npts,)
    value : float
        :math:`m`.
    node : int, optional
        The node of ``field`` whose row is replaced; default 0.

    Raises
    ------
    TypeError
        If ``layout`` does not satisfy ``StateRowsLayoutProtocol``.
    ValueError
        If ``field`` or ``node`` is out of range, or ``weights`` does not
        have one entry per node of the field.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        layout: StateRowsLayoutProtocol,
        field: int,
        weights: Array,
        value: float,
        node: int = 0,
    ) -> None:
        check_state_layout(layout)
        if not 0 <= field < layout.nfields():
            raise ValueError(
                f"field {field} is not one of the layout's {layout.nfields()} fields"
            )
        npts = layout.npts(field)
        if tuple(weights.shape) != (npts,):
            raise ValueError(
                f"weights must have shape ({npts},), one per node of field "
                f"{field}; got {tuple(weights.shape)}"
            )
        if not 0 <= node < npts:
            raise ValueError(f"node {node} is not one of field {field}'s {npts} nodes")
        self._bkd, self._nstates = bkd, layout.nstates()
        self._start = layout.offset(field)
        self._columns = bkd.arange(self._start, self._start + npts)
        self._row = self._start + node
        self._weights, self._value = weights, value
        self._rows = BackendRowSet(
            backend_state_space(bkd, self._nstates).descriptor(), [self._row], bkd
        )

    def rows(self) -> BackendRowSet[Array]:
        """The replaced row."""
        return self._rows

    def _check_state(self, state: Array) -> None:
        if tuple(state.shape) != (self._nstates,):
            raise ValueError(
                f"state must have shape ({self._nstates},), got {tuple(state.shape)}"
            )

    def residual(self, state: Array, time: float) -> Array:
        r""":math:`\sum_i w_i u_i - m` on the row, zero elsewhere.
        Shape: (nstates,)"""
        self._check_state(state)
        bkd = self._bkd
        npts = self._weights.shape[0]
        mean = bkd.sum(self._weights * state[self._start : self._start + npts])
        return bkd.index_update(
            bkd.zeros((self._nstates,)),
            bkd.asarray([self._row], dtype=bkd.index_dtype()),
            bkd.reshape(mean - self._value, (1,)),
        )

    def jacobian(self, state: Array, time: float) -> Array:
        """:math:`w` in the field's columns of the row, zero elsewhere.
        Shape: (nstates, nstates)"""
        self._check_state(state)
        bkd = self._bkd
        row = bkd.index_update(
            bkd.zeros((self._nstates,)), self._columns, self._weights
        )
        return bkd.index_update(
            bkd.zeros((self._nstates, self._nstates)),
            bkd.asarray([self._row], dtype=bkd.index_dtype()),
            bkd.reshape(row, (1, self._nstates)),
        )

    def derivatives(
        self, normal_term_param: Optional[NormalTermParamFn[Array]] = None
    ) -> ClosureRowsDerivatives[Array]:
        """Every derivative stated zero: the row is linear in the state
        and reads no parameter."""
        return ClosureRowsDerivatives(
            STATED_ZERO, STATED_ZERO, STATED_ZERO, STATED_ZERO
        )

    def is_time_invariant(self) -> bool:
        """True: neither the weights nor the value depend on time."""
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(row={self._row}, value={self._value!r})"
