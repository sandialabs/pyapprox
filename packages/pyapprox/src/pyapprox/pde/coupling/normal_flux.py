r"""A port flux computed pointwise from the state, as collocation does.

In a weak (Galerkin) discretization, integration by parts leaves a
boundary term on every row, so the residual left on a pinned row at a
solution, the constraint reaction, is the boundary flux integrated
against that row's basis function. A discretization whose rows are the
PDE evaluated pointwise (spectral collocation, any physics) integrates
nothing by parts: what remains on a pinned row is the PDE's pointwise
error at that node, not a flux. Its port flux is instead the normal flux
:math:`q = \mathbf{F}(u) \cdot n`, evaluated at the trace's quadrature
points by operators the discretization supplies, and integrated against
each of the port's trace functions, :math:`r_n = \int q\, \phi_n`, by the
trace's quadrature (:func:`~pyapprox.pde.coupling.pointwise_flux.
flux_moments`). Its tangent is the operators' Jacobian applied to the
state's change, integrated the same way.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, runtime_checkable

from pyapprox.implicitfunction.steady_system import SteadyPoint
from pyapprox.pde.coupling.pointwise_flux import flux_moments
from pyapprox.pde.coupling.ports import PortTraceProtocol, trace_shape
from pyapprox.util.backends.protocols import Array


@runtime_checkable
class NormalFluxOperatorProtocol(Protocol[Array]):
    """What :class:`NormalFluxMoments` asks of a pointwise normal-flux
    operator, e.g. a collocation ``FluxNormalOperator`` or
    ``TractionNormalOperator`` on one side of a subdomain."""

    def __call__(self, state: Array, time: float) -> Array:
        """The outward normal flux at the operator's boundary nodes.
        ``state`` has shape (nstates,). Shape: (nboundary,)."""
        ...

    def jacobian(self, state: Array, time: float) -> Array:
        """Its Jacobian. Shape: (nboundary, nstates)."""
        ...


@dataclass(frozen=True)
class NormalFluxPart(Generic[Array]):
    """One operator's contribution to a port: which of its boundary nodes
    are trace quadrature points, in the trace's point order."""

    operator: NormalFluxOperatorProtocol[Array]
    positions: Sequence[int]


class NormalFluxMoments(Generic[Array]):
    r"""The outward normal flux of a state, tested against the port's
    trace functions.

    Satisfies :class:`~pyapprox.pde.coupling.pinned_response.
    PortFluxProtocol` with backend columns as states. The parts' selected
    nodes, concatenated in order, are the trace's quadrature points: for a
    :class:`~pyapprox.pde.coupling.traces.LumpedNodalTrace`, the port
    nodes. Scalar fluxes only (one component).

    Parameters
    ----------
    parts
        The operators and their selected boundary nodes.
    trace
        The port's trace functions and quadrature.
    time
        The time the operators' coefficients are evaluated at.

    Raises
    ------
    TypeError
        If an operator or ``trace`` does not satisfy its protocol.
    ValueError
        If there are no parts, or the parts select a different number of
        nodes than the trace has quadrature points.
    """

    def __init__(
        self,
        parts: Sequence[NormalFluxPart[Array]],
        trace: PortTraceProtocol[Array],
        time: float,
    ) -> None:
        nelems, nquad, _ = trace_shape(trace)
        if not parts:
            raise ValueError("a port flux needs at least one part")
        for part in parts:
            if not isinstance(part.operator, NormalFluxOperatorProtocol):
                raise TypeError(
                    "operators must satisfy NormalFluxOperatorProtocol, got "
                    f"{type(part.operator).__name__}"
                )
        nselected = sum(len(part.positions) for part in parts)
        if nselected != nelems * nquad:
            raise ValueError(
                f"the parts select {nselected} nodes, the trace has "
                f"{nelems * nquad} quadrature points"
            )
        bkd = trace.bkd()
        self._parts = [
            (part.operator, bkd.asarray(list(part.positions), dtype=bkd.int64_dtype()))
            for part in parts
        ]
        self._trace, self._time, self._bkd = trace, time, bkd

    def flux(self, state: Array) -> Array:
        """The tested outward flux at ``state``, shape (nstates, 1).
        Shape: (nport, 1)."""
        values = self._bkd.concatenate(
            [op(state[:, 0], self._time)[rows] for op, rows in self._parts], axis=0
        )
        return flux_moments(self._trace, self._bkd.reshape(values, (-1, 1)))

    def flux_tangent(
        self, state: Array, delta: Array, point: SteadyPoint[Array, Array]
    ) -> Array:
        """The tested flux's change for ``delta``, shape (nstates, ncols):
        the operators' Jacobian at ``state`` applied to it. ``point`` is not
        read: the operators carry no linearization label. Shape: (nport,
        ncols)."""
        jacobian = self._bkd.concatenate(
            [op.jacobian(state[:, 0], self._time)[rows] for op, rows in self._parts],
            axis=0,
        )
        return flux_moments(self._trace, jacobian @ delta)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nparts={len(self._parts)})"
