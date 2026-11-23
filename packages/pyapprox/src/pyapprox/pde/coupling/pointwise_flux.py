r"""Tested fluxes for models that return a flux pointwise.

A port returns its flux tested against its trace functions,
:math:`r_n = \int \phi_n q`: a Galerkin reaction already is one. A model
that computes the flux pointwise (collocation, a learned field model)
becomes one here, inside its response, by the port's trace quadrature:
:math:`r_n = \sum_q w_q\, q(x_q)\, \phi_n(x_q)`. No transfer then needs
to know what a producer returns.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.pde.coupling.ports import PortTraceProtocol, trace_shape
from pyapprox.pde.coupling.responses import (
    Flux,
    PortQuantity,
    ResponseDerivatives,
    ResponsePointProtocol,
    TimeWindow,
)
from pyapprox.util.backends.protocols import Array, Backend


def flux_moments(
    trace: PortTraceProtocol[Array], values: Array, ncomponents: int = 1
) -> Array:
    r"""Pointwise flux values tested against the trace functions:
    :math:`r_{k a + c} = \sum_e \sum_q w_{eq}\, \phi_{eqa}\, q_{(e,q),c}`.

    Linear in ``values``, and computed by backend products only (the
    element sums enter through a constant 0/1 placement), so a torch graph
    through ``values`` survives.

    Parameters
    ----------
    trace
        The port's trace functions and quadrature.
    values
        The flux at the trace's points, node-major per point (row
        ``k p + c`` for point ``p = e * nquad + q``). Shape:
        (k * nelems * nquad, ncols).
    ncomponents
        :math:`k`.

    Returns
    -------
    Array
        Shape (k * trace.nnodes(), ncols), node-major.

    Raises
    ------
    TypeError
        If ``trace`` does not satisfy :class:`PortTraceProtocol`.
    ValueError
        If ``ncomponents`` is not positive or ``values`` has the wrong
        shape.
    """
    nelems, nquad, nlocal = trace_shape(trace)
    if ncomponents < 1:
        raise ValueError(f"ncomponents must be >= 1, got {ncomponents}")
    k, bkd = ncomponents, trace.bkd()
    if values.ndim != 2 or values.shape[0] != k * nelems * nquad:
        raise ValueError(
            f"values must have shape ({k * nelems * nquad}, ncols), "
            f"got {tuple(values.shape)}"
        )
    ncols = int(values.shape[1])
    q = bkd.reshape(values, (nelems, nquad, k, ncols))
    local = bkd.einsum("eq,eqa,eqcj->eacj", trace.weights(), trace.values(), q)
    flat = bkd.reshape(local, (nelems * nlocal, k * ncols))
    return bkd.reshape(_placement(trace) @ flat, (trace.nnodes() * k, ncols))


def flux_moments_transpose(
    trace: PortTraceProtocol[Array], moments: Array, ncomponents: int = 1
) -> Array:
    r"""The transpose of :func:`flux_moments`: :math:`q_{(e,q),c} = w_{eq}
    \sum_a \phi_{eqa}\, r_{k a + c}`, the moments' weights read back at
    the trace's points.

    Parameters
    ----------
    trace
        The port's trace functions and quadrature.
    moments
        Weights on the moments, node-major. Shape: (k * trace.nnodes(),
        ncols).
    ncomponents
        :math:`k`.

    Returns
    -------
    Array
        Shape (k * nelems * nquad, ncols), node-major per point.

    Raises
    ------
    TypeError
        If ``trace`` does not satisfy :class:`PortTraceProtocol`.
    ValueError
        If ``ncomponents`` is not positive or ``moments`` has the wrong
        shape.
    """
    nelems, nquad, nlocal = trace_shape(trace)
    if ncomponents < 1:
        raise ValueError(f"ncomponents must be >= 1, got {ncomponents}")
    k, bkd = ncomponents, trace.bkd()
    if moments.ndim != 2 or moments.shape[0] != k * trace.nnodes():
        raise ValueError(
            f"moments must have shape ({k * trace.nnodes()}, ncols), "
            f"got {tuple(moments.shape)}"
        )
    ncols = int(moments.shape[1])
    nodal = bkd.reshape(moments, (trace.nnodes(), k * ncols))
    local = bkd.reshape(
        bkd.transpose(_placement(trace)) @ nodal, (nelems, nlocal, k, ncols)
    )
    q = bkd.einsum("eq,eqa,eacj->eqcj", trace.weights(), trace.values(), local)
    return bkd.reshape(q, (nelems * nquad * k, ncols))


def _placement(trace: PortTraceProtocol[Array]) -> Array:
    """The 0/1 matrix summing element-local node rows into the trace's
    nodes. Shape: (trace.nnodes(), nelems * nlocal)."""
    nelems, _, nlocal = trace_shape(trace)
    nodes = trace.element_nodes()
    owner = [int(nodes[e, a]) for e in range(nelems) for a in range(nlocal)]
    return trace.bkd().asarray(
        [[1.0 if o == n else 0.0 for o in owner] for n in range(trace.nnodes())]
    )


@runtime_checkable
class PointwiseFluxModelProtocol(Protocol[Array]):
    """What :class:`PointwiseFluxResponse` asks of a model: a response
    whose points' ``output`` is the flux at the trace's quadrature points
    (shape (k * npoints, 1), as :func:`flux_moments` reads it) rather than
    a tested flux, and whose ``jvp_trace``, if any, differentiates those
    values."""

    def bkd(self) -> Backend[Array]: ...

    def nport(self) -> int: ...

    def receives(self) -> PortQuantity: ...

    def nparams(self) -> int: ...

    def nqoi(self) -> int: ...

    def linearize(
        self, port_data: Array, params: Array, window: TimeWindow
    ) -> ResponsePointProtocol[Array]: ...


class PointwiseFluxResponse(Generic[Array]):
    """A response returning the tested flux of a model that computes it
    pointwise.

    Satisfies
    :class:`~pyapprox.pde.coupling.responses.LocalResponseProtocol`,
    returning :class:`~pyapprox.pde.coupling.responses.Flux`; everything
    else is the model's. The port data are the model's, one row per trace
    function and component, so the trace's nodes are the port's.

    Parameters
    ----------
    model
        The pointwise-flux model.
    trace
        The port's trace functions and quadrature; the model's flux is
        given at its points.
    ncomponents
        :math:`k`, the components of the flux.

    Raises
    ------
    TypeError
        If ``model`` or ``trace`` does not satisfy its protocol.
    ValueError
        If the model's port size is not ``k * trace.nnodes()``.
    """

    def __init__(
        self,
        model: PointwiseFluxModelProtocol[Array],
        trace: PortTraceProtocol[Array],
        ncomponents: int = 1,
    ) -> None:
        if not isinstance(model, PointwiseFluxModelProtocol):
            raise TypeError(
                "model must satisfy PointwiseFluxModelProtocol, got "
                f"{type(model).__name__}"
            )
        trace_shape(trace)
        if model.nport() != ncomponents * trace.nnodes():
            raise ValueError(
                f"the model has {model.nport()} port rows, the trace "
                f"{ncomponents} x {trace.nnodes()}"
            )
        self._model, self._trace, self._k = model, trace, ncomponents

    def bkd(self) -> Backend[Array]:
        return self._model.bkd()

    def nport(self) -> int:
        return self._model.nport()

    def receives(self) -> PortQuantity:
        return self._model.receives()

    def returns(self) -> Flux:
        return Flux()

    def nparams(self) -> int:
        return self._model.nparams()

    def nqoi(self) -> int:
        return self._model.nqoi()

    def linearize(
        self, port_data: Array, params: Array, window: TimeWindow
    ) -> "PointwiseFluxPoint[Array]":
        return PointwiseFluxPoint(
            self._model.linearize(port_data, params, window), self._trace, self._k
        )


class PointwiseFluxPoint(Generic[Array]):
    """A :class:`PointwiseFluxResponse` at one linearization: the model's
    point with its flux, and its trace derivatives if it has them,
    tested."""

    def __init__(
        self,
        point: ResponsePointProtocol[Array],
        trace: PortTraceProtocol[Array],
        ncomponents: int,
    ) -> None:
        self._point, self._trace, self._k = point, trace, ncomponents
        self._flux = flux_moments(trace, point.output(), ncomponents)
        jvp = point.derivatives().jvp_trace
        self._derivs: ResponseDerivatives[Array] = (
            ResponseDerivatives.none()
            if jvp is None
            else ResponseDerivatives.tangent(
                lambda directions: flux_moments(trace, jvp(directions), ncomponents)
            )
        )

    def output(self) -> Array:
        """The tested flux. Shape: (k * nnodes, 1)."""
        return self._flux

    def qoi(self) -> Array:
        return self._point.qoi()

    def derivatives(self) -> ResponseDerivatives[Array]:
        return self._derivs
