r"""A port flux computed pointwise from the state, as collocation does.

In a weak (Galerkin) discretization, integration by parts leaves a
boundary term on every row, so the residual left on a pinned row at a
solution, the constraint reaction, is the boundary flux integrated
against that row's basis function. A discretization whose rows are the
PDE evaluated pointwise (spectral collocation, any physics) integrates
nothing by parts: what remains on a pinned row is the PDE's pointwise
error at that node, not a flux. Its port flux is instead the normal flux
:math:`q = \mathbf{F}(u) \cdot n`, evaluated at the operators' boundary
nodes, sampled at the trace's quadrature points (selected, or
interpolated from the nodes), and integrated against
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
from pyapprox.util.backends.protocols import Array, Backend


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
    """One boundary's contribution to a port: ``operators`` give each
    flux component at the boundary's nodes (one operator for a scalar
    flux, e.g. one per traction component), and ``sampling`` maps those
    nodes to the trace quadrature points the part supplies, in the
    trace's point order. Shape of ``sampling``: (npoints, nboundary).

    Rows of the identity select nodes that are quadrature points
    (:func:`selecting`); interpolation rows sample the flux's nodal
    interpolant elsewhere."""

    operators: Sequence[NormalFluxOperatorProtocol[Array]]
    sampling: Array


def selecting(positions: Sequence[int], nboundary: int, bkd: Backend[Array]) -> Array:
    """The sampling matrix of a :class:`NormalFluxPart` whose quadrature
    points are the boundary nodes ``positions``, of ``nboundary``. Shape:
    (len(positions), nboundary)."""
    return bkd.eye(nboundary)[bkd.asarray(list(positions), dtype=bkd.int64_dtype())]


class NormalFluxMoments(Generic[Array]):
    r"""The outward normal flux of a state, tested against the port's
    trace functions.

    Satisfies :class:`~pyapprox.pde.coupling.pinned_response.
    PortFluxProtocol` with backend columns as states. The parts' sampled
    points, concatenated in order, are the trace's quadrature points: for a
    :class:`~pyapprox.pde.coupling.traces.LumpedNodalTrace`, the port
    nodes. A flux of :math:`k` components (every part's operator count)
    is tested componentwise, the output node-major (row :math:`k a + c`).

    Parameters
    ----------
    parts
        The operators and their sampling, each with the same number of
        operators.
    trace
        The port's trace functions and quadrature.
    time
        The time the operators' coefficients are evaluated at.

    Raises
    ------
    TypeError
        If an operator or ``trace`` does not satisfy its protocol.
    ValueError
        If there are no parts, a part has no operator or another count
        than the first, a sampling matrix is not 2D, or the parts sample a
        different number of points than the trace has quadrature points.
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
        k = len(parts[0].operators)
        if k == 0:
            raise ValueError("a part needs at least one operator")
        for part in parts:
            if len(part.operators) != k:
                raise ValueError(
                    f"every part needs {k} operators, got {len(part.operators)}"
                )
            for operator in part.operators:
                if not isinstance(operator, NormalFluxOperatorProtocol):
                    raise TypeError(
                        "operators must satisfy NormalFluxOperatorProtocol, got "
                        f"{type(operator).__name__}"
                    )
            if part.sampling.ndim != 2:
                raise ValueError(
                    "a sampling matrix must have shape (npoints, nboundary), "
                    f"got {tuple(part.sampling.shape)}"
                )
        nsampled = sum(int(part.sampling.shape[0]) for part in parts)
        if nsampled != nelems * nquad:
            raise ValueError(
                f"the parts sample {nsampled} points, the trace has "
                f"{nelems * nquad} quadrature points"
            )
        self._parts = [(list(part.operators), part.sampling) for part in parts]
        self._k = k
        self._trace, self._time, self._bkd = trace, time, trace.bkd()

    def flux(self, state: Array) -> Array:
        """The tested outward flux at ``state``, shape (nstates, 1).
        Shape: (k * nport, 1)."""
        bkd = self._bkd
        # Per part (npoints, k), flattened point-major: row k p + c.
        values = bkd.concatenate(
            [
                bkd.reshape(
                    bkd.stack(
                        [sampling @ op(state[:, 0], self._time) for op in ops], axis=1
                    ),
                    (-1,),
                )
                for ops, sampling in self._parts
            ],
            axis=0,
        )
        return flux_moments(self._trace, bkd.reshape(values, (-1, 1)), self._k)

    def flux_tangent(
        self, state: Array, delta: Array, point: SteadyPoint[Array, Array]
    ) -> Array:
        """The tested flux's change for ``delta``, shape (nstates, ncols):
        the operators' Jacobian at ``state`` applied to it. ``point`` is not
        read: the operators carry no linearization label. Shape: (k *
        nport, ncols)."""
        bkd = self._bkd
        blocks = []
        for ops, sampling in self._parts:
            # (npoints, k, nstates), flattened point-major.
            stacked = bkd.stack(
                [sampling @ op.jacobian(state[:, 0], self._time) for op in ops],
                axis=1,
            )
            blocks.append(bkd.reshape(stacked, (-1, int(stacked.shape[2]))))
        jacobian = bkd.concatenate(blocks, axis=0)
        return flux_moments(self._trace, jacobian @ delta, self._k)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nparts={len(self._parts)}, k={self._k})"
