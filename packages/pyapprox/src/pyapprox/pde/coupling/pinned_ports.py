r"""A subdomain's port rows pinned to its port data, the data as parameters.

A Dirichlet-to-Neumann response pins the subdomain state's port rows to
its port data :math:`\lambda_i`, beside its physical essential rows. The
port data are declared as pinned data (:class:`PortPinnedData`, a
:class:`~pyapprox.implicitfunction.pinned_rows.PinnedValuesProtocol`)
whose parameters are the port data themselves: :math:`g = B \lambda`, for
:math:`B` the placement of a port block on the port rows
(:class:`PortPlacement`). So :math:`\partial g/\partial \lambda = B` and
the second derivatives are stated zero.

:func:`pin_ports` pins the port rows on top of a residual whose physical
essential rows are already pinned. The two row sets are disjoint, so the
nested pin is the pin of their union: its rows read :math:`u - g` on
both, and its Jacobian is :math:`F_u` with both sets' rows identity rows.
The residual below gets the port data as its parameters
(:class:`PortParameterResidual`, with :math:`F_\lambda = 0`), so the
pinned residual's parameter Jacobian is :math:`-B` on the port rows and a
steady solve's tangent and adjoint in :math:`\lambda` follow from the
implicit-function machinery unchanged.

Port blocks are of the stack's port type ``P``; states and the port data
held as parameters are vectors ``V`` of their spaces, reached only
through :class:`~pyapprox.pde.coupling.ports.PortRowsProtocol`.
"""

from typing import Generic, Optional, TypeVar

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.pinned_rows import (
    PinnedValuesParamDerivatives,
    RowPin,
)
from pyapprox.implicitfunction.residual_derivatives import (
    STATED_ZERO,
    DerivativeOperator,
    ResidualDerivatives,
)
from pyapprox.pde.coupling.ports import PortRowsProtocol
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    MatrixFree,
    OperatorProperties,
    RowSetProtocol,
    SpaceDescriptor,
    VectorSpaceProtocol,
    ZeroOperator,
)

V = TypeVar("V")
P = TypeVar("P")
F = TypeVar("F", bound=FormatProtocol)


class PortPlacement(Generic[V, P]):
    r""":math:`B`: the port data, held as a parameter vector, placed on
    the port rows of a zero state; ``MatrixFree``. :math:`B^\top` reads a
    state's port rows back into a parameter vector. ``out`` is ignored.

    Parameters
    ----------
    state_ports : PortRowsProtocol
        The port rows of the state space, in port order.
    param_ports : PortRowsProtocol
        Every row of the parameter space, in port order.
    states, params : VectorSpaceProtocol
        The state and parameter spaces.
    """

    def __init__(
        self,
        state_ports: PortRowsProtocol[V, P],
        param_ports: PortRowsProtocol[V, P],
        states: VectorSpaceProtocol[V],
        params: VectorSpaceProtocol[V],
    ) -> None:
        self._state_ports, self._param_ports = state_ports, param_ports
        self._states, self._params = states, params

    def domain(self) -> SpaceDescriptor:
        return self._params.descriptor()

    def range(self) -> SpaceDescriptor:
        return self._states.descriptor()

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        zeros = self._states.zeros(self._params.ncols(x))
        return self._state_ports.add_to(zeros, self._param_ports.take(x))

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        zeros = self._params.zeros(self._states.ncols(y))
        return self._param_ports.add_to(zeros, self._state_ports.take(y))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.domain()!r} -> {self.range()!r})"


class _Placement(Generic[V, P]):
    r""":math:`\partial g/\partial \lambda = B`, the same at every port
    data."""

    def __init__(self, placement: PortPlacement[V, P]) -> None:
        self._placement = placement

    def __call__(self) -> DerivativeOperator[V]:
        return self._placement


class PortPinnedData(Generic[V, P]):
    r"""The port rows of a subdomain state and their data :math:`g =
    B \lambda`, the port data :math:`\lambda` being the parameters.

    Satisfies :class:`~pyapprox.implicitfunction.pinned_rows.
    PinnedValuesProtocol`. Holds the port data, zero until
    :meth:`set_port_data`; every change increments :meth:`version`.

    Parameters
    ----------
    rows : RowSetProtocol
        The port rows of the state space, as the stack's row set.
    state_ports : PortRowsProtocol
        The same rows, in port order, as the stack's port rows.
    param_ports : PortRowsProtocol
        Every row of ``params``, in port order.
    states, params : VectorSpaceProtocol
        The state space and the port data's space, of ``nport`` rows.

    Raises
    ------
    TypeError
        If a port rows argument does not satisfy ``PortRowsProtocol``.
    ValueError
        If ``rows`` belongs to another space than ``states``, or the row
        counts of ``rows``, ``state_ports``, ``param_ports`` and
        ``params`` differ.
    """

    def __init__(
        self,
        rows: RowSetProtocol[V],
        state_ports: PortRowsProtocol[V, P],
        param_ports: PortRowsProtocol[V, P],
        states: VectorSpaceProtocol[V],
        params: VectorSpaceProtocol[V],
    ) -> None:
        for name, ports in (("state_ports", state_ports), ("param_ports", param_ports)):
            if not isinstance(ports, PortRowsProtocol):
                raise TypeError(
                    f"{name} must satisfy PortRowsProtocol, got {type(ports).__name__}"
                )
        states.descriptor().require_same(rows.descriptor(), "the port rows")
        counts = (
            rows.count(),
            state_ports.nrows(),
            param_ports.nrows(),
            params.descriptor().size,
        )
        if len(set(counts)) != 1:
            raise ValueError(
                "rows, state_ports, param_ports and params must have one row "
                f"count, got {counts}"
            )
        self._rows, self._param_ports = rows, param_ports
        self._states, self._params = states, params
        self._placement = PortPlacement(state_ports, param_ports, states, params)
        self._bundle: PinnedValuesParamDerivatives[V] = (
            PinnedValuesParamDerivatives.second_order(
                _Placement(self._placement), STATED_ZERO
            )
        )
        self._lam = params.zeros(1)
        self._version = 0

    def nport(self) -> int:
        """The number of port rows."""
        return self._rows.count()

    def set_port_data(self, block: P) -> None:
        r"""Set :math:`\lambda` from a port block of shape (nport, 1)."""
        self._lam = self._param_ports.add_to(self._params.zeros(1), block)
        self._version += 1

    def port_data(self) -> V:
        r""":math:`\lambda`, a single vector of the parameter space."""
        return self._lam

    def version(self) -> int:
        """The number of port data changes so far."""
        return self._version

    def param_space(self) -> VectorSpaceProtocol[V]:
        """The port data's space."""
        return self._params

    def state_space(self) -> VectorSpaceProtocol[V]:
        """The state space."""
        return self._states

    def placement(self) -> PortPlacement[V, P]:
        """:math:`B`."""
        return self._placement

    def rows(self) -> RowSetProtocol[V]:
        return self._rows

    def values(self) -> V:
        r""":math:`B \lambda`: the port data on the port rows, zero
        elsewhere."""
        return self._placement.apply(self._lam)

    def parameter_derivatives(self) -> PinnedValuesParamDerivatives[V]:
        return self._bundle

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nport={self.nport()})"


class _ZeroParam(Generic[V]):
    def __init__(self, op: ZeroOperator[V]) -> None:
        self._op = op

    def __call__(self, state: V) -> DerivativeOperator[V]:
        return self._op


class PortParameterResidual(Generic[V, F]):
    r""":math:`F(u)`, its essential rows pinned, with the port data as its
    parameters, on which it does not depend: :math:`F_\lambda = 0`.

    Satisfies :class:`~pyapprox.implicitfunction.residual.
    ResidualProtocol`. Its parameter version counts both :math:`F`'s and
    the port data's changes. Pinning the port rows over it is the pin of
    the union of the essential and port rows only because the two sets
    are disjoint, which construction checks.

    Parameters
    ----------
    residual : PinnedResidual
        :math:`F` with its essential rows pinned, without parameters of
        its own.
    ports : PortPinnedData
        The port data.

    Raises
    ------
    TypeError
        If ``residual`` is not a ``PinnedResidual``.
    ValueError
        If a port row is also an essential row, the port rows belong to
        another space, or ``residual`` has parameter derivatives: its own
        parameters and the port data would need one combined parameter
        space.
    """

    def __init__(
        self, residual: PinnedResidual[V, F], ports: PortPinnedData[V, P]
    ) -> None:
        if not isinstance(residual, PinnedResidual):
            raise TypeError(
                f"residual must be a PinnedResidual, got {type(residual).__name__}"
            )
        residual.space().descriptor().require_same(
            ports.rows().descriptor(), "the port rows"
        )
        free = ports.rows().difference(residual.pin().rows())
        if free.count() != ports.nport():
            raise ValueError(
                f"{ports.nport() - free.count()} port rows are also essential "
                "rows; a port must not carry an essential condition"
            )
        bundle = residual.derivatives()
        if bundle.param is not None:
            raise ValueError(
                f"{type(residual).__name__} has parameters of its own; the port "
                "data are the only parameters supported"
            )
        self._residual, self._ports = residual, ports
        params = ports.param_space()
        self._derivatives: ResidualDerivatives[V, F] = (
            ResidualDerivatives.none()
            if bundle.jacobian is None
            else ResidualDerivatives.first_order(
                params,
                bundle.jacobian,
                _ZeroParam(ZeroOperator(params, ports.state_space())),
            )
        )

    def __call__(self, state: V) -> V:
        return self._residual(state)

    def derivatives(self) -> ResidualDerivatives[V, F]:
        return self._derivatives

    def parameters_version(self) -> int:
        return self._residual.parameters_version() + self._ports.version()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._residual!r}, {self._ports!r})"


def pin_ports(
    pinned: PinnedResidual[V, F], ports: PortPinnedData[V, P]
) -> PinnedResidual[V, F]:
    r"""``pinned`` with the port rows also pinned to the port data, which
    become its parameters (see the module docstring).

    Parameters
    ----------
    pinned : PinnedResidual
        :math:`F` with its physical essential rows pinned; no parameters.
    ports : PortPinnedData
        The port rows and data, on ``pinned``'s state space.

    Raises
    ------
    ValueError
        If a port row is also a pinned row of ``pinned``, the spaces
        differ, or ``pinned`` has parameters of its own.
    """
    return PinnedResidual(
        PortParameterResidual(pinned, ports), RowPin(ports), pinned.space()
    )
