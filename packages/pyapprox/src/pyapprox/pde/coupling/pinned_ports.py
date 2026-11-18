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
The residual below gets the parameters :math:`\theta = [\lambda; p]`
(:class:`PortParameterResidual`): the port data, with :math:`F_\lambda =
0`, beside its own parameters :math:`p`, if any. One vector, because a
residual's derivatives carry one parameter space; which rows are which
is fixed by the caller, through the port data's rows and the placement
of :math:`p`. The pinned residual's parameter Jacobian is then
:math:`[-B,\, P F_p]`, and a steady solve's tangent and adjoint in both
the port data and the coefficients follow from the implicit-function
machinery unchanged.

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
    ResidualParamFn,
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
from pyapprox.util.linalg.operators.algebra import Product, Transposed

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

    The parameter space may be larger than the port data: with a
    residual's own parameters :math:`p` it is the space of
    :math:`\theta = [\lambda; p]`, ``param_ports`` selecting the rows of
    :math:`\lambda`, and :math:`\partial g/\partial \theta = B` reads
    those rows only.

    Parameters
    ----------
    rows : RowSetProtocol
        The port rows of the state space, as the stack's row set.
    state_ports : PortRowsProtocol
        The same rows, in port order, as the stack's port rows.
    param_ports : PortRowsProtocol
        The rows of ``params`` holding the port data, in port order.
    states, params : VectorSpaceProtocol
        The state space and the parameter space.

    Raises
    ------
    TypeError
        If a port rows argument does not satisfy ``PortRowsProtocol``.
    ValueError
        If ``rows`` belongs to another space than ``states``, the row
        counts of ``rows``, ``state_ports`` and ``param_ports`` differ,
        or ``params`` has fewer rows.
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
        counts = (rows.count(), state_ports.nrows(), param_ports.nrows())
        if len(set(counts)) != 1 or params.descriptor().size < counts[0]:
            raise ValueError(
                "rows, state_ports and param_ports must have one row count, "
                f"at most params' size {params.descriptor().size}; got {counts}"
            )
        self._rows, self._param_ports = rows, param_ports
        self._state_ports = state_ports
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
        r""":math:`\lambda` on its rows of the parameter space, zero on the
        others: their values are held by whoever owns them."""
        return self._lam

    def version(self) -> int:
        """The number of port data changes so far."""
        return self._version

    def param_space(self) -> VectorSpaceProtocol[V]:
        """The parameter space."""
        return self._params

    def state_space(self) -> VectorSpaceProtocol[V]:
        """The state space."""
        return self._states

    def state_ports(self) -> PortRowsProtocol[V, P]:
        """The port rows of the state space, in port order."""
        return self._state_ports

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


class _Embedded(Generic[V, P]):
    r""":math:`F_\theta = F_p E_p^\top`: the residual's own parameter
    Jacobian read through the rows of :math:`p` in :math:`\theta`."""

    def __init__(self, fn: ResidualParamFn[V], physical: PortPlacement[V, P]) -> None:
        self._fn, self._select = fn, Transposed(physical)

    def __call__(self, state: V) -> DerivativeOperator[V]:
        return Product([self._fn(state), self._select])


class PortParameterResidual(Generic[V, F]):
    r""":math:`F(u; p)`, its essential rows pinned, with parameters
    :math:`\theta = [\lambda; p]`: the port data, on which it does not
    depend, and its own parameters, if any, so :math:`F_\theta = [0,\,
    F_p]`.

    Satisfies :class:`~pyapprox.implicitfunction.residual.
    ResidualProtocol`, first order: second derivatives in :math:`\theta`
    are not given. Its parameter version counts both :math:`F`'s and the
    port data's changes. Pinning the port rows over it is the pin of the
    union of the essential and port rows only because the two sets are
    disjoint, which construction checks.

    Parameters
    ----------
    residual : PinnedResidual
        :math:`F` with its essential rows pinned.
    ports : PortPinnedData
        The port data; its parameter space is that of :math:`\theta`.
    physical : PortPlacement, optional
        :math:`E_p`, placing :math:`F`'s parameters on their rows of
        :math:`\theta`, disjoint from the port data's rows; required
        exactly when :math:`F` has parameter derivatives. The values of
        :math:`p` stay with :math:`F`'s owner; only the derivatives are
        read through :math:`\theta`.

    Raises
    ------
    TypeError
        If ``residual`` is not a ``PinnedResidual``.
    ValueError
        If a port row is also an essential row, the port rows belong to
        another space, ``physical`` is given without parameter derivatives
        or missing with them, or its spaces are not :math:`F`'s parameter
        space and :math:`\theta`'s.
    """

    def __init__(
        self,
        residual: PinnedResidual[V, F],
        ports: PortPinnedData[V, P],
        physical: Optional[PortPlacement[V, P]] = None,
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
        self._residual, self._ports = residual, ports
        self._derivatives = self._theta_bundle(residual.derivatives(), physical)

    def _theta_bundle(
        self,
        bundle: ResidualDerivatives[V, F],
        physical: Optional[PortPlacement[V, P]],
    ) -> ResidualDerivatives[V, F]:
        own, own_space = bundle.param, bundle.param_space
        if (own is None) != (physical is None):
            raise ValueError(
                "physical places the residual's own parameters in theta: give "
                "it exactly when the residual has parameter derivatives"
            )
        params = self._ports.param_space()
        if bundle.jacobian is None:
            return ResidualDerivatives.none()
        if own is None or own_space is None or physical is None:
            zero = ZeroOperator(params, self._ports.state_space())
            return ResidualDerivatives.first_order(
                params, bundle.jacobian, _ZeroParam(zero)
            )
        physical.domain().require_same(
            own_space.descriptor(), "the residual's parameter space"
        )
        physical.range().require_same(params.descriptor(), "theta's space")
        return ResidualDerivatives.first_order(
            params, bundle.jacobian, _Embedded(own, physical)
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
    pinned: PinnedResidual[V, F],
    ports: PortPinnedData[V, P],
    physical: Optional[PortPlacement[V, P]] = None,
) -> PinnedResidual[V, F]:
    r"""``pinned`` with the port rows also pinned to the port data, its
    parameters becoming :math:`\theta = [\lambda; p]` (see the module
    docstring and :class:`PortParameterResidual`).

    Parameters
    ----------
    pinned : PinnedResidual
        :math:`F` with its physical essential rows pinned.
    ports : PortPinnedData
        The port rows and data, on ``pinned``'s state space.
    physical : PortPlacement, optional
        :math:`E_p`, required exactly when ``pinned`` has parameters.

    Raises
    ------
    ValueError
        As :class:`PortParameterResidual`.
    """
    return PinnedResidual(
        PortParameterResidual(pinned, ports, physical), RowPin(ports), pinned.space()
    )
