r"""A subdomain response whose port data enter as a load on its port rows.

A pinned response (:mod:`~pyapprox.pde.coupling.pinned_response`)
receives a trace and pins its port rows to it. A port-loaded response
receives data it applies as a natural load instead, e.g. a tested flux
(a Neumann-to-Dirichlet response) or a traction: with the port data
:math:`\lambda` as the residual's parameters,

.. math::

    R(u; \lambda) = F(u) + B\, T \lambda,

for :math:`F` the steady residual in right-hand-side form with its
essential rows pinned, :math:`B` the placement of a port block on the
port rows and :math:`T` an injected map of port blocks: the identity when
the data are already tested (a flux tested against the port's trace
functions), a trace mass when they are nodal values to be tested. The
port rows stay free rows, so :math:`R_u = F_u` and :math:`R_\lambda = B
T`; the trace tangent is the pinned response's, :math:`R_u\, \delta u =
-B T d`, then the output's tangent (:class:`~pyapprox.pde.coupling.
pinned_response.PortTangent`).

In right-hand-side form a Galerkin residual is :math:`F = f - K u` and
natural data :math:`g` enter as :math:`+\int g\, \psi`, so a tested
outward flux in Neumann form adds with sign +.

The output is read from the solved state by a
:class:`~pyapprox.pde.coupling.pinned_response.PortFluxProtocol`, e.g.
:class:`PortRowsOutput`: the state on the port rows, optionally mapped
(tested) by a port-block map.

The response declares the kinds it receives and returns, so the same
class serves a flux-to-trace response and one between physical
quantities of another kind (an interface pressure in, a tested normal
velocity out).
"""

from typing import Generic, Optional, TypeVar

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.residual_derivatives import (
    DerivativeOperator,
    ResidualDerivatives,
)
from pyapprox.implicitfunction.setups import SetupStrategyProtocol, SteadyRequest
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.pde.coupling.pinned_ports import PortPinnedData
from pyapprox.pde.coupling.pinned_response import (
    PortFluxProtocol,
    PortTangent,
    SteadyPinnedPoint,
)
from pyapprox.pde.coupling.ports import PortMapProtocol
from pyapprox.pde.coupling.responses import Flux, PortQuantity, TimeWindow, Trace
from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    InnerProductSpaceProtocol,
    MatrixFree,
    OperatorProperties,
    SpaceDescriptor,
)
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)


class PortLoad(Generic[V, A]):
    r""":math:`B T`: the port data, held as a parameter vector, mapped by
    :math:`T` and placed on the port rows of a zero state; ``MatrixFree``.
    The transpose reads a state's port rows, maps them by :math:`T^\top`
    and places them on the port data's rows. ``out`` is ignored.

    Parameters
    ----------
    ports : PortPinnedData
        The port rows and the port data's parameter rows.
    load : PortMapProtocol, optional
        :math:`T`, square on port blocks; omitted, the identity.
    """

    def __init__(
        self, ports: PortPinnedData[V, A], load: Optional[PortMapProtocol[A]] = None
    ) -> None:
        self._ports, self._load = ports, load

    def domain(self) -> SpaceDescriptor:
        return self._ports.param_space().descriptor()

    def range(self) -> SpaceDescriptor:
        return self._ports.state_space().descriptor()

    def properties(self) -> OperatorProperties:
        return OperatorProperties()

    def format(self) -> MatrixFree:
        return MatrixFree()

    def apply(self, x: V, out: Optional[V] = None) -> V:
        ports = self._ports
        block = ports.param_ports().take(x)
        if self._load is not None:
            block = self._load.apply(block)
        zeros = ports.state_space().zeros(ports.param_space().ncols(x))
        return ports.state_ports().add_to(zeros, block)

    def apply_transpose(self, y: V, out: Optional[V] = None) -> V:
        ports = self._ports
        block = ports.state_ports().take(y)
        if self._load is not None:
            block = self._load.apply_transpose(block)
        zeros = ports.param_space().zeros(ports.state_space().ncols(y))
        return ports.param_ports().add_to(zeros, block)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.domain()!r} -> {self.range()!r})"


class _Constant(Generic[V]):
    """A parameter Jacobian independent of the state."""

    def __init__(self, op: DerivativeOperator[V]) -> None:
        self._op = op

    def __call__(self, state: V) -> DerivativeOperator[V]:
        return self._op


class PortLoadResidual(Generic[V, F]):
    r""":math:`R(u; \lambda) = F(u) + B T \lambda` (see the module
    docstring): ``pinned`` with the port data added as a load on its port
    rows, the port data its parameters.

    Satisfies :class:`~pyapprox.implicitfunction.residual.
    ResidualProtocol`, first order. Its parameter version counts both
    :math:`F`'s and the port data's changes.

    Parameters
    ----------
    pinned : PinnedResidual
        :math:`F` with its essential rows pinned, without parameters of its
        own.
    ports : PortPinnedData
        The port rows and data, on ``pinned``'s state space.
    load : PortMapProtocol, optional
        :math:`T`; omitted, the identity.

    Raises
    ------
    TypeError
        If ``pinned`` is not a ``PinnedResidual``.
    ValueError
        If a port row is also an essential row (the pin would overwrite the
        load), the port rows belong to another space, or ``pinned`` has no
        state Jacobian or has parameters of its own.
    """

    def __init__(
        self,
        pinned: PinnedResidual[V, F],
        ports: PortPinnedData[V, A],
        load: Optional[PortMapProtocol[A]] = None,
    ) -> None:
        if not isinstance(pinned, PinnedResidual):
            raise TypeError(
                f"pinned must be a PinnedResidual, got {type(pinned).__name__}"
            )
        pinned.space().descriptor().require_same(
            ports.rows().descriptor(), "the port rows"
        )
        free = ports.rows().difference(pinned.pin().rows())
        if free.count() != ports.nport():
            raise ValueError(
                f"{ports.nport() - free.count()} port rows are also essential "
                "rows; a load on them would be overwritten"
            )
        bundle = pinned.derivatives()
        if bundle.jacobian is None:
            raise ValueError("a port-loaded residual needs the state Jacobian")
        if bundle.param is not None:
            raise ValueError(
                "a port-loaded residual does not yet take parameters of its own"
            )
        self._pinned, self._ports = pinned, ports
        self._load = PortLoad(ports, load)
        self._derivatives: ResidualDerivatives[V, F] = ResidualDerivatives.first_order(
            ports.param_space(), bundle.jacobian, _Constant(self._load)
        )

    def __call__(self, state: V) -> V:
        space = self._ports.state_space()
        return space.axpby(
            1.0, self._pinned(state), 1.0, self._load.apply(self._ports.port_data())
        )

    def derivatives(self) -> ResidualDerivatives[V, F]:
        return self._derivatives

    def parameters_version(self) -> int:
        return self._pinned.parameters_version() + self._ports.version()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._pinned!r}, {self._ports!r})"


class PortRowsOutput(Generic[V, A]):
    r"""The state on the port rows as a response's output, mapped by a
    port-block map when given: :math:`T R u` and its tangent :math:`T R\,
    \delta u`. The trace for :math:`T = I`; a tested trace (moments
    against the port's trace functions) for :math:`T` a trace mass.

    Satisfies :class:`~pyapprox.pde.coupling.pinned_response.
    PortFluxProtocol`.

    Parameters
    ----------
    ports : PortPinnedData
        The port rows.
    output : PortMapProtocol, optional
        :math:`T`; omitted, the identity.
    """

    def __init__(
        self, ports: PortPinnedData[V, A], output: Optional[PortMapProtocol[A]] = None
    ) -> None:
        self._ports, self._output = ports, output

    def _mapped(self, block: A) -> A:
        return block if self._output is None else self._output.apply(block)

    def flux(self, state: V) -> A:
        return self._mapped(self._ports.state_ports().take(state))

    def flux_tangent(self, state: V, delta: V, point: SteadyPoint[V, A]) -> A:
        return self._mapped(self._ports.state_ports().take(delta))


class SteadyLoadedResponse(Generic[V, A, F]):
    r"""A subdomain whose port data enter as a load on its port rows (see
    the module docstring), returning what ``output`` reads from its
    solved state. Steady, without quantities of interest or parameters of
    its own.

    Satisfies :class:`~pyapprox.pde.coupling.responses.
    LocalResponseProtocol`. Each solve starts from the last state solved.

    Parameters
    ----------
    pinned : PinnedResidual
        The subdomain's steady residual in right-hand-side form, with its
        essential rows pinned and its port rows not.
    ports : PortPinnedData
        Its port rows; port blocks are backend arrays.
    output : PortFluxProtocol
        Reads the output from the solved state, e.g.
        :class:`PortRowsOutput`.
    space : InnerProductSpaceProtocol
        The state space.
    setups : SetupStrategyProtocol
        Sets up the Jacobian for its solves.
    nonlinear : NonlinearSolverProtocol
        Solves the subdomain problem.
    bkd : Backend
        The backend of the port blocks.
    load : PortMapProtocol, optional
        :math:`T`, mapping the port data to the load on the port rows;
        omitted, the identity (the data are tested already).
    receives, returns : PortQuantity
        The kinds of the port data and the output; a flux in and a trace
        out by default.

    Raises
    ------
    TypeError
        If ``output`` does not satisfy ``PortFluxProtocol``.
    ValueError
        As :class:`PortLoadResidual`.
    """

    def __init__(
        self,
        pinned: PinnedResidual[V, F],
        ports: PortPinnedData[V, A],
        output: PortFluxProtocol[V, A],
        space: InnerProductSpaceProtocol[V, A],
        setups: SetupStrategyProtocol[SteadyRequest[V, F], V, F],
        nonlinear: NonlinearSolverProtocol[V, A],
        bkd: Backend[A],
        load: Optional[PortMapProtocol[A]] = None,
        receives: PortQuantity = Flux(),
        returns: PortQuantity = Trace(),
    ) -> None:
        if not isinstance(output, PortFluxProtocol):
            raise TypeError(
                f"output must satisfy PortFluxProtocol, got {type(output).__name__}"
            )
        self._ports, self._output = ports, output
        self._residual = PortLoadResidual(pinned, ports, load)
        param = self._residual.derivatives().param
        if param is None:
            raise ValueError("a port-loaded residual has a parameter Jacobian")
        self._system = SteadySystem(self._residual, space, setups)
        self._tangent = PortTangent(param, ports, output)
        self._nonlinear, self._bkd = nonlinear, bkd
        self._receives, self._returns = receives, returns
        self._guess = space.zeros(1)

    def bkd(self) -> Backend[A]:
        return self._bkd

    def nport(self) -> int:
        return self._ports.nport()

    def receives(self) -> PortQuantity:
        return self._receives

    def returns(self) -> PortQuantity:
        return self._returns

    def nparams(self) -> int:
        return 0

    def nqoi(self) -> int:
        return 0

    def residual(self) -> PortLoadResidual[V, F]:
        """The port-loaded residual the response solves."""
        return self._residual

    def linearize(
        self, port_data: A, params: A, window: TimeWindow
    ) -> SteadyPinnedPoint[V, A]:
        """Solve with ``port_data`` (nport, 1); ``params`` has shape (0, 1).

        Raises
        ------
        ValueError
            If a shape is wrong, the window is not steady, or the solve
            does not converge.
        """
        for name, block, rows in (
            ("port_data", port_data, self.nport()),
            ("params", params, 0),
        ):
            if tuple(block.shape) != (rows, 1):
                raise ValueError(
                    f"{name} must have shape ({rows}, 1), got {tuple(block.shape)}"
                )
        if not window.is_steady():
            raise ValueError("SteadyLoadedResponse is steady: the window must be too")
        self._ports.set_port_data(port_data)
        point = self._system.solve(self._nonlinear, self._guess)
        if not point.converged():
            raise ValueError(
                f"the subdomain solve did not converge ({point.info().message})"
            )
        self._guess = point.state()
        output = self._output.flux(point.state())
        return SteadyPinnedPoint(point, output, self._bkd, self._tangent)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nport={self.nport()})"
