r"""A subdomain response from a steady residual with pinned rows.

A mesh subdomain (Galerkin, or any discretization giving a steady
residual with its essential rows pinned) becomes a Dirichlet-to-Neumann
response: its port rows are pinned to the port data
(:func:`~pyapprox.pde.coupling.pinned_ports.pin_ports`), the steady
problem is solved, and the output is the reaction on the port rows.

The residual is in right-hand-side form, :math:`M \dot u = F(u)`, as the
Galerkin steady residual is: :math:`F = f - K u` for diffusion. On a
pinned row :math:`F` does not vanish at a solution; what remains is the
boundary term the pinning left out, :math:`-\kappa\, \partial u/\partial
n` tested against the row's basis function. The output is its negative,
the outward flux tested against the port basis, which is what
:class:`~pyapprox.pde.coupling.responses.Flux` holds. Summed over the
subdomains at an interface, it is the single-domain residual of the
interface rows, so a skeleton solve reproduces the single-domain
discrete solution and the flux balance holds to solver tolerance.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.setups import SetupStrategyProtocol, SteadyRequest
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.pde.coupling.pinned_ports import (
    PortPinnedData,
    PortPlacement,
    pin_ports,
)
from pyapprox.pde.coupling.responses import (
    Flux,
    ResponseDerivatives,
    TimeWindow,
    Trace,
)
from pyapprox.util.backends.protocols import ArrayProtocol, Backend
from pyapprox.util.linalg.operators import (
    FormatProtocol,
    InnerProductSpaceProtocol,
)
from pyapprox.util.rootfinding.nonlinear import NonlinearSolverProtocol

V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
F = TypeVar("F", bound=FormatProtocol)
A_contra = TypeVar("A_contra", bound=ArrayProtocol, contravariant=True)


@runtime_checkable
class ResponseParametersProtocol(Protocol[A_contra]):
    """What a response sets its subdomain's own parameters through."""

    def nparams(self) -> int:
        """The number of parameters."""
        ...

    def set_param(self, param: A_contra) -> None:
        """Apply ``param``, of shape (nparams,)."""
        ...


class SteadyPinnedPoint(Generic[V, A]):
    """A :class:`SteadyPinnedResponse` solved at one set of port data and
    parameters."""

    def __init__(self, point: SteadyPoint[V, A], output: A, bkd: Backend[A]) -> None:
        self._point, self._output, self._bkd = point, output, bkd

    def output(self) -> A:
        """The outward flux on the ports. Shape: (nport, 1)."""
        return self._output

    def qoi(self) -> A:
        return self._bkd.zeros((0, 1))

    def derivatives(self) -> ResponseDerivatives[A]:
        return ResponseDerivatives.none()

    def state(self) -> V:
        """The subdomain's solved state."""
        return self._point.state()

    def steady_point(self) -> SteadyPoint[V, A]:
        """The solve, for sensitivities at it."""
        return self._point


class SteadyPinnedResponse(Generic[V, A, F]):
    r"""A subdomain as a Dirichlet-to-Neumann response: receives the
    trace on its ports and returns the outward flux there (see the module
    docstring). Steady, without quantities of interest.

    Satisfies :class:`~pyapprox.pde.coupling.responses.
    LocalResponseProtocol`. Each solve starts from the last state solved.

    Parameters
    ----------
    pinned : PinnedResidual
        The subdomain's steady residual, in right-hand-side form, with its
        essential rows pinned.
    ports : PortPinnedData
        Its port rows; port blocks are backend arrays.
    space : InnerProductSpaceProtocol
        The state space.
    setups : SetupStrategyProtocol
        Sets up the Jacobian of the port-pinned residual for its solves.
    nonlinear : NonlinearSolverProtocol
        Solves the subdomain problem.
    bkd : Backend
        The backend of the port blocks and parameters.
    parameters : ResponseParametersProtocol, optional
        Sets the subdomain's own parameters; given exactly when
        ``physical`` is.
    physical : PortPlacement, optional
        Places those parameters in the residual's parameter vector beside
        the port data (:func:`~pyapprox.pde.coupling.pinned_ports.
        pin_ports`).

    Raises
    ------
    TypeError
        If ``parameters`` does not satisfy its protocol.
    ValueError
        If ``parameters`` and ``physical`` are not given together, or as
        :func:`~pyapprox.pde.coupling.pinned_ports.pin_ports`.
    """

    def __init__(
        self,
        pinned: PinnedResidual[V, F],
        ports: PortPinnedData[V, A],
        space: InnerProductSpaceProtocol[V, A],
        setups: SetupStrategyProtocol[SteadyRequest[V, F], V, F],
        nonlinear: NonlinearSolverProtocol[V, A],
        bkd: Backend[A],
        parameters: Optional[ResponseParametersProtocol[A]] = None,
        physical: Optional[PortPlacement[V, A]] = None,
    ) -> None:
        if (parameters is None) != (physical is None):
            raise ValueError("parameters and physical come together")
        if parameters is not None and not isinstance(
            parameters, ResponseParametersProtocol
        ):
            raise TypeError(
                "parameters must satisfy ResponseParametersProtocol, got "
                f"{type(parameters).__name__}"
            )
        self._pinned, self._ports = pinned, ports
        self._residual = pin_ports(pinned, ports, physical)
        self._system = SteadySystem(self._residual, space, setups)
        self._nonlinear, self._bkd = nonlinear, bkd
        self._parameters = parameters
        self._guess = space.zeros(1)

    def bkd(self) -> Backend[A]:
        return self._bkd

    def nport(self) -> int:
        return self._ports.nport()

    def receives(self) -> Trace:
        return Trace()

    def returns(self) -> Flux:
        return Flux()

    def nparams(self) -> int:
        return 0 if self._parameters is None else self._parameters.nparams()

    def nqoi(self) -> int:
        return 0

    def residual(self) -> PinnedResidual[V, F]:
        """The port-pinned residual the response solves."""
        return self._residual

    def linearize(
        self, port_data: A, params: A, window: TimeWindow
    ) -> SteadyPinnedPoint[V, A]:
        """Solve with ``port_data`` (nport, 1) and ``params`` (nparams,
        1).

        Raises
        ------
        ValueError
            If a shape is wrong, the window is not steady, or the solve
            does not converge.
        """
        for name, block, rows in (
            ("port_data", port_data, self.nport()),
            ("params", params, self.nparams()),
        ):
            if tuple(block.shape) != (rows, 1):
                raise ValueError(
                    f"{name} must have shape ({rows}, 1), got {tuple(block.shape)}"
                )
        if not window.is_steady():
            raise ValueError("SteadyPinnedResponse is steady: the window must be too")
        if self._parameters is not None:
            self._parameters.set_param(params[:, 0])
        self._ports.set_port_data(port_data)
        point = self._system.solve(self._nonlinear, self._guess)
        if not point.converged():
            raise ValueError(
                f"the subdomain solve did not converge ({point.info().message})"
            )
        self._guess = point.state()
        # The essentially pinned residual keeps F on the port rows.
        reaction = self._ports.state_ports().take(self._pinned(point.state()))
        return SteadyPinnedPoint(point, -reaction, self._bkd)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nport={self.nport()})"
