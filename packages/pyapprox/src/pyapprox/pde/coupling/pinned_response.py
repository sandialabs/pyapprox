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

The trace tangent, what a skeleton Newton step needs, is one tangent
solve at the subdomain's solution: the port data are the residual's
parameters, so :math:`R_u\, \delta u = -R_\theta [d; 0]` gives the
state's change, and the flux changes by :math:`-B^\top F_u\, \delta u`.
With a setup strategy keeping the Jacobian's setup at the solution, each
skeleton Krylov product costs one back-substitution per subdomain.
"""

from typing import Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.implicitfunction.pinned_residual import PinnedResidual
from pyapprox.implicitfunction.residual_derivatives import (
    ResidualJacobian,
    ResidualParamFn,
)
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
    LinearizationPoint,
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


class _TraceTangent(Generic[V, A, F]):
    r"""What a point's trace tangent reads: the port-pinned residual
    :math:`R`, the essentially pinned :math:`F` below it, and the port
    data."""

    def __init__(
        self,
        residual: PinnedResidual[V, F],
        pinned: PinnedResidual[V, F],
        ports: PortPinnedData[V, A],
    ) -> None:
        self.pinned, self.ports = pinned, ports
        param, jacobian = residual.derivatives().param, pinned.derivatives().jacobian
        if param is None or jacobian is None:
            raise ValueError(
                "a trace tangent needs the port-pinned residual's parameter "
                "Jacobian and the essentially pinned residual's Jacobian"
            )
        self.param: ResidualParamFn[V] = param
        self.jacobian: ResidualJacobian[V, F] = jacobian


class SteadyPinnedPoint(Generic[V, A, F]):
    r"""A :class:`SteadyPinnedResponse` solved at one set of port data and
    parameters.

    Its derivatives hold while the response is not linearized again: a
    later solve changes the port data, and a tangent requested after it
    is refused.
    """

    def __init__(
        self,
        point: SteadyPoint[V, A],
        output: A,
        bkd: Backend[A],
        tangent: _TraceTangent[V, A, F],
    ) -> None:
        self._point, self._output, self._bkd = point, output, bkd
        self._tangent = tangent
        self._derivs = ResponseDerivatives.tangent(self.jvp_trace)

    def output(self) -> A:
        """The outward flux on the ports. Shape: (nport, 1)."""
        return self._output

    def qoi(self) -> A:
        return self._bkd.zeros((0, 1))

    def derivatives(self) -> ResponseDerivatives[A]:
        return self._derivs

    def jvp_trace(self, directions: A) -> A:
        r""":math:`(\partial r/\partial \lambda)` ``directions``, for a
        block of shape (nport, ncols): one tangent solve :math:`R_u\,
        \delta u = -R_\theta [d; 0]` for all columns, then the flux
        tangent :math:`-B^\top F_u\, \delta u`.

        Raises
        ------
        ValueError
            If ``directions`` has the wrong shape, or the response has been
            linearized again since this point.
        """
        t = self._tangent
        nport = t.ports.nport()
        if directions.ndim != 2 or directions.shape[0] != nport:
            raise ValueError(
                f"directions must have shape ({nport}, ncols), got "
                f"{tuple(directions.shape)}"
            )
        state = self._point.solution()
        params = t.ports.param_space()
        # theta = [d; 0]: the directions on the port data's rows.
        theta = t.ports.param_ports().add_to(
            params.zeros(int(directions.shape[1])), directions
        )
        space = t.ports.state_space()
        rhs = space.scale(-1.0, t.param(state).apply(theta))
        delta, info = self._point.jacobian_handle().solve(rhs)
        if not info.converged:
            raise ValueError("the tangent solve did not converge")
        label = LinearizationPoint(t.pinned.parameters_version(), self._point)
        change = t.jacobian(state, label).apply(delta)
        return -t.ports.state_ports().take(change)

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
        self._tangent = _TraceTangent(self._residual, pinned, ports)
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
    ) -> SteadyPinnedPoint[V, A, F]:
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
        return SteadyPinnedPoint(point, -reaction, self._bkd, self._tangent)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nport={self.nport()})"
