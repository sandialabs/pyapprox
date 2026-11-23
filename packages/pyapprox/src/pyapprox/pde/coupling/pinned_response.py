r"""A subdomain response from a steady residual with pinned rows.

A mesh subdomain (Galerkin, collocation, or any discretization giving a
steady residual with its essential rows pinned) becomes a
Dirichlet-to-Neumann response: its port rows are pinned to the port data
(:func:`~pyapprox.pde.coupling.pinned_ports.pin_ports`), the steady
problem is solved, and the output is the outward flux on the ports,
tested against the port's trace functions, as an injected
:class:`PortFluxProtocol` reads it from the solved state.

:class:`PinnedReactionFlux` reads the reaction. The residual is in
right-hand-side form, :math:`M \dot u = F(u)`, as the Galerkin steady
residual is: :math:`F = f - K u` for diffusion. On a pinned row :math:`F`
does not vanish at a solution; what remains is the boundary term the
pinning left out, :math:`-\kappa\, \partial u/\partial n` tested against
the row's basis function. The output is its negative, the outward flux
tested against the port basis, which is what
:class:`~pyapprox.pde.coupling.responses.Flux` holds. Summed over the
subdomains at an interface, it is the single-domain residual of the
interface rows, so a skeleton solve reproduces the single-domain
discrete solution and the flux balance holds to solver tolerance. A
discretization whose pinned rows are pointwise equations (collocation)
has no such reaction and injects a flux computed from the state instead.

The trace tangent, what a skeleton Newton step needs, is one tangent
solve at the subdomain's solution: the port data are the residual's
parameters, so :math:`R_u\, \delta u = -R_\theta [d; 0]` gives the
state's change, which the flux maps to its own change. With a setup
strategy keeping the Jacobian's setup at the solution, each skeleton
Krylov product costs one back-substitution per subdomain. Its transpose,
what adjoints need, is one transposed solve :math:`R_u^\top z = C^\top
w` at the same solution, :math:`C` the output's derivative in the state,
read back through :math:`-R_\theta^\top z` on the port data's rows.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

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
from pyapprox.pde.coupling.ports import PortRowsProtocol
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


FluxParamTangent = Callable[[V, V], A]
r"""``(state, v) ->`` the output's explicit change :math:`(\partial
r/\partial p)\, v` at fixed state, for a block ``v`` of the response's
own parameter space (ncols columns). Shape: (nport, ncols)."""

FluxParamAdjoint = Callable[[V, A], V]
r"""``(state, w) ->`` its transpose applied to weights ``w`` of shape
(nport, ncols): a block of the response's own parameter space."""


@dataclass(frozen=True)
class FluxParamDerivatives(Generic[V, A]):
    r"""How a port output depends on the response's own parameters
    :math:`p` at a fixed state.

    - ``reads_params``: whether the output reads :math:`p` directly; when
      it does not, the explicit derivative is zero and needs no fields.
    - ``tangent``, ``adjoint``: the explicit derivative and its transpose,
      together; absent when the output reads :math:`p` but cannot
      differentiate in it.

    Build with the named constructors.
    """

    reads_params: bool
    tangent: Optional[FluxParamTangent[V, A]] = None
    adjoint: Optional[FluxParamAdjoint[V, A]] = None

    def __post_init__(self) -> None:
        if (self.tangent is None) != (self.adjoint is None):
            raise ValueError("tangent and adjoint come together")
        if not self.reads_params and self.tangent is not None:
            raise ValueError("an output reading no parameters has no derivative")

    @staticmethod
    def independent() -> "FluxParamDerivatives[V, A]":
        """The output reads no parameters directly: a zero derivative."""
        return FluxParamDerivatives(False)

    @staticmethod
    def first_order(
        tangent: FluxParamTangent[V, A], adjoint: FluxParamAdjoint[V, A]
    ) -> "FluxParamDerivatives[V, A]":
        """The output reads the parameters, with its derivative."""
        return FluxParamDerivatives(True, tangent, adjoint)

    @staticmethod
    def unavailable() -> "FluxParamDerivatives[V, A]":
        """The output reads the parameters, without a derivative."""
        return FluxParamDerivatives(True)

    def available(self) -> bool:
        """Whether the explicit derivative is known (zero or given)."""
        return not self.reads_params or self.tangent is not None


@runtime_checkable
class PortFluxProtocol(Protocol[V, A]):
    """How a steady port response reads its output from a solved state,
    and its change with the state: for :class:`SteadyPinnedResponse` the
    outward flux on the ports, tested against the port's trace functions;
    for a port-loaded response (:mod:`~pyapprox.pde.coupling.
    loaded_response`) the trace, or a tested trace."""

    def flux(self, state: V) -> A:
        """The output at ``state``. Shape: (nport, 1)."""
        ...

    def flux_tangent(self, state: V, delta: V, point: SteadyPoint[V, A]) -> A:
        """The output's change for state changes ``delta`` (ncols columns)
        at the solve ``point`` of ``state``. Shape: (nport, ncols)."""
        ...

    def flux_adjoint(self, state: V, weights: A, point: SteadyPoint[V, A]) -> V:
        """The transpose of :meth:`flux_tangent`: the output's derivative
        in the state, transposed, applied to ``weights`` (nport, ncols).
        A state-space vector of ncols columns."""
        ...

    def param_derivatives(self) -> FluxParamDerivatives[V, A]:
        """How the output depends on the response's own parameters at a
        fixed state."""
        ...


class PinnedReactionFlux(Generic[V, A, F]):
    r"""The reaction on the port rows as the tested outward flux:
    :math:`-B^\top F(u)` and its tangent :math:`-B^\top F_u\, \delta u`
    (see the module docstring).

    Satisfies :class:`PortFluxProtocol`. Conservative for a
    discretization whose rows are tested equations (Galerkin, a nested
    skeleton); not for one whose pinned rows are pointwise equations.

    Parameters
    ----------
    pinned : PinnedResidual
        The subdomain's steady residual in right-hand-side form, with its
        essential rows pinned and its port rows not: :math:`F` keeps its
        value on them.
    ports : PortPinnedData
        Its port rows.

    Raises
    ------
    ValueError
        If ``pinned`` has no state Jacobian.
    """

    def __init__(
        self, pinned: PinnedResidual[V, F], ports: PortPinnedData[V, A]
    ) -> None:
        bundle = pinned.derivatives()
        if bundle.jacobian is None:
            raise ValueError("a reaction flux needs the residual's state Jacobian")
        self._pinned, self._ports = pinned, ports
        self._jacobian: ResidualJacobian[V, F] = bundle.jacobian
        self._param: Optional[ResidualParamFn[V]] = bundle.param
        self._param_derivs: FluxParamDerivatives[V, A] = (
            FluxParamDerivatives.independent()
            if bundle.param_space is None
            else FluxParamDerivatives.unavailable()
            if bundle.param is None
            else FluxParamDerivatives.first_order(
                self._param_tangent, self._param_adjoint
            )
        )

    def flux(self, state: V) -> A:
        return -self._ports.state_ports().take(self._pinned(state))

    def flux_tangent(self, state: V, delta: V, point: SteadyPoint[V, A]) -> A:
        label = LinearizationPoint(self._pinned.parameters_version(), point)
        change = self._jacobian(state, label).apply(delta)
        return -self._ports.state_ports().take(change)

    def flux_adjoint(self, state: V, weights: A, point: SteadyPoint[V, A]) -> V:
        r""":math:`-F_u^\top B\, W`: the weights placed on the port rows,
        then the Jacobian's transpose."""
        space = self._ports.state_space()
        placed = self._ports.state_ports().add_to(
            space.zeros(int(weights.shape[1])), weights
        )
        label = LinearizationPoint(self._pinned.parameters_version(), point)
        return space.scale(-1.0, self._jacobian(state, label).apply_transpose(placed))

    def param_derivatives(self) -> FluxParamDerivatives[V, A]:
        r""":math:`-B^\top F_p` when the residual has a parameter
        Jacobian; independent when it has no parameters."""
        return self._param_derivs

    def _own_param(self) -> ResidualParamFn[V]:
        if self._param is None:
            raise RuntimeError(
                "the residual has no parameter Jacobian; check "
                "param_derivatives() before calling"
            )
        return self._param

    def _param_tangent(self, state: V, direction: V) -> A:
        return -self._ports.state_ports().take(
            self._own_param()(state).apply(direction)
        )

    def _param_adjoint(self, state: V, weights: A) -> V:
        space = self._ports.state_space()
        placed = self._ports.state_ports().add_to(
            space.zeros(int(weights.shape[1])), weights
        )
        return self._own_param()(state).apply_transpose(space.scale(-1.0, placed))


class PortTangent(Generic[V, A]):
    r"""What a point's trace tangent reads: the port residual's parameter
    Jacobian :math:`R_\theta`, the port data and the output. The port
    residual is the port-pinned one of :class:`SteadyPinnedResponse` or
    the port-loaded one of :class:`~pyapprox.pde.coupling.
    loaded_response.SteadyLoadedResponse`; the tangent solve is the same
    for both. ``physical``, the placement of the response's own
    parameters in :math:`\theta`, or ``None`` without them."""

    def __init__(
        self,
        param: ResidualParamFn[V],
        ports: PortPinnedData[V, A],
        flux: PortFluxProtocol[V, A],
        physical: Optional[PortPlacement[V, A]] = None,
    ) -> None:
        self.param, self.ports, self.flux = param, ports, flux
        self.physical = physical


class SteadyPinnedPoint(Generic[V, A]):
    r"""A steady port response (:class:`SteadyPinnedResponse`, or a
    port-loaded one) solved at one set of port data and parameters.

    Its derivatives hold while the response is not linearized again: a
    later solve changes the port data, and a tangent requested after it
    is refused. With parameters of its own its bundle also carries
    ``jvp_param`` and ``vjp_param``: the same tangent and adjoint solves,
    placed on and read from the parameters' rows of :math:`\\theta`.
    """

    def __init__(
        self,
        point: SteadyPoint[V, A],
        output: A,
        bkd: Backend[A],
        tangent: PortTangent[V, A],
    ) -> None:
        self._point, self._output, self._bkd = point, output, bkd
        self._tangent = tangent
        self._explicit = tangent.flux.param_derivatives()
        self._derivs: ResponseDerivatives[A] = (
            ResponseDerivatives.parametric(
                self.jvp_trace, self.vjp_trace, self._jvp_param, self._vjp_param
            )
            if tangent.physical is not None and self._explicit.available()
            else ResponseDerivatives.adjoint(self.jvp_trace, self.vjp_trace)
        )

    def output(self) -> A:
        """The response's output on the ports: the outward flux for a
        pinned response. Shape: (nport, 1)."""
        return self._output

    def qoi(self) -> A:
        return self._bkd.zeros((0, 1))

    def derivatives(self) -> ResponseDerivatives[A]:
        return self._derivs

    def jvp_trace(self, directions: A) -> A:
        r""":math:`(\partial r/\partial \lambda)` ``directions``, for a
        block of shape (nport, ncols): one tangent solve :math:`R_u\,
        \delta u = -R_\theta [d; 0]` for all columns, then the output's
        tangent at :math:`\delta u`.

        Raises
        ------
        ValueError
            If ``directions`` has the wrong shape, or the response has been
            linearized again since this point.
        """
        return self._tangent_on(self._tangent.ports.param_ports(), directions)

    def vjp_trace(self, weights: A) -> A:
        r""":math:`(\partial r/\partial \lambda)^\top` ``weights``, for a
        block of shape (nport, ncols): the transpose of :meth:`jvp_trace`,
        one transposed solve :math:`R_u^\top z = C^\top W` for all columns,
        :math:`C` the output's derivative in the state, then the port data's
        rows of :math:`-R_\theta^\top z`.

        Raises
        ------
        ValueError
            If ``weights`` has the wrong shape, the response has been
            linearized again since this point, or the solve does not
            converge.
        TypeError
            If the Jacobian's solver has no transposed solve.
        """
        return self._adjoint_on(self._tangent.ports.param_ports(), weights)

    def _jvp_param(self, directions: A) -> A:
        r""":math:`(\partial r/\partial p)` ``directions``, for a block of
        shape (nparams, ncols): :meth:`jvp_trace`'s solve with the
        directions on the parameters' rows of :math:`\theta = [\lambda;
        p]`, plus the output's explicit change in :math:`p`. Raises as
        :meth:`jvp_trace`, and RuntimeError without parameters."""
        physical = self._physical()
        through_state = self._tangent_on(physical.range_rows(), directions)
        if self._explicit.tangent is None:
            return through_state
        theta = physical.range_rows().add_to(
            self._tangent.ports.param_space().zeros(int(directions.shape[1])),
            directions,
        )
        own = physical.apply_transpose(theta)
        explicit = self._explicit.tangent(self._point.solution(), own)
        return through_state + explicit

    def _vjp_param(self, weights: A) -> A:
        r""":math:`(\partial r/\partial p)^\top` ``weights``, for a block of
        shape (nport, ncols): :meth:`vjp_trace`'s solve, read on the
        parameters' rows of :math:`-R_\theta^\top z`, plus the explicit
        change's transpose. Shape: (nparams, ncols). Raises as
        :meth:`vjp_trace`, and RuntimeError without parameters."""
        physical = self._physical()
        rows = physical.range_rows()
        through_state = self._adjoint_on(rows, weights)
        if self._explicit.adjoint is None:
            return through_state
        own = self._explicit.adjoint(self._point.solution(), weights)
        return through_state + rows.take(physical.apply(own))

    def _physical(self) -> PortPlacement[V, A]:
        physical = self._tangent.physical
        if physical is None:
            raise RuntimeError(
                "the response has no parameters; check derivatives() before calling"
            )
        return physical

    def _tangent_on(self, rows: PortRowsProtocol[V, A], directions: A) -> A:
        """The output's change for ``directions`` placed on ``rows`` of
        :math:`\\theta`, one tangent solve for all columns."""
        t = self._tangent
        if directions.ndim != 2 or directions.shape[0] != rows.nrows():
            raise ValueError(
                f"directions must have shape ({rows.nrows()}, ncols), got "
                f"{tuple(directions.shape)}"
            )
        state = self._point.solution()
        params = t.ports.param_space()
        theta = rows.add_to(params.zeros(int(directions.shape[1])), directions)
        space = t.ports.state_space()
        rhs = space.scale(-1.0, t.param(state).apply(theta))
        delta, info = self._point.jacobian_handle().solve(rhs)
        if not info.converged:
            raise ValueError("the tangent solve did not converge")
        return t.flux.flux_tangent(state, delta, self._point)

    def _adjoint_on(self, rows: PortRowsProtocol[V, A], weights: A) -> A:
        """``rows`` of :math:`-R_\\theta^\\top z`, :math:`R_u^\\top z =
        C^\\top W`, one transposed solve for all columns."""
        t = self._tangent
        nport = t.ports.nport()
        if weights.ndim != 2 or weights.shape[0] != nport:
            raise ValueError(
                f"weights must have shape ({nport}, ncols), got {tuple(weights.shape)}"
            )
        state = self._point.solution()
        transposed = self._point.jacobian_handle().transpose()
        if transposed is None:
            raise TypeError("the Jacobian's solver has no transposed solve")
        adjoint, info = transposed.solve(
            t.flux.flux_adjoint(state, weights, self._point)
        )
        if not info.converged:
            raise ValueError("the adjoint solve did not converge")
        theta = t.param(state).apply_transpose(adjoint)
        return -rows.take(theta)

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
    flux : PortFluxProtocol
        Reads the tested outward flux from the solved state, e.g.
        :class:`PinnedReactionFlux` over ``pinned`` and ``ports``.
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
        If ``flux`` or ``parameters`` does not satisfy its protocol.
    ValueError
        If ``parameters`` and ``physical`` are not given together, the
        port-pinned residual has no parameter Jacobian, or as
        :func:`~pyapprox.pde.coupling.pinned_ports.pin_ports`.
    """

    def __init__(
        self,
        pinned: PinnedResidual[V, F],
        ports: PortPinnedData[V, A],
        flux: PortFluxProtocol[V, A],
        space: InnerProductSpaceProtocol[V, A],
        setups: SetupStrategyProtocol[SteadyRequest[V, F], V, F],
        nonlinear: NonlinearSolverProtocol[V, A],
        bkd: Backend[A],
        parameters: Optional[ResponseParametersProtocol[A]] = None,
        physical: Optional[PortPlacement[V, A]] = None,
    ) -> None:
        if not isinstance(flux, PortFluxProtocol):
            raise TypeError(
                f"flux must satisfy PortFluxProtocol, got {type(flux).__name__}"
            )
        if (parameters is None) != (physical is None):
            raise ValueError("parameters and physical come together")
        if parameters is not None and not isinstance(
            parameters, ResponseParametersProtocol
        ):
            raise TypeError(
                "parameters must satisfy ResponseParametersProtocol, got "
                f"{type(parameters).__name__}"
            )
        self._ports, self._flux = ports, flux
        self._residual = pin_ports(pinned, ports, physical)
        param = self._residual.derivatives().param
        if param is None:
            raise ValueError(
                "a trace tangent needs the port-pinned residual's parameter Jacobian"
            )
        self._system = SteadySystem(self._residual, space, setups)
        self._tangent = PortTangent(param, ports, flux, physical)
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
        output = self._flux.flux(point.state())
        return SteadyPinnedPoint(point, output, self._bkd, self._tangent)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nport={self.nport()})"
