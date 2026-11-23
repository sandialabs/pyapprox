r"""Local responses: what a subdomain returns for its interface data.

A response :math:`r_i(\lambda_i, p)` maps a subdomain's port data
:math:`\lambda_i` (and parameters :math:`p`) to an output on its ports.
Each response declares what it :meth:`~LocalResponseProtocol.receives`
and what it :meth:`~LocalResponseProtocol.returns` as
:class:`PortQuantity` values: a Dirichlet-to-Neumann response receives
the :class:`Trace` and returns the :class:`Flux` tested against the
skeleton basis. The skeleton sums the outputs through the port
transfers, :math:`S(\lambda) = \sum_i E_i^\top r_i(E_i \lambda, p)`, so a
response is all the coupling sees of a subdomain: how it discretizes or
solves its interior is its own.

A response *linearizes, then answers queries*: :meth:`LocalResponseProtocol.
linearize` does the subdomain's work once for the given data and returns
a point whose output, quantities of interest and derivatives refer to
that data. Port data, outputs, parameters and QoIs are single-column
backend arrays: port blocks are the only data crossing between stacks,
and small replicated data stay arrays.

Derivatives are optional and travel in a frozen
:class:`ResponseDerivatives` bundle; an absent capability is ``None``.
"""

from dataclasses import dataclass, replace
from typing import Callable, Generic, Optional, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend

TraceJVP = Callable[[Array], Array]
r"""Directional derivatives of the output in the port data,
:math:`(\partial r_i/\partial \lambda_i) V` for a block :math:`V` of
shape (nport, ncols)."""

TraceVJP = Callable[[Array], Array]
r"""Transposed products of the output in the port data,
:math:`(\partial r_i/\partial \lambda_i)^\top W` for a block :math:`W` of
shape (nport, ncols)."""

ParamJVP = Callable[[Array], Array]
r"""Directional derivatives of the output in the response's parameters,
:math:`(\partial r_i/\partial p_i) V` for a block :math:`V` of shape
(nparams, ncols); shape (nport, ncols)."""

ParamVJP = Callable[[Array], Array]
r"""Transposed products of the output in the response's parameters,
:math:`(\partial r_i/\partial p_i)^\top W` for a block :math:`W` of
shape (nport, ncols); shape (nparams, ncols)."""


@dataclass(frozen=True)
class PortQuantity:
    """What a port block holds. Compared by value; a new kind of port
    data is a new subclass."""


@dataclass(frozen=True)
class Trace(PortQuantity):
    """The field's values on the ports, as skeleton coefficients
    (Dirichlet data)."""

    def matching_sign(self) -> int:
        """-1: the traces ``a``, ``b`` two pieces hold on a shared port
        match when ``a - b = 0`` (continuity)."""
        return -1


@dataclass(frozen=True)
class Flux(PortQuantity):
    """The outward normal flux on the ports, tested against the skeleton
    basis (Neumann data)."""

    def matching_sign(self) -> int:
        """+1: the outward fluxes ``a``, ``b`` two pieces hold on a shared
        port match when ``a + b = 0`` (conservation)."""
        return 1


@dataclass(frozen=True)
class TimeWindow:
    """The time interval a response covers; ``start == end`` is steady.

    Parameters
    ----------
    start, end
        The window's ends, ``end >= start``.
    """

    start: float = 0.0
    end: float = 0.0

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"window end {self.end} precedes its start {self.start}")

    @staticmethod
    def steady() -> "TimeWindow":
        """The degenerate window of a steady problem."""
        return TimeWindow()

    def is_steady(self) -> bool:
        return self.start == self.end


@dataclass(frozen=True)
class ResponseDerivatives(Generic[Array]):
    r"""The optional derivatives of a response at one linearization point.

    - ``jvp_trace``: :math:`(\partial r_i/\partial \lambda_i) V` for a
      block :math:`V` of port directions, what Newton on the skeleton
      needs.
    - ``vjp_trace``: :math:`(\partial r_i/\partial \lambda_i)^\top W` for
      a block :math:`W` of output weights, shape (nport, ncols) both ways:
      the skeleton Jacobian's transpose, what adjoints need. It comes only
      with ``jvp_trace``.
    - ``jvp_param``: :math:`(\partial r_i/\partial p_i) V` for a block of
      parameter directions, what a coupled tangent needs beside
      ``jvp_trace``. It comes only with ``jvp_trace``.
    - ``vjp_param``: its transpose, what a coupled gradient needs beside
      ``vjp_trace``. It comes only with ``jvp_param`` and ``vjp_trace``.

    A response without parameters carries no parameter fields: its output
    does not depend on any.

    The same four derivatives of the response's quantities of interest,
    for a response with them: ``qoi_jvp_trace`` (nport, ncols) to (nqoi,
    ncols), ``qoi_vjp_trace`` (nqoi, ncols) to (nport, ncols),
    ``qoi_jvp_param`` (nparams, ncols) to (nqoi, ncols), ``qoi_vjp_param``
    (nqoi, ncols) to (nparams, ncols), with the same pairing rules.

    Build with the named constructors, adding the QoI fields by
    :meth:`with_qoi`.
    """

    jvp_trace: Optional[TraceJVP[Array]] = None
    vjp_trace: Optional[TraceVJP[Array]] = None
    jvp_param: Optional[ParamJVP[Array]] = None
    vjp_param: Optional[ParamVJP[Array]] = None
    qoi_jvp_trace: Optional[Callable[[Array], Array]] = None
    qoi_vjp_trace: Optional[Callable[[Array], Array]] = None
    qoi_jvp_param: Optional[Callable[[Array], Array]] = None
    qoi_vjp_param: Optional[Callable[[Array], Array]] = None

    def __post_init__(self) -> None:
        _check_quartet(
            self.jvp_trace, self.vjp_trace, self.jvp_param, self.vjp_param, ""
        )
        _check_quartet(
            self.qoi_jvp_trace,
            self.qoi_vjp_trace,
            self.qoi_jvp_param,
            self.qoi_vjp_param,
            "qoi_",
        )

    def with_qoi(
        self,
        jvp_trace: Callable[[Array], Array],
        vjp_trace: Callable[[Array], Array],
        jvp_param: Optional[Callable[[Array], Array]] = None,
        vjp_param: Optional[Callable[[Array], Array]] = None,
    ) -> "ResponseDerivatives[Array]":
        """This bundle with the derivatives of the quantities of interest:
        in the port data, and in the parameters when given."""
        return replace(
            self,
            qoi_jvp_trace=jvp_trace,
            qoi_vjp_trace=vjp_trace,
            qoi_jvp_param=jvp_param,
            qoi_vjp_param=vjp_param,
        )

    @staticmethod
    def none() -> "ResponseDerivatives[Array]":
        """No derivatives."""
        return ResponseDerivatives()

    @staticmethod
    def tangent(jvp_trace: TraceJVP[Array]) -> "ResponseDerivatives[Array]":
        """Trace directional derivatives: a forward skeleton solve."""
        return ResponseDerivatives(jvp_trace=jvp_trace)

    @staticmethod
    def adjoint(
        jvp_trace: TraceJVP[Array], vjp_trace: TraceVJP[Array]
    ) -> "ResponseDerivatives[Array]":
        """Trace directional derivatives and their transpose: forward
        solves and adjoints."""
        return ResponseDerivatives(jvp_trace=jvp_trace, vjp_trace=vjp_trace)

    @staticmethod
    def parametric(
        jvp_trace: TraceJVP[Array],
        vjp_trace: TraceVJP[Array],
        jvp_param: ParamJVP[Array],
        vjp_param: ParamVJP[Array],
    ) -> "ResponseDerivatives[Array]":
        """Trace and parameter directional derivatives and their
        transposes: coupled tangents and gradients in the parameters."""
        return ResponseDerivatives(
            jvp_trace=jvp_trace,
            vjp_trace=vjp_trace,
            jvp_param=jvp_param,
            vjp_param=vjp_param,
        )


def _check_quartet(
    jvp_trace: object,
    vjp_trace: object,
    jvp_param: object,
    vjp_param: object,
    prefix: str,
) -> None:
    """One quartet's fields are callables or None and come in the order
    the derivatives need: a transpose with its forward product, a
    parameter derivative with the trace one."""
    for name, value in (
        ("jvp_trace", jvp_trace),
        ("vjp_trace", vjp_trace),
        ("jvp_param", jvp_param),
        ("vjp_param", vjp_param),
    ):
        if value is not None and not callable(value):
            raise TypeError(
                f"ResponseDerivatives field '{prefix}{name}' must be callable or "
                f"None; got {type(value).__name__}"
            )
    if vjp_trace is not None and jvp_trace is None:
        raise ValueError(f"{prefix}vjp_trace comes only with {prefix}jvp_trace")
    if jvp_param is not None and jvp_trace is None:
        raise ValueError(f"{prefix}jvp_param comes only with {prefix}jvp_trace")
    if vjp_param is not None and (jvp_param is None or vjp_trace is None):
        raise ValueError(
            f"{prefix}vjp_param comes only with {prefix}jvp_param and {prefix}vjp_trace"
        )


@runtime_checkable
class ResponsePointProtocol(Protocol[Array]):
    """A response linearized at one set of port data, parameters and
    window."""

    def output(self) -> Array:
        """The quantity the response :meth:`~LocalResponseProtocol.returns`
        on its ports (the tested flux, for a Dirichlet-to-Neumann
        response). Shape: (nport, 1)."""
        ...

    def qoi(self) -> Array:
        """The subdomain's quantities of interest. Shape: (nqoi, 1)."""
        ...

    def derivatives(self) -> ResponseDerivatives[Array]:
        """The optional derivatives at this point."""
        ...


@runtime_checkable
class LocalResponseProtocol(Protocol[Array]):
    """A subdomain as the skeleton sees it."""

    def bkd(self) -> Backend[Array]:
        """The backend of the port blocks, parameters and QoIs."""
        ...

    def nport(self) -> int:
        """The number of rows of the port data and output."""
        ...

    def receives(self) -> "PortQuantity":
        """What the port data are: :class:`Trace` for Dirichlet data."""
        ...

    def returns(self) -> "PortQuantity":
        """What the output is: :class:`Flux` for the tested flux."""
        ...

    def nparams(self) -> int:
        """The number of parameters; 0 without parameters."""
        ...

    def nqoi(self) -> int:
        """The number of quantities of interest; may be 0."""
        ...

    def linearize(
        self, port_data: Array, params: Array, window: TimeWindow
    ) -> ResponsePointProtocol[Array]:
        """Do the subdomain's work for ``port_data`` (nport, 1) and
        ``params`` (nparams, 1) over ``window``."""
        ...


class DeclaredKinds(Generic[Array]):
    """``response`` declaring other kinds than its own: the physical
    quantities its port data and output are, where its class names a
    generic role. A pinned response receives a :class:`Trace` and returns
    a :class:`Flux`; as the Darcy side of a Stokes-Darcy interface the
    trace is an interface pressure and the flux a normal velocity, the
    kinds the Stokes side receives and returns too, so the interface has
    one unknown per DOF (:mod:`~pyapprox.pde.coupling.kind_layout`).

    Satisfies :class:`LocalResponseProtocol`; everything but the kinds is
    ``response``'s.

    Raises
    ------
    TypeError
        If ``response`` does not satisfy :class:`LocalResponseProtocol`.
    """

    def __init__(
        self,
        response: LocalResponseProtocol[Array],
        receives: PortQuantity,
        returns: PortQuantity,
    ) -> None:
        if not isinstance(response, LocalResponseProtocol):
            raise TypeError(
                "response must satisfy LocalResponseProtocol, got "
                f"{type(response).__name__}"
            )
        self._response = response
        self._receives, self._returns = receives, returns

    def bkd(self) -> Backend[Array]:
        return self._response.bkd()

    def nport(self) -> int:
        return self._response.nport()

    def receives(self) -> PortQuantity:
        return self._receives

    def returns(self) -> PortQuantity:
        return self._returns

    def nparams(self) -> int:
        return self._response.nparams()

    def nqoi(self) -> int:
        return self._response.nqoi()

    def linearize(
        self, port_data: Array, params: Array, window: TimeWindow
    ) -> ResponsePointProtocol[Array]:
        return self._response.linearize(port_data, params, window)

    def wrapped(self) -> LocalResponseProtocol[Array]:
        """The response whose kinds this replaces."""
        return self._response
