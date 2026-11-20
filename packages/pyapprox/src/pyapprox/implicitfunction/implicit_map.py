r"""What an implicit map :math:`p \mapsto Q(u(p);\, p)` can differentiate.

An implicit map solves for a state :math:`u(p)` (a steady solution or a
trajectory) and returns a functional of it. A solve at one parameter
gives a point, and the derivatives are taken at the point: a tangent
:math:`dQ/dp\, v` by one forward solve, a gradient
:math:`d(\omega^\top Q)/dp` by one adjoint solve, and a Hessian-vector
product :math:`(d^2 \omega^\top Q/dp^2)\, v` by both.
:class:`ImplicitMapDerivatives` declares which of these a map has, as
callables over its point. A map fixes it when it is built, as the
weakest of its parts' bundles, so a consumer reads it before any solve.

A map works in its stack's types: a parameter vector in, a value of
``nqoi`` entries out. :class:`ImplicitMapFunction` makes one a
:class:`~pyapprox.interface.functions.protocols.function.FunctionProtocol`
of backend sample columns, converting each sample to the stack's
parameter vector through a :class:`ParameterSpaceMapProtocol`, and
offers the map's derivatives in a
:class:`~pyapprox.interface.functions.derivatives.Derivatives` bundle: the
Jacobian, formed by rows of gradients or columns of tangents, and the
weighted Hessian-vector product for every ``nqoi``.
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.validation import (
    validate_sample,
    validate_samples,
)
from pyapprox.util.backends.protocols import ArrayProtocol, Backend

P = TypeVar("P")
V = TypeVar("V")
A = TypeVar("A", bound=ArrayProtocol)
Q = TypeVar("Q")
W = TypeVar("W")
B = TypeVar("B", bound=ArrayProtocol)

PointTangentFn = Callable[[P, V], A]
r"""``(point, direction) ->`` :math:`dQ/dp\, v`, shape ``(nqoi, 1)``."""

PointGradientFn = Callable[[P, Optional[A]], V]
r"""``(point, weights) ->`` :math:`d(\omega^\top Q)/dp`, a parameter
vector; ``weights`` of shape ``(nqoi, 1)``, None only when ``nqoi`` is
1."""

PointHVPFn = Callable[[P, V, Optional[A]], V]
r"""``(point, direction, weights) ->`` :math:`(d^2 \omega^\top Q/dp^2)\,
v`, a parameter vector."""

_FIELDS = ("tangent", "gradient", "hvp")


@dataclass(frozen=True)
class ImplicitMapDerivatives(Generic[P, V, A]):
    r"""The optional derivatives of an implicit map, as callables over a
    point it solved.

    ``tangent`` and ``gradient`` are separate fields: a tangent solves
    with the state equation's Jacobian and a gradient with its transpose,
    which a solver may lack. Build with :meth:`none`,
    :meth:`first_order` or :meth:`second_order`.

    Parameters
    ----------
    tangent : callable, optional
        :math:`dQ/dp\, v` at a point.
    gradient : callable, optional
        :math:`d(\omega^\top Q)/dp` at a point.
    hvp : callable, optional
        :math:`(d^2 \omega^\top Q/dp^2)\, v` at a point, for value
        weights :math:`\omega` (every ``nqoi``).

    Raises
    ------
    TypeError
        If a field is neither callable nor None.
    ValueError
        If ``hvp`` is given without both ``tangent`` and ``gradient``,
        which the second-order adjoint uses.
    """

    tangent: Optional[PointTangentFn[P, V, A]] = None
    gradient: Optional[PointGradientFn[P, A, V]] = None
    hvp: Optional[PointHVPFn[P, V, A]] = None

    def __post_init__(self) -> None:
        for name in _FIELDS:
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"ImplicitMapDerivatives field '{name}' must be callable or "
                    f"None; got {type(value).__name__}"
                )
        if self.hvp is not None and (self.tangent is None or self.gradient is None):
            raise ValueError("hvp needs tangent and gradient")

    @staticmethod
    def none() -> "ImplicitMapDerivatives[Q, W, B]":
        """No derivatives: the map is only evaluated."""
        return ImplicitMapDerivatives()

    @staticmethod
    def first_order(
        tangent: PointTangentFn[Q, W, B],
        gradient: PointGradientFn[Q, B, W],
    ) -> "ImplicitMapDerivatives[Q, W, B]":
        """Tangents and gradients."""
        return ImplicitMapDerivatives(tangent, gradient)

    @staticmethod
    def second_order(
        tangent: PointTangentFn[Q, W, B],
        gradient: PointGradientFn[Q, B, W],
        hvp: PointHVPFn[Q, W, B],
    ) -> "ImplicitMapDerivatives[Q, W, B]":
        """Tangents, gradients and Hessian-vector products."""
        return ImplicitMapDerivatives(tangent, gradient, hvp)


@runtime_checkable
class ImplicitMapProtocol(Protocol[P, V, A]):
    r"""What :class:`ImplicitMapFunction` reads of an implicit map: its
    sizes, its value, a solved point, and which derivatives a point has."""

    def nparams(self) -> int:
        """The number of parameters."""
        ...

    def nqoi(self) -> int:
        """The number of entries of the value."""
        ...

    def derivatives(self) -> ImplicitMapDerivatives[P, V, A]:
        """Which derivatives a point has, fixed when the map is built."""
        ...

    def value(self, param: V) -> A:
        """:math:`Q` at the parameter vector ``param``, shape ``(nqoi, 1)``."""
        ...

    def point(self, param: V) -> P:
        """The solve at ``param``, for the derivatives."""
        ...


@runtime_checkable
class ParameterSpaceMapProtocol(Protocol[V, A]):
    """Converts between a backend sample column and the stack's parameter
    vector; the backend of those columns is its."""

    def bkd(self) -> Backend[A]:
        """The backend of the sample columns."""
        ...

    def from_array(self, sample: A) -> V:
        """The parameter vector of the column ``sample``, shape
        ``(nparams, 1)``."""
        ...

    def to_array(self, param: V) -> A:
        """The column of the parameter vector ``param``, shape
        ``(nparams, 1)``."""
        ...


class ArrayParameters(Generic[A]):
    """The parameter-space map of a stack whose parameter vectors are
    backend columns: both conversions return their argument.

    Parameters
    ----------
    bkd : Backend
        The backend of the columns.
    """

    def __init__(self, bkd: Backend[A]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[A]:
        return self._bkd

    def from_array(self, sample: A) -> A:
        return sample

    def to_array(self, param: A) -> A:
        return param

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@runtime_checkable
class ImplicitJacobianProtocol(Protocol):
    """How an implicit map's Jacobian is formed at a solved point from
    the derivatives the map declares."""

    def supported(self, derivatives: ImplicitMapDerivatives[P, V, A]) -> bool:
        """Whether ``derivatives`` has the field this strategy uses."""
        ...

    def jacobian(
        self,
        derivatives: ImplicitMapDerivatives[P, V, A],
        point: P,
        parameters: ParameterSpaceMapProtocol[V, A],
        nqoi: int,
        nparams: int,
    ) -> A:
        """:math:`dQ/dp` at ``point``, shape ``(nqoi, nparams)``, on the
        backend of ``parameters``.

        Raises
        ------
        TypeError
            If ``derivatives`` lacks the field this strategy uses.
        """
        ...


class AdjointRows:
    """The Jacobian by rows: one gradient (adjoint solve) per entry of
    the value. Cheap with few entries; needs the map's ``gradient``."""

    def supported(self, derivatives: ImplicitMapDerivatives[P, V, A]) -> bool:
        return derivatives.gradient is not None

    def jacobian(
        self,
        derivatives: ImplicitMapDerivatives[P, V, A],
        point: P,
        parameters: ParameterSpaceMapProtocol[V, A],
        nqoi: int,
        nparams: int,
    ) -> A:
        gradient = derivatives.gradient
        if gradient is None:
            raise TypeError("the map has no gradient, so no Jacobian by rows")
        bkd = parameters.bkd()
        eye = bkd.eye(nqoi)
        return bkd.vstack(
            [
                parameters.to_array(gradient(point, eye[:, ii : ii + 1])).T
                for ii in range(nqoi)
            ]
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class TangentColumns:
    """The Jacobian by columns: one tangent (forward solve) per
    parameter. Cheap with few parameters; needs the map's ``tangent``."""

    def supported(self, derivatives: ImplicitMapDerivatives[P, V, A]) -> bool:
        return derivatives.tangent is not None

    def jacobian(
        self,
        derivatives: ImplicitMapDerivatives[P, V, A],
        point: P,
        parameters: ParameterSpaceMapProtocol[V, A],
        nqoi: int,
        nparams: int,
    ) -> A:
        tangent = derivatives.tangent
        if tangent is None:
            raise TypeError("the map has no tangent, so no Jacobian by columns")
        bkd = parameters.bkd()
        eye = bkd.eye(nparams)
        return bkd.hstack(
            [
                tangent(point, parameters.from_array(eye[:, jj : jj + 1]))
                for jj in range(nparams)
            ]
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ImplicitMapFunction(Generic[P, V, A]):
    r"""An implicit map as a
    :class:`~pyapprox.interface.functions.protocols.function.FunctionProtocol`
    of backend sample columns (see the module docstring).

    ``derivatives()`` is decided once, here, from the map's bundle: none
    when no Jacobian strategy is supported; otherwise the Jacobian, and
    with the map's ``hvp`` the weighted Hessian-vector product ``whvp``
    for every ``nqoi`` (``resolved_hvp`` gives the plain one when
    ``nqoi`` is 1). Each derivative solves once at its sample.

    Parameters
    ----------
    implicit : ImplicitMapProtocol
        The map.
    parameters : ParameterSpaceMapProtocol
        Converts the sample columns to the map's parameter vectors; its
        backend is the function's.
    jacobian : ImplicitJacobianProtocol, optional
        How the Jacobian is formed. Default: of the strategies the map's
        bundle supports, :class:`AdjointRows` when there are no more
        entries than parameters, else :class:`TangentColumns`, whichever
        takes fewer solves.

    Raises
    ------
    TypeError
        If an argument does not satisfy its protocol, or ``jacobian`` uses
        a derivative the map does not declare.
    """

    def __init__(
        self,
        implicit: ImplicitMapProtocol[P, V, A],
        parameters: ParameterSpaceMapProtocol[V, A],
        jacobian: Optional[ImplicitJacobianProtocol] = None,
    ) -> None:
        for argument, protocol, name in (
            (implicit, ImplicitMapProtocol, "implicit"),
            (parameters, ParameterSpaceMapProtocol, "parameters"),
        ):
            if not isinstance(argument, protocol):
                raise TypeError(
                    f"{name} must satisfy {protocol.__name__}, got "
                    f"{type(argument).__name__}"
                )
        self._implicit, self._parameters = implicit, parameters
        self._map_derivatives = implicit.derivatives()
        self._strategy: Optional[ImplicitJacobianProtocol] = self._choose(jacobian)
        self._derivs: Derivatives[A] = (
            Derivatives.none()
            if self._strategy is None
            else Derivatives.first_order(jacobian=self._jacobian)
            if self._map_derivatives.hvp is None
            else Derivatives.second_order_weighted(
                jacobian=self._jacobian, whvp=self._whvp
            )
        )

    def _choose(
        self, given: Optional[ImplicitJacobianProtocol]
    ) -> Optional[ImplicitJacobianProtocol]:
        """The Jacobian strategy: ``given``, or the cheaper one the map's
        bundle supports; None when it supports neither."""
        bundle = self._map_derivatives
        if given is not None:
            if not isinstance(given, ImplicitJacobianProtocol):
                raise TypeError(
                    "jacobian must satisfy ImplicitJacobianProtocol, got "
                    f"{type(given).__name__}"
                )
            if not given.supported(bundle):
                raise TypeError(f"{given!r} uses a derivative the map does not declare")
            return given
        rows, columns = AdjointRows(), TangentColumns()
        preferred = (rows, columns) if self.nqoi() <= self.nvars() else (columns, rows)
        for strategy in preferred:
            if strategy.supported(bundle):
                return strategy
        return None

    def bkd(self) -> Backend[A]:
        """The backend of the samples and values."""
        return self._parameters.bkd()

    def nvars(self) -> int:
        """The number of parameters."""
        return self._implicit.nparams()

    def nqoi(self) -> int:
        """The number of entries of the value."""
        return self._implicit.nqoi()

    def derivatives(self) -> Derivatives[A]:
        """The derivative bundle, with respect to the parameters."""
        return self._derivs

    def __call__(self, samples: A) -> A:
        """The value at each sample.

        Parameters
        ----------
        samples : Array
            Shape: ``(nvars, nsamples)``.

        Returns
        -------
        Array
            Shape: ``(nqoi, nsamples)``.
        """
        validate_samples(self.nvars(), samples)
        implicit, parameters = self._implicit, self._parameters
        return self.bkd().hstack(
            [
                implicit.value(parameters.from_array(samples[:, ii : ii + 1]))
                for ii in range(samples.shape[1])
            ]
        )

    def _jacobian(self, sample: A) -> A:
        """:math:`dq/dp` at one sample. Shape: ``(nqoi, nvars)``."""
        strategy = self._strategy
        if strategy is None:
            raise RuntimeError(
                "jacobian is unavailable; check derivatives() before calling"
            )
        validate_sample(self.nvars(), sample)
        point = self._implicit.point(self._parameters.from_array(sample))
        return strategy.jacobian(
            self._map_derivatives, point, self._parameters, self.nqoi(), self.nvars()
        )

    def _whvp(self, sample: A, vec: A, weights: A) -> A:
        r""":math:`(\sum_i w_i \nabla^2 q_i)\, v` at one sample. Shape:
        ``(nvars, 1)``."""
        hvp = self._map_derivatives.hvp
        if hvp is None:
            raise RuntimeError(
                "whvp is unavailable; check derivatives() before calling"
            )
        validate_sample(self.nvars(), sample)
        validate_sample(self.nvars(), vec)
        if tuple(weights.shape) != (self.nqoi(), 1):
            raise ValueError(
                f"weights must have shape ({self.nqoi()}, 1), got "
                f"{tuple(weights.shape)}"
            )
        parameters = self._parameters
        point = self._implicit.point(parameters.from_array(sample))
        return parameters.to_array(hvp(point, parameters.from_array(vec), weights))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nvars={self.nvars()}, nqoi={self.nqoi()}, "
            f"jacobian={self._strategy!r}, derivatives={self._derivs!r})"
        )
