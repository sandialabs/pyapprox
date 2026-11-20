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
"""

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

P = TypeVar("P")
V = TypeVar("V")
A = TypeVar("A")
Q = TypeVar("Q")
W = TypeVar("W")
B = TypeVar("B")

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
