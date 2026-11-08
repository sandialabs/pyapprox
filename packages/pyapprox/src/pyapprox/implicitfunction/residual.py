r"""The residual of an implicit function :math:`R(u; p) = 0`.

A residual evaluates :math:`R` at a state, returns its optional
derivatives as a bundle
(:class:`~pyapprox.implicitfunction.residual_derivatives.ResidualDerivatives`)
and reports a **parameter version**: a counter that every change of the
parameters increments. Whatever is computed at one parameter value (a
Jacobian label, a factorization) records the version, so it can be
recognized as current or stale by comparing integers. A residual without
parameters reports 0 for its whole life.

States are single-column vectors of the state space, as are the values.
"""

from typing import Protocol, TypeVar, runtime_checkable

from pyapprox.implicitfunction.residual_derivatives import ResidualDerivatives

V = TypeVar("V")
F_co = TypeVar("F_co", covariant=True)


@runtime_checkable
class ResidualProtocol(Protocol[V, F_co]):
    """:math:`R(u; p)`, its derivatives bundle, whose Jacobian is in format
    ``F_co``, and its parameter version."""

    def __call__(self, state: V) -> V:
        """:math:`R` at ``state``, a single column of the state space."""
        ...

    def derivatives(self) -> ResidualDerivatives[V, F_co]:
        """The optional derivatives; which fields are present is fixed for
        the residual's life."""
        ...

    def parameters_version(self) -> int:
        """The number of parameter changes so far; 0, unchanging, without
        parameters."""
        ...
