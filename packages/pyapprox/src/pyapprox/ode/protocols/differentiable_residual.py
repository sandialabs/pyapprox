r"""Protocols for ODE residuals whose derivatives come as a bundle.

A residual :math:`f(y, t; p)` of :math:`M \dot y = f` returns its
optional derivatives from ``derivatives()`` (see
:mod:`pyapprox.ode.residual_derivatives`). A parameterized one also sets
its parameters and reports a **parameter version**: a counter that every
``set_param`` increments, even when the value is unchanged. Whatever is
computed from the residual at one parameter value (a Jacobian label, a
solver setup, an assembled operator) records the version, so it can be
recognized as current or stale by comparing integers. The version
carries a caller's decision that the parameters changed; deciding
whether two parameter values are the same point is the caller's.

A residual without parameters has one parameter version for its whole
life, so a consumer reads such a residual's version as 0.
"""

from typing import Protocol, runtime_checkable

from pyapprox.ode.residual_derivatives import ODEResidualDerivatives
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import FormatProtocol


@runtime_checkable
class DifferentiableODEResidualProtocol(Protocol[Array]):
    """A residual :math:`f(y, t)` with a derivatives bundle."""

    def bkd(self) -> Backend[Array]: ...

    def __call__(self, state: Array, time: float) -> Array: ...

    def derivatives(self) -> ODEResidualDerivatives[Array, FormatProtocol]: ...


@runtime_checkable
class ParameterizedODEResidualProtocol(Protocol[Array]):
    """A residual :math:`f(y, t; p)` with a derivatives bundle whose
    parameters can be set.

    A bundle is valid for the parameters set when it was fetched, so a
    consumer fetches ``derivatives()`` again after ``set_param``.
    """

    def bkd(self) -> Backend[Array]: ...

    def __call__(self, state: Array, time: float) -> Array: ...

    def derivatives(self) -> ODEResidualDerivatives[Array, FormatProtocol]: ...

    def set_param(self, param: Array) -> None:
        """Set the parameters and increment the parameter version."""
        ...

    def parameters_version(self) -> int:
        """The number of ``set_param`` calls so far."""
        ...
