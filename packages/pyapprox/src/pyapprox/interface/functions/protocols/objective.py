"""Optimizer-facing protocols built on the Derivatives bundle.

Structural: users implement these without inheriting anything. The only
capability surface is ``derivatives()``; consumers decide off the bundle
and NEVER probe attributes with ``hasattr`` or catch
``NotImplementedError`` around the :class:`Function` sugar.

Objectives and constraints differ deliberately:

- constraints carry bounds (``lb``/``ub``); objectives do not
- objectives must have ``nqoi() == 1`` (enforced by validation, not the
  type); constraints are legitimately vector-valued
- consumers read objectives via ``Derivatives.resolved_hvp`` and
  constraints via ``Derivatives.resolved_whvp`` (multiplier-weighted
  adjoint Hessian), always passing the OWNING object's ``nqoi()``

Typing note: these protocols use the invariant ``Array`` type variable,
which is correct because Array appears in both argument and return
positions. A future protocol whose methods are genuinely return-only must
declare a COVARIANT type variable instead — never add a dummy argument to
silence mypy's variance check.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Protocol, runtime_checkable

from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.interface.functions.protocols.function import FunctionProtocol
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ObjectiveProtocol(FunctionProtocol[Array], Protocol):
    """A function an optimizer can minimize: evaluation plus a bundle."""

    def derivatives(self) -> Derivatives[Array]:
        """Return this objective's derivative capabilities.

        The bundle's ``jacobian`` differentiates w.r.t. THIS function's
        input (shape contracts in
        :mod:`pyapprox.interface.functions.derivatives`).
        """
        ...


class Function(ABC, Generic[Array]):
    """OPTIONAL human-facing sugar; never a consumer surface.

    Provides delegating ``jacobian``/``hvp`` that raise
    ``NotImplementedError`` when the capability is absent — convenient at
    a REPL. Framework consumers must decide off ``derivatives()`` and
    never try/except around this sugar (enforced by a repo grep).

    Subclasses that implement real derivative methods should point their
    bundle at those implementations directly — never at this class's
    delegating methods (that would recurse).
    """

    @abstractmethod
    def bkd(self) -> Backend[Array]: ...

    @abstractmethod
    def nvars(self) -> int: ...

    @abstractmethod
    def nqoi(self) -> int: ...

    @abstractmethod
    def __call__(self, samples: Array) -> Array: ...

    def derivatives(self) -> Derivatives[Array]:
        """Default: no capabilities. Override to declare them."""
        empty: Derivatives[Array] = Derivatives.none()
        return empty

    def jacobian(self, sample: Array) -> Array:
        jacobian = self.derivatives().jacobian
        if jacobian is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not provide a jacobian"
            )
        return jacobian(sample)

    def hvp(self, sample: Array, vec: Array) -> Array:
        hvp = self.derivatives().resolved_hvp(self.nqoi(), self.bkd())
        if hvp is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not provide an hvp"
            )
        return hvp(sample, vec)
