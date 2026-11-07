"""Frozen bundle of an implicit-function operator's optional derivatives.

An operator pairing a state equation with a functional always provides
the functional value and its parameter Jacobian. Whether it also provides
the second-order adjoint Hessian-vector product depends on what its
state equation and functional provide, so it is listed here and read from
data rather than from the operator's class
(``docs/conventions/optional_methods.md``). Absence is ``None``.

Field signature (``init_fwd_state`` ``(nstates, 1)``, the initial guess
of the state solve; ``param`` ``(nparams, 1)``):

- ``hvp``: ``(init_fwd_state, param, vvec (nparams, 1)) -> (nparams, 1)``,
  the Hessian of the scalar functional of the solved state with respect
  to the parameters, applied to ``vvec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from pyapprox.util.backends.protocols import Array, ArrayProtocol

A = TypeVar("A", bound=ArrayProtocol)

OperatorHVPFn = Callable[[Array, Array, Array], Array]


@dataclass(frozen=True)
class ImplicitOperatorDerivatives(Generic[Array]):
    """Optional derivatives of an implicit-function operator.

    Fields hold callables, typically bound methods of the operator.
    """

    hvp: Optional[OperatorHVPFn[Array]] = None

    def __post_init__(self) -> None:
        if self.hvp is not None and not callable(self.hvp):
            raise TypeError(
                "ImplicitOperatorDerivatives field 'hvp' must be callable "
                f"or None; got {type(self.hvp).__name__}"
            )

    @staticmethod
    def none() -> "ImplicitOperatorDerivatives[A]":
        """No optional derivatives."""
        return ImplicitOperatorDerivatives()

    @staticmethod
    def second_order(
        hvp: OperatorHVPFn[A],
    ) -> "ImplicitOperatorDerivatives[A]":
        """The second-order adjoint Hessian-vector product."""
        return ImplicitOperatorDerivatives(hvp=hvp)
