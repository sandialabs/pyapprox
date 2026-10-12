"""Frozen bundle of a steady functional's optional derivatives.

A steady functional ``Q(u, p)`` always provides its value and its state
and parameter Jacobians. What it may lack is listed here, so a consumer
reads capability from data rather than from which protocol the
functional satisfies (``docs/conventions/optional_methods.md``). Absence
is ``None``.

Field signatures (``state`` ``(nstates, 1)``; ``param`` ``(nparams, 1)``):

- ``state_state_hvp``: ``(state, param, wvec (nstates, 1))
  -> (nstates, 1)`` — ``(d^2 Q/du^2) w``.
- ``state_param_hvp``: ``(state, param, vvec (nparams, 1))
  -> (nstates, 1)`` — ``(d^2 Q/du dp) v``.
- ``param_state_hvp``: ``(state, param, wvec (nstates, 1))
  -> (nparams, 1)`` — ``(d^2 Q/dp du) w``.
- ``param_param_hvp``: ``(state, param, vvec (nparams, 1))
  -> (nparams, 1)`` — ``(d^2 Q/dp^2) v``.

The four come together or not at all: the second-order adjoint needs
every one, so a partial set is not a capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from pyapprox.util.backends.protocols import Array, ArrayProtocol

A = TypeVar("A", bound=ArrayProtocol)

SteadyFunctionalHVPFn = Callable[[Array, Array, Array], Array]

_HVP_FIELDS = (
    "state_state_hvp",
    "state_param_hvp",
    "param_state_hvp",
    "param_param_hvp",
)


@dataclass(frozen=True)
class SteadyFunctionalDerivatives(Generic[Array]):
    """Optional derivatives of a steady functional.

    Fields hold callables, typically bound methods of the functional.
    Bundles are frozen: a functional whose capability changes builds a
    new bundle.
    """

    state_state_hvp: Optional[SteadyFunctionalHVPFn[Array]] = None
    state_param_hvp: Optional[SteadyFunctionalHVPFn[Array]] = None
    param_state_hvp: Optional[SteadyFunctionalHVPFn[Array]] = None
    param_param_hvp: Optional[SteadyFunctionalHVPFn[Array]] = None

    def __post_init__(self) -> None:
        for name in _HVP_FIELDS:
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"SteadyFunctionalDerivatives field '{name}' must be "
                    f"callable or None; got {type(value).__name__}"
                )
        present = [getattr(self, name) is not None for name in _HVP_FIELDS]
        if any(present) and not all(present):
            raise TypeError(
                "the four HVP fields must be given together or not at "
                "all; the second-order adjoint needs every one"
            )

    @staticmethod
    def none() -> "SteadyFunctionalDerivatives[A]":
        """No optional derivatives."""
        return SteadyFunctionalDerivatives()

    @staticmethod
    def second_order(
        state_state_hvp: SteadyFunctionalHVPFn[A],
        state_param_hvp: SteadyFunctionalHVPFn[A],
        param_state_hvp: SteadyFunctionalHVPFn[A],
        param_param_hvp: SteadyFunctionalHVPFn[A],
    ) -> "SteadyFunctionalDerivatives[A]":
        """All four HVPs."""
        return SteadyFunctionalDerivatives(
            state_state_hvp=state_state_hvp,
            state_param_hvp=state_param_hvp,
            param_state_hvp=param_state_hvp,
            param_param_hvp=param_param_hvp,
        )
