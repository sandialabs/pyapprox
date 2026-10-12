"""Frozen bundle of a steady state equation's optional derivatives.

A state equation ``R(u, p) = 0`` always provides its residual, its solve
and its state and parameter Jacobians. What it may lack is listed here,
so a consumer reads capability from data rather than from which protocol
the state equation satisfies (``docs/conventions/optional_methods.md``).
Absence is ``None``.

Field signatures (``state`` ``(nstates, 1)``; ``param`` ``(nparams, 1)``;
``adj_state`` ``(nstates, 1)``), each the adjoint-weighted second
derivative ``lambda^T (d^2 R) vec``:

- ``state_state_hvp``: ``(state, param, adj_state, wvec (nstates, 1))
  -> (nstates, 1)``.
- ``state_param_hvp``: ``(state, param, adj_state, vvec (nparams, 1))
  -> (nstates, 1)``.
- ``param_state_hvp``: ``(state, param, adj_state, wvec (nstates, 1))
  -> (nparams, 1)``.
- ``param_param_hvp``: ``(state, param, adj_state, vvec (nparams, 1))
  -> (nparams, 1)``.

The four come together or not at all: the second-order adjoint needs
every one, so a partial set is not a capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from pyapprox.util.backends.protocols import Array, ArrayProtocol

A = TypeVar("A", bound=ArrayProtocol)

StateEquationHVPFn = Callable[[Array, Array, Array, Array], Array]

_HVP_FIELDS = (
    "state_state_hvp",
    "state_param_hvp",
    "param_state_hvp",
    "param_param_hvp",
)


@dataclass(frozen=True)
class SteadyStateEquationDerivatives(Generic[Array]):
    """Optional derivatives of a steady state equation.

    Fields hold callables, typically bound methods of the state
    equation. Bundles are frozen: a state equation whose capability
    changes builds a new bundle.
    """

    state_state_hvp: Optional[StateEquationHVPFn[Array]] = None
    state_param_hvp: Optional[StateEquationHVPFn[Array]] = None
    param_state_hvp: Optional[StateEquationHVPFn[Array]] = None
    param_param_hvp: Optional[StateEquationHVPFn[Array]] = None

    def __post_init__(self) -> None:
        for name in _HVP_FIELDS:
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"SteadyStateEquationDerivatives field '{name}' must be "
                    f"callable or None; got {type(value).__name__}"
                )
        present = [getattr(self, name) is not None for name in _HVP_FIELDS]
        if any(present) and not all(present):
            raise TypeError(
                "the four HVP fields must be given together or not at "
                "all; the second-order adjoint needs every one"
            )

    @staticmethod
    def none() -> "SteadyStateEquationDerivatives[A]":
        """No optional derivatives."""
        return SteadyStateEquationDerivatives()

    @staticmethod
    def second_order(
        state_state_hvp: StateEquationHVPFn[A],
        state_param_hvp: StateEquationHVPFn[A],
        param_state_hvp: StateEquationHVPFn[A],
        param_param_hvp: StateEquationHVPFn[A],
    ) -> "SteadyStateEquationDerivatives[A]":
        """All four adjoint-weighted HVPs."""
        return SteadyStateEquationDerivatives(
            state_state_hvp=state_state_hvp,
            state_param_hvp=state_param_hvp,
            param_state_hvp=param_state_hvp,
            param_param_hvp=param_param_hvp,
        )
