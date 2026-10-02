"""Frozen bundle of a spatial operator's optional state derivatives.

The first state derivative of ``F`` (its Jacobian) is required of every
spatial operator; second derivatives are optional, so they travel in a
bundle exposed through a ``state_derivatives()`` accessor. Absence of a
capability is ``None``, never a missing attribute, and consumers choose
their derivative tier by None-checking the bundle once rather than by
``isinstance`` on the operator: one operator class built from parts
(an interior and boundary terms) cannot conditionally have a method,
but it can return a bundle whose fields depend on its parts.

Field signatures:

- ``state_state_hvp``: ``(state, adj_state, wvec, time) -> (nstates,)``
  --- ``lambda^T (d^2 F/du^2) w`` of the raw operator (no constraint
  handling). Zero must be declared (``StateDerivatives.linear``), never
  assumed from absence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend

StateStateHVPFn = Callable[[Array, Array, Array, float], Array]


class _ZeroStateStateHVP(Generic[Array]):
    """The exact zero curvature of an operator linear in the state.

    Module-level, so a linear operator's bundle pickles.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        return self._bkd.zeros((state.shape[0],))


@dataclass(frozen=True)
class StateDerivatives(Generic[Array]):
    """Bundle of optional second state derivatives of ``F``.

    Build with the named constructors. Bundles are frozen: a producer
    whose capability changes rebuilds its bundle, never mutates one.
    """

    state_state_hvp: Optional[StateStateHVPFn[Array]] = None

    @staticmethod
    def none() -> "StateDerivatives[Array]":
        """No second derivatives: consumers stay at the Jacobian tier."""
        return StateDerivatives()

    @staticmethod
    def second_order(
        state_state_hvp: StateStateHVPFn[Array],
    ) -> "StateDerivatives[Array]":
        """Second derivatives supplied as ``state_state_hvp``."""
        return StateDerivatives(state_state_hvp=state_state_hvp)

    @staticmethod
    def linear(bkd: Backend[Array]) -> "StateDerivatives[Array]":
        """Exact zero curvature, declared for an operator linear in u."""
        return StateDerivatives(state_state_hvp=_ZeroStateStateHVP(bkd))
