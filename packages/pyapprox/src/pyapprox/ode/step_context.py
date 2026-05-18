"""Immutable step context for time-stepper methods."""

from dataclasses import dataclass
from typing import Generic

from pyapprox.util.backends.protocols import Array


@dataclass(frozen=True)
class StepContext(Generic[Array]):
    """Describes residual R_k: y_k computed from y_{k-1} over (t_{k-1}, t_k].

    Parameters
    ----------
    t_prev : float
        Time at start of step (t_{k-1}).
    deltat : float
        Time step size (t_k - t_{k-1}).
    y_prev : Array
        State at previous time step y_{k-1}. Shape: (nstates,)
    """

    t_prev: float
    deltat: float
    y_prev: Array

    @property
    def t_curr(self) -> float:
        """Time at end of step: t_{k-1} + dt."""
        return self.t_prev + self.deltat
