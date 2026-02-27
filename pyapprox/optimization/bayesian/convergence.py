"""Convergence criteria for Bayesian optimization.

Provides a protocol and concrete implementations for early stopping
when the optimizer has converged.
"""

from typing import Generic, Protocol, runtime_checkable

import numpy as np

from pyapprox.optimization.bayesian.state import ConvergenceContext
from pyapprox.util.backends.protocols import Array


@runtime_checkable
class ConvergenceCriterionProtocol(Protocol, Generic[Array]):
    """Protocol for BO convergence criteria."""

    def has_converged(self, ctx: ConvergenceContext[Array]) -> bool:
        """Return True if optimization should stop.

        Parameters
        ----------
        ctx : ConvergenceContext[Array]
            Context with current and previous best results.

        Returns
        -------
        bool
            True if optimization has converged.
        """
        ...


class ValueToleranceCriterion(Generic[Array]):
    """Stop when recommended_y changes by less than atol.

    Triggers when ``|recommended_y_new - recommended_y_prev| < atol``
    for ``patience`` consecutive steps.

    Parameters
    ----------
    atol : float
        Absolute tolerance for value change. Default 1e-6.
    patience : int
        Number of consecutive steps below tolerance required. Default 3.
    """

    def __init__(self, atol: float = 1e-6, patience: int = 3) -> None:
        self._atol = atol
        self._patience = patience
        self._count = 0

    def has_converged(self, ctx: ConvergenceContext[Array]) -> bool:
        """Return True if value change is below tolerance for patience steps."""
        if ctx.prev_best is None:
            self._count = 0
            return False

        curr_y = float(np.asarray(ctx.best.recommended_y).flat[0])
        prev_y = float(np.asarray(ctx.prev_best.recommended_y).flat[0])

        if abs(curr_y - prev_y) < self._atol:
            self._count += 1
        else:
            self._count = 0

        return self._count >= self._patience


class AcquisitionToleranceCriterion(Generic[Array]):
    """Stop when the maximum acquisition value is below atol.

    Parameters
    ----------
    atol : float
        Absolute tolerance. Default 1e-8.
    """

    def __init__(self, atol: float = 1e-8) -> None:
        self._atol = atol

    def has_converged(self, ctx: ConvergenceContext[Array]) -> bool:
        """Return True if max acquisition value is below tolerance."""
        if ctx.max_acquisition_value is None:
            return False
        return ctx.max_acquisition_value < self._atol


class DistanceToleranceCriterion(Generic[Array]):
    """Stop when recommended_x moves by less than atol.

    Triggers when ``||recommended_x_new - recommended_x_prev|| < atol``
    for ``patience`` consecutive steps.

    Parameters
    ----------
    atol : float
        Absolute tolerance for distance. Default 1e-6.
    patience : int
        Number of consecutive steps below tolerance required. Default 3.
    """

    def __init__(self, atol: float = 1e-6, patience: int = 3) -> None:
        self._atol = atol
        self._patience = patience
        self._count = 0

    def has_converged(self, ctx: ConvergenceContext[Array]) -> bool:
        """Return True if point movement is below tolerance for patience steps."""
        if ctx.prev_best is None:
            self._count = 0
            return False

        curr_x = np.asarray(ctx.best.recommended_x).flatten()
        prev_x = np.asarray(ctx.prev_best.recommended_x).flatten()
        dist = float(np.linalg.norm(curr_x - prev_x))

        if dist < self._atol:
            self._count += 1
        else:
            self._count = 0

        return self._count >= self._patience
