"""Quadrature mixin for time-consistent quadrature rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from pyapprox.ode.time_quadrature import TrajectoryQuadratureProtocol
from pyapprox.util.backends.protocols import Array, Backend


class QuadratureMixin(ABC, Generic[Array]):
    """Mixin providing trajectory_quadrature.

    Subclasses override trajectory_quadrature to return the rule
    object matching their time discretization (typically one of the
    constructors in :mod:`pyapprox.ode.time_quadrature`).
    """

    if TYPE_CHECKING:
        _bkd: Backend[Array]

    @abstractmethod
    def trajectory_quadrature(
        self, times: Array
    ) -> TrajectoryQuadratureProtocol[Array]:
        """Return the scheme-implied quadrature over a stored trajectory.

        Parameters
        ----------
        times : Array
            Time nodes of the solve. Shape: (ntimes,)

        Returns
        -------
        TrajectoryQuadratureProtocol
            The rule mapped onto stored trajectory columns.
        """
        ...
