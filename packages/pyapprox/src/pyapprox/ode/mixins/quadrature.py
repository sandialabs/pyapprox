"""Quadrature mixin for time-consistent quadrature rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class QuadratureMixin(ABC, Generic[Array]):
    """Mixin providing quadrature_samples_weights.

    Subclasses override quadrature_samples_weights to return the
    quadrature rule matching their time discretization.
    """

    if TYPE_CHECKING:
        _bkd: Backend[Array]

    @abstractmethod
    def quadrature_samples_weights(
        self, times: Array
    ) -> Tuple[Array, Array]:
        """Compute quadrature rule consistent with time discretization.

        Parameters
        ----------
        times : Array
            Time nodes. Shape: (ntimes,)

        Returns
        -------
        quadx : Array
            Quadrature sample points. Shape depends on rule:
            (ntimes-1,) for constant rules, (ntimes,) for linear.
        quadw : Array
            Quadrature weights. Shape: (ntimes-1,) for constant
            rules, (ntimes,) for linear (trapezoidal).
        """
        ...
