"""Quadrature mixin for time-consistent quadrature rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class QuadratureMixin(ABC, Generic[Array]):
    """Mixin providing quadrature_samples_weights via template method.

    Subclasses override _get_quadrature_class() to specify their
    quadrature type.
    """

    if TYPE_CHECKING:
        _bkd: Backend[Array]

    @abstractmethod
    def _get_quadrature_class(self) -> type:
        """Return the quadrature class for this time stepper."""
        ...

    def quadrature_samples_weights(self, times: Array) -> Tuple[Array, Array]:
        """Compute quadrature rule consistent with time discretization."""
        quadrature_class = self._get_quadrature_class()
        quadrature = quadrature_class(times, self._bkd)
        quadx, quadw = quadrature.quadrature_rule()
        if quadw.ndim > 1:
            quadw = quadw[:, 0]
        return quadx, quadw
