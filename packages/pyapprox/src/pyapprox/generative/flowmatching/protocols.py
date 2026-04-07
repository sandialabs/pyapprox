"""Protocols for the flow matching module."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Generic,
    Optional,
    Protocol,
    runtime_checkable,
)

from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.generative.flowmatching.quad_data import (
        FlowMatchingQuadData,
    )


@runtime_checkable
class ProbabilityPathProtocol(Protocol, Generic[Array]):
    """Protocol for probability paths interpolating between distributions.

    A probability path defines how to interpolate between a source sample
    x0 and a target sample x1 as a function of time t in [0, 1].
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def alpha(self, t: Array) -> Array:
        """Signal coefficient. t: (1, ns) -> (1, ns)."""
        ...

    def sigma(self, t: Array) -> Array:
        """Noise coefficient. t: (1, ns) -> (1, ns)."""
        ...

    def d_alpha(self, t: Array) -> Array:
        """Time derivative of alpha. t: (1, ns) -> (1, ns)."""
        ...

    def d_sigma(self, t: Array) -> Array:
        """Time derivative of sigma. t: (1, ns) -> (1, ns)."""
        ...

    def interpolate(self, t: Array, x0: Array, x1: Array) -> Array:
        """Interpolate between x0 and x1 at time t.

        Parameters
        ----------
        t : Array
            Time values, shape ``(1, ns)``.
        x0 : Array
            Source samples, shape ``(d, ns)``.
        x1 : Array
            Target samples, shape ``(d, ns)``.

        Returns
        -------
        Array
            Interpolated samples, shape ``(d, ns)``.
        """
        ...

    def target_field(self, t: Array, x0: Array, x1: Array) -> Array:
        """Conditional target vector field u_t(x | x0, x1).

        Parameters
        ----------
        t : Array
            Time values, shape ``(1, ns)``.
        x0 : Array
            Source samples, shape ``(d, ns)``.
        x1 : Array
            Target samples, shape ``(d, ns)``.

        Returns
        -------
        Array
            Target vector field, shape ``(d, ns)``.
        """
        ...


@runtime_checkable
class TimeWeightProtocol(Protocol, Generic[Array]):
    """Protocol for time-dependent weighting in the CFM loss."""

    def __call__(self, t: Array) -> Array:
        """Evaluate time weight. t: (1, ns) -> (1, ns)."""
        ...


@runtime_checkable
class CFMLossProtocol(Protocol, Generic[Array]):
    """Protocol for the conditional flow matching loss."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def integrand(
        self,
        vf: object,
        path: ProbabilityPathProtocol[Array],
        t: Array,
        x0: Array,
        x1: Array,
        c: Optional[Array] = None,
    ) -> Array:
        """Pointwise loss integrand.

        Returns
        -------
        Array
            Per-sample loss values, shape ``(ns,)``.
        """
        ...

    def __call__(
        self,
        vf: object,
        path: ProbabilityPathProtocol[Array],
        t: Array,
        x0: Array,
        x1: Array,
        weights: Array,
        c: Optional[Array] = None,
    ) -> Array:
        """Weighted loss (quadrature approximation to the integral).

        Returns
        -------
        Array
            Scalar loss value.
        """
        ...


@runtime_checkable
class FlowMatchingFitterProtocol(Protocol, Generic[Array]):
    """Protocol for flow matching fitters."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def fit(
        self,
        vf: object,
        path: ProbabilityPathProtocol[Array],
        loss: CFMLossProtocol[Array],
        quad_data: FlowMatchingQuadData[Array],
    ) -> object:
        """Fit a vector field to minimize the CFM loss.

        Parameters
        ----------
        vf : object
            Vector field (e.g. BasisExpansion) to fit. Deep-cloned internally.
        path : ProbabilityPathProtocol[Array]
            Probability path defining interpolation.
        loss : CFMLossProtocol[Array]
            CFM loss function.
        quad_data : FlowMatchingQuadData[Array]
            Pre-assembled quadrature data.

        Returns
        -------
        object
            Fit result containing the fitted surrogate.
        """
        ...
