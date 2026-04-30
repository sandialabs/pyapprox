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
    from pyapprox.util.hyperparameter import HyperParameterList


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
class ParameterizedVFProtocol(Protocol, Generic[Array]):
    """Protocol for vector fields with optimizable parameters."""

    def __call__(self, vf_input: Array) -> Array: ...

    def hyp_list(self) -> "HyperParameterList[Array]": ...

    def sync_params(self) -> None: ...


@runtime_checkable
class DifferentiableVFProtocol(ParameterizedVFProtocol[Array], Protocol):
    """Parameterized VF that also provides analytical parameter jacobians."""

    def jacobian_wrt_params(self, vf_input: Array) -> Array:
        """Jacobian of output w.r.t. active parameters.

        Parameters
        ----------
        vf_input : Array
            Shape ``(nvars_in, ns)``.

        Returns
        -------
        Array
            Shape ``(ns, nqoi, nactive)``.
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
        quad_data: FlowMatchingQuadData[Array],
        time_weight: Optional[TimeWeightProtocol[Array]] = None,
    ) -> object:
        """Fit a vector field to minimize the CFM loss.

        Parameters
        ----------
        vf : object
            Vector field (e.g. BasisExpansion) to fit. Deep-cloned internally.
        path : ProbabilityPathProtocol[Array]
            Probability path defining interpolation.
        quad_data : FlowMatchingQuadData[Array]
            Pre-assembled quadrature data.
        time_weight : TimeWeightProtocol[Array], optional
            Time-dependent weight. Defaults to uniform.

        Returns
        -------
        object
            Fit result containing the fitted surrogate.
        """
        ...
