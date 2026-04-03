"""Protocols for quadrature rules.

This module defines protocols for:
- UnivariateQuadratureRuleProtocol: 1D quadrature rules
- MultivariateQuadratureRuleProtocol: Fixed multivariate rules
- ParameterizedQuadratureRuleProtocol: Rules parameterized by sample count
- AdaptiveQuadratureRuleProtocol: Adaptive quadrature with error estimation
"""

from typing import (
    Callable,
    Generic,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class UnivariateQuadratureRuleProtocol(Protocol, Generic[Array]):
    """Protocol for univariate quadrature rules.

    A univariate quadrature rule provides samples and weights for
    numerical integration in one dimension.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def __call__(self, npoints: int) -> Tuple[Array, Array]:
        """Generate quadrature rule with specified number of points.

        Parameters
        ----------
        npoints : int
            Number of quadrature points.

        Returns
        -------
        Tuple[Array, Array]
            (samples, weights) with shapes (npoints,) and (npoints,)
        """
        ...


@runtime_checkable
class MultivariateQuadratureRuleProtocol(Protocol, Generic[Array]):
    """Protocol for multivariate quadrature rules (fixed sample count).

    A multivariate quadrature rule provides pre-computed samples and
    weights for numerical integration in multiple dimensions.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of variables."""
        ...

    def nsamples(self) -> int:
        """Return the number of samples."""
        ...

    def __call__(self) -> Tuple[Array, Array]:
        """Return quadrature samples and weights.

        Returns
        -------
        Tuple[Array, Array]
            (samples, weights) with shapes (nvars, nsamples) and (nsamples,)
        """
        ...

    def integrate(self, func: Callable[[Array], Array]) -> Array:
        """Integrate a function using this quadrature rule.

        Parameters
        ----------
        func : Callable[[Array], Array]
            Function to integrate. Takes samples (nvars, nsamples) and
            returns values (nsamples, nqoi).

        Returns
        -------
        Array
            Integral estimate of shape (nqoi,)
        """
        ...


@runtime_checkable
class ParameterizedQuadratureRuleProtocol(Protocol, Generic[Array]):
    """Protocol for quadrature rules parameterized by sample count or level.

    These rules can generate quadrature points for different accuracy
    levels on demand.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of variables."""
        ...

    def __call__(self, level: int) -> Tuple[Array, Array]:
        """Generate quadrature rule for given level.

        Parameters
        ----------
        level : int
            Quadrature level (higher = more accurate).

        Returns
        -------
        Tuple[Array, Array]
            (samples, weights) with shapes (nvars, nsamples) and (nsamples,)
        """
        ...

    def integrate(
        self,
        func: Callable[[Array], Array],
        level: int,
    ) -> Array:
        """Integrate a function at given quadrature level.

        Parameters
        ----------
        func : Callable[[Array], Array]
            Function to integrate.
        level : int
            Quadrature level.

        Returns
        -------
        Array
            Integral estimate of shape (nqoi,)
        """
        ...


@runtime_checkable
class AdaptiveQuadratureRuleProtocol(Protocol, Generic[Array]):
    """Protocol for adaptive quadrature (unknown sample count a priori).

    Adaptive quadrature rules iteratively refine the integration
    estimate until a tolerance is met.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of variables."""
        ...

    def step_samples(self) -> Optional[Array]:
        """Get samples for next refinement step.

        Returns
        -------
        Optional[Array]
            New samples of shape (nvars, nnew) or None if converged
        """
        ...

    def step_values(self, values: Array) -> None:
        """Provide function values for the samples from step_samples.

        Parameters
        ----------
        values : Array
            Values of shape (nnew, nqoi)
        """
        ...

    def error_estimate(self) -> float:
        """Return current error estimate.

        Returns
        -------
        float
            Estimated integration error
        """
        ...

    def integral_estimate(self) -> Array:
        """Return current integral estimate.

        Returns
        -------
        Array
            Current integral estimate of shape (nqoi,)
        """
        ...

    def nsamples_used(self) -> int:
        """Return total number of samples used so far."""
        ...

    def integrate(
        self,
        func: Callable[[Array], Array],
        tolerance: float,
        max_samples: int = 10000,
    ) -> Tuple[Array, float, int]:
        """Integrate function adaptively until tolerance is met.

        Parameters
        ----------
        func : Callable[[Array], Array]
            Function to integrate.
        tolerance : float
            Target error tolerance.
        max_samples : int, optional
            Maximum number of samples.

        Returns
        -------
        Tuple[Array, float, int]
            (integral, error_estimate, nsamples_used)
        """
        ...
