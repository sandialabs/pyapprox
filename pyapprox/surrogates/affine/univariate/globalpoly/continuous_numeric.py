"""Continuous numeric orthonormal polynomials for arbitrary measures.

This module provides orthonormal polynomials for arbitrary continuous
probability measures using the predictor-corrector method to compute
recursion coefficients via numerical integration.

Classes
-------
GaussLegendreIntegrator : Bounded domain integrator using Gauss-Legendre
UnboundedIntegrator : Unbounded domain integrator using interval expansion
PredictorCorrector : Computes recursion coefficients via modified moments
ContinuousNumericOrthonormalPolynomial1D : Orthonormal polynomials for
    arbitrary continuous marginals

Notes
-----
This implementation supports continuous marginals only. For discrete
marginals, use DiscreteNumericOrthonormalPolynomial1D with the Lanczos
algorithm instead.
"""

import math
from typing import Any, Callable, Generic, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from pyapprox.surrogates.affine.univariate.globalpoly.jacobi import (
    LegendrePolynomial1D,
)
from pyapprox.surrogates.affine.univariate.globalpoly.orthopoly_base import (
    OrthonormalPolynomial1D,
    evaluate_orthonormal_polynomial_1d,
)
from pyapprox.surrogates.affine.univariate.globalpoly.quadrature import (
    GaussQuadratureRule,
)
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class IntegratorProtocol(Protocol, Generic[Array]):
    """Protocol for numerical integrators."""

    def set_bounds(self, bounds: Tuple[float, float]) -> None:
        """Set integration bounds."""
        ...

    def set_integrand(self, integrand: Callable[[Array], Array]) -> None:
        """Set the integrand function."""
        ...

    def __call__(self) -> float:
        """Compute the integral."""
        ...


class GaussLegendreIntegrator(Generic[Array]):
    """Gauss-Legendre quadrature integrator for bounded domains.

    Computes definite integrals using Gauss-Legendre quadrature on
    arbitrary finite intervals [a, b].

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    npoints : int
        Number of quadrature points. Default: 100.
    """

    def __init__(self, bkd: Backend[Array], npoints: int = 100):
        self._bkd = bkd
        self._npoints = npoints
        self._bounds: Tuple[float, float] = (-1.0, 1.0)
        self._integrand: Optional[Callable[[Array], Array]] = None

        # Setup Legendre polynomial and quadrature rule
        self._legendre = LegendrePolynomial1D(bkd)
        self._legendre.set_nterms(npoints)
        self._quad_rule = GaussQuadratureRule(self._legendre, store=True)

    def set_bounds(self, bounds: Tuple[float, float]) -> None:
        """Set integration bounds [a, b]."""
        self._bounds = bounds

    def set_integrand(self, integrand: Callable[[Array], Array]) -> None:
        """Set the integrand function.

        Parameters
        ----------
        integrand : Callable[[Array], Array]
            Function that takes samples of shape (1, nsamples) and
            returns values of shape (nsamples, 1).
        """
        self._integrand = integrand

    def __call__(self) -> float:
        """Compute the integral.

        Returns
        -------
        float
            The value of the integral.

        Notes
        -----
        Uses Gauss-Legendre quadrature. The Legendre polynomial uses
        probability measure weights (sum to 1), so for Lebesgue integration
        on [-1, 1], we multiply by 2. Combined with change of variables
        Jacobian, the total factor is (b - a).
        """
        if self._integrand is None:
            raise ValueError("Must set integrand before calling")

        # Get Gauss-Legendre points and weights on [-1, 1]
        points, weights = self._quad_rule(self._npoints)
        # points: (1, npoints), weights: (npoints, 1)
        # weights sum to 1 (probability measure)

        # Map from [-1, 1] to [a, b]
        a, b = self._bounds
        # x = (b - a) / 2 * t + (a + b) / 2
        mapped_points = (b - a) / 2.0 * points + (a + b) / 2.0

        # For Lebesgue integral:
        # ∫_{a}^{b} f(x) dx = ∫_{-1}^{1} f(x(t)) * (b-a)/2 dt
        # Using probability measure weights (sum to 1):
        # ∫_{-1}^{1} g(t) dt = 2 * Σ w_i g(t_i)
        # So: ∫_{a}^{b} f(x) dx = (b-a) * Σ w_i f(x(t_i))
        jacobian = b - a

        # Evaluate integrand at mapped points
        vals = self._integrand(mapped_points)  # (npoints, 1)

        # Compute weighted sum
        result = self._bkd.sum(vals[:, 0] * weights[:, 0]) * jacobian
        return self._bkd.to_float(result)


class UnboundedIntegrator(Generic[Array]):
    """Integrator for unbounded domains using interval expansion.

    Computes integrals over unbounded domains by expanding outward from
    a starting point until the integral contribution drops below tolerance.
    Assumes the integrand decays towards infinity.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    npoints : int
        Number of quadrature points per interval. Default: 50.
    interval_size : float
        Size of each integration interval. Default: 2.0.
    atol : float
        Absolute tolerance for convergence. Default: 1e-8.
    rtol : float
        Relative tolerance for convergence. Default: 1e-8.
    maxiters : int
        Maximum number of interval expansions per direction. Default: 1000.
    adaptive : bool
        Whether to use adaptive refinement. Default: True.
    maxinner_iters : int
        Maximum refinement iterations per interval. Default: 10.
    """

    def __init__(
        self,
        bkd: Backend[Array],
        npoints: int = 50,
        interval_size: float = 2.0,
        atol: float = 1e-8,
        rtol: float = 1e-8,
        maxiters: int = 1000,
        adaptive: bool = True,
        maxinner_iters: int = 10,
    ):
        self._bkd = bkd
        self._npoints = npoints
        self._interval_size = interval_size
        self._atol = atol
        self._rtol = rtol
        self._maxiters = maxiters
        self._adaptive = adaptive
        self._maxinner_iters = maxinner_iters
        self._bounds: Tuple[float, float] = (-np.inf, np.inf)
        self._integrand: Optional[Callable[[Array], Array]] = None

        # Setup Legendre quadrature for each interval
        self._legendre = LegendrePolynomial1D(bkd)
        self._legendre.set_nterms(npoints * 2 + 1)
        self._quad_rule = GaussQuadratureRule(self._legendre, store=True)

    def set_bounds(self, bounds: Tuple[float, float]) -> None:
        """Set integration bounds.

        At least one bound must be infinite.
        """
        lb, ub = bounds
        if np.isfinite(lb) and np.isfinite(ub):
            raise ValueError(
                "UnboundedIntegrator requires at least one infinite bound. "
                "Use GaussLegendreIntegrator for bounded domains."
            )
        self._bounds = bounds

    def set_integrand(self, integrand: Callable[[Array], Array]) -> None:
        """Set the integrand function."""
        self._integrand = integrand

    def _integrate_interval(self, lb: float, ub: float, npoints: int) -> float:
        """Integrate over a single interval [lb, ub]."""
        points, weights = self._quad_rule(npoints)
        # Map from [-1, 1] to [lb, ub]
        mapped_points = (ub - lb) / 2.0 * points + (lb + ub) / 2.0
        jacobian = ub - lb

        vals = self._integrand(mapped_points)
        return self._bkd.to_float(self._bkd.sum(vals[:, 0] * weights[:, 0]) * jacobian)

    def _adaptive_integrate_interval(self, lb: float, ub: float) -> float:
        """Adaptively integrate over an interval with refinement."""
        npoints = self._npoints
        integral = self._integrate_interval(lb, ub, npoints)

        if not self._adaptive:
            return integral

        for _ in range(self._maxinner_iters):
            npoints = (npoints - 1) * 2 + 1
            prev_integral = integral
            integral = self._integrate_interval(lb, ub, npoints)

            diff = abs(integral - prev_integral)
            if diff < self._rtol * abs(integral) + self._atol:
                break

        return integral

    def _initial_interval_bounds(self) -> Tuple[float, float]:
        """Determine initial integration interval."""
        lb, ub = self._bounds

        if np.isfinite(lb) and not np.isfinite(ub):
            # Semi-infinite [lb, inf)
            return (lb, lb + self._interval_size)
        elif not np.isfinite(lb) and np.isfinite(ub):
            # Semi-infinite (-inf, ub]
            return (ub - self._interval_size, ub)
        else:
            # Doubly infinite (-inf, inf)
            return (-self._interval_size / 2, self._interval_size / 2)

    def _left_integrate(self, lb: float, ub: float) -> float:
        """Integrate leftward from initial interval."""
        integral = 0.0
        prev_integral = np.inf
        it = 0

        while (
            abs(integral - prev_integral)
            >= self._rtol * abs(prev_integral) + self._atol
            and lb >= self._bounds[0]
            and it < self._maxiters
        ):
            result = self._adaptive_integrate_interval(lb, ub)
            prev_integral = integral if it > 0 else np.inf
            integral += result
            ub = lb
            lb -= self._interval_size
            it += 1

        return integral

    def _right_integrate(self, lb: float, ub: float) -> float:
        """Integrate rightward from initial interval."""
        integral = 0.0
        prev_integral = np.inf
        it = 0

        while (
            abs(integral - prev_integral)
            >= self._rtol * abs(prev_integral) + self._atol
            and ub <= self._bounds[1]
            and it < self._maxiters
        ):
            result = self._adaptive_integrate_interval(lb, ub)
            prev_integral = integral if it > 0 else np.inf
            integral += result
            lb = ub
            ub += self._interval_size
            it += 1

        return integral

    def __call__(self) -> float:
        """Compute the integral over unbounded domain."""
        if self._integrand is None:
            raise ValueError("Must set integrand before calling")

        lb, ub = self._initial_interval_bounds()

        # Integrate leftward and rightward
        left_integral = self._left_integrate(lb, ub)
        right_integral = self._right_integrate(ub, ub + self._interval_size)

        return left_integral + right_integral


class PredictorCorrector(Generic[Array]):
    """Compute recursion coefficients via predictor-corrector method.

    Uses numerical integration to compute recursion coefficients for
    orthonormal polynomials with respect to an arbitrary probability
    measure.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    References
    ----------
    W. Gautschi, "Orthogonal Polynomials: Computation and Approximation",
    Oxford University Press, 2004.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd
        self._integrator: Optional[IntegratorProtocol[Array]] = None
        self._measure: Optional[Callable[[Array], Array]] = None
        self._idx: int = 0
        self._ab: Optional[Array] = None

    def set_measure(self, measure: Callable[[Array], Array]) -> None:
        """Set the probability measure (PDF function).

        Parameters
        ----------
        measure : Callable[[Array], Array]
            Function that takes samples of shape (1, nsamples) and
            returns PDF values of shape (nsamples, 1).
        """
        self._measure = measure

    def set_integrator(self, integrator: IntegratorProtocol[Array]) -> None:
        """Set the numerical integrator."""
        self._integrator = integrator

    def _integrand_0(self, x: Array) -> Array:
        """Integrand for computing b_0: integral of measure."""
        return self._measure(x)

    def _integrand_1(self, x: Array) -> Array:
        """Integrand for computing a_{n-1}: <x * p_n, p_{n-1}>."""
        pvals = evaluate_orthonormal_polynomial_1d(self._ab, self._bkd, x)
        # measure(x): (nsamples, 1)
        # pvals: (nsamples, nterms)
        measure_vals = self._measure(x)[:, 0]  # (nsamples,)
        return (measure_vals * pvals[:, self._idx] * pvals[:, self._idx - 1])[:, None]

    def _integrand_2(self, x: Array) -> Array:
        """Integrand for computing b_n: <p_n, p_n>."""
        pvals = evaluate_orthonormal_polynomial_1d(self._ab, self._bkd, x)
        measure_vals = self._measure(x)[:, 0]  # (nsamples,)
        return (measure_vals * pvals[:, self._idx] ** 2)[:, None]

    def __call__(self, nterms: int) -> Array:
        """Compute recursion coefficients.

        Parameters
        ----------
        nterms : int
            Number of recursion coefficients to compute.

        Returns
        -------
        Array
            Recursion coefficients. Shape: (nterms, 2)
            Column 0: alpha (a) coefficients
            Column 1: beta (b) coefficients
        """
        if self._integrator is None:
            raise ValueError("Must set integrator before calling")
        if self._measure is None:
            raise ValueError("Must set measure before calling")

        ab = self._bkd.zeros((nterms + 1, 2))

        # Compute b_0 = sqrt(integral of measure)
        self._integrator.set_integrand(self._integrand_0)
        ab = self._set_value(ab, 0, 1, math.sqrt(self._integrator()))

        for idx in range(1, nterms + 1):
            # Predict step
            ab = self._set_value(ab, idx, 1, float(ab[idx - 1, 1]))
            if idx > 1:
                ab = self._set_value(ab, idx - 1, 0, float(ab[idx - 2, 0]))
            else:
                ab = self._set_value(ab, idx - 1, 0, 0.0)

            # Set state for integrands
            self._idx = idx
            self._ab = ab[: idx + 1, :]

            # Correct a_{idx-1}
            self._integrator.set_integrand(self._integrand_1)
            G_idx_idxm1 = self._integrator()
            new_a = float(ab[idx - 1, 0]) + float(ab[idx - 1, 1]) * G_idx_idxm1
            ab = self._set_value(ab, idx - 1, 0, new_a)

            # Update _ab with corrected a_{idx-1} before computing b_idx
            self._ab = ab[: idx + 1, :]

            # Correct b_idx
            self._integrator.set_integrand(self._integrand_2)
            G_idx_idx = self._integrator()
            new_b = float(ab[idx, 1]) * math.sqrt(G_idx_idx)
            ab = self._set_value(ab, idx, 1, new_b)

        return ab[:nterms, :]

    def _set_value(self, ab: Array, i: int, j: int, val: float) -> Array:
        """Set value in array (backend-agnostic)."""
        ab_np = self._bkd.to_numpy(ab).copy()
        ab_np[i, j] = val
        return self._bkd.asarray(ab_np)


class _ContinuousNumericOrthonormalPolynomial1DBase(
    OrthonormalPolynomial1D[Array], Generic[Array]
):
    """Base class for continuous numeric orthonormal polynomials.

    Uses the predictor-corrector method to compute recursion coefficients
    via numerical integration of the marginal's PDF.

    Domain Conventions
    ------------------
    - For bounded marginals: polynomials are defined on [-1, 1]
      (canonical). Samples must be in canonical domain [-1, 1].
      Use BoundedContinuousNumericOrthonormalPolynomial1D.
    - For unbounded marginals: polynomials are defined on the
      physical domain of the random variable (no transformation).
      Use UnboundedContinuousNumericOrthonormalPolynomial1D.

    This is an internal base class. Use BoundedContinuousNumericOrthonormalPolynomial1D
    or UnboundedContinuousNumericOrthonormalPolynomial1D instead.

    Parameters
    ----------
    marginal : Any
        The continuous marginal distribution. Must have:
        - __call__(samples) method returning PDF values
        - is_bounded() method returning bool
        - interval(alpha) method returning bounds
    bkd : Backend[Array]
        Computational backend.
    nquad_points : int
        Number of quadrature points for integration. Default: 100.
    integrator_options : dict, optional
        Options for unbounded integrator (only used for unbounded domains):
        - interval_size : float (default 2.0)
        - atol : float (default 1e-8)
        - rtol : float (default 1e-8)
        - maxiters : int (default 1000)
        - adaptive : bool (default True)
        - maxinner_iters : int (default 10)
    """

    def __init__(
        self,
        marginal,
        bkd: Backend[Array],
        nquad_points: int = 100,
        integrator_options: Optional[dict[str, Any]] = None,
    ):
        self._marginal = marginal
        self._nquad_points = nquad_points
        self._integrator_options = integrator_options or {}
        self._predictor_corrector = PredictorCorrector(bkd)
        self._recursion_coef: Optional[Array] = None

        # Setup canonical domain transformation
        self._setup_domain(bkd)

        # Setup integration before calling parent __init__
        self._setup_integration(bkd)

        super().__init__(bkd)

    def _setup_domain(self, bkd: Backend[Array]) -> None:
        """Setup canonical domain transformation."""
        if self._marginal.is_bounded():
            # Bounded marginal: use exact bounds and transform to [-1, 1]
            interval = self._marginal.interval(1.0)
            lb = float(interval[0, 0])
            ub = float(interval[0, 1])

            self._can_lb = -1.0
            self._can_ub = 1.0
            # Transform: x = loc + scale * t where t in [-1, 1]
            # loc = (lb + ub) / 2, scale = (ub - lb) / 2
            self._loc = (lb + ub) / 2.0
            self._scale = (ub - lb) / 2.0
        else:
            # Unbounded marginal: use infinite bounds directly
            # Check if distribution has finite lower or upper bound
            interval = self._marginal.interval(1 - 1e-10)
            lb_finite = float(interval[0, 0])
            ub_finite = float(interval[0, 1])

            # Determine true bounds based on distribution type
            # If lower bound is very close to 0, assume semi-infinite [0, inf)
            # Otherwise, assume doubly infinite (-inf, inf)
            # Use a heuristic: if lb_finite > -1e10, it's likely finite
            if lb_finite > -1e6:
                # Semi-infinite on right: [finite_lb, inf)
                # Determine true lower bound (check for 0 or other natural boundaries)
                self._can_lb = 0.0 if lb_finite >= -1e-10 else lb_finite
                self._can_ub = np.inf
            elif ub_finite < 1e6:
                # Semi-infinite on left: (-inf, finite_ub]
                self._can_lb = -np.inf
                self._can_ub = ub_finite
            else:
                # Doubly infinite: (-inf, inf)
                self._can_lb = -np.inf
                self._can_ub = np.inf

            # For unbounded, use identity transform (no scaling)
            self._loc = 0.0
            self._scale = 1.0

    def _canonical_to_physical(self, can_samples: Array) -> Array:
        """Map from canonical to physical domain."""
        return can_samples * self._scale + self._loc

    def _physical_to_canonical(self, samples: Array) -> Array:
        """Map from physical to canonical domain."""
        return (samples - self._loc) / self._scale

    def _canonical_pdf(self, can_samples: Array) -> Array:
        """Evaluate PDF in canonical domain.

        The PDF in canonical domain is:
        p_can(t) = p_phys(x(t)) * |dx/dt| = p_phys(x(t)) * scale
        """
        # can_samples: (1, nsamples)
        phys_samples = self._canonical_to_physical(can_samples)
        pdf_vals = self._marginal(phys_samples)  # (1, nsamples)
        return pdf_vals * self._scale

    def _setup_integration(self, bkd: Backend[Array]) -> None:
        """Setup numerical integration for the marginal."""

        # Setup measure (canonical PDF function)
        def measure(samples: Array) -> Array:
            # samples: (1, nsamples) in canonical domain
            # Return: (nsamples, 1)
            pdf_vals = self._canonical_pdf(samples)  # (1, nsamples)
            return pdf_vals.T  # (nsamples, 1)

        # Choose integrator based on domain boundedness
        if self._marginal.is_bounded():
            integrator: IntegratorProtocol[Array] = GaussLegendreIntegrator(
                bkd, self._nquad_points
            )
        else:
            # Use unbounded integrator with configurable options
            integrator = UnboundedIntegrator(
                bkd,
                npoints=self._nquad_points,
                **self._integrator_options,
            )

        integrator.set_bounds((self._can_lb, self._can_ub))
        self._predictor_corrector.set_integrator(integrator)
        self._predictor_corrector.set_measure(measure)

    def _get_recursion_coefficients(self, nterms: int) -> Array:
        """Compute recursion coefficients via predictor-corrector.

        Parameters
        ----------
        nterms : int
            Number of coefficients needed.

        Returns
        -------
        Array
            Recursion coefficients. Shape: (nterms, 2)
        """
        # Cache recursion coefficients
        if self._recursion_coef is None or self._recursion_coef.shape[0] < nterms:
            self._recursion_coef = self._predictor_corrector(nterms)

        return self._recursion_coef[:nterms, :]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"marginal={type(self._marginal).__name__}, nterms={self.nterms()})"
        )


class BoundedContinuousNumericOrthonormalPolynomial1D(
    _ContinuousNumericOrthonormalPolynomial1DBase[Array]
):
    """Numeric orthonormal polynomials for bounded continuous marginals.

    Uses the predictor-corrector method to compute recursion coefficients
    via numerical integration of the marginal's PDF.

    This polynomial expects samples in canonical domain [-1, 1]. The caller
    must transform physical domain samples to [-1, 1] before evaluation.
    Use TransformedBasis1D wrapper for physical domain samples.

    Parameters
    ----------
    marginal : Any
        The continuous marginal distribution. Must be bounded
        (is_bounded() returns True).
        Must have:
        - __call__(samples) method returning PDF values
        - is_bounded() method returning True
        - interval(alpha) method returning bounds
    bkd : Backend[Array]
        Computational backend.
    nquad_points : int
        Number of quadrature points for integration. Default: 100.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import BetaMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = BetaMarginal(3.0, 2.0, bkd)
    >>> poly = BoundedContinuousNumericOrthonormalPolynomial1D(marginal, bkd)
    >>> poly.set_nterms(5)
    >>> # Samples in canonical domain [-1, 1]
    >>> samples = bkd.array([[-0.6, 0.0, 0.6]])
    >>> values = poly(samples)  # Shape: (3, 5)
    """

    # Operates in canonical domain [-1, 1], requires TransformedBasis1D wrapper
    _operates_in_physical_domain = False

    def __init__(
        self,
        marginal,
        bkd: Backend[Array],
        nquad_points: int = 100,
    ):
        if not marginal.is_bounded():
            raise ValueError(
                "BoundedContinuousNumericOrthonormal"
                "Polynomial1D requires bounded marginal"
                f", got {type(marginal).__name__} with "
                "is_bounded()=False. Use Unbounded"
                "ContinuousNumericOrthonormal"
                "Polynomial1D instead."
            )
        super().__init__(marginal, bkd, nquad_points)


class UnboundedContinuousNumericOrthonormalPolynomial1D(
    _ContinuousNumericOrthonormalPolynomial1DBase[Array]
):
    """Numeric orthonormal polynomials for unbounded continuous marginals.

    Uses the predictor-corrector method to compute recursion coefficients
    via numerical integration of the marginal's PDF.

    This polynomial operates in the physical domain - it expects samples
    directly from the marginal distribution's support (e.g., [0, ∞) for Gamma).
    No transform wrapper is needed.

    Parameters
    ----------
    marginal : Any
        The continuous marginal distribution. Must be unbounded
        (is_bounded() returns False).
        Must have:
        - __call__(samples) method returning PDF values
        - is_bounded() method returning False
        - interval(alpha) method returning bounds
    bkd : Backend[Array]
        Computational backend.
    nquad_points : int
        Number of quadrature points for integration. Default: 100.
    integrator_options : dict, optional
        Options for unbounded integrator:
        - interval_size : float (default 2.0)
        - atol : float (default 1e-8)
        - rtol : float (default 1e-8)
        - maxiters : int (default 1000)
        - adaptive : bool (default True)
        - maxinner_iters : int (default 10)

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import GammaMarginal
    >>> bkd = NumpyBkd()
    >>> marginal = GammaMarginal(3.0, 2.0, bkd)
    >>> poly = UnboundedContinuousNumericOrthonormalPolynomial1D(marginal, bkd)
    >>> poly.set_nterms(5)
    >>> # Samples in physical domain [0, ∞)
    >>> samples = bkd.array([[0.5, 1.0, 2.0]])
    >>> values = poly(samples)  # Shape: (3, 5)
    """

    # Operates in physical domain (no transform needed)
    _operates_in_physical_domain = True

    def __init__(
        self,
        marginal,
        bkd: Backend[Array],
        nquad_points: int = 100,
        integrator_options: Optional[dict[str, Any]] = None,
    ):
        if marginal.is_bounded():
            raise ValueError(
                "UnboundedContinuousNumericOrthonormal"
                "Polynomial1D requires unbounded "
                f"marginal, got {type(marginal).__name__}"
                " with is_bounded()=True. Use Bounded"
                "ContinuousNumericOrthonormal"
                "Polynomial1D instead."
            )
        super().__init__(marginal, bkd, nquad_points, integrator_options)


def ContinuousNumericOrthonormalPolynomial1D(
    marginal,
    bkd: Backend[Array],
    nquad_points: int = 100,
    integrator_options: Optional[dict[str, Any]] = None,
) -> _ContinuousNumericOrthonormalPolynomial1DBase[Array]:
    """Factory function for continuous numeric orthonormal polynomials.

    Automatically selects the appropriate polynomial class based on whether
    the marginal distribution is bounded or unbounded.

    Parameters
    ----------
    marginal : Any
        The continuous marginal distribution. Must have:
        - __call__(samples) method returning PDF values
        - is_bounded() method returning bool
        - interval(alpha) method returning bounds
    bkd : Backend[Array]
        Computational backend.
    nquad_points : int
        Number of quadrature points for integration. Default: 100.
    integrator_options : dict, optional
        Options for unbounded integrator (ignored for bounded marginals):
        - interval_size : float (default 2.0)
        - atol : float (default 1e-8)
        - rtol : float (default 1e-8)
        - maxiters : int (default 1000)
        - adaptive : bool (default True)
        - maxinner_iters : int (default 10)

    Returns
    -------
    _ContinuousNumericOrthonormalPolynomial1DBase[Array]
        Either BoundedContinuousNumericOrthonormalPolynomial1D or
        UnboundedContinuousNumericOrthonormalPolynomial1D depending on the marginal.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> from pyapprox.probability.univariate import BetaMarginal, GammaMarginal
    >>> bkd = NumpyBkd()
    >>> # Bounded marginal - returns BoundedContinuousNumericOrthonormalPolynomial1D
    >>> poly_bounded = ContinuousNumericOrthonormalPolynomial1D(
    ...     BetaMarginal(3.0, 2.0, bkd), bkd
    ... )
    >>> # Unbounded marginal - returns UnboundedContinuousNumericOrthonormalPolynomial1D
    >>> poly_unbounded = ContinuousNumericOrthonormalPolynomial1D(
    ...     GammaMarginal(3.0, 2.0, bkd=bkd), bkd
    ... )
    """
    if marginal.is_bounded():
        return BoundedContinuousNumericOrthonormalPolynomial1D(
            marginal, bkd, nquad_points
        )
    else:
        return UnboundedContinuousNumericOrthonormalPolynomial1D(
            marginal, bkd, nquad_points, integrator_options
        )
