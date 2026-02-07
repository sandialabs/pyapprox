"""
Gamma univariate distribution.

Provides an analytically-defined Gamma distribution that implements
MarginalWithJacobianProtocol with full autograd support.
"""

from typing import Generic, Any, Tuple, Optional
import math

import numpy as np
from scipy import special

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import (
    LogHyperParameter,
    HyperParameterList,
)
from pyapprox.typing.probability.protocols import UniformQuadratureRule01Protocol
from pyapprox.typing.probability.univariate.beta import ScipyGaussLegendreQuadrature01
from pyapprox.typing.optimization.rootfinding.newton import (
    NewtonSolver,
    NewtonSolverResidualProtocol,
)


class _GammaCDFNewtonResidual(Generic[Array]):
    """Newton residual for computing Gamma inverse CDF."""

    def __init__(self, marginal: "GammaMarginal[Array]") -> None:
        self._marginal = marginal
        self._bkd = marginal.bkd()
        self._usamples: Array = None  # type: ignore

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def set_usamples(self, usamples: Array) -> None:
        """Set the target probability values."""
        if usamples.ndim != 1:
            raise ValueError("usamples must be 1D array")
        self._usamples = usamples

    def __call__(self, iterate: Array) -> Array:
        """Compute residual: CDF(x) - p."""
        # Convert 1D iterate to 2D for marginal.cdf
        iterate_2d = self._bkd.reshape(iterate, (1, -1))
        cdf_vals = self._marginal.cdf(iterate_2d)
        return cdf_vals[0] - self._usamples

    def linsolve(self, iterate: Array, res: Array) -> Array:
        """Solve J * delta = res where J = diag(pdf(x))."""
        # CDF jacobian is pdf, so linsolve is res / pdf
        iterate_2d = self._bkd.reshape(iterate, (1, -1))
        pdf_vals = self._marginal(iterate_2d)
        return res / pdf_vals[0]


class GammaMarginal(Generic[Array]):
    """
    Gamma distribution on [0, infinity) with autograd support.

    Implements MarginalWithJacobianProtocol with analytical formulas for
    PDF and logpdf, and numerical quadrature for CDF (autograd-compatible).

    The Gamma distribution has PDF:
        f(x; k, theta) = x^{k-1} * exp(-x/theta) / (theta^k * Gamma(k))

    where k is the shape parameter and theta is the scale parameter.

    Parameters
    ----------
    shape : float or Array
        Shape parameter k (k > 0). Can be Array for autograd w.r.t. parameters.
    scale : float or Array, optional
        Scale parameter theta (theta > 0). Default is 1.0.
    bkd : Backend[Array]
        The backend to use for computations.
    quadrature_rule : UniformQuadratureRule01Protocol[Array], optional
        Quadrature rule on [0, 1] with Lebesgue measure for CDF computation.
        If not provided, uses ScipyGaussLegendreQuadrature01.
    nquad_samples : int, optional
        Number of quadrature samples for CDF computation. Default is 50.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Default quadrature rule is used automatically
    >>> dist = GammaMarginal(shape=2.0, scale=1.0, bkd=bkd)
    >>> samples = np.array([[0.5, 1.0, 2.0]])  # Shape: (1, 3)
    >>> pdf_vals = dist(samples)  # PDF values, shape: (1, 3)
    >>> cdf_vals = dist.cdf(samples)  # CDF values, shape: (1, 3)
    """

    def __init__(
        self,
        shape: float,
        scale: float = 1.0,
        bkd: Backend[Array] = None,  # type: ignore
        quadrature_rule: Optional[UniformQuadratureRule01Protocol[Array]] = None,
        nquad_samples: int = 50,
    ):
        if bkd is None:
            raise ValueError("bkd must be provided")

        self._bkd = bkd

        # Store parameters - convert to floats for validation
        shape_val = float(shape)
        scale_val = float(scale)

        if shape_val <= 0:
            raise ValueError(f"shape must be positive, got {shape_val}")
        if scale_val <= 0:
            raise ValueError(f"scale must be positive, got {scale_val}")

        # Create hyperparameter list for parameter optimization
        # Both parameters are positive, so use log transform
        self._shape_hyp = LogHyperParameter(
            name="shape",
            nparams=1,
            user_values=shape_val,
            user_bounds=(1e-10, 1e10),
            bkd=bkd,
        )
        self._scale_hyp = LogHyperParameter(
            name="scale",
            nparams=1,
            user_values=scale_val,
            user_bounds=(1e-10, 1e10),
            bkd=bkd,
        )
        self._hyp_list = HyperParameterList([self._shape_hyp, self._scale_hyp])

        # Setup quadrature for CDF (autograd-compatible)
        # Use default GaussLegendreQuadrature01 if none provided
        if quadrature_rule is None:
            quadrature_rule = ScipyGaussLegendreQuadrature01(bkd)
        self._quadrature_rule = quadrature_rule
        self._quadx_01: Optional[Array] = None
        self._quadw_01: Optional[Array] = None
        self._newton_solver: Optional[NewtonSolver[Array]] = None
        self._newton_residual: Optional[_GammaCDFNewtonResidual[Array]] = None

        self._setup_quadrature(quadrature_rule, nquad_samples)
        # Setup Newton solver for inverse CDF
        self._setup_newton_solver()

        # Store scipy distribution for initial guess and rvs
        from scipy import stats
        self._scipy_rv = stats.gamma(shape_val, scale=scale_val)

    def _setup_quadrature(
        self,
        quadrature_rule: UniformQuadratureRule01Protocol[Array],
        nquad_samples: int,
    ) -> None:
        """Setup quadrature for Gamma CDF.

        The quadrature rule must be on [0, 1] with the Lebesgue measure.

        Raises
        ------
        ValueError
            If the quadrature rule does not integrate the Lebesgue measure
            on [0, 1] (validated by integrating f(x)=x^2 which should be 1/3).
        """
        points, weights = quadrature_rule(nquad_samples)
        self._quadx_01 = self._bkd.flatten(points)
        self._quadw_01 = self._bkd.flatten(weights)

        # Validate: integral of f(x)=x^2 on [0,1] should be 1/3
        x_squared = self._quadx_01 ** 2
        integral = self._bkd.sum(x_squared * self._quadw_01)
        expected = 1.0 / 3.0
        if not self._bkd.allclose(
            self._bkd.atleast_1d(integral),
            self._bkd.atleast_1d(self._bkd.asarray(expected)),
            rtol=1e-6,
            atol=1e-8,
        ):
            integral_val = float(self._bkd.to_numpy(integral))
            raise ValueError(
                f"Quadrature rule does not integrate Lebesgue measure on [0, 1]. "
                f"Expected integral of x^2 to be {expected:.10f}, got {integral_val:.10f}. "
                f"Ensure the quadrature rule has points in [0, 1] and weights sum to 1."
            )

    def _setup_newton_solver(self) -> None:
        """Setup Newton solver for inverse CDF computation."""
        self._newton_residual = _GammaCDFNewtonResidual(self)
        self._newton_solver = NewtonSolver(self._newton_residual)
        # Only 1 iteration since initial guess is good (from scipy)
        # This allows autograd to differentiate through the Newton step
        self._newton_solver.set_options(
            maxiters=1,
            verbosity=0,
            atol=10.0,  # Large tolerance since we only do 1 iteration
            rtol=10.0,
        )

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list for parameter optimization."""
        return self._hyp_list

    def nparams(self) -> int:
        """Return the number of distribution parameters (shape and scale)."""
        return self._hyp_list.nparams()

    def _get_shape(self) -> Array:
        """Get shape as array (preserves autograd graph)."""
        return self._shape_hyp.exp_values()[0]

    def _get_scale(self) -> Array:
        """Get scale as array (preserves autograd graph)."""
        return self._scale_hyp.exp_values()[0]

    def _get_rate(self) -> Array:
        """Get rate (1/scale) as array (preserves autograd graph)."""
        return 1.0 / self._get_scale()

    def _validate_input(self, samples: Array) -> Array:
        """Validate that input is 2D with shape (1, nsamples)."""
        if samples.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (1, nsamples), got {samples.ndim}D"
            )
        if samples.shape[0] != 1:
            raise ValueError(
                f"Univariate distribution expects shape (1, nsamples), "
                f"got {samples.shape}"
            )
        return samples[0]  # Return 1D for internal computation

    def nvars(self) -> int:
        """Return the number of variables (always 1 for univariate)."""
        return 1

    def shape(self) -> float:
        """Return the shape parameter."""
        return float(self._bkd.to_numpy(self._shape_hyp.exp_values())[0])

    def scale(self) -> float:
        """Return the scale parameter."""
        return float(self._bkd.to_numpy(self._scale_hyp.exp_values())[0])

    def rate(self) -> float:
        """Return the rate parameter (1/scale)."""
        return 1.0 / self.scale()

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the probability density function (PDF).

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the PDF. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            The evaluated PDF values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        return self._bkd.exp(self.logpdf(samples))

    def __call__(self, samples: Array) -> Array:
        """Evaluate the PDF (alias for pdf())."""
        return self.pdf(samples)

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the log probability density function.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the log PDF. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            The evaluated log PDF values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        shape = self._get_shape()
        rate = self._get_rate()
        # log f(x) = k*log(rate) - gammaln(k) + (k-1)*log(x) - rate*x
        log_const = (
            shape * self._bkd.log(rate)
            - self._bkd.gammaln(shape)
        )
        log_x = self._bkd.log(samples_1d)
        result = log_const + (shape - 1.0) * log_x - rate * samples_1d
        return self._bkd.reshape(result, (1, -1))

    def _require_quadrature(self) -> None:
        """Raise RuntimeError if quadrature rule was not provided."""
        if self._quadx_01 is None or self._quadw_01 is None:
            raise RuntimeError(
                "CDF/invcdf methods require a quadrature rule. "
                "Pass a UniformQuadratureRule01Protocol to the constructor."
            )

    def _gammainc(self, samples: Array) -> Array:
        """
        Compute regularized lower incomplete gamma function via quadrature.

        This is autograd-compatible since it uses backend operations only.

        gammainc(k, x) = (1/Gamma(k)) * integral_0^x t^{k-1} * exp(-t) dt
        """
        self._require_quadrature()
        assert self._quadx_01 is not None  # for type checker
        assert self._quadw_01 is not None

        shape = self._get_shape()
        rate = self._get_rate()
        # Transform integral_0^x to integral_0^1 with substitution t = x*u
        # integral_0^x t^{k-1} exp(-t) dt = x^k * integral_0^1 u^{k-1} exp(-x*u) du
        # Using rate: for rate*x instead of x
        rate_x = rate * samples
        quadx = rate_x[:, None] * self._quadx_01[None, :]
        quadw = rate_x[:, None] * self._quadw_01[None, :]
        integrand_vals = quadx ** (shape - 1.0) * self._bkd.exp(-quadx)
        integral = self._bkd.sum(integrand_vals * quadw, axis=1)
        return integral / self._bkd.exp(self._bkd.gammaln(shape))

    def cdf(self, samples: Array) -> Array:
        """
        Evaluate the cumulative distribution function.

        Uses numerical quadrature for autograd compatibility.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the CDF. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            CDF values in [0, 1]. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        RuntimeError
            If quadrature rule was not provided at construction
        """
        samples_1d = self._validate_input(samples)
        result = self._gammainc(samples_1d)
        return self._bkd.reshape(result, (1, -1))

    def invcdf(self, probs: Array) -> Array:
        """
        Evaluate the inverse CDF (quantile function).

        Uses Newton iteration with scipy for initial guess.
        Autograd-compatible through the Newton step.

        Parameters
        ----------
        probs : Array
            Probability values in [0, 1]. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Quantile values in [0, infinity). Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        RuntimeError
            If quadrature rule was not provided at construction
        """
        self._require_quadrature()
        assert self._newton_solver is not None  # for type checker
        assert self._newton_residual is not None

        probs_1d = self._validate_input(probs)

        # Handle boundary cases
        idx0 = self._bkd.where(probs_1d == 0.0)[0]
        idx1 = self._bkd.where(probs_1d == 1.0)[0]
        jdx = self._bkd.where((probs_1d != 0.0) & (probs_1d != 1.0))[0]

        # Initialize result
        result = self._bkd.zeros_like(probs_1d)

        if len(idx0) > 0:
            result = self._set_indices(result, idx0, 0.0)
        if len(idx1) > 0:
            result = self._set_indices(result, idx1, float("inf"))

        if len(jdx) == 0:
            return self._bkd.reshape(result, (1, -1))

        # Get scipy initial guess
        probs_np = self._bkd.to_numpy(probs_1d[jdx])
        from scipy import stats
        scipy_rv = stats.gamma(self.shape(), scale=self.scale())
        init_guess = self._bkd.asarray(scipy_rv.ppf(probs_np))

        # Newton iteration (1 step for autograd)
        self._newton_residual.set_usamples(probs_1d[jdx])
        quantiles = self._newton_solver.solve(init_guess)

        # Update result at interior indices
        for ii, idx in enumerate(self._bkd.to_numpy(jdx)):
            result = self._set_index(result, int(idx), quantiles[ii])

        return self._bkd.reshape(result, (1, -1))

    def _set_indices(self, arr: Array, indices: Array, value: float) -> Array:
        """Set values at indices (helper for in-place modification)."""
        arr_np = self._bkd.to_numpy(arr).copy()
        indices_np = self._bkd.to_numpy(indices)
        arr_np[indices_np] = value
        return self._bkd.asarray(arr_np)

    def _set_index(self, arr: Array, idx: int, value: Any) -> Array:
        """Set value at single index."""
        arr_np = self._bkd.to_numpy(arr).copy()
        arr_np[idx] = float(self._bkd.to_numpy(self._bkd.atleast_1d(value))[0])
        return self._bkd.asarray(arr_np)

    # Alias for compatibility
    ppf = invcdf

    def invcdf_jacobian(self, probs: Array) -> Array:
        """
        Compute Jacobian of inverse CDF.

        d(F^{-1})/dp = 1 / pdf(F^{-1}(p))

        Parameters
        ----------
        probs : Array
            Probability values in [0, 1]. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        # Validation done by invcdf and __call__
        self._validate_input(probs)
        samples = self.invcdf(probs)
        pdf_vals = self(samples)
        return 1.0 / pdf_vals

    def rvs(self, nsamples: int) -> Array:
        """
        Generate random samples from the distribution.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Random samples. Shape: (1, nsamples) for protocol compliance.
        """
        samples = np.random.gamma(self.shape(), self.scale(), nsamples)
        return self._bkd.reshape(self._bkd.asarray(samples), (1, nsamples))

    def mean_value(self) -> float:
        """
        Return the mean of the distribution.

        mean = shape * scale

        Returns
        -------
        float
            Mean value.
        """
        return self.shape() * self.scale()

    def variance(self) -> float:
        """
        Return the variance of the distribution.

        variance = shape * scale^2

        Returns
        -------
        float
            Variance value.
        """
        return self.shape() * self.scale() ** 2

    def std(self) -> float:
        """
        Return the standard deviation.

        Returns
        -------
        float
            Standard deviation.
        """
        return math.sqrt(self.variance())

    def is_bounded(self) -> bool:
        """
        Check if the distribution is bounded.

        Returns
        -------
        bool
            False for Gamma (unbounded on right).
        """
        return False

    def bounds(self) -> Tuple[float, float]:
        """
        Return the support bounds.

        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds (0, infinity).
        """
        return (0.0, float("inf"))

    def interval(self, alpha: float) -> Array:
        """
        Compute the interval with given probability content.

        Parameters
        ----------
        alpha : float
            Probability content of the interval (0 < alpha < 1).

        Returns
        -------
        Array
            Interval [lower, upper] such that P(lower < X < upper) = alpha.
            Shape: (1, 2)
        """
        eps = (1.0 - alpha) / 2.0
        probs_2d = self._bkd.array([[eps, 1.0 - eps]])  # Shape: (1, 2)
        return self.invcdf(probs_2d)

    def logpdf_jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the log PDF w.r.t. samples.

        d/dx log f(x) = (k-1)/x - rate

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        shape = self._get_shape()
        rate = self._get_rate()
        grad = (shape - 1.0) / samples_1d - rate
        return self._bkd.reshape(grad, (1, -1))

    def pdf_jacobian(self, samples: Array) -> Array:
        """
        Compute the Jacobian of the PDF.

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        # Validation done by __call__ and logpdf_jacobian
        pdf_vals = self(samples)
        logpdf_jac = self.logpdf_jacobian(samples)
        return pdf_vals * logpdf_jac

    def logpdf_jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute the Jacobian of log PDF w.r.t. distribution parameters.

        For Gamma(shape=k, scale=theta), log PDF is:
        logpdf = (k-1)*log(x) - x/theta - k*log(theta) - gammaln(k)

        Derivatives in log-space (optimizer sees log_shape, log_scale):
        d(logpdf)/d(log_shape) = shape * (log(x) - log(scale) - psi(shape))
        d(logpdf)/d(log_scale) = -shape + x/scale

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian.
            Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian matrix with shape (nsamples, nparams).
            Column 0: d(logpdf)/d(log_shape)
            Column 1: d(logpdf)/d(log_scale)
        """
        samples_1d = self._validate_input(samples)
        shape = self._get_shape()
        scale = self._get_scale()

        log_x = self._bkd.log(samples_1d)
        log_scale = self._bkd.log(scale)

        # Digamma term
        psi_shape = self._bkd.digamma(shape)

        # d(logpdf)/d(shape) = log(x) - log(scale) - psi(shape)
        # d(logpdf)/d(log_shape) = shape * d(logpdf)/d(shape)
        d_log_shape = shape * (log_x - log_scale - psi_shape)

        # d(logpdf)/d(scale) = -shape/scale + x/scale^2
        # d(logpdf)/d(log_scale) = scale * d(logpdf)/d(scale) = -shape + x/scale
        d_log_scale = -shape + samples_1d / scale

        # Stack columns: shape (nsamples, 2)
        return self._bkd.stack([d_log_shape, d_log_scale], axis=1)

    def base_distribution(self) -> Any:
        """Return the base distribution for reparameterization (Uniform(0,1))."""
        from pyapprox.typing.probability.univariate.uniform import UniformMarginal
        return UniformMarginal(0.0, 1.0, self._bkd)

    def reparameterize(self, base_samples: Array) -> Array:
        """Transform Uniform(0,1) base samples to samples from this distribution.

        Parameters
        ----------
        base_samples : Array
            Samples from U(0,1), shape ``(1, nsamples)``.

        Returns
        -------
        Array
            Samples from this distribution, shape ``(1, nsamples)``.
        """
        return self.invcdf(base_samples)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another GammaMarginal."""
        if not isinstance(other, GammaMarginal):
            return False
        return self.shape() == other.shape() and self.scale() == other.scale()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"GammaMarginal(shape={self.shape()}, scale={self.scale()})"
