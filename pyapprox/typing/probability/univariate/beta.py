"""
Beta univariate distribution.

Provides an analytically-defined Beta distribution that implements
MarginalWithJacobianProtocol with full autograd support.
"""

from typing import Generic, Any, Tuple, Optional
import math

import numpy as np
from scipy import special, stats
from scipy.special import roots_legendre

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.util.hyperparameter import (
    LogHyperParameter,
    HyperParameterList,
)
from pyapprox.typing.probability.protocols import UniformQuadratureRule01Protocol
from pyapprox.typing.optimization.rootfinding.newton import (
    NewtonSolver,
    NewtonSolverResidualProtocol,
)


class ScipyGaussLegendreQuadrature01(Generic[Array]):
    """Gauss-Legendre quadrature on [0, 1] with Lebesgue measure.

    This uses scipy.special.roots_legendre which returns points and weights
    for integrating on [-1, 1] with the Lebesgue measure (not probability
    measure). The weights sum to 2 (length of interval). We transform to
    [0, 1] and normalize weights to sum to 1.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> quad = ScipyGaussLegendreQuadrature01(bkd)
    >>> points, weights = quad(5)
    >>> # Points are in [0, 1], weights sum to 1
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(self, npoints: int) -> Tuple[Array, Array]:
        """Get quadrature points and weights.

        Parameters
        ----------
        npoints : int
            Number of quadrature points.

        Returns
        -------
        points : Array
            Quadrature points in [0, 1], shape (npoints,).
        weights : Array
            Quadrature weights summing to 1, shape (npoints,).
        """
        # scipy returns points in [-1, 1] and weights that sum to 2
        points_11, weights_11 = roots_legendre(npoints)

        # Transform points from [-1, 1] to [0, 1]
        points_01 = (points_11 + 1.0) / 2.0

        # Scale weights: integral on [0,1] = (1/2) * integral on [-1,1]
        # scipy weights sum to 2, so divide by 2 to get weights summing to 1
        weights_01 = weights_11 / 2.0

        return self._bkd.asarray(points_01.tolist()), self._bkd.asarray(
            weights_01.tolist()
        )


class _BetaCDFNewtonResidual(Generic[Array]):
    """Newton residual for computing Beta inverse CDF."""

    def __init__(self, marginal: "BetaMarginal[Array]") -> None:
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


class BetaMarginal(Generic[Array]):
    """
    Beta distribution on [lb, ub] with autograd support.

    Implements MarginalWithJacobianProtocol with analytical formulas for
    PDF and logpdf, and numerical quadrature for CDF (autograd-compatible).

    The Beta distribution has PDF:
        f(x; a, b) = x^{a-1} (1-x)^{b-1} / B(a, b)

    where B(a, b) is the beta function. When lb and ub are specified,
    the distribution is scaled and shifted to [lb, ub].

    Parameters
    ----------
    alpha : float
        Shape parameter alpha (a > 0).
    beta : float
        Shape parameter beta (b > 0).
    bkd : Backend[Array]
        The backend to use for computations.
    lb : float, optional
        Lower bound of the distribution support. Default is 0.0.
    ub : float, optional
        Upper bound of the distribution support. Default is 1.0.
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
    >>> # Default: Beta on [0, 1]
    >>> dist = BetaMarginal(alpha=2.0, beta=5.0, bkd=bkd)
    >>> # Beta on [2, 5]
    >>> dist2 = BetaMarginal(alpha=2.0, beta=5.0, bkd=bkd, lb=2.0, ub=5.0)
    >>> samples = np.array([[0.1, 0.3, 0.5]])  # Shape: (1, 3)
    >>> pdf_vals = dist(samples)  # PDF values, shape: (1, 3)
    >>> cdf_vals = dist.cdf(samples)  # CDF values, shape: (1, 3)
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        bkd: Backend[Array],
        lb: float = 0.0,
        ub: float = 1.0,
        quadrature_rule: Optional[UniformQuadratureRule01Protocol[Array]] = None,
        nquad_samples: int = 50,
    ):
        # Validate parameters
        alpha_val = float(alpha)
        beta_val = float(beta)

        if alpha_val <= 0:
            raise ValueError(f"alpha must be positive, got {alpha_val}")
        if beta_val <= 0:
            raise ValueError(f"beta must be positive, got {beta_val}")
        if lb >= ub:
            raise ValueError(f"lb must be < ub, got lb={lb}, ub={ub}")

        self._bkd = bkd

        # Store bounds (NOT hyperparameters - fixed values)
        self._lb = float(lb)
        self._ub = float(ub)
        self._scale = self._ub - self._lb

        # Create hyperparameter list for parameter optimization
        # Both alpha and beta are positive, use log transform
        self._alpha_hyp = LogHyperParameter(
            name="alpha",
            nparams=1,
            user_values=alpha_val,
            user_bounds=(1e-10, 1e10),
            bkd=bkd,
        )
        self._beta_hyp = LogHyperParameter(
            name="beta",
            nparams=1,
            user_values=beta_val,
            user_bounds=(1e-10, 1e10),
            bkd=bkd,
        )
        self._hyp_list = HyperParameterList([self._alpha_hyp, self._beta_hyp])

        # Setup quadrature for CDF (autograd-compatible)
        # Use default GaussLegendreQuadrature01 if none provided
        if quadrature_rule is None:
            quadrature_rule = ScipyGaussLegendreQuadrature01(bkd)
        self._quadrature_rule = quadrature_rule
        self._quadx_01: Optional[Array] = None
        self._quadw_01: Optional[Array] = None
        self._newton_solver: Optional[NewtonSolver[Array]] = None
        self._newton_residual: Optional[_BetaCDFNewtonResidual[Array]] = None

        self._setup_quadrature(quadrature_rule, nquad_samples)
        # Setup Newton solver for inverse CDF
        self._setup_newton_solver()

        # Store scipy distribution for initial guess and rvs
        # Use loc=lb, scale=ub-lb for bounded Beta
        self._scipy_rv = stats.beta(alpha_val, beta_val, loc=self._lb, scale=self._scale)

    def _setup_quadrature(
        self,
        quadrature_rule: UniformQuadratureRule01Protocol[Array],
        nquad_samples: int,
    ) -> None:
        """Setup quadrature for Beta CDF.

        The quadrature rule must be on [0, 1] with the Lebesgue measure.
        For Beta(a, b), we compute the CDF by integrating the unnormalized
        PDF against this uniform measure.

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
        self._newton_residual = _BetaCDFNewtonResidual(self)
        self._newton_solver = NewtonSolver(self._newton_residual)
        # Only 1 iteration since initial guess is good (from scipy)
        self._newton_solver.set_options(
            maxiters=1,
            verbosity=0,
            atol=10.0,
            rtol=10.0,
        )

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        return self._bkd

    def hyp_list(self) -> HyperParameterList:
        """Return the hyperparameter list for parameter optimization."""
        return self._hyp_list

    def nparams(self) -> int:
        """Return the number of distribution parameters (alpha and beta)."""
        return self._hyp_list.nparams()

    def _get_alpha(self) -> Array:
        """Get alpha as array (preserves autograd graph)."""
        return self._alpha_hyp.exp_values()[0]

    def _get_beta(self) -> Array:
        """Get beta as array (preserves autograd graph)."""
        return self._beta_hyp.exp_values()[0]

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

    def alpha(self) -> float:
        """Return the alpha shape parameter."""
        return float(self._bkd.to_numpy(self._alpha_hyp.exp_values())[0])

    def beta(self) -> float:
        """Return the beta shape parameter."""
        return float(self._bkd.to_numpy(self._beta_hyp.exp_values())[0])

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the probability density function (PDF).

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the PDF. Shape: (1, nsamples) - must be 2D.
            Values must be in [0, 1].

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
            Points at which to evaluate the log PDF. Shape: (1, nsamples) - must be 2D.
            Values must be in [lb, ub].

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

        # Transform from [lb, ub] to [0, 1]
        samples_01 = (samples_1d - self._lb) / self._scale

        alpha = self._get_alpha()
        beta = self._get_beta()
        # log f(x) = (a-1)*log(x) + (b-1)*log(1-x) - log(B(a,b)) - log(scale)
        # log(B(a,b)) = gammaln(a) + gammaln(b) - gammaln(a+b)
        log_beta_func = (
            self._bkd.gammaln(alpha)
            + self._bkd.gammaln(beta)
            - self._bkd.gammaln(alpha + beta)
        )
        log_x = self._bkd.log(samples_01)
        log_1mx = self._bkd.log(1.0 - samples_01)
        result = (
            (alpha - 1.0) * log_x
            + (beta - 1.0) * log_1mx
            - log_beta_func
            - math.log(self._scale)  # Jacobian correction
        )
        return self._bkd.reshape(result, (1, -1))

    def _require_quadrature(self) -> None:
        """Raise RuntimeError if quadrature rule was not provided."""
        if self._quadx_01 is None or self._quadw_01 is None:
            raise RuntimeError(
                "CDF/invcdf methods require a quadrature rule. "
                "Pass a UniformQuadratureRule01Protocol to the constructor."
            )

    def _betainc(self, samples_1d: Array) -> Array:
        """
        Compute regularized incomplete beta function via quadrature.

        I_x(a, b) = (1/B(a,b)) * integral_0^x t^{a-1} (1-t)^{b-1} dt

        This is autograd-compatible since it uses backend operations only.

        Parameters
        ----------
        samples_1d : Array
            1D array of sample points (already validated and extracted).
        """
        self._require_quadrature()
        assert self._quadx_01 is not None  # for type checker
        assert self._quadw_01 is not None

        alpha = self._get_alpha()
        beta = self._get_beta()

        # Transform integral_0^x to integral_0^1 with substitution t = x*u
        # integral_0^x t^{a-1}(1-t)^{b-1} dt = x^a * integral_0^1 u^{a-1}(1-x*u)^{b-1} du
        # Note: simpler is just evaluating the integrand at quadrature points
        # scaled by x
        quadx = samples_1d[:, None] * self._quadx_01[None, :]
        quadw = samples_1d[:, None] * self._quadw_01[None, :]

        # Integrand: t^{a-1} (1-t)^{b-1}
        integrand_vals = (
            quadx ** (alpha - 1.0) * (1.0 - quadx) ** (beta - 1.0)
        )
        integral = self._bkd.sum(integrand_vals * quadw, axis=1)

        # Normalize by B(a, b)
        log_beta_func = (
            self._bkd.gammaln(alpha)
            + self._bkd.gammaln(beta)
            - self._bkd.gammaln(alpha + beta)
        )
        return integral / self._bkd.exp(log_beta_func)

    def cdf(self, samples: Array) -> Array:
        """
        Evaluate the cumulative distribution function.

        Uses numerical quadrature for autograd compatibility.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the CDF. Shape: (1, nsamples) - must be 2D.
            Values must be in [lb, ub].

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
        # Transform from [lb, ub] to [0, 1]
        samples_01 = (samples_1d - self._lb) / self._scale
        result = self._betainc(samples_01)
        return self._bkd.reshape(result, (1, -1))

    def invcdf(self, probs: Array) -> Array:
        """
        Evaluate the inverse CDF (quantile function).

        Uses Newton iteration with scipy for initial guess.
        Autograd-compatible through the Newton step.

        Parameters
        ----------
        probs : Array
            Probability values in [0, 1]. Shape: (1, nsamples) - must be 2D.

        Returns
        -------
        Array
            Quantile values in [lb, ub]. Shape: (1, nsamples)

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
        probs_flat = self._bkd.flatten(probs_1d)

        # Handle boundary cases (use lb and ub instead of 0 and 1)
        idx0 = self._bkd.where(probs_flat == 0.0)[0]
        idx1 = self._bkd.where(probs_flat == 1.0)[0]
        jdx = self._bkd.where((probs_flat != 0.0) & (probs_flat != 1.0))[0]

        # Initialize result
        result = self._bkd.zeros_like(probs_flat)

        if len(idx0) > 0:
            result = self._set_indices(result, idx0, self._lb)
        if len(idx1) > 0:
            result = self._set_indices(result, idx1, self._ub)

        if len(jdx) == 0:
            return self._bkd.reshape(result, (1, -1))

        # Get scipy initial guess (already returns values in [lb, ub])
        probs_np = self._bkd.to_numpy(probs_flat[jdx])
        init_guess = self._bkd.asarray(self._scipy_rv.ppf(probs_np))

        # Newton iteration (1 step for autograd)
        self._newton_residual.set_usamples(probs_flat[jdx])
        quantiles = self._newton_solver.solve(init_guess)

        # Update result at interior indices
        for ii, idx in enumerate(self._bkd.to_numpy(jdx)):
            result = self._set_index(result, int(idx), quantiles[ii])

        return self._bkd.reshape(result, (1, -1))

    def _set_indices(self, arr: Array, indices: Array, value: float) -> Array:
        """Set values at indices (helper for modification)."""
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
            Probability values in [0, 1]. Shape: (1, nsamples) - must be 2D.

        Returns
        -------
        Array
            Jacobian values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        # Validate input (also done in invcdf and __call__)
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
            Random samples in [lb, ub]. Shape: (1, nsamples) for protocol compliance.
        """
        # Generate samples in [0, 1] and transform to [lb, ub]
        samples_01 = np.random.beta(self.alpha(), self.beta(), nsamples)
        samples = self._lb + samples_01 * self._scale
        return self._bkd.reshape(self._bkd.asarray(samples), (1, nsamples))

    def mean_value(self) -> float:
        """
        Return the mean of the distribution.

        mean = lb + scale * alpha / (alpha + beta)

        For Beta on [0, 1], this is alpha / (alpha + beta).

        Returns
        -------
        float
            Mean value.
        """
        mean_01 = self.alpha() / (self.alpha() + self.beta())
        return self._lb + self._scale * mean_01

    def variance(self) -> float:
        """
        Return the variance of the distribution.

        variance = scale^2 * alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))

        For Beta on [0, 1], this is alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1)).

        Returns
        -------
        float
            Variance value.
        """
        ab = self.alpha() + self.beta()
        var_01 = (self.alpha() * self.beta()) / (ab**2 * (ab + 1))
        return self._scale**2 * var_01

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
            True for Beta (bounded on [lb, ub]).
        """
        return True

    def lower(self) -> float:
        """
        Return the lower bound.

        Returns
        -------
        float
            Lower bound of the support.
        """
        return self._lb

    def upper(self) -> float:
        """
        Return the upper bound.

        Returns
        -------
        float
            Upper bound of the support.
        """
        return self._ub

    def bounds(self) -> Tuple[float, float]:
        """
        Return the support bounds.

        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds (lb, ub).
        """
        return (self._lb, self._ub)

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
        Compute the Jacobian of the log PDF.

        For x in [lb, ub], let t = (x - lb) / scale be in [0, 1].
        d/dx log f(x) = (d/dt log f_01(t)) * (1/scale)
                      = ((a-1)/t - (b-1)/(1-t)) / scale

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
        # Transform to [0, 1]
        samples_01 = (samples_1d - self._lb) / self._scale
        alpha = self._get_alpha()
        beta = self._get_beta()
        # Derivative w.r.t. t on [0, 1]
        grad_01 = (alpha - 1.0) / samples_01 - (beta - 1.0) / (1.0 - samples_01)
        # Chain rule: d/dx = (1/scale) * d/dt
        grad = grad_01 / self._scale
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
        # Input validation is done by __call__ and logpdf_jacobian
        pdf_vals = self(samples)
        logpdf_jac = self.logpdf_jacobian(samples)
        return pdf_vals * logpdf_jac

    def logpdf_jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute the Jacobian of log PDF w.r.t. distribution parameters.

        For Beta(alpha, beta) on [lb, ub], log PDF is:
        logpdf = (a-1)*log(t) + (b-1)*log(1-t) - log(B(a,b)) - log(scale)

        where t = (x - lb) / scale is in [0, 1].
        log(B(a,b)) = gammaln(a) + gammaln(b) - gammaln(a+b)

        Derivatives in log-space (optimizer sees log_alpha, log_beta):
        d(logpdf)/d(log_alpha) = alpha * (log(t) - psi(alpha) + psi(alpha+beta))
        d(logpdf)/d(log_beta) = beta * (log(1-t) - psi(beta) + psi(alpha+beta))

        Note: The -log(scale) term is constant w.r.t. alpha, beta, so it doesn't
        affect the parameter jacobian.

        Parameters
        ----------
        samples : Array
            Points at which to compute the Jacobian.
            Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian matrix with shape (nsamples, nparams).
            Column 0: d(logpdf)/d(log_alpha)
            Column 1: d(logpdf)/d(log_beta)
        """
        samples_1d = self._validate_input(samples)
        # Transform to [0, 1]
        samples_01 = (samples_1d - self._lb) / self._scale

        alpha = self._get_alpha()
        beta = self._get_beta()

        log_x = self._bkd.log(samples_01)
        log_1mx = self._bkd.log(1.0 - samples_01)

        # Digamma terms
        psi_alpha = self._bkd.digamma(alpha)
        psi_beta = self._bkd.digamma(beta)
        psi_sum = self._bkd.digamma(alpha + beta)

        # d(logpdf)/d(alpha) = log(t) - psi(alpha) + psi(alpha+beta)
        # d(logpdf)/d(log_alpha) = alpha * d(logpdf)/d(alpha)
        d_log_alpha = alpha * (log_x - psi_alpha + psi_sum)

        # d(logpdf)/d(beta) = log(1-t) - psi(beta) + psi(alpha+beta)
        # d(logpdf)/d(log_beta) = beta * d(logpdf)/d(beta)
        d_log_beta = beta * (log_1mx - psi_beta + psi_sum)

        # Stack columns: shape (nsamples, 2)
        return self._bkd.stack([d_log_alpha, d_log_beta], axis=1)

    def __eq__(self, other: Any) -> bool:
        """Check equality with another BetaMarginal."""
        if not isinstance(other, BetaMarginal):
            return False
        return (
            self.alpha() == other.alpha()
            and self.beta() == other.beta()
            and self._lb == other._lb
            and self._ub == other._ub
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"BetaMarginal(alpha={self.alpha()}, beta={self.beta()}, "
            f"lb={self._lb}, ub={self._ub})"
        )
