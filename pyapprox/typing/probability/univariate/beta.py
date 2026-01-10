"""
Beta univariate distribution.

Provides an analytically-defined Beta distribution that implements
MarginalWithJacobianProtocol with full autograd support.
"""

from typing import Generic, Any, Tuple
import math

import numpy as np
from scipy import special, stats

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.affine.univariate.globalpoly import (
    JacobiPolynomial1D,
    GaussQuadratureRule,
)
from pyapprox.typing.optimization.rootfinding.newton import (
    NewtonSolver,
    NewtonSolverResidualProtocol,
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
    Beta distribution on [0, 1] with autograd support.

    Implements MarginalWithJacobianProtocol with analytical formulas for
    PDF and logpdf, and numerical quadrature for CDF (autograd-compatible).

    The Beta distribution has PDF:
        f(x; a, b) = x^{a-1} (1-x)^{b-1} / B(a, b)

    where B(a, b) is the beta function.

    Parameters
    ----------
    alpha : float
        Shape parameter alpha (a > 0).
    beta : float
        Shape parameter beta (b > 0).
    bkd : Backend[Array]
        The backend to use for computations.
    nquad_samples : int, optional
        Number of quadrature samples for CDF computation. Default is 50.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> dist = BetaMarginal(alpha=2.0, beta=5.0, bkd=bkd)
    >>> samples = np.array([[0.1, 0.3, 0.5]])  # Shape: (1, 3)
    >>> pdf_vals = dist(samples)  # PDF values, shape: (1, 3)
    >>> cdf_vals = dist.cdf(samples)  # CDF values, shape: (1, 3)
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        bkd: Backend[Array],
        nquad_samples: int = 50,
    ):
        # Validate parameters
        alpha_val = float(alpha)
        beta_val = float(beta)

        if alpha_val <= 0:
            raise ValueError(f"alpha must be positive, got {alpha_val}")
        if beta_val <= 0:
            raise ValueError(f"beta must be positive, got {beta_val}")

        self._bkd = bkd

        # Store as backend arrays for autograd support
        self._alpha = self._bkd.asarray(alpha_val)
        self._beta = self._bkd.asarray(beta_val)

        # Setup quadrature for CDF (autograd-compatible)
        self._setup_quadrature(nquad_samples)

        # Setup Newton solver for inverse CDF
        self._setup_newton_solver()

        # Store scipy distribution for initial guess and rvs
        self._scipy_rv = stats.beta(alpha_val, beta_val)

    def _setup_quadrature(self, nquad_samples: int) -> None:
        """Setup Gauss-Jacobi quadrature for Beta CDF.

        For Beta(a, b), the CDF can be computed using quadrature with
        the Jacobi weight function w(x) = x^{a-1} (1-x)^{b-1} on [0, 1].
        We use Legendre (uniform weight) on [0, 1] and multiply by the
        unnormalized PDF.
        """
        # Use Legendre polynomial for quadrature, mapped to [0, 1]
        from pyapprox.typing.surrogates.affine.univariate.globalpoly import (
            LegendrePolynomial1D,
        )

        legendre = LegendrePolynomial1D(self._bkd)
        quad_rule = GaussQuadratureRule(legendre, store=True)
        points, weights = quad_rule(nquad_samples)

        # Map from [-1, 1] to [0, 1]: x_01 = (x + 1) / 2
        self._quadx_01 = self._bkd.flatten((points + 1.0) / 2.0)
        self._quadw_01 = self._bkd.flatten(weights)

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

    @property
    def alpha(self) -> float:
        """Return the alpha shape parameter."""
        return float(self._bkd.to_numpy(self._alpha))

    @property
    def beta(self) -> float:
        """Return the beta shape parameter."""
        return float(self._bkd.to_numpy(self._beta))

    def __call__(self, samples: Array) -> Array:
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

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the log probability density function.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the log PDF. Shape: (1, nsamples) - must be 2D.
            Values must be in [0, 1].

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
        # log f(x) = (a-1)*log(x) + (b-1)*log(1-x) - log(B(a,b))
        # log(B(a,b)) = gammaln(a) + gammaln(b) - gammaln(a+b)
        log_beta = (
            self._bkd.gammaln(self._alpha)
            + self._bkd.gammaln(self._beta)
            - self._bkd.gammaln(self._alpha + self._beta)
        )
        log_x = self._bkd.log(samples_1d)
        log_1mx = self._bkd.log(1.0 - samples_1d)
        result = (
            (self._alpha - 1.0) * log_x
            + (self._beta - 1.0) * log_1mx
            - log_beta
        )
        return self._bkd.reshape(result, (1, -1))

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
        # Transform integral_0^x to integral_0^1 with substitution t = x*u
        # integral_0^x t^{a-1}(1-t)^{b-1} dt = x^a * integral_0^1 u^{a-1}(1-x*u)^{b-1} du
        # Note: simpler is just evaluating the integrand at quadrature points
        # scaled by x
        quadx = samples_1d[:, None] * self._quadx_01[None, :]
        quadw = samples_1d[:, None] * self._quadw_01[None, :]

        # Integrand: t^{a-1} (1-t)^{b-1}
        integrand_vals = (
            quadx ** (self._alpha - 1.0) * (1.0 - quadx) ** (self._beta - 1.0)
        )
        integral = self._bkd.sum(integrand_vals * quadw, axis=1)

        # Normalize by B(a, b)
        log_beta = (
            self._bkd.gammaln(self._alpha)
            + self._bkd.gammaln(self._beta)
            - self._bkd.gammaln(self._alpha + self._beta)
        )
        return integral / self._bkd.exp(log_beta)

    def cdf(self, samples: Array) -> Array:
        """
        Evaluate the cumulative distribution function.

        Uses numerical quadrature for autograd compatibility.

        Parameters
        ----------
        samples : Array
            Points at which to evaluate the CDF. Shape: (1, nsamples) - must be 2D.
            Values must be in [0, 1].

        Returns
        -------
        Array
            CDF values in [0, 1]. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        samples_1d = self._validate_input(samples)
        result = self._betainc(samples_1d)
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
            Quantile values in [0, 1]. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        probs_1d = self._validate_input(probs)
        probs_flat = self._bkd.flatten(probs_1d)

        # Handle boundary cases
        idx0 = self._bkd.where(probs_flat == 0.0)[0]
        idx1 = self._bkd.where(probs_flat == 1.0)[0]
        jdx = self._bkd.where((probs_flat != 0.0) & (probs_flat != 1.0))[0]

        # Initialize result
        result = self._bkd.zeros_like(probs_flat)

        if len(idx0) > 0:
            result = self._set_indices(result, idx0, 0.0)
        if len(idx1) > 0:
            result = self._set_indices(result, idx1, 1.0)

        if len(jdx) == 0:
            return self._bkd.reshape(result, (1, -1))

        # Get scipy initial guess
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
            Random samples. Shape: (1, nsamples) for protocol compliance.
        """
        samples = np.random.beta(self.alpha, self.beta, nsamples)
        return self._bkd.reshape(self._bkd.asarray(samples), (1, nsamples))

    def mean_value(self) -> float:
        """
        Return the mean of the distribution.

        mean = alpha / (alpha + beta)

        Returns
        -------
        float
            Mean value.
        """
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """
        Return the variance of the distribution.

        variance = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))

        Returns
        -------
        float
            Variance value.
        """
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab**2 * (ab + 1))

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
            True for Beta (bounded on [0, 1]).
        """
        return True

    def bounds(self) -> Tuple[float, float]:
        """
        Return the support bounds.

        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds (0, 1).
        """
        return (0.0, 1.0)

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

        d/dx log f(x) = (a-1)/x - (b-1)/(1-x)

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
        grad = (self._alpha - 1.0) / samples_1d - (self._beta - 1.0) / (1.0 - samples_1d)
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

    def __eq__(self, other: Any) -> bool:
        """Check equality with another BetaMarginal."""
        if not isinstance(other, BetaMarginal):
            return False
        return self.alpha == other.alpha and self.beta == other.beta

    def __repr__(self) -> str:
        """Return string representation."""
        return f"BetaMarginal(alpha={self.alpha}, beta={self.beta})"
