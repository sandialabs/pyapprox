"""
Protocols for probability distributions.

These protocols define the interface for marginal and joint distributions
at different capability levels.

Protocol Hierarchy
------------------
DistributionProtocol
    Base protocol with sampling and logpdf.
MarginalProtocol
    Adds CDF, inverse CDF for 1D distributions.
MarginalWithJacobianProtocol
    Adds Jacobian of CDF for sensitivity analysis.
MarginalWithParamJacobianProtocol
    Adds Jacobian w.r.t. distribution parameters for MLE/VI.
JointDistributionProtocol
    Multivariate distribution with marginal access.
"""

from typing import Protocol, Generic, runtime_checkable, Sequence, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


@runtime_checkable
class DistributionProtocol(Protocol, Generic[Array]):
    """
    Base protocol for probability distributions.

    All distributions must support sampling and log-PDF evaluation.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Number of random variables.
    rvs(nsamples)
        Generate random samples.
    logpdf(samples)
        Evaluate log probability density function.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Return the number of random variables.

        Returns
        -------
        int
            Number of variables (dimension of the distribution).
        """
        ...

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
            Random samples. Shape: (nvars, nsamples)
        """
        ...

    def logpdf(self, samples: Array) -> Array:
        """
        Evaluate the log probability density function.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Log PDF values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        ...


@runtime_checkable
class MarginalProtocol(Protocol, Generic[Array]):
    """
    Protocol for 1D marginal distributions.

    Extends DistributionProtocol with CDF and inverse CDF for
    probability integral transforms.

    All marginal distributions must implement both `pdf()` and `__call__()`,
    where `__call__` is an alias for `pdf()`. This applies to both continuous
    and discrete marginals. When writing code that uses marginals, prefer
    `pdf()` for clarity.

    Derivatives: `jacobian()` returns the derivative of `pdf()` (equivalently,
    `__call__`) with respect to the input.

    Methods
    -------
    pdf(samples)
        Evaluate probability density function.
    __call__(samples)
        Alias for pdf(). Evaluate probability density function.
    cdf(samples)
        Evaluate cumulative distribution function.
    invcdf(probs)
        Evaluate inverse CDF (quantile function).
    is_bounded()
        Check if the distribution has bounded support.
    interval(alpha)
        Compute credible interval with given probability content.
    """

    def bkd(self) -> Backend[Array]:
        ...

    def nvars(self) -> int:
        ...

    def rvs(self, nsamples: int) -> Array:
        ...

    def logpdf(self, samples: Array) -> Array:
        ...

    def pdf(self, samples: Array) -> Array:
        """
        Evaluate the probability density function.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            PDF values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        ...

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the probability density function (alias for pdf).

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            PDF values. Shape: (1, nsamples)
        """
        ...

    def cdf(self, samples: Array) -> Array:
        """
        Evaluate the cumulative distribution function.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            CDF values in [0, 1]. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        ...

    def invcdf(self, probs: Array) -> Array:
        """
        Evaluate the inverse CDF (quantile function).

        Parameters
        ----------
        probs : Array
            Probability values in [0, 1]. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Quantile values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        ...

    def is_bounded(self) -> bool:
        """
        Check if the distribution has bounded support.

        Returns
        -------
        bool
            True if the distribution has finite lower and upper bounds.
        """
        ...

    def interval(self, alpha: float) -> Array:
        """
        Compute credible interval with given probability content.

        Parameters
        ----------
        alpha : float
            Probability content of the interval (0 < alpha <= 1).

        Returns
        -------
        Array
            Interval [lower, upper] such that P(lower < X < upper) = alpha.
            Shape: (1, 2)
        """
        ...


@runtime_checkable
class MarginalWithJacobianProtocol(Protocol, Generic[Array]):
    """
    Marginal distribution with Jacobian support for sensitivity analysis.

    The Jacobian of the inverse CDF is the reciprocal of the PDF:
        d(F^{-1})/dp = 1 / f(F^{-1}(p))

    Methods
    -------
    invcdf_jacobian(probs)
        Jacobian of inverse CDF.
    """

    def bkd(self) -> Backend[Array]:
        ...

    def nvars(self) -> int:
        ...

    def rvs(self, nsamples: int) -> Array:
        ...

    def logpdf(self, samples: Array) -> Array:
        ...

    def pdf(self, samples: Array) -> Array:
        ...

    def __call__(self, samples: Array) -> Array:
        ...

    def cdf(self, samples: Array) -> Array:
        ...

    def invcdf(self, probs: Array) -> Array:
        ...

    def is_bounded(self) -> bool:
        ...

    def interval(self, alpha: float) -> Array:
        ...

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
        ...


@runtime_checkable
class MarginalWithParamJacobianProtocol(Protocol, Generic[Array]):
    """
    Marginal distribution with parameter Jacobian for optimization.

    Extends MarginalWithJacobianProtocol with gradients w.r.t. distribution
    parameters. This enables gradient-based optimization for:
    - Maximum likelihood estimation (MLE)
    - Variational inference (VI)

    The HyperParameterList manages parameter transformations (e.g., log-space
    for positive parameters) to enable unconstrained optimization.

    Methods
    -------
    hyp_list()
        Return the hyperparameter list for parameter optimization.
    nparams()
        Return the number of distribution parameters.
    logpdf_jacobian_wrt_params(samples)
        Jacobian of log-PDF w.r.t. distribution parameters.
    """

    def bkd(self) -> Backend[Array]:
        ...

    def nvars(self) -> int:
        ...

    def rvs(self, nsamples: int) -> Array:
        ...

    def logpdf(self, samples: Array) -> Array:
        ...

    def pdf(self, samples: Array) -> Array:
        ...

    def __call__(self, samples: Array) -> Array:
        ...

    def cdf(self, samples: Array) -> Array:
        ...

    def invcdf(self, probs: Array) -> Array:
        ...

    def is_bounded(self) -> bool:
        ...

    def interval(self, alpha: float) -> Array:
        ...

    def invcdf_jacobian(self, probs: Array) -> Array:
        ...

    def hyp_list(self) -> HyperParameterList:
        """
        Return the hyperparameter list for parameter optimization.

        The hyperparameter list manages:
        - Parameter values (possibly in transformed space)
        - Parameter bounds
        - Active/fixed parameter selection

        Returns
        -------
        HyperParameterList
            Hyperparameter list containing distribution parameters.
        """
        ...

    def nparams(self) -> int:
        """
        Return the total number of distribution parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        ...

    def logpdf_jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute Jacobian of log-PDF w.r.t. distribution parameters.

        Gradients are computed in the optimizer's parameter space
        (i.e., log-space for log-transformed parameters).

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples) - must be 2D

        Returns
        -------
        Array
            Jacobian values. Shape: (nsamples, nparams)
            Each row contains gradients for one sample.

        Raises
        ------
        ValueError
            If input is not 2D or has wrong first dimension
        """
        ...


@runtime_checkable
class JointDistributionProtocol(Protocol, Generic[Array]):
    """
    Protocol for multivariate joint distributions.

    Provides access to marginal distributions and joint operations.

    Methods
    -------
    marginals()
        Return list of marginal distributions.
    correlation_matrix()
        Return the correlation matrix (if defined).
    """

    def bkd(self) -> Backend[Array]:
        ...

    def nvars(self) -> int:
        ...

    def rvs(self, nsamples: int) -> Array:
        ...

    def logpdf(self, samples: Array) -> Array:
        ...

    def marginals(self) -> Sequence[MarginalProtocol[Array]]:
        """
        Return the marginal distributions.

        Returns
        -------
        Sequence[MarginalProtocol]
            List of marginal distributions.
        """
        ...

    def correlation_matrix(self) -> Array:
        """
        Return the correlation matrix.

        Returns
        -------
        Array
            Correlation matrix. Shape: (nvars, nvars)
        """
        ...


@runtime_checkable
class UniformQuadratureRule01Protocol(Protocol, Generic[Array]):
    """
    Protocol for quadrature rules on [0, 1] with the Lebesgue measure.

    This protocol is for quadrature rules that approximate integrals of the form:
        integral_0^1 f(x) dx ≈ sum_i w_i f(x_i)

    where points x_i are in [0, 1] and weights w_i sum to 1.

    This is used by distributions that need numerical integration for CDF
    computation (e.g., BetaMarginal), where the integrand is the unnormalized
    PDF and the quadrature must be on [0, 1] with uniform weight.

    Methods
    -------
    bkd()
        Get the computational backend.
    __call__(npoints)
        Compute quadrature points and weights on [0, 1].
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def __call__(self, npoints: int) -> Tuple[Array, Array]:
        """
        Compute quadrature points and weights on [0, 1].

        The quadrature rule must satisfy:
        - Points are in [0, 1]
        - Weights sum to 1 (Lebesgue measure on [0, 1])
        - Approximates: integral_0^1 f(x) dx ≈ sum_i w_i f(x_i)

        Parameters
        ----------
        npoints : int
            Number of quadrature points.

        Returns
        -------
        points : Array
            Quadrature points in [0, 1]. Shape: (1, npoints)
        weights : Array
            Quadrature weights (sum to 1). Shape: (npoints, 1)
        """
        ...
