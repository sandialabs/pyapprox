"""
Protocols for conditional probability distributions.

Conditional distributions define p(y | x) where:
- x is the conditioning variable (e.g., input features)
- y is the random variable being modeled

The distribution parameters are functions of x, enabling heteroscedastic
models where variance (or other parameters) depends on the input.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


@runtime_checkable
class ConditionalDistributionProtocol(Protocol, Generic[Array]):
    """
    Protocol for conditional distributions p(y | x).

    A conditional distribution has parameters that depend on a conditioning
    variable x. For example, a conditional Gaussian might have:
        p(y | x) = N(y; mu(x), sigma(x)^2)
    where mu(x) and sigma(x) are functions (e.g., neural networks, polynomials).

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Number of conditioning variables (dimension of x).
    nqoi()
        Number of output variables (dimension of y).
    logpdf(x, y)
        Evaluate log probability density.
    rvs(x)
        Generate random samples given conditioning variable.

    Notes
    -----
    Optional capabilities are declared, never probed with hasattr:
    - hyp_list() / nparams() raise RuntimeError when unavailable; gate on
      has_hyp_list().
    - logpdf_jacobian_wrt_x(x, y) / logpdf_jacobian_wrt_params(x, y) raise
      RuntimeError when the component functions lack the required
      derivative capability; gate on has_logpdf_jacobian_wrt_x() /
      has_logpdf_jacobian_wrt_params(). (These are two-argument
      derivative families, so capability lives behind predicates rather
      than a Derivatives bundle.)

    Optional VI (variational inference) capabilities (checked via
    runtime protocol isinstance):
    - reparameterize(x, base_samples) -> Array: Transform base samples to
      distribution samples. Differentiable w.r.t. distribution parameters.
    - kl_divergence(x, prior) -> Array: Analytical KL(q(.|x) || prior),
      shape (1, nsamples).
    - base_distribution() -> marginal/joint: The base sampling distribution
      (e.g., N(0,1) for Gaussian, U(0,1) for Beta).
    - reparameterize_jacobian_wrt_params(x, base_samples) -> Array: Jacobian
      of reparameterize w.r.t. active parameters.

    The rvs(x) method returns one y sample per column of x. For multiple
    samples per conditioning point, the caller should repeat x:
        rvs(bkd.repeat(x, n, axis=1))
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Return the number of conditioning variables.

        Returns
        -------
        int
            Dimension of the conditioning variable x.
        """
        ...

    def nqoi(self) -> int:
        """
        Return the number of output variables.

        Returns
        -------
        int
            Dimension of the output variable y.
        """
        ...

    def logpdf(self, x: Array, y: Array) -> Array:
        """
        Evaluate the log probability density function.

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples) - must be 2D
        y : Array
            Output variable values. Shape: (nqoi, nsamples) - must be 2D

        Returns
        -------
        Array
            Log PDF values. Shape: (1, nsamples)

        Raises
        ------
        ValueError
            If inputs are not 2D or have mismatched sample counts
        """
        ...

    def rvs(self, x: Array) -> Array:
        """
        Generate random samples given conditioning variable.

        Returns one y sample per column of x. For multiple samples per
        conditioning point, repeat x before calling this method.

        Parameters
        ----------
        x : Array
            Conditioning variable values. Shape: (nvars, nsamples) - must be 2D

        Returns
        -------
        Array
            Random samples. Shape: (nqoi, nsamples)
            One sample per conditioning point.

        Raises
        ------
        ValueError
            If input is not 2D
        """
        ...


@runtime_checkable
class ComponentWithHypListProtocol(Protocol, Generic[Array]):
    """Component function exposing trainable hyperparameters."""

    def hyp_list(self) -> HyperParameterList[Array]:
        """Return the hyperparameter list."""
        ...


@runtime_checkable
class ComponentWithSyncParamsProtocol(Protocol):
    """Component function whose coefficients sync from its hyp_list."""

    def sync_params(self) -> None:
        """Sync internal coefficients from hyp_list values."""
        ...


@runtime_checkable
class ComponentWithParamJacobianProtocol(Protocol, Generic[Array]):
    """Component function providing a parameter Jacobian.

    Parameter-family derivatives are public named methods (the name
    carries the "wrt params" context), so this structural gate — not the
    Derivatives bundle, which covers derivatives w.r.t. inputs — is how
    consumers detect them.
    """

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """Compute Jacobian w.r.t. active parameters.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Jacobians. Shape: (nsamples, nqoi, nactive_params)
        """
        ...


@runtime_checkable
class ConditionalWithHypListProtocol(Protocol, Generic[Array]):
    """Conditional exposing (possibly unavailable) hyperparameters."""

    def has_hyp_list(self) -> bool:
        """Whether hyperparameters are available."""
        ...

    def hyp_list(self) -> HyperParameterList[Array]:
        """Return the hyperparameter list (raises when unavailable)."""
        ...


@runtime_checkable
class ConditionalWithXJacobianProtocol(Protocol, Generic[Array]):
    """Conditional exposing (possibly unavailable) d(logpdf)/dx."""

    def has_logpdf_jacobian_wrt_x(self) -> bool:
        """Whether the conditioning-variable jacobian is available."""
        ...

    def logpdf_jacobian_wrt_x(self, x: Array, y: Array) -> Array:
        """Compute d(logpdf)/dx (raises when unavailable)."""
        ...


@runtime_checkable
class ConditionalWithParamJacobianProtocol(Protocol, Generic[Array]):
    """Conditional exposing (possibly unavailable) d(logpdf)/dparams."""

    def has_logpdf_jacobian_wrt_params(self) -> bool:
        """Whether the parameter jacobian is available."""
        ...

    def logpdf_jacobian_wrt_params(self, x: Array, y: Array) -> Array:
        """Compute d(logpdf)/dparams (raises when unavailable)."""
        ...
