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
    Optional capabilities are checked via hasattr:
    - hyp_list() -> HyperParameterList: Parameters for optimization
    - logpdf_jacobian_wrt_x(x, y) -> Array: Gradient w.r.t. conditioning variable
    - logpdf_jacobian_wrt_params(x, y) -> Array: Gradient w.r.t. parameters

    Optional VI (variational inference) capabilities (checked via hasattr):
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
