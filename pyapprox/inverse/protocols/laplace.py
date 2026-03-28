"""
Protocols for Laplace approximation.

The Laplace approximation approximates a posterior distribution as Gaussian
centered at the maximum a posteriori (MAP) point with covariance equal to
the inverse Hessian of the negative log-posterior at the MAP.

This is distinct from conjugate Gaussian priors which give exact posteriors
for linear models. Laplace approximation is used when the forward model
is nonlinear.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class LaplacePosteriorProtocol(Protocol, Generic[Array]):
    """
    Protocol for Laplace approximation posteriors.

    The Laplace approximation gives:
        p(theta | data) ≈ N(theta_MAP, H^{-1})

    where:
        - theta_MAP is the maximum a posteriori estimate
        - H is the Hessian of the negative log-posterior at theta_MAP
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Return the number of variables.

        Returns
        -------
        int
            Number of posterior variables.
        """
        ...

    def compute(self, **kwargs: object) -> None:
        """
        Compute the Laplace approximation.

        This finds the MAP point and computes the Hessian at that point.
        Implementation-specific keyword arguments may control optimization.
        """
        ...

    def map_point(self) -> Array:
        """
        Return the MAP (maximum a posteriori) point.

        Returns
        -------
        Array
            MAP estimate. Shape: (nvars, 1)
        """
        ...

    def posterior_covariance(self) -> Array:
        """
        Return the posterior covariance.

        For full-rank approximation, this is the inverse Hessian.
        For low-rank approximation, this may be a low-rank plus diagonal form.

        Returns
        -------
        Array
            Posterior covariance. Shape: (nvars, nvars)
        """
        ...

    def covariance_diagonal(self) -> Array:
        """
        Return the diagonal of the posterior covariance.

        More efficient than computing full covariance for large problems.

        Returns
        -------
        Array
            Diagonal of posterior covariance. Shape: (nvars,)
        """
        ...


@runtime_checkable
class HessianMatVecOperatorProtocol(Protocol, Generic[Array]):
    """
    Protocol for Hessian matrix-vector product operators.

    Instead of forming the full Hessian matrix (which can be expensive
    for high-dimensional problems), this protocol provides matrix-vector
    products: H @ v.

    This is used by low-rank Laplace approximation methods that use
    randomized SVD to find the dominant eigenspace of the Hessian.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Return the dimension of the operator.

        Returns
        -------
        int
            Number of variables (operator is nvars x nvars).
        """
        ...

    def apply(self, vecs: Array) -> Array:
        """
        Apply Hessian to vectors.

        Parameters
        ----------
        vecs : Array
            Vectors to multiply. Shape: (nvars, nvecs)

        Returns
        -------
        Array
            Result H @ vecs. Shape: (nvars, nvecs)
        """
        ...

    def apply_transpose(self, vecs: Array) -> Array:
        """
        Apply Hessian transpose to vectors.

        For symmetric Hessians (which is typical), this is the same as apply().

        Parameters
        ----------
        vecs : Array
            Vectors to multiply. Shape: (nvars, nvecs)

        Returns
        -------
        Array
            Result H.T @ vecs. Shape: (nvars, nvecs)
        """
        ...


@runtime_checkable
class PriorConditionedHessianProtocol(Protocol, Generic[Array]):
    """
    Protocol for prior-conditioned Hessian operators.

    For efficient low-rank approximation, we work with the
    prior-conditioned Hessian: L @ H_misfit @ L.T

    where:
        - L is the Cholesky factor of the prior covariance
        - H_misfit is the Hessian of the data misfit (negative log-likelihood)

    The eigenvalues of this operator indicate how much each direction
    in parameter space is informed by the data relative to the prior.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """Return the dimension of the operator."""
        ...

    def apply(self, vecs: Array) -> Array:
        """
        Apply prior-conditioned Hessian: L @ H @ L.T @ vecs.

        Parameters
        ----------
        vecs : Array
            Vectors to multiply. Shape: (nvars, nvecs)

        Returns
        -------
        Array
            Result. Shape: (nvars, nvecs)
        """
        ...
