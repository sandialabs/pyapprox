"""
Protocols for observation/forward model operators.

Forward models (observation operators) map parameters to observables:
    observations = G(parameters) + noise

The protocols are tiered to allow users to implement only what they need:
- Base: Just callable evaluation
- WithJacobian: Adds first derivative
- WithHessian: Adds second derivative (Hessian-vector products)
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend

#TODO: Isnt this protcol really just FunctionProtocol with nobs instead of nqoi
# should we just use FunctionProtocol instead. Similarly should we
# swtich nobs to nqoi for protocols with derivatives so usual function
# api can be used with addition of additional methods needed like
# apply_jacobian_transpose

@runtime_checkable
class ObservationOperatorProtocol(Protocol, Generic[Array]):
    """
    Base protocol for observation operators (forward models).

    An observation operator maps model parameters to predicted observations.

    Examples
    --------
    Linear model: G(x) = A @ x + b
    Nonlinear model: G(x) = f(x) for some nonlinear f
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """
        Return the number of input parameters.

        Returns
        -------
        int
            Number of model parameters.
        """
        ...

    def nobs(self) -> int:
        """
        Return the number of observations (outputs).

        Returns
        -------
        int
            Number of predicted observations.
        """
        ...

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the forward model.

        Parameters
        ----------
        samples : Array
            Input parameters. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Model predictions. Shape: (nobs, nsamples)
        """
        ...


@runtime_checkable
class ObservationOperatorWithJacobianProtocol(Protocol, Generic[Array]):
    """
    Observation operator with Jacobian computation.

    Extends the base protocol with first-order derivative information.
    The Jacobian is the matrix of partial derivatives:
        J[i,j] = d(output_i) / d(input_j)
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """Return the number of input parameters."""
        ...

    def nobs(self) -> int:
        """Return the number of observations."""
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate the forward model."""
        ...

    def jacobian(self, sample: Array) -> Array:
        """
        Compute Jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Input parameters. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian matrix. Shape: (nobs, nvars)
        """
        ...

    def apply_jacobian(self, sample: Array, vec: Array) -> Array:
        """
        Apply Jacobian to a vector: J @ vec.

        More efficient than forming the full Jacobian for large problems.

        Parameters
        ----------
        sample : Array
            Input parameters. Shape: (nvars, 1)
        vec : Array
            Vector to multiply. Shape: (nvars, 1)

        Returns
        -------
        Array
            Result J @ vec. Shape: (nobs, 1)
        """
        ...

    def apply_jacobian_transpose(self, sample: Array, vec: Array) -> Array:
        """
        Apply Jacobian transpose to a vector: J.T @ vec.

        Parameters
        ----------
        sample : Array
            Input parameters. Shape: (nvars, 1)
        vec : Array
            Vector to multiply. Shape: (nobs, 1)

        Returns
        -------
        Array
            Result J.T @ vec. Shape: (nvars, 1)
        """
        ...


@runtime_checkable
class ObservationOperatorWithHessianProtocol(Protocol, Generic[Array]):
    """
    Observation operator with Hessian-vector product computation.

    Extends the Jacobian protocol with second-order derivative information.
    Full Hessian computation is avoided; instead we provide Hessian-vector
    products which are sufficient for most optimization and inference algorithms.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def nvars(self) -> int:
        """Return the number of input parameters."""
        ...

    def nobs(self) -> int:
        """Return the number of observations."""
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate the forward model."""
        ...

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample."""
        ...

    def apply_jacobian(self, sample: Array, vec: Array) -> Array:
        """Apply Jacobian to a vector."""
        ...

    def apply_jacobian_transpose(self, sample: Array, vec: Array) -> Array:
        """Apply Jacobian transpose to a vector."""
        ...

    def apply_hessian(self, sample: Array, vec: Array) -> Array:
        """
        Apply Hessian-vector product for each output.

        For scalar output f(x), computes H @ vec where H = d²f/dx².
        For vector output, computes sum_i H_i @ vec where H_i is the
        Hessian of the i-th output.

        Parameters
        ----------
        sample : Array
            Input parameters. Shape: (nvars, 1)
        vec : Array
            Vector to multiply. Shape: (nvars, 1)

        Returns
        -------
        Array
            Result sum_i H_i @ vec. Shape: (nvars, 1)
        """
        ...

    def apply_weighted_hessian(
        self, sample: Array, vec: Array, weights: Array
    ) -> Array:
        """
        Apply weighted Hessian-vector product.

        Computes sum_i weights[i] * H_i @ vec where H_i is the Hessian
        of the i-th output. Used in adjoint-based optimization.

        Parameters
        ----------
        sample : Array
            Input parameters. Shape: (nvars, 1)
        vec : Array
            Vector to multiply. Shape: (nvars, 1)
        weights : Array
            Weights for each output. Shape: (nobs, 1)

        Returns
        -------
        Array
            Weighted Hessian-vector product. Shape: (nvars, 1)
        """
        ...
