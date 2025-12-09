"""
Protocols for log unnormalized posterior.

The log unnormalized posterior combines likelihood and prior:
    log p(theta | data) = log p(data | theta) + log p(theta) + const

This is used for:
- MAP estimation via optimization
- MCMC sampling
- Laplace approximation (via Hessian at MAP)
"""

from typing import Protocol, Generic, runtime_checkable, Optional

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class LogUnNormalizedPosteriorProtocol(Protocol, Generic[Array]):
    """
    Protocol for log unnormalized posterior.

    Combines a log-likelihood and log-prior to give:
        log p(theta | data) propto log p(data | theta) + log p(theta)

    Provides methods for:
    - Evaluation: __call__(samples)
    - Gradient: jacobian(sample) for optimization/HMC
    - Hessian: hessian(sample) for Laplace approximation
    - MAP finding: maximum_aposteriori_point()
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
            Number of model parameters.
        """
        ...

    def __call__(self, samples: Array) -> Array:
        """
        Evaluate the log unnormalized posterior.

        Parameters
        ----------
        samples : Array
            Parameter samples. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Log posterior values. Shape: (nsamples,)
        """
        ...

    def jacobian(self, sample: Array) -> Array:
        """
        Compute gradient of log posterior at a single sample.

        Parameters
        ----------
        sample : Array
            Parameter sample. Shape: (nvars, 1)

        Returns
        -------
        Array
            Gradient. Shape: (nvars, 1)
        """
        ...

    def hessian(self, sample: Array) -> Array:
        """
        Compute Hessian of log posterior at a single sample.

        Parameters
        ----------
        sample : Array
            Parameter sample. Shape: (nvars, 1)

        Returns
        -------
        Array
            Hessian matrix. Shape: (nvars, nvars)
        """
        ...

    def apply_hessian(self, sample: Array, vec: Array) -> Array:
        """
        Apply Hessian-vector product.

        More efficient than forming full Hessian for large problems.

        Parameters
        ----------
        sample : Array
            Parameter sample. Shape: (nvars, 1)
        vec : Array
            Vector to multiply. Shape: (nvars, 1)

        Returns
        -------
        Array
            Hessian-vector product. Shape: (nvars, 1)
        """
        ...

    def maximum_aposteriori_point(
        self, initial_guess: Optional[Array] = None
    ) -> Array:
        """
        Find the maximum a posteriori (MAP) point.

        Parameters
        ----------
        initial_guess : Array, optional
            Initial point for optimization. Shape: (nvars, 1)
            If None, uses prior mean or zeros.

        Returns
        -------
        Array
            MAP estimate. Shape: (nvars, 1)
        """
        ...
