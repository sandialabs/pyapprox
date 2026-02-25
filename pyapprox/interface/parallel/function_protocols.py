"""Protocol definitions for parallel function capabilities.

This module defines protocols for functions that support parallel
batch execution of derivatives.
"""

from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class ParallelFunctionProtocol(Protocol, Generic[Array]):
    """Protocol for functions with parallel batch support.

    Extends basic function protocol with parallel execution capabilities.
    """

    def bkd(self) -> Backend[Array]:
        """Return the array backend."""
        ...

    def nvars(self) -> int:
        """Return number of input variables."""
        ...

    def nqoi(self) -> int:
        """Return number of outputs (quantities of interest)."""
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate function at samples.

        Parameters
        ----------
        samples : Array
            Input samples, shape (nvars, nsamples).

        Returns
        -------
        Array
            Function values, shape (nqoi, nsamples).
        """
        ...

    def parallel_backend(self) -> Optional[str]:
        """Return name of parallel backend, or None if sequential."""
        ...

    def n_workers(self) -> int:
        """Return number of parallel workers."""
        ...


@runtime_checkable
class ParallelFunctionWithJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for functions with parallel jacobian batch support.

    Extends ParallelFunctionProtocol with jacobian methods.
    """

    def bkd(self) -> Backend[Array]:
        """Return the array backend."""
        ...

    def nvars(self) -> int:
        """Return number of input variables."""
        ...

    def nqoi(self) -> int:
        """Return number of outputs (quantities of interest)."""
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate function at samples."""
        ...

    def jacobian(self, sample: Array) -> Array:
        """Compute jacobian at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample, shape (nvars, 1).

        Returns
        -------
        Array
            Jacobian matrix, shape (nqoi, nvars).
        """
        ...

    def jacobian_batch(self, samples: Array) -> Array:
        """Compute jacobians at multiple samples.

        Parameters
        ----------
        samples : Array
            Samples, shape (nvars, nsamples).

        Returns
        -------
        Array
            Jacobians, shape (nsamples, nqoi, nvars).
        """
        ...


@runtime_checkable
class ParallelFunctionWithHVPProtocol(Protocol, Generic[Array]):
    """Protocol for functions with parallel HVP batch support.

    Requires nqoi == 1 for Hessian-vector products.
    """

    def bkd(self) -> Backend[Array]:
        """Return the array backend."""
        ...

    def nvars(self) -> int:
        """Return number of input variables."""
        ...

    def nqoi(self) -> int:
        """Return number of outputs (must be 1 for HVP)."""
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate function at samples."""
        ...

    def hvp(self, sample: Array, vec: Array) -> Array:
        """Compute Hessian-vector product at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample, shape (nvars, 1).
        vec : Array
            Direction vector, shape (nvars, 1).

        Returns
        -------
        Array
            HVP result, shape (nvars, 1).
        """
        ...

    def hvp_batch(self, samples: Array, vecs: Array) -> Array:
        """Compute HVPs at multiple samples.

        Parameters
        ----------
        samples : Array
            Samples, shape (nvars, nsamples).
        vecs : Array
            Direction vectors, shape (nvars, nsamples).

        Returns
        -------
        Array
            HVP results, shape (nsamples, nvars).
        """
        ...


@runtime_checkable
class ParallelFunctionWithWHVPProtocol(Protocol, Generic[Array]):
    """Protocol for functions with parallel weighted HVP batch support.

    Supports any nqoi with weighted combination of Hessians.
    """

    def bkd(self) -> Backend[Array]:
        """Return the array backend."""
        ...

    def nvars(self) -> int:
        """Return number of input variables."""
        ...

    def nqoi(self) -> int:
        """Return number of outputs."""
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate function at samples."""
        ...

    def whvp(self, sample: Array, vec: Array, weights: Array) -> Array:
        """Compute weighted Hessian-vector product at a single sample.

        Parameters
        ----------
        sample : Array
            Single sample, shape (nvars, 1).
        vec : Array
            Direction vector, shape (nvars, 1).
        weights : Array
            Weights for each QoI, shape (nqoi, 1).

        Returns
        -------
        Array
            Weighted HVP result, shape (nvars, 1).
        """
        ...

    def whvp_batch(
        self, samples: Array, vecs: Array, weights: Array
    ) -> Array:
        """Compute weighted HVPs at multiple samples.

        Parameters
        ----------
        samples : Array
            Samples, shape (nvars, nsamples).
        vecs : Array
            Direction vectors, shape (nvars, nsamples).
        weights : Array
            Weights for each QoI, shape (nqoi, 1).

        Returns
        -------
        Array
            Weighted HVP results, shape (nsamples, nvars).
        """
        ...
