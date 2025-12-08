"""Protocols for sparse grid surrogates.

This module defines protocols for:
- SubspaceProtocol: Individual tensor product subspaces
- SparseGridProtocol: Sparse grid surrogates (combinations of subspaces)
- AdaptiveSparseGridProtocol: Adaptive sparse grids with refinement
"""

from typing import Generic, List, Optional, Protocol, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array, Backend


@runtime_checkable
class SubspaceProtocol(Protocol, Generic[Array]):
    """Protocol for tensor product subspace in sparse grid.

    A subspace represents a single tensor product of univariate bases,
    identified by a multi-index specifying the level in each dimension.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def get_index(self) -> Array:
        """Return the multi-index identifying this subspace.

        Returns
        -------
        Array
            Multi-index of shape (nvars,)
        """
        ...

    def nvars(self) -> int:
        """Return the number of variables."""
        ...

    def nsamples(self) -> int:
        """Return the number of samples in this subspace."""
        ...

    def get_samples(self) -> Array:
        """Return sample locations for this subspace.

        Returns
        -------
        Array
            Samples of shape (nvars, nsamples)
        """
        ...

    def get_values(self) -> Optional[Array]:
        """Return function values at samples, if set.

        Returns
        -------
        Optional[Array]
            Values of shape (nsamples, nqoi) or None if not set
        """
        ...

    def set_values(self, values: Array) -> None:
        """Set function values at samples.

        Parameters
        ----------
        values : Array
            Values of shape (nsamples, nqoi)
        """
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate subspace interpolant at given samples.

        Parameters
        ----------
        samples : Array
            Evaluation points of shape (nvars, npoints)

        Returns
        -------
        Array
            Interpolant values of shape (npoints, nqoi)
        """
        ...


@runtime_checkable
class SparseGridProtocol(Protocol, Generic[Array]):
    """Protocol for sparse grid surrogates.

    A sparse grid is a linear combination of tensor product subspaces
    using Smolyak combination coefficients.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of variables."""
        ...

    def nsubspaces(self) -> int:
        """Return the number of subspaces."""
        ...

    def get_subspaces(self) -> List[SubspaceProtocol[Array]]:
        """Return list of all subspaces.

        Returns
        -------
        List[SubspaceProtocol[Array]]
            All tensor product subspaces in the sparse grid
        """
        ...

    def get_smolyak_coefficients(self) -> Array:
        """Return Smolyak combination coefficients.

        Returns
        -------
        Array
            Coefficients of shape (nsubspaces,)
        """
        ...

    def get_samples(self) -> Array:
        """Return all unique sample locations.

        Returns
        -------
        Array
            Unique samples of shape (nvars, nsamples)
        """
        ...

    def set_values(self, values: Array) -> None:
        """Set function values at all unique samples.

        Parameters
        ----------
        values : Array
            Values of shape (nsamples, nqoi)
        """
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate sparse grid interpolant.

        Parameters
        ----------
        samples : Array
            Evaluation points of shape (nvars, npoints)

        Returns
        -------
        Array
            Interpolant values of shape (npoints, nqoi)
        """
        ...


@runtime_checkable
class AdaptiveSparseGridProtocol(Protocol, Generic[Array]):
    """Protocol for adaptive sparse grid surrogates.

    Adaptive sparse grids refine subspaces based on error indicators,
    following the step_samples/step_values pattern for incremental
    construction.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nvars(self) -> int:
        """Return the number of variables."""
        ...

    def step_samples(self) -> Optional[Array]:
        """Get samples for next refinement step.

        Returns
        -------
        Optional[Array]
            New samples of shape (nvars, nnew) or None if converged
        """
        ...

    def step_values(self, values: Array) -> None:
        """Provide function values for the samples from step_samples.

        Parameters
        ----------
        values : Array
            Values of shape (nnew, nqoi)
        """
        ...

    def error_estimate(self) -> float:
        """Return current error estimate.

        Returns
        -------
        float
            Estimated interpolation error
        """
        ...

    def get_candidate_subspaces(self) -> List[Array]:
        """Return indices of candidate subspaces for refinement.

        Returns
        -------
        List[Array]
            Multi-indices of candidate subspaces
        """
        ...

    def __call__(self, samples: Array) -> Array:
        """Evaluate adaptive sparse grid interpolant.

        Parameters
        ----------
        samples : Array
            Evaluation points of shape (nvars, npoints)

        Returns
        -------
        Array
            Interpolant values of shape (npoints, nqoi)
        """
        ...
