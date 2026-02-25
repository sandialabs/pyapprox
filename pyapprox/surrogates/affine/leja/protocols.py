"""Protocols for Leja sequence generation.

This module defines protocols for:
- Weighting strategies for Leja optimization
- Univariate Leja sequence generation
- Multivariate Leja sampling

Note: Leja optimization uses optimizers from pyapprox.optimization.minimize.
The default is ScipyTrustConstrOptimizer. Custom optimizers can be provided via
a factory callable that accepts (objective, bounds) and returns an optimizer.
"""

from typing import Callable, Generic, List, Optional, Protocol, Tuple, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class LejaWeightingProtocol(Protocol, Generic[Array]):
    """Protocol for Leja weighting strategies.

    Weighting strategies determine how candidate points are weighted
    during Leja sequence optimization.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def __call__(self, samples: Array, basis_values: Array) -> Array:
        """Compute weights for samples.

        Parameters
        ----------
        samples : Array
            Sample locations. Shape: (1, nsamples) for univariate
        basis_values : Array
            Basis function values at samples. Shape: (nsamples, nterms)

        Returns
        -------
        Array
            Weights for each sample. Shape: (nsamples, 1)
        """
        ...


@runtime_checkable
class LejaWeightingWithJacobianProtocol(LejaWeightingProtocol[Array], Protocol):
    """Protocol for Leja weighting with Jacobian support."""

    def jacobian(self, samples: Array, basis_values: Array, basis_jacobians: Array) -> Array:
        """Compute Jacobian of weights with respect to samples.

        Parameters
        ----------
        samples : Array
            Sample locations. Shape: (1, nsamples)
        basis_values : Array
            Basis function values at samples. Shape: (nsamples, nterms)
        basis_jacobians : Array
            Jacobians of basis functions. Shape: (nsamples, nterms)

        Returns
        -------
        Array
            Weight Jacobians. Shape: (nsamples, 1)
        """
        ...


@runtime_checkable
class LejaSequence1DProtocol(Protocol, Generic[Array]):
    """Protocol for univariate Leja sequences.

    Univariate Leja sequences generate nested sequences of points
    on a 1D domain that are optimal for polynomial interpolation.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def quadrature_rule(self, npoints: int) -> Tuple[Array, Array]:
        """Get Leja sequence with specified number of points.

        Parameters
        ----------
        npoints : int
            Number of points in sequence.

        Returns
        -------
        Tuple[Array, Array]
            (samples, weights) with shapes (1, npoints) and (npoints, 1)
        """
        ...

    def extend(self, n_new_points: int) -> None:
        """Extend sequence by adding new points.

        Parameters
        ----------
        n_new_points : int
            Number of new points to add.
        """
        ...

    def npoints(self) -> int:
        """Return current number of points in sequence."""
        ...

    def clear_cache(self) -> None:
        """Clear cached sequence data."""
        ...


@runtime_checkable
class LejaSamplerProtocol(Protocol, Generic[Array]):
    """Protocol for multivariate Leja-based samplers.

    Multivariate Leja samplers select points from a candidate set
    using pivoted factorization techniques.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def sample(self, nsamples: int) -> Array:
        """Generate Leja samples.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.

        Returns
        -------
        Array
            Selected samples. Shape: (nvars, nsamples)
        """
        ...

    def sample_incremental(self, n_new_samples: int) -> Array:
        """Add new samples incrementally.

        Parameters
        ----------
        n_new_samples : int
            Number of new samples to add.

        Returns
        -------
        Array
            The new samples only. Shape: (nvars, n_new_samples)
        """
        ...

    def nsamples(self) -> int:
        """Return current number of samples."""
        ...

    def get_selected_indices(self) -> Array:
        """Return indices of selected samples from candidate set."""
        ...


@runtime_checkable
class FeketeSamplerProtocol(Protocol, Generic[Array]):
    """Protocol for Fekete point samplers.

    Fekete samplers select all points at once (not incrementally)
    using pivoted QR factorization.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def sample(self, nsamples: int) -> Array:
        """Select Fekete points.

        Parameters
        ----------
        nsamples : int
            Number of points to select.

        Returns
        -------
        Array
            Selected samples. Shape: (nvars, nsamples)
        """
        ...

    def get_selected_indices(self) -> Array:
        """Return indices of selected samples from candidate set."""
        ...
