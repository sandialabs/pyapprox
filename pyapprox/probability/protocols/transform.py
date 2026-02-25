"""
Protocols for probability transforms.

These protocols define the interface for transformations between
probability spaces, including:
- Affine transforms
- Gaussian transforms (to/from standard normal)
- Nataf transform (correlated to independent)
- Rosenblatt transform (general joint to independent)

Protocol Hierarchy
------------------
TransformProtocol
    Base transform (forward only).
InvertibleTransformProtocol
    Bidirectional transform with inverse.
TransformWithJacobianProtocol
    Transform with Jacobian for change of variables.
"""

from typing import Protocol, Generic, runtime_checkable, Tuple

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class TransformProtocol(Protocol, Generic[Array]):
    """
    Protocol for forward probability transforms.

    A transform maps samples from one distribution to another.

    Methods
    -------
    bkd()
        Get the computational backend.
    nvars()
        Number of variables.
    map_to_canonical(samples)
        Transform to canonical (target) space.
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
            Number of variables.
        """
        ...

    def map_to_canonical(self, samples: Array) -> Array:
        """
        Transform samples to the canonical (target) space.

        For Gaussian transforms, canonical = standard normal.
        For uniform transforms, canonical = [0, 1]^d.

        Parameters
        ----------
        samples : Array
            Samples in original space. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Samples in canonical space. Shape: (nvars, nsamples)
        """
        ...


@runtime_checkable
class InvertibleTransformProtocol(Protocol, Generic[Array]):
    """
    Protocol for invertible probability transforms.

    Extends TransformProtocol with the inverse mapping.

    Methods
    -------
    map_from_canonical(canonical_samples)
        Transform from canonical (target) space back to original.
    """

    def bkd(self) -> Backend[Array]:
        ...

    def nvars(self) -> int:
        ...

    def map_to_canonical(self, samples: Array) -> Array:
        ...

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        """
        Transform samples from canonical space to original space.

        This is the inverse of map_to_canonical.

        Parameters
        ----------
        canonical_samples : Array
            Samples in canonical space. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Samples in original space. Shape: (nvars, nsamples)
        """
        ...


@runtime_checkable
class TransformWithJacobianProtocol(Protocol, Generic[Array]):
    """
    Transform with Jacobian for change of variables.

    The Jacobian is needed for:
    - Density transformation: p(x) = p(y) * |det(dy/dx)|
    - Gradient computation in optimization
    - Sensitivity analysis

    Methods
    -------
    map_to_canonical_with_jacobian(samples)
        Transform with Jacobian of the mapping.
    map_from_canonical_with_jacobian(canonical_samples)
        Inverse transform with Jacobian.
    """

    def bkd(self) -> Backend[Array]:
        ...

    def nvars(self) -> int:
        ...

    def map_to_canonical(self, samples: Array) -> Array:
        ...

    def map_from_canonical(self, canonical_samples: Array) -> Array:
        ...

    def map_to_canonical_with_jacobian(
        self, samples: Array
    ) -> Tuple[Array, Array]:
        """
        Transform to canonical space with Jacobian.

        Parameters
        ----------
        samples : Array
            Samples in original space. Shape: (nvars, nsamples)

        Returns
        -------
        Tuple[Array, Array]
            canonical_samples : Array
                Samples in canonical space. Shape: (nvars, nsamples)
            jacobian_diag : Array
                Diagonal of Jacobian (for separable transforms).
                Shape: (nvars, nsamples)
                For non-separable, returns full Jacobian: (nvars, nvars, nsamples)
        """
        ...

    def map_from_canonical_with_jacobian(
        self, canonical_samples: Array
    ) -> Tuple[Array, Array]:
        """
        Transform from canonical space with Jacobian.

        Parameters
        ----------
        canonical_samples : Array
            Samples in canonical space. Shape: (nvars, nsamples)

        Returns
        -------
        Tuple[Array, Array]
            samples : Array
                Samples in original space. Shape: (nvars, nsamples)
            jacobian_diag : Array
                Diagonal of Jacobian. Shape: (nvars, nsamples)
        """
        ...
