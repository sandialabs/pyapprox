"""Field and operator protocols for spectral collocation methods.

Defines interfaces for fields (scalar and vector) on meshes and
differential operators.
"""

from typing import Protocol, Generic, runtime_checkable, Optional

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class FieldProtocol(Protocol, Generic[Array]):
    """Protocol for fields on a mesh.

    A field represents values at mesh points with a specified number
    of components. Scalar fields have ncomponents=1, vector fields
    have ncomponents=ndim.

    Shape convention: (ncomponents, npts)
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def npts(self) -> int:
        """Return the number of mesh points."""
        ...

    def ncomponents(self) -> int:
        """Return the number of field components."""
        ...

    def values(self) -> Array:
        """Return field values.

        Returns
        -------
        Array
            Field values of shape (ncomponents, npts).
        """
        ...

    def is_scalar(self) -> bool:
        """Return True if this is a scalar field (ncomponents=1)."""
        ...

    def as_flat(self) -> Array:
        """Return flattened values.

        Returns
        -------
        Array
            For scalar: shape (npts,)
            For vector: shape (ncomponents * npts,)
        """
        ...


@runtime_checkable
class FieldWithJacobianProtocol(Protocol, Generic[Array]):
    """Protocol for fields with Jacobian dependency tracking.

    Tracks how field values depend on input fields for computing
    derivatives of composed operations.
    """

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def npts(self) -> int:
        """Return the number of mesh points."""
        ...

    def ncomponents(self) -> int:
        """Return the number of field components."""
        ...

    def values(self) -> Array:
        """Return field values. Shape: (ncomponents, npts)."""
        ...

    def is_scalar(self) -> bool:
        """Return True if this is a scalar field."""
        ...

    def as_flat(self) -> Array:
        """Return flattened values."""
        ...

    def has_jacobian(self) -> bool:
        """Return True if Jacobian tracking is enabled."""
        ...

    def jacobian(self) -> Optional[Array]:
        """Return Jacobian with respect to input field.

        Returns
        -------
        Optional[Array]
            Jacobian matrix of shape (ncomponents * npts, ninputs * npts)
            or None if not tracking.
        """
        ...


@runtime_checkable
class DifferentialOperatorProtocol(Protocol, Generic[Array]):
    """Protocol for differential operators on fields.

    Operators like gradient, divergence, and Laplacian that act on fields.
    """

    def __call__(self, field: FieldProtocol[Array]) -> FieldProtocol[Array]:
        """Apply operator to field.

        Parameters
        ----------
        field : FieldProtocol
            Input field.

        Returns
        -------
        FieldProtocol
            Result field.
        """
        ...

    def jacobian(self, field: FieldProtocol[Array]) -> Array:
        """Compute Jacobian of operator output with respect to input.

        Parameters
        ----------
        field : FieldProtocol
            Input field (may be needed for nonlinear operators).

        Returns
        -------
        Array
            Jacobian matrix. Shape depends on input/output dimensions.
        """
        ...
