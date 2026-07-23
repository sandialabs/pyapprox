"""FieldMap protocol: maps parameter vector to spatial field."""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.util.backends.protocols import Array


@runtime_checkable
class FieldMapProtocol(Protocol, Generic[Array]):
    """Protocol for field maps: parameter vector -> spatial field.

    ``jacobian`` is required: every field map is an analytic leaf of the
    parameterization chain, and a missing jacobian would silently drop
    derivative capability from every consumer above it (ending in
    finite differences of full PDE solves). Field maps use the pde
    1D-array convention, so their derivatives are named methods rather
    than a Derivatives bundle; optional capability (hvp) is declared by
    FieldMapWithHVPProtocol.
    """

    def nvars(self) -> int: ...

    def __call__(self, params_1d: Array) -> Array: ...

    def jacobian(self, params_1d: Array) -> Array:
        """Compute d(field)/d(params). Shape: (npts, nvars)."""
        ...


@runtime_checkable
class FieldMapWithHVPProtocol(FieldMapProtocol[Array], Protocol):
    """Field map additionally providing an adjoint-weighted HVP."""

    def hvp(self, params_1d: Array, adj_state: Array, vvec: Array) -> Array:
        """Compute adj_state-weighted Hessian-vector product.

        Shape: (nvars,).
        """
        ...


@runtime_checkable
class GuardedHVPFieldMapProtocol(FieldMapWithHVPProtocol[Array], Protocol):
    """Field map whose hvp availability is guarded (TransformedFieldMap
    exposes ``hvp`` structurally but honors it only when constructed
    with a second transform derivative)."""

    def has_hvp(self) -> bool: ...


def field_map_has_hvp(field_map: FieldMapProtocol[Array]) -> bool:
    """Whether the field map provides a USABLE adjoint-weighted HVP."""
    if isinstance(field_map, GuardedHVPFieldMapProtocol):
        return field_map.has_hvp()
    return isinstance(field_map, FieldMapWithHVPProtocol)
