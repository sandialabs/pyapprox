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
class LinearFieldMapProtocol(FieldMapProtocol[Array], Protocol):
    """A field map declaring it is linear in its parameters.

    Only a linear map may carry a temporal modulation. For a linear map
    the separable form :math:`\\sum_k p_k b_k(t) s_k(x)` is exact, so the
    modulation is a per-column scaling of a constant jacobian. Compose a
    modulation with a POINTWISE NONLINEAR map and the two do not
    commute: :math:`\\exp(\\sum_k p_k b_k(t) s_k)` is not
    :math:`b(t)\\exp(\\sum_k p_k s_k)`, so scaling the jacobian columns
    would describe a field the forward solve never evaluates.

    Declared rather than inferred: nothing can detect linearity from a
    callable, and the failure is a silently wrong field rather than an
    error. Consumers requiring the guarantee ``isinstance``-check this at
    CONSTRUCTION, so no capability sniffing happens during assembly.
    """

    def is_linear(self) -> bool:
        """Whether the map is linear in ``params_1d``."""
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


def validate_params_1d(params_1d: Array, nvars: int) -> None:
    """Reject a parameter vector that is not 1D of length ``nvars``.

    Field maps take a 1-D vector, unlike most of the library, where a
    sample is a ``(nvars, 1)`` column. That makes the column the natural
    mistake, and an unchecked one is silent: a map evaluating
    ``mean_field + W @ params`` broadcasts a ``(npts,)`` mean against a
    ``(npts, 1)`` product and returns a ``(npts, npts)`` matrix. Every
    value in it is finite and plausible, so the error surfaces far away
    -- as a coefficient that appears to be zero somewhere it is indexed,
    or as a solve that quietly uses the wrong field.

    Called at the top of ``__call__`` rather than left to each map,
    since the failure mode is shared and the check costs two
    comparisons.

    Parameters
    ----------
    params_1d : Array
        The parameter vector to check.
    nvars : int
        The number of parameters the map expects.

    Raises
    ------
    ValueError
        If ``params_1d`` is not 1D, or its length is not ``nvars``.
    """
    if params_1d.ndim != 1:
        raise ValueError(
            f"params must be 1D with shape ({nvars},), got ndim="
            f"{params_1d.ndim} with shape {tuple(params_1d.shape)}. A "
            "field map takes a bare parameter vector rather than the "
            "(nvars, 1) column used elsewhere; passing a column "
            "broadcasts into a square field instead of raising."
        )
    if int(params_1d.shape[0]) != nvars:
        raise ValueError(
            f"params has {int(params_1d.shape[0])} entries but this "
            f"field map takes {nvars}"
        )
