"""FieldMap protocol: maps parameter vector to spatial field."""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.typing.util.backends.protocols import Array


@runtime_checkable
class FieldMapProtocol(Protocol, Generic[Array]):
    """Protocol for field maps: parameter vector -> spatial field.

    Required methods:
        nvars() -> int
        __call__(params_1d) -> Array

    Optional methods (detected via hasattr):
        jacobian(params_1d) -> Array  shape (npts, nvars)
        hvp(params_1d, adj_state, vvec) -> Array  shape (nvars,)
    """

    def nvars(self) -> int: ...

    def __call__(self, params_1d: Array) -> Array: ...
