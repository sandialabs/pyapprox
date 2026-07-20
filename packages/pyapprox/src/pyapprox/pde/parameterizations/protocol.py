"""Protocols for parameterizations module."""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.protocols import Array


@runtime_checkable
class DerivativeMatrixBasisProtocol(Protocol, Generic[Array]):
    """Minimal basis interface needed by parameterization factories.

    Any TensorProductBasisProtocol or BasisProtocol satisfies this
    via structural subtyping.
    """

    def ndim(self) -> int: ...

    def derivative_matrix(self, order: int, dim: int) -> Array: ...


@runtime_checkable
class ParameterizationProtocol(Protocol, Generic[Array]):
    """Protocol for physics parameterizations.

    Maps a parameter vector to physics inputs. The physics is bound at
    construction — one parameterization instance serves ONE physics
    instance (ensembles construct one per physics); ``physics()``
    returns it so consumers can validate identity. Optional derivative
    capability is expressed through the :class:`ParamDerivatives` bundle
    returned by ``param_derivatives()`` — absence of a capability is a
    ``None`` field, never a missing attribute.
    """

    def nparams(self) -> int: ...

    def physics(self) -> object: ...

    def apply(self, params_1d: Array) -> None: ...

    def param_derivatives(self) -> ParamDerivatives[Array]: ...
