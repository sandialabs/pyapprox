"""Protocols for parameterizations module.

Third-party solver modules plug in through the four-seam contract
documented in ``docs/conventions/pde_solver_extension.md``.
"""

from typing import Generic, Protocol, Tuple, runtime_checkable

from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.protocols import Array, Array_co


@runtime_checkable
class DerivativeMatrixBasisProtocol(Protocol, Generic[Array_co]):
    """Minimal basis interface needed by parameterization factories.

    Any TensorProductBasisProtocol or BasisProtocol satisfies this
    via structural subtyping. ``Array`` appears only in return
    position, so the protocol is covariant.
    """

    def ndim(self) -> int: ...

    def derivative_matrix(self, order: int, dim: int) -> Array_co: ...


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

    def owned_coefficients(self) -> Tuple[str, ...]:
        """Identifiers of the physics coefficient fields ``apply``
        writes.

        ``CompositeParameterization`` rejects parts with overlapping
        identifiers: two parts writing the same coefficient would be
        last-writer-wins in ``apply`` while both still report nonzero
        derivative blocks — silently wrong numbers. Identifiers must
        be consistent across all parameterizations of one physics
        family (e.g. ``"diffusion"``, ``"mu"``, ``"lamda"``).
        """
        ...

    def apply(self, params_1d: Array) -> None: ...

    def param_derivatives(self) -> ParamDerivatives[Array]: ...
