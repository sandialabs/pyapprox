"""The composed Galerkin spatial operator, F = F_Omega + F_Gamma.

``ComposedSpatialOperator`` joins an interior operator (a physics, or any
object satisfying ``GalerkinInteriorOperatorProtocol``) with the natural-BC
terms (``NaturalBCOperator``):

    spatial_residual = interior_residual + sum_k c_k
    spatial_jacobian = interior_jacobian + sum_k dc_k/du

It satisfies ``SpatialOperatorProtocol``. This is the one place the
natural-BC part of ``F`` is added, and it needs no base class: a new
physics implements only its interior and is composed here.
"""

from typing import Generic

from pyapprox.pde.boundary import NaturalBCOperator
from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinInteriorOperatorProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


class ComposedSpatialOperator(Generic[Array]):
    """``F = F_Omega + F_Gamma`` from an interior and natural-BC terms.

    Parameters
    ----------
    interior : GalerkinInteriorOperatorProtocol
        The interior operator ``F_Omega``; it never sees a BC.
    natural_bcs : NaturalBCOperator
        The natural-BC terms ``F_Gamma = sum_k c_k``; may be empty.
    """

    def __init__(
        self,
        interior: GalerkinInteriorOperatorProtocol[Array],
        natural_bcs: NaturalBCOperator[Array],
    ) -> None:
        if not isinstance(interior, GalerkinInteriorOperatorProtocol):
            raise TypeError(
                "interior must satisfy GalerkinInteriorOperatorProtocol, got "
                f"{type(interior).__name__}"
            )
        if not isinstance(natural_bcs, NaturalBCOperator):
            raise TypeError(
                "natural_bcs must be a NaturalBCOperator, got "
                f"{type(natural_bcs).__name__}"
            )
        self._interior = interior
        self._natural_bcs = natural_bcs

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._interior.bkd()

    def nstates(self) -> int:
        """Return the number of states."""
        return self._interior.nstates()

    def interior(self) -> GalerkinInteriorOperatorProtocol[Array]:
        """Return the interior operator."""
        return self._interior

    def natural_bcs(self) -> NaturalBCOperator[Array]:
        """Return the natural-BC operator."""
        return self._natural_bcs

    def spatial_residual(self, state: Array, time: float) -> Array:
        """Compute ``F = F_Omega + F_Gamma``. Shape: (nstates,)."""
        return self._natural_bcs.add_to_residual(
            self._interior.interior_residual(state, time), state, time
        )

    def spatial_jacobian(self, state: Array, time: float) -> Array:
        """Compute ``dF/du = dF_Omega/du + dF_Gamma/du``.

        Shape: (nstates, nstates); sparse when the interior's is.
        """
        return self._natural_bcs.add_to_jacobian(
            self._interior.interior_jacobian(state, time), state, time
        )

    def __repr__(self) -> str:
        return (
            f"ComposedSpatialOperator(interior={type(self._interior).__name__}, "
            f"natural_bcs={self._natural_bcs!r})"
        )
