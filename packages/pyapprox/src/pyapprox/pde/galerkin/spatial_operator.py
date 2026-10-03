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

from typing import Generic, List, Optional, Sequence

from pyapprox.ode.state_derivatives import StateDerivatives, StateStateHVPFn
from pyapprox.pde.boundary import NaturalBCOperator
from pyapprox.pde.galerkin.protocols.physics import (
    GalerkinInteriorOperatorProtocol,
)
from pyapprox.pde.ownership import owns
from pyapprox.util.backends.protocols import Array, Backend


class _SummedStateStateHVP(Generic[Array]):
    """The interior curvature plus each term's, as one contraction."""

    def __init__(
        self,
        interior_hvp: StateStateHVPFn[Array],
        term_hvps: Sequence[StateStateHVPFn[Array]],
    ) -> None:
        self._interior_hvp = interior_hvp
        self._term_hvps = tuple(term_hvps)

    def __call__(
        self, state: Array, adj_state: Array, wvec: Array, time: float
    ) -> Array:
        result = self._interior_hvp(state, adj_state, wvec, time)
        for term_hvp in self._term_hvps:
            result = result + term_hvp(state, adj_state, wvec, time)
        return result


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
        self._state_derivatives: Optional[StateDerivatives[Array]] = None

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

    def is_time_invariant(self) -> bool:
        """Whether the interior and every term are declared
        time-independent."""
        return self._interior.interior_is_time_invariant() and all(
            term.is_time_invariant() for term in self._natural_bcs.terms()
        )

    def owns(self, target: object) -> bool:
        """Whether ``target`` is part of ``F``: the interior (and what it
        owns) or one of the terms."""
        return (
            owns(self._interior, target)
            or any(target is term for term in self._natural_bcs.terms())
        )

    def state_derivatives(self) -> StateDerivatives[Array]:
        """Return ``d^2 F/du^2 = d^2 F_Omega/du^2 + sum_k d^2 c_k/du^2``.

        Present only when the interior and EVERY term supply it: a part
        without curvature would otherwise be silently dropped from the
        sum. Composed once, at the first call.
        """
        if self._state_derivatives is None:
            self._state_derivatives = self._compose_state_derivatives()
        return self._state_derivatives

    def _compose_state_derivatives(self) -> StateDerivatives[Array]:
        interior_hvp = self._interior.interior_state_derivatives().state_state_hvp
        term_hvps: List[StateStateHVPFn[Array]] = []
        for term in self._natural_bcs.terms():
            term_hvp = term.state_derivatives().state_state_hvp
            if term_hvp is None:
                return StateDerivatives.none()
            term_hvps.append(term_hvp)
        if interior_hvp is None:
            return StateDerivatives.none()
        if not term_hvps:
            return StateDerivatives.second_order(interior_hvp)
        return StateDerivatives.second_order(
            _SummedStateStateHVP(interior_hvp, term_hvps)
        )

    def __repr__(self) -> str:
        return (
            f"ComposedSpatialOperator(interior={type(self._interior).__name__}, "
            f"natural_bcs={self._natural_bcs!r})"
        )
