"""Composed Galerkin systems: spatial operator, constraints, and mass.

``GalerkinSteadySystem`` holds the spatial operator and the essential
constraint set; ``GalerkinSystem`` adds the mass. Both take the parts
already composed --- never a physics plus a BC list --- so anything that
satisfies the part protocols can be solved, with no base class.
``compose_galerkin_system`` builds the parts from a physics and its
boundary conditions.
"""

from typing import Generic

from pyapprox.ode.protocols import SpatialOperatorProtocol
from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.galerkin.protocols.system import GalerkinMassProtocol
from pyapprox.pde.ownership import owns
from pyapprox.pde.steady_view import SteadyView
from pyapprox.util.backends.protocols import Array, Backend


class GalerkinSteadySystem(Generic[Array]):
    """The parts of a steady problem: ``F`` and the essential constraints.

    Satisfies ``GalerkinSteadySystemProtocol``.

    Parameters
    ----------
    spatial_operator : SpatialOperatorProtocol
        ``F``, the interior composed with the natural-BC terms.
    constraint_set : ConstraintSetProtocol
        The essential constraints on rows of ``F``.

    Raises
    ------
    TypeError
        If a part does not satisfy its protocol.
    ValueError
        If a constrained DOF lies outside the operator's states.
    """

    def __init__(
        self,
        spatial_operator: SpatialOperatorProtocol[Array],
        constraint_set: ConstraintSetProtocol[Array],
    ) -> None:
        if not isinstance(spatial_operator, SpatialOperatorProtocol):
            raise TypeError(
                "spatial_operator must satisfy SpatialOperatorProtocol, got "
                f"{type(spatial_operator).__name__}"
            )
        if not isinstance(constraint_set, ConstraintSetProtocol):
            raise TypeError(
                "constraint_set must satisfy ConstraintSetProtocol, got "
                f"{type(constraint_set).__name__}"
            )
        nstates = spatial_operator.nstates()
        if constraint_set.ndofs() > 0:
            bkd = spatial_operator.bkd()
            largest = int(bkd.to_float(bkd.max(constraint_set.dofs())))
            if largest >= nstates:
                raise ValueError(
                    f"constraint set constrains DOF {largest}, but the "
                    f"spatial operator has {nstates} states"
                )
        self._spatial_operator = spatial_operator
        self._constraint_set = constraint_set

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._spatial_operator.bkd()

    def nstates(self) -> int:
        """Return the number of states."""
        return self._spatial_operator.nstates()

    def spatial_operator(self) -> SpatialOperatorProtocol[Array]:
        """Return ``F``, without essential constraints applied."""
        return self._spatial_operator

    def constraint_set(self) -> ConstraintSetProtocol[Array]:
        """Return the essential constraints."""
        return self._constraint_set

    def owns(self, target: object) -> bool:
        """Whether ``target`` is part of this system's ``F``."""
        return owns(self._spatial_operator, target)

    def steady(self) -> SteadyView[Array]:
        """The time-free steady problem ``F(u) = 0``.

        Raises unless ``F`` and the constraints are declared
        time-invariant; for time-dependent data use ``steady_snapshot``.
        """
        return SteadyView.of(self._spatial_operator, self._constraint_set)

    def steady_snapshot(self, time: float) -> SteadyView[Array]:
        """The steady problem of the data frozen at ``time``."""
        return SteadyView.snapshot(
            self._spatial_operator, self._constraint_set, time
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(nstates={self.nstates()}, "
            f"spatial_operator={type(self._spatial_operator).__name__}, "
            f"nconstrained={self._constraint_set.ndofs()})"
        )


class GalerkinSystem(GalerkinSteadySystem[Array]):
    """The parts of a transient problem ``M du/dt = F(u, t)``.

    Satisfies ``GalerkinTransientSystemProtocol`` (and so the steady
    one too: a transient system can be solved for its steady state).

    Parameters
    ----------
    spatial_operator : SpatialOperatorProtocol
        ``F``, the interior composed with the natural-BC terms.
    constraint_set : ConstraintSetProtocol
        The essential constraints on rows of ``F``.
    mass : GalerkinMassProtocol
        Supplies ``M``; read on every call, so it is never stale.
    """

    def __init__(
        self,
        spatial_operator: SpatialOperatorProtocol[Array],
        constraint_set: ConstraintSetProtocol[Array],
        mass: GalerkinMassProtocol[Array],
    ) -> None:
        super().__init__(spatial_operator, constraint_set)
        if not isinstance(mass, GalerkinMassProtocol):
            raise TypeError(
                "mass must satisfy GalerkinMassProtocol, got "
                f"{type(mass).__name__}"
            )
        self._mass = mass

    def mass_matrix(self) -> Array:
        """Return the mass matrix. Shape: (nstates, nstates)."""
        mass = self._mass.mass_matrix()
        nstates = self.nstates()
        if tuple(mass.shape) != (nstates, nstates):
            raise ValueError(
                f"mass matrix has shape {tuple(mass.shape)}, but the "
                f"spatial operator has {nstates} states"
            )
        return mass
