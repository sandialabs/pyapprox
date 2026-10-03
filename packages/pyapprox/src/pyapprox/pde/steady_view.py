"""A time-free steady view of a time-dependent operator.

Physics keep one definition, ``F(u, t)``, for transient and steady use.
Steady consumers (Newton for ``F(u) = 0``, steady adjoints) never see
time: they take a ``SteadyOperatorProtocol``, and a ``SteadyView`` binds
the time once, immutably, while applying the essential constraints.

The time is never a silent default:

- ``SteadyView.of(op, cs)`` accepts only an operator and constraint set
  DECLARED time-invariant; their value is then the same at every time,
  so the bound time cannot matter.
- ``SteadyView.snapshot(op, cs, time)`` is the explicit steady state of
  time-dependent data frozen at ``time``.

The view copies no data: every call delegates to the operator, so a
parameter change (``set_param``) is seen immediately.
"""

from typing import Generic, Protocol, runtime_checkable

from pyapprox.ode.protocols import SpatialOperatorProtocol
from pyapprox.pde.boundary import ConstraintSetProtocol
from pyapprox.pde.ownership import owns
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class SteadyOperatorProtocol(Protocol, Generic[Array]):
    """What a steady solver needs: the constrained ``F(u)``, no time."""

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        ...

    def nstates(self) -> int:
        """Return the number of states."""
        ...

    def steady_residual(self, state: Array) -> Array:
        """``F(u)`` with constrained rows replaced. Shape: (nstates,)."""
        ...

    def steady_jacobian(self, state: Array) -> Array:
        """``dF/du`` with identity rows at constrained DOFs."""
        ...


@runtime_checkable
class SteadyViewProtocol(SteadyOperatorProtocol[Array], Protocol):
    """A steady operator that also exposes its parts.

    For the one place that binds the view's time into derivatives
    (``steady_constrained_derivatives``): the operator, its constraint
    set, and the bound time. Steady consumers themselves never read the
    time.
    """

    def spatial_operator(self) -> SpatialOperatorProtocol[Array]:
        """Return the viewed operator ``F(u, t)``."""
        ...

    def constraint_set(self) -> ConstraintSetProtocol[Array]:
        """Return the essential constraints."""
        ...

    def time(self) -> float:
        """Return the time at which the operator is evaluated."""
        ...

    def owns(self, target: object) -> bool:
        """Whether ``target`` is part of the viewed operator."""
        ...


class SteadyView(Generic[Array]):
    """``F(u, t)`` with constraints, at one bound time. Build with
    ``SteadyView.of`` or ``SteadyView.snapshot``.

    Satisfies ``SteadyViewProtocol``.
    """

    def __init__(
        self,
        spatial_operator: SpatialOperatorProtocol[Array],
        constraint_set: ConstraintSetProtocol[Array],
        time: float,
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
        self._spatial_operator = spatial_operator
        self._constraint_set = constraint_set
        self._time = float(time)

    @classmethod
    def of(
        cls,
        spatial_operator: SpatialOperatorProtocol[Array],
        constraint_set: ConstraintSetProtocol[Array],
    ) -> "SteadyView[Array]":
        """The steady problem of time-invariant data.

        Raises
        ------
        ValueError
            If the operator or the constraints are declared
            time-dependent: their steady state depends on which time is
            meant, so use ``snapshot`` and say.
        """
        if not spatial_operator.is_time_invariant():
            raise ValueError(
                f"{type(spatial_operator).__name__} has data declared "
                "time-dependent, so its steady state depends on the time; "
                "use SteadyView.snapshot(op, constraints, time)"
            )
        if not constraint_set.is_time_invariant():
            raise ValueError(
                "the essential constraints are time-dependent, so the "
                "steady state depends on the time; use "
                "SteadyView.snapshot(op, constraints, time)"
            )
        # Any time gives the same values; 0.0 is never consulted for a
        # result.
        return cls(spatial_operator, constraint_set, 0.0)

    @classmethod
    def snapshot(
        cls,
        spatial_operator: SpatialOperatorProtocol[Array],
        constraint_set: ConstraintSetProtocol[Array],
        time: float,
    ) -> "SteadyView[Array]":
        """The steady problem of the data frozen at ``time``."""
        return cls(spatial_operator, constraint_set, time)

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._spatial_operator.bkd()

    def nstates(self) -> int:
        """Return the number of states."""
        return self._spatial_operator.nstates()

    def time(self) -> float:
        """Return the bound time."""
        return self._time

    def spatial_operator(self) -> SpatialOperatorProtocol[Array]:
        """Return the viewed operator ``F(u, t)``."""
        return self._spatial_operator

    def constraint_set(self) -> ConstraintSetProtocol[Array]:
        """Return the essential constraints."""
        return self._constraint_set

    def owns(self, target: object) -> bool:
        """Whether ``target`` is part of the viewed operator."""
        return owns(self._spatial_operator, target)

    def steady_residual(self, state: Array) -> Array:
        """``F(u, t)`` with rows ``d`` replaced by ``u_d - g(t)``."""
        return self._constraint_set.apply_to_residual(
            self._spatial_operator.spatial_residual(state, self._time),
            state,
            self._time,
        )

    def steady_jacobian(self, state: Array) -> Array:
        """``dF/du`` with identity rows at constrained DOFs."""
        return self._constraint_set.apply_to_jacobian(
            self._spatial_operator.spatial_jacobian(state, self._time)
        )

    def __repr__(self) -> str:
        return (
            f"SteadyView({type(self._spatial_operator).__name__}, "
            f"time={self._time})"
        )
