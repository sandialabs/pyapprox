r"""A Galerkin system's essential rows and data, for pinning.

The essential (Dirichlet) rows of a discretized system and their data
:math:`g(t)`, declared once as :mod:`pyapprox.ode.pinned_rows` reads
them. The strategy is then which consumer gets the declaration: a
stepper pinning them inside its solves
(:class:`~pyapprox.ode.pinned_rows.RowPinning`; at the step end, with the
coupled row treatment, total content is conserved exactly for every
stepper) or the integrator pinning them after each step
(:class:`~pyapprox.ode.pinned_rows.PinnedRowsConstraint`).
"""

from typing import Callable, Generic, Optional, TypeVar

from pyapprox.ode.pinned_rows import PinnedRowsParamDerivatives
from pyapprox.pde.galerkin.time_integration.stack_discretization import (
    GalerkinStackDiscretization,
)
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import FormatProtocol, RowSetProtocol

F = TypeVar("F", bound=FormatProtocol)


class _EssentialRates(Generic[Array]):
    r""":math:`\dot g(t)` placed on the essential rows of a zero column."""

    def __init__(
        self,
        velocity: Callable[[float], Array],
        dofs: Array,
        nstates: int,
        bkd: Backend[Array],
    ) -> None:
        self._velocity, self._dofs = velocity, dofs
        self._nstates, self._bkd = nstates, bkd

    def __call__(self, time: float) -> Array:
        placed = self._bkd.index_update(
            self._bkd.zeros((self._nstates,)), self._dofs, self._velocity(time)
        )
        return self._bkd.reshape(placed, (self._nstates, 1))


class EssentialPinnedRows(Generic[Array]):
    """The essential rows of a discretized Galerkin system and their data,
    as :class:`~pyapprox.ode.pinned_rows.PinnedRowsProtocol` declares
    them.

    Parameters
    ----------
    discretization : GalerkinStackDiscretization
        Gives the essential rows (a row set of its state space) and,
        through its system's constraint set, the data. The data must not
        depend on the parameters.
    """

    def __init__(
        self, discretization: GalerkinStackDiscretization[Array, Array, F]
    ) -> None:
        system = discretization.system()
        self._rows = discretization.essential_rows()
        self._constraint_set = system.constraint_set()
        self._bkd = system.bkd()
        self._nstates = system.nstates()
        # Fixed for the constraint set's life: exact zeros when it is time
        # invariant, None when a time-varying condition lacks its rate.
        velocity = self._constraint_set.values_derivative(1)
        self._rates: Optional[_EssentialRates[Array]] = (
            None
            if velocity is None
            else _EssentialRates(
                velocity, self._constraint_set.dofs(), self._nstates, self._bkd
            )
        )

    def rows(self) -> RowSetProtocol[Array]:
        """The essential rows."""
        return self._rows

    def values(self, time: float) -> Array:
        """A column holding :math:`g(t)` on the essential rows and zero
        elsewhere. Shape: (nstates, 1)."""
        injected = self._constraint_set.inject(self._bkd.zeros((self._nstates,)), time)
        return self._bkd.reshape(injected, (self._nstates, 1))

    def rates(self) -> Optional[Callable[[float], Array]]:
        r""":math:`t \mapsto \dot g(t)` as a column laid out as
        :meth:`values`, from the boundary conditions' declared rates; None
        when a time-varying condition supplies none."""
        return self._rates

    def parameter_derivatives(self) -> PinnedRowsParamDerivatives[Array]:
        """Independent of the parameters: a parameterization writes
        coefficients of the physics, never essential boundary values."""
        return PinnedRowsParamDerivatives.independent()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nrows={self._rows.count()})"
