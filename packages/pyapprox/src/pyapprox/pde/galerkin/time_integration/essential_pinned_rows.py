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

from typing import Generic, TypeVar

from pyapprox.pde.galerkin.time_integration.stack_discretization import (
    GalerkinStackDiscretization,
)
from pyapprox.util.backends.protocols import Array
from pyapprox.util.linalg.operators import FormatProtocol, RowSetProtocol

F = TypeVar("F", bound=FormatProtocol)


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

    def rows(self) -> RowSetProtocol[Array]:
        """The essential rows."""
        return self._rows

    def values(self, time: float) -> Array:
        """A column holding :math:`g(t)` on the essential rows and zero
        elsewhere. Shape: (nstates, 1)."""
        injected = self._constraint_set.inject(self._bkd.zeros((self._nstates,)), time)
        return self._bkd.reshape(injected, (self._nstates, 1))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nrows={self._rows.count()})"
