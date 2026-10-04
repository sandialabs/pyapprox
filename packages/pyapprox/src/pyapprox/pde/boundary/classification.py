"""Boundary DOF classification shared by the time-integration layer."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BCDofClassification:
    r"""Classification of boundary DOFs for adjoint operations.

    Produced by the physics/solver layer, consumed by the time
    integration layer. The time integration layer uses these index
    sets without knowing solver internals.

    Attributes
    ----------
    essential : list[int]
        DOFs where the solution is prescribed (e.g., Dirichlet), the set
        :math:`E`. The adjoint is NOT zero there: for :math:`n \geq 1`
        it is the reaction

        .. math::

            \lambda_{n,E} = -\partial_{y_{n,E}} Q
                - \sum_{i \notin E} (A_n)_{iE}\, \lambda_{n,i}
                - \sum_{i \notin R} (B_{n+1})_{iE}\, \lambda_{n+1,i},

        with :math:`\lambda_{N+1} = 0`, so the cross-step sum vanishes
        only at :math:`n = N`. The :math:`-\partial_{y_{n,E}} Q` term is
        dropped under the default ``zero_adjoint_rhs(zero_essential=True)``.
        For coefficient parameters :math:`\lambda_{n,E}` does not enter
        the gradient (rows :math:`E` of :math:`\partial r / \partial p`
        and columns :math:`R` of :math:`B_{n+1}^T` are zero); for
        BC-data parameters it is the multiplier of the constraint.
        Always a subset of row_replaced.
    row_replaced : list[int]
        DOFs where the solver replaced the PDE residual row with a BC
        equation, the set :math:`R`. At these rows :math:`B_n` has no
        entries (no dependence on the previous step) and
        :math:`\partial r_n / \partial p` is zero (BC data independent of
        the coefficient parameters).

        For collocation: all BC DOFs (both Dirichlet and Robin).
        For Galerkin FEM: only strongly-enforced Dirichlet DOFs.
        Natural BCs in Galerkin are assembled into the weak form
        without row replacement.
    """

    essential: list[Any]
    row_replaced: list[Any]

    def __post_init__(self) -> None:
        if not set(self.essential) <= set(self.row_replaced):
            raise ValueError(
                "essential must be a subset of row_replaced: "
                "every prescribed-value BC must replace its residual row"
            )
