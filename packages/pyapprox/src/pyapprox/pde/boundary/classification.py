"""Boundary DOF classification shared by the time-integration layer."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BCDofClassification:
    """Classification of boundary DOFs for adjoint operations.

    Produced by the physics/solver layer, consumed by the time
    integration layer. The time integration layer uses these index
    sets without knowing solver internals.

    Attributes
    ----------
    essential : list[int]
        DOFs where the solution is prescribed (e.g., Dirichlet).
        The adjoint variable is zero at these DOFs (lambda[b] = 0)
        when differentiating w.r.t. PDE parameters. When differentiating
        w.r.t. BC parameters, lambda[b] = -dq/dy[b] acts as the
        Lagrange multiplier for the constraint.
        Always a subset of row_replaced.
    row_replaced : list[int]
        DOFs where the solver replaced the PDE residual row with a BC
        equation. At these DOFs: B_n[b,:] = 0 (no dependence on
        previous time step) and dR_n/dp[b,:] = 0 (assuming BCs are
        independent of PDE parameters).

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
