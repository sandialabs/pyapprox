"""Machinery shared by the two directions of GroupACV allocation.

Minimizing an estimator criterion under a budget and minimizing cost
under an accuracy requirement differ in which object fills the objective
role, which fills the constraint role, where the bound on total cost
comes from, and which way sample counts round. Everything between those
choices -- deriving per-partition bounds, moving the problem into the
optimizer's variable space, solving, and moving the answer back -- is
identical, and lives here so a change to it cannot land in one direction
and be missed in the other.
"""

from typing import TYPE_CHECKING, Generic, List, Optional

from pyapprox.interface.functions.autograd import WithAutogradJacobian
from pyapprox.statest.groupacv.variable_space import (
    AllocationProblemConfig,
    VariableSpace,
    _AllocationConstraint,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.interface.functions.protocols.objective import (
        ObjectiveProtocol,
    )
    from pyapprox.optimization.minimize.protocols import (
        BindableOptimizerProtocol,
    )
    from pyapprox.statest.groupacv.base import BaseGroupACVEstimator


class VariableSpaceSolution(Generic[Array]):
    """Outcome of a solve, in n-space.

    ``npartition_samples`` is populated only when the solve succeeded;
    otherwise ``message`` says why. The two directions report failure
    differently -- they build different result types from different
    fallback allocations -- so this reports the outcome and leaves that
    to the caller.
    """

    def __init__(
        self,
        npartition_samples: Optional[Array],
        message: str,
    ) -> None:
        self._npartition_samples = npartition_samples
        self._message = message

    def succeeded(self) -> bool:
        """Whether the solve produced a usable allocation."""
        return self._npartition_samples is not None

    def npartition_samples(self) -> Array:
        """The n-space allocation. Only valid when :meth:`succeeded`."""
        if self._npartition_samples is None:
            raise RuntimeError(
                "no allocation available; check succeeded() first"
            )
        return self._npartition_samples

    def message(self) -> str:
        """Failure reason, empty when the solve succeeded."""
        return self._message


def partition_costs(
    est: "BaseGroupACVEstimator[Array]",
) -> Array:
    """Cost of one sample in each partition. Shape (npartitions,)."""
    return est._bkd.einsum(
        "m,mp->p", est._costs, est._partitions_per_model
    )


def raw_bounds(
    est: "BaseGroupACVEstimator[Array]",
    config: AllocationProblemConfig,
    cost_ceiling: float,
) -> Array:
    """Per-partition n-space bounds implied by a cost ceiling.

    A partition cannot hold more samples than the ceiling buys of it
    alone. The ceiling is a budget in one direction and the cost of a
    known-feasible allocation in the other; the bound is the same either
    way.

    Returns
    -------
    Array (npartitions, 2)
        Lower and upper bound for each partition.
    """
    bkd = est._bkd
    bounds_lb = config.resolve_bounds_lb(est._stat)
    costs = partition_costs(est)
    bounds_list: List[List[float]] = []
    for m in range(est.npartitions()):
        max_n_m = cost_ceiling / bkd.to_float(costs[m])
        bounds_list.append([bounds_lb, max_n_m])
    return bkd.array(bounds_list)


def solve_in_variable_space(
    objective: "ObjectiveProtocol[Array]",
    constraint: _AllocationConstraint[Array],
    n_bounds: Array,
    init_guess: Array,
    optimizer: "BindableOptimizerProtocol[Array]",
    config: AllocationProblemConfig,
    est: "BaseGroupACVEstimator[Array]",
) -> VariableSpaceSolution[Array]:
    """Solve an allocation problem in the configured variable space.

    Moves objective, constraint, bounds and initial guess out of sample
    counts into whatever coordinates the configuration asks for, solves,
    and moves the answer back.

    Parameters
    ----------
    objective : ObjectiveProtocol
        The quantity to minimize, in n-space.
    constraint : _AllocationConstraint
        The constraint to respect, in n-space.
    n_bounds : Array (npartitions, 2)
        Bounds in n-space, from :func:`raw_bounds`.
    init_guess : Array (npartitions, 1)
        Starting allocation, in n-space.
    optimizer : BindableOptimizerProtocol
        The optimizer to bind and run.
    config : AllocationProblemConfig
        Supplies the variable space.
    est : BaseGroupACVEstimator
        Supplies the per-partition costs used to scale the space.

    Returns
    -------
    VariableSpaceSolution
        The n-space allocation, or the reason none was produced.
    """
    bkd: Backend[Array] = est._bkd
    space: VariableSpace[Array] = config.build_variable_space(bkd)
    scale = space.compute_scale(partition_costs(est), bkd)
    opt_bounds = space.transform_bounds(n_bounds, scale, bkd)
    wrapped_obj: "ObjectiveProtocol[Array]" = space.wrap_objective(
        objective, scale
    )
    wrapped_con = space.wrap_constraint(constraint, scale)

    # Autograd is a composition source: when no analytical jacobian is
    # available (stat lacks sigma-block derivatives or estimator is not
    # IS) and the backend can autodiff, differentiate the
    # optimizer-space objective the optimizer actually sees.
    if wrapped_obj.derivatives().jacobian is None and isinstance(
        bkd, AutodiffBackend
    ):
        wrapped_obj = WithAutogradJacobian(wrapped_obj, bkd)

    optimizer.bind(wrapped_obj, opt_bounds, [wrapped_con])
    result = optimizer.minimize(space.transform_init_guess(init_guess, scale))
    if not result.success():
        return VariableSpaceSolution(None, "Optimization failed")

    npartition_samples = space.transform_from_optimizer(
        result.optima()[:, 0], scale
    )
    if bkd.any_bool(npartition_samples < 0):
        return VariableSpaceSolution(
            None, "Negative sample counts in n-space"
        )
    return VariableSpaceSolution(npartition_samples, "")
