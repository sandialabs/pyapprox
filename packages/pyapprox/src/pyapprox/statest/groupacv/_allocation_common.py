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

from typing import (
    TYPE_CHECKING,
    Generic,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from pyapprox.interface.functions.autograd import (
    WithAutogradJacobian,
    WithAutogradJacobianConstraint,
)
from pyapprox.optimization.minimize.constraints.linear import (
    PyApproxLinearConstraint,
)
from pyapprox.statest.groupacv.variable_space import (
    AllocationProblemConfig,
    VariableSpace,
    _AllocationConstraint,
    _ConstraintLike,
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


@runtime_checkable
class _AffineConstraint(Protocol):
    """A constraint that can say whether it is affine in its input."""

    def is_affine(self) -> bool:
        """Whether every row is affine in the sample counts."""
        ...


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


def _as_linear_if_affine(
    constraint: _ConstraintLike[Array],
    space: VariableSpace[Array],
    opt_guess: Array,
    bkd: Backend[Array],
) -> Union[_ConstraintLike[Array], PyApproxLinearConstraint[Array]]:
    """Express an affine constraint as a linear one where sound.

    The cost constraint is affine in the sample counts, but reaches
    scipy as a general nonlinear constraint. Scipy then re-evaluates it
    and its jacobian at every iterate and trial step, and approximates
    a Hessian that is identically zero -- work its linear-constraint
    path skips entirely.

    Only correct where the variable space's transform is itself linear.
    Under log-space variables the same constraint is genuinely
    nonlinear in the optimizer's coordinates, and describing it as
    linear would misstate the feasible set rather than merely cost
    time, so the space is asked first.

    Both the constraint and the space must agree: an accuracy
    requirement curves in the sample counts however the variables are
    scaled, and a constraint with no analytical jacobian has no
    coefficient matrix to hand over.
    """
    if not space.preserves_affinity():
        return constraint
    if not isinstance(constraint, _AffineConstraint):
        return constraint
    if not constraint.is_affine():
        return constraint
    jacobian = constraint.derivatives().jacobian
    if jacobian is None:
        return constraint
    # g(m) = A m + b, so b follows from one evaluation, and the bounds
    # absorb it: lb <= g(m) <= ub becomes lb - b <= A m <= ub - b.
    amat = jacobian(opt_guess)
    offset = constraint(opt_guess)[:, 0] - amat @ opt_guess[:, 0]
    return PyApproxLinearConstraint(
        amat, constraint.lb() - offset, constraint.ub() - offset, bkd
    )


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
    # optimizer-space function the optimizer actually sees. Either side
    # can lack one -- a criterion held to a tolerance carries the
    # criterion's capability into the constraint role -- so both are
    # composed.
    if wrapped_obj.derivatives().jacobian is None and isinstance(
        bkd, AutodiffBackend
    ):
        wrapped_obj = WithAutogradJacobian(wrapped_obj, bkd)
    if wrapped_con.derivatives().jacobian is None and isinstance(
        bkd, AutodiffBackend
    ):
        wrapped_con = WithAutogradJacobianConstraint(wrapped_con, bkd)

    opt_guess = space.transform_init_guess(init_guess, scale)
    solver_con = _as_linear_if_affine(wrapped_con, space, opt_guess, bkd)

    optimizer.bind(wrapped_obj, opt_bounds, [solver_con])
    result = optimizer.minimize(opt_guess)
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
