"""Variable space and budget constraint strategies for GroupACV allocation.

This module provides strategy classes that decouple variable rescaling and
budget constraint form from the optimization loop:

- VariableSpace protocol: transforms between n-space and optimizer-space
- BudgetConstraintForm protocol: configures constraint equality/inequality
- AllocationProblemConfig: composes both via registry-based factory methods
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generic,
    Literal,
    Protocol,
    runtime_checkable,
)

from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.statest.groupacv.optimization import (
        GroupACVCostConstraint,
        GroupACVObjective,
    )
    from pyapprox.statest.statistics import MultiOutputStatistic


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class _ObjectiveLike(Protocol[Array]):
    """Structural type for objective-like objects (original or wrapped)."""

    def bkd(self) -> Backend[Array]: ...
    def nvars(self) -> int: ...
    def nqoi(self) -> int: ...
    def __call__(self, npartition_samples: Array) -> Array: ...


@runtime_checkable
class _ConstraintLike(Protocol[Array]):
    """Structural type for constraint-like objects (original or wrapped)."""

    def bkd(self) -> Backend[Array]: ...
    def nvars(self) -> int: ...
    def nqoi(self) -> int: ...
    def lb(self) -> Array: ...
    def ub(self) -> Array: ...
    def __call__(self, npartition_samples: Array) -> Array: ...
    def jacobian(self, npartition_samples: Array) -> Array: ...
    def whvp(
        self, npartition_samples: Array, vec: Array, weights: Array
    ) -> Array: ...


@runtime_checkable
class VariableSpace(Protocol[Array]):
    """Protocol for variable-space transformations in GroupACV allocation."""

    def compute_scale(
        self, partition_costs: Array, bkd: Backend[Array]
    ) -> Array:
        """Return per-partition scale factors, shape (npartitions,)."""
        ...

    def transform_bounds(
        self, n_bounds: Array, scale: Array, bkd: Backend[Array]
    ) -> Array:
        """Transform n-space bounds to optimizer-space bounds."""
        ...

    def transform_init_guess(self, n_guess: Array, scale: Array) -> Array:
        """Transform n-space init guess to optimizer-space."""
        ...

    def transform_from_optimizer(self, m_opt: Array, scale: Array) -> Array:
        """Transform optimizer result back to n-space."""
        ...

    def wrap_objective(
        self, objective: "GroupACVObjective[Array]", scale: Array
    ) -> "_ObjectiveLike[Array]":
        """Wrap objective to accept optimizer-space variables."""
        ...

    def wrap_constraint(
        self, constraint: "GroupACVCostConstraint[Array]", scale: Array
    ) -> "_ConstraintLike[Array]":
        """Wrap constraint to accept optimizer-space variables."""
        ...


@runtime_checkable
class BudgetConstraintForm(Protocol[Array]):
    """Protocol for budget constraint form strategies."""

    def adjust_bounds(
        self, constraint: "GroupACVCostConstraint[Array]"
    ) -> None:
        """Modify constraint lb/ub for the desired form."""
        ...


# ---------------------------------------------------------------------------
# Wrapper classes
# ---------------------------------------------------------------------------


class _RescaledObjective(Generic[Array]):
    """Wraps objective to accept m-space variables, converting via n = m/scale."""

    def __init__(
        self, inner: "GroupACVObjective[Array]", scale: Array
    ) -> None:
        self._inner = inner
        self._scale = scale
        if hasattr(inner, "jacobian"):
            self.jacobian = self._jacobian_impl
        if hasattr(inner, "hessian"):
            self._inner_hessian: Callable[[Array], Array] = inner.hessian
            self.hessian = self._hessian_impl
        if hasattr(inner, "hvp"):
            self.hvp = self._hvp_impl

    def bkd(self) -> Backend[Array]:
        return self._inner.bkd()

    def nvars(self) -> int:
        return self._inner.nvars()

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def __call__(self, npartition_samples: Array) -> Array:
        n = npartition_samples / self._scale[:, None]
        return self._inner(n)

    def _jacobian_impl(self, npartition_samples: Array) -> Array:
        n = npartition_samples / self._scale[:, None]
        J_n = self._inner.jacobian(n)
        return J_n / self._scale[None, :]

    def _hessian_impl(self, npartition_samples: Array) -> Array:
        n = npartition_samples / self._scale[:, None]
        H_n = self._inner_hessian(n)
        outer = self._scale[:, None] * self._scale[None, :]
        return H_n / outer

    def _hvp_impl(self, npartition_samples: Array, vec: Array) -> Array:
        return self._hessian_impl(npartition_samples) @ vec


class _RescaledConstraint(Generic[Array]):
    """Wraps constraint to accept m-space variables, converting via n = m/scale."""

    def __init__(
        self, inner: "GroupACVCostConstraint[Array]", scale: Array
    ) -> None:
        self._inner = inner
        self._scale = scale
        if hasattr(inner, "jacobian"):
            self.jacobian = self._jacobian_impl
        if hasattr(inner, "whvp"):
            self.whvp = self._whvp_impl

    def bkd(self) -> Backend[Array]:
        return self._inner.bkd()

    def nvars(self) -> int:
        return self._inner.nvars()

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def lb(self) -> Array:
        return self._inner.lb()

    def ub(self) -> Array:
        return self._inner.ub()

    def __call__(self, npartition_samples: Array) -> Array:
        n = npartition_samples / self._scale[:, None]
        return self._inner(n)

    def _jacobian_impl(self, npartition_samples: Array) -> Array:
        n = npartition_samples / self._scale[:, None]
        J_n = self._inner.jacobian(n)
        return J_n / self._scale[None, :]

    def _whvp_impl(
        self, npartition_samples: Array, vec: Array, weights: Array
    ) -> Array:
        bkd = self._inner.bkd()
        return bkd.zeros((self.nvars(), 1))


class _LogObjective(Generic[Array]):
    """Wraps objective to accept log-space variables: n = exp(m).

    Chain rule: df/dm_k = (df/dn_k) * n_k
    Hessian: d²f/dm_k dm_p = (d²f/dn_k dn_p)*n_k*n_p + δ_{kp}*(df/dn_k)*n_k
    """

    def __init__(
        self, inner: "GroupACVObjective[Array]", bkd: Backend[Array]
    ) -> None:
        self._inner = inner
        self._bkd = bkd
        if hasattr(inner, "jacobian"):
            self.jacobian = self._jacobian_impl
        if hasattr(inner, "hessian"):
            self._inner_hessian: Callable[[Array], Array] = inner.hessian
            self.hessian = self._hessian_impl
        if hasattr(inner, "hvp"):
            self.hvp = self._hvp_impl

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._inner.nvars()

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def __call__(self, m: Array) -> Array:
        n = self._bkd.exp(m)
        return self._inner(n)

    def _jacobian_impl(self, m: Array) -> Array:
        n = self._bkd.exp(m)
        J_n = self._inner.jacobian(n)
        return J_n * n[:, 0][None, :]

    def _hessian_impl(self, m: Array) -> Array:
        n = self._bkd.exp(m)
        n_1d = n[:, 0]
        H_n = self._inner_hessian(n)
        H_m = H_n * (n_1d[:, None] * n_1d[None, :])
        J_n = self._inner.jacobian(n)
        diag_correction = J_n[0, :] * n_1d
        nv = self.nvars()
        for k in range(nv):
            H_m[k, k] = H_m[k, k] + diag_correction[k]
        return H_m

    def _hvp_impl(self, m: Array, vec: Array) -> Array:
        return self._hessian_impl(m) @ vec


class _LogConstraint(Generic[Array]):
    """Wraps constraint to accept log-space variables: n = exp(m).

    Chain rule: dg/dm_k = (dg/dn_k) * n_k
    The constraint is no longer linear in m, so whvp is nonzero.
    """

    def __init__(
        self, inner: "GroupACVCostConstraint[Array]", bkd: Backend[Array]
    ) -> None:
        self._inner = inner
        self._bkd = bkd
        if hasattr(inner, "jacobian"):
            self.jacobian = self._jacobian_impl
        if hasattr(inner, "whvp"):
            self.whvp = self._whvp_impl

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._inner.nvars()

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def lb(self) -> Array:
        return self._inner.lb()

    def ub(self) -> Array:
        return self._inner.ub()

    def __call__(self, m: Array) -> Array:
        n = self._bkd.exp(m)
        return self._inner(n)

    def _jacobian_impl(self, m: Array) -> Array:
        n = self._bkd.exp(m)
        J_n = self._inner.jacobian(n)
        return J_n * n[:, 0][None, :]

    def _whvp_impl(
        self, m: Array, vec: Array, weights: Array
    ) -> Array:
        # g_i(m) = g_i(exp(m)), constraint g_i is linear in n:
        #   g_i(n) = a_i^T n + b_i
        # So d²g_i/dm_k dm_p = δ_{kp} * (dg_i/dn_k) * n_k
        # whvp = Σ_i w_i * diag(dg_i/dn * n) @ vec
        n = self._bkd.exp(m)
        n_1d = n[:, 0]
        J_n = self._inner.jacobian(n)
        # J_n shape: (nqoi, nvars), weights shape: (nqoi, 1)
        weighted_diag = self._bkd.einsum(
            "i,ij->j", weights[:, 0], J_n
        ) * n_1d
        return (weighted_diag[:, None] * vec)


class _NormalizedConstraint(Generic[Array]):
    """Divides constraint output by normalization factors so values are ~O(1)."""

    def __init__(
        self, inner: _ConstraintLike[Array], normalization: Array
    ) -> None:
        bkd = inner.bkd()
        if bkd.any_bool(normalization <= 0):
            raise ValueError("Normalization factors must be positive")
        self._inner: _ConstraintLike[Array] = inner
        self._norm = normalization
        if hasattr(inner, "jacobian"):
            self.jacobian = self._jacobian_impl
        if hasattr(inner, "whvp"):
            self.whvp = self._whvp_impl

    def bkd(self) -> Backend[Array]:
        return self._inner.bkd()

    def nvars(self) -> int:
        return self._inner.nvars()

    def nqoi(self) -> int:
        return self._inner.nqoi()

    def lb(self) -> Array:
        return self._inner.lb() / self._norm

    def ub(self) -> Array:
        return self._inner.ub() / self._norm

    def __call__(self, npartition_samples: Array) -> Array:
        return self._inner(npartition_samples) / self._norm[:, None]

    def _jacobian_impl(self, npartition_samples: Array) -> Array:
        return self._inner.jacobian(npartition_samples) / self._norm[:, None]

    def _whvp_impl(
        self, npartition_samples: Array, vec: Array, weights: Array
    ) -> Array:
        bkd = self._inner.bkd()
        return bkd.zeros((self.nvars(), 1))


# ---------------------------------------------------------------------------
# VariableSpace implementations
# ---------------------------------------------------------------------------


class IdentitySpace(Generic[Array]):
    """No variable rescaling — optimizer works in raw n-space."""

    def compute_scale(
        self, partition_costs: Array, bkd: Backend[Array]
    ) -> Array:
        return bkd.ones((partition_costs.shape[0],))

    def transform_bounds(
        self, n_bounds: Array, scale: Array, bkd: Backend[Array]
    ) -> Array:
        return n_bounds

    def transform_init_guess(self, n_guess: Array, scale: Array) -> Array:
        return n_guess

    def transform_from_optimizer(self, m_opt: Array, scale: Array) -> Array:
        return m_opt

    def wrap_objective(
        self, objective: "GroupACVObjective[Array]", scale: Array
    ) -> _ObjectiveLike[Array]:
        return objective

    def wrap_constraint(
        self, constraint: "GroupACVCostConstraint[Array]", scale: Array
    ) -> _ConstraintLike[Array]:
        return constraint


class ConstraintScaledSpace(Generic[Array]):
    """No variable rescaling; constraint output normalized to ~O(1)."""

    def compute_scale(
        self, partition_costs: Array, bkd: Backend[Array]
    ) -> Array:
        return bkd.ones((partition_costs.shape[0],))

    def transform_bounds(
        self, n_bounds: Array, scale: Array, bkd: Backend[Array]
    ) -> Array:
        return n_bounds

    def transform_init_guess(self, n_guess: Array, scale: Array) -> Array:
        return n_guess

    def transform_from_optimizer(self, m_opt: Array, scale: Array) -> Array:
        return m_opt

    def wrap_objective(
        self, objective: "GroupACVObjective[Array]", scale: Array
    ) -> _ObjectiveLike[Array]:
        return objective

    def wrap_constraint(
        self, constraint: "GroupACVCostConstraint[Array]", scale: Array
    ) -> _ConstraintLike[Array]:
        bkd = constraint.bkd()
        target_cost = constraint.target_cost()
        norm_val = target_cost if target_cost and target_cost != 0 else 1.0
        norm = bkd.array([norm_val, 1.0])
        return _NormalizedConstraint(constraint, norm)


class FullCostSpace(Generic[Array]):
    """Full cost-based rescaling: m_k = (c_k / c_ref) * n_k."""

    def compute_scale(
        self, partition_costs: Array, bkd: Backend[Array]
    ) -> Array:
        return partition_costs / bkd.min(partition_costs)

    def transform_bounds(
        self, n_bounds: Array, scale: Array, bkd: Backend[Array]
    ) -> Array:
        return n_bounds * scale[:, None]

    def transform_init_guess(self, n_guess: Array, scale: Array) -> Array:
        return n_guess * scale[:, None]

    def transform_from_optimizer(self, m_opt: Array, scale: Array) -> Array:
        return m_opt / scale

    def wrap_objective(
        self, objective: "GroupACVObjective[Array]", scale: Array
    ) -> _ObjectiveLike[Array]:
        return _RescaledObjective(objective, scale)

    def wrap_constraint(
        self, constraint: "GroupACVCostConstraint[Array]", scale: Array
    ) -> _ConstraintLike[Array]:
        bkd = constraint.bkd()
        target_cost = constraint.target_cost()
        norm_val = target_cost if target_cost and target_cost != 0 else 1.0
        norm = bkd.array([norm_val, 1.0])
        rescaled = _RescaledConstraint(constraint, scale)
        return _NormalizedConstraint(rescaled, norm)


class LogSpace(Generic[Array]):
    """Log-space rescaling: m_k = log(n_k), so n_k = exp(m_k).

    Equalizes the dynamic range of partition sample counts when costs
    span many orders of magnitude. The cost constraint becomes nonlinear
    in m-space but the objective landscape is much better conditioned.
    """

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def compute_scale(
        self, partition_costs: Array, bkd: Backend[Array]
    ) -> Array:
        return bkd.ones((partition_costs.shape[0],))

    def transform_bounds(
        self, n_bounds: Array, scale: Array, bkd: Backend[Array]
    ) -> Array:
        lb = bkd.maximum(
            n_bounds[:, 0], bkd.full(n_bounds[:, 0].shape, 1e-30)
        )
        ub = n_bounds[:, 1]
        return bkd.stack([bkd.log(lb), bkd.log(ub)], axis=1)

    def transform_init_guess(self, n_guess: Array, scale: Array) -> Array:
        safe = self._bkd.maximum(
            n_guess, self._bkd.full(n_guess.shape, 1e-30)
        )
        return self._bkd.log(safe)

    def transform_from_optimizer(self, m_opt: Array, scale: Array) -> Array:
        return self._bkd.exp(m_opt)

    def wrap_objective(
        self, objective: "GroupACVObjective[Array]", scale: Array
    ) -> _ObjectiveLike[Array]:
        return _LogObjective(objective, self._bkd)

    def wrap_constraint(
        self, constraint: "GroupACVCostConstraint[Array]", scale: Array
    ) -> _ConstraintLike[Array]:
        target_cost = constraint.target_cost()
        norm_val = target_cost if target_cost and target_cost != 0 else 1.0
        norm = self._bkd.array([norm_val, 1.0])
        log_con = _LogConstraint(constraint, self._bkd)
        return _NormalizedConstraint(log_con, norm)


# ---------------------------------------------------------------------------
# BudgetConstraintForm implementations
# ---------------------------------------------------------------------------


class InequalityBudget(Generic[Array]):
    """Budget as inequality constraint: cost <= target_cost."""

    def adjust_bounds(
        self, constraint: "GroupACVCostConstraint[Array]"
    ) -> None:
        constraint._set_cost_inequality()


class EqualityBudget(Generic[Array]):
    """Budget as equality constraint: cost == target_cost."""

    def adjust_bounds(
        self, constraint: "GroupACVCostConstraint[Array]"
    ) -> None:
        constraint._set_cost_equality()


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_SPACE_REGISTRY: Dict[
    str, Callable[[Backend[Array]], VariableSpace[Array]]
] = {
    "none": lambda bkd: IdentitySpace(),
    "constraint_only": lambda bkd: ConstraintScaledSpace(),
    "full": lambda bkd: FullCostSpace(),
    "log": LogSpace,
}

_FORM_REGISTRY: Dict[str, Callable[[], BudgetConstraintForm[Array]]] = {
    "inequality": InequalityBudget,
    "equality": EqualityBudget,
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationProblemConfig:
    """Configuration for GroupACV allocation optimization.

    Parameters
    ----------
    variable_scaling : {"none", "constraint_only", "full", "log"}
        Variable rescaling strategy:
        - "none": optimizer works in raw n-space
        - "constraint_only": n-space with normalized constraint values
        - "full": optimizer works in m-space where m_k = (c_k/c_ref) * n_k
        - "log": optimizer works in m-space where m_k = log(n_k)

    budget_constraint_form : {"inequality", "equality"}
        Budget constraint form:
        - "inequality": cost <= target_cost
        - "equality": cost == target_cost

    bounds_lb : float or "dead_threshold"
        Lower bound for partition sample counts during optimization.
        - "dead_threshold": use max(1e-8, stat.continuous_dead_threshold())
        - explicit float: use that value directly
    """

    variable_scaling: Literal["none", "constraint_only", "full", "log"] = "none"
    budget_constraint_form: Literal["inequality", "equality"] = "inequality"
    bounds_lb: Literal["dead_threshold"] | float = "dead_threshold"

    def resolve_bounds_lb(
        self, stat: "MultiOutputStatistic[Array]"
    ) -> float:
        """Resolve the bounds lower bound to a concrete float value."""
        if self.bounds_lb == "dead_threshold":
            return max(1e-8, stat.continuous_dead_threshold())
        return self.bounds_lb

    def build_variable_space(self, bkd: Backend[Array]) -> VariableSpace[Array]:
        """Create the variable space strategy from config."""
        return _SPACE_REGISTRY[self.variable_scaling](bkd)

    def build_budget_form(self) -> BudgetConstraintForm[Array]:
        """Create the budget constraint form strategy from config."""
        return _FORM_REGISTRY[self.budget_constraint_form]()
