"""Unified search comparing ACV and GroupACV estimators."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Generic, Optional

from pyapprox.util.backends.protocols import Array

if TYPE_CHECKING:
    from pyapprox.statest.acv.search import ACVSearch, SearchResult
    from pyapprox.statest.groupacv.search import (
        GroupACVSearch,
        GroupACVSearchResult,
    )


class EstimatorFamily(Enum):
    """Enumeration of estimator families."""

    ACV = "acv"
    GROUPACV = "groupacv"


@dataclass
class UnifiedSearchResult(Generic[Array]):
    """Result of unified search across ACV and GroupACV."""

    # Best overall
    best_estimator: object
    best_objective: float
    best_family: EstimatorFamily

    # Individual search results (may be None if not searched)
    acv_result: Optional["SearchResult[Array]"]
    groupacv_result: Optional["GroupACVSearchResult[Array]"]

    # Objectives stored as floats for comparison_summary
    acv_objective: Optional[float] = None
    groupacv_objective: Optional[float] = None

    def comparison_summary(self) -> str:
        """Return summary comparing both families."""
        lines = [
            f"Best overall: {self.best_family.value} (obj={self.best_objective:.6f})"
        ]
        if self.acv_result is not None and self.acv_objective is not None:
            lines.append(
                f"  ACV best: {type(self.acv_result.estimator).__name__} "
                f"(obj={self.acv_objective:.6f})"
            )
        if self.groupacv_result is not None and self.groupacv_objective is not None:
            lines.append(
                f"  GroupACV best: {type(self.groupacv_result.estimator).__name__} "
                f"(obj={self.groupacv_objective:.6f})"
            )
        return "\n".join(lines)


def unified_search(
    acv_search: Optional["ACVSearch[Array]"] = None,
    groupacv_search: Optional["GroupACVSearch[Array]"] = None,
    target_cost: float = 1000.0,
    allow_failures: bool = True,
) -> UnifiedSearchResult[Array]:
    """Run both searches and compare results.

    Parameters
    ----------
    acv_search : ACVSearch, optional
        Configured ACV search. If None, skips ACV.
    groupacv_search : GroupACVSearch, optional
        Configured GroupACV search. If None, skips GroupACV.
    target_cost : float
        The computational budget.
    allow_failures : bool
        If True, continue on allocation failures.

    Returns
    -------
    UnifiedSearchResult
        Contains best estimator and comparison info.

    Raises
    ------
    ValueError
        If both searches are None.
    RuntimeError
        If no successful allocations found.
    """
    if acv_search is None and groupacv_search is None:
        raise ValueError(
            "At least one of acv_search or groupacv_search must be provided"
        )

    acv_result: Optional["SearchResult[Array]"] = None
    groupacv_result: Optional["GroupACVSearchResult[Array]"] = None
    acv_obj: Optional[float] = None
    groupacv_obj: Optional[float] = None

    # Run ACV search
    if acv_search is not None:
        try:
            acv_result = acv_search.search(target_cost, allow_failures=allow_failures)
            # search() only returns successfully if allocation succeeded
            bkd = acv_search.bkd()
            acv_obj = bkd.to_float(acv_result.allocation.objective_value[0])
        except RuntimeError:
            if not allow_failures:
                raise

    # Run GroupACV search
    if groupacv_search is not None:
        try:
            groupacv_result = groupacv_search.search(
                target_cost, allow_failures=allow_failures
            )
            # search() only returns successfully if allocation succeeded
            bkd = groupacv_search.bkd()
            groupacv_obj = bkd.to_float(
                groupacv_result.allocation.objective_value[0]
            )
        except RuntimeError:
            if not allow_failures:
                raise

    # Compare and select best
    acv_effective = acv_obj if acv_obj is not None else float("inf")
    groupacv_effective = groupacv_obj if groupacv_obj is not None else float("inf")

    if acv_effective == float("inf") and groupacv_effective == float("inf"):
        raise RuntimeError("No successful allocations found in either search")

    if acv_effective <= groupacv_effective:
        return UnifiedSearchResult(
            best_estimator=acv_result.estimator,
            best_objective=acv_effective,
            best_family=EstimatorFamily.ACV,
            acv_result=acv_result,
            groupacv_result=groupacv_result,
            acv_objective=acv_obj,
            groupacv_objective=groupacv_obj,
        )
    else:
        return UnifiedSearchResult(
            best_estimator=groupacv_result.estimator,
            best_objective=groupacv_effective,
            best_family=EstimatorFamily.GROUPACV,
            acv_result=acv_result,
            groupacv_result=groupacv_result,
            acv_objective=acv_obj,
            groupacv_objective=groupacv_obj,
        )
