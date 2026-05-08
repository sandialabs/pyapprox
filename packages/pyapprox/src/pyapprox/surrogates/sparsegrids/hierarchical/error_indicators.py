"""Per-point error indicators for hierarchical sparse grid priority queue."""

from typing import Generic, Tuple, runtime_checkable

from typing_extensions import Protocol

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class HierarchicalErrorIndicator(Protocol[Array]):
    """Per-point error indicator for the priority queue."""

    def __call__(
        self,
        surplus: Array,
        quad_weight: float,
        cost: float,
    ) -> Tuple[float, float]:
        """Compute priority and error from a point's surplus.

        Parameters
        ----------
        surplus : Array
            Hierarchical surplus, shape (nqoi,).
        quad_weight : float
            ND quadrature weight for this point.
        cost : float
            Evaluation cost from the cost model.

        Returns
        -------
        priority : float
            Priority for the max-heap (higher = refined sooner).
        error : float
            Error contribution of this point.
        """
        ...


class GammaIndicator(Generic[Array]):
    """gamma = |sum_q v_q| * w / cost."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(
        self,
        surplus: Array,
        quad_weight: float,
        cost: float,
    ) -> Tuple[float, float]:
        if cost <= 0:
            raise ValueError(f"cost must be positive, got {cost}")
        bkd = self._bkd
        abs_weighted = float(bkd.to_float(bkd.sum(bkd.abs(surplus)))) * quad_weight
        gamma = abs_weighted / cost
        return (gamma, abs_weighted)


class L2SurplusPointIndicator(Generic[Array]):
    """priority = ||v||_2 / cost."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self._bkd = bkd

    def __call__(
        self,
        surplus: Array,
        quad_weight: float,
        cost: float,
    ) -> Tuple[float, float]:
        if cost <= 0:
            raise ValueError(f"cost must be positive, got {cost}")
        bkd = self._bkd
        l2 = float(bkd.to_float(bkd.norm(surplus)))
        priority = l2 / cost
        error = l2 * quad_weight
        return (priority, error)
