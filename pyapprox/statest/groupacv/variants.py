"""GroupACV estimator variants.

This module provides concrete implementations of BaseGroupACVEstimator:
    - GroupACVEstimatorIS: Independent sampling estimator
    - GroupACVEstimatorNested: Nested sampling estimator
"""

from typing import List, TYPE_CHECKING

from pyapprox.util.backends.protocols import Array

from pyapprox.statest.groupacv.base import BaseGroupACVEstimator
from pyapprox.statest.groupacv.utils import (
    _get_allocation_matrix_is,
    _get_allocation_matrix_nested,
    _nest_subsets,
)

if TYPE_CHECKING:
    from pyapprox.statest.statistics import MultiOutputStatistic


class GroupACVEstimatorIS(BaseGroupACVEstimator[Array]):
    """GroupACV estimator with Independent Sampling.

    This estimator uses independent partitions where each subset has its own
    separate sample partition. The allocation matrix is identity-like.

    Parameters
    ----------
    stat : MultiOutputStatistic
        The statistic object containing covariance information

    costs : Array
        The computational costs of each model

    reg_blue : float, optional
        Regularization parameter for BLUE. Default is 0.

    model_subsets : List[Array], optional
        List of model subsets. If None, all subsets are generated.

    asketch : Array, optional
        Sketch matrix for extracting statistics. If None, identity-like
        matrix extracting high-fidelity model statistics.

    use_pseudo_inv : bool, optional
        Whether to use pseudo-inverse. Default is True.
    """

    def _get_allocation_matrix(self, subsets: List[Array]) -> Array:
        """Get independent sampling allocation matrix."""
        return _get_allocation_matrix_is(subsets, self._bkd)


class GroupACVEstimatorNested(BaseGroupACVEstimator[Array]):
    """GroupACV estimator with Nested Sampling.

    This estimator uses nested partitions where samples are shared across
    subsets in a hierarchical manner. The allocation matrix is lower
    triangular.

    Parameters
    ----------
    stat : MultiOutputStatistic
        The statistic object containing covariance information

    costs : Array
        The computational costs of each model

    reg_blue : float, optional
        Regularization parameter for BLUE. Default is 0.

    model_subsets : List[Array], optional
        List of model subsets. If None, all subsets are generated.

    asketch : Array, optional
        Sketch matrix for extracting statistics. If None, identity-like
        matrix extracting high-fidelity model statistics.

    use_pseudo_inv : bool, optional
        Whether to use pseudo-inverse. Default is True.
    """

    def _preprocess_model_subsets(
        self, model_subsets: List[Array]
    ) -> List[Array]:
        """Preprocess model subsets for nested sampling.

        Filters out the zero subset (if present) and nests the remaining
        subsets for hierarchical sampling.
        """
        zero = self._bkd.zeros((1,), dtype=int)
        filtered = []
        for subset in model_subsets:
            if not isinstance(subset, self._bkd.array_type()):
                raise ValueError(
                    f"subset must be an instance of {self._bkd.array_type()}"
                )
            if not self._bkd.allclose(subset, zero):
                filtered.append(subset)
        return _nest_subsets(filtered, self.nmodels(), self._bkd)[0]

    def _get_allocation_matrix(self, subsets: List[Array]) -> Array:
        """Get nested sampling allocation matrix."""
        return _get_allocation_matrix_nested(subsets, self._bkd)
