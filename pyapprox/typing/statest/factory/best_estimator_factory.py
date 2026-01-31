"""Best estimator factory for statest module.

Provides BestEstimatorFactory that searches over estimator types, recursion
indices, and model subsets to find the optimal ACV estimator configuration.
"""

from itertools import combinations
from typing import (
    Generic,
    List,
    Dict,
    Any,
    Optional,
    Iterator,
)
import time

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.statest.protocols import (
    StatisticProtocol,
    EstimatorProtocol,
)
from pyapprox.typing.statest.statistics import MultiOutputStatistic
from pyapprox.typing.statest.factory.tree_enumeration import (
    get_acv_recursion_indices,
)
from pyapprox.typing.statest.factory.registry import (
    CandidateResult,
    compute_objective,
    create_estimator,
    get_estimator_registry,
    get_statistic_registry,
)


class BestEstimatorFactory(Generic[Array]):
    """Factory to find the best estimator configuration.

    Searches over estimator types (GIS, GRD, GMF), recursion indices
    (all valid tree structures), and optionally model subsets to find
    the configuration with the smallest objective value.

    This is NOT an estimator - it's a factory that returns the best
    estimator after searching over configurations.

    New estimator types can be registered using `register_estimator()`.

    Parameters
    ----------
    stat : StatisticProtocol[Array]
        Statistic to estimate. Must have pilot quantities set.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array]
        Computational backend.
    estimator_types : List[str], optional
        Estimator types to compare. Default: ["gmf", "gis", "grd"].
    max_depth : int, optional
        Maximum recursion tree depth. Default: 4.
    max_nmodels : int, optional
        Maximum models to use. Default: 4.
        Set to None to use all available models.
    min_nmodels : int, optional
        Minimum models to use. Default: 2.
    require_hf : bool, optional
        Always include model 0 (HF). Default: True.
    allow_failures : bool, optional
        Continue if some configurations fail. Default: True.
    save_candidates : bool, optional
        Store all candidate results for analysis. Default: False.
    verbosity : int, optional
        Verbosity level (0=silent, 1=summary, 2=progress, 3=debug).

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.statest import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> cov = bkd.asarray([[1.0, 0.9, 0.8],
    ...                    [0.9, 1.0, 0.85],
    ...                    [0.8, 0.85, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([100.0, 10.0, 1.0])
    >>>
    >>> factory = BestEstimatorFactory(stat, costs, bkd)
    >>> factory.allocate_samples(target_cost=1000.0)
    >>> best_est = factory.best_estimator()
    """

    def __init__(
        self,
        stat: StatisticProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
        estimator_types: Optional[List[str]] = None,
        max_depth: int = 4,
        max_nmodels: int = 4,
        min_nmodels: int = 2,
        require_hf: bool = True,
        allow_failures: bool = True,
        save_candidates: bool = False,
        verbosity: int = 0,
    ) -> None:
        self._stat = stat
        self._costs = costs
        self._bkd = bkd
        self._nmodels_total = costs.shape[0]

        if estimator_types is None:
            estimator_types = ["gmf", "gis", "grd"]
        self._estimator_types = estimator_types

        self._max_depth = max_depth
        self._max_nmodels = max_nmodels
        self._min_nmodels = min_nmodels
        self._require_hf = require_hf
        self._allow_failures = allow_failures
        self._save_candidates = save_candidates
        self._verbosity = verbosity

        # Results
        self._best_result: Optional[CandidateResult] = None
        self._candidate_results: List[CandidateResult] = []
        self._search_stats: Dict[str, Any] = {}

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nmodels_total(self) -> int:
        """Return total number of available models."""
        return self._nmodels_total

    # --- Model Subset Enumeration ---

    def _generate_model_subsets(self) -> Iterator[List[int]]:
        """Generate model subsets to search.

        Yields
        ------
        List[int]
            Model indices for each subset.
        """
        nmodels = self._nmodels_total
        max_n = self._max_nmodels if self._max_nmodels else nmodels
        max_n = min(max_n, nmodels)
        min_n = self._min_nmodels

        if self._require_hf:
            # Always include model 0, vary LF models
            lf_models = list(range(1, nmodels))
            for nlf in range(min_n - 1, max_n):
                for lf_subset in combinations(lf_models, nlf):
                    yield [0] + list(lf_subset)
        else:
            # All combinations (less common)
            for n in range(min_n, max_n + 1):
                for subset in combinations(range(nmodels), n):
                    yield list(subset)

    def _count_model_subsets(self) -> int:
        """Count total number of model subsets to search."""
        return sum(1 for _ in self._generate_model_subsets())

    # --- Subset Statistic Creation ---

    def _create_subset_stat(
        self, model_indices: List[int]
    ) -> MultiOutputStatistic[Array]:
        """Create a statistic for a model subset.

        Parameters
        ----------
        model_indices : List[int]
            Indices of models in the subset.

        Returns
        -------
        MultiOutputStatistic[Array]
            Statistic with subset covariance.
        """
        bkd = self._bkd
        cov = self._stat.pilot_covariance()
        nqoi = self._stat.nqoi()
        stat_type_name = type(self._stat).__name__

        # Extract subset covariance
        cov_np = bkd.to_numpy(cov)
        if nqoi == 1:
            # Simple case: cov is nmodels x nmodels
            subset_cov = cov_np[np.ix_(model_indices, model_indices)]
        else:
            # Multi-QoI: cov is (nmodels*nqoi) x (nmodels*nqoi)
            indices = []
            for m in model_indices:
                indices.extend(range(m * nqoi, (m + 1) * nqoi))
            subset_cov = cov_np[np.ix_(indices, indices)]

        subset_cov = bkd.asarray(subset_cov)

        # Create same type as original stat using registry
        stat_registry = get_statistic_registry()
        if stat_type_name not in stat_registry:
            raise ValueError(
                f"Unknown statistic type: {stat_type_name}. "
                f"Registered types: {list(stat_registry.keys())}. "
                f"Use register_statistic() to register new types."
            )
        new_stat = stat_registry[stat_type_name](nqoi=nqoi, bkd=bkd)

        new_stat.set_pilot_quantities(subset_cov)
        return new_stat

    # --- Single Candidate Evaluation ---

    def _evaluate_candidate(
        self,
        est_type: str,
        model_indices: List[int],
        recursion_index: Optional[Array],
        target_cost: float,
    ) -> CandidateResult:
        """Evaluate a single candidate configuration.

        Parameters
        ----------
        est_type : str
            Estimator type.
        model_indices : List[int]
            Model indices to use.
        recursion_index : Optional[Array]
            Recursion index (for ACV types).
        target_cost : float
            Budget.

        Returns
        -------
        CandidateResult
            Evaluation result.
        """
        bkd = self._bkd

        try:
            # Create subset stat and costs
            subset_stat = self._create_subset_stat(model_indices)
            costs_np = bkd.to_numpy(self._costs)
            subset_costs = bkd.asarray(costs_np[model_indices])

            # Create estimator using registry
            estimator = create_estimator(
                est_type, subset_stat, subset_costs, recursion_index
            )

            # Allocate samples
            estimator.allocate_samples(target_cost)

            # Compute objective (log-det of covariance)
            cov = estimator.optimized_covariance()
            objective_value = compute_objective(cov, bkd)

            return CandidateResult(
                estimator_type=est_type,
                model_indices=model_indices,
                recursion_index=recursion_index,
                objective_value=objective_value,
                estimator=estimator,
                success=True,
            )

        except Exception as e:
            if self._verbosity >= 3:
                print(f"  Failed: {est_type}, models={model_indices}, error={e}")
            return CandidateResult(
                estimator_type=est_type,
                model_indices=model_indices,
                recursion_index=recursion_index,
                objective_value=float("inf"),
                estimator=None,
                success=False,
                error_message=str(e),
            )

    # --- Main Search Algorithm ---

    def allocate_samples(self, target_cost: float) -> None:
        """Run comprehensive search and find best estimator.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        start_time = time.time()
        bkd = self._bkd

        best_objective = float("inf")
        best_result: Optional[CandidateResult] = None
        total_candidates = 0
        successful = 0
        failed = 0

        if self._save_candidates:
            self._candidate_results = []

        estimator_registry = get_estimator_registry()

        # Enumerate all configurations
        for model_indices in self._generate_model_subsets():
            nsubset = len(model_indices)

            for est_type in self._estimator_types:
                # Check if this estimator type requires recursion index
                key = est_type.lower()
                if key not in estimator_registry:
                    continue
                _, requires_recursion = estimator_registry[key]

                # For types requiring recursion index, enumerate them
                if requires_recursion:
                    depth = min(self._max_depth, nsubset - 1)
                    recursion_iter: Iterator[Optional[Array]] = iter(
                        get_acv_recursion_indices(nsubset, depth, bkd)
                    )
                else:
                    # Types that don't need recursion index (mfmc, mlmc)
                    recursion_iter = iter([None])

                for recursion_index in recursion_iter:
                    total_candidates += 1

                    if self._verbosity >= 2:
                        if recursion_index is not None:
                            rec_str = str(
                                bkd.to_numpy(recursion_index).astype(int).tolist()
                            )
                        else:
                            rec_str = "N/A"
                        print(
                            f"  Candidate {total_candidates}: "
                            f"{est_type}, models={model_indices}, rec={rec_str}"
                        )

                    result = self._evaluate_candidate(
                        est_type, model_indices, recursion_index, target_cost
                    )

                    if result.success:
                        successful += 1
                        if result.objective_value < best_objective:
                            best_objective = result.objective_value
                            best_result = result
                            if self._verbosity >= 1:
                                print(
                                    f"  New best: {est_type}, "
                                    f"models={model_indices}, "
                                    f"objective={best_objective:.6f}"
                                )
                    else:
                        failed += 1
                        if not self._allow_failures:
                            raise RuntimeError(
                                f"Candidate failed: {result.error_message}"
                            )

                    if self._save_candidates:
                        self._candidate_results.append(result)

        elapsed = time.time() - start_time

        self._search_stats = {
            "total_candidates": total_candidates,
            "successful": successful,
            "failed": failed,
            "elapsed_seconds": elapsed,
        }

        if self._verbosity >= 1:
            print(
                f"Search complete: {successful}/{total_candidates} successful, "
                f"{elapsed:.2f}s"
            )

        if best_result is None:
            raise RuntimeError("No valid configurations found")

        self._best_result = best_result

    # --- Result Accessors ---

    def best_estimator(self) -> EstimatorProtocol[Array]:
        """Return the best estimator.

        Raises
        ------
        ValueError
            If allocate_samples has not been called.
        """
        if self._best_result is None:
            raise ValueError("Call allocate_samples first.")
        return self._best_result.estimator

    def best_type(self) -> str:
        """Return the type of the best estimator."""
        if self._best_result is None:
            raise ValueError("Call allocate_samples first.")
        return self._best_result.estimator_type

    def best_models(self) -> List[int]:
        """Return model indices used by the best estimator."""
        if self._best_result is None:
            raise ValueError("Call allocate_samples first.")
        return self._best_result.model_indices

    def best_recursion_index(self) -> Optional[Array]:
        """Return recursion index of the best estimator."""
        if self._best_result is None:
            raise ValueError("Call allocate_samples first.")
        return self._best_result.recursion_index

    def best_objective_value(self) -> float:
        """Return objective value (log-det) of the best estimator."""
        if self._best_result is None:
            raise ValueError("Call allocate_samples first.")
        return self._best_result.objective_value

    def candidate_results(self) -> List[CandidateResult]:
        """Return all candidate results (if save_candidates=True)."""
        return self._candidate_results

    def search_stats(self) -> Dict[str, Any]:
        """Return search statistics."""
        return self._search_stats

    def nsamples_per_model(self) -> Array:
        """Return samples per model for best estimator."""
        return self.best_estimator().nsamples_per_model()

    def optimized_covariance(self) -> Array:
        """Return covariance of the best estimator."""
        return self.best_estimator().optimized_covariance()

    def __call__(self, values: List[Array]) -> Array:
        """Compute estimate using best estimator.

        Parameters
        ----------
        values : List[Array]
            Model outputs for the models in best_models().

        Returns
        -------
        Array
            Estimated statistic.
        """
        return self.best_estimator()(values)

    def __repr__(self) -> str:
        if self._best_result is None:
            return "BestEstimatorFactory(not searched)"
        return (
            f"BestEstimatorFactory("
            f"type={self._best_result.estimator_type!r}, "
            f"models={self._best_result.model_indices}, "
            f"objective={self._best_result.objective_value:.4f})"
        )
