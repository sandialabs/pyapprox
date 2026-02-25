"""AETCBLUE implementation.

This module implements the AETCBLUE class which uses MLBLUEEstimator
for optimal sample allocation in the exploitation phase.
"""

from typing import Generic, List, Tuple, Optional, Callable, Any, Dict
from functools import partial

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.statest.aetc.base import AETC
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.statest.groupacv import (
    MLBLUEEstimator,
    GroupACVAllocationOptimizer,
    GroupACVAllocationResult,
    default_groupacv_optimizer,
)


class AETCBLUE(AETC[Array]):
    """AETC with MLBLUE-based sample allocation.

    This class uses the Multi-Level Best Linear Unbiased Estimator (MLBLUE)
    for optimal sample allocation during the exploitation phase.

    Parameters
    ----------
    models : List[Callable]
        List of callable functions with signature fun(samples) -> Array.
        Returns shape (nqoi, nsamples).
    rvs : Callable
        Function to generate random samples with signature rvs(nsamples) -> Array.
        Returns shape (nvars, nsamples).
    costs : Array
        Cost of evaluating each model.
    oracle_stats : Optional[List[Array]]
        Oracle statistics for testing. First element is covariance, second is means.
    reg_blue : float
        Regularization parameter for BLUE. Default is 1e-15.
    optimizer : Optional[Any]
        Optimizer for MLBLUE allocation. If None, uses default optimizer.
    bkd : Backend[Array]
        Backend for numerical computations.
    """

    def __init__(
        self,
        models: List[Callable[[Array], Array]],
        rvs: Callable[[int], Array],
        costs: Optional[Array],
        oracle_stats: Optional[List[Array]],
        bkd: Backend[Array],
        reg_blue: float = 1e-15,
        optimizer: Optional[Any] = None,
    ) -> None:
        if optimizer is None:
            optimizer = default_groupacv_optimizer()
        self._optimizer = optimizer
        self._reg_blue = reg_blue
        super().__init__(models, rvs, costs, oracle_stats, bkd)

    def set_optimizer(self, optimizer: Any) -> None:
        """Set the optimizer for MLBLUE allocation.

        Parameters
        ----------
        optimizer : Any
            Optimizer instance (GroupACVOptimizer or ChainedOptimizer).
        """
        self._optimizer = optimizer

    def _find_k2(
        self,
        beta_Sp: Array,
        Sigma_S: Array,
        costs_S: Array,
        round_nsamples: bool = False,
    ) -> Tuple[Array, Array]:
        """Find optimal k2 and sample allocation using MLBLUE.

        Parameters
        ----------
        beta_Sp : Array
            Least squares coefficients, shape (ncovariates + 1, 1).
        Sigma_S : Array
            Covariance matrix of covariates, shape (ncovariates, ncovariates).
        costs_S : Array
            Costs of models in subset.
        round_nsamples : bool
            Whether to round sample counts to integers.

        Returns
        -------
        k2 : Array
            Optimal k2 value (scalar).
        nsamples_per_subset : Array
            Optimal sample allocation per model.
        """
        bkd = self._bkd

        # Remove high-fidelity coefficient to get asketch
        asketch = beta_Sp[1:].T

        # Create MultiOutputMean statistic with covariance
        stat_S = MultiOutputMean(1, bkd)
        stat_S.set_pilot_quantities(Sigma_S)

        # Create MLBLUE estimator
        est = MLBLUEEstimator(
            stat_S,
            costs_S,
            reg_blue=self._reg_blue,
            asketch=asketch,
        )

        # Set target cost and allocate samples using new API
        target_cost = 10 * float(bkd.to_numpy(costs_S[0]))
        allocator = GroupACVAllocationOptimizer(est, optimizer=self._optimizer)
        result = allocator.optimize(
            target_cost, min_nhf_samples=0, round_nsamples=round_nsamples
        )
        est.set_allocation(result)

        # Get allocation and scale by target cost
        nsamples_per_subset = bkd.maximum(
            bkd.zeros(est.npartition_samples().shape),
            est.npartition_samples(),
        )

        # Get k2 from the estimator's optimized criteria (same as legacy)
        # Note: optimized_criteria is the objective value from the optimizer,
        # which is NOT the same as logdet(covariance). Legacy uses
        # k2 = est.optimized_criteria().squeeze() * target_cost
        criteria = est.optimized_criteria()
        if hasattr(criteria, 'squeeze'):
            criteria = criteria.squeeze()
        elif criteria.ndim > 0:
            criteria = criteria.ravel()[0]
        k2 = criteria * target_cost

        nsamples_per_subset = nsamples_per_subset / target_cost

        return k2, nsamples_per_subset

    def _create_exploit_estimator(self, result: Tuple) -> MLBLUEEstimator:
        """Create MLBLUEEstimator for exploitation phase.

        Parameters
        ----------
        result : Tuple
            Exploration result tuple.

        Returns
        -------
        MLBLUEEstimator
            Configured estimator for exploitation.
        """
        bkd = self._bkd
        best_subset = result[1]
        beta_Sp, Sigma_best_S, rounded_nsamples_per_subset = result[3:6]
        costs_best_S = self._costs[best_subset + 1]
        beta_best_S = beta_Sp[1:]

        stat_best_S = MultiOutputMean(1, bkd)
        stat_best_S.set_pilot_quantities(Sigma_best_S)
        # Note: Legacy incorrectly passes Sigma_best_S as reg_blue (a matrix
        # instead of a scalar). This implementation uses the correct reg_blue.
        est = MLBLUEEstimator(
            stat_best_S,
            costs_best_S,
            asketch=beta_best_S.T,
            reg_blue=self._reg_blue,
        )
        # Create allocation result and set it
        nsamples_per_model = est._compute_nsamples_per_model(
            rounded_nsamples_per_subset
        )
        actual_cost = float(est._estimator_cost(rounded_nsamples_per_subset))
        allocation = GroupACVAllocationResult(
            npartition_samples=rounded_nsamples_per_subset,
            nsamples_per_model=nsamples_per_model,
            actual_cost=actual_cost,
            objective_value=bkd.array([0.0]),  # Placeholder
            success=True,
            message="",
        )
        est.set_allocation(allocation)
        return est

    def get_exploit_samples(
        self, result: Tuple, random_states: Optional[Any] = None
    ) -> Tuple[List[Array], List[int]]:
        """Get samples for exploitation phase.

        Parameters
        ----------
        result : Tuple
            Exploration result tuple.
        random_states : Optional[Any]
            Random states for reproducibility.

        Returns
        -------
        samples_per_model : List[Array]
            Samples for each model in the best subset.
        best_subset_HF : List[int]
            Indices of models in best subset (1-indexed for HF model).
        """
        if random_states is not None:
            rvs = partial(self._rvs, random_states=random_states)
        else:
            rvs = self._rvs

        best_subset = result[1]
        est = self._create_exploit_estimator(result)
        samples_per_model = est.generate_samples_per_model(rvs)

        # Convert to HF-indexed subset (add 1 to account for HF model at index 0)
        best_subset_HF = [int(s) + 1 for s in best_subset]

        return samples_per_model, best_subset_HF

    def find_exploit_mean(
        self, values_per_model: List[Array], result: Tuple
    ) -> Array:
        """Compute exploitation mean estimate using MLBLUE.

        Parameters
        ----------
        values_per_model : List[Array]
            Model evaluations for each model in subset.
            Each array has shape (nqoi, nsamples).
        result : Tuple
            Exploration result tuple.

        Returns
        -------
        mean : Array
            Estimated mean (scalar).
        """
        beta_Sp = result[3]
        est = self._create_exploit_estimator(result)

        # Use MLBLUE estimator to compute the weighted estimate
        product = est(values_per_model)
        # Handle both scalar and array returns
        if hasattr(product, 'item'):
            product = product.item()
        elif product.ndim > 0:
            product = product.flatten()[0]

        return beta_Sp[0, 0] + product

    def exploit(self, result: Tuple) -> Array:
        """Run exploitation phase to compute final estimate.

        Parameters
        ----------
        result : Tuple
            Exploration result tuple.

        Returns
        -------
        mean : Array
            Estimated mean (scalar).
        """
        samples_per_model, best_subset = self.get_exploit_samples(result)

        # Evaluate models
        values_per_model = [
            self._models[s](samples)
            for s, samples in zip(best_subset, samples_per_model)
        ]

        return self.find_exploit_mean(values_per_model, result)

    def _explore_result_to_dict(self, result: Tuple) -> Dict[str, Any]:
        """Convert exploration result tuple to dictionary.

        Parameters
        ----------
        result : Tuple
            Exploration result tuple.

        Returns
        -------
        Dict[str, Any]
            Dictionary with named result fields.
        """
        return {
            "nexplore_samples": result[0],
            "subset": result[1],
            "subset_cost": result[2],
            "beta_Sp": result[3],
            "sigma_S": result[4],
            "rounded_nsamples_per_subset": result[5],
            "nsamples_per_subset": result[6],
            "loss": result[7],
            "k1": result[8],
            # BLUE_variance is for unrounded nsamples_per_subset
            "BLUE_variance": result[9],
            "exploit_budget": result[10],
            "mlblue_subset_costs": result[11],
            "explore_budget": result[0] * (result[2] + self._costs[0]),
        }
