"""AETC base class implementation.

This module implements the AETC (Adaptive Efficient Test Collection) base class
for multi-fidelity Monte Carlo estimation with explore/exploit phases.
"""

from typing import Generic, List, Tuple, Optional, Callable, Any, Dict
from functools import partial

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg import extract_submatrix
from pyapprox.statest.groupacv.utils import get_model_subsets


class AETC(Generic[Array]):
    """Adaptive Efficient Test Collection base class.

    This class implements the core AETC algorithm that balances exploration
    (gathering pilot samples) and exploitation (computing estimates) for
    multi-fidelity Monte Carlo estimation.

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
    ) -> None:
        self._bkd = bkd
        self._models = models
        self._nmodels = len(models)
        if not callable(rvs):
            raise ValueError("rvs must be callable")
        self._rvs = rvs
        self._costs = self._validate_costs(costs)
        self._oracle_stats = oracle_stats

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def _validate_costs(self, costs: Optional[Array]) -> Optional[Array]:
        """Validate and convert costs array."""
        if costs is None:
            return None
        if len(costs) != self._nmodels:
            raise ValueError("costs must be provided for each model")
        return self._bkd.asarray(costs)

    def _least_squares(
        self, hf_values: Array, covariate_values: Array
    ) -> Tuple[Array, Array, Array]:
        """Compute least squares fit of high-fidelity values to covariates.

        Parameters
        ----------
        hf_values : Array
            High-fidelity model evaluations, shape (nqoi, nsamples).
            For AETC, nqoi=1 so shape is (1, nsamples).
        covariate_values : Array
            Low-fidelity model evaluations, shape (ncovariates, nsamples).

        Returns
        -------
        beta_Sp : Array
            Least squares coefficients, shape (ncovariates + 1, 1).
            First element is intercept, rest are covariate coefficients.
        sigma_S_sq : Array
            Estimated residual variance (scalar as 0-d array).
        X_Sp : Array
            Design matrix with leading column of ones, shape (nsamples, ncovariates + 1).
        """
        bkd = self._bkd
        # Input is (ncovariates, nsamples), transpose to (nsamples, ncovariates)
        covariate_values_T = covariate_values.T
        hf_values_T = hf_values.T  # (nsamples, nqoi) = (nsamples, 1)
        nsamples, ncovariates = covariate_values_T.shape

        # X_Sp is (1, X_S) - design matrix with intercept
        X_Sp = bkd.hstack([bkd.ones((nsamples, 1)), covariate_values_T])

        # beta_Sp = least squares solution
        beta_Sp = bkd.lstsq(X_Sp, hf_values_T)

        # Ensure beta_Sp is 2D column vector
        if beta_Sp.ndim == 1:
            beta_Sp = bkd.reshape(beta_Sp, (-1, 1))

        # sigma_S_sq = residual variance estimate
        residuals = hf_values_T - X_Sp @ beta_Sp
        sigma_S_sq = bkd.sum(residuals**2) / (nsamples - 1)

        return beta_Sp, sigma_S_sq, X_Sp

    def _subset_oracle_stats(
        self, oracle_stats: List[Array], covariate_subset: Array
    ) -> Tuple[Array, Array, Array]:
        """Extract oracle statistics for a subset of covariates.

        Parameters
        ----------
        oracle_stats : List[Array]
            List containing [covariance, means] where:
            - covariance has shape (nmodels, nmodels)
            - means has shape (nmodels, 1)
        covariate_subset : Array
            Indices of low-fidelity models in the subset (0-indexed for LF models,
            so model 0 is the first LF model, not the HF model).

        Returns
        -------
        Sigma_S : Array
            Covariance matrix of covariates in subset, shape (nsubset, nsubset).
        Lambda_Sp : Array
            Second moment matrix E[X_Sp X_Sp^T], shape (nsubset+1, nsubset+1).
        x_Sp : Array
            Mean vector (1, means of subset), shape (nsubset+1, 1).
        """
        bkd = self._bkd
        cov, means = oracle_stats[0], oracle_stats[1]

        # covariate_subset indexes LF models (0-based), so add 1 to get model indices
        # since model 0 is HF
        subset_indices = bkd.asarray(covariate_subset + 1, dtype=int)

        # Sigma_S = covariance of subset models
        Sigma_S = extract_submatrix(cov, subset_indices, subset_indices)

        # Sp_subset = [0, subset_indices] for indexing into full matrices
        Sp_subset = bkd.hstack([bkd.zeros((1,), dtype=int), subset_indices])

        # x_Sp = (1, means[subset])^T - mean vector with leading 1
        x_Sp = bkd.vstack([bkd.ones((1, 1)), means[subset_indices]])

        # Lambda_Sp = E[X_Sp X_Sp^T] where X_Sp = (1, X_S)
        # Build tmp1: zeros with cov[1:,1:] in lower-right block
        tmp1 = bkd.zeros(cov.shape)
        tmp1[1:, 1:] = cov[1:, 1:]

        # tmp2 = (1, means[1:])^T
        tmp2 = bkd.vstack([bkd.ones((1, 1)), means[1:]])

        # Lambda = tmp1 + tmp2 @ tmp2^T, then extract subset
        Lambda_full = tmp1 + tmp2 @ tmp2.T
        Lambda_Sp = extract_submatrix(Lambda_full, Sp_subset, Sp_subset)

        return Sigma_S, Lambda_Sp, x_Sp

    def _find_k2(
        self,
        beta_Sp: Array,
        Sigma_S: Array,
        costs_S: Array,
        round_nsamples: bool = False,
    ) -> Tuple[Array, Array]:
        """Find optimal k2 and sample allocation.

        Must be implemented by subclasses (AETCBLUE, AETCMC).

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
        raise NotImplementedError("Subclasses must implement _find_k2")

    def _allocate_samples(
        self,
        beta_Sp: Array,
        Sigma_S: Array,
        sigma_S_sq: Array,
        x_Sp: Array,
        Lambda_Sp: Array,
        costs_S: Array,
        exploit_budget: Array,
    ) -> Tuple[Array, Array, Array]:
        """Allocate samples for exploitation phase.

        Parameters
        ----------
        beta_Sp : Array
            Least squares coefficients, shape (ncovariates + 1, 1).
        Sigma_S : Array
            Covariance matrix of covariates, shape (ncovariates, ncovariates).
        sigma_S_sq : Array
            Estimated residual variance (scalar).
        x_Sp : Array
            Mean vector (1, means of subset), shape (ncovariates + 1, 1).
        Lambda_Sp : Array
            Second moment matrix, shape (ncovariates + 1, ncovariates + 1).
        costs_S : Array
            Costs of models in subset.
        exploit_budget : Array
            Budget for exploitation phase.

        Returns
        -------
        k1 : Array
            k1 value for optimal loss calculation (scalar).
        k2 : Array
            k2 value for optimal loss calculation (scalar).
        nsamples_per_subset : Array
            Sample allocation per model in subset.
        """
        bkd = self._bkd
        nmodels = len(costs_S)

        # k1 = sigma_S_sq * trace(x_Sp @ x_Sp.T @ inv(Lambda_Sp))
        k1 = sigma_S_sq * bkd.trace(
            bkd.multidot([x_Sp, x_Sp.T, bkd.inv(Lambda_Sp)])
        )

        # Build Sigma_Sp: zeros with Sigma_S in lower-right block
        Sigma_Sp = bkd.zeros((Sigma_S.shape[0] + 1, Sigma_S.shape[1] + 1))
        Sigma_Sp[1:, 1:] = Sigma_S

        # Special case: single covariate
        if nmodels == 1:
            exploit_cost = bkd.sum(costs_S)
            nsamples_per_subset = 1.0 / exploit_cost
            k2 = exploit_cost * bkd.trace(
                bkd.multidot([Sigma_Sp, beta_Sp, beta_Sp.T])
            )
            return k1, k2, nsamples_per_subset * bkd.ones((1,))

        # General case: use subclass-specific _find_k2
        k2, nsamples_per_subset = self._find_k2(beta_Sp, Sigma_S, costs_S)

        return k1, k2, nsamples_per_subset

    def _optimal_loss(
        self,
        total_budget: Array,
        hf_values: Array,
        covariate_values: Array,
        costs: Array,
        covariate_subset: Array,
        alpha: float,
        exploit_budget: Array,
    ) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        """Compute optimal loss for a given covariate subset.

        Parameters
        ----------
        total_budget : Array
            Total budget for exploration and exploitation.
        hf_values : Array
            High-fidelity model evaluations, shape (nqoi, nsamples).
            For AETC, nqoi=1 so shape is (1, nsamples).
        covariate_values : Array
            Low-fidelity model evaluations, shape (nmodels-1, nsamples).
        costs : Array
            Costs of all models, shape (nmodels,).
        covariate_subset : Array
            Indices of LF models in subset (0-indexed).
        alpha : float
            Regularization parameter.
        exploit_budget : Array
            Budget for exploitation phase.

        Returns
        -------
        opt_loss : Array
            Optimal loss value.
        nsamples_per_subset : Array
            Sample allocation per model.
        explore_rate : Array
            Optimal exploration rate.
        beta_Sp : Array
            Least squares coefficients.
        Sigma_S : Array
            Covariance matrix of subset.
        k1 : Array
            k1 value.
        k2 : Array
            k2 value.
        exploit_budget : Array
            Updated exploitation budget.
        """
        bkd = self._bkd
        nsamples = hf_values.shape[1]  # Typing convention: (nqoi, nsamples)

        # Compute least squares solution
        # covariate_values is (nmodels-1, nsamples), select subset rows
        beta_Sp, sigma_S_sq, X_Sp = self._least_squares(
            hf_values, covariate_values[covariate_subset, :]
        )

        # Get statistics (from oracle or samples)
        if self._oracle_stats is None:
            # Compute from samples
            x_Sp = bkd.mean(X_Sp, axis=0)[:, None]
            # covariate_values[covariate_subset, :] is (nsubset, nsamples)
            # cov expects each row to be a variable, which is what we have
            Sigma_S = bkd.atleast_2d(
                bkd.cov(covariate_values[covariate_subset, :], ddof=1)
            )
            Lambda_Sp = X_Sp.T @ X_Sp / nsamples
        else:
            # Use oracle statistics
            Sigma_S, Lambda_Sp, x_Sp = self._subset_oracle_stats(
                self._oracle_stats, covariate_subset
            )

        # Extract costs of models in subset
        costs_S = costs[covariate_subset + 1]

        # Find optimal sample allocation
        k1, k2, nsamples_per_subset = self._allocate_samples(
            beta_Sp,
            Sigma_S,
            sigma_S_sq,
            x_Sp,
            Lambda_Sp,
            costs_S,
            exploit_budget,
        )

        # Cost of exploration (evaluates all models)
        explore_cost = bkd.sum(costs)

        # Estimate optimal exploration rate (Equation 4.34)
        explore_rate = bkd.maximum(
            total_budget / (
                explore_cost + bkd.sqrt(
                    explore_cost * k2 / (k1 + alpha ** (-nsamples))
                )
            ),
            bkd.asarray(nsamples, dtype=float),
        )

        # Update exploitation budget
        exploit_budget = total_budget - explore_cost * explore_rate

        # Compute optimal loss
        opt_loss = k2 / exploit_budget + (k1 + alpha ** (-nsamples)) / explore_rate

        # Scale sample allocation by exploitation budget
        nsamples_per_subset = nsamples_per_subset * exploit_budget

        return (
            opt_loss,
            nsamples_per_subset,
            explore_rate,
            beta_Sp,
            Sigma_S,
            k1,
            k2,
            exploit_budget,
        )

    def _validate_subsets(
        self, subsets: Optional[List[Array]]
    ) -> Tuple[List[Array], int]:
        """Validate and convert model subsets.

        Parameters
        ----------
        subsets : Optional[List[Array]]
            List of arrays containing low-fidelity model indices (0-indexed).
            If None, generates all possible subsets.

        Returns
        -------
        validated_subsets : List[Array]
            Validated list of subset arrays.
        max_ncovariates : int
            Maximum number of covariates across all subsets.
        """
        bkd = self._bkd

        # Generate default subsets if not provided
        if subsets is None:
            subsets = get_model_subsets(self._nmodels - 1, bkd)

        validated_subsets: List[Array] = []
        max_ncovariates = 0

        for subset in subsets:
            # Check for duplicate indices
            unique_count = bkd.unique(subset).shape[0]
            if unique_count != len(subset):
                raise ValueError(
                    f"subsets provided are not valid. First invalid subset {subset}"
                )
            # Check indices are within valid range (0 to nmodels-2 for LF models)
            if bkd.max(subset) >= self._nmodels - 1:
                raise ValueError(
                    f"subsets provided are not valid. First invalid subset {subset}"
                )
            validated_subsets.append(bkd.asarray(subset, dtype=int))
            max_ncovariates = max(max_ncovariates, len(subset))

        return validated_subsets, max_ncovariates

    def _explore_step(
        self,
        total_budget: Array,
        lf_model_subsets: List[Array],
        values: Array,
        alpha: float,
    ) -> Tuple[
        int, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
    ]:
        """Perform one exploration step.

        Parameters
        ----------
        total_budget : Array
            Total budget for exploration and exploitation.
        lf_model_subsets : List[Array]
            List of LF model subset indices.
        values : Array
            Model evaluations, shape (nmodels, nsamples).
        alpha : float
            Regularization parameter.

        Returns
        -------
        Tuple containing exploration results.
        """
        bkd = self._bkd
        nsamples = values.shape[1]  # Typing convention: (nmodels, nsamples)
        explore_cost = bkd.sum(self._costs)

        # Compute exploitation budget
        exploit_budget = total_budget - nsamples * explore_cost
        if exploit_budget < 0:
            raise RuntimeError("Exploitation budget is negative")

        # Evaluate all subsets
        results = []
        for subset in lf_model_subsets:
            result = self._optimal_loss(
                total_budget,
                values[:1, :],  # HF values (1, nsamples)
                values[1:, :],  # LF values (nmodels-1, nsamples)
                self._costs,
                subset,
                alpha,
                exploit_budget,
            )
            results.append(result)

        # Find best subset (minimum loss)
        losses = bkd.asarray([result[0] for result in results])
        best_subset_idx = int(bkd.argmin(losses))
        best_result = results[best_subset_idx]

        (
            best_loss,
            best_allocation,
            best_rate,
            best_beta_Sp,
            best_Sigma_S,
            best_k1,
            best_k2,
            best_exploit_budget,
        ) = best_result

        best_subset = lf_model_subsets[best_subset_idx]
        best_cost = bkd.sum(self._costs[best_subset + 1])

        # Compute subset group costs
        best_subset_costs = self._costs[best_subset + 1]
        best_subset_groups = get_model_subsets(best_subset.shape[0], bkd)
        best_subset_group_costs = bkd.asarray(
            [bkd.sum(best_subset_costs[group]) for group in best_subset_groups]
        )

        # Compute sample allocations
        best_nsamples_per_subset = bkd.asarray(best_allocation)
        rounded_best_nsamples_per_subset = bkd.floor(best_nsamples_per_subset)
        best_variance = best_k2 / best_exploit_budget

        # Determine number of exploration samples for next iteration
        if best_rate > 2 * nsamples:
            nexplore_samples = 2 * nsamples
        elif best_rate > nsamples:
            nexplore_samples = int(bkd.ceil((nsamples + best_rate) / 2))
        else:
            nexplore_samples = nsamples

        # Ensure we don't exceed budget
        if (total_budget - nexplore_samples * explore_cost) < 0:
            nexplore_samples = int(total_budget / explore_cost)

        return (
            nexplore_samples,
            best_subset,
            best_cost,
            best_beta_Sp,
            best_Sigma_S,
            rounded_best_nsamples_per_subset,
            best_nsamples_per_subset,
            best_loss,
            best_k1,
            best_variance,
            best_exploit_budget,
            best_subset_group_costs,
        )

    def explore(
        self,
        total_budget: Array,
        lf_model_subsets: Optional[List[Array]] = None,
        alpha: float = 4.0,
        random_states: Optional[Any] = None,
    ) -> Tuple[Array, Array, Tuple]:
        """Run exploration phase to find optimal model subset.

        Parameters
        ----------
        total_budget : Array
            Total budget for exploration and exploitation.
        lf_model_subsets : Optional[List[Array]]
            List of LF model subset indices. If None, uses all subsets.
        alpha : float
            Regularization parameter. Default is 4.0.
        random_states : Optional[Any]
            Random states for reproducibility.

        Returns
        -------
        samples : Array
            Samples used in exploration, shape (nvars, nsamples).
        values : Array
            Model evaluations, shape (nmodels, nsamples).
        result : Tuple
            Exploration result tuple.
        """
        bkd = self._bkd

        if self._costs is None:
            raise NotImplementedError("Costs must be provided")

        lf_model_subsets, max_ncovariates = self._validate_subsets(lf_model_subsets)

        # Set up random sampling function
        if random_states is not None:
            rvs = partial(self._rvs, random_states=random_states)
        else:
            rvs = self._rvs

        # Initialize exploration
        nexplore_samples = max_ncovariates + 2
        nexplore_samples_prev = 0
        samples: Optional[Array] = None
        values: Optional[Array] = None
        last_result: Optional[Tuple] = None

        while nexplore_samples - nexplore_samples_prev > 0:
            nnew_samples = nexplore_samples - nexplore_samples_prev
            new_samples = rvs(nnew_samples)
            # Each model returns (nqoi, nnew_samples), stack vertically to get
            # (nmodels*nqoi, nnew_samples) = (nmodels, nnew_samples) for nqoi=1
            new_values = [model(new_samples) for model in self._models]

            if nexplore_samples_prev == 0:
                samples = new_samples
                # Stack models vertically: (nmodels, nsamples)
                values = bkd.vstack(new_values)
                assert values.ndim == 2
            else:
                # Append new samples: (nvars, total_nsamples)
                samples = bkd.hstack([samples, new_samples])
                # Stack new values horizontally: (nmodels, total_nsamples)
                new_values_stacked = bkd.vstack(new_values)
                values = bkd.hstack([values, new_values_stacked])

            nexplore_samples_prev = nexplore_samples
            result = self._explore_step(
                total_budget, lf_model_subsets, values, alpha
            )
            nexplore_samples = result[0]
            last_result = result

        return samples, values, last_result

    def get_exploit_samples(
        self, result: Tuple, random_states: Optional[Any] = None
    ) -> Array:
        """Get samples for exploitation phase.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_exploit_samples")

    def find_exploit_mean(
        self, values_per_model: List[Array], result: Tuple
    ) -> Array:
        """Compute exploitation mean estimate.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement find_exploit_mean")

    def exploit(self, result: Tuple) -> Array:
        """Run exploitation phase to compute final estimate.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement exploit")

    def _explore_result_to_dict(self, result: Tuple) -> Dict[str, Any]:
        """Convert exploration result tuple to dictionary.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _explore_result_to_dict")

    def estimate(
        self,
        total_budget: Array,
        subsets: Optional[List[Array]] = None,
        return_dict: bool = True,
    ) -> Tuple[Array, Array, Any]:
        """Run full AETC estimation (explore + exploit).

        Parameters
        ----------
        total_budget : Array
            Total budget for estimation.
        subsets : Optional[List[Array]]
            Model subsets to consider. If None, uses all subsets.
        return_dict : bool
            If True, convert result to dictionary.

        Returns
        -------
        mean : Array
            Estimated mean.
        values : Array
            Model evaluations from exploration.
        result : Any
            Exploration result (dict or tuple).
        """
        samples, values, result = self.explore(total_budget, subsets)
        mean = self.exploit(result)
        if not return_dict:
            return mean, values, result
        result = self._explore_result_to_dict(result)
        return mean, values, result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
