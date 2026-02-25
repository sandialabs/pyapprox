"""AETCMC implementation.

This module implements the AETCMC class which uses a simpler Monte Carlo-based
approach for sample allocation in the exploitation phase.
"""

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

from pyapprox.statest.aetc.base import AETC
from pyapprox.util.backends.protocols import Array, Backend


class AETCMC(AETC[Array]):
    """AETC with Monte Carlo-based sample allocation.

    This class uses a simpler approach than AETCBLUE where all models in the
    subset share the same samples during exploitation.

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
        super().__init__(models, rvs, costs, oracle_stats, bkd)

    def _find_k2(
        self,
        beta_Sp: Array,
        Sigma_S: Array,
        costs_S: Array,
        round_nsamples: bool = False,
    ) -> Tuple[Array, Array]:
        """Find k2 and sample allocation using MC approach.

        Parameters
        ----------
        beta_Sp : Array
            Least squares coefficients, shape (ncovariates + 1, 1).
        Sigma_S : Array
            Covariance matrix of covariates, shape (ncovariates, ncovariates).
        costs_S : Array
            Costs of models in subset.
        round_nsamples : bool
            Whether to round sample counts to integers (not used in MC).

        Returns
        -------
        k2 : Array
            k2 value (scalar).
        nsamples_per_subset : Array
            Sample allocation (uniform for MC).
        """
        bkd = self._bkd
        asketch = beta_Sp[1:]  # remove high-fidelity coefficient

        exploit_cost = bkd.sum(costs_S)

        assert len(asketch.shape) == 2
        k2 = exploit_cost * bkd.trace(bkd.multidot([asketch.T, Sigma_S, asketch]))

        return k2, 1 / exploit_cost * bkd.ones((1,))

    def get_exploit_samples(
        self, result: Tuple, random_states: Optional[Any] = None
    ) -> Tuple[List[Array], List[int]]:
        """Get samples for exploitation phase.

        In AETCMC, all models share the same samples.

        Parameters
        ----------
        result : Tuple
            Exploration result tuple.
        random_states : Optional[Any]
            Random states for reproducibility.

        Returns
        -------
        samples_per_model : List[Array]
            Samples for each model (same samples for all).
        best_subset_HF : List[int]
            Indices of models in best subset (1-indexed for HF model).
        """
        if random_states is not None:
            rvs = partial(self._rvs, random_states=random_states)
        else:
            rvs = self._rvs

        best_subset = result[1]
        rounded_nsamples_per_subset = result[5]

        # Generate shared samples
        nsamples = int(self._bkd.to_numpy(rounded_nsamples_per_subset).sum())
        samples = rvs(nsamples)

        # All models use the same samples
        samples_per_model = [
            samples for _ in range(len(self._bkd.to_numpy(best_subset)))
        ]

        # Convert to HF-indexed subset
        best_subset_HF = [int(s) + 1 for s in self._bkd.to_numpy(best_subset)]

        return samples_per_model, best_subset_HF

    def find_exploit_mean(self, values_per_model: List[Array], result: Tuple) -> Array:
        """Compute exploitation mean estimate using MC approach.

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
        bkd = self._bkd
        beta_Sp = result[3]
        beta_best_S = beta_Sp[1:]

        # Stack values and compute mean across samples
        # values_per_model: list of (nqoi, nsamples) arrays
        values_stacked = bkd.stack(
            [bkd.mean(v, axis=1) for v in values_per_model], axis=0
        )  # (nmodels, nqoi)

        # Compute weighted sum
        product = bkd.squeeze(bkd.dot(beta_best_S.T, values_stacked[:, :1]))

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
