"""Base Approximate Control Variate (ACV) estimator.

Provides the foundation for all ACV-type estimators with multiple low-fidelity
models as control variates.
"""

from typing import Generic, List, Callable, Optional, Tuple, Any
from abc import abstractmethod

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.stats.protocols import StatisticWithDiscrepancyProtocol
from pyapprox.typing.stats.estimators.base import AbstractEstimator
from pyapprox.typing.stats.estimators.bootstrap import BootstrapMixin
from pyapprox.typing.stats.allocation.matrices import (
    get_allocation_matrix_from_recursion,
    get_allocation_matrix_gmf,
    get_nsamples_per_model,
    get_npartitions_from_nmodels,
    compute_samples_per_model,
)
from pyapprox.typing.optimization.minimize.protocols import (
    BindableOptimizerProtocol,
)
from pyapprox.typing.optimization.minimize.scipy.diffevol import (
    ScipyDifferentialEvolutionOptimizer,
)
from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.typing.optimization.minimize.chained.chained_optimizer import (
    ChainedOptimizer,
)


class ACVEstimator(BootstrapMixin[Array], AbstractEstimator[Array], Generic[Array]):
    """Base Approximate Control Variate estimator.

    The ACV estimator uses multiple low-fidelity models as control variates:
        Q_ACV = Q_0 + sum_{m=1}^{M-1} eta_m * (mu_m - Q_m)

    where:
    - Q_0 is the HF sample mean
    - Q_m are LF sample means on shared samples
    - mu_m are LF sample means on all LF samples
    - eta_m are optimal weights

    The allocation of samples across models is determined by the recursion
    index, which specifies how models are coupled.

    Parameters
    ----------
    stat : StatisticWithDiscrepancyProtocol[Array]
        Statistic to estimate.
    costs : Array
        Cost per sample for each model. Shape: (nmodels,)
    bkd : Backend[Array]
        Computational backend.
    recursion_index : Array, optional
        Recursion index. Shape: (nmodels-1,)
        If None, uses MFMC structure (all coupled with HF).

    Examples
    --------
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> from pyapprox.typing.stats import MultiOutputMean
    >>> bkd = NumpyBkd()
    >>> stat = MultiOutputMean(nqoi=1, bkd=bkd)
    >>> cov = bkd.asarray([[1.0, 0.9, 0.8], [0.9, 1.0, 0.95], [0.8, 0.95, 1.0]])
    >>> stat.set_pilot_quantities(cov)
    >>> costs = bkd.asarray([10.0, 1.0, 0.1])
    >>> acv = ACVEstimator(stat, costs, bkd)
    >>> acv.allocate_samples(target_cost=100.0)
    """

    def __init__(
        self,
        stat: StatisticWithDiscrepancyProtocol[Array],
        costs: Array,
        bkd: Backend[Array],
        recursion_index: Optional[Array] = None,
    ):
        super().__init__(stat, costs, bkd)

        nmodels = costs.shape[0]
        if nmodels < 2:
            raise ValueError("ACV requires at least 2 models")

        # Set recursion index (default: all coupled to HF)
        if recursion_index is None:
            self._recursion_index = bkd.zeros((nmodels - 1,))
        else:
            if recursion_index.shape[0] != nmodels - 1:
                raise ValueError(
                    f"recursion_index must have length {nmodels-1}, got {recursion_index.shape[0]}"
                )
            self._recursion_index = recursion_index

        self._weights: Optional[Array] = None
        self._npartition_samples: Optional[Array] = None
        self._allocation_mat: Optional[Array] = None
        self._optimizer: Optional[BindableOptimizerProtocol[Array]] = None

    def recursion_index(self) -> Array:
        """Return the recursion index."""
        return self._recursion_index

    def set_recursion_index(self, index: Array) -> None:
        """Set the recursion index."""
        if index.shape[0] != self.nmodels() - 1:
            raise ValueError(
                f"recursion_index must have length {self.nmodels()-1}"
            )
        self._recursion_index = index
        # Invalidate cached values
        self._nsamples = None
        self._weights = None
        self._npartition_samples = None
        self._allocation_mat = None

    def get_allocation_matrix(self) -> Array:
        """Return the allocation matrix.

        The allocation matrix has shape (nmodels, 2*nmodels) where:
        - Rows: independent partitions k = 0, ..., M-1
        - Columns: sample sets Z₀*, Z₀, Z₁*, Z₁, ...
          - Column 2m: Z_m* (starred set for model m)
          - Column 2m+1: Z_m (unstarred set for model m)

        Subclasses should override this to use their specific allocation
        matrix function (get_allocation_matrix_gmf, get_allocation_matrix_mfmc, etc.)
        """
        if self._allocation_mat is None:
            # Default: use GMF-style allocation (hierarchical inclusion)
            self._allocation_mat = get_allocation_matrix_gmf(
                self.nmodels(), self._recursion_index, self._bkd
            )
        return self._allocation_mat

    def npartitions(self) -> int:
        """Return number of independent partitions (rows in allocation matrix)."""
        return self.nmodels()

    def nsets(self) -> int:
        """Return number of sample sets (columns in allocation matrix)."""
        return 2 * self.nmodels()

    def _compute_optimal_weights(self) -> Array:
        """Compute optimal control variate weights.

        For ACV, the optimal weights minimize estimator variance:
            eta = cf^{-1} @ CF^T

        where CF and cf are the discrepancy covariances.
        """
        bkd = self._bkd
        npart = self.npartition_samples()

        alloc_mat = self.get_allocation_matrix()
        CF, cf = self._stat.get_acv_discrepancy_covariances(alloc_mat, npart)

        nqoi = self._stat.nqoi()
        ncontrols = self.nmodels() - 1

        # Solve for weights: eta = cf^{-1} @ CF^T
        if nqoi == 1:
            # Simple case: direct solution
            cf_flat = bkd.flatten(cf)
            CF_flat = bkd.flatten(CF)
            if cf.shape == (1, 1):
                cf_val = cf[0, 0]
                # Avoid division by zero
                eta = CF_flat / bkd.maximum(bkd.abs(cf_val), bkd.asarray(1e-14))
            else:
                eta = bkd.flatten(bkd.solve(cf, bkd.reshape(CF_flat, (-1, 1))))
        else:
            # Multi-QoI: block-wise solution
            eta = bkd.solve(cf, CF.T)

        return eta

    def _compute_sample_ratios(self) -> Array:
        """Compute optimal sample ratios n_m / n_0.

        Returns array of ratios for each model.
        """
        bkd = self._bkd
        cov = self._stat.cov()
        nmodels = self.nmodels()
        nqoi = self._stat.nqoi()

        # Extract variances (diagonal elements)
        if nqoi == 1:
            variances = bkd.get_diagonal(cov)
        else:
            # Use trace-based averaging for multi-QoI
            # Build indices for each model's block diagonal
            traces = []
            for m in range(nmodels):
                block = cov[m * nqoi : (m + 1) * nqoi, m * nqoi : (m + 1) * nqoi]
                traces.append(bkd.trace(block) / nqoi)
            variances = bkd.stack(traces)

        # Extract correlations with model 0
        if nqoi == 1:
            # cov[0, m] / sqrt(cov[0,0] * cov[m,m])
            correlations = cov[0, :] / bkd.sqrt(cov[0, 0] * variances)
        else:
            # Use trace-based averaging for multi-QoI
            corrs = []
            for m in range(nmodels):
                cov_0m = cov[:nqoi, m * nqoi : (m + 1) * nqoi]
                corr = bkd.trace(cov_0m) / nqoi / bkd.sqrt(variances[0] * variances[m])
                corrs.append(corr)
            correlations = bkd.stack(corrs)

        # Compute sample ratios using optimization
        c0 = self._costs[0]
        rho_sq = correlations ** 2

        # Optimal ratio: r_m = sqrt((c0/c_m) * rho_m^2 / (1 - rho_m^2))
        # For m=0, ratio is always 1
        # For valid correlations 0 < rho^2 < 1
        ratios = bkd.ones((nmodels,))

        # Vectorized computation for m >= 1
        cm = self._costs[1:]
        rho_sq_m = rho_sq[1:]

        # Compute ratios where valid (0 < rho^2 < 1)
        valid_mask = (rho_sq_m > 0) & (rho_sq_m < 1)
        denom = bkd.where(valid_mask, 1 - rho_sq_m, bkd.ones_like(rho_sq_m))
        r = bkd.sqrt((c0 / cm) * rho_sq_m / denom)
        r = bkd.maximum(r, bkd.ones_like(r))

        # Apply mask: use r where valid, else 1.0
        r = bkd.where(valid_mask, r, bkd.ones_like(r))

        # Combine with first element
        ratios = bkd.concatenate([bkd.asarray([1.0]), r])

        return ratios

    def allocate_samples(self, target_cost: float) -> None:
        """Allocate samples optimally across models.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        """
        bkd = self._bkd

        # Get optimal sample ratios
        ratios = self._compute_sample_ratios()

        # Compute HF samples from budget
        # Total cost = sum_m (n_m * c_m) = n_0 * sum_m (r_m * c_m)
        weighted_cost = bkd.sum(ratios * self._costs)
        n0 = target_cost / weighted_cost
        n0 = bkd.maximum(n0, bkd.asarray(float(self._stat.min_nsamples())))

        # Compute samples per model
        nsamples = bkd.round(ratios * n0)
        min_samples = bkd.asarray(float(self._stat.min_nsamples()))
        nsamples = bkd.maximum(nsamples, min_samples * bkd.ones_like(nsamples))

        self._nsamples = nsamples

        # Compute partition samples from allocation matrix
        alloc_mat = self.get_allocation_matrix()

        # Solve for partition samples: A @ npart = nsamples
        npart = self._solve_partition_allocation(alloc_mat, nsamples)
        self._npartition_samples = npart

        # Compute optimal weights
        self._weights = self._compute_optimal_weights()

    def set_optimizer(
        self, optimizer: BindableOptimizerProtocol[Array]
    ) -> None:
        """Set the optimizer for sample allocation optimization.

        Parameters
        ----------
        optimizer : BindableOptimizerProtocol[Array]
            An optimizer configured with options but NOT bound to an objective.
            During allocate_samples_numerical(), the optimizer will be cloned
            and bound to the log-determinant objective and partition constraints.

        Examples
        --------
        >>> from pyapprox.typing.optimization.minimize.scipy.trust_constr import (
        ...     ScipyTrustConstrOptimizer
        ... )
        >>> optimizer = ScipyTrustConstrOptimizer(maxiter=500, gtol=1e-8)
        >>> est.set_optimizer(optimizer)
        >>> est.allocate_samples_numerical(target_cost=100.0)

        >>> # Or use ChainedOptimizer for global + local search
        >>> from pyapprox.typing.optimization.minimize.chained import ChainedOptimizer
        >>> global_opt = ScipyTrustConstrOptimizer(maxiter=50, gtol=1e-4)
        >>> local_opt = ScipyTrustConstrOptimizer(maxiter=500, gtol=1e-8)
        >>> chained = ChainedOptimizer(global_opt, local_opt)
        >>> est.set_optimizer(chained)
        >>> est.allocate_samples_numerical(target_cost=100.0)
        """
        if not isinstance(optimizer, BindableOptimizerProtocol):
            raise TypeError(
                f"optimizer must satisfy BindableOptimizerProtocol, "
                f"got {type(optimizer).__name__}"
            )
        self._optimizer = optimizer

    def _get_default_optimizer(self) -> ChainedOptimizer[Array]:
        """Create default optimizer for sample allocation.

        Uses a chained optimizer: ScipyDifferentialEvolutionOptimizer for
        global search followed by ScipyTrustConstrOptimizer for local
        refinement. This provides robust convergence for the non-convex
        ACV sample allocation problem.
        """
        global_opt = ScipyDifferentialEvolutionOptimizer(
            maxiter=3,
            raise_on_failure=False,  # Just need a good starting point
        )
        local_opt = ScipyTrustConstrOptimizer(
            maxiter=500,
            gtol=1e-8,
            xtol=1e-10,
        )
        return ChainedOptimizer(global_opt, local_opt)

    def allocate_samples_numerical(
        self,
        target_cost: float,
        init_ratios: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        """Allocate samples using numerical optimization.

        This method uses constrained optimization to find the partition
        ratios that minimize the estimator variance for a given budget.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        init_ratios : Array, optional
            Initial partition ratios. If None, uses unit ratios.

        Returns
        -------
        partition_ratios : Array
            Optimal partition ratios. Shape: (npartitions-1,)
        log_variance : Array
            Log of estimator variance.

        Raises
        ------
        RuntimeError
            If optimization fails to converge or produces invalid results
            (NaN/Inf in objective, negative partition samples).

        Notes
        -----
        Use set_optimizer() before calling this method to use a custom
        optimizer. If no optimizer is set, uses ScipyTrustConstrOptimizer
        with default settings.

        Examples
        --------
        >>> # Default optimization
        >>> ratios, log_var = est.allocate_samples_numerical(target_cost=100.0)

        >>> # Custom optimizer
        >>> est.set_optimizer(ScipyTrustConstrOptimizer(maxiter=1000))
        >>> ratios, log_var = est.allocate_samples_numerical(target_cost=100.0)
        """
        from .optimization import ACVLogDeterminantObjective, ACVPartitionConstraint

        bkd = self._bkd

        # Get or create optimizer
        if self._optimizer is not None:
            optimizer = self._optimizer.copy()
        else:
            optimizer = self._get_default_optimizer()

        # Create objective
        objective = ACVLogDeterminantObjective()
        objective.set_target_cost(target_cost)
        objective.set_estimator(self)

        # Create constraint
        constraint = ACVPartitionConstraint(self, target_cost)

        # Get bounds for partition ratios
        nvars = self.npartitions() - 1
        lb = bkd.full((nvars,), 0.01)
        ub = bkd.full((nvars,), 1000.0)
        bounds = bkd.stack([lb, ub], axis=1)  # Shape: (nvars, 2)

        # Initial guess
        if init_ratios is not None:
            if init_ratios.shape[0] != nvars:
                raise ValueError(
                    f"init_ratios must have shape ({nvars},), "
                    f"got {init_ratios.shape}"
                )
            init_guess = bkd.reshape(init_ratios, (nvars, 1))
        else:
            init_guess = bkd.ones((nvars, 1))

        # Bind and optimize
        optimizer.bind(objective, bounds, [constraint])
        result = optimizer.minimize(init_guess)

        # Extract results
        partition_ratios = result.optima()[:, 0]  # Shape: (nvars,)
        log_variance = bkd.asarray(result.fun())  # Convert float to array

        # Validate optimization result
        if not result.success():
            raise RuntimeError(
                f"Sample allocation optimization failed: {result.message()}"
            )

        # Check for NaN/Inf in objective
        log_var_arr = bkd.asarray(log_variance) if not hasattr(log_variance, 'shape') else log_variance
        if bkd.any_bool(bkd.isnan(log_var_arr)) or bkd.any_bool(bkd.isinf(log_var_arr)):
            raise RuntimeError(
                "Optimization produced NaN/Inf objective value. "
                "Try different initial ratios or optimizer settings."
            )

        # Update estimator state
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )

        # Check for negative partition samples (constraint violation)
        if bkd.any_bool(npartition_samples < 0):
            raise RuntimeError(
                "Optimization produced negative partition samples. "
                "Constraint violation detected. Try different initial ratios."
            )

        self._npartition_samples = npartition_samples

        # Compute model samples from partition samples using correct formula
        alloc_mat = self.get_allocation_matrix()
        self._nsamples = compute_samples_per_model(
            npartition_samples, alloc_mat, self._bkd
        )

        # Compute optimal weights
        self._weights = self._compute_optimal_weights()

        return partition_ratios, log_variance

    def _solve_partition_allocation(
        self, alloc_mat: Array, nsamples: Array
    ) -> Array:
        """Solve for partition samples from model samples.

        Computes partition samples (shape: (nmodels,)) from model samples.
        This is estimator-specific - subclasses may override.

        Parameters
        ----------
        alloc_mat : Array
            Allocation matrix. Shape: (nmodels, 2*nmodels)
        nsamples : Array
            Samples per model. Shape: (nmodels,)

        Returns
        -------
        npartition_samples : Array
            Samples per partition. Shape: (nmodels,)
        """
        bkd = self._bkd
        nmodels = alloc_mat.shape[0]

        # Simple approach: partition k gets samples from model differences
        # Partition 0 = HF samples
        # Partition k (k > 0) = increment for model k
        parts = [bkd.reshape(nsamples[0], (1,))]

        for k in range(1, nmodels):
            # Partition k: samples for model k beyond model k-1
            diff = bkd.maximum(nsamples[k] - nsamples[k - 1], bkd.asarray(0.0))
            parts.append(bkd.reshape(diff, (1,)))

        return bkd.concatenate(parts)

    def npartition_samples(self) -> Array:
        """Return samples per partition."""
        if self._npartition_samples is None:
            raise ValueError("Samples not allocated.")
        return self._npartition_samples

    def weights(self) -> Array:
        """Return optimal control variate weights."""
        if self._weights is None:
            raise ValueError("Weights not computed.")
        return self._weights

    def _partition_ratios_to_model_ratios(self, partition_ratios: Array) -> Array:
        """Convert partition ratios to model sample ratios.

        Computes model_ratios[m] = n_{m+1} / n_0 from partition_ratios.

        Parameters
        ----------
        partition_ratios : Array
            Shape: (nmodels-1,)

        Returns
        -------
        model_ratios : Array
            Shape: (nmodels-1,) where model_ratios[m] = n_{m+1} / n_0
        """
        bkd = self._bkd
        alloc_mat = self.get_allocation_matrix()
        nmodels = self.nmodels()

        model_ratios_list = []
        for ii in range(1, nmodels):
            # Find active partitions for model ii (excluding partition 0)
            # Model ii is active in partition k if A[k, 2*ii] or A[k, 2*ii+1] == 1
            col_starred = 2 * ii
            col_unstarred = 2 * ii + 1
            active_in_partitions = (
                (alloc_mat[1:, col_starred] == 1) |
                (alloc_mat[1:, col_unstarred] == 1)
            )
            # Sum of partition_ratios where model is active (partitions 1 to M-1)
            ratio = bkd.sum(bkd.where(
                active_in_partitions,
                partition_ratios,
                bkd.zeros_like(partition_ratios)
            ))

            # Add 1 if model uses partition 0 (the HF partition)
            if (alloc_mat[0, col_starred] == 1) or (alloc_mat[0, col_unstarred] == 1):
                ratio = ratio + 1

            model_ratios_list.append(bkd.reshape(ratio, (1,)))

        return bkd.concatenate(model_ratios_list)

    def _get_nhf_samples_from_partition_ratios(
        self, target_cost: float, partition_ratios: Array
    ) -> Array:
        """Compute number of HF samples from partition ratios.

        Uses model ratios for cost computation:
        target_cost = n_0 * (c_0 + sum_m model_ratios[m] * c_{m+1})

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        partition_ratios : Array
            Partition sample ratios relative to first partition. Shape: (nmodels-1,)

        Returns
        -------
        Array
            Number of HF samples (scalar).
        """
        bkd = self._bkd
        model_ratios = self._partition_ratios_to_model_ratios(partition_ratios)

        # Cost formula: target_cost = n_0 * (c_0 + sum_m model_ratios[m] * c_{m+1})
        weighted_cost = self._costs[0] + bkd.sum(model_ratios * self._costs[1:])
        nhf = target_cost / weighted_cost
        return nhf

    def _npartition_samples_from_partition_ratios(
        self, target_cost: float, partition_ratios: Array
    ) -> Array:
        """Compute partition samples from partition ratios.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        partition_ratios : Array
            Partition sample ratios relative to first partition. Shape: (npartitions-1,)

        Returns
        -------
        Array
            Samples per partition. Shape: (npartitions,)
        """
        bkd = self._bkd
        nhf = self._get_nhf_samples_from_partition_ratios(target_cost, partition_ratios)
        npartition_samples = bkd.concatenate([
            bkd.reshape(nhf, (1,)),
            partition_ratios * nhf
        ])
        return npartition_samples

    def _covariance_from_npartition_samples(
        self, npartition_samples: Array
    ) -> Array:
        """Compute estimator covariance from partition samples.

        Parameters
        ----------
        npartition_samples : Array
            Samples per partition. Shape: (npartitions,)

        Returns
        -------
        Array
            Estimator covariance matrix.
        """
        bkd = self._bkd
        alloc_mat = self.get_allocation_matrix()

        # Get discrepancy covariances
        CF, cf = self._stat.get_acv_discrepancy_covariances(alloc_mat, npartition_samples)
        nqoi = self._stat.nqoi()

        # Compute weights from covariances: eta = cf^{-1} @ CF^T
        if nqoi == 1:
            # Scalar case
            cf_flat = bkd.flatten(cf)
            CF_flat = bkd.flatten(CF)
            # Check if cf is essentially a scalar
            if cf.shape == (1, 1):
                cf_val = cf[0, 0]
                # Avoid division by zero
                eta = CF_flat / bkd.maximum(bkd.abs(cf_val), bkd.asarray(1e-14))
            else:
                # Solve linear system
                eta = bkd.flatten(bkd.solve(cf, bkd.reshape(CF_flat, (-1, 1))))
        else:
            # Multi-QoI: use least squares
            eta_sol, _, _, _ = bkd.lstsq(cf, CF.T)
            eta = eta_sol.T

        # Compute HF covariance
        nhf = npartition_samples[0]
        hf_cov = self._stat.high_fidelity_estimator_covariance(nhf)

        # Compute variance reduction
        if nqoi == 1:
            # V(Q_ACV) = V(Q_0) - eta^T @ cf @ eta
            eta_row = bkd.reshape(eta, (1, -1))
            var_red = eta_row @ cf @ eta_row.T
            return hf_cov - bkd.reshape(var_red, hf_cov.shape)
        else:
            return hf_cov

    def _covariance_from_partition_ratios(
        self, target_cost: float, partition_ratios: Array
    ) -> Array:
        """Compute estimator covariance from partition ratios.

        Parameters
        ----------
        target_cost : float
            Total computational budget.
        partition_ratios : Array
            Partition sample ratios relative to first partition.

        Returns
        -------
        Array
            Estimator covariance matrix.
        """
        npartition_samples = self._npartition_samples_from_partition_ratios(
            target_cost, partition_ratios
        )
        return self._covariance_from_npartition_samples(npartition_samples)

    def generate_samples_per_model(
        self, rvs: Callable[[int], Array], npilot_samples: int = 0
    ) -> List[Array]:
        """Generate samples for each model.

        The sample structure follows the allocation matrix:
        - Model 0 (HF) uses samples from partitions where row 0 has 1
        - Model m uses samples from partitions where row m has 1

        Parameters
        ----------
        rvs : Callable[[int], Array]
            Random variable sampler.
        npilot_samples : int, optional
            Number of pilot samples to skip from the first partition.
            These samples are assumed to have been generated previously
            for pilot covariance estimation.

        Returns
        -------
        List[Array]
            Samples for each model.
        """
        bkd = self._bkd
        npart = bkd.copy(self.npartition_samples())
        npartitions = len(npart)

        # Adjust first partition for pilot samples
        if npilot_samples > 0:
            if npilot_samples > npart[0].item():
                raise ValueError(
                    f"npilot_samples ({npilot_samples}) exceeds first partition "
                    f"size ({npart[0].item()})"
                )
            npart = bkd.concatenate([
                bkd.asarray([npart[0] - npilot_samples]),
                npart[1:]
            ])

        nmodels = self.nmodels()
        alloc_mat = self.get_allocation_matrix()

        # Generate all unique samples needed
        total_samples = int(bkd.sum(npart).item())
        all_samples = rvs(total_samples) if total_samples > 0 else rvs(1)[:, :0]

        # Assign samples to partitions
        partition_samples: List[Array] = []
        start = 0
        for p in range(npartitions):
            count = int(npart[p].item())
            end = start + count
            partition_samples.append(all_samples[:, start:end])
            start = end

        # Collect samples for each model
        model_samples: List[Array] = []
        for m in range(nmodels):
            model_samps: List[Array] = []
            for p in range(npartitions):
                count = int(npart[p].item())
                if alloc_mat[m, p].item() == 1 and count > 0:
                    model_samps.append(partition_samples[p])

            if model_samps:
                model_samples.append(bkd.concatenate(model_samps, axis=1))
            else:
                # Empty array with correct shape
                model_samples.append(all_samples[:, :0])

        return model_samples

    def insert_pilot_values(
        self, pilot_values: List[Array], values_per_model: List[Array]
    ) -> List[Array]:
        """Insert pilot sample values at the beginning of model values.

        Pilot samples are added to models that use the first partition
        (partition 0), which typically includes the high-fidelity model.

        Parameters
        ----------
        pilot_values : List[Array]
            Pilot sample values for each model. pilot_values[m] has shape
            (npilot, nqoi).
        values_per_model : List[Array]
            Model values without pilot samples.

        Returns
        -------
        List[Array]
            Model values with pilot samples prepended where appropriate.
        """
        bkd = self._bkd
        alloc_mat = self.get_allocation_matrix()
        nmodels = self.nmodels()

        new_values_per_model: List[Array] = []
        for m in range(nmodels):
            # Check if this model uses the first partition
            uses_first_partition = alloc_mat[m, 0].item() == 1

            if uses_first_partition:
                # Prepend pilot values
                new_values_per_model.append(
                    bkd.vstack([pilot_values[m], values_per_model[m]])
                )
            else:
                # Copy without modification
                new_values_per_model.append(bkd.copy(values_per_model[m]))

        return new_values_per_model

    def __call__(self, values: List[Array]) -> Array:
        """Compute ACV estimate.

        Q_ACV = Q_0 + sum_m eta_m * (mu_m - Q_m)

        Parameters
        ----------
        values : List[Array]
            Model outputs. values[m] has shape (nsamples_m, nqoi)

        Returns
        -------
        Array
            Estimated statistic. Shape: (nstats,)
        """
        if len(values) != self.nmodels():
            raise ValueError(
                f"Expected {self.nmodels()} model outputs, got {len(values)}"
            )

        bkd = self._bkd
        nmodels = self.nmodels()
        nsamples = self.nsamples_per_model()
        weights = self.weights()

        nhf = int(nsamples[0].item())

        # Q_0: HF mean
        Q0 = bkd.sum(values[0], axis=0) / nhf

        # Control variate correction
        correction = bkd.zeros((self._stat.nstats(),))

        for m in range(1, nmodels):
            nlf = int(nsamples[m].item())
            n_shared = min(nhf, nlf)

            # Q_m: LF mean on shared samples
            Q_m = bkd.sum(values[m][:n_shared], axis=0) / n_shared

            # mu_m: LF mean on all samples
            mu_m = bkd.sum(values[m], axis=0) / nlf

            # Weight for this control
            eta_m = weights[m - 1] if weights.ndim == 1 else weights[m - 1, :]

            correction = correction + eta_m * (mu_m - Q_m)

        return Q0 + correction

    def optimized_covariance(self) -> Array:
        """Return covariance of the ACV estimator.

        Returns
        -------
        Array
            Covariance matrix. Shape: (nstats, nstats)
        """
        bkd = self._bkd
        nsamples = self.nsamples_per_model()
        nhf = nsamples[0]

        # HF covariance
        hf_cov = self._stat.high_fidelity_estimator_covariance(nhf)

        # Compute variance reduction from control variates
        cov = self._stat.cov()
        nqoi = self._stat.nqoi()

        if nqoi == 1:
            # Compute total correlation
            total_rho_sq = bkd.asarray(0.0)
            for m in range(1, self.nmodels()):
                rho = cov[0, m] / bkd.sqrt(cov[0, 0] * cov[m, m])
                total_rho_sq = total_rho_sq + rho ** 2

            # Variance reduction factor (approximate)
            var_red = bkd.maximum(1 - total_rho_sq, bkd.asarray(0.01))
            return hf_cov * var_red
        else:
            return hf_cov

    def variance_reduction(self) -> float:
        """Return variance reduction factor compared to MC.

        Returns
        -------
        float
            Ratio Var(Q_ACV) / Var(Q_MC).
        """
        bkd = self._bkd
        cov = self._stat.cov()
        nqoi = self._stat.nqoi()

        if nqoi == 1:
            total_rho_sq = bkd.asarray(0.0)
            for m in range(1, self.nmodels()):
                rho = cov[0, m] / bkd.sqrt(cov[0, 0] * cov[m, m])
                total_rho_sq = total_rho_sq + rho ** 2
            result = bkd.maximum(1 - total_rho_sq, bkd.asarray(0.01))
            return float(bkd.to_numpy(result))
        else:
            return 0.5

    def _estimate_with_weights(
        self, values_per_model: List[Array], weights: Any
    ) -> Array:
        """Compute ACV estimate using specified weights.

        Parameters
        ----------
        values_per_model : List[Array]
            Model values.
        weights : Any
            Control variate weights. If None, uses stored weights.

        Returns
        -------
        Array
            Estimated statistic.
        """
        if weights is None:
            return self(values_per_model)

        if len(values_per_model) != self.nmodels():
            raise ValueError(
                f"Expected {self.nmodels()} model outputs, got {len(values_per_model)}"
            )

        bkd = self._bkd
        nmodels = self.nmodels()
        nsamples = self.nsamples_per_model()

        nhf = int(nsamples[0].item())

        # Q_0: HF mean
        Q0 = bkd.sum(values_per_model[0], axis=0) / nhf

        # Control variate correction
        correction = bkd.zeros((self._stat.nstats(),))

        for m in range(1, nmodels):
            nlf = int(nsamples[m].item())
            n_shared = min(nhf, nlf)

            # Q_m: LF mean on shared samples
            Q_m = bkd.sum(values_per_model[m][:n_shared], axis=0) / n_shared

            # mu_m: LF mean on all samples
            mu_m = bkd.sum(values_per_model[m], axis=0) / nlf

            # Weight for this control
            eta_m = weights[m - 1] if weights.ndim == 1 else weights[m - 1, :]

            correction = correction + eta_m * (mu_m - Q_m)

        return Q0 + correction

    def __repr__(self) -> str:
        nsamples_str = "not allocated"
        if self._nsamples is not None:
            ns = self._bkd.to_numpy(self._nsamples)
            nsamples_str = str(list(ns.astype(int)))
        return f"ACVEstimator(nmodels={self.nmodels()}, nsamples={nsamples_str})"
