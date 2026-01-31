"""Optimization objectives and constraints for ACV estimators.

Provides classes for optimizing sample allocation in ACV estimators.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional
from functools import partial

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


class ACVObjective(ABC, Generic[Array]):
    """Base class for ACV optimization objectives.

    Computes the objective function value and derivatives for sample
    allocation optimization. The optimization variable is the partition
    ratios (samples in each partition relative to the first partition).

    Parameters
    ----------
    scaling : float, optional
        Scaling factor for the objective. Default: 1.0.

    Notes
    -----
    This class follows the function protocol pattern expected by
    DerivativeChecker, with methods:
    - __call__(samples) for evaluation
    - jacobian(sample) for single-sample Jacobian
    - nvars(), nqoi(), bkd()
    """

    def __init__(self, scaling: float = 1.0):
        self._scaling = scaling
        self._est: Optional["ACVEstimator"] = None  # type: ignore
        self._bkd: Optional[Backend[Array]] = None
        self._target_cost: Optional[float] = None

    def set_target_cost(self, target_cost: float) -> None:
        """Set the target cost budget."""
        self._target_cost = target_cost

    def set_estimator(self, est: "ACVEstimator") -> None:  # type: ignore
        """Set the ACV estimator to optimize.

        Parameters
        ----------
        est : ACVEstimator
            The estimator to optimize sample allocation for.
        """
        self._est = est
        self._bkd = est._bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        if self._bkd is None:
            raise ValueError("Estimator not set. Call set_estimator() first.")
        return self._bkd

    def nvars(self) -> int:
        """Return number of optimization variables (partition ratios)."""
        if self._est is None:
            raise ValueError("Estimator not set. Call set_estimator() first.")
        return self._est.npartitions() - 1

    def nqoi(self) -> int:
        """Return number of outputs (always 1 for scalar objective)."""
        return 1

    @abstractmethod
    def _optimization_criteria(self, est_covariance: Array) -> Array:
        """Compute the optimization criteria from estimator covariance.

        Parameters
        ----------
        est_covariance : Array
            Estimator covariance matrix.

        Returns
        -------
        Array
            Scalar objective value.
        """
        raise NotImplementedError

    def _objective_value(self, partition_ratios: Array) -> Array:
        """Compute objective value from partition ratios.

        Parameters
        ----------
        partition_ratios : Array
            1D array of partition ratios. Shape: (npartitions-1,)

        Returns
        -------
        Array
            Scalar objective value.
        """
        if self._est is None or self._target_cost is None:
            raise ValueError("Estimator and target cost must be set.")

        est_covariance = self._est._covariance_from_partition_ratios(
            self._target_cost, partition_ratios
        )
        return self._optimization_criteria(est_covariance) * self._scaling

    def __call__(self, samples: Array) -> Array:
        """Evaluate objective at given partition ratios.

        Parameters
        ----------
        samples : Array
            Partition ratios. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Objective values. Shape: (1, nsamples)
        """
        bkd = self.bkd()
        if samples.ndim == 1:
            samples = bkd.reshape(samples, (-1, 1))

        nsamples = samples.shape[1]
        results = []
        for i in range(nsamples):
            val = self._objective_value(samples[:, i])
            results.append(bkd.reshape(val, (1,)))

        return bkd.stack(results, axis=1)

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample using autograd.

        Parameters
        ----------
        sample : Array
            Single partition ratio sample. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (1, nvars)
        """
        bkd = self.bkd()
        # Use backend's autograd jacobian
        partition_ratios = sample[:, 0] if sample.ndim == 2 else sample
        jac = bkd.jacobian(self._objective_value, partition_ratios)
        return bkd.reshape(jac, (1, -1))


class ACVLogDeterminantObjective(ACVObjective[Array]):
    """Log-determinant objective for ACV optimization.

    Minimizes log(det(Covariance)) which is equivalent to minimizing
    the generalized variance of the estimator.
    """

    def _optimization_criteria(self, est_covariance: Array) -> Array:
        """Compute log-determinant of estimator covariance.

        Parameters
        ----------
        est_covariance : Array
            Estimator covariance matrix.

        Returns
        -------
        Array
            Log-determinant value.
        """
        bkd = self.bkd()

        # Handle scalar case (nqoi=1) - covariance is (1, 1) or scalar
        if est_covariance.ndim == 0:
            return bkd.log(bkd.abs(est_covariance) + 1e-14)

        if est_covariance.shape == (1, 1):
            cov_scalar = est_covariance[0, 0]
            return bkd.log(bkd.abs(cov_scalar) + 1e-14)

        # For matrix case, compute log-det via Cholesky
        # log(det(A)) = 2 * sum(log(diag(L))) where A = L @ L.T
        L = bkd.cholesky(est_covariance)
        diag_L = bkd.diag(L)
        return 2 * bkd.sum(bkd.log(bkd.abs(diag_L) + 1e-14))


class ACVPartitionConstraint(Generic[Array]):
    """Constraint ensuring non-negative partition samples.

    Enforces that partition samples >= min_nsamples for all partitions.

    Parameters
    ----------
    est : ACVEstimator
        The ACV estimator.
    target_cost : float
        Total computational budget.
    """

    def __init__(self, est: "ACVEstimator", target_cost: float):  # type: ignore
        self._est = est
        self._target_cost = target_cost
        self._bkd = est._bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return number of optimization variables."""
        return self._est.npartitions() - 1

    def nqoi(self) -> int:
        """Return number of constraint outputs."""
        return self._est.npartitions()

    def lb(self) -> Array:
        """Lower bound: constraint >= 0 (non-negative samples)."""
        return self._bkd.zeros((self.nqoi(),))

    def ub(self) -> Array:
        """Upper bound: no upper limit."""
        return self._bkd.full((self.nqoi(),), float("inf"))

    def _eval_constraint(self, partition_ratios: Array) -> Array:
        """Evaluate constraint values.

        Returns partition samples minus minimum, so constraint >= 0 is satisfied
        when all partitions have at least min_nsamples.

        Parameters
        ----------
        partition_ratios : Array
            1D array of partition ratios.

        Returns
        -------
        Array
            Constraint values (should be >= 0).
        """
        nsamples = self._est._npartition_samples_from_partition_ratios(
            self._target_cost, partition_ratios
        )
        min_samples = self._est._stat.min_nsamples()
        return nsamples - min_samples

    def __call__(self, samples: Array) -> Array:
        """Evaluate constraints at given partition ratios.

        Parameters
        ----------
        samples : Array
            Partition ratios. Shape: (nvars, nsamples)

        Returns
        -------
        Array
            Constraint values. Shape: (nqoi, nsamples)
        """
        bkd = self.bkd()
        if samples.ndim == 1:
            samples = bkd.reshape(samples, (-1, 1))

        nsamples = samples.shape[1]
        results = []
        for i in range(nsamples):
            val = self._eval_constraint(samples[:, i])
            results.append(val)

        return bkd.stack(results, axis=1)

    def jacobian(self, sample: Array) -> Array:
        """Compute Jacobian at a single sample using autograd.

        Parameters
        ----------
        sample : Array
            Single partition ratio sample. Shape: (nvars, 1)

        Returns
        -------
        Array
            Jacobian. Shape: (nqoi, nvars)
        """
        bkd = self.bkd()
        partition_ratios = sample[:, 0] if sample.ndim == 2 else sample
        return bkd.jacobian(self._eval_constraint, partition_ratios)
