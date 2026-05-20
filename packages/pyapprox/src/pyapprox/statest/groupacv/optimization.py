"""Optimization infrastructure for GroupACV estimators.

This module provides objective functions and constraints for
GroupACV sample allocation optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Generic, List, Optional, Tuple

import numpy as np

from pyapprox.statest.groupacv.utils import _grouped_acv_sigma
from pyapprox.util.backends.protocols import Array, Backend

if TYPE_CHECKING:
    from pyapprox.statest.groupacv.base import BaseGroupACVEstimator


class GroupACVObjective(ABC, Generic[Array]):
    """Abstract base class for GroupACV optimization objectives.

    Satisfies ObjectiveProtocol for use with ScipyTrustConstrOptimizer.
    """

    def __init__(self, bkd: Optional[Backend[Array]] = None):
        self._bkd: Optional[Backend[Array]] = bkd
        self._est: Optional[BaseGroupACVEstimator[Array]] = None

    def _ensure_bound(
        self,
    ) -> Tuple[Backend[Array], BaseGroupACVEstimator[Array]]:
        if self._bkd is None or self._est is None:
            raise RuntimeError("Call set_estimator() before using objective")
        return self._bkd, self._est

    def bkd(self) -> Backend[Array]:
        bkd, _ = self._ensure_bound()
        return bkd

    def set_estimator(self, estimator: BaseGroupACVEstimator[Array]) -> None:
        """Set the estimator and update backend."""
        self._est = estimator
        self._bkd = self._est._bkd

    def nvars(self) -> int:
        """Number of optimization variables (npartitions)."""
        _, est = self._ensure_bound()
        return int(est.npartitions())

    def nqoi(self) -> int:
        """Number of quantities of interest (always 1 for scalar objective)."""
        return 1

    @abstractmethod
    def _objective_wrapper(self, npartition_samples_1d: Array) -> Array:
        """
        Compute the objective value.

        Parameters
        ----------
        npartition_samples_1d : Array
            1D array of partition sample counts

        Returns
        -------
        Array
            Scalar objective value as shape (1, 1)
        """
        raise NotImplementedError

    def _objective_value(self, npartition_samples: Array) -> Array:
        """Wrapper for objective computation from 2D input."""
        return self._objective_wrapper(npartition_samples[:, 0])

    def __call__(self, npartition_samples: Array) -> Array:
        """
        Evaluate the objective.

        Parameters
        ----------
        npartition_samples : Array (nvars, 1)
            Partition sample counts as column vector

        Returns
        -------
        Array (1, 1)
            Objective value
        """
        return self._objective_value(npartition_samples)

    def _scalar_objective_wrapper(self, npartition_samples_1d: Array) -> Array:
        """Wrapper that returns scalar for bkd.jacobian compatibility."""
        bkd, _ = self._ensure_bound()
        result = self._objective_wrapper(npartition_samples_1d)
        return bkd.flatten(result)[0]

    def jacobian(self, npartition_samples: Array) -> Array:
        """
        Compute the Jacobian of the objective.

        Parameters
        ----------
        npartition_samples : Array (nvars, 1)
            Partition sample counts as column vector

        Returns
        -------
        Array (1, nvars)
            Jacobian row vector
        """
        bkd, _ = self._ensure_bound()
        # bkd.jacobian is only on TorchBkd (autograd); not on Backend protocol
        bkd_jacobian: Optional[
            Callable[[Callable[[Array], Array], Array], Array]
        ] = getattr(bkd, "jacobian", None)
        if bkd_jacobian is None:
            raise NotImplementedError(
                "AD jacobian requires TorchBkd; override jacobian() "
                "for other backends"
            )
        jac = bkd_jacobian(
            self._scalar_objective_wrapper, npartition_samples[:, 0]
        )
        return jac[None, ...]


class GroupACVTraceObjective(GroupACVObjective[Array]):
    """Trace objective for GroupACV optimization.

    Minimizes the trace of the estimator covariance matrix.
    """

    def _objective_wrapper(self, npartition_samples_1d: Array) -> Array:
        bkd, est = self._ensure_bound()
        trace = bkd.trace(
            est._covariance_from_npartition_samples(npartition_samples_1d)
        )
        # conversion below is necessary for torch
        return bkd.hstack((trace,))[:, None]


class GroupACVLogDetObjective(GroupACVObjective[Array]):
    """Log-determinant objective for GroupACV optimization.

    Minimizes the log-determinant of the estimator covariance matrix.
    """

    def _objective_wrapper(self, npartition_samples_1d: Array) -> Array:
        bkd, est = self._ensure_bound()
        cov = est._covariance_from_npartition_samples(npartition_samples_1d)
        sign, logdet = bkd.slogdet(cov)
        if logdet < -1e16:
            # when cov is singular logdet returns np.inf
            # make sure to return positive value to indicate
            # to minimizer this is a bad point. Only really is
            # an issue if starting from poor initial guess or using
            # global optimizer
            return bkd.asarray([[np.inf]])
        # conversion below is necessary for torch
        return bkd.hstack((logdet,))[:, None]


class MLBLUEObjective(GroupACVTraceObjective[Array]):
    """MLBLUE-specific trace objective with analytical derivatives.

    Provides analytical Jacobian and Hessian for MLBLUE optimization.
    """

    def jacobian(self, npartition_samples: Array) -> Array:
        """
        Compute analytical Jacobian for MLBLUE.

        Uses the derivative of inverse matrix:
        d_m X^{-1} = X^{-1} (d_mX) X^{-1}
        where X = psi_matrix and d_mX is RC_mR.T (not multiplied by nsamples)

        Objective is e^T X e so
        grad is e^T X^{-1} d_mX X^{-1} e = gamma^T(d_mX)gamma
        """
        bkd, est = self._ensure_bound()
        # compute sigma blocks with npartition_samples = 1
        Sigma_blocks = _grouped_acv_sigma(
            est.nmodels(),
            bkd.eye(npartition_samples.shape[0]),
            est._subsets,
            est._stat,
        )
        # compute psi matrix with partition sizes
        psi_matrix = est._psi_matrix(npartition_samples[:, 0])
        psi_inv = est._inv(psi_matrix)
        Rmats = est._restriction_matrices
        jacobian: Array = bkd.zeros((1, npartition_samples.shape[0]))
        for kk in range(est._stat.nstats()):
            gamma = psi_inv @ est._asketch[kk : kk + 1].T
            jacobian = jacobian + bkd.hstack(
                [
                    bkd.multidot(
                        [
                            -gamma.T,
                            Rmats[ii],
                            est._inv(Sigma_blocks[ii][ii]),
                            Rmats[ii].T,
                            gamma,
                        ]
                    )
                    for ii in range(len(Sigma_blocks))
                ]
            )
        return jacobian

    def hessian(self, npartition_samples: Array) -> Array:
        """
        Compute analytical Hessian for MLBLUE.

        Uses the derivative of inverse matrix twice:
        d_mn X^{-1} = d_n(X^{-1} (d_mX) X^{-1})
        = X^{-1}(d_nX)X^{-1}d_mX^{-1} + X^{-1}(d_mX)X^{-1}d_nX^{-1}

        Hessian is gamma^T(d_nX)xi + xi^Td_nX^{-1}gamma
        = eta^T + eta, where eta = xi^Td_nX^{-1}gamma
        """
        bkd, est = self._ensure_bound()
        Sigma_blocks = _grouped_acv_sigma(
            est.nmodels(),
            bkd.eye(npartition_samples.shape[0]),
            est._subsets,
            est._stat,
        )
        psi_matrix = est._psi_matrix(npartition_samples[:, 0])
        psi_inv = est._inv(psi_matrix)
        Rmats = est._restriction_matrices
        nblocks = len(Sigma_blocks)
        hess: List[List[Array]] = [
            [bkd.zeros((1, 1)) for _jj in range(nblocks)]
            for _ii in range(nblocks)
        ]
        sigma_invs = [
            est._inv(Sigma_blocks[ii][ii]) for ii in range(nblocks)
        ]
        psi_derivs = [
            bkd.multidot([Rmats[ii], sigma_invs[ii], Rmats[ii].T])
            for ii in range(nblocks)
        ]
        for kk in range(est._stat.nstats()):
            gamma = psi_inv @ est._asketch[kk : kk + 1].T
            for ii in range(nblocks):
                xi = bkd.multidot([psi_inv, psi_derivs[ii], gamma])
                for jj in range(ii, nblocks):
                    eta = bkd.multidot([xi.T, psi_derivs[jj], gamma])
                    hess[ii][jj] = hess[ii][jj] + eta.T + eta
                    hess[jj][ii] = hess[ii][jj]
        return bkd.vstack([bkd.hstack(row) for row in hess])

    def hvp(self, npartition_samples: Array, vec: Array) -> Array:
        """Compute Hessian-vector product.

        Parameters
        ----------
        npartition_samples : Array (nvars, 1)
            Partition sample counts as column vector
        vec : Array (nvars, 1)
            Vector to multiply with Hessian

        Returns
        -------
        Array (nvars, 1)
            Hessian-vector product
        """
        hess = self.hessian(npartition_samples)
        return hess @ vec


class GroupACVCostConstraint(Generic[Array]):
    """Cost and minimum HF samples constraint for GroupACV optimization.

    Enforces:
    1. Total cost <= target_cost
    2. Number of HF samples >= min_nhf_samples

    Satisfies NonlinearConstraintProtocol for use with ScipyTrustConstrOptimizer.
    """

    def __init__(self, bkd: Optional[Backend[Array]] = None):
        """
        Initialize the constraint.

        Parameters
        ----------
        bkd : Backend[Array], optional
            Backend for array operations. Set via set_estimator() if not provided.
        """
        self._bkd: Optional[Backend[Array]] = bkd
        self._est: Optional[BaseGroupACVEstimator[Array]] = None
        self._target_cost: Optional[float] = None
        self._min_nhf_samples: Optional[int] = None
        self._lb: Optional[Array] = None
        self._ub: Optional[Array] = None

    def _ensure_bound(
        self,
    ) -> Tuple[Backend[Array], BaseGroupACVEstimator[Array]]:
        if self._bkd is None or self._est is None:
            raise RuntimeError("Call set_estimator() before using constraint")
        return self._bkd, self._est

    def _ensure_budget(self) -> Tuple[float, int]:
        if self._target_cost is None or self._min_nhf_samples is None:
            raise RuntimeError("Call set_budget() before using constraint")
        return self._target_cost, self._min_nhf_samples

    def bkd(self) -> Backend[Array]:
        bkd, _ = self._ensure_bound()
        return bkd

    def set_estimator(self, estimator: BaseGroupACVEstimator[Array]) -> None:
        """Set the estimator and update backend."""
        self._est = estimator
        self._bkd = self._est._bkd
        # Bounds: both constraints must be >= 0
        self._lb = self._bkd.zeros((self.nqoi(),))
        self._ub = self._bkd.full((self.nqoi(),), np.inf)

    def set_budget(self, target_cost: float, min_nhf_samples: int) -> None:
        """
        Set the budget constraints.

        Parameters
        ----------
        target_cost : float
            Maximum total computational cost

        min_nhf_samples : int
            Minimum number of high-fidelity samples
        """
        self._target_cost = target_cost
        self._min_nhf_samples = min_nhf_samples
        self._validate_target_cost_min_nhf_samples()

    def _validate_target_cost_min_nhf_samples(self) -> None:
        """Validate that target_cost is sufficient for min_nhf_samples."""
        _, est = self._ensure_bound()
        target_cost, min_nhf_samples = self._ensure_budget()
        lb = min_nhf_samples * est._costs[0]
        ub = target_cost
        if ub < lb:
            msg = "target_cost {0} & cost of min_nhf_samples {1} ".format(
                target_cost, lb
            )
            msg += "are inconsistent"
            raise ValueError(msg)

    def nvars(self) -> int:
        """Number of optimization variables."""
        _, est = self._ensure_bound()
        return int(est.npartitions())

    def nqoi(self) -> int:
        """Number of constraints (cost + min HF samples)."""
        return 2

    def lb(self) -> Array:
        """Lower bounds for constraints."""
        if self._lb is None:
            raise RuntimeError("Call set_estimator() before accessing bounds")
        return self._lb

    def ub(self) -> Array:
        """Upper bounds for constraints."""
        if self._ub is None:
            raise RuntimeError("Call set_estimator() before accessing bounds")
        return self._ub

    def _eval_constraint(self, npartition_samples_1d: Array) -> Array:
        """Evaluate constraint values from 1D input."""
        bkd, est = self._ensure_bound()
        target_cost, min_nhf_samples = self._ensure_budget()
        return bkd.array(
            [
                target_cost - est._estimator_cost(npartition_samples_1d),
                bkd.sum(
                    est._partitions_per_model[0] * npartition_samples_1d
                )
                - min_nhf_samples,
            ]
        )

    def __call__(self, npartition_samples: Array) -> Array:
        """
        Evaluate constraints.

        Parameters
        ----------
        npartition_samples : Array (nvars, 1)
            Partition sample counts as column vector

        Returns
        -------
        Array (nqoi, 1)
            Constraint values (should be >= 0 for feasibility)
        """
        return self._eval_constraint(npartition_samples[:, 0])[:, None]

    def jacobian(self, npartition_samples: Array) -> Array:
        """
        Compute the Jacobian of the constraints.

        Parameters
        ----------
        npartition_samples : Array (nvars, 1)
            Partition sample counts as column vector

        Returns
        -------
        Array (nqoi, nvars)
            Jacobian matrix
        """
        bkd, est = self._ensure_bound()
        return bkd.vstack(
            (
                -(est._costs[None, :] @ est._partitions_per_model),
                est._partitions_per_model[0][None, :],
            )
        )

    def hessian(self, npartition_samples: Array) -> Array:
        """
        Compute the Hessian of the constraints.

        Returns zero matrix since constraints are linear.

        Parameters
        ----------
        npartition_samples : Array (nvars, 1)
            Partition sample counts as column vector

        Returns
        -------
        Array (nqoi, nvars, nvars)
            Hessian tensor (all zeros)
        """
        bkd, _ = self._ensure_bound()
        return bkd.zeros(
            (
                self.nqoi(),
                npartition_samples.shape[0],
                npartition_samples.shape[0],
            )
        )

    def whvp(self, npartition_samples: Array, vec: Array, weights: Array) -> Array:
        """
        Compute weighted Hessian-vector product.

        Returns zeros since constraints are linear (Hessian is zero).

        Parameters
        ----------
        npartition_samples : Array (nvars, 1)
            Partition sample counts as column vector
        vec : Array (nvars, 1)
            Vector to multiply with Hessian
        weights : Array (nqoi, 1)
            Weights for each constraint

        Returns
        -------
        Array (nvars, 1)
            Weighted Hessian-vector product (all zeros)
        """
        bkd, _ = self._ensure_bound()
        return bkd.zeros((npartition_samples.shape[0], 1))
