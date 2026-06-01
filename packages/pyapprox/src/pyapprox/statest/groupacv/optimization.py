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
        self._use_analytical: bool = False

    def _ensure_bound(
        self,
    ) -> Tuple[Backend[Array], BaseGroupACVEstimator[Array]]:
        if self._bkd is None or self._est is None:
            raise RuntimeError("Call set_estimator() before using objective")
        return self._bkd, self._est

    def bkd(self) -> Backend[Array]:
        bkd, _ = self._ensure_bound()
        return bkd

    def _check_analytical_support(self) -> bool:
        """Check if analytical derivatives are available for IS estimators."""
        _, est = self._ensure_bound()
        if not hasattr(est._stat, "_group_acv_sigma_block_derivs"):
            return False
        # Only support IS (allocation_mat is identity)
        amat = est._allocation_mat
        bkd = est._bkd
        if amat.shape[0] != amat.shape[1]:
            return False
        if not bkd.allclose(amat, bkd.eye(amat.shape[0])):
            return False
        try:
            subset = est._subsets[0]
            est._stat._group_acv_sigma_block_derivs(subset, 10.0)
            return True
        except NotImplementedError:
            return False

    def set_estimator(self, estimator: BaseGroupACVEstimator[Array]) -> None:
        """Set the estimator and update backend."""
        self._est = estimator
        self._bkd = self._est._bkd
        self._use_analytical = self._check_analytical_support()

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

    def _compute_dpsi_components(
        self, npartition_samples_1d: Array
    ) -> Tuple[List[Array], List[Array], List[Array]]:
        r"""Compute per-partition Psi derivative building blocks for IS.

        For IS estimators, Σ is block-diagonal and ∂_m Σ is zero except in
        block m, so cross partitions vanish (∂_m ∂_p Ψ = 0 for m ≠ p).

        Returns (sigma_invs, dpsi_blocks, d2psi_blocks) where:
          ∂_m Ψ = -R_m Σ_m^{-1} (∂_m Σ_m) Σ_m^{-1} R_m^T
          ∂²_m Ψ = R_m Σ_m^{-1} [2(∂_m Σ_m)Σ_m^{-1}(∂_m Σ_m) - ∂²_m Σ_m] Σ_m^{-1} R_m^T

                """
        bkd, est = self._ensure_bound()
        npartitions = npartition_samples_1d.shape[0]
        Rmats = est._restriction_matrices
        sigma_invs: List[Array] = []
        dpsi_blocks: List[Array] = []
        d2psi_blocks: List[Array] = []
        for m in range(npartitions):
            n_m = npartition_samples_1d[m]
            subset = est._subsets[m]
            sigma_m = est._stat._group_acv_sigma_block(
                subset, subset, n_m, n_m, n_m
            )
            sigma_m_inv = est._inv(sigma_m)
            d1_m, d2_m = est._stat._group_acv_sigma_block_derivs(subset, n_m)
            R_m = Rmats[m]
            sinv_d1_sinv = sigma_m_inv @ d1_m @ sigma_m_inv
            dpsi_m = bkd.multidot([-R_m, sinv_d1_sinv, R_m.T])
            # d²(Σ⁻¹)/dn² = 2 Σ⁻¹ Σ' Σ⁻¹ Σ' Σ⁻¹ - Σ⁻¹ Σ'' Σ⁻¹
            d2_sinv = (
                2 * sinv_d1_sinv @ d1_m @ sigma_m_inv
                - sigma_m_inv @ d2_m @ sigma_m_inv
            )
            d2psi_m = bkd.multidot([R_m, d2_sinv, R_m.T])
            sigma_invs.append(sigma_m_inv)
            dpsi_blocks.append(dpsi_m)
            d2psi_blocks.append(d2psi_m)
        return sigma_invs, dpsi_blocks, d2psi_blocks

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
    f(n) = tr(A Ψ⁻¹ Aᵀ) = tr(Ψ⁻¹ Ã) where Ã = Aᵀ A.
    """

    def _objective_wrapper(self, npartition_samples_1d: Array) -> Array:
        bkd, est = self._ensure_bound()
        trace = bkd.trace(
            est._covariance_from_npartition_samples(npartition_samples_1d)
        )
        # conversion below is necessary for torch
        return bkd.hstack((trace,))[:, None]

    def jacobian(self, npartition_samples: Array) -> Array:
        r"""Analytical jacobian for trace objective.

        ∂f/∂n_m = -tr(G ∂_m Ψ)  where G = Ψ⁻¹ Ã Ψ⁻¹, Ã = Aᵀ A
        """
        if not self._use_analytical:
            return super().jacobian(npartition_samples)
        bkd, est = self._ensure_bound()
        n1d = npartition_samples[:, 0]
        psi = est._psi_matrix(n1d)
        psi_inv = est._inv(psi)
        A_tilde = est._asketch.T @ est._asketch
        G = psi_inv @ A_tilde @ psi_inv
        _, dpsi_blocks, _ = self._compute_dpsi_components(n1d)
        npartitions = n1d.shape[0]
        jac_entries = [
            -bkd.trace(G @ dpsi_blocks[m]) for m in range(npartitions)
        ]
        return bkd.hstack(jac_entries)[None, :]

    def hessian(self, npartition_samples: Array) -> Array:
        r"""Analytical hessian for trace objective.

        H^f_{mp} = 2 tr(G ∂_p Ψ Ψ⁻¹ ∂_m Ψ) - δ_{mp} tr(G ∂²_m Ψ)
        """
        if not self._use_analytical:
            raise NotImplementedError(
                "Analytical hessian not available; stat does not provide "
                "_group_acv_sigma_block_derivs or estimator is not IS"
            )
        bkd, est = self._ensure_bound()
        n1d = npartition_samples[:, 0]
        psi = est._psi_matrix(n1d)
        psi_inv = est._inv(psi)
        A_tilde = est._asketch.T @ est._asketch
        G = psi_inv @ A_tilde @ psi_inv
        _, dpsi_blocks, d2psi_blocks = self._compute_dpsi_components(n1d)
        npartitions = n1d.shape[0]
        G_dpsi_psiinv = [G @ dpsi_blocks[m] @ psi_inv for m in range(npartitions)]
        hess_rows = []
        for m in range(npartitions):
            row_entries = []
            for p in range(npartitions):
                val = 2 * bkd.trace(G_dpsi_psiinv[m] @ dpsi_blocks[p])
                if m == p:
                    val = val - bkd.trace(G @ d2psi_blocks[m])
                row_entries.append(val)
            hess_rows.append(bkd.hstack(row_entries))
        return bkd.vstack(hess_rows)

    def hvp(self, npartition_samples: Array, vec: Array) -> Array:
        hess = self.hessian(npartition_samples)
        return hess @ vec


class GroupACVLogDetObjective(GroupACVObjective[Array]):
    """Log-determinant objective for GroupACV optimization.

    Minimizes the log-determinant of the estimator covariance matrix.
    g(n) = log det(A Ψ⁻¹ Aᵀ) = log det(cov(n)).
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

    def _compute_logdet_matrices(
        self, npartition_samples_1d: Array
    ) -> Tuple[Array, Array, Array]:
        """Compute shared matrices for log-det jacobian and hessian.

        Returns (psi_inv, Lambda, dpsi_blocks + d2psi_blocks via caller).
        Λ = Ψ⁻¹ Aᵀ C⁻¹ A Ψ⁻¹
        P = Φ Φᵀ where Φ = Ψ⁻¹ Aᵀ C⁻¹ (for jacobian only, P = Λ Ψ Ψ⁻¹ = Λ
        when Φ^T Φ is not needed, but we compute Λ directly).
        """
        bkd, est = self._ensure_bound()
        psi = est._psi_matrix(npartition_samples_1d)
        psi_inv = est._inv(psi)
        A = est._asketch
        cov = A @ psi_inv @ A.T
        cov_inv = est._inv(cov)
        # Λ = Ψ⁻¹ Aᵀ C⁻¹ A Ψ⁻¹
        Lambda = psi_inv @ A.T @ cov_inv @ A @ psi_inv
        return psi_inv, Lambda, cov_inv

    def jacobian(self, npartition_samples: Array) -> Array:
        r"""Analytical jacobian for log-det objective.

        ∂g/∂n_m = -tr(Λ ∂_m Ψ)
        where Λ = Ψ⁻¹ Aᵀ C⁻¹ A Ψ⁻¹, C = A Ψ⁻¹ Aᵀ
        """
        if not self._use_analytical:
            return super().jacobian(npartition_samples)
        bkd, est = self._ensure_bound()
        n1d = npartition_samples[:, 0]
        psi_inv, Lambda, _ = self._compute_logdet_matrices(n1d)
        _, dpsi_blocks, _ = self._compute_dpsi_components(n1d)
        npartitions = n1d.shape[0]
        jac_entries = [
            -bkd.trace(Lambda @ dpsi_blocks[m]) for m in range(npartitions)
        ]
        return bkd.hstack(jac_entries)[None, :]

    def hessian(self, npartition_samples: Array) -> Array:
        r"""Analytical hessian for log-det objective.

        H^g_{mp} = -tr(Λ ∂_m Ψ Λ ∂_p Ψ) + 2 tr(Λ ∂_p Ψ Ψ⁻¹ ∂_m Ψ)
                   - δ_{mp} tr(Λ ∂²_m Ψ)
        where Λ = Ψ⁻¹ Aᵀ C⁻¹ A Ψ⁻¹
        """
        if not self._use_analytical:
            raise NotImplementedError(
                "Analytical hessian not available; stat does not provide "
                "_group_acv_sigma_block_derivs or estimator is not IS"
            )
        bkd, est = self._ensure_bound()
        n1d = npartition_samples[:, 0]
        psi_inv, Lambda, _ = self._compute_logdet_matrices(n1d)
        _, dpsi_blocks, d2psi_blocks = self._compute_dpsi_components(n1d)
        npartitions = n1d.shape[0]
        Lam_dpsi_psiinv = [
            Lambda @ dpsi_blocks[m] @ psi_inv for m in range(npartitions)
        ]
        Lam_dpsi = [Lambda @ dpsi_blocks[m] for m in range(npartitions)]
        hess_rows = []
        for m in range(npartitions):
            row_entries = []
            for p in range(npartitions):
                # -tr(Λ ∂_m Ψ Λ ∂_p Ψ) + 2 tr(Λ ∂_p Ψ Ψ⁻¹ ∂_m Ψ)
                cross = -bkd.trace(Lam_dpsi[m] @ Lam_dpsi[p])
                linear = 2 * bkd.trace(Lam_dpsi_psiinv[m] @ dpsi_blocks[p])
                val = cross + linear
                if m == p:
                    val = val - bkd.trace(Lambda @ d2psi_blocks[m])
                row_entries.append(val)
            hess_rows.append(bkd.hstack(row_entries))
        return bkd.vstack(hess_rows)

    def hvp(self, npartition_samples: Array, vec: Array) -> Array:
        hess = self.hessian(npartition_samples)
        return hess @ vec


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
