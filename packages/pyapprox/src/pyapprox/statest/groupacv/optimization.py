"""Optimization infrastructure for GroupACV estimators.

This module provides objective functions and constraints for
GroupACV sample allocation optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Generic, List, Optional, Tuple

import numpy as np

from pyapprox.statest.groupacv.utils import (
    _grouped_acv_sigma,
    _grouped_acv_sigma_block,
)
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

    def _compute_psi_and_derivs(
        self, npartition_samples_1d: Array
    ) -> Tuple[Array, Array, Array, Array]:
        r"""Compute Ψ, Ψ⁻¹, and stacked derivative blocks in one pass.

        For IS estimators, Σ is block-diagonal so each partition's sigma
        block, its inverse, and derivatives are computed once and reused
        for both Ψ construction and derivative blocks.

        Returns (psi, psi_inv, dpsi_stack, d2psi_stack) where:
          dpsi_stack[m] = ∂_m Ψ = -R_m Σ_m⁻¹ (∂_m Σ_m) Σ_m⁻¹ R_mᵀ
          d2psi_stack[m] = ∂²_m Ψ
        """
        bkd, est = self._ensure_bound()
        npartitions = npartition_samples_1d.shape[0]
        Rmats = est._restriction_matrices
        dpsi_blocks: List[Array] = []
        d2psi_blocks: List[Array] = []
        # Build Ψ = Σ_m R_m Σ_m⁻¹ R_mᵀ + λI incrementally
        nT = est._nT_stats
        psi = bkd.eye(nT) * est._reg_blue
        zero_block = bkd.zeros((nT, nT))
        for m in range(npartitions):
            n_m = npartition_samples_1d[m]
            subset = est._subsets[m]
            sigma_m = _grouped_acv_sigma_block(
                subset, subset, n_m, n_m, n_m, est._stat
            )
            if bkd.all_bool(sigma_m == 0):
                dpsi_blocks.append(zero_block)
                d2psi_blocks.append(zero_block)
                continue
            sigma_m_inv = est._inv(sigma_m)
            R_m = Rmats[m]
            # Accumulate Ψ = Σ R_m Σ_m⁻¹ R_mᵀ
            psi = psi + bkd.multidot([R_m, sigma_m_inv, R_m.T])
            d1_m, d2_m = est._stat._group_acv_sigma_block_derivs(subset, n_m)
            sinv_d1_sinv = sigma_m_inv @ d1_m @ sigma_m_inv
            dpsi_m = bkd.multidot([-R_m, sinv_d1_sinv, R_m.T])
            # d²(Σ⁻¹)/dn² = 2 Σ⁻¹ Σ' Σ⁻¹ Σ' Σ⁻¹ - Σ⁻¹ Σ'' Σ⁻¹
            d2_sinv = (
                2 * sinv_d1_sinv @ d1_m @ sigma_m_inv
                - sigma_m_inv @ d2_m @ sigma_m_inv
            )
            d2psi_m = bkd.multidot([R_m, d2_sinv, R_m.T])
            dpsi_blocks.append(dpsi_m)
            d2psi_blocks.append(d2psi_m)
        psi_inv = est._inv(psi)
        return psi, psi_inv, bkd.stack(dpsi_blocks), bkd.stack(d2psi_blocks)

    @staticmethod
    def _batch_trace_product(
        bkd: Backend[Array], A_stack: Array, B_stack: Array
    ) -> Array:
        """Compute tr(A_m @ B_p) for all (m, p) pairs via einsum.

        Parameters
        ----------
        A_stack : Array, shape (M, d, d)
        B_stack : Array, shape (P, d, d)

        Returns
        -------
        Array, shape (M, P)
            result[m, p] = tr(A_stack[m] @ B_stack[p])
        """
        return bkd.einsum("mij,pji->mp", A_stack, B_stack)

    @staticmethod
    def _batch_trace_single(
        bkd: Backend[Array], G: Array, B_stack: Array
    ) -> Array:
        """Compute tr(G @ B_m) for all m via einsum.

        Parameters
        ----------
        G : Array, shape (d, d)
        B_stack : Array, shape (M, d, d)

        Returns
        -------
        Array, shape (M,)
            result[m] = tr(G @ B_stack[m])
        """
        return bkd.einsum("ij,mji->m", G, B_stack)

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
        _, psi_inv, dpsi_stack, _ = self._compute_psi_and_derivs(n1d)
        A_tilde = est._asketch.T @ est._asketch
        G = psi_inv @ A_tilde @ psi_inv
        jac = -self._batch_trace_single(bkd, G, dpsi_stack)
        return jac[None, :]

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
        _, psi_inv, dpsi_stack, d2psi_stack = self._compute_psi_and_derivs(
            n1d
        )
        A_tilde = est._asketch.T @ est._asketch
        G = psi_inv @ A_tilde @ psi_inv
        # G_dpsi_psiinv[m] = G @ dpsi[m] @ Ψ⁻¹, stacked as (M, d, d)
        G_dpsi_psiinv_stack = bkd.einsum(
            "ij,mjk,kl->mil", G, dpsi_stack, psi_inv
        )
        # H[m,p] = 2 tr(G_dpsi_psiinv[m] @ dpsi[p])
        H = 2 * self._batch_trace_product(
            bkd, G_dpsi_psiinv_stack, dpsi_stack
        )
        # diagonal correction: H[m,m] -= tr(G @ d2psi[m])
        diag_corr = self._batch_trace_single(bkd, G, d2psi_stack)
        H = H - bkd.diag(diag_corr)
        return H

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
        self, psi_inv: Array
    ) -> Array:
        """Compute Λ = Ψ⁻¹ Aᵀ C⁻¹ A Ψ⁻¹ from precomputed psi_inv."""
        bkd, est = self._ensure_bound()
        A = est._asketch
        cov = A @ psi_inv @ A.T
        cov_inv = est._inv(cov)
        return psi_inv @ A.T @ cov_inv @ A @ psi_inv

    def jacobian(self, npartition_samples: Array) -> Array:
        r"""Analytical jacobian for log-det objective.

        ∂g/∂n_m = -tr(Λ ∂_m Ψ)
        where Λ = Ψ⁻¹ Aᵀ C⁻¹ A Ψ⁻¹, C = A Ψ⁻¹ Aᵀ
        """
        if not self._use_analytical:
            return super().jacobian(npartition_samples)
        bkd, est = self._ensure_bound()
        n1d = npartition_samples[:, 0]
        _, psi_inv, dpsi_stack, _ = self._compute_psi_and_derivs(n1d)
        Lambda = self._compute_logdet_matrices(psi_inv)
        jac = -self._batch_trace_single(bkd, Lambda, dpsi_stack)
        return jac[None, :]

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
        _, psi_inv, dpsi_stack, d2psi_stack = self._compute_psi_and_derivs(
            n1d
        )
        Lambda = self._compute_logdet_matrices(psi_inv)
        # Lam_dpsi[m] = Λ @ dpsi[m], stacked as (M, d, d)
        Lam_dpsi_stack = bkd.einsum("ij,mjk->mik", Lambda, dpsi_stack)
        # Lam_dpsi_psiinv[m] = Λ @ dpsi[m] @ Ψ⁻¹, stacked as (M, d, d)
        Lam_dpsi_psiinv_stack = bkd.einsum(
            "mij,jk->mik", Lam_dpsi_stack, psi_inv
        )
        # cross[m,p] = -tr(Lam_dpsi[m] @ Lam_dpsi[p])
        cross = -self._batch_trace_product(
            bkd, Lam_dpsi_stack, Lam_dpsi_stack
        )
        # linear[m,p] = 2 tr(Lam_dpsi_psiinv[m] @ dpsi[p])
        linear = 2 * self._batch_trace_product(
            bkd, Lam_dpsi_psiinv_stack, dpsi_stack
        )
        H = cross + linear
        # diagonal correction: H[m,m] -= tr(Λ @ d2psi[m])
        diag_corr = self._batch_trace_single(bkd, Lambda, d2psi_stack)
        H = H - bkd.diag(diag_corr)
        return H

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
