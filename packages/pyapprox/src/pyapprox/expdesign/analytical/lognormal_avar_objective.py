"""
Differentiable exact objective for E_y[AVaR_alpha over vector lognormal Std].

Provides a standalone class satisfying OEDObjectiveProtocol with all
computation using Backend methods so PyTorch autograd can compute
gradients through the entire formula.
"""

import itertools
import math
from typing import Generic, List, Optional, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


def _compute_crossing_thresholds(
    x_vals: List[float], log_K_vals: List[float]
) -> List[float]:
    """Compute all C(Q,2) crossing thresholds on mu_1^*.

    D_i > D_j iff (x_i - x_j)*mu_1^* > log(K_j/K_i).
    Threshold: mu_1^* = log(K_j/K_i) / (x_i - x_j) when x_i != x_j.
    """
    thresholds: List[float] = []
    npred = len(x_vals)
    for i, j in itertools.combinations(range(npred), 2):
        dx = x_vals[i] - x_vals[j]
        if abs(dx) < 1e-15:
            continue
        t = (log_K_vals[j] - log_K_vals[i]) / dx
        thresholds.append(t)
    return sorted(set(thresholds))


class LogNormalDataMeanQoIAVaRStdDevObjective(Generic[Array]):
    """
    Differentiable exact objective: E_y[AVaR_alpha over vector lognormal Std].

    Satisfies OEDObjectiveProtocol: __call__(weights) -> Array (1,1),
    jacobian(weights) -> Array (1, nobs).

    For a linear Gaussian model with Q prediction QoIs where
    W_j = exp(psi_j^T theta), computes:

        U(w) = E_y[ AVaR_alpha({Std(W_1|y), ..., Std(W_Q|y)}) ]

    Uses the general formula that handles arbitrary (non-equal) posterior
    variances across QoI locations. Enumerates all C(Q,2) crossing
    thresholds on mu_1^* and integrates piecewise.

    All operations use Backend methods so PyTorch autograd can compute
    gradients through the entire formula.

    Parameters
    ----------
    obs_mat : Array
        Observation matrix A. Shape: (nobs, nparams)
    prior_mean : Array
        Prior mean. Shape: (nparams, 1)
    prior_cov : Array
        Prior covariance. Shape: (nparams, nparams)
    qoi_mat : Array
        QoI matrix B. Shape: (npred, nparams). Must be degree-1 basis
        (2 columns: [1, x_j]).
    noise_variances : Array
        Base noise variances. Shape: (nobs,)
    alpha : float
        AVaR level in [0, 1). alpha=0 = I-optimality,
        alpha->(Q-1)/Q = G-optimality.
    bkd : Backend[Array]
    """

    def __init__(
        self,
        obs_mat: Array,
        prior_mean: Array,
        prior_cov: Array,
        qoi_mat: Array,
        noise_variances: Array,
        alpha: float,
        bkd: Backend[Array],
    ) -> None:
        self._obs_mat = obs_mat
        self._prior_mean = prior_mean
        self._prior_cov = prior_cov
        self._prior_cov_inv = bkd.inv(prior_cov)
        self._qoi_mat = qoi_mat
        self._noise_variances = noise_variances
        self._alpha = alpha
        self._bkd = bkd
        self._nobs = obs_mat.shape[0]
        self._npred = qoi_mat.shape[0]
        self._m = math.ceil(self._npred * (1 - alpha))

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nobs

    def nqoi(self) -> int:
        return 1

    def __call__(self, design_weights: Array) -> Array:
        """Evaluate exact utility. Shape: (nobs, 1) -> (1, 1).

        All computation uses backend ops to preserve autograd graph.
        """
        bkd = self._bkd
        npred = self._npred
        m = self._m
        w = bkd.reshape(design_weights, (self._nobs,))

        # Build noise precision from weights
        noise_cov_inv = bkd.diag(w / self._noise_variances)

        # Posterior covariance
        info_mat = self._obs_mat.T @ (noise_cov_inv @ self._obs_mat)
        Sigma_star = bkd.inv(info_mat + self._prior_cov_inv)

        # nu_vec = E[mu_*], Cmat = Cov[mu_*]
        # Same formula as DenseGaussianConjugatePosterior
        Rmat = Sigma_star @ (self._obs_mat.T @ noise_cov_inv)
        ROmat = Rmat @ self._obs_mat
        nu_vec = (
            ROmat @ self._prior_mean
            + Sigma_star @ (self._prior_cov_inv @ self._prior_mean)
        )
        noise_cov = bkd.diag(self._noise_variances / w)
        Cmat = ROmat @ (self._prior_cov @ ROmat.T) + Rmat @ (noise_cov @ Rmat.T)

        c_11 = Cmat[1, 1]
        c_01 = Cmat[0, 1]
        nu_1 = nu_vec[1, 0]
        std_mu1 = bkd.sqrt(c_11)

        # Per-QoI: K_j, nu_j, sigma_tau_j^2, cov(tau_j, mu_1^*)
        K_arr = []
        base_arr = []  # K_j * exp(nu_j + sigma_tau_j^2/2)
        shift_arr = []  # cov_j1 / std_mu1
        x_vals_np = []
        log_K_np = []

        for j in range(npred):
            psi_j = self._qoi_mat[j : j + 1]
            sigma_post_j_sq = bkd.sum(psi_j * (psi_j @ Sigma_star), axis=1)
            K_j = bkd.exp(sigma_post_j_sq / 2) * bkd.sqrt(
                bkd.exp(sigma_post_j_sq) - 1
            )
            nu_j = (psi_j @ nu_vec)[0, 0]
            sigma_tau_j_sq = (psi_j @ Cmat @ psi_j.T)[0, 0]
            cov_j1 = c_01 + self._qoi_mat[j, 1] * c_11

            K_arr.append(K_j)
            base_arr.append(K_j * bkd.exp(nu_j + sigma_tau_j_sq / 2))
            shift_arr.append(cov_j1 / std_mu1)

            x_vals_np.append(float(bkd.to_numpy(self._qoi_mat[j, 1])))
            log_K_np.append(float(bkd.to_numpy(bkd.log(K_j)).flat[0]))

        # alpha=0: all terms contribute, integral telescopes to 1
        if self._alpha == 0.0:
            total = bkd.zeros((1,))
            for j in range(npred):
                total = total + bkd.reshape(base_arr[j], (1,))
            return bkd.reshape(total / m, (1, 1))

        # Crossing thresholds (computed in numpy for sorting)
        thresholds = _compute_crossing_thresholds(x_vals_np, log_K_np)

        if not thresholds:
            span = 1.0
        else:
            span = (
                thresholds[-1] - thresholds[0]
                if len(thresholds) > 1
                else 1.0
            )

        sentinel_lo = (
            (thresholds[0] - 10 * span - 1) if thresholds
            else float(bkd.to_numpy(nu_1))
        )
        sentinel_hi = (
            (thresholds[-1] + 10 * span + 1) if thresholds
            else float(bkd.to_numpy(nu_1))
        )

        boundaries: List[Optional[float]] = (
            [None] + [float(t) for t in thresholds] + [None]
        )

        total = bkd.zeros((1,))
        for k in range(len(boundaries) - 1):
            t_lo = boundaries[k]
            t_hi = boundaries[k + 1]

            # Representative for D ordering
            if t_lo is None and t_hi is None:
                rep = float(bkd.to_numpy(nu_1))
            elif t_lo is None:
                rep = sentinel_lo
            elif t_hi is None:
                rep = sentinel_hi
            else:
                rep = (t_lo + t_hi) / 2

            log_D_at_rep = [
                log_K_np[j] + x_vals_np[j] * rep for j in range(npred)
            ]
            ranked = sorted(
                range(npred), key=lambda j: log_D_at_rep[j], reverse=True
            )
            tail_indices = ranked[:m]

            for j in tail_indices:
                base_j = bkd.reshape(base_arr[j], (1,))
                shift_j = shift_arr[j]

                if t_hi is None:
                    phi_hi = bkd.ones((1,))
                else:
                    arg_hi = (t_hi - nu_1) / std_mu1 - shift_j
                    phi_hi = bkd.ndtr(bkd.reshape(arg_hi, (1,)))
                if t_lo is None:
                    phi_lo = bkd.zeros((1,))
                else:
                    arg_lo = (t_lo - nu_1) / std_mu1 - shift_j
                    phi_lo = bkd.ndtr(bkd.reshape(arg_lo, (1,)))

                total = total + base_j * (phi_hi - phi_lo)

        return bkd.reshape(total / m, (1, 1))

    def evaluate(self, design_weights: Array) -> Array:
        """Alias for __call__."""
        return self(design_weights)

    def value(self, design_weights: Array) -> float:
        """Return utility as a float (for diagnostics)."""
        return float(self._bkd.to_numpy(self(design_weights)).flat[0])

    def jacobian(self, design_weights: Array) -> Array:
        """Jacobian via finite differences (numpy) or autograd (torch).

        Parameters
        ----------
        design_weights : Array
            Shape: (nobs, 1)

        Returns
        -------
        Array
            Shape: (1, nobs)
        """
        bkd = self._bkd
        nobs = self._nobs
        eps = 1e-7
        f0 = self(design_weights)
        jac_cols = []
        for i in range(nobs):
            one_hot = bkd.reshape(bkd.eye(nobs)[i], (nobs, 1))
            w_plus = design_weights + eps * one_hot
            f_plus = self(w_plus)
            jac_cols.append((f_plus[0, 0] - f0[0, 0]) / eps)
        return bkd.reshape(bkd.stack(jac_cols), (1, nobs))
