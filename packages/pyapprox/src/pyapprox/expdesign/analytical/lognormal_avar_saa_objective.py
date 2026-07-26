"""
Sample-average exact objective for E_y[AVaR_alpha over lognormal Std].

Defines the objective with respect to two FIXED measures: the quadrature
measure on prediction space (atom masses over the QoI locations) and a
fixed outer rule for the data expectation, supplied as an outer-data
generator (see :mod:`.outer_data`). Conditional on those measures every
term is computed exactly, for any polynomial degree:

1. per outer node the conjugate posterior pushforward at each prediction
   point is exact (no inner loop, no evidence estimation);
2. the weighted discrete AVaR over the prediction atoms uses the exact
   cumulative-mass tail rule (no smoothing);
3. the outer average is a weighted sum over the fixed nodes.

The only gap to the continuous-data objective is the outer rule's
discretization error, which is an explicit, refinable choice. For a
degree-1 basis this evaluator converges to the piecewise-Gaussian closed
form in ``lognormal_avar_objective`` as the outer rule refines.

All computation uses backend operations so autograd backends
differentiate through the whole formula; the per-node atom rankings and
tail masses are piecewise constant in the design weights and are
computed detached (the objective is continuous across ranking
crossings, so the envelope gradient is exact).
"""

from typing import Callable, Generic, Optional

import numpy as np

from pyapprox.interface.functions.autograd import autograd_derivatives
from pyapprox.interface.functions.derivatives import Derivatives
from pyapprox.util.backends.autodiff import AutodiffBackend
from pyapprox.util.backends.protocols import Array, Backend


class LogNormalDataMeanQoIAVaRStdDevSAAObjective(Generic[Array]):
    """
    Exact fixed-measure objective for any polynomial degree.

    U(w) = sum_k omega_k AVaR_alpha({D_j(y_k; w)}_j)

    with D_j(y) = K_j(w) exp(psi_j^T mu*(y; w)) the exact posterior
    lognormal standard deviation of the j-th QoI, and the AVaR taken
    over the prediction atoms with quadrature masses
    ``qoi_quad_weights``.

    Parameters
    ----------
    obs_mat : Array
        Observation matrix A. Shape: (nobs, nparams)
    prior_mean : Array
        Prior mean. Shape: (nparams, 1)
    prior_cov : Array
        Prior covariance. Shape: (nparams, nparams)
    qoi_mat : Array
        QoI matrix B. Shape: (npred, nparams). Any degree.
    noise_variances : Array
        Base noise variances. Shape: (nobs,)
    alpha : float
        AVaR level in [0, 1).
    outer_data : Callable[[Array], Array]
        Outer-data generator: maps design weights (nobs, 1) to data
        nodes (nobs, nouter). See :mod:`.outer_data` for
        implementations, including the double-loop pipeline's
        reparameterization so outer data can be shared exactly.
    bkd : Backend[Array]
    outer_weights : Array, optional
        Weights of the outer rule, shape (nouter,). Default uniform.
        Normalized to sum to one.
    qoi_quad_weights : Array, optional
        Masses of the prediction atoms, shape (npred,). Default
        uniform. Normalized to sum to one.
    """

    def __init__(
        self,
        obs_mat: Array,
        prior_mean: Array,
        prior_cov: Array,
        qoi_mat: Array,
        noise_variances: Array,
        alpha: float,
        outer_data: Callable[[Array], Array],
        bkd: Backend[Array],
        outer_weights: Optional[Array] = None,
        qoi_quad_weights: Optional[Array] = None,
    ) -> None:
        self._obs_mat = obs_mat
        self._prior_mean = prior_mean
        self._prior_cov = prior_cov
        self._prior_cov_inv = bkd.inv(prior_cov)
        self._qoi_mat = qoi_mat
        self._noise_variances = noise_variances
        self._alpha = alpha
        self._outer_data = outer_data
        self._bkd = bkd
        self._nobs = obs_mat.shape[0]
        self._npred = qoi_mat.shape[0]
        self._nouter = outer_data.nouter() if hasattr(
            outer_data, "nouter"
        ) else None

        if outer_weights is not None:
            outer_weights = bkd.reshape(outer_weights, (-1,))
            outer_weights = outer_weights / bkd.sum(outer_weights)
        self._outer_weights = outer_weights

        if qoi_quad_weights is None:
            qoi_quad_weights = bkd.ones((self._npred,)) / self._npred
        qoi_quad_weights = bkd.reshape(qoi_quad_weights, (-1,))
        self._qoi_quad_weights = qoi_quad_weights / bkd.sum(
            qoi_quad_weights
        )

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nobs

    def nqoi(self) -> int:
        return 1

    def _deviations(self, design_weights: Array) -> Array:
        """Exact per-node deviations. Shape: (nouter, npred)."""
        bkd = self._bkd
        w = bkd.reshape(design_weights, (self._nobs,))

        noise_cov_inv = bkd.diag(w / self._noise_variances)
        info_mat = self._obs_mat.T @ (noise_cov_inv @ self._obs_mat)
        post_cov = bkd.inv(info_mat + self._prior_cov_inv)

        # Pushforward variances and lognormal factors, any degree:
        # sigma2[j] = psi_j^T post_cov psi_j
        sigma2 = bkd.sum(
            (self._qoi_mat @ post_cov) * self._qoi_mat, axis=1
        )  # (npred,)
        K = bkd.exp(sigma2 / 2) * bkd.sqrt(bkd.exp(sigma2) - 1)

        y_nodes = self._outer_data(design_weights)  # (nobs, nouter)

        # Posterior means mu*(y) = post_cov (A^T Sn^-1 y + Sigma0^-1 mu0)
        rhs = (
            self._obs_mat.T @ (noise_cov_inv @ y_nodes)
            + self._prior_cov_inv @ self._prior_mean
        )
        mu_star = post_cov @ rhs  # (nparams, nouter)

        # nu[j, k] = psi_j^T mu*(y_k); D = K_j exp(nu)
        nu = self._qoi_mat @ mu_star  # (npred, nouter)
        return (bkd.reshape(K, (-1, 1)) * bkd.exp(nu)).T

    def __call__(self, design_weights: Array) -> Array:
        """Evaluate the objective. Shape: (nobs, 1) -> (1, 1)."""
        bkd = self._bkd
        deviations = self._deviations(design_weights)  # (nouter, npred)

        # Per-node exact weighted AVaR via the cumulative-mass tail
        # rule. The rankings and tail masses are piecewise constant in
        # the design weights, so they are computed detached (numpy) and
        # scattered back to the original atom positions; the autograd
        # graph sees only a constant-weighted sum of the deviations,
        # whose gradient is the exact envelope gradient.
        target = 1.0 - self._alpha
        dev_np = bkd.to_numpy(deviations)
        p_np = bkd.to_numpy(self._qoi_quad_weights)
        order = np.argsort(dev_np, axis=1)[:, ::-1]  # descending

        # Tail mass per sorted position, scattered to atom positions
        p_sorted = np.take_along_axis(
            np.broadcast_to(p_np, dev_np.shape), order, axis=1
        )
        cum = np.cumsum(p_sorted, axis=1)
        prev_cum = cum - p_sorted
        tail_mass = np.clip(target - prev_cum, 0.0, None)
        tail_mass = np.minimum(tail_mass, p_sorted)  # (nouter, npred)
        tail_mass_unsorted = np.empty_like(tail_mass)
        np.put_along_axis(tail_mass_unsorted, order, tail_mass, axis=1)

        tail_w = bkd.asarray(tail_mass_unsorted / target)
        avar_per_node = bkd.sum(tail_w * deviations, axis=1)  # (nouter,)

        if self._outer_weights is None:
            total = bkd.sum(avar_per_node) / avar_per_node.shape[0]
        else:
            total = bkd.sum(self._outer_weights * avar_per_node)
        return bkd.reshape(total, (1, 1))

    def evaluate(self, design_weights: Array) -> Array:
        """Alias for __call__."""
        return self(design_weights)

    def value(self, design_weights: Array) -> float:
        """Return the objective as a float (for diagnostics)."""
        return float(self._bkd.to_numpy(self(design_weights)).flat[0])

    def derivatives(self) -> Derivatives[Array]:
        """Autograd bundle when the backend supports it, else empty."""
        bkd = self._bkd
        if isinstance(bkd, AutodiffBackend):
            return autograd_derivatives(self, bkd)
        empty: Derivatives[Array] = Derivatives.none()
        return empty
