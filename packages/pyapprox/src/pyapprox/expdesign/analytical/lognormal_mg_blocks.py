r"""Exact moment-Gaussian blocks for a lognormal QoI.

For Gaussian inputs :math:`x \sim N(\mu, P)`, noise-free observations
:math:`g = Hx` and a lognormal QoI :math:`q = \exp(Fx)` (elementwise), the
first two moments of :math:`(q, g)` are available in closed form. With
:math:`L = Fx`, :math:`c = F\mu`, :math:`C_L = FPF^\top` and
:math:`p = \mathrm{diag}\, C_L`:

.. math::

    E[q_i] = e^{c_i + p_i / 2}, \qquad
    \mathrm{Cov}(q_i, q_j) = E[q_i]\, E[q_j]\, (e^{(C_L)_{ij}} - 1), \qquad
    \mathrm{Cov}(q, g) = \mathrm{diag}(E[q])\, F P H^\top,

the last by Stein's lemma, since :math:`(L, g)` is jointly Gaussian. A
moment-Gaussian approximation built from these blocks carries no
quadrature error, so comparing it with the exact posterior of :math:`q`
isolates the error of keeping only two moments.

Nuisance parameters are included by stacking them into :math:`x`, with
:math:`H` and :math:`F` padded accordingly.
"""

from dataclasses import dataclass
from typing import Generic

from pyapprox.util.backends.protocols import Array, Backend


@dataclass(frozen=True)
class LogNormalMGBlocks(Generic[Array]):
    """Means and covariance blocks of a lognormal QoI and Gaussian data.

    Parameters
    ----------
    qoi_mean : Array
        Mean of the QoI. Shape: (nqoi, 1)
    obs_mean : Array
        Mean of the noise-free observations. Shape: (nobs, 1)
    qoi_cov : Array
        Covariance of the QoI. Shape: (nqoi, nqoi)
    qoi_obs_cov : Array
        Cross-covariance of the QoI and the noise-free observations.
        Shape: (nqoi, nobs)
    obs_cov : Array
        Covariance of the noise-free observations; observation noise is not
        included. Shape: (nobs, nobs)
    """

    qoi_mean: Array
    obs_mean: Array
    qoi_cov: Array
    qoi_obs_cov: Array
    obs_cov: Array


def _check_2d(name: str, array: Array, shape: tuple[int, int]) -> None:
    if array.ndim != 2 or tuple(array.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {tuple(array.shape)}"
        )


def lognormal_goal_mg_blocks(
    obs_mat: Array,
    qoi_mat: Array,
    prior_mean: Array,
    prior_cov: Array,
    bkd: Backend[Array],
) -> LogNormalMGBlocks[Array]:
    """Return the exact moment-Gaussian blocks of ``(exp(F x), H x)``.

    Parameters
    ----------
    obs_mat : Array
        Observation matrix ``H``. Shape: (nobs, nvars)
    qoi_mat : Array
        Log-QoI matrix ``F``, so that ``q = exp(F x)``. Shape: (nqoi, nvars)
    prior_mean : Array
        Mean of ``x``. Shape: (nvars, 1)
    prior_cov : Array
        Covariance of ``x``. Shape: (nvars, nvars)
    bkd : Backend[Array]
        Computational backend.

    Returns
    -------
    LogNormalMGBlocks[Array]
        The means and covariance blocks of the QoI and the noise-free
        observations.
    """
    nvars = prior_mean.shape[0]
    _check_2d("prior_mean", prior_mean, (nvars, 1))
    _check_2d("prior_cov", prior_cov, (nvars, nvars))
    _check_2d("obs_mat", obs_mat, (obs_mat.shape[0], nvars))
    _check_2d("qoi_mat", qoi_mat, (qoi_mat.shape[0], nvars))

    log_mean = bkd.dot(qoi_mat, prior_mean)
    log_cov = bkd.dot(bkd.dot(qoi_mat, prior_cov), qoi_mat.T)
    log_var = bkd.reshape(bkd.diag(log_cov), (qoi_mat.shape[0], 1))
    qoi_mean = bkd.exp(log_mean + 0.5 * log_var)
    qoi_cov = bkd.dot(qoi_mean, qoi_mean.T) * (bkd.exp(log_cov) - 1.0)
    log_obs_cov = bkd.dot(bkd.dot(qoi_mat, prior_cov), obs_mat.T)
    qoi_obs_cov = qoi_mean * log_obs_cov
    obs_mean = bkd.dot(obs_mat, prior_mean)
    obs_cov = bkd.dot(bkd.dot(obs_mat, prior_cov), obs_mat.T)
    return LogNormalMGBlocks(
        qoi_mean=qoi_mean,
        obs_mean=obs_mean,
        qoi_cov=qoi_cov,
        qoi_obs_cov=qoi_obs_cov,
        obs_cov=obs_cov,
    )
