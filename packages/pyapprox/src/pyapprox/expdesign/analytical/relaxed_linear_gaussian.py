r"""Exact references for linear-Gaussian models at relaxed design weights.

Inputs :math:`x \sim N(\mu, P)` are observed through :math:`y = Hx + e`,
:math:`e \sim N(0, \Gamma_e)`. A relaxed design :math:`w \in [0, 1]^d`
observes the blended data

.. math::

    z_i = \sqrt{w_i}\,(Hx + e)_i + \sqrt{\nu_i}\,\epsilon_i, \qquad
    \nu_i = (1 - w_i)\, s_i^2, \qquad \epsilon \sim N(0, I),

with :math:`\epsilon` independent of everything else and reference scales
:math:`s_i` (by default :math:`s_i^2 = (\Gamma_e)_{ii}`). At :math:`w_i = 1`
the datum is observed; at :math:`w_i = 0` it is pure independent noise. For
diagonal :math:`\Gamma_e` and the default scales this is the usual
precision-weighted relaxation, with noise variance :math:`\sigma_i^2 / w_i`.

Every quantity is computed without square roots through

.. math::

    \mathcal A_w = W\Gamma_{yy} + \Lambda_w, \qquad
    W = \mathrm{diag}(w), \quad \Lambda_w = \mathrm{diag}(\nu), \quad
    \Gamma_{yy} = HPH^\top + \Gamma_e,

using the push-through identity
:math:`W^{1/2}(W^{1/2}\Gamma_{yy}W^{1/2} + \Lambda_w)^{-1}W^{1/2}
= \mathcal A_w^{-1} W`, valid because :math:`\Lambda_w` is diagonal. For a
linear target :math:`t = Tx`,

.. math::

    \Gamma_{t|z} = TPT^\top - TPH^\top \mathcal A_w^{-1} W H P T^\top,
    \qquad
    \det \Gamma_{zz} = \det \mathcal A_w .

These hold for any :math:`w \in [0, 1]^d`, zero weights and correlated
:math:`\Gamma_e` included, as long as :math:`\nu_i > 0` wherever
:math:`w_i = 0`.

Nuisance parameters are marginalized exactly by stacking them into
:math:`x`, with :math:`H` and :math:`T` padded accordingly. A lognormal QoI
:math:`q = \exp(Fx)` is handled through its linear log-target :math:`L = Fx`.
"""

from typing import Optional

from pyapprox.util.backends.protocols import Array, Backend


def _check_shape(name: str, array: Array, shape: tuple[int, int]) -> None:
    if array.ndim != 2 or tuple(array.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {tuple(array.shape)}"
        )


def _check_inputs(
    target_mat: Array,
    obs_mat: Array,
    prior_cov: Array,
    noise_cov: Array,
    weights: Array,
    ref_var: Optional[Array],
) -> None:
    nobs, nvars = obs_mat.shape[0], prior_cov.shape[0]
    _check_shape("prior_cov", prior_cov, (nvars, nvars))
    _check_shape("obs_mat", obs_mat, (nobs, nvars))
    _check_shape("target_mat", target_mat, (target_mat.shape[0], nvars))
    _check_shape("noise_cov", noise_cov, (nobs, nobs))
    _check_shape("weights", weights, (nobs, 1))
    if ref_var is not None:
        _check_shape("ref_var", ref_var, (nobs, 1))


def _relaxed_system(
    obs_mat: Array,
    prior_cov: Array,
    noise_cov: Array,
    weights: Array,
    ref_var: Optional[Array],
    bkd: Backend[Array],
) -> tuple[Array, Array, Array]:
    """Return ``(Gamma_yy, A_w, W)`` for the blended observation."""
    nobs = obs_mat.shape[0]
    if ref_var is None:
        ref_var = bkd.reshape(bkd.diag(noise_cov), (nobs, 1))
    syy = bkd.dot(bkd.dot(obs_mat, prior_cov), obs_mat.T) + noise_cov
    wdiag = bkd.diag(weights[:, 0])
    filler = bkd.diag(((1.0 - weights) * ref_var)[:, 0])
    a_w = bkd.dot(wdiag, syy) + filler
    return syy, a_w, wdiag


def relaxed_linear_target_covariance(
    target_mat: Array,
    obs_mat: Array,
    prior_cov: Array,
    noise_cov: Array,
    weights: Array,
    bkd: Backend[Array],
    ref_var: Optional[Array] = None,
) -> Array:
    """Return the exact posterior covariance of ``T x`` at relaxed weights.

    Parameters
    ----------
    target_mat : Array
        Linear target ``T``. Shape: (ntarget, nvars)
    obs_mat : Array
        Observation matrix ``H``. Shape: (nobs, nvars)
    prior_cov : Array
        Prior covariance ``P`` of ``x``. Shape: (nvars, nvars)
    noise_cov : Array
        Noise covariance ``Gamma_e``. Shape: (nobs, nobs)
    weights : Array
        Design weights in [0, 1]. Shape: (nobs, 1)
    bkd : Backend[Array]
        Computational backend.
    ref_var : Array, optional
        Reference variances ``s^2`` of the blended relaxation.
        Shape: (nobs, 1). Defaults to ``diag(noise_cov)``.

    Returns
    -------
    Array
        ``Gamma_{t|z}``. Shape: (ntarget, ntarget)
    """
    _check_inputs(target_mat, obs_mat, prior_cov, noise_cov, weights, ref_var)
    _, a_w, wdiag = _relaxed_system(
        obs_mat, prior_cov, noise_cov, weights, ref_var, bkd
    )
    ctt = bkd.dot(bkd.dot(target_mat, prior_cov), target_mat.T)
    cty = bkd.dot(bkd.dot(target_mat, prior_cov), obs_mat.T)
    return ctt - bkd.dot(cty, bkd.solve(a_w, bkd.dot(wdiag, cty.T)))


def relaxed_linear_target_eig(
    target_mat: Array,
    obs_mat: Array,
    prior_cov: Array,
    noise_cov: Array,
    weights: Array,
    bkd: Backend[Array],
    ref_var: Optional[Array] = None,
) -> Array:
    """Return the expected information gain about ``T x`` at relaxed weights.

    The mutual information between ``t = T x`` and the blended observation
    ``z``, ``1/2 logdet A_w - 1/2 logdet A_{w|t}``, with
    ``A_{w|t} = W Gamma_{yy|t} + Lambda_w``. ``T P T^T`` must be nonsingular.
    For a lognormal QoI ``q = exp(F x)`` pass ``T = F``: mutual information is
    unchanged by the elementwise exponential.

    Parameters
    ----------
    target_mat, obs_mat, prior_cov, noise_cov, weights, bkd, ref_var
        As in :func:`relaxed_linear_target_covariance`.

    Returns
    -------
    Array
        The expected information gain. Shape: (1, 1)
    """
    _check_inputs(target_mat, obs_mat, prior_cov, noise_cov, weights, ref_var)
    syy, a_w, wdiag = _relaxed_system(
        obs_mat, prior_cov, noise_cov, weights, ref_var, bkd
    )
    ctt = bkd.dot(bkd.dot(target_mat, prior_cov), target_mat.T)
    cyt = bkd.dot(bkd.dot(obs_mat, prior_cov), target_mat.T)
    syy_given_t = syy - bkd.dot(cyt, bkd.solve(ctt, cyt.T))
    a_w_given_t = a_w - bkd.dot(wdiag, syy) + bkd.dot(wdiag, syy_given_t)
    _, logdet_a = bkd.slogdet(a_w)
    _, logdet_a_t = bkd.slogdet(a_w_given_t)
    return bkd.reshape(0.5 * (logdet_a - logdet_a_t), (1, 1))


def relaxed_lognormal_expected_variance(
    qoi_mat: Array,
    obs_mat: Array,
    prior_mean: Array,
    prior_cov: Array,
    noise_cov: Array,
    weights: Array,
    bkd: Backend[Array],
    ref_var: Optional[Array] = None,
) -> Array:
    r"""Return ``E_z[Var(q_i | z)]`` for ``q = exp(F x)`` at relaxed weights.

    With :math:`L = Fx`, :math:`c = F\mu`, :math:`p = \mathrm{diag}(FPF^\top)`
    and :math:`v = \mathrm{diag}\,\Gamma_{L|z}`,
    :math:`E_z[\mathrm{Var}(q_i \mid z)] = (e^{v_i} - 1)\,
    e^{2c_i + 2p_i - v_i}`.

    Parameters
    ----------
    qoi_mat : Array
        Log-QoI matrix ``F``. Shape: (nqoi, nvars)
    prior_mean : Array
        Prior mean of ``x``. Shape: (nvars, 1)
    obs_mat, prior_cov, noise_cov, weights, bkd, ref_var
        As in :func:`relaxed_linear_target_covariance`.

    Returns
    -------
    Array
        Expected posterior variance of each QoI. Shape: (nqoi, 1)
    """
    _check_shape("prior_mean", prior_mean, (prior_cov.shape[0], 1))
    nqoi = qoi_mat.shape[0]
    cov_l = relaxed_linear_target_covariance(
        qoi_mat, obs_mat, prior_cov, noise_cov, weights, bkd, ref_var
    )
    v = bkd.reshape(bkd.diag(cov_l), (nqoi, 1))
    prior_log_cov = bkd.dot(bkd.dot(qoi_mat, prior_cov), qoi_mat.T)
    p = bkd.reshape(bkd.diag(prior_log_cov), (nqoi, 1))
    c = bkd.dot(qoi_mat, prior_mean)
    return (bkd.exp(v) - 1.0) * bkd.exp(2.0 * c + 2.0 * p - v)
