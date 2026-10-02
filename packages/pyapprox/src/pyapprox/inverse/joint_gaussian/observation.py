r"""The relaxed observation of a joint Gaussian, without square roots.

Design weights :math:`w \in [0, 1]^d` and independent-noise variances
:math:`\nu` define the blended observation
:math:`z_i = \sqrt{w_i}\, y_i + \sqrt{\nu_i}\, \epsilon_i`. With
:math:`W = \mathrm{diag}(w)`, :math:`\Lambda = \mathrm{diag}(\nu)` and

.. math::

    \mathcal A_w = W \Gamma_{yy} + \Lambda,

every quantity follows without square roots, which is what keeps them
defined at :math:`w_i = 0`:

.. math::

    \Gamma_{t|z} = \Gamma_{tt} - \Gamma_{ty} \mathcal A_w^{-1} W \Gamma_{yt},
    \qquad
    \log\det\Gamma_{zz} = \log\det \mathcal A_w,

    \mu_{t|z} = \mu_t + \Gamma_{ty} \mathcal A_w^{-1} W (y - \mu_y),

and :math:`\log\det\Gamma_{zz|t} = \log\det\mathcal A_{w|t}` with
:math:`\mathcal A_{w|t} = W \Gamma_{yy|t} + \Lambda`. :math:`\mathcal A_w` is
not symmetric, so it is factored by LU; its determinant equals that of the
symmetric positive definite :math:`\Gamma_{zz}`.
"""

from typing import Generic, Optional

from pyapprox.util.backends.protocols import Array, Backend


class LinearGaussianObservation(Generic[Array]):
    """One target observed through the relaxed observation.

    Built by ``JointGaussian.observe``; not constructed directly.

    Parameters
    ----------
    target_mean, target_cov : Array
        ``mu_t`` (n_t, 1) and ``Gamma_tt`` (n_t, n_t).
    target_obs_cov : Array
        ``Gamma_ty = Gamma_tg``. Shape: (n_t, d)
    obs_mean, obs_cov : Array
        ``mu_y`` (d, 1) and ``Gamma_yy`` (d, d), noise included.
    weights, variances : Array
        ``w`` and ``nu``. Shape: (d, 1)
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        target_mean: Array,
        target_cov: Array,
        target_obs_cov: Array,
        obs_mean: Array,
        obs_cov: Array,
        weights: Array,
        variances: Array,
        bkd: Backend[Array],
    ) -> None:
        self._mu_t, self._ctt, self._cty = target_mean, target_cov, target_obs_cov
        self._mu_y, self._syy = obs_mean, obs_cov
        self._w, self._nu = weights, variances
        self._bkd = bkd
        self._lu, self._piv = bkd.lu_factor(self._a_matrix(obs_cov))
        # X = A_w^{-1} W Gamma_yt, shared by the covariance and the mean.
        self._x = bkd.lu_solve(self._lu, self._piv, weights * target_obs_cov.T)
        self._logdet_given_t: Optional[Array] = None

    def _a_matrix(self, syy: Array) -> Array:
        return self._w * syy + self._bkd.diag(self._nu[:, 0])

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def weights(self) -> Array:
        """Design weights ``w``. Shape: (d, 1)"""
        return self._w

    def variances(self) -> Array:
        """Independent-noise variances ``nu``. Shape: (d, 1)"""
        return self._nu

    def covariance(self) -> Array:
        """``Gamma_t|z``. Shape: (n_t, n_t)"""
        return self._ctt - self._bkd.dot(self._cty, self._x)

    def logdet_zz(self) -> Array:
        """``log det Gamma_zz = log det A_w``. Shape: (1,)"""
        return self._logdet_lu(self._lu)

    def logdet_zz_given_t(self) -> Array:
        """``log det Gamma_zz|t = log det A_w|t``. Shape: (1,)

        Needs ``Gamma_tt`` to be invertible, and is computed on first use.
        """
        if self._logdet_given_t is None:
            bkd = self._bkd
            syy_given_t = self._syy - bkd.dot(
                self._cty.T, bkd.solve(self._ctt, self._cty)
            )
            lu, _ = bkd.lu_factor(self._a_matrix(syy_given_t))
            self._logdet_given_t = self._logdet_lu(lu)
        return self._logdet_given_t

    def mean(self, data: Array) -> Array:
        """``mu_t|z`` given data ``y`` (d, 1). Shape: (n_t, 1)

        Entries of ``data`` where ``w_i = 0`` are multiplied by zero, so
        any placeholder value there is fine.
        """
        d = self._w.shape[0]
        if tuple(data.shape) != (d, 1):
            raise ValueError(f"data must have shape ({d}, 1), got {tuple(data.shape)}")
        bkd = self._bkd
        rhs = self._w * (data - self._mu_y)
        return self._mu_t + bkd.dot(self._cty, bkd.lu_solve(self._lu, self._piv, rhs))

    def _logdet_lu(self, lu: Array) -> Array:
        bkd = self._bkd
        return bkd.reshape(bkd.sum(bkd.log(bkd.abs(bkd.diag(lu)))), (1,))
