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

from typing import Callable, Generic, Optional, Tuple

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
    obs_cov_given_target : Callable[[], Array]
        Returns ``Gamma_yy|t`` (d, d). Called only when a quantity given
        the target is asked for, so its rank guard and cost are paid only
        then, and the caller can cache it across designs.
    check_target_rank : Callable[[], None]
        Raises if the target's moments are sampled from too few samples
        for log-determinant criteria.
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
        obs_cov_given_target: Callable[[], Array],
        check_target_rank: Callable[[], None],
    ) -> None:
        self._mu_t, self._ctt, self._cty = target_mean, target_cov, target_obs_cov
        self._mu_y, self._syy = obs_mean, obs_cov
        self._w, self._nu = weights, variances
        self._bkd = bkd
        self._obs_cov_given_target = obs_cov_given_target
        self._check_target_rank = check_target_rank
        self._lu, self._piv = bkd.lu_factor(self._a_matrix(obs_cov))
        # X = A_w^{-1} W Gamma_yt, shared by the covariance and the mean.
        self._x = bkd.lu_solve(self._lu, self._piv, weights * target_obs_cov.T)
        # LU factors of A_w|t and Gamma_yy|t, computed on first use.
        self._given_t: Optional[Tuple[Array, Array, Array]] = None

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

    def check_target_rank(self) -> None:
        """Raise if log-determinant criteria are meaningless for this target.

        That is when the target's own moments are sampled and
        ``N <= n_t + 1``: the estimate of ``Gamma_yy|t`` then collapses, and
        since ``log det Gamma_t|z = log det Gamma_tt - 2 EIG``, D-optimality
        is as meaningless as the EIG.
        """
        self._check_target_rank()

    def covariance(self) -> Array:
        """``Gamma_t|z``. Shape: (n_t, n_t)"""
        return self._ctt - self._bkd.dot(self._cty, self._x)

    def logdet_zz(self) -> Array:
        """``log det Gamma_zz = log det A_w``. Shape: (1,)"""
        return self._logdet_lu(self._lu)

    def logdet_zz_given_t(self) -> Array:
        """``log det Gamma_zz|t = log det A_w|t``. Shape: (1,)

        Computed on first use, from ``Gamma_yy|t``; raises if the target
        is sampled with too few samples (see
        ``JointGaussian.observation_covariance_given_target``).
        """
        lu, _, _ = self._factor_given_t()
        return self._logdet_lu(lu)

    def covariance_vjp(self, cov_bar: Array) -> Tuple[Array, Array]:
        r"""Gradients in ``w`` and ``nu`` of ``<cov_bar, Gamma_t|z>``.

        With :math:`C = \Gamma_{ty}`, :math:`S = \Gamma_{yy}`,
        :math:`X = \mathcal A_w^{-1} W C^\top` and
        :math:`M = \mathcal A_w^{-\top} C^\top \bar\Gamma`,

        .. math::

            \partial_{w_i} = \sum_k M_{ik} (S X - C^\top)_{ik}, \qquad
            \partial_{\nu_i} = \sum_k M_{ik} X_{ik},

        from :math:`d\Gamma_{t|z} = C\mathcal A_w^{-1}\,dW\,(SX - C^\top)
        + C\mathcal A_w^{-1}\,d\Lambda\,X`.

        Parameters
        ----------
        cov_bar : Array
            ``df / dGamma_t|z``. Shape: (n_t, n_t)

        Returns
        -------
        Tuple[Array, Array]
            ``df/dw`` and ``df/dnu``, each of shape (d, 1).
        """
        ntarget = self._ctt.shape[0]
        if tuple(cov_bar.shape) != (ntarget, ntarget):
            raise ValueError(
                f"cov_bar must have shape ({ntarget}, {ntarget}), got "
                f"{tuple(cov_bar.shape)}"
            )
        bkd = self._bkd
        m = bkd.lu_solve(
            self._lu, self._piv, bkd.dot(self._cty.T, cov_bar), adjoint=True
        )
        residual = bkd.dot(self._syy, self._x) - self._cty.T
        return self._row_sums(m * residual), self._row_sums(m * self._x)

    def logdet_zz_gradient(self) -> Tuple[Array, Array]:
        r"""Gradients of ``log det A_w`` in ``w`` and ``nu``.

        :math:`\partial_{w} = \mathrm{diag}(S\mathcal A_w^{-1})` and
        :math:`\partial_{\nu} = \mathrm{diag}(\mathcal A_w^{-1})`, from
        :math:`d\log\det\mathcal A_w
        = \mathrm{tr}(\mathcal A_w^{-1}(dW\,S + d\Lambda))`.

        Returns
        -------
        Tuple[Array, Array]
            Each of shape (d, 1).
        """
        return self._logdet_gradient(self._lu, self._piv, self._syy)

    def logdet_zz_given_t_gradient(self) -> Tuple[Array, Array]:
        """Gradients of ``log det A_w|t`` in ``w`` and ``nu``, each (d, 1).

        As ``logdet_zz_gradient`` with ``Gamma_yy|t`` in place of
        ``Gamma_yy``.
        """
        lu, piv, syy_given_t = self._factor_given_t()
        return self._logdet_gradient(lu, piv, syy_given_t)

    def _factor_given_t(self) -> Tuple[Array, Array, Array]:
        if self._given_t is None:
            syy_given_t = self._obs_cov_given_target()
            lu, piv = self._bkd.lu_factor(self._a_matrix(syy_given_t))
            self._given_t = (lu, piv, syy_given_t)
        return self._given_t

    def _logdet_gradient(
        self, lu: Array, piv: Array, syy: Array
    ) -> Tuple[Array, Array]:
        bkd = self._bkd
        ainv = bkd.lu_solve(lu, piv, bkd.eye(syy.shape[0]))
        # diag(S A^{-1})_i = sum_k S_ik (A^{-1})_ki
        return self._row_sums(syy * ainv.T), bkd.reshape(bkd.diag(ainv), (-1, 1))

    def _row_sums(self, values: Array) -> Array:
        return self._bkd.reshape(self._bkd.sum(values, axis=1), (-1, 1))

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
