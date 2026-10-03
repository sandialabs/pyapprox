r"""Design criteria on a relaxed linear-Gaussian observation.

Each criterion is minimized. With :math:`\Gamma = \Gamma_{t|z}` the
posterior covariance of the target and :math:`C` an optional linear map of
the target:

- ``AOptimal``: :math:`\mathrm{tr}(C\Gamma C^\top)`. A row vector :math:`C = c^\top`
  gives c-optimality, :math:`C = W_t^{1/2}` weighted A-optimality.
- ``DOptimal``: :math:`\log\det(C\Gamma C^\top)`.
- ``ExpectedInformationGain``: :math:`-\mathrm{EIG}`, with
  :math:`\mathrm{EIG} = \tfrac12(\log\det\mathcal A_w - \log\det\mathcal A_{w|t})`.

Because :math:`\log\det\Gamma_{t|z} = \log\det\Gamma_{tt} - 2\,\mathrm{EIG}` and
:math:`\Gamma_{tt}` does not depend on the design, ``DOptimal`` with
:math:`C = I` and ``ExpectedInformationGain`` rank designs identically. They
differ in cost: D-optimality works with :math:`n_t \times n_t` matrices,
the information gain with :math:`d \times d` ones, so the information gain
is the cheaper choice for a large target.

Gradients come from the observation's vector-Jacobian products: a criterion
of :math:`\Gamma` supplies :math:`\partial f / \partial\Gamma`, which
``covariance_vjp`` maps to gradients in :math:`w` and :math:`\nu`.
"""

from typing import Generic, Optional, Tuple

from pyapprox.inverse.joint_gaussian import LinearGaussianObservation
from pyapprox.util.backends.protocols import Array, Backend


def _mapped(bkd: Backend[Array], cov: Array, target_map: Optional[Array]) -> Array:
    """``C Gamma C^T``, or ``Gamma`` when ``C`` is None."""
    if target_map is None:
        return cov
    if target_map.ndim != 2 or target_map.shape[1] != cov.shape[0]:
        raise ValueError(
            f"target_map must have shape (*, {cov.shape[0]}), got "
            f"{tuple(target_map.shape)}"
        )
    return bkd.dot(bkd.dot(target_map, cov), target_map.T)


class AOptimal(Generic[Array]):
    """``tr(C Gamma_t|z C^T)``, the mean posterior variance of ``C t``.

    Parameters
    ----------
    target_map : Array, optional
        ``C``. Shape: (k, n_t). Default the identity.
    """

    def __init__(self, target_map: Optional[Array] = None) -> None:
        self._map = target_map

    def value(self, observation: LinearGaussianObservation[Array]) -> Array:
        """``tr(C Gamma C^T)``. Shape: (1,)"""
        bkd = observation.bkd()
        mapped = _mapped(bkd, observation.covariance(), self._map)
        return bkd.reshape(bkd.trace(mapped), (1,))

    def gradient(
        self, observation: LinearGaussianObservation[Array]
    ) -> Tuple[Array, Array]:
        """From ``d tr(C Gamma C^T) / d Gamma = C^T C``."""
        bkd = observation.bkd()
        ntarget = observation.covariance().shape[0]
        cov_bar = (
            bkd.eye(ntarget) if self._map is None else bkd.dot(self._map.T, self._map)
        )
        return observation.covariance_vjp(cov_bar)


class DOptimal(Generic[Array]):
    """``log det(C Gamma_t|z C^T)``.

    Requires ``C Gamma_t|z C^T`` positive definite, which needs a
    full-rank ``C Gamma_tt C^T``; raises otherwise. A sampled target with
    ``N <= n_t + 1`` is refused, as for the information gain.

    Parameters
    ----------
    target_map : Array, optional
        ``C``. Shape: (k, n_t), ``k <= n_t``. Default the identity.
    """

    def __init__(self, target_map: Optional[Array] = None) -> None:
        self._map = target_map

    def _mapped_cov(self, observation: LinearGaussianObservation[Array]) -> Array:
        observation.check_target_rank()
        return _mapped(observation.bkd(), observation.covariance(), self._map)

    def value(self, observation: LinearGaussianObservation[Array]) -> Array:
        """``log det(C Gamma C^T)``. Shape: (1,)"""
        bkd = observation.bkd()
        sign, logdet = bkd.slogdet(self._mapped_cov(observation))
        if bkd.to_float(sign) <= 0.0:
            raise ValueError(
                "C Gamma_t|z C^T is not positive definite, so its log "
                "determinant is undefined; D-optimality needs a full-rank "
                "C Gamma_tt C^T"
            )
        return bkd.reshape(logdet, (1,))

    def gradient(
        self, observation: LinearGaussianObservation[Array]
    ) -> Tuple[Array, Array]:
        """From ``d log det(C Gamma C^T) / d Gamma = C^T (C Gamma C^T)^{-1} C``."""
        bkd = observation.bkd()
        mapped_inv = bkd.inv(self._mapped_cov(observation))
        if self._map is None:
            cov_bar = mapped_inv
        else:
            cov_bar = bkd.dot(bkd.dot(self._map.T, mapped_inv), self._map)
        return observation.covariance_vjp(cov_bar)


class ExpectedInformationGain(Generic[Array]):
    """``-EIG``, so that minimizing it maximizes the information gain.

    ``EIG = (log det A_w - log det A_w|t) / 2``, the data form, which works
    with ``d x d`` matrices only. It takes no target map: for a function of
    a target, make that function its own target block, since the data form
    needs ``Gamma_yy|t`` for exactly that target. A sampled target with
    ``N <= n_t + 1`` is refused.
    """

    def value(self, observation: LinearGaussianObservation[Array]) -> Array:
        """``-EIG``. Shape: (1,)"""
        return -0.5 * (observation.logdet_zz() - observation.logdet_zz_given_t())

    def gradient(
        self, observation: LinearGaussianObservation[Array]
    ) -> Tuple[Array, Array]:
        """``-(grad log det A_w - grad log det A_w|t) / 2`` in ``w`` and ``nu``."""
        dw_zz, dnu_zz = observation.logdet_zz_gradient()
        dw_t, dnu_t = observation.logdet_zz_given_t_gradient()
        return -0.5 * (dw_zz - dw_t), -0.5 * (dnu_zz - dnu_t)
