"""Protocols for joint moments of targets and observations."""

from typing import Generic, Optional, Protocol, runtime_checkable

from pyapprox.interface.functions.joint import JointOutputs
from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class CovarianceBlocksProtocol(Protocol, Generic[Array]):
    r"""Means and covariance blocks of targets and noise-free observations.

    With target blocks :math:`t_1, \dots, t_r` and observations :math:`g`,
    exposes the means, each :math:`\Gamma_{t_k t_k}`, each
    :math:`\Gamma_{t_k g}` and :math:`\Gamma_{gg}`.
    """

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        ...

    def target_sizes(self) -> tuple[int, ...]:
        """Number of rows in each target block."""
        ...

    def nobs(self) -> int:
        """Number of observations."""
        ...

    def target_mean(self, index: int) -> Array:
        """Mean of target block ``index``. Shape: (target_sizes[index], 1)"""
        ...

    def obs_mean(self) -> Array:
        """Mean of the observations. Shape: (nobs, 1)"""
        ...

    def target_covariance(self, index: int) -> Array:
        """``Gamma_tt`` of block ``index``. Shape: (n_t, n_t)"""
        ...

    def target_obs_covariance(self, index: int) -> Array:
        """``Gamma_tg`` of block ``index``. Shape: (n_t, nobs)"""
        ...

    def obs_covariance(self) -> Array:
        """``Gamma_gg``, without noise. Shape: (nobs, nobs)"""
        ...

    def nsamples(self) -> Optional[int]:
        """Number of samples behind the blocks, or None if exact."""
        ...


@runtime_checkable
class MomentAccumulatorProtocol(Protocol, Generic[Array]):
    """Accumulates weighted joint moments over batches of outputs."""

    def update(self, weights: Array, outputs: JointOutputs[Array]) -> None:
        """Add one batch.

        Parameters
        ----------
        weights : Array
            Quadrature weights of the batch. Shape: (1, nsamples)
        outputs : JointOutputs[Array]
            Targets and observations at the batch's samples.
        """
        ...

    def finalize(self) -> CovarianceBlocksProtocol[Array]:
        """Blocks from every batch added so far."""
        ...
