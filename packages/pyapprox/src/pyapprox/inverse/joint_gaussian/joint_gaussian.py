r"""The Gaussian defined by covariance blocks and additive noise.

Targets :math:`t_k` and noise-free observations :math:`g` have joint
moments given by blocks; data are :math:`y = g + e` with
:math:`e \sim N(0, \Gamma_e)` independent of the inputs, so
:math:`\Gamma_{ty} = \Gamma_{tg}` and :math:`\Gamma_{yy} = \Gamma_{gg} + \Gamma_e`.
Treating the joint as Gaussian, target :math:`k` given data has

.. math::

    \mu_{t|y} = \mu_t + \Gamma_{tg} \Gamma_{yy}^{-1} (y - \mu_g), \qquad
    \Gamma_{t|y} = \Gamma_{tt} - \Gamma_{tg} \Gamma_{yy}^{-1} \Gamma_{gt}.

This is exact when targets and observations are jointly Gaussian and the
best linear predictor otherwise.
"""

from typing import Generic, Optional, Sequence, Tuple

from pyapprox.probability.covariance import DenseCholeskyCovarianceOperator
from pyapprox.probability.moments import (
    CovarianceRepairProtocol,
    DenseBlocks,
    NoRepair,
)
from pyapprox.probability.protocols import CovarianceOperatorProtocol
from pyapprox.util.backends.protocols import Array, Backend


class _AlreadyChecked(Generic[Array]):
    """Pass-through for blocks known to be positive semidefinite.

    Used only for principal submatrices of blocks a ``JointGaussian``
    has already accepted, which are positive semidefinite by construction.
    """

    def repair(self, blocks: DenseBlocks[Array]) -> DenseBlocks[Array]:
        return blocks


class JointGaussian(Generic[Array]):
    """Targets and noisy observations as one Gaussian.

    Parameters
    ----------
    blocks : DenseBlocks[Array]
        Means and covariances of the targets and noise-free observations.
    noise : CovarianceOperatorProtocol[Array]
        Covariance of the additive observation noise, of size ``nobs``.
    repair : CovarianceRepairProtocol[Array], optional
        Applied to the blocks at construction. Default ``NoRepair()``,
        which raises if the blocks are indefinite; pass ``EigenClip`` to
        repair them instead. Nothing is repaired unless asked for.
    """

    def __init__(
        self,
        blocks: DenseBlocks[Array],
        noise: CovarianceOperatorProtocol[Array],
        repair: Optional[CovarianceRepairProtocol[Array]] = None,
    ) -> None:
        if not isinstance(blocks, DenseBlocks):
            raise TypeError(f"blocks must be DenseBlocks, got {type(blocks).__name__}")
        if not isinstance(noise, CovarianceOperatorProtocol):
            raise TypeError(
                "noise must satisfy CovarianceOperatorProtocol, got "
                f"{type(noise).__name__}"
            )
        if noise.nvars() != blocks.nobs():
            raise ValueError(
                f"noise has size {noise.nvars()} but the blocks have "
                f"{blocks.nobs()} observations"
            )
        repair = NoRepair() if repair is None else repair
        if not isinstance(repair, CovarianceRepairProtocol):
            raise TypeError(
                "repair must satisfy CovarianceRepairProtocol, got "
                f"{type(repair).__name__}"
            )
        self._blocks = repair.repair(blocks)
        self._noise = noise
        self._bkd = blocks.bkd()

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def blocks(self) -> DenseBlocks[Array]:
        """The (repaired, if asked for) blocks."""
        return self._blocks

    def noise(self) -> CovarianceOperatorProtocol[Array]:
        """The observation noise covariance."""
        return self._noise

    def nobs(self) -> int:
        """Number of observations."""
        return self._blocks.nobs()

    def target_sizes(self) -> tuple[int, ...]:
        """Number of rows in each target block."""
        return self._blocks.target_sizes()

    def obs_covariance(self) -> Array:
        """``Gamma_yy = Gamma_gg + Gamma_e``. Shape: (nobs, nobs)"""
        return self._blocks.obs_covariance() + self._noise.covariance()

    def condition(self, data: Array, index: int) -> Tuple[Array, Array]:
        """Mean and covariance of target ``index`` given data.

        Parameters
        ----------
        data : Array
            Observed data. Shape: (nobs, 1)
        index : int
            Which target block.

        Returns
        -------
        Tuple[Array, Array]
            Mean of shape (n_t, 1) and covariance of shape (n_t, n_t).
        """
        nobs = self.nobs()
        if tuple(data.shape) != (nobs, 1):
            raise ValueError(
                f"data must have shape ({nobs}, 1), got {tuple(data.shape)}"
            )
        self._check_index(index)
        bkd, blocks = self._bkd, self._blocks
        ctg = blocks.target_obs_covariance(index)
        gain_t = bkd.solve(self.obs_covariance(), ctg.T)
        mean = blocks.target_mean(index) + bkd.dot(gain_t.T, data - blocks.obs_mean())
        cov = blocks.target_covariance(index) - bkd.dot(ctg, gain_t)
        return mean, cov

    def select(self, rows: Sequence[int]) -> "JointGaussian[Array]":
        """The joint of the targets and the observations in ``rows``.

        Marginalizing the other observations keeps the rows and columns
        of ``rows`` in the observation blocks and the noise. The result
        is positive semidefinite whenever this one is, so it is not
        repaired again.

        Parameters
        ----------
        rows : Sequence[int]
            Observations to keep, in order, without repeats.
        """
        nobs = self.nobs()
        selected = list(rows)
        if len(selected) == 0:
            raise ValueError("rows is empty")
        if len(set(selected)) != len(selected):
            raise ValueError(f"rows has repeats: {selected}")
        bad = [row for row in selected if not 0 <= row < nobs]
        if bad:
            raise ValueError(f"rows {bad} are outside the {nobs} observations")
        blocks = self._blocks
        ntarget = sum(blocks.target_sizes())
        keep = list(range(ntarget)) + [ntarget + row for row in selected]
        sub_blocks = DenseBlocks(
            blocks.mean()[keep],
            blocks.covariance()[keep][:, keep],
            blocks.target_sizes(),
            len(selected),
            self._bkd,
            blocks.nsamples(),
        )
        noise_cov = self._noise.covariance()[selected][:, selected]
        sub_noise = DenseCholeskyCovarianceOperator(noise_cov, self._bkd)
        return JointGaussian(sub_blocks, sub_noise, _AlreadyChecked())

    def _check_index(self, index: int) -> None:
        ntargets = len(self.target_sizes())
        if not 0 <= index < ntargets:
            raise ValueError(f"target {index} does not exist; there are {ntargets}")
