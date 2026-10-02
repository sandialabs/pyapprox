r"""Dense covariance blocks of stacked targets and observations.

The stacked vector is :math:`\chi = (t_1, \dots, t_r, g)`, with target
blocks :math:`t_k` in order followed by the noise-free observations
:math:`g`. ``DenseBlocks`` holds its mean :math:`\mu_\chi` and covariance
:math:`\Gamma_{\chi\chi}` and slices the blocks out on request.
"""

from typing import Generic, Optional, Sequence

from pyapprox.util.backends.protocols import Array, Backend


class DenseBlocks(Generic[Array]):
    """Mean and full covariance of the stacked targets and observations.

    Parameters
    ----------
    mean : Array
        Mean of the stacked vector. Shape: (nstacked, 1)
    covariance : Array
        Covariance of the stacked vector. Shape: (nstacked, nstacked)
    target_sizes : Sequence[int]
        Rows of each target block, in stacking order.
    nobs : int
        Number of observations, the last rows of the stacked vector.
    bkd : Backend[Array]
        Computational backend.
    nsamples : int, optional
        Number of samples the moments were estimated from. None if exact.
    """

    def __init__(
        self,
        mean: Array,
        covariance: Array,
        target_sizes: Sequence[int],
        nobs: int,
        bkd: Backend[Array],
        nsamples: Optional[int] = None,
    ) -> None:
        sizes = tuple(int(size) for size in target_sizes)
        if any(size < 1 for size in sizes) or nobs < 1:
            raise ValueError(
                f"block sizes must be positive, got {sizes} and nobs={nobs}"
            )
        nstacked = sum(sizes) + nobs
        if tuple(mean.shape) != (nstacked, 1):
            raise ValueError(
                f"mean must have shape ({nstacked}, 1), got {tuple(mean.shape)}"
            )
        if tuple(covariance.shape) != (nstacked, nstacked):
            raise ValueError(
                f"covariance must have shape ({nstacked}, {nstacked}), got "
                f"{tuple(covariance.shape)}"
            )
        self._mean = mean
        self._cov = covariance
        self._sizes = sizes
        self._nobs = nobs
        self._bkd = bkd
        self._nsamples = nsamples
        starts = [sum(sizes[:ii]) for ii in range(len(sizes))]
        self._target_slices = [
            slice(start, start + size) for start, size in zip(starts, sizes)
        ]
        self._obs_slice = slice(nstacked - nobs, nstacked)

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def target_sizes(self) -> tuple[int, ...]:
        """Number of rows in each target block."""
        return self._sizes

    def nobs(self) -> int:
        """Number of observations."""
        return self._nobs

    def nsamples(self) -> Optional[int]:
        """Number of samples behind the blocks, or None if exact."""
        return self._nsamples

    def mean(self) -> Array:
        """Mean of the stacked vector. Shape: (nstacked, 1)"""
        return self._mean

    def covariance(self) -> Array:
        """Covariance of the stacked vector. Shape: (nstacked, nstacked)"""
        return self._cov

    def target_mean(self, index: int) -> Array:
        """Mean of target block ``index``. Shape: (n_t, 1)"""
        return self._mean[self._target_slices[index]]

    def obs_mean(self) -> Array:
        """Mean of the observations. Shape: (nobs, 1)"""
        return self._mean[self._obs_slice]

    def target_covariance(self, index: int) -> Array:
        """``Gamma_tt`` of block ``index``. Shape: (n_t, n_t)"""
        rows = self._target_slices[index]
        return self._cov[rows, rows]

    def target_obs_covariance(self, index: int) -> Array:
        """``Gamma_tg`` of block ``index``. Shape: (n_t, nobs)"""
        return self._cov[self._target_slices[index], self._obs_slice]

    def obs_covariance(self) -> Array:
        """``Gamma_gg``, without noise. Shape: (nobs, nobs)"""
        return self._cov[self._obs_slice, self._obs_slice]

    def min_eigenvalue(self) -> Array:
        """Smallest eigenvalue of the stacked covariance. Shape: (1,)

        Negative when a rule with negative weights, such as a sparse grid,
        produced an indefinite estimate.
        """
        return self._bkd.reshape(self._bkd.min(self._bkd.eigvalsh(self._cov)), (1,))
