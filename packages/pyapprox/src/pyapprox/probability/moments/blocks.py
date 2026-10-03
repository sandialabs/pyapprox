r"""Dense covariance blocks of stacked targets and observations.

The stacked vector is :math:`\chi = (t_1, \dots, t_r, g)`, with target
blocks :math:`t_k` in order followed by the noise-free observations
:math:`g`. ``DenseBlocks`` holds its mean :math:`\mu_\chi` and covariance
:math:`\Gamma_{\chi\chi}` and slices the blocks out on request.
"""

from typing import Generic, Mapping, Optional, Sequence

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
    exact_targets : Sequence[bool], optional
        Whether each target's own mean and covariance are exact rather
        than estimated. Default: all exact when ``nsamples`` is None, none
        otherwise. Log-determinant criteria refuse a sampled target with
        ``nsamples <= n_t + 1``, where the estimate collapses.
    """

    def __init__(
        self,
        mean: Array,
        covariance: Array,
        target_sizes: Sequence[int],
        nobs: int,
        bkd: Backend[Array],
        nsamples: Optional[int] = None,
        exact_targets: Optional[Sequence[bool]] = None,
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
        if exact_targets is None:
            exact_targets = [nsamples is None] * len(sizes)
        if len(exact_targets) != len(sizes):
            raise ValueError(
                f"exact_targets has {len(exact_targets)} entries for "
                f"{len(sizes)} targets"
            )
        self._exact = tuple(bool(flag) for flag in exact_targets)
        starts = [sum(sizes[:ii]) for ii in range(len(sizes))]
        self._target_slices = [
            slice(start, start + size) for start, size in zip(starts, sizes)
        ]
        self._obs_slice = slice(nstacked - nobs, nstacked)

    @classmethod
    def from_linear_model(
        cls,
        obs_mat: Array,
        mean: Array,
        covariance: Array,
        target_mats: Sequence[Array],
        bkd: Backend[Array],
    ) -> "DenseBlocks[Array]":
        r"""Exact blocks of linear maps of a Gaussian input.

        For :math:`\xi \sim N(\mu, \Gamma)`, observations :math:`g = A\xi`
        and targets :math:`t_k = A_k \xi`, the stacked map
        :math:`M = (A_1; \dots; A_r; A)` gives mean :math:`M\mu` and
        covariance :math:`M \Gamma M^\top`, with no samples.

        Parameters
        ----------
        obs_mat : Array
            Observation map ``A``. Shape: (nobs, nvars)
        mean : Array
            Input mean. Shape: (nvars, 1)
        covariance : Array
            Input covariance. Shape: (nvars, nvars)
        target_mats : Sequence[Array]
            Target maps ``A_k``, each of shape (n_k, nvars).
        bkd : Backend[Array]
            Computational backend.
        """
        nvars = covariance.shape[0]
        for name, mat in [("obs_mat", obs_mat)] + [
            (f"target_mats[{ii}]", mat) for ii, mat in enumerate(target_mats)
        ]:
            if mat.ndim != 2 or mat.shape[1] != nvars:
                raise ValueError(
                    f"{name} must have shape (*, {nvars}), got {tuple(mat.shape)}"
                )
        stacked = bkd.vstack(list(target_mats) + [obs_mat])
        return cls(
            bkd.dot(stacked, mean),
            bkd.dot(bkd.dot(stacked, covariance), stacked.T),
            [int(mat.shape[0]) for mat in target_mats],
            int(obs_mat.shape[0]),
            bkd,
        )

    def with_known_targets(
        self, overrides: Mapping[int, tuple[Array, Array]]
    ) -> "DenseBlocks[Array]":
        """Replace chosen targets' mean and covariance with known values.

        Use when a target's own moments are known exactly, for example a
        parameter target whose prior covariance is given, while its
        cross-covariance with the observations is estimated. Cross
        covariances are kept, so the result may be indefinite; repair it
        if so.

        Parameters
        ----------
        overrides : Mapping[int, tuple[Array, Array]]
            Target index to ``(mean (n_k, 1), covariance (n_k, n_k))``.
        """
        mean, cov = self._bkd.copy(self._mean), self._bkd.copy(self._cov)
        exact = list(self._exact)
        for index, (known_mean, known_cov) in overrides.items():
            if not 0 <= index < len(self._sizes):
                raise ValueError(
                    f"target {index} does not exist; there are {len(self._sizes)}"
                )
            size = self._sizes[index]
            if tuple(known_mean.shape) != (size, 1) or tuple(known_cov.shape) != (
                size,
                size,
            ):
                raise ValueError(
                    f"target {index} needs shapes ({size}, 1) and ({size}, {size}), "
                    f"got {tuple(known_mean.shape)} and {tuple(known_cov.shape)}"
                )
            rows = self._target_slices[index]
            mean[rows] = known_mean
            cov[rows, rows] = known_cov
            exact[index] = True
        return DenseBlocks(
            mean, cov, self._sizes, self._nobs, self._bkd, self._nsamples, exact
        )

    def bkd(self) -> Backend[Array]:
        """Get the computational backend."""
        return self._bkd

    def target_sizes(self) -> tuple[int, ...]:
        """Number of rows in each target block."""
        return self._sizes

    def nobs(self) -> int:
        """Number of observations."""
        return self._nobs

    def exact_targets(self) -> tuple[bool, ...]:
        """Whether each target's own mean and covariance are exact."""
        return self._exact

    def target_is_exact(self, index: int) -> bool:
        """Whether target ``index``'s own mean and covariance are exact."""
        return self._exact[index]

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
