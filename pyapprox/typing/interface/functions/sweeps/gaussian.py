"""Gaussian parameter sweeps.

Parameter sweeps along directions in Gaussian-distributed parameter space.
"""

from typing import Callable, Generic, Optional, Tuple

import numpy as np

from pyapprox.typing.util.backends.protocols import Array, Backend


class GaussianParameterSweeper(Generic[Array]):
    """Parameter sweeper for Gaussian-distributed parameters.

    Generates parameter sweeps along random orthogonal directions
    in a Gaussian parameter space. Each sweep samples points along a line
    in standard normal space, transformed to the parameter space via
    the covariance square root operator.

    Parameters
    ----------
    mean : Array
        Shape (nvars,) - mean of the Gaussian distribution.
    cov_sqrt_op : Callable[[Array], Array]
        Operator that applies the square root of the covariance matrix:
        samples = mean + cov_sqrt_op(standard_normal_samples)
    sweep_radius : float
        Radius of the sweep in standard normal units (number of std devs).
    nsamples_per_sweep : int
        Number of samples in each sweep.
    bkd : Backend[Array]
        Backend for array operations.

    Examples
    --------
    >>> import numpy as np
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> mean = bkd.asarray([0.0, 0.0])
    >>> L = bkd.asarray([[1.0, 0.0], [0.5, 0.866]])  # Cholesky factor
    >>> cov_sqrt_op = lambda x: L @ x
    >>> sweeper = GaussianParameterSweeper(
    ...     mean, cov_sqrt_op, sweep_radius=3.0, nsamples_per_sweep=50, bkd=bkd
    ... )
    >>> samples = sweeper.rvs(nsweeps=5)
    """

    def __init__(
        self,
        mean: Array,
        cov_sqrt_op: Callable[[Array], Array],
        sweep_radius: float,
        nsamples_per_sweep: int,
        bkd: Backend[Array],
    ) -> None:
        self._bkd = bkd
        self._mean = mean
        self._cov_sqrt_op = cov_sqrt_op
        self._sweep_radius = sweep_radius
        self._nsamples_per_sweep = nsamples_per_sweep
        self._nvars = mean.shape[0]

        self._rotation_mat: Optional[Array] = None
        self._canonical_active_samples: Optional[Array] = None
        self._samples: Optional[Array] = None

    def bkd(self) -> Backend[Array]:
        """Return the backend used for array operations."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of input variables."""
        return self._nvars

    def nsamples_per_sweep(self) -> int:
        """Return the number of samples in each sweep."""
        return self._nsamples_per_sweep

    def set_nsamples_per_sweep(self, nsamples_per_sweep: int) -> None:
        """Set the number of samples per sweep."""
        self._nsamples_per_sweep = nsamples_per_sweep

    def random_rotation_matrices(self, nsweeps: int) -> Array:
        """Generate random orthogonal rotation vectors.

        Parameters
        ----------
        nsweeps : int
            Number of sweep directions to generate.

        Returns
        -------
        Array
            Shape (nvars, nsweeps) - rotation vectors for each sweep.
        """
        ii = 0
        mats = []
        while ii * self._nvars < nsweeps:
            A = np.random.normal(0, 1, (self._nvars, nsweeps))
            Q, R = np.linalg.qr(A)
            ncols = min(self._nvars, nsweeps - ii * self._nvars)
            mats.append(self._bkd.asarray(Q[:, :ncols]))
            ii += 1
        return self._bkd.hstack(mats)

    def set_sweep_rotation_matrices(self, mat: Array) -> None:
        """Set the rotation matrix for sweeps.

        Parameters
        ----------
        mat : Array
            Shape (nvars, nsweeps) - each column is a rotation vector.
        """
        if mat.shape[0] != self._nvars:
            raise ValueError(
                f"Rotation matrix must have {self._nvars} rows, got {mat.shape[0]}"
            )
        self._rotation_mat = mat

    def sweep_bounds(self, rotation_vec: Array) -> Tuple[float, float]:
        """Return the sweep bounds.

        For Gaussian sweeps, bounds are [-sweep_radius, sweep_radius].

        Parameters
        ----------
        rotation_vec : Array
            Shape (nvars, 1) - rotation vector (unused for Gaussian).

        Returns
        -------
        Tuple[float, float]
            (-sweep_radius, sweep_radius) bounds.
        """
        return -self._sweep_radius, self._sweep_radius

    def canonical_sweep_samples(self, rotation_vec: Array) -> Array:
        """Generate 1D canonical samples for a sweep direction.

        Parameters
        ----------
        rotation_vec : Array
            Shape (nvars, 1) - rotation vector (sweep direction).

        Returns
        -------
        Array
            Shape (1, nsamples_per_sweep) - 1D sweep samples.
        """
        y_lb, y_ub = self.sweep_bounds(rotation_vec)
        return self._bkd.reshape(
            self._bkd.linspace(y_lb, y_ub, self._nsamples_per_sweep),
            (1, -1),
        )

    def _transform_to_parameter_space(self, standard_samples: Array) -> Array:
        """Transform standard normal samples to parameter space.

        Parameters
        ----------
        standard_samples : Array
            Shape (nvars, nsamples) - samples in standard normal space.

        Returns
        -------
        Array
            Shape (nvars, nsamples) - samples in parameter space.
        """
        mean_col = self._bkd.reshape(self._mean, (-1, 1))
        return mean_col + self._cov_sqrt_op(standard_samples)

    def rvs(self, nsweeps: int) -> Array:
        """Generate parameter sweep samples.

        Parameters
        ----------
        nsweeps : int
            Number of sweeps to generate.

        Returns
        -------
        Array
            Shape (nvars, nsamples_per_sweep * nsweeps) - all sweep samples.
            Samples from each sweep are stored consecutively.
        """
        if self._rotation_mat is None:
            self._rotation_mat = self.random_rotation_matrices(nsweeps)

        standard_samples = self._bkd.zeros(
            (self._nvars, self._nsamples_per_sweep * nsweeps)
        )
        self._canonical_active_samples = self._bkd.zeros(
            (nsweeps, self._nsamples_per_sweep)
        )

        for ii in range(nsweeps):
            rotation_vec = self._rotation_mat[:, ii : ii + 1]
            y_samples = self.canonical_sweep_samples(rotation_vec)
            self._canonical_active_samples[ii, :] = self._bkd.reshape(
                y_samples, (-1,)
            )

            start = ii * self._nsamples_per_sweep
            end = (ii + 1) * self._nsamples_per_sweep
            standard_samples[:, start:end] = rotation_vec @ y_samples

        self._samples = self._transform_to_parameter_space(standard_samples)
        return self._samples

    def canonical_active_samples(self) -> Array:
        """Return the canonical (1D) sweep samples.

        Returns
        -------
        Array
            Shape (nsweeps, nsamples_per_sweep) - 1D samples for each sweep.
        """
        if self._canonical_active_samples is None:
            raise RuntimeError("Must call rvs() first")
        return self._canonical_active_samples

    def samples(self) -> Array:
        """Return the generated sweep samples.

        Returns
        -------
        Array
            Shape (nvars, nsamples_total) - all sweep samples.
        """
        if self._samples is None:
            raise RuntimeError("Must call rvs() first")
        return self._samples

    def __repr__(self) -> str:
        return (
            f"GaussianParameterSweeper(nvars={self._nvars}, "
            f"sweep_radius={self._sweep_radius}, "
            f"nsamples_per_sweep={self._nsamples_per_sweep})"
        )
