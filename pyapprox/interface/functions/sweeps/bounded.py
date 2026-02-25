"""Bounded parameter sweeps.

Parameter sweeps over bounded hypercube domains.
"""

import math
from typing import Generic, Optional, Tuple

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.probability.transforms.affine import AffineTransform


class BoundedParameterSweeper(Generic[Array]):
    """Parameter sweeper for bounded hypercube domains.

    Generates parameter sweeps along random orthogonal directions
    in a bounded hypercube. Each sweep samples points along a line
    in the input space.

    Parameters
    ----------
    bounds : Array
        Shape (nvars, 2) - lower and upper bounds for each variable.
    nsamples_per_sweep : int
        Number of samples in each sweep.
    bkd : Backend[Array]
        Backend for array operations.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> bounds = bkd.asarray([[0.0, 1.0], [0.0, 2.0], [0.0, 1.0]])
    >>> sweeper = BoundedParameterSweeper(bounds, nsamples_per_sweep=50, bkd=bkd)
    >>> samples = sweeper.rvs(nsweeps=5)  # Generate 5 sweeps
    >>> # samples has shape (3, 250) - 50 samples per sweep * 5 sweeps
    """

    def __init__(
        self,
        bounds: Array,
        nsamples_per_sweep: int,
        bkd: Backend[Array],
    ) -> None:
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (nvars, 2)")
        self._bkd = bkd
        self._bounds = bounds
        self._nvars = bounds.shape[0]
        self._nsamples_per_sweep = nsamples_per_sweep

        # Create affine transform from canonical [-1, 1]^d to bounds
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        loc = (lb + ub) / 2.0
        scale = (ub - lb) / 2.0
        self._transform = AffineTransform(loc, scale, bkd)

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

        Each column is a unit vector defining a sweep direction.
        Uses QR decomposition of random matrices to get orthogonal directions.

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
        """Compute bounds for a sweep in the canonical [-1,1]^d hypercube.

        Finds the extent of the line defined by rotation_vec that lies
        within the unit hypercube.

        Parameters
        ----------
        rotation_vec : Array
            Shape (nvars, 1) - rotation vector (sweep direction).

        Returns
        -------
        Tuple[float, float]
            (lower, upper) bounds for the sweep parameter.
        """
        # Maximum extent in any direction from origin is sqrt(nvars)
        maxdist = math.sqrt(self._nvars * 4)
        y = self._bkd.linspace(-maxdist / 2.0, maxdist / 2.0, 1000)
        y = self._bkd.reshape(y, (1, -1))
        x = rotation_vec @ y  # Shape: (nvars, 1000)

        # Find points inside the canonical hypercube [-1, 1]^d
        inside_mask = self._bkd.all_array(x >= -1, axis=0) & self._bkd.all_array(
            x <= 1, axis=0
        )
        indices = self._bkd.where(inside_mask)[0]

        if len(indices) == 0:
            raise RuntimeError("No valid sweep bounds found")

        y_lb = float(self._bkd.to_numpy(y[0, indices[0]]))
        y_ub = float(self._bkd.to_numpy(y[0, indices[-1]]))
        return y_lb, y_ub

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

        canonical_samples = self._bkd.zeros(
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
            canonical_samples[:, start:end] = rotation_vec @ y_samples

        self._samples = self._transform.map_from_canonical(canonical_samples)
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
            f"BoundedParameterSweeper(nvars={self._nvars}, "
            f"nsamples_per_sweep={self._nsamples_per_sweep})"
        )
