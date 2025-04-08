import math
from typing import Tuple
from abc import ABC, abstractmethod

import numpy as np

from pyapprox.util.transforms import Transform
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.variables.transforms import OperatorBasedGaussianTransform


class ParameterSweeper(ABC):
    def __init__(
        self,
        nvars: int,
        trans: Transform,
        nsamples_per_sweep: int,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Parameters
        ----------
        nsamples_per_sweep : int
            The number of samples in each of the parameter sweeps
        """
        self._bkd = backend
        self._nvars = nvars
        self.set_nsamples_per_sweep(nsamples_per_sweep)
        self.set_transform(trans)

    def set_transform(self, trans: Transform):
        if not isinstance(trans, Transform):
            raise ValueError("trans must be an instance of Transform")
        self._trans = trans

    def set_sweep_rotation_matrices(self, mat: Array):
        """
        Parameters
        ----------
        mat : Array (nvars, nsweeps)
            Each row contains the rotation vector of each sweep.
        """

        if mat.shape[0] != self._nvars:
            raise ValueError("Rotation matrix has the wrong shape")
        self._rotation_mat = mat

    def set_nsamples_per_sweep(self, nsamples_per_sweep: int):
        self._nsamples_per_sweep = nsamples_per_sweep

    def random_rotation_matrices(self, nsweeps: int) -> Array:
        # can only generate a maximum of nvar sweeps for each rotation matrix
        # to generate more create different rotation matrices until enough
        # sweeps are generated
        ii = 0
        mats = []
        while ii * self._nvars < nsweeps:
            A = np.random.normal(0, 1, (self._nvars, nsweeps))
            Q, R = np.linalg.qr(A)
            mats.append(Q[:, : min(self._nvars, nsweeps - ii * self._nvars)])
            ii += 1
        return self._bkd.hstack(mats)

    @abstractmethod
    def sweep_bounds(self, Wmat: Array) -> Tuple[float, float]:
        raise NotImplementedError

    def set_trans(self, trans: Transform):
        if not isinstance(trans, Transform):
            raise ValueError("trans must be an instance of Transform")
        self._trans = trans

    def canonical_sweep_samples(self, Wmat: Array) -> Array:
        y_lb, y_ub = self.sweep_bounds(Wmat)
        return self._bkd.linspace(y_lb, y_ub, self._nsamples_per_sweep)[
            None, :
        ]

    # y = np.random.uniform(y_lb, y_ub, (1, nsamples_per_sweep))

    def rvs(self, nsweeps: int) -> Tuple[Array, Array]:
        """
        Parameters
        ----------
        nsweeps : int
            The number of sweeps

        Returns
        -------
        samples : Array (nvars, nsamples_per_sweep * nsweeps)
            The samples in the D-dimensional space. Each sweep is listed
            consecutivelty. That is nsamples_per_sweep for first sweep
            are the first rows, then the second sweep are the next set of
            nsamples_per_sweep rows, and so on.

        active_samples : Aarray (nsweeps, nsamples_per_sweep)
            The univariate samples of the parameter sweeps. These samples are
            for normalized hypercubes [-1,1]^D.
        """
        if not hasattr(self, "_rotation_mat"):
            self._rotation_mat = self.random_rotation_matrices(nsweeps)
        canonical_samples = self._bkd.empty(
            (self._nvars, self._nsamples_per_sweep * nsweeps)
        )
        self._canonical_active_samples = self._bkd.empty(
            (nsweeps, self._nsamples_per_sweep)
        )

        for ii in range(nsweeps):
            Wmat = self._rotation_mat[:, ii : ii + 1]
            # find approximate upper and lower bounds for active variable
            # define samples in sweep inside approximate upper and lower bounds
            y_samples = self.canonical_sweep_samples(Wmat)
            self._canonical_active_samples[ii, :] = y_samples
            canonical_samples[
                :,
                ii
                * self._nsamples_per_sweep : (ii + 1)
                * self._nsamples_per_sweep,
            ] = (
                Wmat @ y_samples
            )
        self._samples = self._trans.map_from_canonical(canonical_samples)
        return self._samples

    def plot_single_qoi_sweep(
        self, sweep_vals: Array, sweep_id: int, ax, **plot_kwargs
    ):
        if not hasattr(self, "_samples"):
            raise RuntimeError("Must call sweep_samples first")
        nsweeps = self._canonical_active_samples.shape[0]
        print(self._canonical_active_samples.shape)
        if sweep_vals.shape != (self._nsamples_per_sweep * nsweeps,):
            raise ValueError(
                "sweep_vals has the wrong shape {0} should be {1}".format(
                    sweep_vals.shape, (self._nsamples_per_sweep * nsweeps,)
                )
            )
        ax.plot(
            self._canonical_active_samples[sweep_id],
            sweep_vals[
                sweep_id
                * self._nsamples_per_sweep : (sweep_id + 1)
                * self._nsamples_per_sweep
            ],
            **plot_kwargs,
        )


class BoundedParameterSweeper(ParameterSweeper):
    """
    Parameter sweep over a bounded hypercube.

    All variables must be bounded
    """

    def sweep_bounds(self, Wmat: Array) -> Tuple[float, float]:
        maxdist = math.sqrt(self._nvars * 4)
        y = self._bkd.linspace(-maxdist / 2.0, maxdist / 2.0, 1000)[None, :]
        x = Wmat @ y
        II = self._bkd.where(
            self._bkd.all(x >= -1, axis=0) & self._bkd.all(x <= 1, axis=0)
        )[0]
        y_lb = y[0, II[0]]
        y_ub = y[0, II[-1]]
        return y_lb, y_ub


class GaussianParameterSweeper(ParameterSweeper):
    def __init__(
        self,
        mean: Array,
        cov_sqrt_op: callable,
        sweep_radius: int,
        nsamples_per_sweep: int,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Parameters
        ----------
        seep_radius : float
            The radius of the parameter sweep as a multiple of
            one standard deviation of the standard normal

        covariance_sqrt : callable
            correlated_samples = covariance_sqrt(stdnormal_samples)
            An operator that applies the sqrt of the Gaussian covariance to a
            set of vectors. Useful for large scale applications.
        """
        self._sweep_radius = sweep_radius
        trans = OperatorBasedGaussianTransform(mean, cov_sqrt_op, None)
        super().__init__(
            mean.shape[0], trans, nsamples_per_sweep, backend=backend
        )

    def sweep_bounds(self, Wmat: Array) -> Tuple[float, float]:
        return -self._sweep_radius, self._sweep_radius
