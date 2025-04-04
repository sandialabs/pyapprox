from abc import ABC, abstractmethod
import math

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array


class Transform(ABC):
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        self._bkd = backend

    @abstractmethod
    def map_from_canonical(self, values: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def map_to_canonical(self, values: Array) -> Array:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "{0}(bkd={1})".format(
            self.__class__.__name__, self._bkd.__name__
        )


class IdentityTransform(Transform):
    def map_from_canonical(self, values: Array) -> Array:
        return values

    def map_to_canonical(self, values: Array) -> Array:
        return values

    def derivatives_from_canonical(
        self, canonical_derivs: Array, order: int = 1
    ) -> Array:
        return canonical_derivs

    def derivatives_to_canonical(self, derivs: Array, order=1) -> Array:
        return derivs

    def map_from_canonical_1d(self, values: Array, dim_id: int) -> Array:
        return values

    def map_to_canonical_1d(self, values: Array, dim_id: int) -> Array:
        return values


class AffineBoundedTransform(Transform):
    def __init__(
        self,
        canonical_ranges: Array,
        user_ranges: Array,
        bkd: LinAlgMixin = NumpyLinAlgMixin,
    ):
        super().__init__(bkd)
        if len(user_ranges) != len(canonical_ranges):
            raise ValueError(
                "user_ranges and canonical_ranges have the wrong shape"
            )
        self._canonical_ranges = self._bkd.asarray(canonical_ranges)
        self._user_ranges = self._bkd.asarray(user_ranges)
        self._nvars = len(self._user_ranges) // 2

    def _map_hypercube_samples(
        self, current_samples: Array, current_ranges: Array, new_ranges: Array
    ) -> Array:
        # no error checking or notion of active_vars
        clbs, cubs = current_ranges[0::2], current_ranges[1::2]
        nlbs, nubs = new_ranges[0::2], new_ranges[1::2]
        return (
            (current_samples.T - clbs) / (cubs - clbs) * (nubs - nlbs) + nlbs
        ).T

    def map_from_canonical(self, canonical_samples: Array) -> Array:

        return self._map_hypercube_samples(
            canonical_samples, self._canonical_ranges, self._user_ranges
        )

    def map_to_canonical(self, user_samples: Array) -> Array:
        return self._map_hypercube_samples(
            user_samples, self._user_ranges, self._canonical_ranges
        )

    def nvars(self) -> int:
        return self._nvars

    def map_from_canonical_1d(
        self, canonical_samples: Array, dim_id: int
    ) -> Array:
        return self._map_hypercube_samples(
            canonical_samples,
            self._canonical_ranges[2 * dim_id : 2 * (dimid + 1)],
            self._user_ranges[2 * dim_id : 2 * (dimid + 1)],
        )

    def map_to_canonical_1d(self, user_samples: Array, dim_id: int) -> Array:
        return self._map_hypercube_samples(
            user_samples,
            self._user_ranges[2 * dim_id : 2 * (dimid + 1)],
            self._canonical_ranges[2 * dim_id : 2 * (dimid + 1)],
        )


class StandardDeviationTransform(Transform):
    def __init__(self, trans=False, backend=None):
        # todo: samples and values should always be (nvars, nsamples)
        # where nvars=nqois but currently values is transpose of this
        # so trans=True is used to deal with this case
        super().__init__(backend)
        self._trans = trans
        self._means = None
        self._stdevs = None

    def map_to_canonical(self, values: Array) -> Array:
        # Assume that first call to map_to_canonical defines the mean
        # and standar deviation
        if self._means is None:
            if not self._trans:
                self._means = self._bkd.mean(values, axis=1)[:, None]
                self._stdevs = self._bkd.std(values, axis=1, ddof=1)[:, None]
            else:
                self._means = self._bkd.mean(values, axis=0)[:, None]
                self._stdevs = self._bkd.std(values, axis=0, ddof=1)[:, None]
        canonical_values = (values - self._means) / self._stdevs
        return canonical_values

    def map_from_canonical(self, canonical_values: Array) -> Array:
        values = canonical_values * self._stdevs + self._means
        return values

    def map_from_canonical_1d(
        self, canonical_values: Array, dim_id: int
    ) -> Array:
        values = canonical_values * self._stdevs[dim_id] + self._means[dim_id]
        return values

    def map_to_canonical_1d(self, user_values: Array, dim_id: int) -> Array:
        # assumes that map_to canonical has already been called and
        # used to define self._means, self._stdevs
        canonical_values = (values - self._means[dim_id]) / self._stdevs[
            dim_id
        ]
        return canonical_values


class NSphereCoordinateTransform(Transform):
    def map_to_nsphere(self, samples):
        nvars, nsamples = samples.shape
        r = self._bkd.sqrt((samples**2).sum(axis=0))
        psi = self._bkd.full(samples.shape, 0.0)
        psi[0] = self._bkd.copy(r)
        psi[1] = self._bkd.arccos(samples[0] / r)
        for ii in range(2, nvars):
            denom = self._bkd.copy(r)
            for jj in range(ii - 1):
                denom *= self._bkd.sin(psi[jj + 1])
            psi[ii] = self._bkd.arccos(samples[ii - 1] / denom)
        psi[-1][samples[-1] < 0] = 2 * math.pi - psi[-1][samples[-1] < 0]
        return psi

    def map_from_nsphere(self, psi):
        nvars, nsamples = psi.shape
        r = self._bkd.copy(psi[0])
        samples = self._bkd.full(psi.shape, 0.0)
        samples[0] = r * self._bkd.cos(psi[1])
        for ii in range(1, nvars):
            samples[ii, :] = self._bkd.copy(r)
            for jj in range(ii):
                samples[ii] *= self._bkd.sin(psi[jj + 1])
            if ii != nvars - 1:
                samples[ii] *= self._bkd.cos(psi[ii + 1])
        return samples

    def map_to_canonical(self, psi):
        return self.map_from_nsphere(psi)

    def map_from_canonical(self, canonical_samples):
        return self.map_to_nsphere(canonical_samples)


class SphericalCorrelationTransform(Transform):
    def __init__(self, noutputs, backend=None):
        super().__init__(backend)
        self.noutputs = noutputs
        self.ntheta = (self.noutputs * (self.noutputs + 1)) // 2
        self._theta_indices = self._bkd.full((self.ntheta, 2), -1, dtype=int)
        self._theta_indices[: self.noutputs, 0] = self._bkd.arange(
            self.noutputs
        )
        self._theta_indices[: self.noutputs, 1] = 0
        for ii in range(1, noutputs):
            for jj in range(1, ii + 1):
                # indices[ii, jj] = (
                #     self.noutputs+((ii-1)*(ii))//2 + (jj-1))
                self._theta_indices[
                    self.noutputs + ((ii - 1) * (ii)) // 2 + (jj - 1)
                ] = self._bkd.asarray([ii, jj])
        self.nsphere_trans = NSphereCoordinateTransform(backend=backend)
        # unconstrained formulation does not seem unique.
        self._unconstrained = False

    def get_spherical_bounds(self):
        inf = self._bkd.inf()
        if not self._unconstrained:
            # l_{i1} > 0,  i = 0,...,noutputs-1
            # l_{ij} in (0, math.pi),    i = 1,...,noutputs-1, j=1,...,i
            eps = 0
            bounds = self._bkd.stack(
                [self._bkd.asarray([eps, inf]) for ii in range(self.noutputs)],
                axis=0,
            )
            other_bounds = self._bkd.stack(
                [
                    self._bkd.asarray([eps, math.pi - eps])
                    for ii in range(self.noutputs, self.ntheta)
                ],
                axis=0,
            )
            bounds = self._bkd.vstack((bounds, other_bounds))
            return bounds

        return self._bkd.stack(
            [self._bkd.asarray([-inf, inf]) for ii in range(self.theta)]
        )

    def map_cholesky_to_spherical(self, L):
        psi = self._bkd.empty(L.shape)
        psi[0, 0] = L[0, 0]
        for ii in range(1, self.noutputs):
            psi[ii, : ii + 1] = self.nsphere_trans.map_to_nsphere(
                L[ii : ii + 1, : ii + 1].T
            ).T
        return psi

    def map_spherical_to_unconstrained_theta(self, psi):
        theta = self._bkd.empty(self.ntheta)
        theta[: self.noutputs] = self._bkd.log(psi[:, 0])
        psi_flat = psi[
            self._theta_indices[self.noutputs :, 0],
            self._theta_indices[self.noutputs :, 1],
        ]
        theta[self.noutputs :] = self._bkd.log(psi_flat / (math.pi - psi_flat))
        return theta

    def map_spherical_to_theta(self, psi):
        if self._unconstrained:
            return self.map_spherical_to_unconstrained_theta(psi)
        return psi[self._theta_indices[:, 0], self._theta_indices[:, 1]]

    def map_from_cholesky(self, L):
        psi = self.map_cholesky_to_spherical(L)
        return self.map_spherical_to_theta(psi)

    def map_unconstrained_theta_to_spherical(self, theta):
        psi = self._bkd.full((self.noutputs, self.noutputs), 0.0)
        # psi[ii, :] are radius of hypersphere of increasing dimension
        # all other psi are angles
        exp_theta = self._bkd.exp(theta)
        psi[:, 0] = exp_theta[: self.noutputs]
        psi[
            self._theta_indices[self.noutputs :, 0],
            self._theta_indices[self.noutputs :, 1],
        ] = (
            exp_theta[self.noutputs :]
            * math.pi
            / (1 + exp_theta[self.noutputs :])
        )
        # cnt = self.noutputs
        # for ii in range(1, self.noutputs):
        #     for jj in range(1, ii+1):
        #         exp_theta = exp(theta[cnt])
        #         psi[ii, jj] = exp_theta*math.pi/(1+exp_theta)
        #         cnt += 1
        return psi

    def map_theta_to_spherical(self, theta):
        if theta.ndim != 1:
            raise ValueError("theta must be 1d array")
        if self._unconstrained:
            psi = self.map_unconstrained_theta_to_spherical(theta)
            return self.map_spherical_to_cholesky(psi)
        psi = self._bkd.full((self.noutputs, self.noutputs), 0.0)
        psi[self._theta_indices[:, 0], self._theta_indices[:, 1]] = theta
        return psi

    def map_spherical_to_cholesky(self, psi):
        L_factor = self._bkd.full((self.noutputs, self.noutputs), 0.0)
        L_factor[0, 0] = psi[0, 0]
        for ii in range(1, self.noutputs):
            L_factor[ii : ii + 1, : ii + 1] = (
                self.nsphere_trans.map_from_nsphere(
                    psi[ii : ii + 1, : ii + 1].T
                ).T
            )
        return L_factor

    def map_to_cholesky(self, theta):
        psi = self.map_theta_to_spherical(theta)
        return self.map_spherical_to_cholesky(psi)

    def map_to_canonical(self, samples):
        return self._map_from_cholesky(samples)

    def map_from_canonical(self, canonical_samples):
        return self._map_to_cholesky(canonical_samples)


class UnivariateAffineTransform(Transform):
    def __init__(self, loc, scale, enforce_bounds=False, backend=None):
        super().__init__(backend)
        self._loc = loc
        self._scale = scale
        self._enforce_bounds = enforce_bounds
        # consider passing in bounds optionally. If provided
        # then check bounds is called. Better than determining
        # bounds from loc, scale which assumes can domain is [-1, 1]

    def _check_bounds(self, user_samples):
        if not self._enforce_bounds:
            return

        bounds = [self._loc - self._scale, self._loc + self._scale]
        if self._bkd.any(user_samples < bounds[0]) or self._bkd.any(
            user_samples > bounds[1]
        ):
            print(user_samples)
            raise ValueError(f"Sample outside the bounds {bounds}")

    def map_from_canonical(self, canonical_samples):
        return canonical_samples * self._scale + self._loc

    def map_to_canonical(self, user_samples):
        self._check_bounds(user_samples)
        return (user_samples - self._loc) / self._scale

    def derivatives_to_canonical(self, user_derivs, order=1):
        return user_derivs * self._scale**order

    def derivatives_from_canonical(self, canonical_derivs, order=1):
        return canonical_derivs / self._scale**order

    def __repr__(self):
        return "{0}(loc={1}, scale={2})".format(
            self.__class__.__name__, self._loc, self._scale
        )


class UnivariateBoundedAffineTransform(UnivariateAffineTransform):
    def __init__(self, bounds, enforce_bounds=False, backend=None):
        super().__init__(
            bounds[0], bounds[1] - bounds[0], enforce_bounds, backend
        )
        self._bounds = bounds
