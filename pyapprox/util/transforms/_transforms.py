from abc import ABC, abstractmethod
import math

from pyapprox.util.linearalgebra.numpylinalg import (
    LinAlgMixin, NumpyLinAlgMixin)


class Transform(ABC):
    def __init__(self, backend: LinAlgMixin = None):
        if backend is None:
            backend = NumpyLinAlgMixin()
        self._bkd = backend

    @abstractmethod
    def map_from_canonical(self, values):
        raise NotImplementedError

    @abstractmethod
    def map_to_canonical(self, values):
        raise NotImplementedError

    def map_stdev_from_canonical(self, canonical_stdevs):
        raise NotImplementedError

    def __repr__(self):
        return "{0}".format(self.__class__.__name__)


class IdentityTransform(Transform):
    def map_from_canonical(self, values):
        return values

    def map_to_canonical(self, values):
        return values

    def map_stdev_from_canonical(self, canonical_stdevs):
        return canonical_stdevs


class StandardDeviationTransform(Transform):
    def __init__(self, trans=False, backend=None):
        # todo: samples and values should always be (nvars, nsamples)
        # where nvars=nqois but currently values is transpose of this
        # so trans=True is used to deal with this case
        super().__init__(backend)
        self._trans = trans
        self._means = None
        self._stdevs = None

    def map_to_canonical(self, values):
        if not self._trans:
            self._means = self._bkd._la_mean(values, axis=1)[:, None]
            self._stdevs = self._bkd._la_std(values, axis=1, ddof=1)[:, None]
        else:
            self._means = self._bkd._la_mean(values, axis=0)[:, None]
            self._stdevs = self._bkd._la_std(values, axis=0, ddof=1)[:, None]
        canonical_values = (values-self._means)/self._stdevs
        return canonical_values

    def map_from_canonical(self, canonical_values):
        values = canonical_values*self._stdevs + self._means
        return values

    def map_stdev_from_canonical(self, canonical_stdevs):
        return canonical_stdevs*self._stdevs


class NSphereCoordinateTransform(Transform):
    def map_to_nsphere(self, samples):
        nvars, nsamples = samples.shape
        r = self._bkd._la_sqrt((samples**2).sum(axis=0))
        psi = self._bkd._la_full(samples.shape, 0.)
        psi[0] = self._bkd._la_copy(r)
        psi[1] = self._bkd._la_arccos(samples[0]/r)
        for ii in range(2, nvars):
            denom = self._bkd._la_copy(r)
            for jj in range(ii-1):
                denom *= self._bkd._la_sin(psi[jj+1])
            psi[ii] = self._bkd._la_arccos(samples[ii-1]/denom)
        psi[-1][samples[-1] < 0] = 2*math.pi-psi[-1][samples[-1] < 0]
        return psi

    def map_from_nsphere(self, psi):
        nvars, nsamples = psi.shape
        r = self._bkd._la_copy(psi[0])
        samples = self._bkd._la_full(psi.shape, 0.)
        samples[0] = r*self._bkd._la_cos(psi[1])
        for ii in range(1, nvars):
            samples[ii, :] = self._bkd._la_copy(r)
            for jj in range(ii):
                samples[ii] *= self._bkd._la_sin(psi[jj+1])
            if ii != nvars-1:
                samples[ii] *= self._bkd._la_cos(psi[ii+1])
        return samples

    def map_to_canonical(self, psi):
        return self.map_from_nsphere(psi)

    def map_from_canonical(self, canonical_samples):
        return self.map_to_nsphere(canonical_samples)


class SphericalCorrelationTransform(Transform):
    def __init__(self, noutputs, backend=None):
        super().__init__(backend)
        self.noutputs = noutputs
        self.ntheta = (self.noutputs*(self.noutputs+1))//2
        self._theta_indices = self._bkd._la_full((self.ntheta, 2), -1, dtype=int)
        self._theta_indices[:self.noutputs, 0] = self._bkd._la_arange(self.noutputs)
        self._theta_indices[:self.noutputs, 1] = 0
        for ii in range(1, noutputs):
            for jj in range(1, ii+1):
                # indices[ii, jj] = (
                #     self.noutputs+((ii-1)*(ii))//2 + (jj-1))
                self._theta_indices[
                    self.noutputs+((ii-1)*(ii))//2 + (jj-1)] = (
                        self._bkd._la_atleast1d([ii, jj]))
        self.nsphere_trans = NSphereCoordinateTransform(backend=backend)
        # unconstrained formulation does not seem unique.
        self._unconstrained = False

    def get_spherical_bounds(self):
        inf = self._bkd._la_inf()
        if not self._unconstrained:
            # l_{i1} > 0,  i = 0,...,noutputs-1
            # l_{ij} in (0, math.pi),    i = 1,...,noutputs-1, j=1,...,i
            eps = 0
            bounds = self._bkd._la_atleast2d(
                [[eps, inf] for ii in range(self.noutputs)])
            other_bounds = self._bkd._la_atleast2d([
                [eps, math.pi-eps]
                for ii in range(self.noutputs, self.ntheta)])
            bounds = self._bkd._la_vstack((bounds, other_bounds))
            return bounds

        return self._bkd._la_atleast2d([[-inf, inf] for ii in range(self.theta)])

    def map_cholesky_to_spherical(self, L):
        psi = self._bkd._la_empty(L.shape)
        psi[0, 0] = L[0, 0]
        for ii in range(1, self.noutputs):
            psi[ii, :ii+1] = self.nsphere_trans.map_to_nsphere(
                L[ii:ii+1, :ii+1].T).T
        return psi

    def map_spherical_to_unconstrained_theta(self, psi):
        theta = self._bkd._la_empty(self.ntheta)
        theta[:self.noutputs] = self._bkd._la_log(psi[:, 0])
        psi_flat = psi[
            self._theta_indices[self.noutputs:, 0],
            self._theta_indices[self.noutputs:, 1]]
        theta[self.noutputs:] = self._bkd._la_log(psi_flat/(math.pi-psi_flat))
        return theta

    def map_spherical_to_theta(self, psi):
        if self._unconstrained:
            return self.map_spherical_to_unconstrained_theta(psi)
        return psi[self._theta_indices[:, 0], self._theta_indices[:, 1]]

    def map_from_cholesky(self, L):
        psi = self.map_cholesky_to_spherical(L)
        return self.map_spherical_to_theta(psi)

    def map_unconstrained_theta_to_spherical(self, theta):
        psi = self._bkd._la_full((self.noutputs, self.noutputs), 0.)
        # psi[ii, :] are radius of hypersphere of increasing dimension
        # all other psi are angles
        exp_theta = self._bkd._la_exp(theta)
        psi[:, 0] = exp_theta[:self.noutputs]
        psi[self._theta_indices[self.noutputs:, 0],
            self._theta_indices[self.noutputs:, 1]] = (
                exp_theta[self.noutputs:]*math.pi/(
                    1+exp_theta[self.noutputs:]))
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
        psi = self._bkd._la_full((self.noutputs, self.noutputs), 0.)
        psi[self._theta_indices[:, 0], self._theta_indices[:, 1]] = theta
        return psi

    def map_spherical_to_cholesky(self, psi):
        L_factor = self._bkd._la_full((self.noutputs, self.noutputs), 0.)
        L_factor[0, 0] = psi[0, 0]
        for ii in range(1, self.noutputs):
            L_factor[ii:ii+1, :ii+1] = self.nsphere_trans.map_from_nsphere(
                psi[ii:ii+1, :ii+1].T).T
        return L_factor

    def map_to_cholesky(self, theta):
        psi = self.map_theta_to_spherical(theta)
        return self.map_spherical_to_cholesky(psi)

    def map_to_canonical(self, samples):
        return self._map_from_cholesky(samples)

    def map_from_canonical(self, canonical_samples):
        return self._map_to_cholesky(canonical_samples)
