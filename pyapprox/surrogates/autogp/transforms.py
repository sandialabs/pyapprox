import numpy as np

from pyapprox.surrogates.autogp._torch_wrappers import (
    sqrt, full, copy, arccos, sin, cos, empty, log, exp)


class IdentityTransform(object):
    def __init__(self, num_vars):
        self.nvars = num_vars

    def map_from_canonical(self, samples):
        return samples

    def map_to_canonical(self, samples):
        return samples

    def map_stdev_from_canonical(self, canonical_stdevs):
        return canonical_stdevs


class StandardDeviationTransform():
    def __init__(self):
        self._means = None
        self._stdevs = None

    def map_to_canonical(self, samples):
        self._means = samples.mean(axis=1)[:, None]
        self._stdevs = samples.std(axis=1, ddof=1)[:, None]
        canonical_samples = (samples-self._means)/self._stdevs
        return canonical_samples

    def map_from_canonical(self, canonical_samples):
        samples = canonical_samples*self._stdevs + self._means
        return samples

    def map_stdev_from_canonical(self, canonical_stdevs):
        return canonical_stdevs*self._stdevs


class NSphereCoordinateTransform():
    def map_to_nsphere(self, samples):
        nvars, nsamples = samples.shape
        r = sqrt((samples**2).sum(axis=0))
        psi = full(samples.shape, 0.)
        psi[0] = copy(r)
        psi[1] = arccos(samples[0]/r)
        for ii in range(2, nvars):
            denom = copy(r)
            for jj in range(ii-1):
                denom *= sin(psi[jj+1])
            psi[ii] = arccos(samples[ii-1]/denom)
        psi[-1][samples[-1] < 0] = 2*np.pi-psi[-1][samples[-1] < 0]
        return psi

    def map_from_nsphere(self, psi):
        nvars, nsamples = psi.shape
        r = copy(psi[0])
        samples = full(psi.shape, 0.)
        samples[0] = r*cos(psi[1])
        for ii in range(1, nvars):
            samples[ii, :] = copy(r)
            for jj in range(ii):
                samples[ii] *= sin(psi[jj+1])
            if ii != nvars-1:
                samples[ii] *= cos(psi[ii+1])
        return samples


class SphericalCorrelationTransformation():
    def __init__(self, noutputs):
        self.noutputs = noutputs
        self.ntheta = (self.noutputs*(self.noutputs+1))//2
        self._theta_indices = np.full((self.ntheta, 2), -1, dtype=int)
        self._theta_indices[:self.noutputs, 0] = np.arange(self.noutputs)
        self._theta_indices[:self.noutputs, 1] = 0
        for ii in range(1, noutputs):
            for jj in range(1, ii+1):
                # indices[ii, jj] = (
                #     self.noutputs+((ii-1)*(ii))//2 + (jj-1))
                self._theta_indices[
                    self.noutputs+((ii-1)*(ii))//2 + (jj-1)] = ii, jj
        self.nsphere_trans = NSphereCoordinateTransform()
        # unconstrained formulation does not seem unique.
        self._unconstrained = False

    def get_spherical_bounds(self):
        if not self._unconstrained:
            # l_{i1} > 0,  i = 0,...,noutputs-1
            # l_{ij} in (0, np.pi),    i = 1,...,noutputs-1, j=1,...,i
            eps = 0
            bounds = np.array([[eps, np.inf] for ii in range(self.noutputs)])
            other_bounds = np.array([
                [eps, np.pi-eps] for ii in range(self.noutputs, self.ntheta)])
            bounds = np.vstack((bounds, other_bounds))
            return bounds

        return np.array([[-np.inf, np.inf] for ii in range(self.theta)])

    def map_cholesky_to_spherical(self, L):
        psi = empty(L.shape)
        psi[0, 0] = L[0, 0]
        for ii in range(1, self.noutputs):
            psi[ii, :ii+1] = self.nsphere_trans.map_to_nsphere(
                L[ii:ii+1, :ii+1].T).T
        return psi

    def map_spherical_to_unconstrained_theta(self, psi):
        theta = empty(self.ntheta)
        theta[:self.noutputs] = log(psi[:, 0])
        psi_flat = psi[
            self._theta_indices[self.noutputs:, 0],
            self._theta_indices[self.noutputs:, 1]]
        theta[self.noutputs:] = log(psi_flat/(np.pi-psi_flat))
        return theta

    def map_spherical_to_theta(self, psi):
        if self._unconstrained:
            return self.map_spherical_to_unconstrained_theta(psi)
        return psi[self._theta_indices[:, 0], self._theta_indices[:, 1]]

    def map_from_cholesky(self, L):
        psi = self.map_cholesky_to_spherical(L)
        return self.map_spherical_to_theta(psi)

    def map_unconstrained_theta_to_spherical(self, theta):
        psi = full((self.noutputs, self.noutputs), 0.)
        # psi[ii, :] are radius of hypersphere of increasing dimension
        # all other psi are angles
        exp_theta = exp(theta)
        psi[:, 0] = exp_theta[:self.noutputs]
        psi[self._theta_indices[self.noutputs:, 0],
            self._theta_indices[self.noutputs:, 1]] = (
                exp_theta[self.noutputs:]*np.pi/(1+exp_theta[self.noutputs:]))
        # cnt = self.noutputs
        # for ii in range(1, self.noutputs):
        #     for jj in range(1, ii+1):
        #         exp_theta = exp(theta[cnt])
        #         psi[ii, jj] = exp_theta*np.pi/(1+exp_theta)
        #         cnt += 1
        return psi

    def map_theta_to_spherical(self, theta):
        if theta.ndim != 1:
            raise ValueError("theta must be 1d array")
        if self._unconstrained:
            psi = self.map_unconstrained_theta_to_spherical(theta)
            return self.map_spherical_to_cholesky(psi)
        psi = full((self.noutputs, self.noutputs), 0.)
        psi[self._theta_indices[:, 0], self._theta_indices[:, 1]] = theta
        return psi

    def map_spherical_to_cholesky(self, psi):
        L_factor = full((self.noutputs, self.noutputs), 0.)
        L_factor[0, 0] = psi[0, 0]
        for ii in range(1, self.noutputs):
            L_factor[ii:ii+1, :ii+1] = self.nsphere_trans.map_from_nsphere(
                psi[ii:ii+1, :ii+1].T).T
        return L_factor

    def map_to_cholesky(self, theta):
        psi = self.map_theta_to_spherical(theta)
        return self.map_spherical_to_cholesky(psi)
