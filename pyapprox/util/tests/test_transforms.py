import unittest
import numpy as np
import torch

from pyapprox.util.transforms import (
    NSphereCoordinateTransform,
    SphericalCorrelationTransform,
    AffineBoundedTransform,
)

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin


class TestTransforms:
    def setUp(self):
        np.random.seed(1)

    def test_affine_bounded_transform(self):
        bkd = self.get_backend()
        nvars, nsamples = 2, 10
        user_ranges = [0, 1, 0, 1]
        canonical_ranges = [-1, 1, -1, 1]
        trans = AffineBoundedTransform(canonical_ranges, user_ranges, bkd=bkd)
        user_samples = bkd.asarray(np.random.uniform(0, 1, (nvars, nsamples)))
        trans_canonical_samples = trans.map_to_canonical(user_samples)
        assert trans_canonical_samples.min() >= -1.0
        assert trans_canonical_samples.max() <= 1.0
        trans_user_samples = trans.map_from_canonical(trans_canonical_samples)
        assert np.allclose(trans_user_samples, user_samples)

    def _check_nsphere_coordinate_transform(self, nvars):
        bkd = self.get_backend()
        nsamples = 10
        trans = NSphereCoordinateTransform(backend=bkd)
        psi = bkd.vstack(
            (
                bkd.array(np.random.uniform(1, 2, (1, nsamples))),
                bkd.array(np.random.uniform(0, np.pi, (nvars - 2, nsamples))),
                bkd.array(np.random.uniform(0, 2 * np.pi, (1, nsamples))),
            )
        )
        samples = trans.map_from_nsphere(psi)
        psi_recovered = trans.map_to_nsphere(samples)
        assert np.allclose(psi_recovered, psi, rtol=1e-12)

    def test_nsphere_coordinate_transform(self):
        for kk in range(2, 6):
            np.random.seed(1)
            self._check_nsphere_coordinate_transform(kk)

    def _check_spherical_correlation_transform(self, noutputs):
        # constrained formulation
        bkd = self.get_backend()
        trans = SphericalCorrelationTransform(noutputs, backend=bkd)

        # if radius is made to small it can create errors in
        # transform to/from spherical coordinates
        theta = bkd.hstack(
            (
                bkd.array(np.random.uniform(0, 10, (trans.noutputs))),
                bkd.array(
                    np.random.uniform(
                        0, np.pi, (trans.ntheta - trans.noutputs)
                    )
                ),
            )
        )

        psi = trans.map_theta_to_spherical(bkd.atleast1d(theta))
        theta_recovered = trans.map_spherical_to_theta(psi)
        assert np.allclose(theta, theta_recovered, rtol=1e-12)

        L = trans.map_to_cholesky(bkd.array(theta))
        theta_recovered = trans.map_from_cholesky(L)
        assert np.allclose(theta, theta_recovered, rtol=1e-12)

    def test_spherical_correlation_transform(self):
        # Use test case from PINHEIRO 1 and BATES
        bkd = self.get_backend()
        noutputs = 3
        trans = SphericalCorrelationTransform(noutputs, backend=bkd)
        trans._unconstrained = True
        L = bkd.asarray([[1, 0, 0], [1, 2, 0], [1, 2, 3]])
        theta_recovered = trans.map_from_cholesky(bkd.atleast2d(L))
        theta = bkd.asarray(
            [0, np.log(5) / 2, np.log(14) / 2, -0.608, -0.348, -0.787]
        )
        # answer is only reported to 3 decimals
        assert np.allclose(theta_recovered, theta, rtol=1e-3)

        for kk in range(2, 6):
            np.random.seed(1)
            self._check_spherical_correlation_transform(kk)


class TestNumpyTransforms(TestTransforms, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchTransforms(TestTransforms, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
