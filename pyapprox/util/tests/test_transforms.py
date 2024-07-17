import unittest
import numpy as np
import torch

from pyapprox.util.transforms.numpytransforms import (
    NumpyNSphereCoordinateTransform, NumpySphericalCorrelationTransform)
from pyapprox.util.transforms.torchtransforms import (
    TorchNSphereCoordinateTransform, TorchSphericalCorrelationTransform)


class TestTransforms(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _check_nsphere_coordinate_transform(
            self, nvars, NSphereCoordinateTransform):
        nsamples = 10
        trans = NSphereCoordinateTransform()
        psi = np.vstack((np.random.uniform(1, 2, (1, nsamples)),
                         np.random.uniform(0, np.pi, (nvars-2, nsamples)),
                         np.random.uniform(0, 2*np.pi, (1, nsamples))))
        samples = trans.map_from_nsphere(trans._la_atleast2d(psi))
        psi_recovered = trans.map_to_nsphere(samples)
        assert np.allclose(psi_recovered, psi, rtol=1e-12)

    def test_nsphere_coordinate_transform(self):
        test_cases = [
            [kk, NumpyNSphereCoordinateTransform] for kk in range(2, 6)]
        test_cases += [
            [kk, TorchNSphereCoordinateTransform] for kk in range(2, 6)]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_nsphere_coordinate_transform(*test_case)

    def _check_spherical_correlation_transform(
            self, noutputs, SphericalCorrelationTransform):
        # constrained formulation
        trans = SphericalCorrelationTransform(noutputs)

        # if radius is made to small it can create errors in
        # transform to/from spherical coordinates
        theta = np.hstack((
            np.random.uniform(0, 10, (trans.noutputs)),
            np.random.uniform(0, np.pi, (trans.ntheta-trans.noutputs)),
        ))

        psi = trans.map_theta_to_spherical(trans._la_atleast1d(theta))
        theta_recovered = trans.map_spherical_to_theta(psi)
        assert np.allclose(theta, theta_recovered, rtol=1e-12)

        L = trans.map_to_cholesky(
            torch.as_tensor(theta, dtype=torch.double))
        theta_recovered = trans.map_from_cholesky(L)
        assert np.allclose(theta, theta_recovered, rtol=1e-12)

    def test_spherical_correlation_transform(self):
        # Use test case from PINHEIRO 1 and BATES
        noutputs = 3
        trans = NumpySphericalCorrelationTransform(noutputs)
        trans._unconstrained = True
        L = trans._la_atleast2d([[1, 0, 0], [1, 2, 0], [1, 2, 3]])
        theta_recovered = trans.map_from_cholesky(trans._la_atleast2d(L))
        theta = trans._la_atleast1d(
            [0, np.log(5)/2, np.log(14)/2, -0.608, -0.348, -0.787])
        # answer is only reported to 3 decimals
        assert np.allclose(theta_recovered, theta, rtol=1e-3)

        test_cases = [
            [kk, NumpySphericalCorrelationTransform] for kk in range(2, 6)]
        test_cases += [
            [kk, TorchSphericalCorrelationTransform] for kk in range(2, 6)]
        for test_case in test_cases:
            np.random.seed(1)
            self._check_spherical_correlation_transform(*test_case)


if __name__ == "__main__":
    transforms_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestTransforms)
    unittest.TextTestRunner(verbosity=2).run(transforms_test_suite)
