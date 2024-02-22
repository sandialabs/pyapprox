import unittest
import numpy as np
import torch

from pyapprox.surrogates.autogp.transforms import (
    NSphereCoordinateTransform, SphericalCorrelationTransform)


class TestTransforms(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def check_nsphere_coordinate_transform(self, nvars):
        nsamples = 10
        trans = NSphereCoordinateTransform()
        psi = np.vstack((np.random.uniform(1, 2, (1, nsamples)),
                         np.random.uniform(0, np.pi, (nvars-2, nsamples)),
                         np.random.uniform(0, 2*np.pi, (1, nsamples))))
        samples = trans.map_from_nsphere(
            torch.as_tensor(psi, dtype=torch.double))
        psi_recovered = trans.map_to_nsphere(samples)
        assert np.allclose(psi_recovered, psi, rtol=1e-12)

    def test_nsphere_coordinate_transform(self):
        test_cases = [
            [2], [3], [4], [5]
        ]
        for test_case in test_cases:
            self.check_nsphere_coordinate_transform(*test_case)

    def check_spherical_correlation_transform(self, noutputs):
        # constrained formulation
        trans = SphericalCorrelationTransform(noutputs)

        # if radius is made to small it can create errors in
        # transform to/from spherical coordinates
        theta = np.hstack((
            np.random.uniform(0, 10, (trans.noutputs)),
            np.random.uniform(0, np.pi, (trans.ntheta-trans.noutputs)),
        ))

        psi = trans.map_theta_to_spherical(
            torch.as_tensor(theta, dtype=torch.double))
        theta_recovered = trans.map_spherical_to_theta(psi)
        assert np.allclose(theta, theta_recovered, rtol=1e-12)

        L = trans.map_to_cholesky(
            torch.as_tensor(theta, dtype=torch.double))
        theta_recovered = trans.map_from_cholesky(
            torch.as_tensor(L, dtype=torch.double))
        assert np.allclose(theta, theta_recovered, rtol=1e-12)

    def test_spherical_correlation_transform(self):
        # Use test case from PINHEIRO 1 and BATES
        noutputs = 3
        trans = SphericalCorrelationTransform(noutputs)
        trans._unconstrained = True
        L = np.array([[1, 0, 0], [1, 2, 0], [1, 2, 3]])
        theta_recovered = trans.map_from_cholesky(
            torch.as_tensor(L, dtype=torch.double))
        theta = np.array(
            [0, np.log(5)/2, np.log(14)/2, -0.608, -0.348, -0.787])
        # answer is only reported to 3 decimals
        assert np.allclose(theta_recovered, theta, rtol=1e-3)

        test_cases = [
            [2], [3], [4], [5]
        ]
        for test_case in test_cases:
            np.random.seed(1)
            self.check_spherical_correlation_transform(*test_case)


if __name__ == "__main__":
    transforms_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestTransforms)
    unittest.TextTestRunner(verbosity=2).run(transforms_test_suite)
