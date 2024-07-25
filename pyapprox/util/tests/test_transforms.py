import unittest
import numpy as np
import torch

from pyapprox.util.transforms import (
    NSphereCoordinateTransform,
    SphericalCorrelationTransform,
)

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin


class TestTransforms:
    def setUp(self):
        np.random.seed(1)

    def _check_nsphere_coordinate_transform(self, nvars):
        bkd = self.get_backend()
        nsamples = 10
        trans = NSphereCoordinateTransform(backend=bkd)
        psi = bkd._la_vstack(
            (
                bkd._la_array(np.random.uniform(1, 2, (1, nsamples))),
                bkd._la_array(
                    np.random.uniform(0, np.pi, (nvars - 2, nsamples))
                ),
                bkd._la_array(np.random.uniform(0, 2 * np.pi, (1, nsamples))),
            )
        )
        samples = trans.map_from_nsphere(bkd._la_atleast2d(psi))
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
        theta = bkd._la_hstack(
            (
                bkd._la_array(np.random.uniform(0, 10, (trans.noutputs))),
                bkd._la_array(
                    np.random.uniform(
                        0, np.pi, (trans.ntheta - trans.noutputs)
                    )
                ),
            )
        )

        psi = trans.map_theta_to_spherical(bkd._la_atleast1d(theta))
        theta_recovered = trans.map_spherical_to_theta(psi)
        assert np.allclose(theta, theta_recovered, rtol=1e-12)

        L = trans.map_to_cholesky(bkd._la_array(theta))
        theta_recovered = trans.map_from_cholesky(L)
        assert np.allclose(theta, theta_recovered, rtol=1e-12)

    def test_spherical_correlation_transform(self):
        # Use test case from PINHEIRO 1 and BATES
        bkd = self.get_backend()
        noutputs = 3
        trans = SphericalCorrelationTransform(noutputs, backend=bkd)
        trans._unconstrained = True
        L = bkd._la_atleast2d([[1, 0, 0], [1, 2, 0], [1, 2, 3]])
        theta_recovered = trans.map_from_cholesky(bkd._la_atleast2d(L))
        theta = bkd._la_atleast1d(
            [0, np.log(5) / 2, np.log(14) / 2, -0.608, -0.348, -0.787]
        )
        # answer is only reported to 3 decimals
        assert np.allclose(theta_recovered, theta, rtol=1e-3)

        for kk in range(2, 6):
            np.random.seed(1)
            self._check_spherical_correlation_transform(kk)


class TestNumpyTransforms(TestTransforms, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin()


class TestTorchTransforms(TestTransforms, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin()


if __name__ == "__main__":
    unittest.main(verbosity=2)
