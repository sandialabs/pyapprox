import unittest
import numpy as np

from pyapprox.util.hyperparameter import (
    LogHyperParameterTransform, IdentityHyperParameterTransform,
    HyperParameter, HyperParameterList)

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin


class TestHyperParameter:
    def setUp(self):
        np.random.seed(1)

    def test_hyperparameter(self):
        bkd = self.get_backend()
        transform_0 = LogHyperParameterTransform(backend=bkd)
        hyp_0 = HyperParameter("P0", 3, 1, [0.01, 2], transform_0)
        assert np.allclose(
            hyp_0.get_active_opt_bounds(), np.log(
                np.array([[0.01, 2], [0.01, 2], [0.01, 2]])))

        # test setting some hyperparameters to inactive using HyperParameter
        transform_1 = IdentityHyperParameterTransform(backend=bkd)
        hyp_1 = HyperParameter(
            "P1", 2, -0.5, [-1, 6, -2, 3], transform_1)
        hyp_1.set_active_indices(bkd._la_asarray([0,], dtype=int))
        hyp_list_0 = HyperParameterList([hyp_0, hyp_1])
        assert np.allclose(
            hyp_list_0.get_active_opt_bounds(), np.vstack((
                np.log(np.array([[0.01, 2], [0.01, 2], [0.01, 2]])),
                np.array([[-1, 6]]))))

        # test setting some hyperparameters to inactive using HyperParameterList
        transform_1 = IdentityHyperParameterTransform(backend=bkd)
        hyp_1 = HyperParameter(
            "P1", 2, -0.5, [-1, 6, -2, 3], transform_1)
        hyp_list_0 = HyperParameterList([hyp_0, hyp_1])
        hyp_list_0.set_active_indices(bkd._la_asarray([0, 1, 2, 3,], dtype=int))
        assert np.allclose(
            hyp_list_0.get_active_opt_bounds(), np.vstack((
                np.log(np.array([[0.01, 2], [0.01, 2], [0.01, 2]])),
                np.array([[-1, 6]]))))


        hyp_2 = HyperParameter("P2", 1, 0.25, [-3, 3], transform_1)
        hyp_list_1 = HyperParameterList([hyp_2])
        hyp_list_2 = hyp_list_0 + hyp_list_1
        assert np.allclose(
            hyp_list_2.get_values(), np.hstack((
                np.full(3, 1), np.full(2, -0.5), np.full(1, 0.25))))
        assert np.allclose(
            hyp_list_2.get_active_opt_bounds(), np.vstack((
                np.log(np.array([[0.01, 2], [0.01, 2], [0.01, 2]])),
                np.array([[-1, 6]]),
                np.array([[-3, 3]]),
            )))


class TestNumpyHyperParameter(TestHyperParameter, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin()


class TestTorchHyperParameter(TestHyperParameter, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin()


if __name__ == "__main__":
    unittest.main(verbosity=2)
