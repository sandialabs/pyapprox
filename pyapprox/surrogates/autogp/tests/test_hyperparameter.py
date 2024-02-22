import unittest
import numpy as np

from pyapprox.surrogates.autogp.hyperparameter import (
    LogHyperParameterTransform, IdentityHyperParameterTransform,
    HyperParameter, HyperParameterList)


class TestHyperParameter(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_hyperparameter(self):
        transform_0 = LogHyperParameterTransform()
        hyp_0 = HyperParameter("P0", 3, 1, [0.01, 2], transform_0)
        assert np.allclose(
            hyp_0.get_active_opt_bounds(), np.log(
                np.array([[0.01, 2], [0.01, 2], [0.01, 2]])))

        transform_1 = IdentityHyperParameterTransform()
        hyp_1 = HyperParameter(
            "P1", 2, -0.5, [-1, 6, np.nan, np.nan], transform_1)
        hyp_list_0 = HyperParameterList([hyp_0, hyp_1])
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

if __name__ == "__main__":
    hyperparameter_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestHyperParameter)
    unittest.TextTestRunner(verbosity=2).run(hyperparameter_test_suite)
