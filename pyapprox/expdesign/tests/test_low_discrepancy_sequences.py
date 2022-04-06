import unittest
import numpy as np
from pyapprox.expdesign.low_discrepancy_sequences import sobol_sequence, halton_sequence


class TestLowDiscrepancySequences(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_sobol_sequence(self):
        samples = sobol_sequence(3, 10)
        true_samples = np.asarray(
            [[0, 0, 0],
             [0.5, 0.5, 0.5],
             [0.75, 0.25, 0.25],
             [0.25, 0.75, 0.75],
             [0.375, 0.375, 0.625],
             [0.875, 0.875, 0.125],
             [0.625, 0.125, 0.875],
             [0.125, 0.625, 0.375],
             [0.1875, 0.3125, 0.9375],
             [0.6875, 0.8125, 0.4375]]).T
        assert np.allclose(true_samples, samples)

    def test_sobol_sequence_nonzero_start_index(self):
        samples = sobol_sequence(3, 8, 2)
        true_samples = np.asarray(
            [[0, 0, 0],
             [0.5, 0.5, 0.5],
             [0.75, 0.25, 0.25],
             [0.25, 0.75, 0.75],
             [0.375, 0.375, 0.625],
             [0.875, 0.875, 0.125],
             [0.625, 0.125, 0.875],
             [0.125, 0.625, 0.375],
             [0.1875, 0.3125, 0.9375],
             [0.6875, 0.8125, 0.4375]]).T
        assert np.allclose(true_samples[:, 2:], samples)

    def test_sobol_sequence_variable_transformation(self):
        from pyapprox.variables.joint import IndependentMarginalsVariable
        from scipy.stats import uniform
        variables = IndependentMarginalsVariable(
            [uniform(-1, 2), uniform(0, 1), uniform(0,3)])
        samples = sobol_sequence(3, 10, variable=variables)
        true_samples = np.asarray(
            [[0,0,0],
             [0.5, 0.5, 0.5],
             [0.75, 0.25, 0.25],
             [0.25, 0.75, 0.75],
             [0.375, 0.375, 0.625],
             [0.875, 0.875, 0.125],
             [0.625, 0.125, 0.875],
             [0.125, 0.625, 0.375],
             [0.1875, 0.3125, 0.9375],
             [0.6875, 0.8125, 0.4375]]).T
        true_samples[0, :] = true_samples[0, :]*2-1
        true_samples[2, :] = true_samples[2, :]*3
        assert np.allclose(true_samples, samples) 

    def test_halton_sequence(self):
        samples = halton_sequence(3, 10, 0)
        true_samples = np.asarray(
            [[0.0,  0.0,  0.0 ],
             [1/2,  1/3,  0.2 ],
             [1/4,  2/3,  0.4 ],
             [3/4,  1/9,  0.6 ],
             [1/8,  4/9,  0.8 ],
             [5/8,  7/9,  0.04],
             [3/8,  2/9,  0.24],
             [7/8,  5/9,  0.44],
             [1/16, 8/9,  0.64],
             [9/16, 1/27, 0.84]]).T
        assert np.allclose(true_samples, samples)

    def test_halton_sequence_nonzero_start_index(self):
        samples = halton_sequence(3, 8, 2)
        true_samples = np.asarray(
            [[0.0,  0.0,  0.0],
             [1/2,  1/3,  0.2],
             [1/4,  2/3,  0.4],
             [3/4,  1/9,  0.6],
             [1/8,  4/9,  0.8],
             [5/8,  7/9,  0.04],
             [3/8,  2/9,  0.24],
             [7/8,  5/9,  0.44],
             [1/16, 8/9,  0.64],
             [9/16, 1/27, 0.84]]).T
        assert np.allclose(true_samples[:, 2:], samples)


if __name__ == "__main__":    
    low_discrepancy_sequences_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestLowDiscrepancySequences)
    unittest.TextTestRunner(verbosity=2).run(
        low_discrepancy_sequences_test_suite)

