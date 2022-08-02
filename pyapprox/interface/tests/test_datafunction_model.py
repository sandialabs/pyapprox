import unittest
import numpy as np

from pyapprox.interface.wrappers import DataFunctionModel


class ModelWithCounter(object):
    def __init__(self):
        self.counter = 0

    def __call__(self, samples):
        self.counter += samples.shape[1]
        return np.sum(samples**2, axis=0)[:, np.newaxis]


class TestDataFunctionModel(unittest.TestCase):
    def test_empty_initial_data_do_not_use_hash(self):

        num_vars = 2
        num_samples = 10

        submodel = ModelWithCounter()
        model = DataFunctionModel(submodel, None, use_hash=False)

        samples = np.random.uniform(-1., 1., (num_vars, num_samples))
        values = model(samples)
        values = model(samples)
        assert submodel.counter == num_samples
        assert np.allclose(values, submodel(samples))

        new_samples = np.random.uniform(-1., 1., (num_vars, num_samples))
        II = np.random.permutation(2*num_samples)[:num_samples+num_samples//2]
        samples = np.hstack(
            (samples, new_samples))[:, II]
        counter = submodel.counter
        values = model(samples)
        assert submodel.counter == np.where(II >= num_samples)[
            0].shape[0]+counter
        assert np.allclose(values, submodel(samples))

    def test_empty_initial_data_use_hash(self):

        num_vars = 2
        num_samples = 10

        submodel = ModelWithCounter()
        model = DataFunctionModel(submodel, None)

        samples = np.random.uniform(-1., 1., (num_vars, num_samples))
        values = model(samples)
        values = model(samples)
        assert submodel.counter == num_samples
        assert np.allclose(values, submodel(samples))

        new_samples = np.random.uniform(-1., 1., (num_vars, num_samples))
        II = np.random.permutation(2*num_samples)[:num_samples+num_samples//2]
        samples = np.hstack(
            (samples, new_samples))[:, II]
        counter = submodel.counter
        values = model(samples)
        assert submodel.counter == np.where(II >= num_samples)[
            0].shape[0]+counter
        assert np.allclose(values, submodel(samples))

    def test_initial_data(self):
        num_vars = 2
        num_samples = 10

        submodel = ModelWithCounter()

        samples = np.random.uniform(-1., 1., (num_vars, num_samples))
        values = submodel(samples)

        model = DataFunctionModel(submodel, (samples, values, None))
        submodel.counter = 0

        values = model(samples)
        assert submodel.counter == 0
        assert np.allclose(values, submodel(samples))  # increments counter

        new_samples = np.random.uniform(-1., 1., (num_vars, num_samples))
        II = np.random.permutation(2*num_samples)[:num_samples+num_samples//2]
        samples = np.hstack(
            (samples, new_samples))[:, II]
        counter = submodel.counter
        values = model(samples)
        assert submodel.counter == np.where(II >= num_samples)[
            0].shape[0]+counter
        assert np.allclose(values, submodel(samples))  # increments counter


if __name__ == "__main__":
    data_function_model_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestDataFunctionModel)
    unittest.TextTestRunner(verbosity=2).run(data_function_model_test_suite)
