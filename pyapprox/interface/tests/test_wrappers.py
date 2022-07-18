import unittest
import numpy as np
import glob
import os
import multiprocessing
import tempfile

from pyapprox.interface.wrappers import (
    ActiveSetVariableModel, PoolModel, DataFunctionModel,
    combine_saved_model_data
)


def function(x):
    return np.sum(x, axis=0)[:, np.newaxis]


def function_with_jac(x, jac=False):
    vals = np.sum(x**2, axis=0)[:, np.newaxis]
    grads = 2*x.T
    if jac:
        return vals, grads
    return vals


class TestModelwrappers(unittest.TestCase):

    def test_active_set_model(self):
        num_vars = 3

        nominal_var_values = np.zeros((num_vars-2, 1))
        active_var_indices = np.array([0, 2])
        model = ActiveSetVariableModel(
            function, num_vars, nominal_var_values, active_var_indices)

        num_samples = 10
        samples = np.random.uniform(0., 1., (num_vars, num_samples))
        samples[1, :] = 0.
        reduced_samples = samples[active_var_indices, :]

        values = model(reduced_samples)
        exact_values = function(samples)
        assert np.allclose(values, exact_values)

        nominal_var_values = np.array([[0], [1]]).T
        active_var_indices = np.array([0, 2])
        model = ActiveSetVariableModel(
            function, num_vars, nominal_var_values, active_var_indices)

        num_samples = 10
        samples = np.random.uniform(0., 1., (num_vars, num_samples))
        reduced_samples = samples[active_var_indices, :]

        samples = np.tile(samples, 2)
        samples[1, :num_samples] = 0.
        samples[1, num_samples:] = 1

        values = model(reduced_samples)
        exact_values = function(samples)
        assert np.allclose(values, exact_values)

    def test_pool_model(self):
        num_vars = 3
        max_eval_concurrency = min(multiprocessing.cpu_count(), 10)

        model = PoolModel(function, max_eval_concurrency, assert_omp=False)

        num_samples = 102
        samples = np.random.uniform(0., 1., (num_vars, num_samples))

        values = model(samples)
        exact_values = function(samples)
        assert np.allclose(values, exact_values)

        num_samples = 3
        samples = np.random.uniform(0., 1., (num_vars, num_samples))
        model = PoolModel(
            function_with_jac, max_eval_concurrency, assert_omp=False)
        values, jacs = model(samples, jac=True)
        exact_values, exact_jacs = function_with_jac(samples, jac=True)
        assert np.allclose(values, exact_values)
        print(np.hstack([j[:, None] for j in jacs]))
        print(exact_jacs)
        assert np.allclose(np.hstack([j[:, None] for j in jacs]), exact_jacs)

    def test_data_function_model(self):
        num_vars = 3
        tmp_dir = tempfile.TemporaryDirectory()
        data_basename = 'data-function-model-data'
        data_basename = os.path.join(tmp_dir.name, data_basename)
        save_frequency = 3
        max_eval_concurrency = min(multiprocessing.cpu_count(), 10)

        pool_model = PoolModel(
            function, max_eval_concurrency, assert_omp=False)
        from pyapprox.util.sys_utilities import has_kwarg
        print(has_kwarg(pool_model, "jac"))
        model = DataFunctionModel(
            pool_model, None, data_basename, save_frequency)

        num_samples = 3 #102
        samples = np.random.uniform(0., 1., (num_vars, num_samples))

        values = model(samples)
        exact_values = function(samples)
        assert np.allclose(values, exact_values)
        assert model.num_evaluations == samples.shape[1]

        filenames = glob.glob(data_basename+'*.npz')
        num_files = len(filenames)
        assert num_files == num_samples//(save_frequency)+min(
            num_samples % (save_frequency), 1)

        model_1 = DataFunctionModel(
            function, None, data_basename, save_frequency)
        samples_1 = np.random.uniform(0., 1., (num_vars, num_samples*2))
        # set half of new samples to be replicates from previous study
        I = np.random.permutation(np.arange(num_samples*2))[:num_samples]
        print(I)
        samples_1[:, I] = samples
        values = model_1(samples_1)
        exact_values = function(samples_1)
        assert np.allclose(values, exact_values)
        assert model_1.num_evaluations == samples_1.shape[1]

        from pyapprox.util.utilities import unique_matrix_rows
        data = combine_saved_model_data(data_basename)
        assert data[0].shape[1] == num_samples*2
        assert unique_matrix_rows(data[0].T).T.shape[1] == 2*num_samples
        model_2 = DataFunctionModel(
            function, data, data_basename, save_frequency)

        # set two thirds of new samples to be replicates from previous study
        samples_2 = np.random.uniform(0., 1., (num_vars, num_samples*3))
        II = np.random.permutation(np.arange(num_samples*3))[:2*num_samples]
        samples_2[:, II] = samples_1
        values = model_2(samples_2)
        assert model_2.num_evaluations == samples_2.shape[1]

        data = combine_saved_model_data(data_basename)
        assert data[0].shape[1] == num_samples*3
        assert unique_matrix_rows(data[0].T).T.shape[1] == 3*num_samples

        data_basename = 'grad-data-function-model-data'
        data_basename = os.path.join(tmp_dir.name, data_basename)
        model_3 = DataFunctionModel(
            function_with_jac, None, data_basename, save_frequency)
        # set half of new samples to be replicates from previous study
        values, grads = model_3(samples, jac=True)
        exact_values, exact_grads = function_with_jac(samples, jac=True)
        assert np.allclose(values, exact_values)
        assert model_3.num_evaluations == samples.shape[1]
        print(grads)
        print(exact_grads)
        print(np.hstack([j[:, None] for j in grads]))
        assert np.allclose(np.hstack([j[:, None] for j in grads]), exact_grads)

        model_4 = DataFunctionModel(
            function_with_jac, None, data_basename, save_frequency)
        samples_4 = np.random.uniform(0., 1., (num_vars, num_samples*2))
        # set half of new samples to be replicates from previous study
        I = np.random.permutation(np.arange(num_samples*2))[:num_samples]
        print(I)
        samples_4[:, I] = samples
        values, grads = model_4(samples_4, jac=True)
        exact_values, exact_grads = function_with_jac(samples_4, jac=True)
        assert np.allclose(values, exact_values)
        assert model_4.num_evaluations == samples_4.shape[1]
        assert np.allclose(np.hstack([j[:, None] for j in grads]), exact_grads)
        
        tmp_dir.cleanup()

if __name__ == "__main__":
    model_wrappers_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestModelwrappers)
    unittest.TextTestRunner(verbosity=2).run(model_wrappers_test_suite)
