import unittest
import numpy as np
import glob
import os
import multiprocessing
import tempfile

from pyapprox.interface.wrappers import (
    ActiveSetVariableModel, PoolModel, DataFunctionModel,
    combine_saved_model_data, TimerModel, WorkTrackingModel,
    evaluate_1darray_function_on_2d_array, ArchivedDataModel
)


def function(x):
    return np.sum(x, axis=0)[:, np.newaxis]


def function_with_jac(x, return_grad=False):
    vals = np.sum(x**2, axis=0)[:, np.newaxis]
    grads = 2*x.T
    if return_grad:
        return vals, grads
    return vals


class TestModelwrappers(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_evaluate_1darray_function_on_2d_array(self):
        num_vars, num_samples = 3, 10
        samples = np.random.uniform(0., 1., (num_vars, num_samples))

        def fun(sample, return_grad=False):
            assert sample.ndim == 1
            if not return_grad:
                return function_with_jac(sample[:, None], return_grad)[:, 0]
            val, grad = function_with_jac(sample[:, None], return_grad)
            return val[:, 0], grad[0, :]

        exact_values, exact_grads = function_with_jac(
            samples, return_grad=True)
        values = evaluate_1darray_function_on_2d_array(fun, samples)
        assert np.allclose(values, exact_values)
        values, grads = evaluate_1darray_function_on_2d_array(
            fun, samples, return_grad=True)
        assert np.allclose(values, exact_values)
        assert np.allclose(grads, exact_grads)

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

        model = ActiveSetVariableModel(
            function_with_jac, num_vars, nominal_var_values,
            active_var_indices)
        values, grads = model(reduced_samples, return_grad=True)
        exact_values, exact_grads = function_with_jac(samples, return_grad=True)
        assert np.allclose(values, exact_values)
        assert np.allclose(grads, exact_grads)

    def test_work_tracking_model(self):
        nvars, nsamples = 3, 4
        max_eval_concurrency = 2
        base_model = function_with_jac
        timer_model = TimerModel(base_model, base_model)
        pool_model = PoolModel(
            timer_model, max_eval_concurrency, base_model=base_model,
            assert_omp=False)
        # pool_model = timer_model
        model = WorkTrackingModel(
            pool_model, base_model, enforce_timer_model=False)

        samples = np.random.normal(0, 1, (nvars, nsamples))
        values = model(samples)
        exact_values, exact_grads = function_with_jac(
            samples, return_grad=True)
        assert np.allclose(values, exact_values)
        values, grads = model(samples, return_grad=True)
        assert np.allclose(values, exact_values)
        assert np.allclose(np.vstack(grads), exact_grads)

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
        values, jacs = model(samples, return_grad=True)
        exact_values, exact_jacs = function_with_jac(samples, return_grad=True)
        assert np.allclose(values, exact_values)
        assert np.allclose(np.hstack([j[:, None] for j in jacs]), exact_jacs)

    def test_archived_data_model(self):
        nvars, nsamples = 2, 100
        samples = np.random.normal(0, 1, (nvars, nsamples))
        values = np.sum(samples**2, axis=0)[:, None]

        model = ArchivedDataModel(samples, values)
        II = np.random.permutation(nsamples)[:nsamples//2]
        valid_samples = samples[:, II]
        valid_values = values[II]
        model_values = model(valid_samples)
        assert np.allclose(model_values, valid_values)

        valid_samples, II = model.rvs(nsamples, return_indices=True)
        valid_values = values[II]
        model_values = model(valid_samples)
        assert np.allclose(model_values, valid_values)

        valid_samples, II = model.rvs(
            nsamples//2, return_indices=True, randomness=None)
        valid_values = values[II]
        model_values = model(valid_samples)
        assert np.allclose(model_values, valid_values)

        valid_samples, II = model.rvs(
            2, return_indices=True, randomness=None)
        valid_values = values[II]
        model_values = model(valid_samples)
        assert np.allclose(model_values, valid_values)

        # check error thrown if two many samples are requested
        # with randomnes=None
        self.assertRaises(ValueError, model.rvs, nsamples, randomness=None)

        # check error thrown if one sample is not found
        valid_samples[:, 1] += np.random.normal(0, 1, nvars)
        self.assertRaises(ValueError, model, valid_samples)

    def test_data_function_model(self):
        num_vars = 3
        tmp_dir = tempfile.TemporaryDirectory()
        data_basename = 'data-function-model-data'
        data_basename = os.path.join(tmp_dir.name, data_basename)
        save_frequency = 3
        max_eval_concurrency = min(multiprocessing.cpu_count(), 10)

        pool_model = PoolModel(
            function, max_eval_concurrency, assert_omp=False)
        model = DataFunctionModel(
            pool_model, None, data_basename, save_frequency)

        num_samples = 25
        samples = np.random.uniform(0., 1., (num_vars, num_samples))

        values = model(samples)
        exact_values = function(samples)
        assert np.allclose(values, exact_values)
        assert model.num_evaluations == samples.shape[1]

        filenames = glob.glob(data_basename+'*.pkl')
        num_files = len(filenames)
        assert num_files == num_samples//(save_frequency)+min(
            num_samples % (save_frequency), 1)

        model_1 = DataFunctionModel(
            function, None, data_basename, save_frequency)
        samples_1 = np.random.uniform(0., 1., (num_vars, num_samples*2))
        # set half of new samples to be replicates from previous study
        II = np.random.permutation(np.arange(num_samples*2))[:num_samples]
        samples_1[:, II] = samples
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

        # test requesting values and grads when function does not return grads
        self.assertRaises(ValueError, model_2, samples_2, return_grad=True)

        num_samples = 4 #102
        samples = np.random.uniform(0., 1., (num_vars, num_samples))
        data_basename = 'grad-data-function-model-data'
        data_basename = os.path.join(tmp_dir.name, data_basename)
        model_3 = DataFunctionModel(
            function_with_jac, None, data_basename, save_frequency)
        # set half of new samples to be replicates from previous study
        values, grads = model_3(samples, return_grad=True)
        exact_values, exact_grads = function_with_jac(
            samples, return_grad=True)
        assert np.allclose(values, exact_values)
        assert model_3.num_evaluations == samples.shape[1]
        assert np.allclose(np.vstack(grads), exact_grads)

        model_4 = DataFunctionModel(
            function_with_jac, None, data_basename, save_frequency,
            use_hash=False)
        samples_4 = np.random.uniform(0., 1., (num_vars, num_samples*2))
        # set half of new samples to be replicates from previous study
        II = np.random.permutation(np.arange(num_samples*2))[:num_samples]
        samples_4[:, II] = samples
        values, grads = model_4(samples_4, return_grad=True)
        exact_values, exact_grads = function_with_jac(
            samples_4, return_grad=True)
        assert np.allclose(values, exact_values)
        assert model_4.num_evaluations == samples_4.shape[1]
        assert np.allclose(np.vstack(grads), exact_grads)

        # test requesting only values when stored samples also have grads
        samples_5 = np.random.uniform(0., 1., (num_vars, num_samples*2))
        samples_5[:, II] = samples
        values = model_4(samples_5)
        assert np.allclose(
            values, function_with_jac(samples_5, return_grad=False))

        # test requesting values and grads when stored samples do not have
        # grads
        samples_6 = np.random.uniform(0., 1., (num_vars, num_samples))
        model_4(samples_6)
        self.assertRaises(ValueError, model_4, samples_6, return_grad=True)

        tmp_dir.cleanup()


if __name__ == "__main__":
    model_wrappers_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestModelwrappers)
    unittest.TextTestRunner(verbosity=2).run(model_wrappers_test_suite)
