import unittest
from pyapprox.models.wrappers import *
import glob, os
import multiprocessing

def function(x):
    return np.sum(x,axis=0)[:,np.newaxis]

class TestModelwrappers(unittest.TestCase):

    def test_active_set_model(self):
        num_vars = 3

        nominal_var_values = np.zeros(num_vars)
        active_var_indices = np.array([0,2])
        model = ActiveSetVariableModel(
            function,nominal_var_values,active_var_indices)

        num_samples = 10
        samples = np.random.uniform(0.,1.,(num_vars,num_samples))
        samples[1,:] = 0.
        reduced_samples = samples[active_var_indices,:]
        
        values = model(reduced_samples)
        exact_values = function(samples)
        assert np.allclose(values,exact_values)


    def test_pool_model(self):
        num_vars = 3
        max_eval_concurrency=min(multiprocessing.cpu_count(),10)
        data_basename='pool-model-data'
        save_frequency=1


        filenames = glob.glob(data_basename+'*.npz')
        for filename in filenames:
           os.remove(filename)

        model = PoolModel(
            function,max_eval_concurrency,data_basename,save_frequency,
            assert_omp=False)

        num_samples = 102
        samples = np.random.uniform(0.,1.,(num_vars,num_samples))
        
        values = model(samples)
        exact_values = function(samples)
        assert np.allclose(values,exact_values)

        filenames = glob.glob(data_basename+'*.npz')
        num_files = len(filenames)
        assert num_files==num_samples//(max_eval_concurrency*save_frequency)+min(num_samples%(max_eval_concurrency*save_frequency),1)

        data = combine_saved_model_data(data_basename)
        model_wrapper=DataFunctionModel(model,data)

        new_samples = np.random.uniform(0.,1.,(num_vars,num_samples*2))
        I = np.random.permutation(np.arange(num_samples*2))[:num_samples]
        new_samples[:,I] = samples
        values = model_wrapper(new_samples)
        
        filenames = glob.glob(data_basename+'*.npz')
        assert len(filenames)==num_files*2

        for filename in filenames:
           os.remove(filename)
        
if __name__== "__main__":    
    model_wrappers_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestModelwrappers)
    unittest.TextTestRunner(verbosity=2).run(model_wrappers_test_suite)
