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

        model = PoolModel(function,max_eval_concurrency,assert_omp=False)

        num_samples = 102
        samples = np.random.uniform(0.,1.,(num_vars,num_samples))
        
        values = model(samples)
        exact_values = function(samples)
        assert np.allclose(values,exact_values)

    def test_data_function_model(self):
        num_vars = 3
        data_basename='data_function-model-data'
        save_frequency=3
        max_eval_concurrency=min(multiprocessing.cpu_count(),10)


        filenames = glob.glob(data_basename+'*.npz')
        for filename in filenames:
           os.remove(filename)

        pool_model = PoolModel(function,max_eval_concurrency,assert_omp=False)  
        model = DataFunctionModel(
            pool_model,None,data_basename,save_frequency)

        num_samples = 102
        samples = np.random.uniform(0.,1.,(num_vars,num_samples))
        
        values = model(samples)
        exact_values = function(samples)
        assert np.allclose(values,exact_values)
        assert model.num_evaluations==samples.shape[1]

        filenames = glob.glob(data_basename+'*.npz')
        num_files = len(filenames)
        assert num_files==num_samples//(save_frequency)+min(
            num_samples%(save_frequency),1)

        model_1=DataFunctionModel(
            function,None,data_basename,save_frequency)
        samples_1 = np.random.uniform(0.,1.,(num_vars,num_samples*2))
        #set half of new samples to be replicates from previous study
        I = np.random.permutation(np.arange(num_samples*2))[:num_samples]
        print(I)
        samples_1[:,I] = samples
        values = model_1(samples_1)
        exact_values = function(samples_1)
        assert np.allclose(values,exact_values)
        assert model_1.num_evaluations==samples_1.shape[1]

        from pyapprox.utilities import unique_matrix_rows
        data=combine_saved_model_data(data_basename)
        assert data[0].shape[1]==num_samples*2
        assert unique_matrix_rows(data[0].T).T.shape[1]==2*num_samples
        model_2=DataFunctionModel(
            function,data,data_basename,save_frequency)

        #set two thirds of new samples to be replicates from previous study
        samples_2 = np.random.uniform(0.,1.,(num_vars,num_samples*3))
        I = np.random.permutation(np.arange(num_samples*3))[:2*num_samples]
        samples_2[:,I] = samples_1
        values = model_2(samples_2)
        assert model_2.num_evaluations==samples_2.shape[1]

        data=combine_saved_model_data(data_basename)
        assert data[0].shape[1]==num_samples*3
        assert unique_matrix_rows(data[0].T).T.shape[1]==3*num_samples

        for filename in filenames:
           os.remove(filename)
        
if __name__== "__main__":    
    model_wrappers_test_suite = unittest.TestLoader().loadTestsFromTestCase(
         TestModelwrappers)
    unittest.TextTestRunner(verbosity=2).run(model_wrappers_test_suite)
