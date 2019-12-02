from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
import subprocess, os, glob
from functools import partial
from multiprocessing import Pool

def get_num_args(function):
    """
    Return the number of arguments of a function.
    If function is a member function of a class the self argument is not 
    counted.

    Parameters
    ----------
    function : callable
        The Python callable to be interrogated

    Return
    ------
    num_args : integer
        The number of arguments to the function including
        args, varargs, keywords
    """
    import inspect
    args = inspect.getfullargspec(function)
    num_args = 0
    if args[0] is not None:
        num_args += len(args[0])
        if 'self' in  args[0]:
            num_args-=1
    if args[1] is not None:
        num_args += len(args[1])
    if args[2] is not None:
        num_args += len(args[2])
    # do not count defaults of keywords conatined in args[3]
    #if args[3] is not None:
    #    num_args += len(args[3])
    return num_args


def evaluate_1darray_function_on_2d_array(function,samples,opts):
    """
    Evaluate a function at a set of samples.

    Parameters
    ----------
    function : callable
    vals = function(sample)
    function accepts a 1d np.ndarray of shape (num_vars) and returns a 1d 
    np.ndarray of values of shape (num_qoi)

    samples : np.ndarray (num_vars, num_samples)
    The samples at which to evaluate the model

    opts : dictionary
    A set of options that are needed to evaluate the model

    Returns
    -------
    values : np.ndarray (num_samples, num_qoi)
    The value of each requested QoI of the model for each sample
    """
    num_args = get_num_args(function)
    assert samples.ndim==2
    num_samples = samples.shape[1]
    if num_args==2:
        values_0 = function(samples[:,0], opts)
    else:
        values_0 = function(samples[:,0])
    assert values_0.ndim==1
    num_qoi = values_0.shape[0]
    values = np.empty((num_samples,num_qoi),float)
    values[0,:]=values_0
    for i in range(1, num_samples):
        if num_args==2:
            values[i,:] = function(samples[:,i], opts)
        else:
            values[i,:] = function(samples[:,i])

    return values

class PyFunction(object):
    def __init__(self,function):
        self.function=function
        
    def __call__(self,samples,opts=dict()):
        return evaluate_1darray_function_on_2d_array(self.function,samples,opts)


def run_shell_command(shell_command, opts={}):
    """
    Execulte a shell command.

    Parameters
    ----------
    shell_command : string
        The command that you want executed

    output_verbosity : integer (default=0)
        0 - supress all model output
        1 - write output to file
        2 - write output to stdout

    filename : string (default=None)
        The filename to which the output of the shell command is written.
        A file is only written if output_verbosity=1.
        If output_verbosity=1 and filename is None then
        filename = shell_command.out

    env : os.environ (default=None)
        Mapping that defines the environment variables for the new process;
        these are used instead of inheriting the current process environment,
        which is the default behavior.
    """
    output_verbosity = opts.get('verbosity',1)
    env = opts.get('env',None)
    filename = opts.get('filename',None)
    
    if output_verbosity==0:
        out = subprocess.check_output(shell_command, shell=True, env=env)
    elif output_verbosity==1:
        if filename is None: filename = 'shell_command.out'
        with open( filename, 'w' ) as f:
            subprocess.call(shell_command, shell=True, stdout=f,
                            stderr=f, env=env)
    else:
        subprocess.call(shell_command, shell=True, env=env)

from pyapprox.utilities import hash_array
class DataFunctionModel(object):
    def hash_sample(self,sample):
        # if samples have undergone a transformation thier value
        # may not be exactly the same so make hash on samples
        # with fixed precision
        # sample = np.round(sample, self.digits)
        # I = np.where(np.abs(sample)<self.tol)[0]
        # sample[I] = 0.
        key = hash_array(sample)
        return key
    
    def __init__(self,function,data,use_hash=True):
        self.function=function

        self.data=dict()
        self.samples=np.zeros((0,0))
        self.values=None
        self.num_evaluations_ran=0
        self.num_evaluations=0 
        self.digits = 16
        self.tol = 10**(-self.digits)
        self.use_hash=use_hash

        if data is not None:
            self.samples,self.values=data
            assert self.samples.shape[1]==self.values.shape[0]
            for ii in range(self.samples.shape[1]):
                key = self.hash_sample(self.samples[:,ii])
                if key in self.data:
                    print((self.samples.sum(axis=0),self.samples[:,ii]))
                    raise Exception('there are duplicate samples')
                else:
                    self.data[key]=ii
            # set counter so that next file takes into account all previously
            # ran samples
            self.num_evaluations_ran=self.samples.shape[1]
        
    def __call__(self,samples):
        evaluated_sample_indices = []
        new_sample_indices = []
        for ii in range(samples.shape[1]):
            if self.use_hash:
                key = self.hash_sample(samples[:,ii])
                if key in self.data:
                    evaluated_sample_indices.append([ii,self.data[key]])
                else:
                    new_sample_indices.append(ii)
            else:
                found = False
                for jj in range(self.samples.shape[1]):
                    if np.allclose(self.samples[:,jj],samples[:,ii],
                                   atol=self.tol):
                        found = True
                        break
                if found:
                    evaluated_sample_indices.append([ii,jj])
                else:
                    new_sample_indices.append(ii)


            
        evaluated_sample_indices = np.asarray(evaluated_sample_indices)
        if len(new_sample_indices)>0:
            new_samples = samples[:,new_sample_indices]
            new_values  = self.function(new_samples)
            num_qoi = new_values.shape[1]
        else:
            num_qoi = self.values.shape[1]

        values = np.empty((samples.shape[1],num_qoi),dtype=float)
        if len(new_sample_indices)>0:
            values[new_sample_indices,:]=new_values
        if len(new_sample_indices)<samples.shape[1]:
            values[evaluated_sample_indices[:,0]] = \
                self.values[evaluated_sample_indices[:,1],:]

        if len(new_sample_indices)>0:
            if self.samples.shape[1]==0:
                jj=0
                self.samples=samples
                self.values=values
            else:
                jj=self.samples.shape[0]
                self.samples=np.hstack(
                    (self.samples,samples[:,new_sample_indices]))
                self.values=np.vstack((self.values,new_values))

            for ii in range(len(new_sample_indices)):
                key = hash_array(samples[:,new_sample_indices[ii]])
                self.data[key]=jj+ii

            self.num_evaluations_ran+=len(new_sample_indices)
        self.num_evaluations+=samples.shape[1]
        print(self.num_evaluations,self.num_evaluations_ran)

        return values

def run_model_samples_in_parallel(model,max_eval_concurrency,samples,pool=None,
                                  assert_omp=True):
    """
    pool.map serializes each argument and so if model is a class, 
    any of its member variables that are updated in __call__ will not
    persist once each __call__ to pool completes.
    """
    num_samples = samples.shape[1]
    if assert_omp:
        assert int(os.environ['OMP_NUM_THREADS'])==1
    if pool is None:
        pool = Pool(max_eval_concurrency)
    result = pool.map(
        model,[(samples[:,ii:ii+1]) for ii in range(samples.shape[1])])
    num_qoi = result[0].shape[1]
    values = np.empty((num_samples,num_qoi))
    for ii in range(len(result)):
        values[ii,:]=result[ii][0,:]
    return values

class PoolModel(object):
    def __init__(self,function,max_eval_concurrency,data_basename=None,
                 save_frequency=None,assert_omp=True):
        self.function=function
        self.max_eval_concurrency=max_eval_concurrency
        self.num_evaluations=0
        self.pool = Pool(self.max_eval_concurrency)
        self.data_basename = data_basename
        self.save_frequency=save_frequency
        if self.data_basename is not None:
            assert save_frequency is not None
        if self.save_frequency and self.data_basename is None:
            msg = 'Warning save_frequency not being used because data_basename'
            msg += ' is None'
            print(msg)
        self.assert_omp=assert_omp

    def __call__(self,samples):
        if self.data_basename is None:
            vals = run_model_samples_in_parallel(
                self.function,self.max_eval_concurrency,samples,pool=self.pool,
                assert_omp=self.assert_omp)
        else:
            assert self.save_frequency>0
            num_batch_samples = self.max_eval_concurrency*self.save_frequency
            lb = 0
            vals = None
            while lb<samples.shape[1]:
                ub = min(lb+num_batch_samples,samples.shape[1])
                batch_vals = run_model_samples_in_parallel(
                    self.function,self.max_eval_concurrency,samples[:,lb:ub],
                    pool=self.pool,assert_omp=self.assert_omp)
                data_filename = self.data_basename+'-%d-%d.npz'%(
                    self.num_evaluations+lb,self.num_evaluations+ub-1)
                np.savez(data_filename,vals=batch_vals,
                         samples=samples[:,lb:ub])
                if vals is None:
                    vals = batch_vals
                else:
                    vals = np.vstack((vals,batch_vals))
                lb=ub
        self.num_evaluations+=samples.shape[1]
        return vals

class ActiveSetVariableModel(object):
    def __init__(self,function,nominal_var_values,
                 active_var_indices):
        self.function = function
        assert nominal_var_values.ndim==1
        self.nominal_var_values = nominal_var_values
        self.active_var_indices = active_var_indices
        assert np.all(self.active_var_indices<nominal_var_values.shape[0])
        
    def __call__(self,reduced_samples):
        samples = np.tile(
            self.nominal_var_values[:,np.newaxis],(1,reduced_samples.shape[1]))
        samples[self.active_var_indices,:] = reduced_samples
        return self.function(samples)

def combine_saved_model_data(saved_data_basename):
    filenames = glob.glob(saved_data_basename+'*.npz')
    ii =0
    for filename in filenames:
        data = np.load(filename)
        if ii==0:
            vals = data['vals']
            samples = data['samples']
        else:
            vals = np.vstack((vals,data['vals']))
            samples = np.hstack((samples,data['samples']))
        ii+=1
    if len(filenames)==0:
        return None,None
    
    return samples,vals

class SingleFidelityWrapper(object):
    def __init__(self,model,config_values):
        self.model=model
        assert config_values.ndim==1
        self.config_values = config_values[:,np.newaxis]

    def __call__(self,samples):
        multif_samples = np.vstack(
            (samples,np.tile(self.config_values,(1,samples.shape[1]))))
        return self.model(multif_samples)

def default_map_to_multidimensional_index(num_config_vars,indices):
    indices = np.atleast_2d(indices)
    assert indices.ndim==2 and indices.shape[0]==1
    multiindex_indices = np.empty(
        (num_config_vars,indices.shape[1]),dtype=indices.dtype)
    for jj in range(indices.shape[1]):
        multiindex_indices[:,jj] = indices[0,jj]
    return multiindex_indices

class MultiLevelWrapper(object):
    """
    Specify a one-dimension model hierachy from a multiple dimensional hierarchy
    For example if model has configure variables which refine the x and y
    physical directions then one can specify a multilevel hierarchy by creating
    new indices with the mapping k=(i,i).

    map_to_multidimensional_index : callable
        Function which maps 1D model index to multi-dimensional index

    See function default_map_to_multidimensional_index
    """
    def __init__(self,model,num_config_vars,multiindex_cost_function,
                 map_to_multidimensional_index=None):
        self.model=model
        self.num_config_vars=num_config_vars
        self.multiindex_cost_function=multiindex_cost_function
        if map_to_multidimensional_index is None:
            self.map_to_multidimensional_index=\
                partial(default_map_to_multidimensional_index,num_config_vars)
        else:
            self.map_to_multidimensional_index=map_to_multidimensional_index
            
        self.num_evaluations=0

    def __call__(self,samples):
        config_values = self.map_to_multidimensional_index(samples[-1:,:])
        assert config_values.shape[0]==self.num_config_vars
        multi_index_samples = np.vstack((samples[:-1],config_values))
        return self.model(multi_index_samples)
    
    def cost_function(self,multilevel_indices):
        indices = self.map_to_multidimensional_index(multilevel_indices)
        return self.multiindex_cost_function(indices)

    @property
    def num_evaluations(self):
        return self.model.num_evaluations

    @num_evaluations.setter
    def num_evaluations(self,nn):
        self.__num_evaluations=nn
        self.model.num_evaluations=nn
