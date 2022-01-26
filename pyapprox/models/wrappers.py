import time
import numpy as np
import subprocess
import os
import glob
from functools import partial
from multiprocessing import Pool

from pyapprox.utilities import get_all_sample_combinations
from pyapprox.utilities import hash_array
from pyapprox.sys_utilities import get_num_args


def evaluate_1darray_function_on_2d_array(function, samples, opts=None):
    """
    Evaluate a function at a set of samples using a function that only takes
    one sample at a time

    Parameters
    ----------
    function : callable
        A function with signature

        ``function(sample) -> np.ndarray```

        where sample is a 1d np.ndarray of shape (num_vars) and the output is
        a np.ndarray of values of shape (num_qoi). The output can also be a
        scalar

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
    assert samples.ndim == 2
    num_samples = samples.shape[1]
    if num_args == 2:
        values_0 = function(samples[:, 0], opts)
    else:
        values_0 = function(samples[:, 0])
    values_0 = np.atleast_1d(values_0)
    assert values_0.ndim == 1
    num_qoi = values_0.shape[0]
    values = np.empty((num_samples, num_qoi), float)
    values[0, :] = values_0
    for i in range(1, num_samples):
        if num_args == 2:
            values[i, :] = function(samples[:, i], opts)
        else:
            values[i, :] = function(samples[:, i])

    return values


class PyFunction(object):
    def __init__(self, function):
        self.function = function

    def __call__(self, samples, opts=dict()):
        return evaluate_1darray_function_on_2d_array(self.function, samples, opts)


def run_shell_command(shell_command, opts={}):
    """
    Execute a shell command.

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
    output_verbosity = opts.get('verbosity', 1)
    env = opts.get('env', None)
    filename = opts.get('filename', None)

    if output_verbosity == 0:
        subprocess.check_output(shell_command, shell=True, env=env)
    elif output_verbosity == 1:
        if filename is None:
            filename = 'shell_command.out'
        with open(filename, 'w') as f:
            subprocess.call(shell_command, shell=True, stdout=f,
                            stderr=f, env=env)
    else:
        subprocess.call(shell_command, shell=True, env=env)


class DataFunctionModel(object):
    def hash_sample(self, sample):
        # if samples have undergone a transformation thier value
        # may not be exactly the same so make hash on samples
        # with fixed precision
        # sample = np.round(sample, self.digits)
        # I = np.where(np.abs(sample)<self.tol)[0]
        # sample[I] = 0.
        key = hash_array(sample)  # ,decimals=self.digits)
        return key

    def __init__(self, function, data=None, data_basename=None,
                 save_frequency=None, use_hash=True, digits=16):
        self.function = function

        self.data = dict()
        self.samples = np.zeros((0, 0))
        self.values = None
        self.num_evaluations_ran = 0
        self.num_evaluations = 0
        self.digits = digits
        self.tol = 10**(-self.digits)
        self.use_hash = use_hash

        self.data_basename = data_basename
        self.save_frequency = save_frequency
        if self.data_basename is not None:
            assert save_frequency is not None
        if self.save_frequency and self.data_basename is None:
            msg = 'Warning save_frequency not being used because data_basename'
            msg += ' is None'
            print(msg)

        if data_basename is not None:
            file_data = combine_saved_model_data(data_basename)
            if file_data[0] is not None:
                self.add_new_data(file_data)

        if data is not None:
            self.samples, self.values = data
            assert self.samples.shape[1] == self.values.shape[0]
            self.add_new_data(data)

    def add_new_data(self, data):
        samples, values = data
        for ii in range(samples.shape[1]):
            if self.use_hash:
                key = self.hash_sample(samples[:, ii])
                if key in self.data:
                    if not np.allclose(self.values[self.data[key]], values[ii]):
                        msg = 'Duplicate samples found but values do not match'
                        raise Exception(msg)
                    found = True
                else:
                    self.data[key] = ii
                    found = False
            else:
                found = False
                for jj in range(self.samples.shape[1]):
                    if np.allclose(self.samples[:, jj], samples[:, ii],
                                   atol=self.tol):
                        found = True
                        break
            if not found:
                if self.samples.shape[1] > 0:
                    self.samples = np.hstack(
                        [self.samples, samples[:, ii:ii+1]])
                    self.values = np.vstack([self.values, values[ii:ii+1, :]])
                else:
                    self.samples = samples[:, ii:ii+1]
                    self.values = values[ii:ii+1, :]

        # set counter so that next file takes into account all previously
        # ran samples
        self.num_evaluations_ran = self.samples.shape[1]

    def _batch_call(self, samples):
        assert self.save_frequency > 0
        num_batch_samples = self.save_frequency
        lb = 0
        vals = None
        while lb < samples.shape[1]:
            ub = min(lb+num_batch_samples, samples.shape[1])
            num_evaluations_ran = self.num_evaluations_ran
            batch_vals, new_sample_indices = self._call(samples[:, lb:ub])
            data_filename = self.data_basename+'-%d-%d.npz' % (
                num_evaluations_ran,
                num_evaluations_ran+len(new_sample_indices)-1)
            np.savez(data_filename, vals=batch_vals[new_sample_indices],
                     samples=samples[:, lb:ub][:, new_sample_indices])
            if vals is None:
                vals = batch_vals
            else:
                vals = np.vstack((vals, batch_vals))
            lb = ub
        return vals

    def _call(self, samples):
        evaluated_sample_indices = []
        new_sample_indices = []
        for ii in range(samples.shape[1]):
            if self.use_hash:
                key = self.hash_sample(samples[:, ii])
                if key in self.data:
                    evaluated_sample_indices.append([ii, self.data[key]])
                else:
                    new_sample_indices.append(ii)
            else:
                found = False
                for jj in range(self.samples.shape[1]):
                    if np.allclose(self.samples[:, jj], samples[:, ii],
                                   atol=self.tol):
                        found = True
                        break
                if found:
                    evaluated_sample_indices.append([ii, jj])
                else:
                    new_sample_indices.append(ii)

        evaluated_sample_indices = np.asarray(evaluated_sample_indices)
        if len(new_sample_indices) > 0:
            new_samples = samples[:, new_sample_indices]
            new_values = self.function(new_samples)
            num_qoi = new_values.shape[1]
        else:
            num_qoi = self.values.shape[1]

        values = np.empty((samples.shape[1], num_qoi), dtype=float)
        if len(new_sample_indices) > 0:
            values[new_sample_indices, :] = new_values
        if len(new_sample_indices) < samples.shape[1]:
            values[evaluated_sample_indices[:, 0]] = \
                self.values[evaluated_sample_indices[:, 1], :]

        if len(new_sample_indices) > 0:
            if self.samples.shape[1] == 0:
                jj = 0
                self.samples = samples
                self.values = values
            else:
                jj = self.samples.shape[0]
                self.samples = np.hstack(
                    (self.samples, samples[:, new_sample_indices]))
                self.values = np.vstack((self.values, new_values))

            for ii in range(len(new_sample_indices)):
                key = hash_array(samples[:, new_sample_indices[ii]])
                self.data[key] = jj+ii

            self.num_evaluations_ran += len(new_sample_indices)
        # increment the number of samples pass to __call__ since object created
        # includes samples drawn from arxiv and samples used to evaluate
        # self.function
        self.num_evaluations += samples.shape[1]

        return values, new_sample_indices

    def __call__(self, samples):
        if self.save_frequency is not None and self.save_frequency > 0:
            values = self._batch_call(samples)
        else:
            values = self._call(samples)[0]
        return values


def run_model_samples_in_parallel(model, max_eval_concurrency, samples,
                                  pool=None, assert_omp=True):
    """
    Warning
    -------
    pool.map serializes each argument and so if model is a class,
    any of its member variables that are updated in __call__ will not
    persist once each __call__ to pool completes.
    """
    num_samples = samples.shape[1]
    if assert_omp and max_eval_concurrency > 1:
        if ('OMP_NUM_THREADS' not in os.environ or
                not int(os.environ['OMP_NUM_THREADS']) == 1):
            msg = 'User set assert_omp=True but OMP_NUM_THREADS has not been '
            msg += 'set to 1. Run script with '
            msg += 'OMP_NUM_THREADS=1 python script.py'
            raise Exception(msg)

    if pool is None:
        pool_given = False
        pool = Pool(max_eval_concurrency)
    else:
        pool_given = True
    result = pool.map(
        model, [(samples[:, ii:ii+1]) for ii in range(samples.shape[1])])
    if pool_given is False:
        pool.close()

    # result  = [model(samples[:, ii:ii+1]) for ii in range(samples.shape[1])]
    num_qoi = result[0].shape[1]
    values = np.empty((num_samples, num_qoi))
    for ii in range(len(result)):
        values[ii, :] = result[ii][0, :]
    return values


def time_function_evaluations(function, samples):
    vals = []
    times = []
    for ii in range(samples.shape[1]):
        t0 = time.time()
        val = function(samples[:, ii:ii+1])[0, :]
        t1 = time.time()
        vals.append(val)
        times.append([t1-t0])
    vals = np.asarray(vals)
    times = np.asarray(times)
    return np.hstack([vals, times])


class TimerModelWrapper(object):
    def __init__(self, function, base_model=None):
        self.function_to_time = function
        self.base_model = base_model

    def x__getattr__(self, name):
        """
        Cannot get following to work

        If defining a custom __getattr__ it seems I cannot have member
        variables with the same name in this class and class definition
        of function

        if self.function is itself a model object allow the access of
        self.function.name using self.name

        Note  __getattr__
        will be invoked on python objects only when the requested
        attribute is not found in the particular object's space.
        """

        if hasattr(self.function_to_time, name):
            attr = getattr(self.function_to_time, name)
            return attr

        raise AttributeError(
            f" {self} or its member {self}.function has no attribute '{name}'")

    def __call__(self, samples):
        return time_function_evaluations(self.function_to_time, samples)


class WorkTracker(object):
    """
    Store the cost needed to evaluate a function under different
    configurations, e.g. mesh resolution of a finite element model
    used to solve a PDE.
    """

    def __init__(self):
        self.costs = dict()

    def __call__(self, config_samples):
        """
        Read the cost of evaluating the functions with the ids given in
        a set of config_samples.

        Parameters
        ----------
        config_samples : np.ndarray (nconfig_vars,nsamples)
            The configuration indices
        """
        num_config_vars, nqueries = config_samples.shape
        costs = np.empty((nqueries))
        for ii in range(nqueries):
            key = tuple([int(ll) for ll in config_samples[:, ii]])
            if key not in self.costs:
                msg = 'Asking for cost before function cost has been provided'
                raise Exception(msg)
            else:
                costs[ii] = np.median(self.costs[key])

        return costs

    def update(self, config_samples, costs):
        """
        Update the cost of evaluating the functions with the ids given in
        a set of config_samples.

        Parameters
        ----------
        config_samples : np.ndarray (nconfig_vars,nsamples)
            The configuration indices

        costs : np.ndarray (nsamples)
            The costs of evaluating the function index by each index in 
            ``config_samples``
        """
        num_config_vars, nqueries = config_samples.shape
        assert costs.shape[0] == nqueries
        assert costs.ndim == 1
        for ii in range(nqueries):
            key = tuple([int(ll) for ll in config_samples[:, ii]])
            if key in self.costs:
                self.costs[key].append(costs[ii])
            else:
                self.costs[key] = [costs[ii]]

    def __str__(self):
        msg = 'WorkTracker Cost Summary\n'
        msg += '{:<10} {:<10}\n'.format('Funtion ID', 'Median Cost')
        for item in self.costs.items():
            msg += '{:<10} {:<10}\n'.format(str(item[0]), np.median(item[1]))
        return msg


def eval(function, samples):
    return function(samples)


class WorkTrackingModel(object):
    def __init__(self, function, base_model=None, num_config_vars=0):
        """
        Keep track of the wall time needed to evaluate a function.

        Parameters
        ----------
        function : callable
            A function with signature

            ``function(w) -> np.ndarray (nsamples,nqoi+1)``

             where ``w`` is a np.ndarray of shape (nvars,nsamples).
             The last qoi returned by function (i.e. the last column of the
             output array) must be the cost of the simulation. This column
             is removed from the output of __call__.

        base_model : callable
            A function with signature

            ``base_model(w) -> float``

             where ``w`` is a np.ndarray of shape (nvars,nsamples).

             This is useful when function is a wrapper of another model, i.e.
             base_model and algorithms or the user want access to the attribtes
             of the base_model.

        num_config_vars : integer
             The number of configuration variables of fun. For most functions
             this will be zero.

        Notes
        -----
        If defining a custom __getattr__ it seems I cannot have member
        variables with the same name in this class and class definition
        of function
        """
        self.wt_function = function
        self.work_tracker = WorkTracker()
        self.base_model = base_model
        self.num_config_vars = num_config_vars

    def __call__(self, samples):
        """
        Evaluate self.function

        Parameters
        ----------
        samples : np.ndarray (nvars,nsamples)
            Samples used to evaluate self.function

        Returns
        -------
        values : np.ndarray (nsamples,nqoi)
            The values of self.function. The last qoi returned by self.function
            (i.e. the last column of the output array of size (nsamples,nqoi+1)
            is the cost of the simulation. This column is not included in
            values.
        """
        data = eval(self.wt_function, samples)
        values = data[:, :-1]
        work = data[:, -1]
        if self.num_config_vars > 0:
            config_samples = samples[-self.num_config_vars:, :]
        else:
            config_samples = np.zeros((1, samples.shape[1]))
        self.work_tracker.update(config_samples, work)
        return values

    def cost_function(self, config_samples):
        """
        Retrun the cost of evaluating the functions with the ids given in
        a set of config_samples. These samples are assumed to be in user space
        not canonical space

        Parameters
        ----------
        config_samples : np.ndarray (nconfig_vars,nsamples)
            The configuration indices
        """
        cost = self.work_tracker(config_samples)
        return cost


class PoolModel(object):
    def __init__(self, function, max_eval_concurrency, assert_omp=True,
                 base_model=None):
        """
        Evaluate a function at multiple samples in parallel using
        multiprocessing.Pool

        Parameters
        ----------
        function : callable
            A function with signature

            ``function(w) -> np.ndarray (nsamples,nqoi+1)``

             where ``w`` is a np.ndarray of shape (nvars,nsamples).

        max_eval_concurrency : integer
            The maximum number of simulations that can be run in parallel.
            Should be no more than the maximum number of cores on the computer
            being used

        assert_omp : boolean
            If True make sure that python is only using one thread per model
            instance. On OSX and Linux machines this means that the
            environement variable OMP_NUM_THREADS has been set to 1 with, e.g.
            export OMP_NUM_THREADS=1

            This is useful because often many python packages,
            e.g. SciPy, NumPy
            use multiple threads and this can cause running multiple
            evaluations of function to be slow because of resource allocation
            issues.

        base_model : callable
            A function with signature

            ``base_model(w) -> float``

             where ``w`` is a np.ndarray of shape (nvars,nsamples).

             This is useful when function is a wrapper of another model, i.e.
             base_model and algorithms or the user want access to the attribtes
             of the base_model.

        Notes
        -----
        If defining a custom __getattr__ it seems I cannot have member
        variables with the same name in this class and class definition
        of function
        """
        self.base_model = base_model
        self.set_max_eval_concurrency(max_eval_concurrency)
        self.num_evaluations = 0
        self.assert_omp = assert_omp
        self.pool_function = function

    def set_max_eval_concurrency(self, max_eval_concurrency):
        """
        Set the number of threads used to evaluate the function

        Parameters
        ----------
        max_eval_concurrency : integer
            The maximum number of simulations that can be run in parallel.
            Should be no more than the maximum number of cores on the computer
            being used
        """
        self.max_eval_concurrency = max_eval_concurrency

    def __call__(self, samples):
        """
        Evaluate a function at multiple samples in parallel using
        multiprocessing.Pool

        Parameters
        ----------
        samples : np.ndarray (nvars,nsamples)
            Samples used to evaluate self.function
        """
        vals = run_model_samples_in_parallel(
            self.pool_function, self.max_eval_concurrency, samples,
            pool=None, assert_omp=self.assert_omp)
        return vals


def get_active_set_model_from_variable(function, variable, active_var_indices,
                                       nominal_values):
    from pyapprox import IndependentMultivariateRandomVariable
    active_variable = IndependentMultivariateRandomVariable(
        [variable.all_variables()[ii] for ii in active_var_indices])
    mask = np.ones(variable.num_vars(), dtype=bool)
    mask[active_var_indices] = False
    inactive_var_values = nominal_values[mask]
    model = ActiveSetVariableModel(
        function, variable.num_vars(), inactive_var_values, active_var_indices)
    return model, active_variable


class ActiveSetVariableModel(object):
    def __init__(self, function, num_vars, inactive_var_values,
                 active_var_indices):
        # num_vars can de determined from inputs but making it
        # necessary allows for better error checking
        self.function = function
        assert inactive_var_values.ndim == 2
        self.inactive_var_values = np.asarray(inactive_var_values)
        self.active_var_indices = np.asarray(active_var_indices)
        assert self.active_var_indices.shape[0] + \
            self.inactive_var_values.shape[0] == num_vars
        self.num_vars = num_vars
        assert np.all(self.active_var_indices < self.num_vars)
        self.inactive_var_indices = np.delete(
            np.arange(self.num_vars), active_var_indices)

    def __call__(self, reduced_samples):
        raw_samples = get_all_sample_combinations(
            self.inactive_var_values, reduced_samples)
        samples = np.empty_like(raw_samples)
        samples[self.inactive_var_indices,
                :] = raw_samples[:self.inactive_var_indices.shape[0]]
        samples[self.active_var_indices,
                :] = raw_samples[self.inactive_var_indices.shape[0]:]
        return self.function(samples)

    def num_active_vars(self):
        return len(self.inactive_var_indices)


def combine_saved_model_data(saved_data_basename):
    filenames = glob.glob(saved_data_basename+'*.npz')
    ii = 0
    for filename in filenames:
        data = np.load(filename)
        if ii == 0:
            vals = data['vals']
            samples = data['samples']
        else:
            vals = np.vstack((vals, data['vals']))
            samples = np.hstack((samples, data['samples']))
        ii += 1
    if len(filenames) == 0:
        return None, None

    return samples, vals


class SingleFidelityWrapper(object):
    def __init__(self, model, config_values):
        self.model = model
        assert config_values.ndim == 1
        self.config_values = config_values[:, np.newaxis]

    def __call__(self, samples):
        multif_samples = np.vstack(
            (samples, np.tile(self.config_values, (1, samples.shape[1]))))
        return self.model(multif_samples)


def default_map_to_multidimensional_index(num_config_vars, indices):
    indices = np.atleast_2d(indices)
    assert indices.ndim == 2 and indices.shape[0] == 1
    multiindex_indices = np.empty(
        (num_config_vars, indices.shape[1]), dtype=indices.dtype)
    for jj in range(indices.shape[1]):
        multiindex_indices[:, jj] = indices[0, jj]
    return multiindex_indices


class MultiLevelWrapper(object):
    """
    Specify a one-dimension model hierachy from a multiple dimensional
    hierarchy
    For example if model has configure variables which refine the x and y
    physical directions then one can specify a multilevel hierarchy by creating
    new indices with the mapping k=(i,i).

    map_to_multidimensional_index : callable
        Function which maps 1D model index to multi-dimensional index

    See function default_map_to_multidimensional_index
    """

    def __init__(self, model, multiindex_num_config_vars,
                 map_to_multidimensional_index=None):
        self.model = model
        self.multiindex_num_config_vars = multiindex_num_config_vars
        if map_to_multidimensional_index is None:
            self.map_to_multidimensional_index =\
                partial(default_map_to_multidimensional_index,
                        multiindex_num_config_vars)
        else:
            self.map_to_multidimensional_index = map_to_multidimensional_index

        self.num_evaluations = 0
        self.num_config_vars = 1

    def __call__(self, samples):
        config_values = self.map_to_multidimensional_index(samples[-1:, :])
        assert config_values.shape[0] == self.multiindex_num_config_vars
        multi_index_samples = np.vstack((samples[:-1], config_values))
        return self.model(multi_index_samples)

    @property
    def num_evaluations(self):
        return self.model.num_evaluations

    @num_evaluations.setter
    def num_evaluations(self, nn):
        self.__num_evaluations = nn
        self.model.num_evaluations = nn
