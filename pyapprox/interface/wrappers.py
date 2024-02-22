import time
import numpy as np
import subprocess
import os
import glob
from functools import partial
from multiprocessing import Pool
import pickle
import copy
# from tqdm import tqdm

from pyapprox.util.utilities import (
    get_all_sample_combinations, hash_array, cartesian_product
)
from pyapprox.util.sys_utilities import has_kwarg


def evaluate_1darray_function_on_2d_array(
        function, samples, statusbar=False, return_grad=False):
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

    statusbar : boolean
        True - print status bar showing progress to stdout
        False - do not print

    return_grad : boolean
        True - values and return gradient
        False - return just gradient
        If function does not accept the return_grad kwarg an exception will
         be raised

    Returns
    -------
    values : np.ndarray (num_samples, num_qoi)
        The value of each requested QoI of the model for each sample
    """
    has_return_grad = has_kwarg(function, "return_grad")
    if return_grad and not has_return_grad:
        msg = "return_grad set to true but function does not return grad"
        raise ValueError(msg)

    assert samples.ndim == 2
    num_samples = samples.shape[1]
    grads = []
    if not has_return_grad or return_grad is False:
        values_0 = function(samples[:, 0])
    else:
        values_0, grad_0 = function(samples[:, 0], return_grad=return_grad)
        assert grad_0.ndim == 1
        grads.append(grad_0)
    values_0 = np.atleast_1d(values_0)
    assert values_0.ndim == 1, values_0.shape
    num_qoi = values_0.shape[0]
    values = np.empty((num_samples, num_qoi), float)
    values[0, :] = values_0
    # if statusbar:
    #     pbar = tqdm(total=num_samples)
    #     pbar.update(1)
    for ii in range(1, num_samples):
        if not has_return_grad or return_grad is False:
            values[ii, :] = function(samples[:, ii])
        else:
            val_ii, grad_ii = function(samples[:, ii], return_grad=return_grad)
            values[ii, :] = val_ii
            grads.append(grad_ii)
        # if statusbar:
        #     pbar.update(1)
    if not return_grad:
        return values
    if num_qoi == 1:
        return values, np.vstack(grads)
    return values, grads


class PyFunction(object):
    def __init__(self, function):
        self.function = function

    def __call__(self, samples, opts=dict()):
        return evaluate_1darray_function_on_2d_array(
            self.function, samples, opts)


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
    r"""
    Create a queriable function that stores samples and
    associated function values and returns stored values
    for samples in the database or otherwise evaluate the
    function.
    """

    def __init__(self, function, data=None, data_basename=None,
                 save_frequency=None, use_hash=True, digits=16,
                 base_model=None):
        """
        Parameters
        ----------
        function : callable
            A function with signature

            ``function(w) -> np.ndarray (nsamples,nqoi+1)``

             where ``w`` is a np.ndarray of shape (nvars,nsamples).
             The last qoi returned by function (i.e. the last column of the
             output array) must be the cost of the simulation. This column
             is removed from the output of __call__.

        data : tuple
            (samples, values) of any previously computed previously samples
            and associated values

        data_basename : string
            The basename of the file used to store the database of samples and
            values.

        save_frequency : integer
            The number of function evaluations run before data is saved.
            E.g. if save frequency is 10 and __call__(samples) is run
            with samples containing 30 samples the values and data will
            be stored at 3 checkpoint, i.e. after 10, 20 and 30 samples
            have been evaluated

        use_hash : boolean
            True - hash samples to determine if values have already been
            collected
            False - np.allclose is used to match samples by looping over
            all samples in the database. This is slower.

        digits : integer
            The number of significant digits used to has or compare samples
            in the database
        """
        self.function = function
        self.base_model = base_model

        self.data = dict()
        self.samples = np.zeros((0, 0))
        self.values = None
        self.grads = None
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
            self.samples, self.values, self.grads = data
            assert self.samples.shape[1] == self.values.shape[0]
            self.add_new_data(data)

    def hash_sample(self, sample):
        # if samples have undergone a transformation thier value
        # may not be exactly the same so make hash on samples
        # with fixed precision
        # sample = np.round(sample, self.digits)
        # I = np.where(np.abs(sample)<self.tol)[0]
        # sample[I] = 0.
        key = hash_array(sample)  # ,decimals=self.digits)
        return key

    def add_new_data(self, data):
        samples, values, grads = data
        for ii in range(samples.shape[1]):
            if self.use_hash:
                key = self.hash_sample(samples[:, ii])
                if key in self.data:
                    if not np.allclose(
                            self.values[self.data[key]], values[ii]):
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
                    if grads is not None:
                        self.grads += grads[ii:ii+1]
                else:
                    self.samples = samples[:, ii:ii+1]
                    self.values = values[ii:ii+1, :]
                    if grads is not None:
                        self.grads = grads[ii:ii+1]

        # set counter so that next file takes into account all previously
        # ran samples
        self.num_evaluations_ran = self.samples.shape[1]

    def _batch_call(self, samples, return_grad):
        assert self.save_frequency > 0
        num_batch_samples = self.save_frequency
        lb = 0
        vals = None
        grads = None
        while lb < samples.shape[1]:
            ub = min(lb+num_batch_samples, samples.shape[1])
            num_evaluations_ran = self.num_evaluations_ran
            batch_vals, batch_grads, new_sample_indices = self._call(
                samples[:, lb:ub], return_grad)
            if return_grad:
                grads_4_save = [batch_grads[ii] for ii in new_sample_indices]
            else:
                grads_4_save = [None for ii in new_sample_indices]
            # I think this code will work only if function always does or does
            # not return grads
            if vals is None:
                vals = batch_vals
                grads = copy.deepcopy(batch_grads)
            else:
                vals = np.vstack((vals, batch_vals))
                grads += copy.deepcopy(batch_grads)

            if len(new_sample_indices) == 0:
                lb = ub
                continue
            data_filename = self.data_basename+'-%d-%d.pkl' % (
                num_evaluations_ran,
                num_evaluations_ran+len(new_sample_indices)-1)
            # np.savez(data_filename, vals=batch_vals[new_sample_indices],
            #          samples=samples[:, lb:ub][:, new_sample_indices],
            #          grads=grads_4_save)
            with open(data_filename, "wb") as f:
                pickle.dump((
                    batch_vals[new_sample_indices],
                    samples[:, lb:ub][:, new_sample_indices], grads_4_save), f)
            lb = ub

        if not return_grad:
            return vals
        return vals, grads

    def _call(self, samples, return_grad):
        has_return_grad = has_kwarg(self.function, "return_grad")
        if return_grad and not has_return_grad:
            msg = "return_grad set to true but function does not return return_grad"
            raise ValueError(msg)

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
            if not has_return_grad or not return_grad:
                new_values = self.function(new_samples)
                num_qoi = new_values.shape[1]
                new_grads = None
            else:
                new_values, new_grads = self.function(
                    new_samples, return_grad=return_grad)
                num_qoi = new_values.shape[1]

        else:
            num_qoi = self.values.shape[1]

        values = np.empty((samples.shape[1], num_qoi), dtype=float)
        grads = [None for ii in range(samples.shape[1])]
        if len(new_sample_indices) > 0:
            values[new_sample_indices, :] = new_values
            new_grads_list = [None for ii in range(len(new_sample_indices))]
            if new_grads is not None:
                for ii in range(len(new_sample_indices)):
                    # TODO need to make sure fun(sampels, return_grad)=True
                    # can return list of grads for each sample
                    # or a 2d array if nqoi=1
                    new_grads_list[ii] = np.atleast_2d(new_grads[ii]).copy()
                    grads[new_sample_indices[ii]] = copy.deepcopy(
                        new_grads[ii])
            new_grads = new_grads_list

        if len(new_sample_indices) < samples.shape[1]:
            values[evaluated_sample_indices[:, 0]] = \
                self.values[evaluated_sample_indices[:, 1], :]
            if has_return_grad:
                for ii in range(evaluated_sample_indices.shape[0]):
                    grads[evaluated_sample_indices[ii, 0]] = copy.deepcopy(
                        self.grads[
                            evaluated_sample_indices[ii, 1]])
        if len(new_sample_indices) > 0:
            if self.samples.shape[1] == 0:
                jj = 0
                self.samples = samples
                self.values = values
                if has_return_grad:
                    self.grads = copy.deepcopy(grads)
            else:
                jj = self.samples.shape[0]
                self.samples = np.hstack(
                    (self.samples, samples[:, new_sample_indices]))
                self.values = np.vstack((self.values, new_values))
                if has_return_grad:
                    self.grads += copy.deepcopy(new_grads)

            for ii in range(len(new_sample_indices)):
                key = hash_array(samples[:, new_sample_indices[ii]])
                self.data[key] = jj+ii

            self.num_evaluations_ran += len(new_sample_indices)
            # increment the number of samples pass to __call__ since object
            # created
            # includes samples drawn from arxiv and samples used to evaluate
            # self.function
        self.num_evaluations += samples.shape[1]

        return values, grads, new_sample_indices

    @staticmethod
    def _grads_valid(grads):
        for g in grads:
            if g is None:
                # TODO remove exception and rerun samples to get grads
                msg = "return_grad=True but previous samples evaluated did "
                msg += "not have grads"
                raise ValueError(msg)

    def __call__(self, samples, return_grad=False):
        if self.save_frequency is not None and self.save_frequency > 0:
            out = self._batch_call(samples, return_grad)
            if not return_grad:
                return out
            self._grads_valid(out[1])
            return out

        values, grads = self._call(samples, return_grad)[:-1]
        if return_grad:
            self._grads_valid(grads)
            return values, grads
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
    if max_eval_concurrency == 1:
        return model(samples)

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

    if type(result[0]) == tuple:
        assert len(result[0]) == 2
        num_qoi = result[0][0].shape[1]
        return_grad = True
        return_grads = []
    else:
        num_qoi = result[0].shape[1]
        return_grad = False
    values = np.empty((num_samples, num_qoi))
    for ii in range(len(result)):
        if not return_grad:
            values[ii, :] = result[ii][0, :]
        else:
            values[ii, :] = result[ii][0][0, :]
            if type(result[ii][1]) == list:
                return_grads += result[ii][1]
            else:
                return_grads += [result[ii][1]]
    if not return_grad:
        return values
    return values, return_grads


def time_function_evaluations(function, samples, return_grad=False):
    has_return_grad = has_kwarg(function, "return_grad")
    if return_grad and not has_return_grad:
        msg = "return_grad set to true but function does not return grad"
        raise ValueError(msg)

    vals = []
    grads = []
    times = []
    has_return_grad = has_kwarg(function, "return_grad")
    for ii in range(samples.shape[1]):
        t0 = time.time()
        if not has_return_grad or not return_grad:
            val = function(samples[:, ii:ii+1])[0, :]
        else:
            out = function(samples[:, ii:ii+1], return_grad=return_grad)
            val = out[0][0, :]
            grads.append(out[1])
        t1 = time.time()
        vals.append(val)
        times.append([t1-t0])
    vals = np.asarray(vals)
    times = np.asarray(times)
    if len(grads) == 0:
        return np.hstack([vals, times])
    if vals.shape[1] == 1:
        grads = np.vstack(grads)
    return np.hstack([vals, times]), grads


class TimerModel(object):
    r"""
    Return the wall-time needed to evaluate a function at each sample
    as an additional quantity of interest.
    """

    def __init__(self, function, base_model=None):
        """
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
        """
        self.function_to_time = function
        self.base_model = base_model

    # def x__getattr__(self, name):
    #     """
    #     Cannot get following to work

    #     If defining a custom __getattr__ it seems I cannot have member
    #     variables with the same name in this class and class definition
    #     of function

    #     if self.function is itself a model object allow the access of
    #     self.function.name using self.name

    #     Note  __getattr__
    #     will be invoked on python objects only when the requested
    #     attribute is not found in the particular object's space.
    #     """

    #     if hasattr(self.function_to_time, name):
    #         attr = getattr(self.function_to_time, name)
    #         return attr

    #     raise AttributeError(
    #         f" {self} or its member {self}.function has no attribute '{name}'")

    def __call__(self, samples, return_grad=False):
        return time_function_evaluations(
            self.function_to_time, samples, return_grad=return_grad)


class WorkTracker(object):
    r"""
    Store the cost needed to evaluate a function under different
    configurations, e.g. mesh resolution of a finite element model
    used to solve a PDE.
    """

    def __init__(self):
        self.costs = dict()

    def __call__(self, config_samples=None):
        """
        Read the cost of evaluating the functions with the ids given in
        a set of config_samples.

        Parameters
        ----------
        config_samples : np.ndarray (nconfig_vars,nsamples)
            The configuration indices. If None the default Id [0] is used
        """
        if config_samples is None:
            config_samples = np.asarray([[0]])
        num_config_vars, nqueries = config_samples.shape
        costs = np.empty((nqueries))
        for ii in range(nqueries):
            # key = tuple([int(ll) for ll in config_samples[:, ii]])
            key = tuple([ll for ll in config_samples[:, ii]])
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
            # key = tuple([int(ll) for ll in config_samples[:, ii]])
            key = tuple([ll for ll in config_samples[:, ii]])
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


# def eval(function, samples):
#     return function(samples)


class WorkTrackingModel(object):
    r"""
    Keep track of the wall time needed to evaluate a function.
    """

    def __init__(self, function, base_model=None, num_config_vars=0,
                 enforce_timer_model=True):
        """
        Keep track of the wall time needed to evaluate a function.

        Parameters
        ----------
        function : callable
            A function with signature

            ``function(w) -> np.ndarray (nsamples, nqoi+1)``

             where ``w`` is a np.ndarray of shape (nvars,nsamples).
             The last qoi returned by function (i.e. the last column of the
             output array) must be the cost of the simulation. This column
             is removed from the output of __call__.
model = WorkTrackingModel(TimerModel(benchmark.fun), num_config_vars=1)
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

        enforce_timer_model : boolean
            If True function must be an instance of TimerModel.
            If False function must return qoi plus an additional column
            which is the model run time

        Notes
        -----
        If defining a custom __getattr__ it seems I cannot have member
        variables with the same name in this class and class definition
        of function
        """
        self.wt_function = function
        if enforce_timer_model and not isinstance(function, TimerModel):
            raise ValueError("Function is not an instance of TimerModel")
        self.work_tracker = WorkTracker()
        self.base_model = base_model
        self.num_config_vars = num_config_vars

    def __call__(self, samples, return_grad=False):
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
        has_return_grad = has_kwarg(self.wt_function, "return_grad")
        if return_grad and not has_return_grad:
            msg = "return_grad set to true but function does not return grad"
            raise ValueError(msg)

        if not has_return_grad or not return_grad:
            data = self.wt_function(samples)
        else:
            data, grads = self.wt_function(samples, return_grad=return_grad)
        if data.shape[1] <= 1:
            raise RuntimeError(
                "function did not return at least one QoI and time")
        values = data[:, :-1]
        work = data[:, -1]
        if self.num_config_vars > 0:
            config_samples = samples[-self.num_config_vars:, :]
        else:
            config_samples = np.zeros((1, samples.shape[1]))
        self.work_tracker.update(config_samples, work)
        if not return_grad:
            return values
        return values, grads

    def cost_function(self, config_samples=None):
        """
        Retrun the cost of evaluating the functions with the ids given in
        a set of config_samples. These samples are assumed to be in user space
        not canonical space

        Parameters
        ----------
        config_samples : np.ndarray (nconfig_vars, nsamples)
            The configuration indices
        """
        if config_samples is None:
            config_samples = np.zeros((1, 1))
        cost = self.work_tracker(config_samples)
        return cost

    def __repr__(self):
        if self.base_model is not None:
            return "{0}(base_model={1})".format(
                self.__class__.__name__, self.base_model)
        return "{0}(model={1})".format(
            self.__class__.__name__, self.wt_function)


class PoolModel(object):
    r"""
    Evaluate a function at multiple samples in parallel using
    multiprocessing.Pool
    """

    def __init__(self, function, max_eval_concurrency, assert_omp=True,
                 base_model=None):
        """
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

    def __call__(self, samples, verbose=False, return_grad=False):
        """
        Evaluate a function at multiple samples in parallel using
        multiprocessing.Pool

        Parameters
        ----------
        samples : np.ndarray (nvars,nsamples)
            Samples used to evaluate self.function
        """
        has_return_grad = has_kwarg(self.pool_function, "return_grad")
        if return_grad and not has_return_grad:
            msg = "return_grad set to true but function does not return grad"
            raise ValueError(msg)

        if not has_return_grad or not return_grad:
            fun = self.pool_function
        else:
            fun = partial(self.pool_function, return_grad=return_grad)

        t0 = time.time()
        vals = run_model_samples_in_parallel(
            fun, self.max_eval_concurrency, samples,
            pool=None, assert_omp=self.assert_omp)
        if verbose:
            msg = f"Evaluating all {samples.shape[1]} samples took "
            msg += f"{time.time()-t0} seconds"
            print(msg)
        return vals


def get_active_set_model_from_variable(function, variable, active_var_indices,
                                       nominal_values, base_model=None):
    from pyapprox.variables.joint import IndependentMarginalsVariable
    active_variable = IndependentMarginalsVariable(
        [variable.marginals()[ii] for ii in active_var_indices])
    mask = np.ones(variable.num_vars(), dtype=bool)
    mask[active_var_indices] = False
    inactive_var_values = nominal_values[mask]
    model = ActiveSetVariableModel(
        function, variable.num_vars(), inactive_var_values, active_var_indices,
        base_model=base_model)
    return model, active_variable


class ActiveSetVariableModel(object):
    r"""
    Create a model wrapper that only accepts a subset of the model variables.
    """

    def __init__(self, function, num_vars, inactive_var_values,
                 active_var_indices, base_model=None):
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
        self.base_model = base_model

    def _expand_samples(self, reduced_samples):
        assert reduced_samples.ndim == 2
        raw_samples = get_all_sample_combinations(
            self.inactive_var_values, reduced_samples)
        samples = np.empty_like(raw_samples)
        samples[self.inactive_var_indices,
                :] = raw_samples[:self.inactive_var_indices.shape[0]]
        samples[self.active_var_indices,
                :] = raw_samples[self.inactive_var_indices.shape[0]:]
        return samples

    def __call__(self, reduced_samples, return_grad=False):
        has_return_grad = has_kwarg(self.function, "return_grad")
        if return_grad and not has_return_grad:
            msg = "return_grad set to true but function does not return grad"
            raise ValueError(msg)
        samples = self._expand_samples(reduced_samples)
        if not has_return_grad:
            return self.function(samples)
        return self.function(samples, return_grad)

    def num_active_vars(self):
        return len(self.inactive_var_indices)


def combine_saved_model_data(saved_data_basename):
    filenames = glob.glob(saved_data_basename+'*.pkl')
    ii = 0
    grads = None
    for filename in filenames:
        # data = np.load(filename, allow_pickle=True)
        # data_vals, data_samples, data_grads = (
        #     data["vals"], data["samples"], data["grads"]
        with open(filename, "rb") as f:
            data_vals, data_samples, data_grads = pickle.load(f)
        if ii == 0:
            vals = data_vals
            samples = data_samples
            grads = data_grads
        else:
            vals = np.vstack((vals, data_vals))
            samples = np.hstack((samples, data_samples))
            grads += data_grads
        ii += 1
    if len(filenames) == 0:
        return None, None, None

    return samples, vals, grads


class SingleFidelityWrapper(object):
    r"""
    Create a single fidelity model that fixes the configuration variables
    to user-defined nominal values.
    """

    def __init__(self, model, config_values, base_model=None):
        self.model = model
        assert config_values.ndim == 1
        self.config_values = config_values[:, np.newaxis]
        if base_model is None:
            base_model = model
        self.base_model = base_model

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
    r"""
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


class ModelEnsemble(object):
    r"""
    Wrapper class to allow easy one-dimensional
    indexing of models in an ensemble.
    """

    def __init__(self, functions, names=None):
        r"""
        Parameters
        ----------
        functions : list of callable
            A list of functions defining the model ensemble. The functions must
            have the call signature values=function(samples)
        """
        self.functions = functions
        self.nmodels = len(self.functions)
        if names is None:
            names = ['f%d' % ii for ii in range(self.nmodels)]
        self.names = names

    def evaluate_at_separated_samples(self, samples_list, active_model_ids):
        r"""
        Evaluate a set of models at different sets of samples.
        The models need not have the same parameters.

        Parameters
        ----------
        samples_list : list[np.ndarray (nvars_ii, nsamples_ii)]
            Realizations of the multivariate random variable model to evaluate
            each model.

        active_model_ids : iterable
            The models to evaluate

        Returns
        -------
        values_list : list[np.ndarray (nsamples, nqoi)]
            The values of the models at the different sets of samples

        """
        values_0 = self.functions[active_model_ids[0]](samples_list[0])
        assert values_0.ndim == 2
        values_list = [values_0]
        for ii in range(1, active_model_ids.shape[0]):
            values_list.append(self.functions[active_model_ids[ii]](
                samples_list[ii]))
        return values_list

    def evaluate_models(self, samples_per_model):
        """
        Evaluate a set of models at a set of samples.

        Parameters
        ----------
        samples_per_model : list (nmodels)
            The ith entry contains the set of samples
            np.narray(nvars, nsamples_ii) used to evaluate the ith model.

        Returns
        -------
        values_per_model : list (nmodels)
            The ith entry contains the set of values
            np.narray(nsamples_ii, nqoi) obtained from the ith model.
        """
        nmodels = len(samples_per_model)
        if nmodels != self.nmodels:
            raise ValueError("Samples must be provided for each model")
        nvars = samples_per_model[0].shape[0]
        nsamples = np.sum([ss.shape[1] for ss in samples_per_model])
        samples = np.empty((nvars+1, nsamples))
        cnt = 0
        ubs = np.cumsum([ss.shape[1] for ss in samples_per_model])
        lbs = np.hstack((0, ubs[:-1]))
        for ii, samples_ii in enumerate(samples_per_model):
            samples[:-1, lbs[ii]:ubs[ii]] = samples_ii
            samples[-1, lbs[ii]:ubs[ii]] = ii
            cnt += samples_ii.shape[1]
        values = self(samples)
        values_per_model = [values[lbs[ii]:ubs[ii]] for ii in range(nmodels)]
        return values_per_model

    def __call__(self, samples):
        r"""
        Evaluate a set of models at a set of samples. The models must have the
        same parameters.

        Parameters
        ----------
        samples : np.ndarray (nvars+1,nsamples)
            Realizations of a multivariate random variable each with an
            additional scalar model id indicating which model to evaluate.

        Returns
        -------
        values : np.ndarray (nsamples,nqoi)
            The values of the models at samples
        """
        model_ids = samples[-1, :]
        assert model_ids.max() < self.nmodels
        active_model_ids = np.unique(model_ids).astype(int)
        active_model_id = active_model_ids[0]
        II = np.where(model_ids == active_model_id)[0]
        values_0 = self.functions[active_model_id](samples[:-1, II])
        assert values_0.ndim == 2
        nqoi = values_0.shape[1]
        values = np.empty((samples.shape[1], nqoi))
        values[II, :] = values_0
        for ii in range(1, active_model_ids.shape[0]):
            active_model_id = active_model_ids[ii]
            II = np.where(model_ids == active_model_id)[0]
            values_II = self.functions[active_model_id](samples[:-1, II])
            assert values_II.ndim == 2
            values[II] = values_II
        return values

    def __repr__(self):
        return "{0}(nmodels={1})".format(
            self.__class__.__name__, self.nmodels)


class MultiIndexModel():
    """
    Define a multi-index model to be used for multi-index collocation

    Parameters
    ----------
    setup_model : callable
       Function with the signature

       ``setup_model(config_values) -> model_instance``

       where config_values np.ndarray (nconfig_vars, 1)
       defines the numerical resolution of model instance
       where model instance is a callable with the signature

       ``setup_model(random_samples) -> np.ndarray (nsamples, nqoi)``

       where random_samples is np.ndarray (nvars, nsamples).

    config_values : 2d list [nconfig_vars, nconfig_values_i]
        Contains the possible model discretization values for each config
        var, e.g. [[100, 200], [0.1, 0.2, 0.3]] would be use to specify
        a 1D spartial mesh with 100 elements or 200 elements and time step
        sizes of [0.1, 0.2, 0.3]
    """

    def __init__(self, setup_model, config_values):
        self._nconfig_vars = len(config_values)
        self._config_values = config_values
        self._model_ensemble, self._multi_index_to_model_id_map = (
            self._create_model_ensemble(setup_model, config_values))

    def _create_model_ensemble(self, setup_model, config_values):
        # config_var_trans = ConfigureVariableTransformation(config_values)
        config_samples = cartesian_product(config_values).astype(np.double)
        models = [None for ii in range(config_samples.shape[1])]
        multi_index_to_model_id_map = {}
        for ii in range(config_samples.shape[1]):
            models[ii] = setup_model(config_samples[:, ii])
            multi_index_to_model_id_map[hash_array(config_samples[:, ii])] = ii
        return ModelEnsemble(models), multi_index_to_model_id_map

    def __call__(self, samples):
        """
        Parameters
        ----------
        samples : np.ndarray (nvars+nconfig_vars, nsamples)
            Each row is the concatenation of a random sample and a
            configuration sample.

        Returns
        -------
        values : np.ndarray (nsamples, nqoi)
           Evaluations of the model at the samples.
        """
        nsamples = samples.shape[1]
        config_samples = samples[-self._nconfig_vars:, :]
        model_ids = np.empty((1, nsamples))
        for ii in range(nsamples):
            key = hash_array(config_samples[:, ii])
            if key not in self._multi_index_to_model_id_map:
                msg = f"Model ID for : {config_samples[:, ii]} not found."
                raise RuntimeError(msg)
            model_ids[0, ii] = self._multi_index_to_model_id_map[key]
        return self._model_ensemble(
            np.vstack((samples[:-self._nconfig_vars, :], model_ids)))

    def __repr__(self):
        return "{0}(nconfig_values_per_config_var={1})".format(
            self.__class__.__name__, [len(v) for v in self._config_values])


class ArchivedDataModel():
    def __init__(self, samples, values):
        # todo add gradients and hess vec prods as optional args

        if values.ndim != 2 or values.shape[0] != samples.shape[1]:
            msg = "values must have shape (nsamples, nqoi) but has shape"
            msg += f" {values.shape}"
            raise ValueError(msg)

        self.samples = samples
        self.values = values
        self._samples_dict = self._set_samples_dict(samples)
        # when randomness is None then rvs just iterates sequentially
        # through samples until none are left. Useful for methods
        # that require unique samples
        self._sample_cnt = 0
        self._samples_dict = self._set_samples_dict(samples)

    def _set_samples_dict(self, samples):
        samples_dict = dict()
        for ii in range(samples.shape[1]):
            key = self._hash_sample(samples[:, ii])
            if key in samples_dict:
                raise ValueError("Duplicate samples detected")
            samples_dict[key] = ii
        return samples_dict

    def num_vars(self):
        return self.samples.shape[0]

    def _hash_sample(self, sample):
        key = hash_array(sample)
        return key

    def __call__(self, samples):
        values = []
        for ii in range(samples.shape[1]):
            key = self._hash_sample(samples[:, ii])
            if key not in self._samples_dict:
                raise ValueError("Sample not found")
            sample_id = self._samples_dict[key]
            values.append(self.values[sample_id])
        return np.array(values)

    def rvs(self, nsamples, weights=None, randomness="wo_replacement",
            return_indices=False):
        """
        Randomly sample with replacement from all available samples
        if weights is None uniform weights are applied to each sample
        otherwise sample according to weights
        """
        if randomness is None:
            if self._sample_cnt+nsamples > self.samples.shape[1]:
                msg = "Too many samples requested when randomness is None. "
                msg += f"self._sample+cnt_nsamples={self._sample_cnt+nsamples}"
                msg += f" but only {self.samples.shape[1]} samples available"
                msg += " This can be overidden by reseting self._sample_cnt=0"
                raise ValueError(msg)
            indices = np.arange(self._sample_cnt, self._sample_cnt+nsamples,
                                dtype=int)
            self._sample_cnt += nsamples
        else:
            indices = np.random.choice(
                np.arange(self.samples.shape[1], dtype=int), nsamples,
                p=weights, replace=(randomness == "replacement"))
        if not return_indices:
            return self.samples[:, indices]
        else:
            return self.samples[:, indices], indices
