r"""
Model interfacing
-----------------
This tutorial demonstrats how to use model wrappers to time function calls and evaluate a function at multiple samples in parallel.

"""
#%%
#Timing function evaluations
#^^^^^^^^^^^^^^^^^^^^^^^^^^^
#It is often useful to be able to track the time needed to evaluate a function. We can track this using the :class:`pyapprox.interface.wrappers.TimerModel` and :class:`pyapprox.interface.wrappers.WorkTrackingModel` objects which are designed to work together. The former time each evaluation of a function that returns output of shape (nsampels,qoi) and appends the time to the quantities of interest returned by the function, i.e returns a 2D np.ndarray with shape (nsamples,qoi+1). The second extracts the time and removes it from the quantities of interest and returns output with the original shape  (nsmaples,nqoi) of the user function.
#
#Lets use the class with a function that takes a random amount of time. We will use the previous function but add a random pause between 0 and .1 seconds. Lets import some functions and define a multi-variate random variable
#
#.. literalinclude:: ../../../examples/__util.py
#  :language: python
#  :start-at: def fun_pause_1
#  :end-before: def fun_pause_2
#
#.. Note for some reason text like this is needed after the literalinclude
#.. Also note that path above is relative to source/auto_examples
#

import os
import time
import numpy as np
from scipy import stats

from pyapprox.variables import IndependentMarginalsVariable
from pyapprox.interface.wrappers import (
    TimerModel, WorkTrackingModel, PoolModel, ModelEnsemble
)

univariate_variables = [stats.uniform(-2, 4), stats.uniform(-2, 4)]
variable = IndependentMarginalsVariable(univariate_variables)

from __util import pyapprox_fun_1, fun_pause_1
timer_fun = TimerModel(pyapprox_fun_1)
worktracking_fun = WorkTrackingModel(timer_fun)

nsamples = 10
samples = variable.rvs(nsamples)
values = worktracking_fun(samples)

#%%
#The :class:`pyapprox.interface.wrappers.WorkTrackingModel` has an attribute :class:`pyapprox.interface.wrappers.WorkTracker` which stores the execution time of each function evaluation as a dictionary. The key corresponds is the model id. For this example the id will always be the same, but the id can vary and this is useful when evaluating mutiple models, e.g. when using multi-fidelity methods. To print the dictionary use
costs = worktracking_fun.work_tracker.costs
print(costs)

#%%
#We can also call the work tracker to query the median cost for a model with a given id. The default id is 0.
fun_id = np.atleast_2d([0])
print(worktracking_fun.work_tracker(fun_id))


#%%
#Evaluating multiple models
#^^^^^^^^^^^^^^^^^^^^^^^^^^
#Now let apply this two an ensemble of models to explore the use of model ids. First create a second function which we import.
#
#.. literalinclude:: ../../../examples/__util.py
#  :language: python
#  :start-at: def fun_pause_2
#
#.. Note for some reason text like this is needed after the literalinclude
#.. Also note that path above is relative to source/auto_examples
#
from __util import pyapprox_fun_2

#%%
#Now using :class:`pyapprox.interface.ModelEnsemble` we can create a function which takes the random samples plus an additional configure variable which defines which model to evaluate. Lets use half the samples to evaluate the first model and evaluate the second model at the remaining samples
model_ensemble = ModelEnsemble([pyapprox_fun_1, pyapprox_fun_2])
timer_fun_ensemble = TimerModel(model_ensemble)
worktracking_fun_ensemble = WorkTrackingModel(
    timer_fun_ensemble, num_config_vars=1)

fun_ids = np.ones(nsamples)
fun_ids[:nsamples//2] = 0
ensemble_samples = np.vstack([samples, fun_ids])
values = worktracking_fun_ensemble(ensemble_samples)

#%%
#Here we had to pass the number (1) of configure variables to the
#WorkTrackingModel. PyApprox assumes that the configure variables are the last rows of the samples 2D array
#
#Now check that the new values are the same as when using the individual functions directly
assert np.allclose(values[:nsamples//2],
                   pyapprox_fun_1(samples[:, :nsamples//2]))
assert np.allclose(values[nsamples//2:],
                   pyapprox_fun_2(samples[:, nsamples//2:]))

#%%
#Again we can query the execution times of each model
costs = worktracking_fun_ensemble.work_tracker.costs
print(costs)

query_fun_ids = np.atleast_2d([0, 1])
print(worktracking_fun_ensemble.work_tracker(query_fun_ids))

#%%
#As expected there are 5 samples tracked for each model and the median evaluation time of the second function is about twice as large as for the first function.

#%%
#Evaluating functions at multiple samples in parallel
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#For expensive models it is often useful to be able to evaluate each model concurrently. This can be achieved using :class:`pyapprox.interface.wrappers.PoolModel`. Note this function is not intended for use with distributed memory systems, but rather is intended to use all the threads of a personal computer or compute node. See :class:`pyapprox.interface.async_model.AynchModel` if you are interested in running multiple simulations in parallel on a distributed memory system.
#
#PoolModel cannot be used to wrap WorkTrackingModel. However it can still
#be used with WorkTrackingModel using the sequence of wrappers below.

max_eval_concurrency = 1  # set higher
# clear WorkTracker counters
pool_model = PoolModel(
    timer_fun_ensemble, max_eval_concurrency, assert_omp=False)
worktracking_fun_ensemble.work_tracker.costs = dict()
worktracking_fun_ensemble = WorkTrackingModel(
    pool_model, num_config_vars=1)

# create more samples to notice improvement in wall time
nsamples = 10
samples = variable.rvs(nsamples)
fun_ids = np.ones(nsamples)
fun_ids[:nsamples//2] = 0
ensemble_samples = np.vstack([samples, fun_ids])

t0 = time.time()
values = worktracking_fun_ensemble(ensemble_samples)
t1 = time.time()
print(f'With {max_eval_concurrency} threads that took {t1-t0} seconds')

if ('OMP_NUM_THREADS' not in os.environ or
    int(os.environ['OMP_NUM_THREADS']) != 1):
    # make sure to set OMP_NUM_THREADS=1 to maximize benefit of pool model
    print('Warning set OMP_NUM_THREADS=1 for best performance')
max_eval_concurrency = 4
pool_model.set_max_eval_concurrency(max_eval_concurrency)
t0 = time.time()
values = worktracking_fun_ensemble(ensemble_samples)
t1 = time.time()
print(f'With {max_eval_concurrency} threads that took {t1-t0} seconds')

#%%
#Lets print a summary of the costs to make sure individual function evaluation
#costs are still being recorded correctly

print(worktracking_fun_ensemble.work_tracker)

#%%
#Note
#^^^^
#PoolModel cannot be used with lambda functions. You will get error similar to pickle.PicklingError: Can't pickle <function <lambda> at 0x12b4e6440>: attribute lookup <lambda> on __main__ failed

# sphinx_gallery_thumbnail_path = './figures/cantilever-beam.png'

#%%
#.. gallery thumbnail will say broken if no plots are made in this file so
#.. specify a default file as above. Must start with a #

