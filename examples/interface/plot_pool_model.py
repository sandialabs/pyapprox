r"""
Parallel Model Evaluations
--------------------------
For expensive models it is often useful to be able to evaluate each model concurrently. This can be achieved using :class:`~pyapprox.interface.model.PoolModelWrapper`. Note this function is not intended for use with distributed memory systems, but rather is intended to use all the threads of a personal computer or compute node. See the documention of :class:`~pyapprox.interface.model.AyncIOModel` if you are interested in running multiple simulations in parallel on a distributed memory system.
"""

import os
import time
import numpy as np
from scipy import stats

from pyapprox.variables import IndependentMarginalsVariable
from pyapprox.interface.model import PoolModelWrapper
from pyapprox.interface.model import ModelFromSingleSampleCallable

nprocs = 1  # set higher

# %%
# Create a function that takes a nontrivial amount of time so we can see
# difference in time taken to evaluate a set of samples


# Define the function
def fun_pause(sample):
    assert sample.ndim == 1
    time.sleep(0.1)
    return np.sum(sample**2)


# Make it a model
nvars = 2
serial_model = ModelFromSingleSampleCallable(
    1, nvars, fun_pause, sample_ndim=1, values_ndim=0
)
# Define the input variables
marginals = [stats.uniform(-2, 4), stats.uniform(-2, 4)]
variable = IndependentMarginalsVariable(marginals)

# %%
# Now lets evaluate the model without the PoolModelWrapper
# and track the time taken
nsamples = 10
samples = variable.rvs(nsamples)

# activate the model work tracker
serial_model.work_tracker().set_active(True)
t0 = time.time()
values = serial_model(samples)
t1 = time.time()
print(f"Serial evaluations took {t1-t0} seconds")

# %%
# Now lets the PoolModelWrapper to use multiprocessing.pool to run the model
# in parallel
# Create a seperate model so that work_tracker of serial_model is not impacted
_model = ModelFromSingleSampleCallable(
    1, nvars, fun_pause, sample_ndim=1, values_ndim=0
)
pool_model = PoolModelWrapper(_model, nprocs, assert_omp=False)

# %%
# Check the user has set OMP_NUM_THREADS=1. If this is not set then
# performance will degrate
if (
    "OMP_NUM_THREADS" not in os.environ
    or int(os.environ["OMP_NUM_THREADS"]) != 1
):
    print("Warning set OMP_NUM_THREADS=1 for best performance")

# %%
# Activate the pool model work tracker and the work tracker of the model
# it wraps. These track two different costs. The pool_model tracks the
# cost including the overhead of using pool which copies data
# The wrapped model tracks the raw costs
pool_model.work_tracker().set_active(True)
pool_model.model().work_tracker().set_active(True)
t0 = time.time()
values = pool_model(samples)
t1 = time.time()
print(f"evaluations took {t1-t0} seconds with {nprocs=} ")

# %%
# Lets print a summary of the costs to make sure individual function evaluation
# costs are still being recorded correctly

print(serial_model.work_tracker())
print(pool_model.work_tracker())
print(pool_model.model().work_tracker())

# %%
# Note
# ^^^^
# PoolModel cannot be used with lambda functions. You will get error similar to pickle.PicklingError: Can't pickle <function <lambda> at 0x12b4e6440>: attribute lookup <lambda> on __main__ failed
#
# Because we use multiprocessing.Pool
# The .py script of this tutorial cannot be run with nprocs > 1
# via the shell command using python plot_pde_convergence.py because Pool
# must be called inside
#
# .. code-block:: python
#
#    if __name__ == '__main__':
#

# %%
# sphinx_gallery_thumbnail_path = './figures/cantilever-beam.png'
# .. gallery thumbnail will say broken if no plots are made in this file so
# .. specify a default file as above. Must start with a #
