r"""
Model Timing
------------
 It is often useful to be able to track the time needed to evaluate a function. 
This tutorial demonstrats how to use model to time function calls.
All model class can track time.

First Lets define a model that takes a random amount of time.
"""

import time
import numpy as np
from scipy import stats

from pyapprox.variables import IndependentMarginalsVariable
from pyapprox.interface.model import (
    ModelFromSingleSampleCallable,
)


# %%
# Timing function evaluations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


# Setup function
def fun_pause(sample):
    assert sample.ndim == 1
    time.sleep(np.random.uniform(0, 0.05))
    return np.sum(sample**2)


def jac_pause(sample):
    assert sample.ndim == 1
    time.sleep(np.random.uniform(0, 0.05))
    return 2 * sample[None, :]


# Make it a model
model = ModelFromSingleSampleCallable(
    1, fun_pause, jacobian=jac_pause, sample_ndim=1, values_ndim=0
)

# Define the distrbution of the input variables
marginals = [stats.uniform(-2, 4), stats.uniform(-2, 4)]
variable = IndependentMarginalsVariable(marginals)

# %%
# Now lets run the model a
nsamples = 10
samples = variable.rvs(nsamples)
values = model(samples)

# %%
# All models contain a :class:`~pyapprox.interface.model.ModelWorkTracker` which stores the execution time of each function evaluation as a dictionary. The key corresponds to the type of model evaluation. Lets print some useful timing information

# print the wall time of all tyes of evaluations
print(model.work_tracker().wall_times())

# print the average wall time of just the function evaluations
print(model.work_tracker().average_wall_time("val"))

# print the number of evaluations
print(model.work_tracker().nevaluations("val"))

# print the work_tracker itself, which includes the number of evaluations and average wall_times for each type of evaluation. nan is returned if no evaluation has been performed
print(model.work_tracker())


# %%
# We have only tracked the time required to evaluate the model. However we can also time derivative computations. The following evaluates the jacobian and print the work_tracker which has been updated to capture the time taken to compute the jacobian
values = model.jacobian(samples[:, :1])
print(model.work_tracker())
