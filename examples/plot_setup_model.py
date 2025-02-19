r"""
Model Definition
----------------
PyApprox can be used with pretty much any function provided an appropriate interface is defined. Here will show how to setup a simple function.

PyApprox requires all functions to take 2D array with shape (nvars, nsamples) and requires a function to return a 2D np.ndarray with shape (nsampels, nqoi). nqoi==1 for scalar valued functions and nqoi>1 for vectored value functions.

Lets define a function which does not match this criteria and use wrappers provided by PyApprox to convert it to the correct format. Specifically we will define a function that only takes a 1D np.ndarray and returns a scalar
"""

import numpy as np
from scipy import stats
from pyapprox.variables import IndependentMarginalsVariable
from pyapprox.interface.model import ModelFromSingleSampleCallable


def fun(samples):
    return np.sum(samples**3)


# %%
# Now lets create a pyapprox model
model = ModelFromSingleSampleCallable(1, 2, fun, sample_ndim=1, values_ndim=0)

# %%
# Now lets define the inputs to the function of interest. For independent random variables we use SciPy random variablest to represent each one-dimensional variables. For documentation refer to the `scipy.stats module <https://docs.scipy.org/doc/scipy/reference/stats.html>`_.
#
# We define multivariate random variables by specifying each 1D variable in a list. Here we will setup a 2D variable which is the tensor product of two independent and identically distributed uniform random variables

marginals = [stats.uniform(-2, 4), stats.uniform(-2, 4)]
variable = IndependentMarginalsVariable(marginals)

# %%
# To print a summary of the random variable use
print(variable)

# %%
# We can draw random samples from variable and evaluate the function using
nsamples = 1000
samples = variable.rvs(nsamples)
values = model(samples)

# %%
# Summary statistics of the samples and values can be printed using
from pyapprox.variables import print_statistics

print_statistics(samples, values)
