r"""
Model Definition
----------------
This tutorial describes how to setup a function with random inputs. 

We start by defining a function of two random variables. We will use the Rosenbrock benchmark. See :func:`pyapprox.benchmarks.benchmarks.setup_rosenbrock_function`
"""
import numpy as np
from scipy import stats
from pyapprox.variables import IndependentMarginalsVariable


def fun(samples):
    return np.sum(samples*2, axis=0)[:, None]


#%%
#Now lets define the inputs to the function of interest. For independent random variables we use SciPy random variablest to represent each one-dimensional variables. For documentation refer to the `scipy.stats module <https://docs.scipy.org/doc/scipy/reference/stats.html>`_.
#
#We define multivariate random variables by specifying each 1D variable in a list. Here we will setup a 2D variable which is the tensor product of two independent and identically distributed uniform random variables

univariate_variables = [stats.uniform(-2, 4), stats.uniform(-2, 4)]
variable = IndependentMarginalsVariable(univariate_variables)

#%%
# To print a summary of the random variable use
print(variable)

#%%
#We can draw random samples from variable and evaluate the function using
nsamples = 1000
samples = variable.rvs(nsamples)
values = fun(samples)

#%%
#Summary statistics of the samples and values can be printed using
from pyapprox.variables import print_statistics
print_statistics(samples, values)

#%%
#User defined functions
#^^^^^^^^^^^^^^^^^^^^^^
#PyApprox can be used with pretty much any function provided an appropriate interface is defined. Here will show how to setup a simple function.
#
#PyApprox requires all functions to take 2D np.ndarray with shape (nvars,nsamples) and requires a function to return a 2D np.ndarray with shape (nsampels,nqoi). nqoi==1 for scalar valued functions and nqoi>1 for vectored value functions.
#
#Lets define a function which does not match this criteria and use wrappers provided by PyApprox to convert it to the correct format. Specifically we will define a function that only takes a 1D np.ndarray and returns a scalar. We import these functions from a separate file
#
#.. literalinclude:: ../../../examples/__util.py
#  :language: python
#  :start-at: def fun_0
#  :end-before: def fun_pause_1
#
#.. Note for some reason text like this is needed after the literalinclude
#.. Also note that path above is relative to source/auto_examples
#

from __util import pyapprox_fun_0, fun_0
values = pyapprox_fun_0(samples)

#%%
#The function :func:`pyapprox.interface.wrappers.evaluate_1darray_function_on_2d_array` avoids the need to write a for loop but we can do this also and does some checking to make sure values is the correct shape

values_loop = np.array([np.atleast_1d(fun_0(s)) for s in samples.T])
assert np.allclose(values, values_loop)
