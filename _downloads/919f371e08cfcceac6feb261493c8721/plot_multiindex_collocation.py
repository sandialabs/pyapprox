"""
Multi-index Collocation
-----------------------
The following provides an example of how to use multivariate quadrature, e.g. multilevel Monte Carlo, control variates to estimate the mean of a high-fidelity model from an ensemble of related models of varying cost and accuracy. Refer to :ref:`sphx_glr_auto_tutorials_multi_fidelity_plot_multindex_collocation.py` for a detailed tutorial on the theory behind multi-index collocation.
"""
#%%
#Load the necessary modules
import numpy as np
from pyapprox.benchmarks.benchmarks import setup_benchmark
from pyapprox import multifidelity
from pyapprox import interface
# set seed for reproducibility
np.random.seed(1)

#%%
#Set up a :py:class:`~pyapprox.interface.wrappers.MultiIndexModel` that takes samples :math:`x=[z_1,\ldots,z_D,v_1,\ldots, v_C]` which is the concatenation of a random sample `z` and configuration values specifying the discretization parameters of the numerical model.
