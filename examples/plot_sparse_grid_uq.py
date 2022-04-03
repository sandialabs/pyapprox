"""
Sparse Grid Interpolation
=========================

We will use the Cantilever Beam benchmark to illustrate how to use a sparse grid
as a surrogate for uncertainty quantification and sensitivity analysis

.. figure:: ../../source/figures/cantilever-beam.png
   :align: center

   Conceptual model of the cantilever-beam

.. table:: Uncertain variables
   :align: center

   =============== ========= =======================
   Uncertainty     Symbol    Prior
   =============== ========= =======================
   Yield stress    :math:`R` :math:`N(40000,2000)`
   Young's modulus :math:`E` :math:`N(2.9e7,1.45e6)`
   Horizontal load :math:`X` :math:`N(500,100)`
   Vertical Load   :math:`Y` :math:`N(1000,100)`
   =============== ========= =======================


.. table:: Design variables
   :align: center

   =============== ========= =======================
   Uncertainty     Symbol    Range
   =============== ========= =======================
   Width           :math:`w` :math:`[1, 4]`
   Thickness       :math:`t` :math:`[1, 4]`
   =============== ========= =======================

First we must load the benchmark
"""
import numpy as np
import pyapprox as pya
from pyapprox.benchmarks.benchmarks import setup_benchmark
import matplotlib.pyplot as plt
benchmark = setup_benchmark('cantilever_beam')
np.random.seed(1)

#%%
#The cantilever beam model is an optimization benchmark. Consequently it
#defines random and design variables. All surrogates require random variable
#information so we define a new variable for this purpose

variable = pya.combine_uncertain_and_bounded_design_variables(
    benchmark.variable, benchmark.design_variable)

#%% The objective fun of the benchmark is not a function of random variables
#So lets just approximate the contraint function.
#
#We can set a maximum number of samples using the options dictionary

options = {"max_nsamples": 1000}
approx = pya.adaptive_approximate(
    benchmark.constraint_fun, variable, "sparse_grid", options).approx

#%%
# We can estimate the error in the surrogate using some validation samples.
validation_samples = variable.rvs(100)
validation_values = benchmark.constraint_fun(validation_samples)
approx_values = approx(validation_samples)
error = np.linalg.norm(validation_values-approx_values, axis=0)/np.sqrt(
    validation_values.shape[0])
print(f"The RMSE error is {error}")

#%% We can estimate the PDF of the two function outputs by sampling the surrogate
#and building a kernel density estimator. Lets first just plot the marginal
#PDFs of the output
surrogate_samples = pya.generate_independent_random_samples(variable, 10000)
approx_values = approx(surrogate_samples)
pya.plot_qoi_marginals(approx_values)
plt.show()
