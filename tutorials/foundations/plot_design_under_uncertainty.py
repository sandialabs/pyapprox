"""
Design Under Uncertainty
========================

We will ue the Cantilever Beam benchmark to illustrate how to design under uncertainty.

.. figure:: ../../figures/cantilever-beam.png
   :align: center

   Conceptual model of the cantilever-beam
   
.. table:: Uncertainties
   :align: center
	    
   =============== ========= =======================
   Uncertainty     Symbol    Prior
   =============== ========= =======================
   Yield stress    :math:`R` :math:`N(40000,2000)`
   Young's modulus :math:`E` :math:`N(2.9e7,1.45e6)`
   Horizontal load :math:`X` :math:`N(500,100)`
   Vertical Load   :math:`Y` :math:`N(1000,100)`
   =============== ========= =======================

First we must specify the distribution of the random variables
"""
import pyapprox as pya
from pyapprox.benchmarks.benchmarks import setup_benchmark
benchmark = setup_benchmark('cantilever_beam')

from pyapprox.models.wrappers import ActiveSetVariableModel
nsamples = 10
samples = pya.generate_independent_random_samples(benchmark.variable,nsamples)
fun = ActiveSetVariableModel(
            benchmark.fun,benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
            samples,benchmark.design_var_indices)
jac = ActiveSetVariableModel(
    benchmark.jac,benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
    samples,benchmark.design_var_indices)

constraint_fun = ActiveSetVariableModel(
    benchmark.constraint_fun,
    benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
    samples,benchmark.design_var_indices)
constraint_jac = ActiveSetVariableModel(
    benchmark.constraint_jac,
    benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
    samples,benchmark.design_var_indices)
