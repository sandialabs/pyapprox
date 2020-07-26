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
import numpy as np
import pyapprox as pya
from pyapprox.benchmarks.benchmarks import setup_benchmark
from functools import partial
from pyapprox.optimization import *
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

generate_random_samples = partial(
    pya.generate_independent_random_samples,benchmark.variable,100)
generate_sample_data = partial(
    generate_monte_carlo_quadrature_data,generate_random_samples,
    benchmark.variable.num_vars(),benchmark.design_var_indices)

num_vars = benchmark.variable.num_vars()+benchmark.design_variable.num_vars()
objective = StatisticalConstraint(
    benchmark.fun,benchmark.jac,expectation_fun,expectation_jac,num_vars,
    benchmark.design_var_indices,generate_sample_data)

init_guess = 2*np.ones((2,1))
errors = pya.check_gradients(
    objective,objective.jacobian,init_guess,disp=True)
constraint = StatisticalConstraint(
    benchmark.constraint_fun,benchmark.constraint_jac,expectation_fun,expectation_jac,
    num_vars,benchmark.design_var_indices,generate_sample_data)

init_guess = 2*np.ones((2,1))
print(constraint(init_guess))
print(constraint.jacobian(init_guess).shape)
errors = pya.check_gradients(
    constraint,constraint.jacobian,init_guess,disp=True)

#Combinations of constraint specifications
#fun,jac = fun,None
#fun,jac = fun,True
#fun,jac = fun,jac
