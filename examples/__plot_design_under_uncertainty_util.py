"""
Design Under Uncertainty
========================

We will ue the Cantilever Beam benchmark to illustrate how to design under
uncertainty.

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
np.random.seed(1)

from pyapprox.interface.wrappers import ActiveSetVariableModel
nsamples = 10
samples = pya.generate_independent_random_samples(benchmark.variable,nsamples)
fun = ActiveSetVariableModel(
            benchmark.fun,benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
            samples,benchmark.design_var_indices)
jac = ActiveSetVariableModel(
    benchmark.jac,benchmark.variable.num_vars()+benchmark.design_variable.num_vars(),
    samples,benchmark.design_var_indices)

nsamples = 10000
generate_random_samples = partial(
    pya.generate_independent_random_samples,benchmark.variable,nsamples)
#set seed so that finite difference jacobian always uses the same set of samples for each
#step size and as used for computing the exact gradient
seed=1
generate_sample_data = partial(
    generate_monte_carlo_quadrature_data,generate_random_samples,
    benchmark.variable.num_vars(),benchmark.design_var_indices,seed=seed)

num_vars = benchmark.variable.num_vars()+benchmark.design_variable.num_vars()
objective = StatisticalConstraint(
    benchmark.fun,benchmark.jac,expectation_fun,expectation_jac,num_vars,
    benchmark.design_var_indices,generate_sample_data,isobjective=True)

init_guess = 2*np.ones((2,1))
errors = pya.check_gradients(
    objective,objective.jacobian,init_guess,disp=False)
print(errors.min())
assert errors.min()<6e-6

from pyapprox.optimization.cvar_regression import \
    smooth_conditional_value_at_risk_composition
smoother_type,eps,alpha=0,1e-3,0.75
#stat_fun = partial(smooth_conditional_value_at_risk,smoother_type,eps,alpha)
#stat_jac = True
tol=0.0
stat_fun = partial(smooth_prob_failure_fun,smoother_type,eps,tol)
stat_jac = partial(smooth_prob_failure_jac,smoother_type,eps,tol)
upper_bound=True
bound=0.05
#stat_fun,stat_jac,upper_bound = expectation_fun,expectation_jac,True
#bound=0

constraint = StatisticalConstraint(
    benchmark.constraint_fun,benchmark.constraint_jac,stat_fun,stat_jac,
    num_vars,benchmark.design_var_indices,generate_sample_data,bound=bound,
    upper_bound=upper_bound)

print('####')
#init_guess = 3*np.ones((2,1))
init_guess = np.array([[2.3,3.4]]).T
print(constraint(init_guess))
print(approx_jacobian(constraint,init_guess,epsilon=1e-5))
print(constraint.jacobian(init_guess))
#assert False
#print(constraint(init_guess))
errors = pya.check_gradients(
    constraint,constraint.jacobian,init_guess,disp=True)
#assert errors.min()<1e-7

from scipy.optimize import minimize, NonlinearConstraint
def run_design(objective,jac,constraints,constraints_jac,bounds,x0,options):
    options=options.copy()
    if constraints_jac is None:
        constraints_jac  = [None]*len(constraints)
    scipy_constraints = []
    for constraint, constraint_jac in zip(constraints,constraints_jac):
        scipy_constraints.append(NonlinearConstraint(
            constraint,0,np.inf,jac=constraint_jac))
    method = options.get('method','slsqp')
    if 'method' in options: del options['method']
    callback=options.get('callback',None)
    if 'callback' in options:
        del options['callback']
    print(x0[:,0])
    res = minimize(
        objective, x0[:,0], method=method, jac=jac, hess=None,
        constraints=scipy_constraints,options=options,callback=callback,
        bounds=bounds)
    return res.x, res

print('$$$$')
jac=None
jac=objective.jacobian
cons_jac='2-point'
#cons_jac=constraint.jacobian
#options={'callback':lambda xk : print(xk),'disp':True,'iprint':5,'method':'slsqp'}
options={'disp':True,'verbose':2,'method':'trust-constr','gtol':1e-6}
x,res=run_design(objective,jac,[constraint],[cons_jac],benchmark.design_variable.bounds,init_guess,options)
print(res)
print(constraint(res.x)+bound)
for ii in range(constraint.fun_values.shape[1]):
    print('b',np.where(constraint.fun_values[:,ii:ii+1]>0)[0].shape[0]/nsamples)
    
#robust design
#min f subject to variance<tol

#reliability design
#min f subject to prob failure<tol
