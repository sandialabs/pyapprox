#!/usr/bin/env python
import numpy as np
from pyapprox.configure_plots import *
from pyapprox.adaptive_polynomial_chaos import *
from pyapprox.variable_transformations import \
    AffineBoundedVariableTransformation, AffineRandomVariableTransformation
from pyapprox.variables import IndependentMultivariateRandomVariable
from scipy.stats import beta
from pyapprox.probability_measure_sampling import \
    generate_independent_random_samples
from pyapprox.adaptive_sparse_grid import max_level_admissibility_function, \
    isotropic_refinement_indicator
from pyapprox.univariate_quadrature import clenshaw_curtis_rule_growth, \
    constant_increment_growth_rule
from functools import partial
from scipy.stats import uniform,beta
from pyapprox.models.genz import GenzFunction

def compute_l2_error(validation_samples,validation_values,pce,relative=True):
    pce_values = pce(validation_samples)
    error = np.linalg.norm(pce_values-validation_values,axis=0)
    if not relative:
        error /=np.sqrt(validation_samples.shape[1])
    else:
        error /= np.linalg.norm(validation_values,axis=0)
    
    return error

def genz_example(max_num_samples):
    error_tol=1e-12

    univariate_variables = [
        uniform(),beta(3,3)]
    variable = IndependentMultivariateRandomVariable(univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)

    c = np.array([10,0.00])
    model = GenzFunction(
        "oscillatory",var_trans.num_vars(),c=c,w=np.zeros_like(c))
    # model.set_coefficients(4,'exponential-decay')

    validation_samples = generate_independent_random_samples(
    var_trans.variable,int(1e3))
    validation_values = model(validation_samples)
    
    errors = []
    num_samples = []
    def callback(pce):
        error = compute_l2_error(validation_samples,validation_values,pce)
        errors.append(error)
        num_samples.append(pce.samples.shape[1])

    candidate_samples=-np.cos(
        np.random.uniform(0,np.pi,(var_trans.num_vars(),int(1e4))))
    pce = AdaptiveLejaPCE(
        var_trans.num_vars(),candidate_samples,factorization_type='fast')

    max_level=np.inf
    max_level_1d=[max_level]*(pce.num_vars)

    admissibility_function = partial(
        max_level_admissibility_function,max_level,max_level_1d,
        max_num_samples,error_tol)

    growth_rule =  partial(constant_increment_growth_rule,2)
    #growth_rule = clenshaw_curtis_rule_growth
    pce.set_function(model,var_trans)
    pce.set_refinement_functions(
        variance_pce_refinement_indicator,admissibility_function,
        growth_rule)
    
    while (not pce.active_subspace_queue.empty() or
           pce.subspace_indices.shape[1]==0):
        pce.refine()
        pce.recompute_active_subspace_priorities()
        if callback is not None:
            callback(pce)


    from pyapprox.sparse_grid import plot_sparse_grid_2d
    plot_sparse_grid_2d(
        pce.samples,np.ones(pce.samples.shape[1]),
        pce.pce.indices, pce.subspace_indices)

    plt.figure()
    plt.loglog(num_samples,errors,'o-')
    plt.show()
    
 
if __name__ == '__main__':
    genz_example(100)
