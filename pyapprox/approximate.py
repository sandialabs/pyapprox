import numpy as np
from pyapprox.adaptive_sparse_grid import variance_refinement_indicator, \
    CombinationSparseGrid, \
    get_sparse_grid_univariate_leja_quadrature_rules_economical, \
    max_level_admissibility_function
from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from functools import partial
def approximate_sparse_grid(fun,univariate_variables,callback=None,refinement_indicator=variance_refinement_indicator,univariate_quad_rule_info=None,max_num_samples=100,tol=0,verbose=False):
    variable = IndependentMultivariateRandomVariable(
        univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    nvars = var_trans.num_vars()
    sparse_grid = CombinationSparseGrid(nvars)
    if univariate_quad_rule_info is None:
        quad_rules, growth_rules, unique_quadrule_indices = \
            get_sparse_grid_univariate_leja_quadrature_rules_economical(
                var_trans)
    else:
        quad_rules,growth_rules=univariate_quad_rule_info
        unique_quadrule_indices=None
    admissibility_function = partial(
        max_level_admissibility_function,np.inf,[np.inf]*nvars,max_num_samples,
        tol,verbose=verbose)
    sparse_grid.setup(fun, None, variance_refinement_indicator,
                      admissibility_function, growth_rules, quad_rules,
                      var_trans, unique_quadrule_indices=unique_quadrule_indices)
    sparse_grid.build(callback)
    return sparse_grid

def approximate(fun,variable,method,callback=None,options=None):
    """
    Approximation of a scalar or vector-valued function of one or more variables
    
    Parameters
    ----------
    fun : callable
        The function to be minimized

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples)

    method : string
        Type of approximation. Should be one of

        - 'sparse-grid'
        - 'polynomial-chaos'
        - 'tensor-train'
        - 'gaussian-process'

    callback : callable
        Function called after each iteration with the signature
        
        ``callback(approx_k)``

        where approx_k is the current approximation object.
        
    Returns
    -------
    approx : Object
       An object which approximates fun.
    """

    methods = {'sparse-grid':approximate_sparse_grid,
               'polynomial-chaos':approximate_polynomial_chaos,
               'tensor-train':approximate_tensor_train,
               'gaussian-process':approximate_gaussian_process}

    if method not in methods:
        msg = f'Method {method} not found.\n Available methods are:\n'
        for key in methods.keys():
            msg += f"\t{key}\n"
        raise Exception(msg)

    return methods['method'](fun,variable,callback,**options)
