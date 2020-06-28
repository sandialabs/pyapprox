import numpy as np
from pyapprox.adaptive_sparse_grid import variance_refinement_indicator, \
    CombinationSparseGrid, \
    get_sparse_grid_univariate_leja_quadrature_rules_economical, \
    max_level_admissibility_function
from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from functools import partial
def approximate_sparse_grid(fun,univariate_variables,callback=None,refinement_indicator=variance_refinement_indicator,univariate_quad_rule_info=None,max_nsamples=100,tol=0,verbose=False):
    """
    Compute a sparse grid approximation of a function.

    Parameters
    ----------
    fun : callable
        The function to be minimized

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shaoe (nsamples,nqoi)

    univariate_variables : list
        A list of scipy.stats random variables of size (nvars)

    callback : callable
        Function called after each iteration with the signature
        
        ``callback(approx_k)``

        where approx_k is the current approximation object.

    refinement_indicator : callable
        A function that retuns an estimate of the error of a sparse grid subspace
        with signature
    
        ``refinement_indicator(subspace_index,nnew_subspace_samples,sparse_grid) -> float, float``

        where ``subspace_index`` is 1D np.ndarray of size (nvars), 
        ``nnew_subspace_samples`` is an integer specifying the number
        of new samples that will be added to the sparse grid by adding the 
        subspace specified by subspace_index and ``sparse_grid`` is the current 
        :class:`pyapprox.adaptive_sparse_grid.CombinationSparseGrid` object. 
        The two outputs are, respectively, the indicator used to control 
        refinement of the sparse grid and the change in error from adding the 
        current subspace. The indicator is typically but now always dependent on 
        the error.

    univariate_quad_rule_info : list
        List containing two entries. The first entry is a list 
        (or single callable) of univariate quadrature rules for each variable
        with signature

        ``quad_rule(l)->np.ndarray,np.ndarray``

        where the integer ``l`` specifies the level of the quadrature rule and 
        ``x`` and ``w`` are 1D np.ndarray of size (nsamples) containing the 
        quadrature rule points and weights, respectively.

        The second entry is a list (or single callable) of growth rules
        with signature

        ``growth_rule(l)->integer``

        where the output ``nsamples`` specifies the numeber of samples in the 
        quadrature rule of level``l``.

        If either entry is a callable then the same quad or growth rule is 
        applied to every variable.

    max_nsamples : integer
        The maximum number of evaluations of fun.

    tol : float
        Tolerance for termination. The construction of the sparse grid is 
        terminated when the estimate error in the sparse grid (determined by 
        ``refinement_indicator`` is below tol.

    verbose: boolean
        Controls messages printed during construction.

    Returns
    -------
    sparse_grid : :class:`pyapprox.adaptive_sparse_grid.CombinationSparseGrid`
        The sparse grid approximation
    """
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

from pyapprox.probability_measure_sampling import \
    generate_independent_random_samples
def compute_l2_error(f,g,variable,nsamples):
    """
    Compute the :math:`\ell^2` error of the output of two functions f and g, i.e.

    .. math:: \lVertf(z)-g(z)\rVert\approx \sum_{m=1}^M f(z^{(m)})

    from a set of random draws :math:`\mathcal{Z}=\{z^{(m)}\}_{m=1}^M` 
    from the PDF of :math:`z`.

    Parameters
    ----------
    f : callable
        Function with signature
    
        ``g(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shaoe (nsamples,nqoi)

    g : callable
        Function with signature
    
        ``f(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shaoe (nsamples,nqoi)

    variable : pya.IndependentMultivariateRandomVariable
        Object containing information of the joint density of the inputs z.
        This is used to generate random samples from this join density

    nsamples : integer
        The number of samples used to compute the :math:`\ell^2` error
    
    Returns
    -------
    error : np.ndarray (nqoi)
    """
    validation_samples = generate_independent_random_samples(variable,nsamples)
    validation_vals = f(validation_samples)
    approx_vals = g(validation_samples)
    error=np.linalg.norm(approx_vals-validation_vals,axis=0)/np.sqrt(
        validation_samples.shape[1])
    return error

def approximate(fun,variable,method,callback=None,options=None):
    """
    Approximation of a scalar or vector-valued function of one or more variables
    
    Parameters
    ----------
    fun : callable
        The function to be minimized

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shaoe (nsamples,nqoi)

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
