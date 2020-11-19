import numpy as np
from pyapprox.adaptive_sparse_grid import variance_refinement_indicator, \
    CombinationSparseGrid, \
    get_sparse_grid_univariate_leja_quadrature_rules_economical, \
    max_level_admissibility_function
from pyapprox.variables import IndependentMultivariateRandomVariable
from pyapprox.variable_transformations import AffineRandomVariableTransformation
from functools import partial

from scipy.optimize import OptimizeResult
class ApproximateResult(OptimizeResult):
    pass

def adaptive_approximate_sparse_grid(
        fun, univariate_variables, callback=None,
        refinement_indicator=variance_refinement_indicator,
        univariate_quad_rule_info=None, max_nsamples=100, tol=0, verbose=0,
        config_variables_idx=None, config_var_trans=None, cost_function=None,
        max_level_1d=None):
    """
    Compute a sparse grid approximation of a function.

    Parameters
    ----------
    fun : callable
        The function to be minimized

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,nqoi)

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

        where the output ``nsamples`` specifies the number of samples in the 
        quadrature rule of level ``l``.

        If either entry is a callable then the same quad or growth rule is 
        applied to every variable.

    max_nsamples : float
        If ``cost_function==None`` then this argument is the maximum number of 
        evaluations of fun. If fun has configure variables

        If ``cost_function!=None`` Then max_nsamples is the maximum cost of 
        constructing the sparse grid, i.e. the sum of the cost of evaluating
        each point in the sparse grid.

        The ``cost_function!=None` same behavior as ``cost_function==None``
        can be obtained by setting cost_function = lambda config_sample: 1.

        This is particularly useful if ``fun`` has configure variables
        and evaluating ``fun`` at these different values of these configure
        variables has different cost. For example if there is one configure
        variable that can take two different values with cost 0.5, and 1
        then 10 samples of both models will be measured as 15 samples and
        so if max_nsamples is 19 the algorithm will not terminate because
        even though the number of samples is the sparse grid is 20.

    tol : float
        Tolerance for termination. The construction of the sparse grid is 
        terminated when the estimate error in the sparse grid (determined by 
        ``refinement_indicator`` is below tol.

    verbose : integer
        Controls messages printed during construction.

    config_variable_idx : integer
        The position in a sample array that the configure variables start

    config_var_trans : pyapprox.adaptive_sparse_grid.ConfigureVariableTransformation
        An object that takes configure indices in [0,1,2,3...] 
        and maps them to the configure values accepted by ``fun``

    cost_function : callable 
        A function with signature

        ``cost_function(config_sample) -> float``
    
        where config_sample is a np.ndarray of shape (nconfig_vars). The output
        is the cost of evaluating ``fun`` at ``config_sample``. The units can be
        anything but the units must be consistent with the units of max_nsamples
        which specifies the maximum cost of constructing the sparse grid.

    max_level_1d : np.ndarray (nvars)
        The maximum level of the sparse grid in each dimension. If None
        There is no limit

    Returns
    -------
    result : :class:`pyapprox.approximate.ApproximateResult`
         Result object with the following attributes

    approx : :class:`pyapprox.adaptive_sparse_grid.CombinationSparseGrid`
        The sparse grid approximation
    """
    variable = IndependentMultivariateRandomVariable(
        univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    nvars = var_trans.num_vars()
    if config_var_trans is not None:
        nvars += config_var_trans.num_vars()
    sparse_grid = CombinationSparseGrid(nvars)
    if univariate_quad_rule_info is None:
        quad_rules, growth_rules, unique_quadrule_indices = \
            get_sparse_grid_univariate_leja_quadrature_rules_economical(
            var_trans)
    else:
        quad_rules,growth_rules = univariate_quad_rule_info
        unique_quadrule_indices = None
    if max_level_1d is None:
        max_level_1d = [np.inf]*nvars
    assert len(max_level_1d) == nvars
    admissibility_function = partial(
        max_level_admissibility_function, np.inf, max_level_1d, max_nsamples,
        tol, verbose=verbose)
    sparse_grid.setup(
        fun, config_variables_idx, refinement_indicator,
        admissibility_function, growth_rules, quad_rules,
        var_trans,unique_quadrule_indices=unique_quadrule_indices,
        verbose=verbose, cost_function=cost_function,
        config_var_trans=config_var_trans)
    sparse_grid.build(callback)
    return ApproximateResult({'approx':sparse_grid})

from pyapprox.adaptive_polynomial_chaos import AdaptiveLejaPCE,\
    variance_pce_refinement_indicator
from pyapprox.variables import is_bounded_continuous_variable
from pyapprox.univariate_quadrature import clenshaw_curtis_rule_growth
def adaptive_approximate_polynomial_chaos(
        fun, univariate_variables, callback=None,
        refinement_indicator=variance_pce_refinement_indicator,
        growth_rules=None, max_nsamples=100, tol=0, verbose=0,
        ncandidate_samples=1e4, generate_candidate_samples=None):
    r"""
    Compute an adaptive Polynomial Chaos Expansion of a function.

    Parameters
    ----------
    fun : callable
        The function to be minimized

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,nqoi)

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

    growth_rules : list or callable
        a list (or single callable) of growth rules with signature

        ``growth_rule(l)->integer``

        where the output ``nsamples`` specifies the number of indices of the 
        univariate basis of level ``l``.

        If the entry is a callable then the same growth rule is 
        applied to every variable.

    max_nsamples : integer
        The maximum number of evaluations of fun.

    tol : float
        Tolerance for termination. The construction of the sparse grid is 
        terminated when the estimate error in the sparse grid (determined by 
        ``refinement_indicator`` is below tol.

    verbose : integer
        Controls messages printed during construction.

    ncandidate_samples : integer
        The number of candidate samples used to generate the Leja sequence
        The Leja sequence will be a subset of these samples.

    generate_candidate_samples : callable
        A function that generates the candidate samples used to build the Leja
        sequence with signature

        ``generate_candidate_samples(ncandidate_samples) -> np.ndarray``
    
        The output is a 2D np.ndarray with size(nvars,ncandidate_samples)

    Returns
    -------
    result : :class:`pyapprox.approximate.ApproximateResult`
         Result object with the following attributes

    approx : :class:`pyapprox.multivariate_polynomials.PolynomialChaosExpansion`
        The PCE approximation
    """
    variable = IndependentMultivariateRandomVariable(
        univariate_variables)
    var_trans = AffineRandomVariableTransformation(variable)
    nvars = var_trans.num_vars()

    bounded_variables = True
    for rv in univariate_variables:
        if not is_bounded_continuous_variable(rv):
            bounded_variables = False
            msg = "For now leja sampling based PCE is only supported for bounded continouous random variablesfor now leja sampling based PCE is only supported for bounded continouous random variables"
            if generate_candidate_samples is None:
                raise Exception (msg)
            else:
                break
    if generate_candidate_samples is None:
        # Todo implement default for non-bounded variables that uses induced
        # sampling
        # candidate samples must be in canonical domain
        from pyapprox import halton_sequence
        candidate_samples = -np.cos(
            np.pi*halton_sequence(nvars,1,int(ncandidate_samples+1)))
        #candidate_samples = -np.cos(
        #    np.random.uniform(0,np.pi,(nvars,int(ncandidate_samples))))
    else:
        candidate_samples = generate_candidate_samples(ncandidate_samples)
        
    pce = AdaptiveLejaPCE(
        nvars,candidate_samples,factorization_type='fast')
    pce.verbose=verbose
    admissibility_function = partial(
        max_level_admissibility_function,np.inf,[np.inf]*nvars,max_nsamples,
        tol,verbose=verbose)
    pce.set_function(fun,var_trans)
    if growth_rules is None:
        growth_rules = clenshaw_curtis_rule_growth
    pce.set_refinement_functions(
        refinement_indicator,admissibility_function,growth_rules)
    pce.build(callback)
    return ApproximateResult({'approx':pce})

from pyapprox.probability_measure_sampling import \
    generate_independent_random_samples
def compute_l2_error(f, g, variable, nsamples, rel=False):
    r"""
    Compute the :math:`\ell^2` error of the output of two functions f and g, i.e.

    .. math:: \lVert f(z)-g(z)\rVert\approx \sum_{m=1}^M f(z^{(m)})

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

    rel : boolean
        True - compute relative error
        False - compute absolute error
    
    Returns
    -------
    error : np.ndarray (nqoi)
    """
    
    validation_samples = generate_independent_random_samples(variable,nsamples)
    validation_vals = f(validation_samples)
    approx_vals = g(validation_samples)
    assert validation_vals.shape==approx_vals.shape
    error=np.linalg.norm(approx_vals-validation_vals,axis=0)
    if not rel:
        error /=np.sqrt(validation_samples.shape[1])
    else:
        error /=np.linalg.norm(validation_vals,axis=0)
    return error

def adaptive_approximate(fun, variable, method, options=None):
    r"""
    Adaptive approximation of a scalar or vector-valued function of one or 
    more variables. These methods choose the samples to at which to 
    evaluate the function being approximated.
    
    
    Parameters
    ----------
    fun : callable
        The function to be minimized

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shaoe (nsamples,nqoi)

    method : string
        Type of approximation. Should be one of

        - 'sparse_grid'
        - 'polynomial_chaos'
        - 'gaussian_process'
        
    Returns
    -------
    result : :class:`pyapprox.approximate.ApproximateResult`
         Result object. For more details see 
    
         - :func:`pyapprox.approximate.adaptive_approximate_sparse_grid` 

         - :func:`pyapprox.approximate.adaptive_approximate_polynomial_chaos`
    """

    methods = {'sparse_grid':adaptive_approximate_sparse_grid,
               'polynomial_chaos':adaptive_approximate_polynomial_chaos}

    if method not in methods:
        msg = f'Method {method} not found.\n Available methods are:\n'
        for key in methods.keys():
            msg += f"\t{key}\n"
        raise Exception(msg)

    if options is None:
        options = {}
    return methods[method](fun, variable, **options)

def approximate_polynomial_chaos(train_samples, train_vals, verbosity=0,
                                 basis_type='expanding_basis',
                                 variable=None, options=None):
    r"""
    Compute a Polynomial Chaos Expansion of a function from a fixed data set.

    Parameters
    ----------
    train_samples : np.ndarray (nvars,nsamples)
        The inputs of the function used to train the approximation

    train_vals : np.ndarray (nvars,nsamples)
        The values of the function at ``train_samples``

    basis_type : string
        Type of approximation. Should be one of

        - 'expanding_basis' see :func:`pyapprox.approximate.cross_validate_pce_degree` 
        - 'hyperbolic_cross' see :func:`pyapprox.approximate.expanding_basis_omp_pce`

    variable : pya.IndependentMultivariateRandomVariable
        Object containing information of the joint density of the inputs z.
        This is used to generate random samples from this join density

    verbosity : integer
        Controls the amount of information printed to screen

    Returns
    -------
    result : :class:`pyapprox.approximate.ApproximateResult`
         Result object. For more details see 
    
         - :func:`pyapprox.approximate.cross_validate_pce_degree` 

         - :func:`pyapprox.approximate.expanding_basis_omp_pce`
    """
    funcs = {'expanding_basis':expanding_basis_omp_pce,
             'hyperbolic_cross':cross_validate_pce_degree}
    if variable is None:
        msg = 'pce requires that variable be defined'
        raise Exception(msg)
    if basis_type not in funcs:
        msg = f'Basis type {basis_type} not found.\n Available types are:\n'
        for key in funcs.keys():
            msg += f"\t{key}\n"
        raise Exception(msg)
    
    from pyapprox.multivariate_polynomials import PolynomialChaosExpansion, \
        define_poly_options_from_variable_transformation
    var_trans = AffineRandomVariableTransformation(variable)
    poly = PolynomialChaosExpansion()
    poly_opts = define_poly_options_from_variable_transformation(
        var_trans)
    poly.configure(poly_opts)
    
    if options is None:
        options = {}

    res = funcs[basis_type](poly, train_samples, train_vals, **options)
    return res

def approximate(train_samples, train_vals, method,options=None):
    r"""
    Approximate a scalar or vector-valued function of one or 
    more variables from a set of points provided by the user
    
    Parameters
    ----------
    train_samples : np.ndarray (nvars,nsamples)
        The inputs of the function used to train the approximation

    train_vals : np.ndarray (nvars,nsamples)
        The values of the function at ``train_samples``

    method : string
        Type of approximation. Should be one of

        - 'polynomial_chaos'
        - 'gaussian_process'

    Returns
    -------
    result : :class:`pyapprox.approximate.ApproximateResult`
    """

    methods = {'polynomial_chaos':approximate_polynomial_chaos,
               'gaussian_process':approximate_gaussian_process}
               #'tensor-train':approximate_tensor_train,

    if method not in methods:
        msg = f'Method {method} not found.\n Available methods are:\n'
        for key in methods.keys():
            msg += f"\t{key}\n"
        raise Exception(msg)

    if options is None:
        options = {}
    return methods[method](train_samples, train_vals, **options)


from sklearn.linear_model import LassoCV, LassoLarsCV, LarsCV, \
    OrthogonalMatchingPursuitCV

def fit_linear_model(basis_matrix, train_vals, solver_type, **kwargs):
    solvers = {'lasso_lars':LassoLarsCV(cv=kwargs['cv']).fit,
               'lasso':LassoCV(cv=kwargs['cv']).fit,
               'lars':LarsCV(cv=kwargs['cv']).fit,
               'omp':OrthogonalMatchingPursuitCV(
                   cv=kwargs['cv'], verbose=5).fit}
    assert train_vals.ndim == 2
    if solver_type in solvers:
        fit = solvers[solver_type]
        res = fit(basis_matrix, train_vals[:, 0])
    else:
        msg = f'Solver type {solver_type} not supported\n'
        msg += 'Supported solvers are:\n'
        for key in solvers.keys():
            msg += f'\t{key}\n'
        raise Exception(msg)

    cv_score = res.score(basis_matrix, train_vals[:,0])
    coef = res.coef_[:, np.newaxis]; coef[0]=res.intercept_
    return coef, cv_score

import copy
from pyapprox import compute_hyperbolic_indices
def cross_validate_pce_degree(
        pce, train_samples, train_vals, min_degree=1, max_degree=3,
        hcross_strength=1, cv=10, solver_type='lars', verbosity=0):
    r"""
    Use cross validation to find the polynomial degree which best fits the data.
    A polynomial is constructed for each degree and the degree with the highest
    cross validation score is returned.
    
    Parameters
    ----------
    train_samples : np.ndarray (nvars,nsamples)
        The inputs of the function used to train the approximation

    train_vals : np.ndarray (nvars,nsamples)
        The values of the function at ``train_samples``
    
    min_degree : integer
        The minimum degree to consider

    min_degree : integer
        The maximum degree to consider. 
        All degrees in ``range(min_degree,max_deree+1)`` are considered

    hcross_strength : float
       The strength of the hyperbolic cross index set. hcross_strength must be 
       in (0,1]. A value of 1 produces total degree polynomials

    cv : integer
        The number of cross validation folds used to compute the cross 
        validation error

    solver_type : string
        The type of regression used to train the polynomial

        - 'lasso_lars'
        - 'lars'
        - 'lasso'
        - 'omp'

    verbosity : integer
        Controls the amount of information printed to screen

    Returns
    -------
    result : :class:`pyapprox.approximate.ApproximateResult`
         Result object with the following attributes

    approx : :class:`pyapprox.multivariate_polynomials.PolynomialChaosExpansion`
        The PCE approximation

    scores : np.ndarray (nqoi)
        The best cross validation score for each QoI

    degrees : np.ndarray (nqoi)
        The best degree for each QoI
    """
    coefs = []
    scores = []
    indices = []
    degrees = []
    indices_dict=dict()
    unique_indices=[]
    nqoi = train_vals.shape[1]
    for ii in range(nqoi):
        if verbosity > 1:
            print(f'Approximating QoI: {ii}')
        pce_ii,score_ii,degree_ii = _cross_validate_pce_degree(
            pce,train_samples, train_vals[:, ii:ii+1], min_degree, max_degree,
            hcross_strength, cv, solver_type,verbosity)
        coefs.append(pce_ii.get_coefficients())
        scores.append(score_ii)
        indices.append(pce_ii.get_indices())
        degrees.append(degree_ii)
        for index in indices[ii].T:
            key = hash_array(index)
            if key not in indices_dict:
                indices_dict[key] = len(unique_indices)
                unique_indices.append(index)

    unique_indices = np.array(unique_indices).T
    all_coefs = np.zeros((unique_indices.shape[1], nqoi))
    for ii in range(nqoi):
        for jj, index in enumerate(indices[ii].T):
            key = hash_array(index)
            all_coefs[indices_dict[key], ii] = coefs[ii][jj, 0]
    pce.set_indices(unique_indices)
    pce.set_coefficients(all_coefs)
    return ApproximateResult({'approx':pce, 'scores':np.array(scores),
                              'degrees':np.array(degrees)})
    
def _cross_validate_pce_degree(
        pce,train_samples, train_vals, min_degree=1, max_degree=3,
        hcross_strength=1,
        cv=10, solver_type='lasso_lars', verbosity=0):
    assert train_vals.shape[1] == 1
    num_samples = train_samples.shape[1]
    if min_degree is None:
        min_degree = 2
    if max_degree is None:
        max_degree = np.iinfo(int).max-1

    best_coef = None
    best_cv_score = -np.finfo(np.double).max
    best_degree = min_degree
    prev_num_terms = 0
    if verbosity>0:
        print ("{:<8} {:<10} {:<18}".format('degree','num_terms','cv score',))
    for degree in range(min_degree, max_degree+1):
        indices = compute_hyperbolic_indices(
            pce.num_vars(), degree, hcross_strength)
        pce.set_indices(indices)
        if ((pce.num_terms() > 100000) and
            (100000-prev_num_terms < pce.num_terms()-100000) ): break

        basis_matrix = pce.basis_matrix(train_samples)
        coef, cv_score = fit_linear_model(
            basis_matrix, train_vals, solver_type, cv=cv)
        pce.set_coefficients(coef)

        if verbosity > 0:
            print("{:<8} {:<10} {:<18} ".format(
                degree ,pce.num_terms(), cv_score))
        if ( cv_score > best_cv_score ):
            best_cv_score = cv_score
            best_coef = coef.copy()
            best_degree = degree
        if ( ( cv_score >= best_cv_score ) and ( degree-best_degree > 1 ) ):
            break
        prev_num_terms = pce.num_terms()

    pce.set_indices(compute_hyperbolic_indices(
        pce.num_vars(), best_degree,hcross_strength))
    pce.set_coefficients(best_coef)
    if verbosity>0:
        print ('best degree:', best_degree)
    return pce, best_cv_score, best_degree

def restrict_basis(indices,coefficients,tol):
    I = np.where(np.absolute(coefficients)>tol)[0]
    restricted_indices = indices[:,I]
    degrees = indices.sum(axis=0)
    J = np.where(degrees==0)[0]
    assert J.shape[0]==1
    if J not in I:
        #always include zero degree polynomial in restricted_indices
        restricted_indices = np.concatenate([indices[:J],restrict_indices])
    return restricted_indices

from pyapprox import hash_array, get_forward_neighbor, get_backward_neighbor
def expand_basis(indices):
    nvars,nindices=indices.shape
    indices_set = set()
    for ii in range(nindices):
        indices_set.add(hash_array(indices[:,ii]))
    
    new_indices = []
    for ii in range(nindices):
        index = indices[:,ii]
        active_vars = np.nonzero(index)
        for dd in range(nvars):
            forward_index = get_forward_neighbor(index,dd)
            key = hash_array(forward_index)
            if key not in indices_set:
                admissible=True
                for kk in active_vars:
                    backward_index = get_backward_neighbor(forward_index,kk)
                    if hash_array(backward_index) not in indices_set:
                        admissible=False
                        break
                if admissible:
                    indices_set.add(key)
                    new_indices.append(forward_index)
    return np.asarray(new_indices).T
    

def expanding_basis_omp_pce(pce, train_samples, train_vals, hcross_strength=1,
                            verbosity=1,max_num_terms=None,
                            solver_type='lasso_lars',cv=10,
                            restriction_tol=np.finfo(float).eps*2):
    r"""
    Iteratively expand and restrict the polynomial basis and use 
    cross validation to find the best basis [JESJCP2015]_
    
    Parameters
    ----------
    train_samples : np.ndarray (nvars,nsamples)
        The inputs of the function used to train the approximation

    train_vals : np.ndarray (nvars,nqoi)
        The values of the function at ``train_samples``
    
    hcross_strength : float
       The strength of the hyperbolic cross index set. hcross_strength must be 
       in (0,1]. A value of 1 produces total degree polynomials

    cv : integer
        The number of cross validation folds used to compute the cross 
        validation error

    solver_type : string
        The type of regression used to train the polynomial

        - 'lasso_lars'
        - 'lars'
        - 'lasso'
        - 'omp'

    verbosity : integer
        Controls the amount of information printed to screen

    restriction_tol : float
        The tolerance used to prune inactive indices

    Returns
    -------
    result : :class:`pyapprox.approximate.ApproximateResult`
         Result object with the following attributes

    approx : :class:`pyapprox.multivariate_polynomials.PolynomialChaosExpansion`
        The PCE approximation

    scores : np.ndarray (nqoi)
        The best cross validation score for each QoI

    References
    ----------
    .. [JESJCP2015] `J.D. Jakeman, M.S. Eldred, and K. Sargsyan. Enhancing l1-minimization estimates of polynomial chaos expansions using basis selection. Journal of Computational Physics, 289(0):18 – 34, 2015 <https://doi.org/10.1016/j.jcp.2015.02.025>`_
    """
    coefs = []
    scores = []
    indices = []
    indices_dict=dict()
    unique_indices=[]
    nqoi = train_vals.shape[1]
    for ii in range(nqoi):
        if verbosity>1:
            print(f'Approximating QoI: {ii}')
        pce_ii,score_ii = _expanding_basis_omp_pce(
            pce, train_samples, train_vals[:,ii:ii+1], hcross_strength,
            verbosity,max_num_terms,solver_type,cv,restriction_tol)
        coefs.append(pce_ii.get_coefficients())
        scores.append(score_ii)
        indices.append(pce_ii.get_indices())
        for index in indices[ii].T:
            key = hash_array(index)
            if key not in indices_dict:
                indices_dict[key]=len(unique_indices)
                unique_indices.append(index)

    unique_indices = np.array(unique_indices).T
    all_coefs = np.zeros((unique_indices.shape[1],nqoi))
    for ii in range(nqoi):
        for jj,index in enumerate(indices[ii].T):
            key = hash_array(index)
            all_coefs[indices_dict[key],ii]=coefs[ii][jj,0]
    pce.set_indices(unique_indices)
    pce.set_coefficients(all_coefs)
    return ApproximateResult({'approx':pce,'scores':np.array(scores)})

def _expanding_basis_omp_pce(pce, train_samples, train_vals, hcross_strength=1,
                             verbosity=1,max_num_terms=None,
                             solver_type='lasso_lars',cv=10,
                             restriction_tol=np.finfo(float).eps*2):
    assert train_vals.shape[1]==1
    num_vars = pce.num_vars()
    if max_num_terms  is None:
        max_num_terms = 10*train_vals.shape[1]
    degree = 2
    prev_num_terms = 0
    while True:
        indices =compute_hyperbolic_indices(num_vars, degree, hcross_strength)
        num_terms = indices.shape[1]
        if ( num_terms > max_num_terms ): break
        degree += 1
        prev_num_terms = num_terms

    if ( abs( num_terms - max_num_terms ) > 
         abs( prev_num_terms - max_num_terms ) ):
        degree -=1
    pce.set_indices(
        compute_hyperbolic_indices(num_vars, degree, hcross_strength))

    if verbosity>0:
        msg = f'Initializing basis with hyperbolic cross of degree {degree} and '
        msg += f' strength {hcross_strength} with {pce.num_terms()} terms'
        print(msg)

    basis_matrix = pce.basis_matrix(train_samples)
    best_coef, best_cv_score = fit_linear_model(
        basis_matrix,train_vals,solver_type,cv=cv)
    pce.set_coefficients(best_coef)
    best_indices = pce.get_indices()
    if verbosity>0:
        print ("{:<10} {:<10} {:<18}".format('nterms', 'nnz terms', 'cv score'))
        print ("{:<10} {:<10} {:<18}".format(
            pce.num_terms(),np.count_nonzero(pce.coefficients),best_cv_score))

    best_cv_score_iter = best_cv_score
    best_num_expansion_steps = 3
    it = 0
    best_it = 0
    while True:
        max_num_expansion_steps = 1
        best_num_expansion_steps_iter = best_num_expansion_steps
        while True:
            # -------------- #
            #  Expand basis  #
            # -------------- #
            num_expansion_steps = 0
            indices=restrict_basis(pce.indices,pce.coefficients,restriction_tol)
            while ( ( num_expansion_steps < max_num_expansion_steps ) and
                    ( num_expansion_steps < best_num_expansion_steps ) ):
                new_indices = expand_basis(pce.indices)
                pce.set_indices(np.hstack([pce.indices,new_indices]))
                num_terms = pce.num_terms()
                num_expansion_steps += 1

            # -----------------#
            # Compute solution #
            # -----------------#
            basis_matrix = pce.basis_matrix(train_samples)
            coef, cv_score = fit_linear_model(
                basis_matrix,train_vals,solver_type,cv=cv)
            pce.set_coefficients(coef)

            if verbosity>0:
                print ("{:<10} {:<10} {:<18}".format(
                    pce.num_terms(), np.count_nonzero(pce.coefficients),
                    cv_score))

            if ( cv_score > best_cv_score_iter ):
                best_cv_score_iter = cv_score
                best_indices_iter = pce.indices.copy()
                best_coef_iter = pce.coefficients.copy()
                best_num_expansion_steps_iter = num_expansion_steps 

            if ( num_terms >= max_num_terms ): break
            if ( max_num_expansion_steps >= 3 ): break

            max_num_expansion_steps += 1


        if ( best_cv_score_iter > best_cv_score):
            best_cv_score = best_cv_score_iter
            best_coef = best_coef_iter.copy()
            best_indices = best_indices_iter.copy()
            best_num_expansion_steps = best_num_expansion_steps_iter
            best_it = it
        elif ( it - best_it >= 2 ):
            break

        it += 1

    nindices = best_indices.shape[1]
    I = np.nonzero(best_coef[:,0])[0]
    pce.set_indices(best_indices[:,I])
    pce.set_coefficients(best_coef[I])
    if verbosity>0:
        msg=f'Final basis has {pce.num_terms()} terms selected from {nindices}'
        msg+=f' using {train_samples.shape[1]} samples'
        print(msg)
    return pce, best_cv_score

def approximate_gaussian_process(train_samples,train_vals,nu=np.inf,n_restarts_optimizer=5,verbosity=0):
    r"""
    Compute a Gaussian process approximation of a function from a fixed data 
    set using the Matern kernel

    .. math::

       k(z_i, z_j) =  \frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(
       \frac{\sqrt{2\nu}}{l} \lVert z_i - z_j \rVert_2\Bigg)^\nu K_\nu\Bigg(
       \frac{\sqrt{2\nu}}{l} \lVert z_i - z_j \rVert_2\Bigg)

    where :math:`\lVert \cdot \rVert_2` is the Euclidean distance, 
    :math:`\Gamma(\cdot)` is the gamma function, :math:`K_\nu(\cdot)` is the 
    modified Bessel function.

    Parameters
    ----------
    train_samples : np.ndarray (nvars,nsamples)
        The inputs of the function used to train the approximation

    train_vals : np.ndarray (nvars,nsamples)
        The values of the function at ``train_samples``

    kernel_nu : string
        The parameter :math:`\nu` of the Matern kernel. When :math:`\nu\to\inf`
        the Matern kernel is equivalent to the squared-exponential kernel.

    n_restarts_optimizer : int
        The number of local optimizeation problems solved to find the 
        GP hyper-parameters

    verbosity : integer
        Controls the amount of information printed to screen

    Returns
    -------
    result : :class:`pyapprox.approximate.ApproximateResult`
         Result object with the following attributes

    approx : :class:`pyapprox.gaussian_process.GaussianProcess`
        The Gaussian process
    """
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    from pyapprox.gaussian_process import GaussianProcess
    kernel = Matern(length_scale_bounds=(1e-2, 10), nu=nu)
    # optimize variance
    kernel = 1*kernel
    # optimize gp noise
    nvars = train_samples.shape[0]
    length_scale = np.array([1]*nvars)
    kernel += WhiteKernel(length_scale,noise_level_bounds=(1e-8, 1))
    gp = GaussianProcess(kernel,n_restarts_optimizer=n_restarts_optimizer)
    gp.fit(train_samples,train_vals)
    return ApproximateResult({'approx':gp})
