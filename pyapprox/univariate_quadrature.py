import numpy as np, os
from pyapprox.orthonormal_polynomials_1d import \
     jacobi_recurrence, hermite_recurrence, gauss_quadrature, \
     evaluate_orthonormal_polynomial_deriv_1d, hahn_recurrence, \
     krawtchouk_recurrence, evaluate_orthonormal_polynomial_1d
from pyapprox.univariate_leja import get_christoffel_leja_sequence_1d, \
    get_christoffel_leja_quadrature_weights_1d, \
    get_leja_sequence_quadrature_weights, \
    get_candidate_based_christoffel_leja_sequence_1d, \
    get_pdf_weighted_leja_sequence_1d, \
    get_pdf_weighted_leja_quadrature_weights_1d
from pyapprox.utilities import beta_pdf, beta_pdf_derivative, gaussian_pdf, \
    gaussian_pdf_derivative
from functools import partial
from pyapprox.utilities import evaluate_tensor_product_function,\
     gradient_of_tensor_product_function
from pyapprox.variables import is_continuous_variable, \
    is_bounded_continuous_variable
from pyapprox.variables import get_distribution_info
from pyapprox.numerically_generate_orthonormal_polynomials_1d import \
    modified_chebyshev_orthonormal, predictor_corrector_known_scipy_pdf
from pyapprox.orthonormal_polynomials_1d import \
    discrete_chebyshev_recurrence


def clenshaw_curtis_rule_growth(level):
    """
    The number of samples in the 1D Clenshaw-Curtis quadrature rule of a given
    level.

    Parameters
    ----------
    level : integer
       The level of the quadrature rule

    Return
    ------
    num_samples_1d : integer
        The number of samples in the quadrature rule
    """
    if level == 0:
        return 1
    else:
        return 2**level+1

    
def clenshaw_curtis_hierarchical_to_nodal_index(level, ll, ii):
    """
    Convert a 1D hierarchical index (ll,ii) to a nodal index for lookup in a
    Clenshaw-Curtis quadrature rule. 

    Given a quadrature rule of the specified max level (level)
    with indices [0,1,...,num_indices] this function can be used
    to convert a hierarchical index, e.g. of the constant function 
    (poly_index=0), to the quadrature index, e.g for poly_index=0, 
    index=num_indices/2). This allows one to take advantage of nestedness of 
    quadrature rule and only store quadrature rule for max level.

    Parameters
    ----------
    level : integer
        The maximum level of the quadrature rule

    ll : integer 
        The level of the polynomial index

    ii : integer
        The polynomial index

    Return
    ------
    nodal_index : integer
        The equivalent nodal index of (ll,ii)
    """
    num_indices = clenshaw_curtis_rule_growth(level)
    # mid point
    if ll == 0:
        return num_indices/2
    # boundaries
    elif ll == 1:
        if ii == 0:
            return 0
        else:
            return num_indices-1
    # higher level points
    return (2*ii+1)*2**(level-ll)

def clenshaw_curtis_poly_indices_to_quad_rule_indices(level):
    """
    Convert all 1D polynomial indices of up to and including a given level
    to their equivalent nodal index for lookup in a Clenshaw-Curtis 
    quadrature rule. 

    Parameters
    ----------
    level : integer
        The maximum level of the quadrature rule

    Return
    ------
    quad_rule_indices : np.ndarray (num_vars x num_indices)
        All the quadrature rule indices
    """
    quad_rule_indices = []
    num_previous_hier_indices = 0
    for ll in range(level+1):
        num_hierarchical_indices=\
          clenshaw_curtis_rule_growth(ll)-num_previous_hier_indices
        for ii in range(num_hierarchical_indices):
            quad_index = clenshaw_curtis_hierarchical_to_nodal_index(
                level, ll, ii)
            quad_rule_indices.append(quad_index)
        num_previous_hier_indices += num_hierarchical_indices
    return np.asarray(quad_rule_indices, dtype=int)


def clenshaw_curtis_in_polynomial_order(level,
                                        return_weights_for_all_levels=True):
    """
    Return the samples and weights of the Clenshaw-Curtis rule using 
    polynomial ordering.

    The first point will be the middle weight of the rule. The second and
    third weights will be the left and right boundary weights. All other weights
    left of mid point will come next followed by all remaining points.

    Parameters
    ----------
    level : integer
        The level of the isotropic sparse grid.

    return_weights_for_all_levels : boolean
        True  - return weights [w(0),w(1),...,w(level)]
        False - return w(level)

    Return
    ------
    ordered_samples_1d : np.ndarray (num_samples_1d)
        The reordered samples.

    ordered_weights_1d : np.ndarray (num_samples_1d)
        The reordered weights.
    """

    #w*=2. #use if want do not want to use probability formulation
   
    if return_weights_for_all_levels:
        ordered_weights_1d = []
        for ll in range(level+1):
            x,w = clenshaw_curtis_pts_wts_1D(ll)
            quad_indices = clenshaw_curtis_poly_indices_to_quad_rule_indices(ll)
            ordered_weights_1d.append(w[quad_indices])
        # ordered samples for last x
        ordered_samples_1d = x[quad_indices]
    else:
        x,w = clenshaw_curtis_pts_wts_1D(level)
        quad_indices = clenshaw_curtis_poly_indices_to_quad_rule_indices(level)
        ordered_samples_1d = x[quad_indices]
        ordered_weights_1d = w[quad_indices]
        
    return ordered_samples_1d, ordered_weights_1d


def clenshaw_curtis_pts_wts_1D(level):
    """
    Generated a nested, exponentially-growing Clenshaw-Curtis quadrature rule 
    that exactly integrates polynomials of degree 2**level+1 with respect to 
    the uniform probability measure on [-1,1].

    Parameters
    ----------
    level : integer
        The level of the nested quadrature rule. The number of samples in the
        quadrature rule will be 2**level+1

    Returns
    -------
    x : np.ndarray(num_samples)
        Quadrature samples

    w : np.ndarray(num_samples)
        Quadrature weights
    """

    try:
        from pyapprox.cython.univariate_quadrature import \
            clenshaw_curtis_pts_wts_1D_pyx
        return clenshaw_curtis_pts_wts_1D_pyx(level)
        # from pyapprox.weave.univariate_quadrature import \
        #     c_clenshaw_curtis_pts_wts_1D
        # return c_clenshaw_curtis_pts_wts_1D(level)
    except:
        print ('clenshaw_curtis_pts_wts failed')

    return __clenshaw_curtis_pts_wts_1D(level)


def __clenshaw_curtis_pts_wts_1D(level):
    num_samples = clenshaw_curtis_rule_growth(level)
  
    wt_factor = 1./2.

    x = np.empty((num_samples))
    w = np.empty_like(x)

    if ( level == 0 ):
        x[0]=0.; w[0]=1.;
    else:
        for jj in range(num_samples):
            if ( jj == 0 ):
                x[jj] = -1.;
                w[jj] = wt_factor / float(num_samples*(num_samples -2.))
            elif ( jj == num_samples-1 ):
                x[jj] = 1.;
                w[jj] = wt_factor / float(num_samples*(num_samples-2.))
            else:
                x[jj] = -np.cos(np.pi*float(jj)/float(num_samples-1))
                mysum = 0.0
                for kk in range(1,(num_samples-3)//2+1):
                    mysum += 1. / float(4.*kk*kk-1.)*\
                      np.cos( 2.*np.pi*float(kk*jj)/float(num_samples-1.))
                w[jj] = 2./float(num_samples-1.)*(1.-np.cos(np.pi*float(jj) ) /
                          float(num_samples*(num_samples -2.))-2.*(mysum))
                w[jj] *= wt_factor;
            if ( abs( x[jj] ) < 2.*np.finfo(float).eps): x[jj] = 0.
    return x, w


def gauss_hermite_pts_wts_1D(num_samples):
    """
    Return Gauss Hermite quadrature rule that exactly integrates polynomials
    of degree 2*num_samples-1 with respect to the Gaussian probability measure
    1/sqrt(2*pi)exp(-x**2/2)

    Parameters
    ----------
    num_samples : integer
        The number of samples in the quadrature rule

    Returns
    -------
    x : np.ndarray(num_samples)
        Quadrature samples

    w : np.ndarray(num_samples)
        Quadrature weights
    """
    rho = 0.0
    ab = hermite_recurrence(num_samples, rho,probability=True)
    x, w = gauss_quadrature(ab, num_samples)
    return x, w


def gauss_jacobi_pts_wts_1D(num_samples, alpha_poly, beta_poly):
    """
    Return Gauss Jacobi quadrature rule that exactly integrates polynomials
    of num_samples 2*num_samples-1 with respect to the probabilty density 
    function of Beta random variables on [-1,1]

    C*(1+x)^(beta_poly)*(1-x)^alpha_poly
    
    where
    
    C = 1/(2**(alpha_poly+beta_poly)*beta_fn(beta_poly+1,alpha_poly+1))

    or equivalently
    
    C*(1+x)**(alpha_stat-1)*(1-x)**(beta_stat-1)

    where 

    C = 1/(2**(alpha_stat+beta_stat-2)*beta_fn(alpha_stat,beta_stat))

    Parameters
    ----------
    num_samples : integer
        The number of samples in the quadrature rule

    alpha_poly : float
        The Jaocbi parameter alpha = beta_stat-1

    beta_poly : float
        The Jacobi parameter beta = alpha_stat-1 

    Returns
    -------
    x : np.ndarray(num_samples)
        Quadrature samples

    w : np.ndarray(num_samples)
        Quadrature weights
    """
    ab = jacobi_recurrence(
        num_samples, alpha=alpha_poly, beta=beta_poly, probability=True)
    return gauss_quadrature(ab, num_samples)


def leja_growth_rule(level):
    """
    The number of samples in the 1D Leja quadrature rule of a given
    level. Most leja rules produce two point quadrature rules which 
    have zero weight assigned to one point. Avoid this by skipping from
    one point rule to 3 point rule and then increment by 1.

    Parameters
    ----------
    level : integer
       The level of the quadrature rule

    Return
    ------
    num_samples_1d : integer
        The number of samples in the quadrature rule
    """
    if level == 0:
        return 1
    return level+2


def constant_increment_growth_rule(increment, level):
    """
    The number of samples in the 1D quadrature rule where number of of points 
    grow by a fixed constant at each level.

    Parameters
    ----------
    level : integer
       The level of the quadrature rule

    Return
    ------
    num_samples_1d : integer
        The number of samples in the quadrature rule
    """
    if level == 1:
        return 3
    return increment*level+1


def algebraic_growth(rate, level):
    return (level)**rate+1


def exponential_growth(level, constant=1):
    """
    The number of samples in an exponentially growing 1D quadrature rule of 
    a given level.

    Parameters
    ----------
    level : integer
       The level of the quadrature rule

    Return
    ------
    num_samples_1d : integer
        The number of samples in the quadrature rule
    """
    if level == 0:
        return 1
    return constant*2**(level+1)-1


def exponential_growth_rule(quad_rule, level):
    return quad_rule(exponential_growth(level))


def get_recursion_coefficients(
        opts,
        num_coefs,
        numerically_generated_poly_accuracy_tolerance=1e-12):

    # variables that require numerically generated polynomials with
    # predictor corrector method
    from scipy import stats
    predcorr_rv_dict = {'gumbel_r': stats.gumbel_r, 'lognorm': stats.lognorm}

    poly_type = opts.get('poly_type', None)
    var_type = None
    if poly_type is None:
        var_type = opts['rv_type']
    if poly_type == 'legendre' or var_type == 'uniform':
        recursion_coeffs = jacobi_recurrence(
            num_coefs, alpha=0, beta=0, probability=True)
    elif poly_type == 'jacobi' or var_type == 'beta':
        if poly_type is not None:
            alpha_poly, beta_poly = opts['alpha_poly'], opts['beta_poly']
        else:
            alpha_poly, beta_poly = opts['shapes']['b'] - \
                1, opts['shapes']['a']-1
        recursion_coeffs = jacobi_recurrence(
            num_coefs, alpha=alpha_poly, beta=beta_poly, probability=True)
    elif poly_type == 'hermite' or var_type == 'norm':
        recursion_coeffs = hermite_recurrence(
            num_coefs, rho=0., probability=True)
    elif poly_type == 'krawtchouk' or var_type == 'binom':
        if poly_type is None:
            opts = opts['shapes']
        n, p = opts['n'], opts['p']
        num_coefs = min(num_coefs, n)
        recursion_coeffs = krawtchouk_recurrence(
            num_coefs, n, p)
    elif poly_type == 'hahn' or var_type == 'hypergeom':
        if poly_type is not None:
            apoly, bpoly = opts['alpha_poly'], opts['beta_poly']
            N = opts['N']
        else:
            M, n, N = [opts['shapes'][key] for key in ['M', 'n', 'N']]
            apoly, bpoly = -(n+1), -M-1+n
        num_coefs = min(num_coefs, N)
        recursion_coeffs = hahn_recurrence(num_coefs, N, apoly, bpoly)
    elif poly_type == 'discrete_chebyshev' or var_type == 'discrete_chebyshev':
        if poly_type is not None:
            N = opts['N']
        else:
            N = opts['shapes']['xk'].shape[0]
            assert np.allclose(opts['shapes']['xk'], np.arange(N))
            assert np.allclose(opts['shapes']['pk'], np.ones(N)/N)
        num_coefs = min(num_coefs, N)
        recursion_coeffs = discrete_chebyshev_recurrence(num_coefs, N)
    elif poly_type == 'discrete_numeric' or var_type == 'float_rv_discrete':
        if poly_type is None:
            opts = opts['shapes']
        xk, pk = opts['xk'], opts['pk']
        # shapes['xk'] will be in [0,1] but canonical domain is [-1,1]
        xk = xk*2-1
        assert xk.min() >= -1 and xk.max() <= 1
        if num_coefs > xk.shape[0]:
            msg = 'Number of coefs requested is larger than number of '
            msg += 'probability masses'
            raise Exception(msg)
        recursion_coeffs = modified_chebyshev_orthonormal(num_coefs, [xk, pk])
        p = evaluate_orthonormal_polynomial_1d(
            np.asarray(xk, dtype=float), num_coefs-1, recursion_coeffs)
        error = np.absolute((p.T*pk).dot(p)-np.eye(num_coefs)).max()
        if error > numerically_generated_poly_accuracy_tolerance:
            msg = f'basis created is ill conditioned. '
            msg += f'Max error: {error}. Max terms: {xk.shape[0]}, '
            msg += f'Terms requested: {num_coefs}'
            raise Exception(msg)
    elif (poly_type == 'continuous_numeric' or
          var_type == 'continuous_rv_sample'):
        if poly_type is None:
            opts = opts['shapes']
        xk, pk = opts['xk'], opts['pk']
        if num_coefs > xk.shape[0]:
            msg = 'Number of coefs requested is larger than number of '
            msg += 'samples'
            raise Exception(msg)
        print(num_coefs)
        recursion_coeffs = modified_chebyshev_orthonormal(num_coefs, [xk, pk])
        p = evaluate_orthonormal_polynomial_1d(
            np.asarray(xk, dtype=float), num_coefs-1, recursion_coeffs)
        error = np.absolute((p.T*pk).dot(p)-np.eye(num_coefs)).max()
        if error > numerically_generated_poly_accuracy_tolerance:
            msg = f'basis created is ill conditioned. '
            msg += f'Max error: {error}. Max terms: {xk.shape[0]}, '
            msg += f'Terms requested: {num_coefs}'
            raise Exception(msg)
    elif poly_type == 'monomial':
        recursion_coeffs = None
    elif var_type in predcorr_rv_dict:
       quad_options = {
           'nquad_samples': 10,
           'atol': numerically_generated_poly_accuracy_tolerance,
           'rtol': numerically_generated_poly_accuracy_tolerance,
           'max_steps': 10000, 'verbose': 0}
       rv = predcorr_rv_dict[var_type](**opts['shapes'])
       recursion_coeffs = predictor_corrector_known_scipy_pdf(
           num_coefs, rv, quad_options)
    else:
        if poly_type is not None:
            raise Exception('poly_type (%s) not supported' % poly_type)
        else:
            raise Exception('var_type (%s) not supported' % var_type)
    return recursion_coeffs


def candidate_based_christoffel_leja_rule_1d(
        recursion_coeffs, generate_candidate_samples,num_candidate_samples,
        level, initial_points=None, growth_rule=leja_growth_rule,
        samples_filename=None, return_weights_for_all_levels=True):

    num_vars = 1
    num_leja_samples = growth_rule(level)
    
    leja_sequence = get_candidate_based_christoffel_leja_sequence_1d(
        num_leja_samples, recursion_coeffs, generate_candidate_samples,
        num_candidate_samples, initial_points, samples_filename)

    from pyapprox.polynomial_sampling import christoffel_weights

    generate_basis_matrix = lambda x: evaluate_orthonormal_polynomial_deriv_1d(
        x[0, :], num_leja_samples, recursion_coeffs, deriv_order=0)

    weight_function = lambda x: christoffel_weights(generate_basis_matrix(x))
    ordered_weights_1d = get_leja_sequence_quadrature_weights(
        leja_sequence, growth_rule, generate_basis_matrix, weight_function,
        level, return_weights_for_all_levels)

    return leja_sequence[0, :], ordered_weights_1d


def univariate_christoffel_leja_quadrature_rule(
        variable, growth_rule, level, return_weights_for_all_levels=True,
        initial_points=None,
        numerically_generated_poly_accuracy_tolerance=1e-12):
    """
    Return the samples and weights of the Leja quadrature rule for any 
    continuous variable using the inverse Christoffel weight function

    By construction these rules have polynomial ordering.

    Parameters
    ----------
    variable : scipy.stats.dist
        The variable used to construct an orthogonormal polynomial

    growth_rule : callable
        Function which returns the number of samples in the quadrature rule
        With signature

        `growth_rule(level) -> integer`

        where level is an integer
    
    level : integer
        The level of the univariate rule.

    return_weights_for_all_levels : boolean
        True  - return weights [w(0),w(1),...,w(level)]
        False - return w(level)

    initial_points : np.ndarray (1, ninit_samples)
        Any points that must be included in the Leja sequence. This argument
        is typically used to pass in previously computed sequence which
        is updated efficiently here.

    Return
    ------
    ordered_samples_1d : np.ndarray (num_samples_1d)
        The reordered samples.

    ordered_weights_1d : np.ndarray (num_samples_1d)
        The reordered weights.
    """
    if not is_continuous_variable(variable):
        raise Exception('Only supports continuous variables')
    
    name, scales, shapes = get_distribution_info(variable)
    opts = {'rv_type':name, 'shapes':shapes, 'var_nums':variable}
    max_nsamples = growth_rule(level)
    ab = get_recursion_coefficients(
        opts, max_nsamples+1, numerically_generated_poly_accuracy_tolerance)
    basis_fun = partial(
            evaluate_orthonormal_polynomial_deriv_1d, ab=ab)
    
    if is_bounded_continuous_variable(variable):
        bounds = variable.interval(1.)
        canonical_bounds = [-1, 1]
        if initial_points is None:
            initial_points = np.asarray(
                [[variable.ppf(0.5)]]).T
            loc, scale = scales['loc'], scales['scale']
            lb, ub = -1, 1
            scale /= (ub-lb)
            loc = loc-scale*lb
            initial_points = (initial_points-loc)/scale
        eps = 1e-13#np.finfo(float).eps
        assert np.all((initial_points>=canonical_bounds[0]-eps)&
                      (initial_points<=canonical_bounds[1]+eps))
        # always produce sequence in canonical space
        bounds = canonical_bounds
    else:
        bounds = list(variable.interval(1))
        if variable.dist.name == 'continuous_rv_sample':
            bounds = [-np.inf, np.inf]
        #if not np.isfinite(bounds[0]):
        #    bounds[0] = -1e6
        #if not np.isfinite(bounds[1]):
        #    bounds[1] = 1e6
        if initial_points is None:
            # creating a leja sequence with initial points == 0
            # e.g. norm(0, 1).ppf(0.5) will cause leja sequence to
            # try to add point at infinity. So use different initial point
            initial_points = np.asarray(
                [[variable.ppf(0.75)]]).T
            loc, scale = scales['loc'], scales['scale']
            initial_points = (initial_points-loc)/scale
        if initial_points.shape[1] == 1:
            assert initial_points[0, 0] != 0

    leja_sequence = get_christoffel_leja_sequence_1d(
        max_nsamples, initial_points, bounds, basis_fun,
        {'gtol':1e-8, 'verbose':False}, callback=None)
    
    __basis_fun = partial(basis_fun, nmax=max_nsamples-1, deriv_order=0)
    ordered_weights_1d =  get_christoffel_leja_quadrature_weights_1d(
            leja_sequence, growth_rule, __basis_fun, level, True)
    return leja_sequence[0, :], ordered_weights_1d


def get_pdf_weight_functions(variable):
    name, scales, shapes = get_distribution_info(variable)
    if name == 'uniform' or name == 'beta':
        if name == 'uniform':
            alpha_stat, beta_stat = 1, 1
        else:
            alpha_stat, beta_stat = shapes['a'], shapes['b']
        def pdf(x):
            return beta_pdf(alpha_stat, beta_stat, (x+1)/2)/2
        def pdf_jac(x):
            return beta_pdf_derivative(alpha_stat, beta_stat, (x+1)/2)/4
        return pdf, pdf_jac
    
    if name == 'norm':
        return partial(gaussian_pdf, 0, 1), \
            partial(gaussian_pdf_derivative, 0, 1)

    raise Exception(f'var_type (name) not supported')


def univariate_pdf_weighted_leja_quadrature_rule(
        variable, growth_rule, level, return_weights_for_all_levels=True,
        initial_points=None):
    """
    Return the samples and weights of the Leja quadrature rule for any 
    continuous variable using the PDF of the random variable as the 
    weight function

    By construction these rules have polynomial ordering.

    Parameters
    ----------
    variable : scipy.stats.dist
        The variable used to construct an orthogonormal polynomial

    growth_rule : callable
        Function which returns the number of samples in the quadrature rule
        With signature

        `growth_rule(level) -> integer`

        where level is an integer
    
    level : integer
        The level of the univariate rule.

    return_weights_for_all_levels : boolean
        True  - return weights [w(0),w(1),...,w(level)]
        False - return w(level)

    initial_points : np.ndarray (1, ninit_samples)
        Any points that must be included in the Leja sequence. This argument
        is typically used to pass in previously computed sequence which
        is updated efficiently here.

    Return
    ------
    ordered_samples_1d : np.ndarray (num_samples_1d)
        The reordered samples.

    ordered_weights_1d : np.ndarray (num_samples_1d)
        The reordered weights.
    """
    if not is_continuous_variable(variable):
        raise Exception('Only supports continuous variables')
    
    name, scales, shapes = get_distribution_info(variable)
    opts = {'rv_type':name, 'shapes':shapes,
            'var_nums':variable}
    max_nsamples = growth_rule(level)
    ab = get_recursion_coefficients(opts, max_nsamples+1)
    basis_fun = partial(evaluate_orthonormal_polynomial_deriv_1d, ab=ab)

    pdf, pdf_jac = get_pdf_weight_functions(variable)
    
    if is_bounded_continuous_variable(variable):
        bounds = variable.interval(1.)
        canonical_bounds = [-1, 1]
        if initial_points is None:
            initial_points = np.asarray(
                [[variable.ppf(0.5)]]).T
            loc, scale = scales['loc'], scales['scale']
            lb, ub = -1, 1
            scale /= (ub-lb)
            loc = loc-scale*lb
            initial_points = (initial_points-loc)/scale  
        assert np.all((initial_points>=canonical_bounds[0])&
                      (initial_points<=canonical_bounds[1]))
        # always produce sequence in canonical space
        bounds = canonical_bounds
    else:
        bounds = list(variable.interval(1))
        if initial_points is None:
            initial_points = np.asarray(
                [[variable.ppf(0.5)]]).T
            loc, scale = scales['loc'], scales['scale']
            initial_points = (initial_points-loc)/scale

    leja_sequence = get_pdf_weighted_leja_sequence_1d(
        max_nsamples, initial_points, bounds, basis_fun, pdf, pdf_jac,
        {'gtol':1e-8, 'verbose':False}, callback=None)
    
    __basis_fun = partial(basis_fun, nmax=max_nsamples-1, deriv_order=0)
    ordered_weights_1d =  get_pdf_weighted_leja_quadrature_weights_1d(
            leja_sequence, growth_rule, pdf, __basis_fun, level, True)

    return leja_sequence[0, :], ordered_weights_1d


def get_discrete_univariate_leja_quadrature_rule(variable, growth_rule):
    var_type, __, shapes = get_distribution_info(variable)
    if var_type == 'binom':
        num_trials = variable_parameters['num_trials']
        prob_success = variable_parameters['prob_success']
        def generate_candidate_samples(num_samples):
            assert num_samples == num_trials+1
            return np.arange(0, num_trials+1)[np.newaxis, :]
        recursion_coeffs = krawtchouk_recurrence(
            num_trials, num_trials, probability=True)
        quad_rule = partial(
            candidate_based_christoffel_leja_rule_1d, recursion_coeffs,
            generate_candidate_samples, num_trials+1, growth_rule=growth_rule,
            initial_points=np.atleast_2d(
                [binomial_rv.ppf(0.5, num_trials, prob_success)]))
    elif var_type == 'float_rv_discrete' or var_type == 'discrete_chebyshev':
        nmasses = shapes['xk'].shape[0]
        if var_type == 'discrete_chebyshev':
            xk = shapes['xk']#do not map discrete_chebyshev
            assert np.allclose(shapes['xk'], np.arange(nmasses))
            assert np.allclose(shapes['pk'], np.ones(nmasses)/nmasses)
            num_coefs = nmasses
            recursion_coeffs = discrete_chebyshev_recurrence(
                num_coefs, nmasses)
        else:
            #shapes['xk'] will be in [0,1] but canonical domain is [-1,1]
            xk = shapes['xk']*2-1
            num_coefs = nmasses
            recursion_coeffs  = modified_chebyshev_orthonormal(
                num_coefs, [xk, shapes['pk']])

        def generate_candidate_samples(num_samples):
            assert num_samples == nmasses
            return xk[np.newaxis, :]

        # do not specify init_samples in partial or a sparse grid cannot
        # update the samples_1d so that next level has same samples_1d
        # TODO: add test that samples_1d[ii] and samples_1d[ii+1] subset
        # are equal
        #init_samples = np.atleast_2d(np.sort(xk)[nmasses//2])
        quad_rule = partial(
            candidate_based_christoffel_leja_rule_1d, recursion_coeffs,
            generate_candidate_samples, nmasses,
            growth_rule=growth_rule)        
    else:
        raise Exception('var_type %s not implemented'%var_type)
    return quad_rule
    

def get_univariate_leja_quadrature_rule(
        variable,
        growth_rule,
        method='pdf',
        numerically_generated_poly_accuracy_tolerance=1e-12,
        initial_points=None):
    
    if not is_continuous_variable(variable):
        return get_discrete_univariate_leja_quadrature_rule(
            variable, growth_rule, initial_points=initial_points)
    
    if method == 'christoffel':
        return partial(
            univariate_christoffel_leja_quadrature_rule,variable, growth_rule,
            numerically_generated_poly_accuracy_tolerance=numerically_generated_poly_accuracy_tolerance, initial_points=initial_points)

    if method == 'pdf':
        return partial(univariate_pdf_weighted_leja_quadrature_rule,
                       variable, growth_rule, initial_points=initial_points)

    assert method == 'deprecated'
    var_type, __, shapes = get_distribution_info(variable)
    if var_type == 'uniform':
        quad_rule = partial(
            beta_leja_quadrature_rule, 1, 1, growth_rule=growth_rule,
            samples_filename=None)
    elif var_type == 'beta':
        quad_rule = partial(
            beta_leja_quadrature_rule,shapes['a'], shapes['b'],
            growth_rule=growth_rule)
    elif var_type == 'norm':
        quad_rule = partial(
            gaussian_leja_quadrature_rule, growth_rule=growth_rule)
    else:
        raise Exception('var_type %s not implemented'%var_type)
    
    return quad_rule



# ------------------------------------------
# The Following functions will be deprecated
# ------------------------------------------

def uniform_leja_quadrature_rule(level, growth_rule=leja_growth_rule,
                                 samples_filename=None,
                                 return_weights_for_all_levels=True):
    return beta_leja_quadrature_rule(1, 1, level,growth_rule,samples_filename,
                                     return_weights_for_all_levels)


def beta_leja_quadrature_rule(alpha_stat, beta_stat, level,
                              growth_rule=leja_growth_rule,
                              samples_filename=None,
                              return_weights_for_all_levels=True,
                              initial_points=None):
    """
    Return the samples and weights of the Leja quadrature rule for the beta
    probability measure. 

    By construction these rules have polynomial ordering.

    Parameters
    ----------
    level : integer
        The level of the isotropic sparse grid.

    alpha_stat : integer
        The alpha shape parameter of the Beta distribution

    beta_stat : integer
        The beta shape parameter of the Beta distribution

    samples_filename : string
         Name of file to save leja samples and weights to

    Return
    ------
    ordered_samples_1d : np.ndarray (num_samples_1d)
        The reordered samples.

    ordered_weights_1d : np.ndarray (num_samples_1d)
        The reordered weights.
    """
    from pyapprox.multivariate_polynomials import PolynomialChaosExpansion
    from pyapprox.leja_sequences import get_leja_sequence_1d,\
        get_quadrature_weights_from_samples
    num_vars = 1
    num_leja_samples = growth_rule(level)
    #print(('num_leja_samples',num_leja_samples))

    # freezing beta rv like below has huge overhead
    # it creates a docstring each time which adds up to many seconds
    # for repeated calls to pdf
    #univariate_weight_function=lambda x: beta_rv(
    #    alpha_stat,beta_stat).pdf((x+1)/2)/2
    #univariate_weight_function = lambda x: beta_rv.pdf(
    #    (x+1)/2,alpha_stat,beta_stat)/2
    univariate_weight_function = lambda x: beta_pdf(
        alpha_stat, beta_stat, (x+1)/2)/2
    univariate_weight_function_deriv = lambda x: beta_pdf_derivative(
       alpha_stat, beta_stat, (x+1)/2)/4
    
    weight_function = partial(
        evaluate_tensor_product_function,
        [univariate_weight_function]*num_vars)
    
    weight_function_deriv = partial(
        gradient_of_tensor_product_function,
        [univariate_weight_function]*num_vars,
        [univariate_weight_function_deriv]*num_vars)

    # assert np.allclose(
    #     (univariate_weight_function(0.5+1e-8)-
    #          univariate_weight_function(0.5))/1e-8,
    #     univariate_weight_function_deriv(0.5),atol=1e-6)

    poly = PolynomialChaosExpansion()
    # must be imported locally otherwise I have a circular dependency
    from pyapprox.variable_transformations import \
        define_iid_random_variable_transformation
    from scipy.stats import uniform
    var_trans = define_iid_random_variable_transformation(
        uniform(-1, 2), num_vars)

    poly_opts = {'poly_type':'jacobi', 'alpha_poly':beta_stat-1,
                 'beta_poly':alpha_stat-1, 'var_trans':var_trans}
    poly.configure(poly_opts) 

    if samples_filename is None or not os.path.exists(samples_filename):
        ranges = [-1, 1]
        from scipy.stats import beta as beta_rv
        if initial_points is None:
            initial_points = np.asarray(
                [[2*beta_rv(alpha_stat, beta_stat).ppf(0.5)-1]]).T
        leja_sequence = get_leja_sequence_1d(
            num_leja_samples, initial_points, poly,
            weight_function, weight_function_deriv, ranges)
        if samples_filename is not None:
            np.savez(samples_filename, samples=leja_sequence)
    else:
        leja_sequence = np.load(samples_filename)['samples']
        #print (leja_sequence.shape[1],growth_rule(level),level)
        assert leja_sequence.shape[1] >= growth_rule(level)
        leja_sequence = leja_sequence[:, :growth_rule(level)]

    indices = np.arange(growth_rule(level))[np.newaxis, :]
    poly.set_indices(indices)
    ordered_weights_1d = get_leja_sequence_quadrature_weights(
        leja_sequence, growth_rule, poly.basis_matrix, weight_function, level,
        return_weights_for_all_levels)
    return leja_sequence[0, :], ordered_weights_1d


def gaussian_leja_quadrature_rule(level,
                                  growth_rule=leja_growth_rule,
                                  samples_filename=None,
                                  return_weights_for_all_levels=True,
                                  initial_points=None):
    """
    Return the samples and weights of the Leja quadrature rule for the beta
    probability measure. 

    By construction these rules have polynomial ordering.

    Parameters
    ----------
    level : integer
        The level of the isotropic sparse grid.

    samples_filename : string
         Name of file to save leja samples and weights to

    Return
    ------
    ordered_samples_1d : np.ndarray (num_samples_1d)
        The reordered samples.

    ordered_weights_1d : np.ndarray (num_samples_1d)
        The reordered weights.
    """
    from pyapprox.multivariate_polynomials import PolynomialChaosExpansion
    from pyapprox.leja_sequences import get_leja_sequence_1d,\
        get_quadrature_weights_from_samples

    # freezing scipy gaussian rv like below has huge overhead
    # it creates a docstring each time which adds up to many seconds
    # for repeated calls to pdf
    from pyapprox.utilities import gaussian_pdf, gaussian_pdf_derivative
    univariate_weight_function = partial(gaussian_pdf, 0, 1)
    univariate_weight_function_deriv = partial(gaussian_pdf_derivative, 0, 1)

    num_vars = 1
    num_leja_samples = growth_rule(level)
    
    weight_function = partial(
        evaluate_tensor_product_function,
        [univariate_weight_function]*num_vars)
    
    weight_function_deriv = partial(
        gradient_of_tensor_product_function,
        [univariate_weight_function]*num_vars,
        [univariate_weight_function_deriv]*num_vars)

    assert np.allclose(
        (univariate_weight_function(0.5+1e-8)-
             univariate_weight_function(0.5))/1e-8,
        univariate_weight_function_deriv(0.5), atol=1e-6)

    poly = PolynomialChaosExpansion()
    # must be imported locally otherwise I have a circular dependency
    from pyapprox.variable_transformations import \
        define_iid_random_variable_transformation
    from scipy.stats import norm
    var_trans = define_iid_random_variable_transformation(
        norm(), num_vars)
    poly_opts = {'poly_type':'hermite', 'var_trans':var_trans}
    poly.configure(poly_opts) 

    if samples_filename is None or not os.path.exists(samples_filename):
        ranges = [None,None]
        if initial_points is None:
            initial_points = np.asarray([[0.0]]).T
        leja_sequence = get_leja_sequence_1d(
            num_leja_samples, initial_points, poly,
            weight_function, weight_function_deriv, ranges)
        if samples_filename is not None:
            np.savez(samples_filename, samples=leja_sequence)
    else:
        leja_sequence = np.load(samples_filename)['samples']
        assert leja_sequence.shape[1]>=growth_rule(level)
        leja_sequence = leja_sequence[:,:growth_rule(level)]

    indices = np.arange(growth_rule(level))[np.newaxis,:]
    poly.set_indices(indices)
    ordered_weights_1d = get_leja_sequence_quadrature_weights(
        leja_sequence,growth_rule,poly.basis_matrix,weight_function,level,
        return_weights_for_all_levels)
    return leja_sequence[0, :], ordered_weights_1d
