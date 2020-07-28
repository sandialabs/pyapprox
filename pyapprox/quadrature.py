from scipy.optimize import OptimizeResult
class QuadratureResult(OptimizeResult):
    pass

def compute_mean_and_variance_sparse_grid(sparse_grid,max_order=2):
    """
    Compute the mean and variance of a sparse_grid by converting it to 
    a polynomial chaos expansion

    Parameters
    ----------
    sparse_grid :class:`pyapprox.adaptive_sparse_grid:CombinationSparseGrid`
       The sparse grid

    Returns
    -------
    result : :class:`pyapprox.quadrature.QuadratureResult`
        Result object with the following attributes

    mean : np.ndarray (nqoi)
        The mean of each quantitity of interest

    variance : np.ndarray (nqoi)
        The variance of each quantitity of interest

    pce : :class:`multivariate_polynomials.PolynomialChaosExpansion`
       The pce respresentation of the sparse grid ``approx``
    """
    from pyapprox.multivariate_polynomials import \
        define_poly_options_from_variable_transformation
    from pyapprox.adaptive_sparse_grid import \
        convert_sparse_grid_to_polynomial_chaos_expansion
    pce_opts=define_poly_options_from_variable_transformation(
        sparse_grid.variable_transformation)
    pce = convert_sparse_grid_to_polynomial_chaos_expansion(
        sparse_grid,pce_opts)
    pce_main_effects,pce_total_effects=\
        get_main_and_total_effect_indices_from_pce(
            pce.get_coefficients(),pce.get_indices())

    interaction_terms, pce_sobol_indices = get_sobol_indices(
            pce.get_coefficients(),pce.get_indices(),max_order=max_order)
    
    return QuadratureResult(
        {'mean':pce.mean(),
         'variance':pce.variance(),
         'pce':pce})
