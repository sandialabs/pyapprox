import copy
import numpy as np
from functools import partial
from scipy.optimize import OptimizeResult
from scipy.linalg import LinAlgWarning

from sklearn.linear_model import (
    LassoCV, LassoLarsCV, LarsCV, OrthogonalMatchingPursuitCV, Lasso,
    LassoLars, Lars, OrthogonalMatchingPursuit
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.linear_model._base import LinearModel

from pyapprox.util.utilities import hash_array
from pyapprox.surrogates.interp.indexing import (
    get_forward_neighbor, get_backward_neighbor,
    compute_hyperbolic_indices
)
from pyapprox.variables.sampling import (
    generate_independent_random_samples
)
from pyapprox.surrogates.polychaos.adaptive_polynomial_chaos import (
    AdaptiveLejaPCE, variance_pce_refinement_indicator, AdaptiveInducedPCE
)
from pyapprox.surrogates.orthopoly.leja_sequences import christoffel_weights
from pyapprox.variables.marginals import is_bounded_continuous_variable
from pyapprox.surrogates.interp.adaptive_sparse_grid import (
    variance_refinement_indicator,
    CombinationSparseGrid, constant_increment_growth_rule,
    get_sparse_grid_univariate_leja_quadrature_rules_economical,
    max_level_admissibility_function, get_unique_max_level_1d,
    get_unique_quadrule_variables
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.transforms import (
    AffineTransform
)
from pyapprox.expdesign.low_discrepancy_sequences import halton_sequence
from pyapprox.surrogates.gaussianprocess.gaussian_process import (
    AdaptiveGaussianProcess, CholeskySampler, GaussianProcess
)
from pyapprox.surrogates.gaussianprocess.kernels import (
    Matern, WhiteKernel, ConstantKernel)
from pyapprox.surrogates.polychaos.gpc import (
    PolynomialChaosExpansion, define_poly_options_from_variable_transformation
)
from pyapprox.surrogates.neural_networks import NeuralNetwork


class ApproximateResult(OptimizeResult):
    pass


def adaptive_approximate_sparse_grid(
        fun, variables, callback=None,
        refinement_indicator=variance_refinement_indicator,
        univariate_quad_rule_info=None, max_nsamples=100, tol=0, verbose=0,
        config_variables_idx=None, config_var_trans=None, cost_function=None,
        max_level_1d=None, max_level=None, basis_type="barycentric"):
    r"""
    Compute a sparse grid approximation of a function.

    Parameters
    ----------
    fun : callable
        The function to be approximated

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,nqoi)

    variables : IndependentMarginalsVariable
        A set of independent univariate random variables

    callback : callable
        Function called after each iteration with the signature

        ``callback(approx_k)``

        where approx_k is the current approximation object.

    refinement_indicator : callable
        A function that retuns an estimate of the error of a sparse grid
        subspace with signature

        ``refinement_indicator(subspace_index,nnew_subspace_samples,sparse_grid) -> float, float``

        where ``subspace_index`` is 1D np.ndarray of size (nvars),
        ``nnew_subspace_samples`` is an integer specifying the number
        of new samples that will be added to the sparse grid by adding the
        subspace specified by subspace_index and ``sparse_grid`` is the current
        :class:`pyapprox.adaptive_sparse_grid.CombinationSparseGrid` object.
        The two outputs are, respectively, the indicator used to control
        refinement of the sparse grid and the change in error from adding the
        current subspace. The indicator is typically but now always dependent
        on the error.

    univariate_quad_rule_info : list
        List containing four entries. The first entry is a list
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

        The third entry is a list of np.ndarray (or single scalar) specifying
        the variable dimensions that each unique quadrature rule is applied to.

        The forth entry is a list which specifies the maximum level of each
        unique quadrature rule. If None then max_level is assumed to be np.inf
        for each quadrature rule. If a scalar then the same value is applied
        to all quadrature rules. This entry is useful for certain quadrature
        rules, e.g. Gauss Patterson, or Leja sequences for bounded discrete
        variables where there is a limit on the number of levels that can be
        used

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
        is the cost of evaluating ``fun`` at ``config_sample``. The units can
        be anything but the units must be consistent with the units of
        max_nsamples which specifies the maximum cost of constructing the
        sparse grid.

    max_level_1d : np.ndarray (nvars)
        The maximum level of the sparse grid in each dimension. If None
        There is no limit

    max_level : integer
        The maximum level l of the sparse grid. Only subspaces with indices
        i that satisfy : math:`\lvert i \rvert_1\le l` can be added. If None
        l=np.inf

    basis_type : string (default="barycentric")
        Specify the basis type to use. Currently the same basis must be used
        for all dimensions. Options "barycentric", "linear", "quadratic"


    Returns
    -------
    result : :class:`pyapprox.surrogates.approximate.ApproximateResult`
         Result object with the following attributes

    approx : :class:`pyapprox.adaptive_sparse_grid.CombinationSparseGrid`
        The sparse grid approximation
    """
    var_trans = AffineTransform(variables)
    nvars = var_trans.num_vars()
    if config_var_trans is not None:
        nvars += config_var_trans.num_vars()
    sparse_grid = CombinationSparseGrid(nvars, basis_type)

    if max_level_1d is None:
        max_level_1d = [np.inf]*nvars
    elif np.isscalar(max_level_1d):
        max_level_1d = [max_level_1d]*nvars

    if max_level is None:
        max_level = np.inf

    if univariate_quad_rule_info is None:
        quad_rules, growth_rules, unique_quadrule_indices, \
            unique_max_level_1d = \
            get_sparse_grid_univariate_leja_quadrature_rules_economical(
                var_trans, method='pdf')
        # Some quadrature rules have max_level enforce this here
        for ii in range(len(unique_quadrule_indices)):
            for ind in unique_quadrule_indices[ii]:
                max_level_1d[ind] = min(
                    max_level_1d[ind], unique_max_level_1d[ii])
    else:
        quad_rules, growth_rules, unique_quadrule_indices, \
            unique_max_level_1d = univariate_quad_rule_info
        if unique_max_level_1d is None:
            max_level_1d = np.minimum([np.inf]*nvars, max_level_1d)
        elif np.isscalar(unique_max_level_1d):
            max_level_1d = np.minimum(
                [unique_max_level_1d]*nvars, max_level_1d)
        else:
            nunique_vars = len(quad_rules)
            assert len(unique_max_level_1d) == nunique_vars
            for ii in range(nunique_vars):
                for jj in unique_quadrule_indices[ii]:
                    max_level_1d[jj] = np.minimum(
                        unique_max_level_1d[ii], max_level_1d[jj])

    if config_var_trans is not None:
        if max_level_1d is None:
            msg = "max_level_1d must be set if config_var_trans is provided"
            #raise ValueError(msg)
        for ii, cv in enumerate(config_var_trans.config_values):
            if len(cv) <= max_level_1d[config_variables_idx+ii]:
                msg = f"maxlevel_1d {max_level_1d} and "
                msg += "config_var_trans.config_values with shapes {0}".format(
                    [len(v) for v in config_var_trans.config_values])
                msg += " are inconsistent."
                raise ValueError(msg)

    assert len(max_level_1d) == nvars
    # todo change np.inf to argument that is passed into approximate
    admissibility_function = partial(
        max_level_admissibility_function, max_level, max_level_1d, max_nsamples,
        tol, verbose=verbose)
    sparse_grid.setup(
        fun, config_variables_idx, refinement_indicator,
        admissibility_function, growth_rules, quad_rules,
        var_trans, unique_quadrule_indices=unique_quadrule_indices,
        verbose=verbose, cost_function=cost_function,
        config_var_trans=config_var_trans)
    sparse_grid.build(callback)
    return ApproximateResult({'approx': sparse_grid})


def adaptive_approximate_polynomial_chaos(
        fun, variable, method="leja", options={}):
    methods = {"leja": adaptive_approximate_polynomial_chaos_leja,
               "induced": adaptive_approximate_polynomial_chaos_induced}
    # "random": adaptive_approximate_polynomial_chaos_random}

    if method not in methods:
        msg = f'Method {method} not found.\n Available methods are:\n'
        for key in methods.keys():
            msg += f"\t{key}\n"
        raise Exception(msg)

    if options is None:
        options = {}
    return methods[method](fun, variable, **options)


def __initialize_leja_pce(
        variables, generate_candidate_samples, ncandidate_samples):

    for rv in variables.marginals():
        if not is_bounded_continuous_variable(rv):
            msg = "For now leja sampling based PCE is only supported for "
            msg += " bounded continouous random variables when"
            msg += " generate_candidate_samples is not provided."
            if generate_candidate_samples is None:
                raise Exception(msg)
            else:
                break

    nvars = variables.num_vars()
    if generate_candidate_samples is None:
        # Todo implement default for non-bounded variables that uses induced
        # sampling
        # candidate samples must be in canonical domain
        candidate_samples = -np.cos(
            np.pi*halton_sequence(nvars, int(ncandidate_samples), 1))
        # candidate_samples = -np.cos(
        #    np.random.uniform(0,np.pi,(nvars,int(ncandidate_samples))))
    else:
        candidate_samples = generate_candidate_samples(ncandidate_samples)

    pce = AdaptiveLejaPCE(
        nvars, candidate_samples, factorization_type='fast')
    return pce


def __setup_adaptive_pce(pce, verbose, fun, var_trans, growth_rules,
                         refinement_indicator, tol, max_nsamples, callback,
                         max_level_1d):
    pce.verbose = verbose
    pce.set_function(fun, var_trans)
    if growth_rules is None:
        growth_incr = 2
        growth_rules = partial(constant_increment_growth_rule, growth_incr)
    assert callable(growth_rules)
    unique_quadrule_variables, unique_quadrule_indices = \
        get_unique_quadrule_variables(var_trans)
    growth_rules = [growth_rules]*len(unique_quadrule_indices)

    admissibility_function = None  # provide after growth_rules have been added
    pce.set_refinement_functions(
        refinement_indicator, admissibility_function, growth_rules,
        unique_quadrule_indices=unique_quadrule_indices)

    nvars = var_trans.num_vars()
    if max_level_1d is None:
        max_level_1d = [np.inf]*nvars
    elif np.isscalar(max_level_1d):
        max_level_1d = [max_level_1d]*nvars

    unique_max_level_1d = get_unique_max_level_1d(
        var_trans, pce.compact_univariate_growth_rule)
    nunique_vars = len(unique_quadrule_indices)
    assert len(unique_max_level_1d) == nunique_vars
    for ii in range(nunique_vars):
        for jj in unique_quadrule_indices[ii]:
            max_level_1d[jj] = np.minimum(
                unique_max_level_1d[ii], max_level_1d[jj])

    admissibility_function = partial(
        max_level_admissibility_function, np.inf, max_level_1d, max_nsamples,
        tol, verbose=verbose)
    pce.admissibility_function = admissibility_function


def adaptive_approximate_polynomial_chaos_induced(
        fun, variables,
        callback=None,
        refinement_indicator=variance_pce_refinement_indicator,
        growth_rules=None, max_nsamples=100, tol=0, verbose=0,
        max_level_1d=None, induced_sampling=True, cond_tol=1e6,
        fit_opts={'omp_tol': 0}):
    r"""
    Compute an adaptive Polynomial Chaos Expansion of a function based upon
    random induced or probability measure sampling.

    Parameters
    ----------
    fun : callable
        The function to be minimized

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,nqoi)

    variables : IndependentMarginalsVariable
        A set of independent univariate random variables

    callback : callable
        Function called after each iteration with the signature

        ``callback(approx_k)``

        where approx_k is the current approximation object.

    refinement_indicator : callable
        A function that retuns an estimate of the error of a sparse grid
        subspace with signature

        ``refinement_indicator(subspace_index,nnew_subspace_samples,sparse_grid) -> float, float``

        where ``subspace_index`` is 1D np.ndarray of size (nvars),
        ``nnew_subspace_samples`` is an integer specifying the number

        of new samples that will be added to the sparse grid by adding the
        subspace specified by subspace_index and ``sparse_grid`` is the current
        :class:`pyapprox.adaptive_sparse_grid.CombinationSparseGrid` object.
        The two outputs are, respectively, the indicator used to control
        refinement of the sparse grid and the change in error from adding the
        current subspace. The indicator is typically but now always dependent
        on the error.

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

    max_level_1d : np.ndarray (nvars)
        The maximum level of the sparse grid in each dimension. If None
        There is no limit

    induced_sampling : boolean
        True - use induced sampling
        False - sample from probability measure

    cond_tol : float
        The maximum allowable condition number of the regression problem.
        If induced_sampling is False and cond_tol < 0 then we do not sample
        until cond number is below cond_tol but rather simply add
        nnew_indices*abs(cond_tol) samples. That is we specify an
        over sampling factor


    Returns
    -------
    result : :class:`pyapprox.surrogates.approximate.ApproximateResult`
         Result object with the following attributes

    approx: :class:`pyapprox.surrogates.polychaos.gpc.PolynomialChaosExpansion`
        The PCE approximation
    """
    var_trans = AffineTransform(variables)

    pce = AdaptiveInducedPCE(
        var_trans.num_vars(), induced_sampling=induced_sampling,
        cond_tol=cond_tol, fit_opts=fit_opts)

    __setup_adaptive_pce(pce, verbose, fun, var_trans, growth_rules,
                         refinement_indicator, tol, max_nsamples, callback,
                         max_level_1d)

    pce.build(callback)
    return ApproximateResult({'approx': pce})


def adaptive_approximate_polynomial_chaos_leja(
        fun, variables,
        callback=None,
        refinement_indicator=variance_pce_refinement_indicator,
        growth_rules=None, max_nsamples=100, tol=0, verbose=0,
        max_level_1d=None,
        ncandidate_samples=1e4, generate_candidate_samples=None):
    r"""
    Compute an adaptive Polynomial Chaos Expansion of a function based upon
    multivariate Leja sequences.

    Parameters
    ----------
    fun : callable
        The function to be minimized

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,nqoi)

    variables : IndependentMarginalsVariable
        A set of independent univariate random variables

    callback : callable
        Function called after each iteration with the signature

        ``callback(approx_k)``

        where approx_k is the current approximation object.

    refinement_indicator : callable
        A function that retuns an estimate of the error of a sparse grid
        subspace with signature

        ``refinement_indicator(subspace_index,nnew_subspace_samples,sparse_grid) -> float, float``

        where ``subspace_index`` is 1D np.ndarray of size (nvars),
        ``nnew_subspace_samples`` is an integer specifying the number

        of new samples that will be added to the sparse grid by adding the
        subspace specified by subspace_index and ``sparse_grid`` is the current
        :class:`pyapprox.adaptive_sparse_grid.CombinationSparseGrid` object.
        The two outputs are, respectively, the indicator used to control
        refinement of the sparse grid and the change in error from adding the
        current subspace. The indicator is typically but now always dependent
        on the error.

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

    max_level_1d : np.ndarray (nvars)
        The maximum level of the sparse grid in each dimension. If None
        There is no limit

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
    result : :class:`pyapprox.surrogates.approximate.ApproximateResult`
         Result object with the following attributes

    approx: :class:`pyapprox.surrogates.polychaos.gpc.PolynomialChaosExpansion`
        The PCE approximation
    """
    var_trans = AffineTransform(variables)

    pce = __initialize_leja_pce(
        variables, generate_candidate_samples, ncandidate_samples)

    __setup_adaptive_pce(pce, verbose, fun, var_trans, growth_rules,
                         refinement_indicator, tol, max_nsamples, callback,
                         max_level_1d)

    pce.build(callback)
    return ApproximateResult({'approx': pce})


def adaptive_approximate_gaussian_process(
        fun, variable, callback=None,
        max_nsamples=100, verbose=0, ncandidate_samples=1e4,
        checkpoints=None, nu=np.inf, n_restarts_optimizer=1,
        normalize_y=False, alpha=0,
        noise_level=None, noise_level_bounds='fixed',
        kernel_variance=None,
        kernel_variance_bounds='fixed',
        length_scale=1,
        length_scale_bounds=(1e-2, 10),
        generate_candidate_samples=None,
        weight_function=None,
        normalize_inputs=False):
    r"""
    Adaptively construct a Gaussian process approximation of a function using
    weighted-pivoted-Cholesky sampling and the Matern kernel

    .. math::

       k(z_i, z_j) =  \frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(
       \frac{\sqrt{2\nu}}{l} \lVert z_i - z_j \rVert_2\Bigg)^\nu K_\nu\Bigg(
       \frac{\sqrt{2\nu}}{l} \lVert z_i - z_j \rVert_2\Bigg)

    where :math:`\lVert \cdot \rVert_2` is the Euclidean distance,
    :math:`\Gamma(\cdot)` is the gamma function, :math:`K_\nu(\cdot)` is the
    modified Bessel function.

    Starting from an initial guess, the algorithm learns the kernel length
    scale as more training data is collected.

    Parameters
    ----------
    fun : callable
        The function to be approximated

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,nqoi)

    variable : IndependentMarginalsVariable
        A set of independent univariate random variables

    callback : callable
        Function called after each iteration with the signature

        ``callback(approx_k)``

        where approx_k is the current approximation object.

    nu : string
        The parameter :math:`\nu` of the Matern kernel. When :math:`\nu\to\inf`
        the Matern kernel is equivalent to the squared-exponential kernel.

    checkpoints : iterable
        The set of points at which the length scale of the kernel will be
        recomputed and new training data obtained. If None then
        ``checkpoints = np.linspace(10, max_nsamples, 10).astype(int)``

    max_nsamples : float
        The maximum number of evaluations of fun. If fun has configure
        variables.

    ncandidate_samples : integer
        The number of candidate samples used to select the training samples
        The final training samples will be a subset of these samples.

    alpha : float
        Nugget added to diagonal of the covariance kernel evaluated at
        the training data. Used to improve numerical conditionining. This
        parameter is different to noise_level which applies to both training
        and test data

    normalize_y : bool
        True - normalize the training values to have zero mean and unit
               variance

    length_scale : float
        The initial length scale used to generate the first batch of training
        samples

    length_scale_bounds : tuple (2)
        The lower and upper bound on length_scale used in optimization of
        the Gaussian process hyper-parameters

    noise_level : float
        The noise_level used when training the GP

    noise_level_bounds : tuple (2)
        The lower and upper bound on noise_level used in optimization of
        the Gaussian process hyper-parameters

    kernel_variance : float
        The kernel_variance used when training the GP

    noise_level_bounds : tuple (2)
        The lower and upper bound on kernel_variance used in optimization of
        the Gaussian process hyper-parameters

    n_restarts_optimizer : int
        The number of local optimizeation problems solved to find the
        GP hyper-parameters

    verbose : integer
        Controls the amount of information printed to screen

    generate_candidate_samples : callable
        A function that generates the candidate samples used to build the Leja
        sequence with signature

        ``generate_candidate_samples(ncandidate_samples) -> np.ndarray``

        The output is a 2D np.ndarray with size(nvars,ncandidate_samples)

    weight_function : callable
        Function used to precondition kernel with the signature

        ``weight_function(samples) -> np.ndarray (num_samples)``

        where samples is a np.ndarray (num_vars,num_samples)

    Returns
    -------
    result : :class:`pyapprox.surrogates.approximate.ApproximateResult`
         Result object with the following attributes

    approx : :class:`pyapprox.surrogates.gaussianprocess.gaussian_process.AdaptiveGaussianProcess`
        The Gaussian process
    """
    assert max_nsamples <= ncandidate_samples

    nvars = variable.num_vars()

    if normalize_inputs:
        var_trans = AffineTransform(variable)
    else:
        var_trans = None

    if normalize_y:
        raise ValueError("normalize_y=True not currently supported")

    kernel = __setup_gaussian_process_kernel(
        nvars, length_scale, length_scale_bounds,
        kernel_variance, kernel_variance_bounds,
        noise_level, noise_level_bounds, nu)

    sampler = CholeskySampler(
        nvars, ncandidate_samples, variable,
        gen_candidate_samples=generate_candidate_samples,
        var_trans=var_trans)
    sampler_kernel = copy.deepcopy(kernel)
    sampler.set_kernel(sampler_kernel)
    sampler.set_weight_function(weight_function)

    gp = AdaptiveGaussianProcess(
        kernel, n_restarts_optimizer=n_restarts_optimizer, alpha=alpha,
        normalize_y=normalize_y)
    gp.setup(fun, sampler)
    if var_trans is not None:
        gp.set_variable_transformation(var_trans)

    if checkpoints is None:
        nsteps = 10
        if max_nsamples-10 < nsteps:
            nsteps = max_nsamples-10
        checkpoints = np.linspace(10, max_nsamples, nsteps).astype(int)
    checkpoints = np.unique(checkpoints.astype(int))
    assert checkpoints[-1] <= max_nsamples

    nsteps = len(checkpoints)
    for kk in range(nsteps):
        chol_flag = gp.refine(checkpoints[kk])
        gp.sampler.set_kernel(copy.deepcopy(gp.kernel_))
        if callback is not None:
            callback(gp)
        if chol_flag != 0:
            msg = "Cannot add additional samples. "
            msg += "Kernel is now ill conditioned. "
            msg += 'If more samples are really required increase alpha or '
            msg += 'manually fix kernel_length to a smaller value'
            print('Exiting: ' + msg)
            # print(gp.kernel_)
            # print(np.linalg.norm(gp.sampler.candidate_samples))
            break
    return ApproximateResult({'approx': gp})


def compute_l2_error(f, g, variable, nsamples, rel=False):
    r"""
    Compute the :math:`\ell^2` error of the output of two functions f and g,
    i.e.

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

    variable : pya.IndependentMarginalsVariable
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

    validation_samples = generate_independent_random_samples(
        variable, nsamples)
    validation_vals = f(validation_samples)
    approx_vals = g(validation_samples)
    assert validation_vals.shape == approx_vals.shape
    error = np.linalg.norm(approx_vals-validation_vals, axis=0)
    if not rel:
        error /= np.sqrt(validation_samples.shape[1])
    else:
        error /= np.linalg.norm(validation_vals, axis=0)
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
    result : :class:`pyapprox.surrogates.approximate.ApproximateResult`
         Result object. For more details see

         - :func:`pyapprox.surrogates.approximate.adaptive_approximate_sparse_grid`

         - :func:`pyapprox.surrogates.approximate.adaptive_approximate_polynomial_chaos`

         - :func:`pyapprox.surrogates.approximate.adaptive_approximate_gaussian_process`
    """

    methods = {'sparse_grid': adaptive_approximate_sparse_grid,
               'polynomial_chaos': adaptive_approximate_polynomial_chaos,
               'gaussian_process': adaptive_approximate_gaussian_process}

    if type(variable) != IndependentMarginalsVariable:
        variable = IndependentMarginalsVariable(
            variable)

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
                                 variable=None, options=None, poly_opts=None):
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

        - 'hyperbolic_cross' see :func:`pyapprox.surrogates.approximate.cross_validate_pce_degree`
        - 'expanding_basis' see :func:`pyapprox.surrogates.approximate.expanding_basis_pce`
        - 'fixed' see :func:`pyapprox.surrogates.approximate.approximate_fixed_pce`

    variable : pya.IndependentMarginalsVariable
        Object containing information of the joint density of the inputs z.
        This is used to generate random samples from this join density

    verbosity : integer
        Controls the amount of information printed to screen

    poly_opts : dictionary
        Dictionary definining the custom configuration of the polynomial
        chaos expansion basis. See :func:`pyapprox.surrogates.polychaos.gpc.PolynomialChaosExpansion.configure`

    Returns
    -------
    result : :class:`pyapprox.surrogates.approximate.ApproximateResult`
         Result object. For more details see

         - :func:`pyapprox.surrogates.approximate.cross_validate_pce_degree`

         - :func:`pyapprox.surrogates.approximate.expanding_basis_pce`

         - :func:`pyapprox.surrogates.approximate.approximate_fixed_pce`
    """
    funcs = {'expanding_basis': expanding_basis_pce,
             'hyperbolic_cross': cross_validate_pce_degree,
             'fixed': approximate_fixed_pce}
    if variable is None:
        msg = 'pce requires that variable be defined'
        raise Exception(msg)
    if basis_type not in funcs:
        msg = f'Basis type {basis_type} not found.\n Available types are:\n'
        for key in funcs.keys():
            msg += f"\t{key}\n"
        raise Exception(msg)

    poly = PolynomialChaosExpansion()
    if poly_opts is None:
        var_trans = AffineTransform(variable)
        poly_opts = define_poly_options_from_variable_transformation(
            var_trans)
    poly.configure(poly_opts)

    if options is None:
        options = {}

    res = funcs[basis_type](poly, train_samples, train_vals, **options)
    return res


def approximate_neural_network(train_samples, train_vals,
                               network_opts, verbosity=0,
                               variable=None, optimizer_opts=None, x0=None):
    print(network_opts)
    network = NeuralNetwork(network_opts)
    if x0 is None:
        nrestarts = 10
        x0 = np.random.uniform(-1, 2, (network.nparams, nrestarts))
    if optimizer_opts is None:
        optimizer_opts = {"method": "L-BFGS-B",
                          "options": {"maxiter": 1000}}
    network.fit(train_samples, train_vals, x0, verbose=verbosity,
                opts=optimizer_opts)
    return ApproximateResult({'approx': network})


def approximate(train_samples, train_vals, method, options=None):
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
    result : :class:`pyapprox.surrogates.approximate.ApproximateResult`
    """

    methods = {'polynomial_chaos': approximate_polynomial_chaos,
               'gaussian_process': approximate_gaussian_process,
               'neural_network': approximate_neural_network}
    # 'tensor-train':approximate_tensor_train,

    if method not in methods:
        msg = f'Method {method} not found.\n Available methods are:\n'
        for key in methods.keys():
            msg += f"\t{key}\n"
        raise Exception(msg)

    if options is None:
        options = {}
    return methods[method](train_samples, train_vals, **options)


class LinearLeastSquares(LinearModel):
    def __init__(self, alpha=0.):
        self.alpha = alpha

    def fit(self, X, y):
        gram_mat = X.T.dot(X)
        gram_mat += self.alpha*np.eye(gram_mat.shape[0])
        self.coef_ = np.linalg.lstsq(
            gram_mat, X.T.dot(y), rcond=None)[0]
        # Do not current support fit_intercept = True
        self.intercept_ = 0
        return self


class LinearLeastSquaresCV(LinearModel):
    """
    Parameters
    ----------
    - None, to use the default 5-fold cross-validation
    - integer to specify the number of folds
    """
    def __init__(self, alphas=[0.], cv=None, random_folds=True):
        if cv is None:
            # sklearn RidgeCV can only be applied with leave one out cross
            # validation. Use this as the default here
            cv = 1
        self.cv = cv
        self.alphas = alphas
        self.random_folds = random_folds

    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:, None]
        assert y.shape[1] == 1
        if self.cv != y.shape[0]:
            fold_sample_indices = get_random_k_fold_sample_indices(
                X.shape[0], self.cv, self.random_folds)
            results = [leave_many_out_lsq_cross_validation(
                X, y, fold_sample_indices, alpha) for alpha in self.alphas]
        else:
            results = [leave_one_out_lsq_cross_validation(
                X, y, alpha) for alpha in self.alphas]
        cv_scores = [r[1] for r in results]
        ii_best_alpha = np.argmin(cv_scores)

        self.cv_score_ = cv_scores[ii_best_alpha][0]
        self.alpha_ = self.alphas[ii_best_alpha]
        self.coef_ = results[ii_best_alpha][2]
        # Do not current support fit_intercept = True
        self.intercept_ = 0
        return self


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=LinAlgWarning)
@ignore_warnings(category=RuntimeWarning)
def fit_linear_model(basis_matrix, train_vals, solver_type, **kwargs):
    # verbose=1 will display lars path on entire data setUp
    # verbose>1 will also show this plus paths on each cross validation set
    solvers = {'lasso': [LassoLarsCV, LassoLars],
               'lasso_grad': [LassoCV, Lasso],
               'lars': [LarsCV, Lars],
               'omp': [OrthogonalMatchingPursuitCV, OrthogonalMatchingPursuit],
               'lstsq': [LinearLeastSquaresCV, LinearLeastSquares]}

    if solver_type not in solvers:
        msg = f'Solver type {solver_type} not supported\n'
        msg += 'Supported solvers are:\n'
        for key in solvers.keys():
            msg += f'\t{key}\n'
        raise Exception(msg)

    if solver_type == 'lars':
        msg = 'Currently lars does not exit when alpha starts to grow '
        msg += 'this causes problems with cross validation. The lasso variant '
        msg += 'lars does work because this exit condition is implemented'
        raise Exception(msg)

    assert train_vals.ndim == 2
    assert train_vals.shape[1] == 1

    # The following comment and two conditional statements are only true
    # for lars which I have switched off.

    # cv interpolates each residual onto a common set of alphas
    # This is problematic if the alpha path is not monotonically decreasing
    # For some problems alpha will increase for last few sample sizes. This
    # messes up interpolation and causes the best_alpha to be estimated
    # very poorly in some cases. I belive all_alphas = np.unique(all_alphas)
    # is the culprit. To avoid the aforementioned issue set max_iter to
    # ntrain_samples//2 This is typically stops the algorithm after
    # what would have been chosen as the best_alpha but before
    # alphas start increasing. Ideally sklearn should exit when
    # alphas increase.
    # if solver_type != 'lstsq' and 'max_iter' not in kwargs:
    #    kwargs['max_iter'] = basis_matrix.shape[0]//2

    # if 'max_iter' in kwargs and kwargs['max_iter'] > basis_matrix.shape[0]//2:
    #     msg = "Warning: max_iter is set large this can effect not just "
    #     msg += "Computational cost but also final accuracy"
    #     print(msg)

    if solver_type == 'omp' and 'max_iter' in kwargs:
        # unlike lasso/lars max iter is not allowed to be greater than
        # number of columns (features/bases)
        kwargs['max_iter'] = min(kwargs['max_iter'], basis_matrix.shape[1])
        # for omp to work sklean must be patched to store mse_path_.
        # Add the line         self.mse_path_ = mse_folds.T
        # as the last line (913) before return self in the function fit of
        # OrthogonalMatchingPursuitCV in
        # site-packages/sklearn/linear_model/_omp.py

    if 'cv' not in kwargs or kwargs['cv'] is False:
        solver_idx = 1
    else:
        solver_idx = 0

    if kwargs.get('precondition', False):
        weights = np.sqrt(christoffel_weights(basis_matrix))[:, None]
        basis_matrix = basis_matrix.copy()*weights
        train_vals = train_vals.copy()*weights

    if 'precondition' in kwargs:
        del kwargs['precondition']

    fit = solvers[solver_type][solver_idx](**kwargs).fit
    res = fit(basis_matrix, train_vals[:, 0])
    coef = res.coef_
    if coef.ndim == 1:
        coef = coef[:, np.newaxis]
        # some methods allow fit_intercept to be set. If True this method
        # extracts of mean of data before computing coefficients.
        # res.predict then makes predictions as X.dot(coef_) + res.intercept_
        # I assume first coefficient is constant basis and want to be able
        # to simply return X.dot(coef_) (e.g. as done be Polynomial Chaos
        # Expansion). Thus add res.intercept_ to first coefficient, i.e.
    coef[0] += res.intercept_
    if 'cv' in kwargs:
        cv_score = extract_cross_validation_score(res)
        best_regularization_param = extract_best_regularization_parameters(res)
    else:
        cv_score = None
        best_regularization_param = None
    return coef, cv_score, best_regularization_param


def extract_best_regularization_parameters(res):
    if (type(res) == LassoLarsCV or type(res) == LinearLeastSquaresCV):
        return res.alpha_
    elif type(res) == LarsCV:
        assert len(res.n_iter_) == 1
        # The Lars (not LarsCV) model takes max_iters as regularization
        # parameter so return it here as well as the alpha. Sklearn
        # has an inconsistency alpha is used to choose best cv score but Lars
        # uses max_iters to stop algorithm early.
        return (res.alpha_, res.n_iter_[0])
    elif type(res) == OrthogonalMatchingPursuitCV:
        return res.n_nonzero_coefs_
    else:
        raise Exception()


def extract_cross_validation_score(linear_model):
    if hasattr(linear_model, 'cv_score_'):
        return linear_model.cv_score_
    elif hasattr(linear_model, 'mse_path_'):
        return np.sqrt(linear_model.mse_path_.mean(axis=-1).min())
    else:
        raise Exception('attribute mse_path_ not found')


def cross_validate_pce_degree(
        pce, train_samples, train_vals, min_degree=1, max_degree=3,
        hcross_strength=1, solver_type='lasso', verbose=0,
        linear_solver_options={'cv': 10}):
    r"""
    Use cross validation to find the polynomial degree which best fits the
    data.
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

    max_degree : integer
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

        - 'lasso'
        - 'lars'
        - 'lasso_grad'
        - 'omp'
        - 'lstsq'

    verbose : integer
        Controls the amount of information printed to screen

    Returns
    -------
    result : :class:`pyapprox.surrogates.approximate.ApproximateResult`
         Result object with the following attributes

    approx: :class:`pyapprox.surrogates.polychaos.gpc.PolynomialChaosExpansion`
        The PCE approximation

    scores : np.ndarray (nqoi)
        The best cross validation score for each QoI

    degrees : np.ndarray (nqoi)
        The best degree for each QoI

    reg_params : np.ndarray (nqoi)
        The best regularization parameters for each QoI chosen by cross
        validation.
    """
    coefs = []
    scores = []
    indices = []
    degrees = []
    reg_params = []
    indices_dict = dict()
    unique_indices = []
    nqoi = train_vals.shape[1]
    for ii in range(nqoi):
        if verbose > 1:
            print(f'Approximating QoI: {ii}')
        pce_ii, score_ii, degree_ii, reg_param_ii = _cross_validate_pce_degree(
            pce, train_samples, train_vals[:, ii:ii+1], min_degree,
            max_degree, hcross_strength, linear_solver_options,
            solver_type, verbose)
        coefs.append(pce_ii.get_coefficients())
        scores.append(score_ii)
        indices.append(pce_ii.get_indices())
        degrees.append(degree_ii)
        reg_params.append(reg_param_ii)
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
    return ApproximateResult({'approx': pce, 'scores': np.array(scores),
                              'degrees': np.array(degrees),
                              'reg_params': reg_params})


def _cross_validate_pce_degree(
        pce, train_samples, train_vals, min_degree=1, max_degree=3,
        hcross_strength=1, linear_solver_options={'cv': 10},
        solver_type='lasso', verbose=0):
    assert train_vals.shape[1] == 1
    if min_degree is None:
        min_degree = 2
    if max_degree is None:
        max_degree = np.iinfo(int).max-1

    best_coef = None
    best_cv_score = np.finfo(np.double).max
    best_degree = min_degree
    prev_num_terms = 0
    if verbose > 0:
        print("{:<8} {:<10} {:<18}".format('degree', 'num_terms', 'cv score',))

    rng_state = np.random.get_state()
    for degree in range(min_degree, max_degree+1):
        indices = compute_hyperbolic_indices(
            pce.num_vars(), degree, hcross_strength)
        pce.set_indices(indices)
        if ((pce.num_terms() > 100000) and
                (100000-prev_num_terms < pce.num_terms()-100000)):
            break

        basis_matrix = pce.basis_matrix(train_samples)

        # use the same state (thus cross validation folds) for each degree
        np.random.set_state(rng_state)
        coef, cv_score, reg_param = fit_linear_model(
            basis_matrix, train_vals, solver_type, **linear_solver_options)
        np.random.set_state(rng_state)
        pce.set_coefficients(coef)
        if verbose > 0:
            print("{:<8} {:<10} {:<18} ".format(
                degree, pce.num_terms(), cv_score))
        if ((cv_score >= best_cv_score) and (degree-best_degree > 1)):
            break
        if (cv_score < best_cv_score):
            best_cv_score = cv_score
            best_coef = coef.copy()
            best_degree = degree
            best_reg_param = reg_param
        prev_num_terms = pce.num_terms()

    pce.set_indices(compute_hyperbolic_indices(
        pce.num_vars(), best_degree, hcross_strength))
    pce.set_coefficients(best_coef)
    if verbose > 0:
        print('best degree:', best_degree)
    return pce, best_cv_score, best_degree, best_reg_param


def restrict_basis(indices, coefficients, tol):
    II = np.where(np.absolute(coefficients) > tol)[0]
    restricted_indices = indices[:, II]
    degrees = indices.sum(axis=0)
    JJ = np.where(degrees == 0)[0]
    assert JJ.shape[0] == 1
    if JJ not in II:
        # always include zero degree polynomial in restricted_indices
        restricted_indices = np.concatenate(
            [indices[:JJ], restricted_indices])
    return restricted_indices


def expand_basis(indices):
    nvars, nindices = indices.shape
    indices_set = set()
    for ii in range(nindices):
        indices_set.add(hash_array(indices[:, ii]))

    new_indices = []
    for ii in range(nindices):
        index = indices[:, ii]
        for dd in range(nvars):
            forward_index = get_forward_neighbor(index, dd)
            key = hash_array(forward_index)
            if key not in indices_set:
                admissible = True
                active_vars = np.nonzero(forward_index)
                for kk in active_vars:
                    backward_index = get_backward_neighbor(forward_index, kk)
                    if hash_array(backward_index) not in indices_set:
                        admissible = False
                        break
                if admissible:
                    indices_set.add(key)
                    new_indices.append(forward_index)
    return np.asarray(new_indices).T


def expanding_basis_pce(pce, train_samples, train_vals, hcross_strength=1,
                        verbose=1, max_num_init_terms=None, max_num_terms=None,
                        solver_type='lasso',
                        linear_solver_options={'cv': 10},
                        restriction_tol=np.finfo(float).eps*2,
                        max_num_expansion_steps_iter=1,
                        max_iters=20,
                        max_num_step_increases=1):
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

        - 'lasso'
        - 'lars'
        - 'lasso_grad'
        - 'omp'
        - 'lstsq'

    verbose : integer
        Controls the amount of information printed to screen

    restriction_tol : float
        The tolerance used to prune inactive indices

    max_num_init_terms : integer
        The number of terms used to initialize the algorithm

    max_num_terms : integer
        The maximum number of terms allowed

    max_iters : integer
        The number of expansion/restriction iterations

    max_num_expansion_steps_iter : integer (1,2,3)
        The number of times a basis can expanded after
        the last restriction step

    max_num_expansion_steps_iter : integer
        The number of iterations error does not decrease before
        terminating

    Returns
    -------
    result : :class:`pyapprox.surrogates.approximate.ApproximateResult`
         Result object with the following attributes

    approx: :class:`pyapprox.surrogates.polychaos.gpc.PolynomialChaosExpansion`
        The PCE approximation

    scores : np.ndarray (nqoi)
        The best cross validation score for each QoI

    reg_params : np.ndarray (nqoi)
        The best regularization parameters for each QoI chosen by cross
        validation.

    References
    ----------
    .. [JESJCP2015] `J.D. Jakeman, M.S. Eldred, and K. Sargsyan. Enhancing l1-minimization estimates of polynomial chaos expansions using basis selection. Journal of Computational Physics, 289(0):18 – 34, 2015 <https://doi.org/10.1016/j.jcp.2015.02.025>`_
    """
    coefs = []
    scores = []
    indices = []
    reg_params = []
    indices_dict = dict()
    unique_indices = []
    nqoi = train_vals.shape[1]
    for ii in range(nqoi):
        if verbose > 1:
            print(f'Approximating QoI: {ii}')
        pce_ii, score_ii, reg_param_ii = _expanding_basis_pce(
            pce, train_samples, train_vals[:, ii:ii+1], hcross_strength,
            verbose, max_num_init_terms, max_num_terms, solver_type,
            linear_solver_options, restriction_tol,
            max_num_expansion_steps_iter, max_iters, max_num_step_increases)
        coefs.append(pce_ii.get_coefficients())
        scores.append(score_ii)
        indices.append(pce_ii.get_indices())
        reg_params.append(reg_param_ii)
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
    return ApproximateResult({'approx': pce, 'scores': np.array(scores),
                              'reg_params': reg_params})


def _expanding_basis_pce(pce, train_samples, train_vals, hcross_strength=1,
                         verbose=1, max_num_init_terms=None,
                         max_num_terms=None,
                         solver_type='lasso',
                         linear_solver_options={'cv': 10},
                         restriction_tol=np.finfo(float).eps*2,
                         max_num_expansion_steps_iter=3,
                         max_iters=20,
                         max_num_step_increases=1):
    assert train_vals.shape[1] == 1
    num_vars = pce.num_vars()

    if max_num_init_terms is None:
        max_num_init_terms = train_vals.shape[0]
    if max_num_terms is None:
        max_num_terms = 10*train_vals.shape[0]

    degree = 2
    prev_num_terms = 0
    while True:
        indices = compute_hyperbolic_indices(num_vars, degree, hcross_strength)
        num_terms = indices.shape[1]
        if (num_terms > max_num_init_terms):
            break
        degree += 1
        prev_num_terms = num_terms

    if (abs(num_terms - max_num_init_terms) >
            abs(prev_num_terms - max_num_init_terms)):
        degree -= 1
    pce.set_indices(
        compute_hyperbolic_indices(num_vars, degree, hcross_strength))

    if verbose > 0:
        msg = f'Initializing basis with hyperbolic cross of degree {degree} '
        msg += f'and strength {hcross_strength} with {pce.num_terms()} terms'
        print(msg)

    rng_state = np.random.get_state()
    basis_matrix = pce.basis_matrix(train_samples)
    best_coef, best_cv_score, best_reg_param = fit_linear_model(
        basis_matrix, train_vals, solver_type, **linear_solver_options)
    np.random.set_state(rng_state)
    pce.set_coefficients(best_coef)
    best_indices = pce.get_indices()
    best_cv_score_iter = best_cv_score
    best_indices_iter = best_indices.copy()
    best_coef_iter = best_coef.copy()
    best_reg_param_iter = best_reg_param
    if verbose > 0:
        print("{:<10} {:<10} {:<18}".format('nterms', 'nnz terms', 'cv score'))
        print("{:<10} {:<10} {:<18}".format(
            pce.num_terms(), np.count_nonzero(pce.coefficients),
            best_cv_score))

    it = 0
    best_it = 0
    while True:
        current_max_num_expansion_steps_iter = 1
        best_cv_score_iter = best_cv_score
        indices_iter = pce.indices.copy()
        coef_iter = pce.coefficients.copy()
        while True:
            # -------------- #
            #  Expand basis  #
            # -------------- #
            num_expansion_steps_iter = 0
            indices = restrict_basis(
                # pce.indices, pce.coefficients, restriction_tol)
                indices_iter, coef_iter, restriction_tol)
            msg = f'Expanding {indices.shape[1]} restricted from '
            msg += f'{pce.indices.shape[1]} terms'
            while (num_expansion_steps_iter <
                   current_max_num_expansion_steps_iter):
                new_indices = expand_basis(indices)
                indices = np.hstack([indices, new_indices])
                num_expansion_steps_iter += 1
            pce.set_indices(indices)
            num_terms = pce.num_terms()
            msg += f' New number of terms {pce.indices.shape[1]}'
            print(msg)

            # -----------------#
            # Compute solution #
            # -----------------#
            basis_matrix = pce.basis_matrix(train_samples)
            np.random.set_state(rng_state)
            coef, cv_score, reg_param = fit_linear_model(
                basis_matrix, train_vals, solver_type, **linear_solver_options)
            np.random.set_state(rng_state)
            pce.set_coefficients(coef)

            if verbose > 0:
                print("{:<10} {:<10} {:<18}".format(
                    pce.num_terms(), np.count_nonzero(pce.coefficients),
                    cv_score))

            if (cv_score < best_cv_score_iter):
                best_cv_score_iter = cv_score
                best_indices_iter = pce.indices.copy()
                best_coef_iter = pce.coefficients.copy()
                best_reg_param_iter = reg_param

            if (num_terms >= max_num_terms):
                if verbose > 0:
                    print(f'Max number of terms {max_num_terms} reached')
                break

            if (current_max_num_expansion_steps_iter >=
                    max_num_expansion_steps_iter):
                if verbose > 0:
                    msg = 'Max number of inner expansion steps '
                    msg += f'({max_num_expansion_steps_iter}) reached'
                    print(msg)
                break
            current_max_num_expansion_steps_iter += 1

        it += 1
        pce.set_indices(best_indices_iter)
        pce.set_coefficients(best_coef_iter)

        if (best_cv_score_iter < best_cv_score):
            best_cv_score = best_cv_score_iter
            best_coef = best_coef_iter.copy()
            best_indices = best_indices_iter.copy()
            best_reg_param = best_reg_param_iter
            best_it = it

        elif (it - best_it >= max_num_step_increases):
            if verbose > 0:
                msg = 'Terminating: error did not decrease'
                msg += f' in last {max_num_step_increases} iterations'
                msg += f'best error: {best_cv_score}'
                print(msg)
            break

        if it >= max_iters:
            if verbose > 0:
                msg = 'Terminating: max iterations reached'
                print(msg)
            break

    nindices = best_indices.shape[1]
    II = np.nonzero(best_coef[:, 0])[0]
    pce.set_indices(best_indices[:, II])
    pce.set_coefficients(best_coef[II])
    if verbose > 0:
        msg = f'Final basis has {pce.num_terms()} terms selected from '
        msg += f'{nindices} using {train_samples.shape[1]} samples'
        print(msg)

    return pce, best_cv_score, best_reg_param


def approximate_fixed_pce(pce, train_samples, train_vals, indices,
                          verbose=1, solver_type='lasso',
                          linear_solver_options={}):
    r"""
    Estimate the coefficients of a polynomial chaos using regression methods
    and pre-specified (fixed) basis and regularization parameters

    Parameters
    ----------
    train_samples : np.ndarray (nvars, nsamples)
        The inputs of the function used to train the approximation

    train_vals : np.ndarray (nvars, nqoi)
        The values of the function at ``train_samples``

    indices : np.ndarray (nvars, nindices)
        The multivariate indices representing each basis in the expansion.

    solver_type : string
        The type of regression used to train the polynomial

        - 'lasso'
        - 'lars'
        - 'lasso_grad'
        - 'omp'
        - 'lstsq'

    verbose : integer
        Controls the amount of information printed to screen

    Returns
    -------
    result : :class:`pyapprox.surrogates.approximate.ApproximateResult`
         Result object with the following attributes

    approx: :class:`pyapprox.surrogates.polychaos.gpc.PolynomialChaosExpansion`
        The PCE approximation

    reg_params : np.ndarray (nqoi)
        The regularization parameters for each QoI.
    """
    nqoi = train_vals.shape[1]
    coefs = []
    if type(linear_solver_options) == dict:
        linear_solver_options = [linear_solver_options]*nqoi
    if type(indices) == np.ndarray:
        indices = [indices.copy() for ii in range(nqoi)]
    unique_indices = []
    indices_dict = dict()
    for ii in range(nqoi):
        pce.set_indices(indices[ii])
        basis_matrix = pce.basis_matrix(train_samples)
        coef_ii, _, reg_param_ii = fit_linear_model(
            basis_matrix, train_vals[:, ii:ii+1], solver_type,
            **linear_solver_options[ii])
        coefs.append(coef_ii)
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
    return ApproximateResult({'approx': pce})


def __setup_gaussian_process_kernel(nvars, length_scale, length_scale_bounds,
                                    kernel_variance, kernel_variance_bounds,
                                    noise_level, noise_level_bounds, nu):
    if np.isscalar(length_scale):
        length_scale = np.array([length_scale]*nvars)
    assert length_scale.shape[0] == nvars
    kernel = Matern(length_scale, length_scale_bounds=length_scale_bounds,
                    nu=nu)
    # optimize variance
    if kernel_variance is not None:
        kernel = ConstantKernel(
            constant_value=kernel_variance,
            constant_value_bounds=kernel_variance_bounds)*kernel
    # optimize gp noise
    if noise_level is not None:
        kernel += WhiteKernel(
            noise_level, noise_level_bounds=noise_level_bounds)
    # Note noise_level is different to alpha
    # noise_kernel applies nugget to both training and test data
    # alpha only applies it to training data
    return kernel


def approximate_gaussian_process(train_samples, train_vals, nu=np.inf,
                                 n_restarts_optimizer=5, verbose=0,
                                 normalize_y=False, alpha=0,
                                 noise_level=None, noise_level_bounds='fixed',
                                 kernel_variance=None,
                                 kernel_variance_bounds='fixed',
                                 var_trans=None,
                                 length_scale=1,
                                 length_scale_bounds=(1e-2, 10)):
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
    train_samples : np.ndarray (nvars, nsamples)
        The inputs of the function used to train the approximation

    train_vals : np.ndarray (nvars, 1)
        The values of the function at ``train_samples``

    nu : string
        The parameter :math:`\nu` of the Matern kernel. When :math:`\nu\to\inf`
        the Matern kernel is equivalent to the squared-exponential kernel.

    alpha : float
        Nugget added to diagonal of the covariance kernel evaluated at
        the training data. Used to improve numerical conditionining. This
        parameter is different to noise_level which applies to both training
        and test data

    normalize_y : bool
        True - normalize the training values to have zero mean and unit
        variance

    length_scale : float
        The initial length scale used to generate the first batch of training
        samples

    length_scale_bounds : tuple (2)
        The lower and upper bound on length_scale used in optimization of
        the Gaussian process hyper-parameters

    noise_level : float
        The noise_level used when training the GP

    noise_level_bounds : tuple (2)
        The lower and upper bound on noise_level used in optimization of
        the Gaussian process hyper-parameters

    kernel_variance : float
        The kernel_variance used when training the GP

    noise_level_bounds : tuple (2)
        The lower and upper bound on kernel_variance used in optimization of
        the Gaussian process hyper-parameters

    n_restarts_optimizer : int
        The number of local optimizeation problems solved to find the
        GP hyper-parameters

    verbose : integer
        Controls the amount of information printed to screen

    Returns
    -------
    result : :class:`pyapprox.surrogates.approximate.ApproximateResult`
         Result object with the following attributes

    approx : :class:`pyapprox.surrogates.gaussianprocess.gaussian_process.GaussianProcess`
        The Gaussian process
    """
    nvars = train_samples.shape[0]
    kernel = __setup_gaussian_process_kernel(
        nvars, length_scale, length_scale_bounds,
        kernel_variance, kernel_variance_bounds,
        noise_level, noise_level_bounds, nu)

    gp = GaussianProcess(kernel, n_restarts_optimizer=n_restarts_optimizer,
                         normalize_y=normalize_y, alpha=alpha)

    if var_trans is not None:
        gp.set_variable_transformation(var_trans)
    gp.fit(train_samples, train_vals)
    return ApproximateResult({'approx': gp})


from pyapprox.util.utilities import get_random_k_fold_sample_indices, \
    leave_many_out_lsq_cross_validation, leave_one_out_lsq_cross_validation
def cross_validate_approximation(
        train_samples, train_vals, options, nfolds, method, random_folds=True):
    ntrain_samples = train_samples.shape[1]
    if random_folds != 'sklearn':
        fold_sample_indices = get_random_k_fold_sample_indices(
            ntrain_samples, nfolds, random_folds)
    else:
        from sklearn.model_selection._split import KFold
        sklearn_cv = KFold(nfolds)
        # indices = np.arange(train_samples.shape[1])
        fold_sample_indices = [
            te for tr, te in sklearn_cv.split(train_vals, train_vals)]

    approx_list = []
    residues_list = []
    cv_score = 0
    for kk in range(len(fold_sample_indices)):
        K = np.ones(ntrain_samples, dtype=bool)
        K[fold_sample_indices[kk]] = False
        train_samples_kk = train_samples[:, K]
        train_vals_kk = train_vals[K, :]
        test_samples_kk = train_samples[:, fold_sample_indices[kk]]
        test_vals_kk = train_vals[fold_sample_indices[kk]]
        approx_kk = approximate(
            train_samples_kk, train_vals_kk, method, options).approx
        residues = approx_kk(test_samples_kk) - test_vals_kk
        approx_list.append(approx_kk)
        residues_list.append(residues)
        cv_score += np.sum(residues**2, axis=0)
    cv_score = np.sqrt(cv_score/ntrain_samples)
    return approx_list, residues_list, cv_score


def quadratic_oversampling_ratio(nindices):
    return nindices**2


def increment_samples_using_oversampling_ratio(
        train_samples, indices, oversampling_ratio, generate_samples):
    ndesired_samples = oversampling_ratio(indices.shape[1])
    nnew_samples = ndesired_samples-train_samples.shape[1]
    new_train_samples = generate_samples(nnew_samples)
    return new_train_samples


def increment_samples_using_condition_number(
        train_samples, indices, pce, generate_samples, sample_growth_factor,
        cond_tol, max_nsamples):
    pce.set_indices(indices)
    ndesired_samples = indices.shape[1]
    cond_num = np.inf
    ncurrent_samples = train_samples.shape[1]
    new_train_samples = np.zeros((pce.num_vars(), 0))
    if train_samples.shape[1] == 0:
        basis_matrix = np.zeros((0, indices.shape[1]))
    else:
        basis_matrix = pce.basis_matrix(train_samples)
    while True:
        ndesired_samples = min(
            int(ndesired_samples*sample_growth_factor), max_nsamples)
        if ndesired_samples < ncurrent_samples:
            ndesired_samples = int(ncurrent_samples*sample_growth_factor)
        assert ndesired_samples > ncurrent_samples
        # generate a new increment of samples
        nsample_incr = ndesired_samples-(
            ncurrent_samples+new_train_samples.shape[1])
        # print(ncurrent_samples, nsample_incr, train_samples.shape,
        #       ndesired_samples, indices.shape)
        samples_incr = generate_samples(nsample_incr)
        # add increment to total set of new samples
        new_train_samples = np.hstack((
            new_train_samples, samples_incr))
        # compute basis matrix for sample increment
        basis_matrix_incr = pce.basis_matrix(samples_incr)
        # compute condition number of entire sample set
        basis_matrix = np.vstack((basis_matrix, basis_matrix_incr))
        cond_num = np.linalg.cond(basis_matrix)
        if cond_num <= cond_tol:
            break
        if ncurrent_samples + new_train_samples.shape[1] >= max_nsamples:
            break
    return new_train_samples


def adaptive_approximate_polynomial_chaos_increment_degree(
        fun, variable, max_degree, max_nsamples=100, cond_tol=None,
        sample_growth_factor=2, verbose=0, hcross_strength=1,
        oversampling_ratio=quadratic_oversampling_ratio,
        solver_type='lasso', linear_solver_options={},
        callback=None):

    var_trans = AffineTransform(variable)
    pce = PolynomialChaosExpansion()
    pce_opts = define_poly_options_from_variable_transformation(var_trans)
    pce.configure(pce_opts)

    if cond_tol is not None and oversampling_ratio is not None:
        raise ValueError("cond_tol or over_sampling_ratio must be None")

    if cond_tol is None and oversampling_ratio is None:
        raise ValueError("cond_tol or over_sampling_ratio must be None")

    degree = 0
    generate_samples = partial(
        generate_independent_random_samples, pce.var_trans.variable)
    train_samples = np.zeros((pce.num_vars(), 0))
    while True:
        if degree > max_degree:
            break
        if train_samples.shape[1] > max_nsamples:
            break

        indices = compute_hyperbolic_indices(
            pce.num_vars(), degree, hcross_strength)
        if cond_tol is not None:
            new_train_samples = increment_samples_using_condition_number(
                train_samples, indices, pce, generate_samples,
                sample_growth_factor, cond_tol, max_nsamples)
        else:
            new_train_samples = increment_samples_using_oversampling_ratio(
                train_samples, indices, oversampling_ratio, generate_samples)
        new_train_values = fun(new_train_samples)
        if train_samples.shape[1] == 0:
            train_values = new_train_values
        else:
            train_values = np.vstack((train_values, new_train_values))
        train_samples = np.hstack((train_samples, new_train_samples))
        # Todo allow expanding basis as well
        result = approximate_fixed_pce(
            pce, train_samples, train_values, indices, verbose,
            solver_type, linear_solver_options)
        if callback is not None:
            callback(result.approx)
        # TODO: if requested add exit condition that checks cross validation
        # error. For now if want to print cross validation error
        # implement that inside a callback
        # if nfolds > 0:
        #     method_opts = {"basis_type": "fixed", "verbose": verbose}
        #     method = approximate_polynomial_chaos()
        #     cv_score = cross_validate_approximation(
        #         train_samples, train_values, method_options, nfolds, method)

        degree += 1
    result = ApproximateResult(
        {'approx': result.approx, 'train_samples': train_samples,
         'train_values': train_values})
    return result
