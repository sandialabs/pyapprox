from functools import partial

import numpy as np
from scipy import stats
from scipy.optimize import OptimizeResult

from pyapprox import PYA_DEV_AVAILABLE
from pyapprox.benchmarks.sensitivity_benchmarks import (
    get_sobol_g_function_statistics, get_ishigami_funciton_statistics,
    oakley_function, oakley_function_statistics, sobol_g_function,
    ishigami_function, ishigami_function_jacobian, ishigami_function_hessian
    )
from pyapprox.benchmarks.surrogate_benchmarks import (
    rosenbrock_function,
    rosenbrock_function_jacobian, rosenbrock_function_hessian_prod,
    rosenbrock_function_mean, cantilever_beam_constraints_jacobian,
    cantilever_beam_constraints, cantilever_beam_objective,
    cantilever_beam_objective_grad, define_beam_random_variables,
    define_piston_random_variables, piston_function,
    define_wing_weight_random_variables, wing_weight_function,
    wing_weight_gradient, define_chemical_reaction_random_variables,
    ChemicalReactionModel, define_random_oscillator_random_variables,
    RandomOscillator, piston_function_gradient, CoupledSprings,
    define_coupled_springs_random_variables, HastingsEcology,
    define_nondim_hastings_ecology_random_variables, NobileBenchmarkFunctions,
    SpectralPDEMultiIndexWrapper, ParameterizedNonlinearModel
    )
from pyapprox.benchmarks.genz import GenzFunction
from pyapprox.benchmarks.multifidelity_benchmarks import (
    PolynomialModelEnsemble, TunableModelEnsemble, ShortColumnModelEnsemble
)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.variables.transforms import (
    ConfigureVariableTransformation
)
from pyapprox.interface.wrappers import (
    TimerModel, PoolModel, WorkTrackingModel
)
from pyapprox.benchmarks.pde_benchmarks import (
    _setup_inverse_advection_diffusion_benchmark,
    _setup_multi_index_advection_diffusion_benchmark
)


class Benchmark(OptimizeResult):
    """
    Contains functions and results needed to implement known
    benchmarks.

    A benchmark can be created with any attribute.
    Only fun and variable are required. Below are these two required attributes
    and other optional attributes used in different PyApprox Benchmarks

    Attributes
    ----------
    fun : callable
        The function being analyzed

    variable : :py:class:`pyapprox.variables.JointVariable`
        Class containing information about each of the nvars inputs to fun

    jac : callable
        The jacobian of fun. (optional)

    hess : callable
        The Hessian of fun. (optional)

    hessp : callable
        Function implementing the hessian of fun multiplied by a vector.
        (optional)

    mean: np.ndarray (nvars)
        The mean of the function with respect to the PDF of var

    variance: np.ndarray (nvars)
        The variance of the function with respect to the PDF of var

    main_effects : np.ndarray (nvars)
        The variance based main effect sensitivity indices

    total_effects : np.ndarray (nvars)
        The variance based total effect sensitivity indices

    sobol_indices : np.ndarray
        The variance based Sobol sensitivity indices

    Notes
    -----
    Use the `keys()` method to see a list of the available
    attributes for a specific benchmark
    """

def setup_sobol_g_function(nvars):
    r"""
    Setup the Sobol-G function benchmark

    .. math:: f(z) = \prod_{i=1}^d\frac{\lvert 4z_i-2\rvert+a_i}{1+a_i}, \quad a_i=\frac{i-2}{2}

    using

    >>> from pyapprox.benchmarks.benchmarks import setup_benchmark
    >>> benchmark=setup_benchmark('sobol_g',nvars=2)
    >>> print(benchmark.keys())
    dict_keys(['fun', 'mean', 'variance', 'main_effects', 'total_effects', 'variable'])

    Parameters
    ----------
    nvars : integer
        The number of variables of the Sobol-G function

    Returns
    -------
    benchmark : :py:class:`pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    fun : callable
        The function being analyzed

    variable : :py:class:`pyapprox.variables.JointVariable`
        Class containing information about each of the nvars inputs to fun

    mean: np.ndarray (nvars)
        The mean of the function with respect to the PDF of var

    variance: np.ndarray (nvars)
        The variance of the function with respect to the PDF of var

    main_effects : np.ndarray (nvars)
        The variance based main effect sensitivity indices

    total_effects : np.ndarray (nvars)
        The variance based total effect sensitivity indices

    References
    ----------
    .. [Saltelli1995] `Saltelli, A., & Sobol, I. M. About the use of rank transformation in sensitivity analysis of model output. Reliability Engineering & System Safety, 50(3), 225-239, 1995. <https://doi.org/10.1016/0951-8320(95)00099-2>`_
    """

    univariate_variables = [stats.uniform(0, 1)]*nvars
    variable = IndependentMarginalsVariable(univariate_variables)
    a_param = (np.arange(1, nvars+1)-2)/2
    mean, variance, main_effects, total_effects = \
        get_sobol_g_function_statistics(a_param)
    return Benchmark({'fun': partial(sobol_g_function, a_param),
                      'mean': mean, 'variance': variance,
                      'main_effects': main_effects,
                      'total_effects': total_effects, 'variable': variable})


def setup_ishigami_function(a, b):
    r"""
    Setup the Ishigami function benchmark

    .. math:: f(z) = \sin(z_1)+a\sin^2(z_2) + bz_3^4\sin(z_0)

    using

    >>> from pyapprox.benchmarks.benchmarks import setup_benchmark
    >>> benchmark=setup_benchmark('ishigami',a=7,b=0.1)
    >>> print(benchmark.keys())
    dict_keys(['fun', 'jac', 'hess', 'variable', 'mean', 'variance', 'main_effects', 'total_effects', 'sobol_indices'])

    Parameters
    ----------
    a : float
        The hyper-parameter a

    b : float
        The hyper-parameter b

    Returns
    -------
    benchmark : :py:class:`pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    fun : callable
        The function being analyzed

    variable : :py:class:`pyapprox.variables.JointVariable`
        Class containing information about each of the nvars inputs to fun

    jac : callable
        The jacobian of fun. (optional)

    hess : callable
        The Hessian of fun. (optional)

    hessp : callable
        Function implementing the hessian of fun multiplied by a vector.
        (optional)

    mean: np.ndarray (nvars)
        The mean of the function with respect to the PDF of var

    variance: np.ndarray (nvars)
        The variance of the function with respect to the PDF of var

    main_effects : np.ndarray (nvars)
        The variance based main effect sensitivity indices

    total_effects : np.ndarray (nvars)
        The variance based total effect sensitivity indices

    sobol_indices : np.ndarray (nsobol_indices)
        The variance based Sobol sensitivity indices

    sobol_interaction_indices : np.ndarray(nsobol_indices)
        The indices of the acitive variable dimensions involved in each
        sobol index

    References
    ----------
    .. [Ishigami1990] `T. Ishigami and T. Homma, "An importance quantification technique in uncertainty analysis for computer models," [1990] Proceedings. First International Symposium on Uncertainty Modeling and Analysis, College Park, MD, USA, 1990, pp. 398-403 <https://doi.org/10.1109/ISUMA.1990.151285>`_
    """
    univariate_variables = [stats.uniform(-np.pi, 2*np.pi)]*3
    variable = IndependentMarginalsVariable(univariate_variables)
    mean, variance, main_effects, total_effects, sobol_indices, \
        sobol_interaction_indices = get_ishigami_funciton_statistics(a, b)
    return Benchmark(
        {'fun': partial(ishigami_function, a=a, b=b),
         'jac': partial(ishigami_function_jacobian, a=a, b=b),
         'hess': partial(ishigami_function_hessian, a=a, b=b),
         'variable': variable, 'mean': mean, 'variance': variance,
         'main_effects': main_effects, 'total_effects': total_effects,
         'sobol_indices': sobol_indices,
         'sobol_interaction_indices': sobol_interaction_indices})


def setup_oakley_function():
    r"""
    Setup the Oakely function benchmark

    .. math:: f(z) = a_1^Tz + a_2^T\sin(z) + a_3^T\cos(z) + z^TMz

    where :math:`z` consists of 15 I.I.D. standard Normal variables and the data :math:`a_1,a_2,a_3` and :math:`M` are defined in the function :func:`pyapprox.benchmarks.sensitivity_benchmarks.get_oakley_function_data`.

    >>> from pyapprox.benchmarks.benchmarks import setup_benchmark
    >>> benchmark=setup_benchmark('oakley')
    >>> print(benchmark.keys())
    dict_keys(['fun', 'variable', 'mean', 'variance', 'main_effects'])

    Returns
    -------
    benchmark : :py:class:`pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    fun : callable
        The function being analyzed

    variable : :py:class:`pyapprox.variables.JointVariable`
        Class containing information about each of the nvars inputs to fun

    mean: np.ndarray (nvars)
        The mean of the function with respect to the PDF of var

    variance: np.ndarray (nvars)
        The variance of the function with respect to the PDF of var

    main_effects : np.ndarray (nvars)
        The variance based main effect sensitivity indices

    References
    ----------
    .. [OakelyOJRSB2004] `Oakley, J.E. and O'Hagan, A. (2004), Probabilistic sensitivity analysis of complex models: a Bayesian approach. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 66: 751-769. <https://doi.org/10.1111/j.1467-9868.2004.05304.x>`_
    """
    univariate_variables = [stats.norm()]*15
    variable = IndependentMarginalsVariable(univariate_variables)
    mean, variance, main_effects = oakley_function_statistics()
    return Benchmark(
        {'fun': oakley_function,
         'variable': variable, 'mean': mean, 'variance': variance,
         'main_effects': main_effects})


def setup_rosenbrock_function(nvars):
    r"""
    Setup the Rosenbrock function benchmark

    .. math:: f(z) = \sum_{i=1}^{d/2}\left[100(z_{2i-1}^{2}-z_{2i})^{2}+(z_{2i-1}-1)^{2}\right]

    This benchmark can also be used to test Bayesian inference methods.
    Specifically this benchmarks returns the log likelihood

    .. math:: l(z) = -f(z)

    which can be used to compute the posterior distribution

    .. math:: \pi_{\text{post}}(\rv)=\frac{\pi(\V{y}|\rv)\pi(\rv)}{\int_{\rvdom} \pi(\V{y}|\rv)\pi(\rv)d\rv}

    where the prior is the tensor product of :math:`d` independent and
    identically distributed uniform variables on :math:`[-2,2]`, i.e.
    :math:`\pi(\rv)=\frac{1}{4^d}`, and the likelihood is given by

    .. math:: \pi(\V{y}|\rv)=\exp\left(l(\rv)\right)

    Parameters
    ----------
    nvars : integer
        The number of variables of the Rosenbrock function

    Returns
    -------
    benchmark : :py:class:`pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes documented below

    fun : callable

        The rosenbrock function with signature

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1)

    jac : callable
        The jacobian of ``fun`` with signature

        ``jac(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nvars,1)

    hessp : callable
        Hessian of  ``fun`` times an arbitrary vector p with signature

        ``hessp(z, p) ->  ndarray shape (nvars,1)``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and p is an
        arbitraty vector with shape (nvars,1)

    variable : :py:class:`pyapprox.variables.IndependentMarginalsVariable`
        Object containing information of the joint density of the inputs z
        which is the tensor product of independent and identically distributed
        uniform variables on :math:`[-2,2]`.

    mean : float
        The mean of the rosenbrock function with respect to the pdf of
        variable.

    loglike : callable
        The log likelihood of the Bayesian inference problem for inferring z
        given the uniform prior specified by variable and the negative
        log likelihood given by the Rosenbrock function. loglike has the
        signature

        ``loglike(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1)

    loglike_grad : callable
        The gradient of the ``loglike`` with the signature

        ``loglike_grad(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1)

    References
    ----------
    .. [DixonSzego1990] `Dixon, L. C. W.; Mills, D. J. "Effect of Rounding Errors on the Variable Metric Method". Journal of Optimization Theory and Applications. 80: 175–179. 1994 <https://doi.org/10.1007%2FBF02196600>`_

    Examples
    --------
    >>> from pyapprox.benchmarks.benchmarks import setup_benchmark
    >>> benchmark=setup_benchmark('rosenbrock',nvars=2)
    >>> print(benchmark.keys())
    dict_keys(['fun', 'jac', 'hessp', 'variable', 'mean', 'loglike', 'loglike_grad'])
    """
    univariate_variables = [stats.uniform(-2, 4)]*nvars
    variable = IndependentMarginalsVariable(univariate_variables)

    benchmark = Benchmark(
        {'fun': rosenbrock_function, 'jac': rosenbrock_function_jacobian,
         'hessp': rosenbrock_function_hessian_prod, 'variable': variable,
         'mean': rosenbrock_function_mean(nvars)})
    benchmark.update({'loglike': lambda x: -benchmark['fun'](x),
                      'loglike_grad': lambda x: -benchmark['jac'](x)})
    return benchmark


def setup_genz_function(nvars, test_name, coefficients=None):
    r"""
    Setup the Genz Benchmarks.

    For example, the two-dimensional oscillatory Genz problem can be defined
    using

    >>> from pyapprox.benchmarks.benchmarks import setup_benchmark
    >>> benchmark=setup_benchmark('genz',nvars=2,test_name='oscillatory')
    >>> print(benchmark.keys())
    dict_keys(['fun', 'mean', 'variable'])

    Parameters
    ----------
    nvars : integer
        The number of variables of the Genz function

    test_name : string
        The test_name of the specific Genz function. See notes
        for options the string needed is given in brackets
        e.g. ('oscillatory')

    coefficients : tuple (ndarray (nvars), ndarray (nvars))
        The coefficients :math:`c_i` and :math:`w_i`
        If None (default) then
        :math:`c_j = \hat{c}_j\left(\sum_{i=1}^d \hat{c}_i\right)^{-1}` where
        :math:`\hat{c}_i=(10^{-15\left(\frac{i}{d}\right)^2)})`

    Returns
    -------
    benchmark : :py:class:`pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    fun : callable
        The function being analyzed

    variable : :py:class:`pyapprox.variables.JointVariable`
        Class containing information about each of the nvars inputs to fun

    mean: np.ndarray (nvars)
        The mean of the function with respect to the PDF of var

    References
    ----------
    .. [Genz1984] `Genz, A. Testing multidimensional integration routines. In Proc. of international conference on Tools, methods and languages for scientific and engineering computation (pp. 81-94), 1984 <https://dl.acm.org/doi/10.5555/2837.2842>`_

    Notes
    -----

    Corner Peak ('corner-peak')

    .. math:: f(z)=\left( 1+\sum_{i=1}^d c_iz_i\right)^{-(d+1)}

    Oscillatory ('oscillatory')

    .. math:: f(z) = \cos\left(2\pi w_1 + \sum_{i=1}^d c_iz_i\right)

    Gaussian Peak ('gaussian-peak')

    .. math:: f(z) = \exp\left( -\sum_{i=1}^d c_i^2(z_i-w_i)^2\right)

    Continuous ('continuous')

    .. math:: f(z) = \exp\left( -\sum_{i=1}^d c_i\lvert z_i-w_i\rvert\right)

    Product Peak ('product-peak')

    .. math:: f(z) = \prod_{i=1}^d \left(c_i^{-2}+(z_i-w_i)^2\right)^{-1}

    Discontinuous ('discontinuous')

    .. math:: f(z) = \begin{cases}0 & x_1>u_1 \;\mathrm{or}\; x_2>u_2\\\exp\left(\sum_{i=1}^d c_iz_i\right) & \mathrm{otherwise}\end{cases}

    """
    genz = GenzFunction(test_name, nvars)
    univariate_variables = [stats.uniform(0, 1)]*nvars
    variable = IndependentMarginalsVariable(univariate_variables)
    if coefficients is None:
        genz.set_coefficients(1, 'squared-exponential-decay', 0.25)
    else:
        genz.c, genz.w = coefficients
    attributes = {'fun': genz, 'mean': genz.integrate(), 'variable': variable}
    if test_name == 'corner-peak':
        attributes['variance'] = genz.variance()
    return Benchmark(attributes)


if PYA_DEV_AVAILABLE:
    from pyapprox_dev.fenics_models.advection_diffusion_wrappers import \
        setup_advection_diffusion_benchmark,\
        setup_advection_diffusion_source_inversion_benchmark,\
        setup_multi_level_advection_diffusion_benchmark
    from pyapprox_dev.fenics_models.helmholtz_benchmarks import \
        setup_mfnets_helmholtz_benchmark


def setup_benchmark(name, **kwargs):
    """
    Setup a PyApprox benchmark.

    Parameters
    ----------
    name : string
        The name of the benchmark

    kwargs: kwargs
     optional keyword arguments

    Returns
    -------
    benchmark : :py:class:`pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    The benchmark object must contain at least the following two attributes

    fun : callable
        A function with signature

        fun(samples) -> np.ndarray(nsamples, nqoi)

        where samples : np.ndarray(nvars, nsamples)

    variable : :py:class:`pyapprox.variables.JointVariable`
        Class containing information about each of the nvars inputs to fun

    """
    benchmarks = {
        'sobol_g': setup_sobol_g_function,
        'ishigami': setup_ishigami_function,
        'oakley': setup_oakley_function,
        'rosenbrock': setup_rosenbrock_function,
        'genz': setup_genz_function,
        'cantilever_beam': setup_cantilever_beam_benchmark,
        'wing_weight': setup_wing_weight_benchmark,
        'piston': setup_piston_benchmark,
        'chemical_reaction': setup_chemical_reaction_benchmark,
        'random_oscillator': setup_random_oscillator_benchmark,
        'coupled_springs': setup_coupled_springs_benchmark,
        'hastings_ecology': setup_hastings_ecology_benchmark,
        'multi_index_advection_diffusion':
        setup_multi_index_advection_diffusion_benchmark,
        'advection_diffusion_kle_inversion':
        setup_advection_diffusion_kle_inversion_benchmark,
        'polynomial_ensemble': setup_polynomial_ensemble,
        'tunable_model_ensemble': setup_tunable_model_ensemble,
        'short_column_ensemble': setup_short_column_ensemble,
        "parameterized_nonlinear_model": setup_parameterized_nonlinear_model}

    if name not in benchmarks:
        msg = f'Benchmark "{name}" not found.\n Available benchmarks are:\n'
        for key in benchmarks.keys():
            msg += f"\t{key}\n"
        raise ValueError(msg)

    return benchmarks[name](**kwargs)


def setup_cantilever_beam_benchmark():
    variable, design_variable = define_beam_random_variables()
    attributes = {'fun': cantilever_beam_objective,
                  'jac': cantilever_beam_objective_grad,
                  'constraint_fun': cantilever_beam_constraints,
                  'constraint_jac': cantilever_beam_constraints_jacobian,
                  'variable': variable,
                  'design_variable': design_variable,
                  'design_var_indices': np.array([4, 5])}
    return Benchmark(attributes)


def setup_piston_benchmark():
    r"""
    Returns
    -------
    benchmark : :py:class:`pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes documented below

    fun : callable

        The piston model with signature

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1)

    variable : :py:class:`pyapprox.variables.IndependentMarginalsVariable`
        Object containing information of the joint density of the inputs z
        which is the tensor product of independent and identically distributed
        uniform variables`.

    References
    ----------
    .. [Moon2010] `Design and Analysis of Computer Experiments for Screening Input Variables (Doctoral dissertation, Ohio State University) <http://rave.ohiolink.edu/etdc/view?acc_num=osu1275422248>`_
    """
    variable = define_piston_random_variables()
    attributes = {'fun': piston_function,
                  "jac": piston_function_gradient,
                  'variable': variable}
    return Benchmark(attributes)


def setup_wing_weight_benchmark():
    r"""
    Setup the wing weight model benchmark.

    The model is given by


    ::math f(x) = 0.036\; S_w^{0.758}W_{fw}^{0.0035}\left(\frac{A}{\cos^2(\Lambda)}\right)^{0.6}q^{0.006}\lambda^{0.04}\left(\frac{100t_c}{\cos(\Lambda)}\right)^{-0.3}(N_zW_{dg})^{0.49}+S_wW_p,

    Returns
    -------
    benchmark : :py:class:`pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes documented below

    fun : callable

        The wing weight model with signature

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1)

    jac : callable
        The jacobian of ``fun`` with signature

        ``jac(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nvars,1)

    variable : :py:class:`pyapprox.variables.IndependentMarginalsVariable`
        Object containing information of the joint density of the inputs z
        which is the tensor product of independent and identically distributed
        uniform variables`.

    References
    ----------
    .. [Moon2012] `Moon, H., Dean, A. M., & Santner, T. J. (2012). Two-stage sensitivity-based group screening in computer experiments. Technometrics, 54(4), 376-387. <https://doi.org/10.1080/00401706.2012.725994>`_
    """
    variable = define_wing_weight_random_variables()
    attributes = {'fun': wing_weight_function,
                  'variable': variable,
                  'jac':  wing_weight_gradient}
    return Benchmark(attributes)


def setup_chemical_reaction_benchmark():
    """
    Setup the chemical reaction model benchmark

    Model of species absorbing onto a surface out of gas phase
    u = y[0] = monomer species
    v = y[1] = dimer species
    w = y[2] = inert species

    Returns
    -------
    benchmark : :py:class:`pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes documented below

    fun : callable

        The piston model with signature

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1)

    variable : :py:class:`pyapprox.variables.IndependentMarginalsVariable`
        Object containing information of the joint density of the inputs z
        which is the tensor product of independent and identically distributed
        uniform variables`.

    References
    ----------
    Vigil et al., Phys. Rev. E., 1996; Makeev et al., J. Chem. Phys., 2002
    Bert Debuschere used this example 2014 talk
    """
    variable = define_chemical_reaction_random_variables()
    model = ChemicalReactionModel()
    attributes = {'fun': model,
                  'variable': variable}
    return Benchmark(attributes)


def setup_random_oscillator_benchmark():
    variable = define_random_oscillator_random_variables()
    model = RandomOscillator()
    attributes = {'fun': model,
                  'variable': variable}
    return Benchmark(attributes)


def setup_coupled_springs_benchmark():
    variable = define_coupled_springs_random_variables()
    model = CoupledSprings()
    attributes = {'fun': model,
                  'variable': variable}
    return Benchmark(attributes)


def setup_hastings_ecology_benchmark(qoi_functional=None, time=None):
    variable = define_nondim_hastings_ecology_random_variables()
    model = HastingsEcology(qoi_functional, True, time)
    attributes = {'fun': model,
                  'variable': variable}
    return Benchmark(attributes)


def setup_polynomial_ensemble():
    r"""
    Return an ensemble of 5 univariate models of the form

    .. math:: f_\alpha(\rv)=\rv^{5-\alpha}, \quad \alpha=0,\ldots,4

    where :mat:`z\sim\mathcal{U}[0, 1]`

    Returns
    -------
    benchmark : :py:class:`pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    fun : callable
        The function being analyzed

    variable : :py:class:`pyapprox.variables.JointVariable`
        Class containing information about each of the nvars inputs to fun

    means : np.ndarray (nmodels)
        The mean of each model fidelity

    model_covariance : np.ndarray (nmodels)
        The covariance between the outputs of each model fidelity

    References
    ----------
    .. [GGEJJCP2020] `A generalized approximate control variate framework for multifidelity uncertainty quantification,  Journal of Computational Physics,  408:109257, 2020. <https://doi.org/10.1016/j.jcp.2020.109257>`_
    """
    model = PolynomialModelEnsemble()
    return Benchmark(
        {'fun': model, 'variable': model.variable, "means": model.get_means(),
         "model_covariance": model.get_covariance_matrix()})


def setup_tunable_model_ensemble(theta1=np.pi/2*0.95, shifts=None):
    model = TunableModelEnsemble(theta1, shifts)
    return Benchmark(
        {'fun': model, 'variable': model.variable, "means": model.get_means(),
         "model_covariance": model.get_covariance_matrix()})


def setup_short_column_ensemble():
    model = ShortColumnModelEnsemble()
    return Benchmark(
        {'fun': model, 'variable': model.variable, "means": model.get_means(),
         "model_covariance": model.get_covariance_matrix()})


def setup_parameterized_nonlinear_model():
    model = ParameterizedNonlinearModel()
    model.qoi = np.array([1])
    marginals = [stats.uniform(lb, ub-lb)
                 for lb, ub in zip(model.ranges[::2], model.ranges[1::2])]
    variable = IndependentMarginalsVariable(marginals)
    return Benchmark(
        {'fun': model, 'variable': variable})


def setup_multi_index_advection_diffusion_benchmark(
        nvars, kle_length_scale, kle_sigma,
        max_eval_concurrency=1, config_values=None):
    r"""
    Compute functionals of the following model of transient advection-diffusion (with 3 configure variables which control the two spatial mesh resolutions and the timestep)

    .. math::

       \frac{\partial u}{\partial t}(x,t,\rv) + \nabla u(x,t,\rv)-\nabla\cdot\left[k(x,\rv) \nabla u(x,t,\rv)\right] &=g(x,t) \qquad (x,t,\rv)\in D\times [0,1]\times\rvdom\\
       \mathcal{B}(x,t,\rv)&=0 \qquad\qquad (x,t,\rv)\in \partial D\times[0,1]\times\rvdom\\
       u(x,t,\rv)&=u_0(x,\rv) \qquad (x,t,\rv)\in D\times\{t=0\}\times\rvdom

    Following [NTWSIAMNA2008]_, [JEGGIJNME2020]_ we set

    .. math:: g(x,t)=(1.5+\cos(2\pi t))\cos(x_1),

    the initial condition as :math:`u(x,z)=0`, :math:`B(x,t,z)` to be zero dirichlet boundary conditions.

    and we model the diffusivity :math:`k` as a random field represented by the
    Karhunen-Loeve (like) expansion (KLE)

    .. math::

       \log(k(x,\rv)-0.5)=1+\rv_1\left(\frac{\sqrt{\pi L}}{2}\right)^{1/2}+\sum_{k=2}^d \lambda_k\phi(x)\rv_k,

    with

    .. math::

       \lambda_k=\left(\sqrt{\pi L}\right)^{1/2}\exp\left(-\frac{(\lfloor\frac{k}{2}\rfloor\pi L)^2}{4}\right) k>1,  \qquad\qquad  \phi(x)=
       \begin{cases}
       \sin\left(\frac{(\lfloor\frac{k}{2}\rfloor\pi x_1)}{L_p}\right) & k \text{ even}\,,\\
       \cos\left(\frac{(\lfloor\frac{k}{2}\rfloor\pi x_1)}{L_p}\right) & k \text{ odd}\,.
       \end{cases}

    where :math:`L_p=\max(1,2L_c)`, :math:`L=\frac{L_c}{L_p}`.

    The quantity of interest :math:`f(z)` is the measurement of the solution at a location :math:`x_k` at the final time :math:`T=1` obtained via the linear functional

    .. math:: f(z)=\int_D u(x,T,z)\frac{1}{2\pi\sigma^2}\exp\left(-\frac{\lVert x-x_k \rVert^2_2}{\sigma^2}\right) dx


    Parameters
    ----------
    nvars : integer
        The number of variables of the KLE

    kle_length_scale : float
        The correlation length :math:`L_c` of the covariance kernel

    kle_sigma : float
        The standard deviation of the KLE kernel

    max_eval_concurrency : integer
        The maximum number of simulations that can be run in parallel. Should be         no more than the maximum number of cores on the computer being used

    Returns
    -------
    benchmark : pyapprox.benchmarks.benchmarks.Benchmark
       Object containing the benchmark attributes documented below

    fun : callable

        The quantity of interest :math:`f(w)` with signature

        ``fun(w) -> np.ndarray``

        where ``w`` is a 2D np.ndarray with shape (nvars+3,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1). The first ``nvars``
        rows of ``w`` are realizations of the random variables. The last 3 rows
        are configuration variables specifying the numerical discretization of
        the PDE model. Specifically the first and second configuration variables
        specify the levels :math:`l_{x_1}` and :math:`l_{x_2}` which dictate
        the resolution of the FEM mesh in the directions :math:`{x_1}` and
        :math:`{x_2}` respectively. The number of cells in the :math:`{x_i}`
        direction is given by :math:`2^{l_{x_i}+2}`. The third configuration
        variable specifies the level :math:`l_t` of the temporal discretization.
        The number of timesteps satisfies :math:`2^{l_{t}+2}` so the timestep
        size is and :math:`T/2^{l_{t}+2}`.

    variable : pya.IndependentMarginalsVariable
        Object containing information of the joint density of the inputs z
        which is the tensor product of independent and identically distributed
        uniform variables on :math:`[-\sqrt{3},\sqrt{3}]`.

    Examples
    --------
    >>> from pyapprox_dev.benchmarks.benchmarks import setup_benchmark
    >>> benchmark = setup_benchmark('multi_index_advection_diffusion', nvars=2)
    >>> print(benchmark.keys())
    dict_keys(['fun', 'variable'])
    """
    base_model, variable = _setup_multi_index_advection_diffusion_benchmark(
        kle_length_scale, kle_sigma, nvars, config_values=config_values)
    timer_model = TimerModel(base_model, base_model)
    pool_model = PoolModel(
        timer_model, max_eval_concurrency, base_model=base_model)
    model = WorkTrackingModel(pool_model, base_model,
                              base_model._nconfig_vars)
    attributes = {'fun': model, 'variable': variable}
    return Benchmark(attributes)


def setup_advection_diffusion_kle_inversion_benchmark(
        source_loc=[0.25, 0.75], source_amp=100, source_width=0.1,
        kle_length_scale=0.5, kle_stdev=1, kle_nvars=2, true_sample=None,
        orders=[20, 20], noise_stdev=0.4, nobs=2, max_eval_concurrency=1):
    r"""
    Compute functionals of the following model of transient diffusion of
    a contaminant

    .. math::

       \frac{\partial u}{\partial t}(x,t,\rv) + \nabla u(x,t,\rv)-\nabla\cdot\left[k(x,\rv) \nabla u(x,t,\rv)\right] &=g(x,t) \qquad (x,t,\rv)\in D\times [0,1]\times\rvdom\\
       \mathcal{B}(x,t,\rv)&=0 \qquad\qquad (x,t,\rv)\in \partial D\times[0,1]\times\rvdom\\
       u(x,t,\rv)&=u_0(x,\rv) \qquad (x,t,\rv)\in D\times\{t=0\}\times\rvdom

    Following [MNRJCP2006]_, [LMSISC2014]_ we set

    .. math:: g(x,t)=\frac{s}{2\pi h^2}\exp\left(-\frac{\lvert x-x_\mathrm{src}\rvert^2}{2h^2}\right)

    the initial condition as :math:`u(x,z)=0`, :math:`B(x,t,z)` to be zero Neumann boundary conditions, i.e.

    .. math:: \nabla u\cdot n = 0 \quad\mathrm{on} \quad\partial D

    and we model the diffusivity :math:`k=1` as a constant.

    The quantities of interest are point observations :math:`u(x_l)`
    taken at :math:`P` points in time :math:`\{t_p\}_{p=1}^P` at :math:`L`
    locations :math:`\{x_l\}_{l=1}^L`. The final time :math:`T` is the last
    observation time.

    These functionals can be used to define the posterior distribution

    .. math::  \pi_{\text{post}}(\rv)=\frac{\pi(\V{y}|\rv)\pi(\rv)}{\int_{\rvdom} \pi(\V{y}|\rv)\pi(\rv)d\rv}

    where the prior is the tensor product of independent and identically
    distributed uniform variables on :math:`[0,1]` i.e.
    :math:`\pi(\rv)=1`, and the likelihood is given by

    .. math:: \pi(\V{y}|\rv)=\frac{1}{(2\pi)^{d/2}\sigma}\exp\left(-\frac{1}{2}\frac{(y-f(\rv))^T(y-f(\rv))}{\sigma^2}\right)

    and :math:`y` are noisy observations of the solution `u` at the 9
    points of a uniform :math:`3\times 3` grid covering the physical domain
    :math:`D` at successive times :math:`\{t_p\}_{p=1}^P`. Here the noise is
    indepenent and Normally distrbuted with mean
    zero and variance :math:`\sigma^2`.

    Parameters
    ----------
    source_loc : np.ndarray (2)
        The center of the source

    source_amp : float
        The source strength :math:`s`

    source_width : float
        The source width :math:`h`

    true_sample : np.ndarray (2)
        The true location of the source used to generate the observations
        used in the likelihood function

    noise_stdev : float
        The standard deviation :math:`sigma` of the observational noise

    max_eval_concurrency : integer
        The maximum number of simulations that can be run in parallel. Should
        be no more than the maximum number of cores on the computer being used

    Returns
    -------
    benchmark : pya.Benchmark
       Object containing the benchmark attributes documented below

    fun : callable

        The quantity of interest :math:`f(w)` with signature

        ``fun(w) -> np.ndarray``

        where ``w`` is a 2D np.ndarray with shape (nvars+3,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1). The first ``nvars``
        rows of ``w`` are realizations of the random variables. The last 3 rows
        are configuration variables specifying the numerical discretization of
        the PDE model. Specifically the first and second configuration variables
        specify the levels :math:`l_{x_1}` and :math:`l_{x_2}` which dictate
        the resolution of the FEM mesh in the directions :math:`{x_1}` and
        :math:`{x_2}` respectively. The number of cells in the :math:`{x_i}`
        direction is given by :math:`2^{l_{x_i}+2}`. The third configuration
        variable specifies the level :math:`l_t` of the temporal discretization.
        The number of timesteps satisfies :math:`2^{l_{t}+2}` so the timestep
        size is and :math:`T/2^{l_{t}+2}`.

    variable : py:class:`pyapprox.variabels.joint.IndependentMarginalsVariable`
        Object containing information of the joint density of the inputs z
        which is the tensor product of independent and identically distributed
        uniform variables on :math:`[0,1]`.

    Examples
    --------
    >>> from pyapprox_dev.benchmarks.benchmarks import setup_benchmark
    >>> benchmark = setup_benchmark('advection_diffusion_kle_inversion', nvars=2)
    >>> print(benchmark.keys())
    dict_keys(['fun', 'variable'])
    """

    base_model, variable, true_sample, noiseless_obs, obs = (
        _setup_inverse_advection_diffusion_benchmark(
            source_amp, source_width, source_loc, nobs, noise_stdev,
            kle_length_scale, kle_stdev, kle_nvars, orders))
    # add wrapper to allow execution times to be captured
    timer_model = TimerModel(base_model, base_model)
    pool_model = PoolModel(
        timer_model, max_eval_concurrency, base_model=base_model)

    # add wrapper that tracks execution times.
    model = WorkTrackingModel(pool_model, base_model)

    attributes = {'fun': model, 'variable': variable,
                  "noiseless_obs": noiseless_obs, "obs": obs,
                  "true_sample": true_sample}
    return Benchmark(attributes)
