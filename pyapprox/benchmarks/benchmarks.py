from functools import partial

import numpy as np
from scipy import stats
from scipy.optimize import OptimizeResult

from pyapprox.benchmarks.sensitivity_benchmarks import (
    get_sobol_g_function_statistics, get_ishigami_funciton_statistics,
    oakley_function, oakley_function_statistics, sobol_g_function,
    ishigami_function, ishigami_function_jacobian, ishigami_function_hessian)
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
    define_nondim_hastings_ecology_random_variables,
    ParameterizedNonlinearModel)
from pyapprox.benchmarks.genz import GenzFunction
from pyapprox.benchmarks.multifidelity_benchmarks import (
    PolynomialModelEnsemble, TunableModelEnsemble, ShortColumnModelEnsemble,
    MultioutputModelEnsemble)
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.interface.wrappers import (
    TimerModel, PoolModel, WorkTrackingModel)
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

    variable : :py:class:`~pyapprox.variables.JointVariable`
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
    def __repr__(self):
        return "Benchmark("+", ".join(
            [str(key) for key, item in self.items()]) + ")"


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
    benchmark : :py:class:`~pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    fun : callable
        The function being analyzed

    variable : :py:class:`~pyapprox.variables.JointVariable`
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
    benchmark : :py:class:`~pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    fun : callable
        The function being analyzed

    variable : :py:class:`~pyapprox.variables.JointVariable`
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

    where :math:`z` consists of 15 I.I.D. standard Normal variables and the data :math:`a_1,a_2,a_3` and :math:`~M` are defined in the function :py:func:`~pyapprox.benchmarks.sensitivity_benchmarks.get_oakley_function_data`.

    >>> from pyapprox.benchmarks.benchmarks import setup_benchmark
    >>> benchmark=setup_benchmark('oakley')
    >>> print(benchmark.keys())
    dict_keys(['fun', 'variable', 'mean', 'variance', 'main_effects'])

    Returns
    -------
    benchmark : :py:class:`~pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    fun : callable
        The function being analyzed

    variable : :py:class:`~pyapprox.variables.JointVariable`
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
    benchmark : :py:class:`~pyapprox.benchmarks.Benchmark`
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

    variable : :py:class:`~pyapprox.variables.IndependentMarginalsVariable`
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


def setup_genz_function(nvars, test_name, coeff_type=None, w=0.25, c_factor=1,
                        coeff=None):
    r"""
    Setup one of the six Genz integration benchmarks
    :math:`f_d(x):\mathbb{R}^D\to\mathbb{R}`,
    where :math:`x=[x_1,\ldots,x_D]^\top`.
    The number of inputs :math:`D` and the anisotropy (relative importance of
    each variable and interactions) of the functions can be adjusted.
    The definition of each function is in the Notes section.

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
        e.g. ('oscillatory'). Choose from
        ["oscillatory", "product_peak", "corner_peak", "c0continuous", "discontinuous"]

    coef_type : string
        Choose from ["no_decay", "quadratic_decay", "quartic_decay",
        "exponential_decay". "squared_exponential_decay"]

    w : float 0<=w<=1
        Set :math:`w_d=w, d=1,\ldots,D`.

    c_factor : float `c_factor>0`
        Scale the integrand.

    coeff : tuple (ndarray (nvars, 1), ndarray (nvars, 1))
        The coefficients :math:`c_d` and :math:`w_d`
        If provided it will overwite the coefficients defined by
        `coeff_type`, `w` and `c_factor`

    Returns
    -------
    benchmark : :py:class:`pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    fun : callable
        The function being analyzed

    variable : :py:class:`~pyapprox.variables.JointVariable`
        Class containing information about each of the nvars inputs to fun

    mean: np.ndarray (nvars)
        The mean of the function with respect to the PDF of var

    References
    ----------
    .. [Genz1984] `Genz, A. Testing multidimensional integration routines. In Proc. of international conference on Tools, methods and languages for scientific and engineering computation (pp. 81-94), 1984 <https://dl.acm.org/doi/10.5555/2837.2842>`_

    Notes
    -----
    The six Genz test function are:

    Oscillatory ('oscillatory')

    .. math:: f(z) = \cos\left(2\pi w_1 + \sum_{d=1}^D c_dz_d\right)

    Product Peak ('product_peak')

    .. math:: f(z) = \prod_{d=1}^D \left(c_d^{-2}+(z_d-w_d)^2\right)^{-1}

    Corner Peak ('corner_peak')

    .. math:: f(z)=\left( 1+\sum_{d=1}^D c_dz_d\right)^{-(D+1)}

    Gaussian Peak ('gaussian')

    .. math:: f(z) = \exp\left( -\sum_{d=1}^D c_d^2(z_d-w_d)^2\right)

    C0 Continuous ('c0continuous')

    .. math:: f(z) = \exp\left( -\sum_{d=1}^D c_d\lvert z_d-w_d\rvert\right)

    Discontinuous ('discontinuous')

    .. math:: f(z) = \begin{cases}0 & z_1>w_1 \;\mathrm{or}\; z_2>w_2\\\exp\left(\sum_{d=1}^D c_d z_d\right) & \mathrm{otherwise}\end{cases}

    Increasing :math:`\lVert c \rVert` will in general make
    the integrands more difficult.

    The :math:`0\le w_d \le 1` parameters do not affect the difficulty
    of the integration problem. We set :math:`w_1=w_2=\ldots=W_D`.

    The coefficient types implement different decay rates for :math:`c_d`.
    This allows testing of methods that can identify and exploit anisotropy.
    They are as follows:

    No decay (none)

    .. math:: \hat{c}_d=\frac{d+0.5}{D}

    Quadratic decay (qudratic)

    .. math:: \hat{c}_d = \frac{1}{(D + 1)^2}

    Quartic decay (quartic)

    .. math:: \hat{c}_d = \frac{1}{(D + 1)^4}

    Exponential decay (exp)

    .. math:: \hat{c}_d=\exp\left(\log(c_\mathrm{min})\frac{d+1}{D}\right)

    Squared-exponential decay (sqexp)

    .. math:: \hat{c}_d=10^{\left(\log_{10}(c_\mathrm{min})\frac{(d+1)^2}{D}\right)}

    Here :math:`c_\mathrm{min}` is argument that sets the minimum value of :math:`c_D`.

    Once the formula are used the coefficients are normalized such that

    .. math:: c_d = c_\text{factor}\frac{\hat{c}_d}{\sum_{d=1}^D \hat{c}_d}.
    """
    genz = GenzFunction()
    univariate_variables = [stats.uniform(0, 1)]*nvars
    variable = IndependentMarginalsVariable(univariate_variables)
    if coeff_type is None:
        coeff_type = 'none'
    genz.set_coefficients(nvars, c_factor, coeff_type, w)
    if coeff is not None:
        genz._c, genz._w = np.asarray(coeff[0]), np.asarray(coeff[1])
        if genz._c.ndim == 1:
            genz._c = genz._c[:, None]
        if genz._w.ndim == 1:
            genz._w = genz._w[:, None]
        assert genz._c.ndim == 2 and genz._w.ndim == 2
    attributes = {'fun': partial(genz, test_name),
                  'mean': genz.integrate(test_name),
                  'variable': variable}
    return Benchmark(attributes)


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
    benchmark : :py:class:`~pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes documented below

    fun : callable

        The piston model with signature

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1)

    variable : :py:class:`~pyapprox.variables.IndependentMarginalsVariable`
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


    .. math::

       f(x) = 0.036\; S_w^{0.758}W_{fw}^{0.0035}\left(\frac{A}{\cos^2(\Lambda)}\right)^{0.6}q^{0.006}\lambda^{0.04}\left(\frac{100t_c}{\cos(\Lambda)}\right)^{-0.3}(N_zW_{dg})^{0.49}+S_wW_p,

    Returns
    -------
    benchmark : :py:class:`~pyapprox.benchmarks.Benchmark`
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

    variable : :py:class:`~pyapprox.variables.IndependentMarginalsVariable`
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
    benchmark : :py:class:`~pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes documented below

    fun : callable

        The piston model with signature

        ``fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1)

    variable : :py:class:`~pyapprox.variables.IndependentMarginalsVariable`
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


def _extract_acv_benchmark_dict(model):
    return {'fun': model, 'variable': model.variable,
            "mean": model.get_means(),
            "covariance": model.get_covariance_matrix(),
            "funs": model.funs, "nqoi": model.nqoi}


def setup_polynomial_ensemble(nmodels=5):
    r"""
    Return an ensemble of 5 univariate models of the form

    .. math:: f_\alpha(\rv)=\rv^{5-\alpha}, \quad \alpha=0,\ldots,4

    where :math:`z\sim\mathcal{U}[0, 1]`

    Returns
    -------
    benchmark : :py:class:`~pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    fun : callable
        The function being analyzed

    variable : :py:class:`~pyapprox.variables.JointVariable`
        Class containing information about each of the nvars inputs to fun

    means : np.ndarray (nmodels)
        The mean of each model fidelity

    covariance : np.ndarray (nmodels)
        The covariance between the outputs of each model fidelity

    References
    ----------
    .. [GGEJJCP2020] `A generalized approximate control variate framework for multifidelity uncertainty quantification,  Journal of Computational Physics,  408:109257, 2020. <https://doi.org/10.1016/j.jcp.2020.109257>`_
    """
    model = PolynomialModelEnsemble(nmodels)
    return Benchmark(_extract_acv_benchmark_dict(model))


def setup_tunable_model_ensemble(theta1=np.pi/2*0.95, shifts=None):
    model = TunableModelEnsemble(theta1, shifts)
    return Benchmark(_extract_acv_benchmark_dict(model))


def setup_short_column_ensemble(nmodels=5):
    model = ShortColumnModelEnsemble(nmodels)
    return Benchmark(_extract_acv_benchmark_dict(model))


def setup_multioutput_model_ensemble():
    model = MultioutputModelEnsemble()
    return Benchmark(_extract_acv_benchmark_dict(model))


def setup_parameterized_nonlinear_model():
    model = ParameterizedNonlinearModel()
    model.qoi = np.array([1])
    marginals = [stats.uniform(lb, ub-lb)
                 for lb, ub in zip(model.ranges[::2], model.ranges[1::2])]
    variable = IndependentMarginalsVariable(marginals)
    return Benchmark(
        {'fun': model, 'variable': variable})


def setup_multi_index_advection_diffusion_benchmark(
        kle_nvars=2, kle_length_scale=0.5, kle_stdev=1,
        max_eval_concurrency=1, time_scenario=None,
        functional=None, config_values=None,
        source_loc=[0.25, 0.75], source_scale=0.1,
        source_amp=100.0, vel_vec=[1., 0.], kle_mean_field=0):
    r"""
    This benchmark is used to test methods for forward propagation of
    uncertainty. The forward simulation model is the transient
    advection-diffusion model

    .. math::

       \frac{\partial u}{\partial t}(x,t,\rv) = \nabla\cdot\left[k(x,\rv) \nabla u(x,t,\rv)\right] -\nabla \cdot (v u(x,t,\rv))+g(x,t) &(x,t,\rv)\in D\times [0,1]\times\rvdom\\
       \mathcal{B}(x,t,\rv)=0  &(x,t,\rv)\in \partial D\times[0,1]\times\rvdom\\
       u(x,t,\rv)=u_0(x,\rv) & (x,t,\rv)\in D\times\{t=0\}\times\rvdom

    where

    .. math::

        g(x,t)=\frac{100}{2\pi 0.1^2}\exp\left(-\frac{\lvert x-[0.25,0.75]^\top\rvert^2}{2\cdot 0.1^2}\right)-\frac{s_\mathrm{sink}}{2\pi h_\mathrm{sink}^2}\exp\left(-\frac{\lvert x-x_\mathrm{sink}\rvert^2}{2h_\mathrm{sink}^2}\right)

    and :math:`B(x,t,z)` enforces Robin boundary conditions, i.e.

    .. math:: K(x,\rv)\nabla u(x,t,\rv)\cdot n -0.1 u(x,t,\rv)= 0 \quad\mathrm{on} \quad\partial D


    As with the :py:func:`pyapprox.benchmarks.setup_advection_diffusion_kle_inversion_benchmark`
    we parameterize the uncertain diffusivity with a Karhunen Loeve Expansion (KLE)

    .. math:: k(x, \rv)=\exp\left(k_0+\sum_{d=1}^D \sqrt{\lambda_d}\psi_d(x)\rv_d\right).

    If no initial condition is provided by the user then the governing equations in :py:func:`pyapprox.benchmarks.setup_advection_diffusion_kle_inversion_benchmark` is used to create an initial condition, where the forcing is set to be the first term of :math:`g` here. I.e. the steady state solution before the second term of :math:`g` is used to remove the concentration :math:`u` from the domain.

    The quantity of interest :math:`f(z)` is the integral of the final solution in the subdomain :math:`S=[0.75, 1]\times[0, 0.25]`, i.e.

    .. math:: f(z)=\int_S u(x,T,z) dx

    This model can be evaluated using different numerical discreizations that control the two spatial mesh resolutions and the timestep. The model is evaluated by specifying the random variables and the three numerical (configuration) variables.

    If not time_scenario is provided. The QoI from the steady state solution is returned.

    This benchmark can be modified by
    changing the default keyword arguments if necessary.

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

    time_scenario : dict
        Options defining the transient simulation. If None a steady state problem will be solved
        If True the default time scenario will be used which corresponds to specifying the dictionary

        .. code-block:: python

           time_scenario = {
               "final_time": 0.2,
               "butcher_tableau": "im_crank2",
               "deltat": 0.1,  # default will be overwritten
               "init_sol_fun": None,
               "sink": None
               }

        Respectively, the entries of sink are :math:`s_\mathrm{sink}, h_\mathrm{sink}, x_\mathrm{sink}`, e.g. [50, 0.1, [0.75, 0.75]]. If None then the sink will be turned off.
        init_sol is a callable function with signature ``init_sol_fun(x) -> np.ndarray (nx, 1)``
        where ``x`` is np.ndarray (nphys_vars, nx) are physical coordinates in the mesh. ``butcher_tableau`` specifies the time-stepping scheme which can be either
        ``im_beuler1`` or ``im_crank2``. ``final_time`` specifies :math:`T`.

    functional : callable
        Function used to compute the Quantities of interest with signature

        ``functional(sol, z) -> float``

        Here ``sol: torch.tensor (ndof)`` is the solution at the mesh points
        and ``z -> np.ndarray(nkle_vars, 1)`` is the value of the KLE
        coefficients that produced ``sol``. If None the subdomain intergral
        of sol at the final time will be used as defined above.

    config_values : list (np.ndarray)
        List with three entries (two if time_scenario=None) The first two are
        the values of the degrees that can be used to construct the
        collocation mesh in each physical direction. The third is an array of
        the timestep sizes that can be used to integrate the PDE in time.

    source_loc : np.ndarray (2)
        The center of the source

    source_amp : float
        The source strength :math:`s`

    source_scale : float
        The source width :math:`h`

    vel_vec: iterable (2) default [1., 0.]
        The spatially independent velocity field :math:`v`

    kle_mean_field : float (default 0)
        The spatially independent mean :math:`k_0` of the KLE field in log space

    Returns
    -------
    benchmark : :py:class:`~pyapprox.benchmarks.benchmarks.Benchmark`
       Object containing the benchmark attributes documented below

    fun : callable

        The quantity of interest :math:`f(w)` with signature

        ``fun(w) -> np.ndarray``

        where ``w`` is a 2D np.ndarray with shape (nvars+3,nsamples) and the
        output is a 2D np.ndarray with shape (nsamples,1). The first ``nvars``
        rows of ``w`` are realizations of the random variables. The last 3 rows
        are configuration variables specifying the numerical discretization of
        the PDE model. See config_values documentation above. This is useful
        for testing multi-index multi-fidelity methods.

    variable : :py:class:`~pyapprox.variables.joint.IndependentMarginalsVariable`
        Object containing information of the joint density of the inputs z
        which is the tensor product of independent and identically distributed
        Gaussian variables :math:`\mathcal{N}(0,1)`.

    get_num_degrees_of_freedom : callable
        Function that returns the number of mesh points multiplied by the
        number of timesteps, with signature

        ``get_num_degrees_of_freedom(v) -> int``

        where ``v->np.ndarray(3)`` are the thre configuration values
        specifiying the numerical discretization

    config_var_trans : :py:class:`~pyapprox.variables.transforms.ConfigureVariableTransformation`
        A transform that maps the configuration values to and from a canonical space.

    model_ensemble : :py:class:`~pyapprox.interface.wrappers.ModelEnsemble`
       Function return the quantities of interest with the signature

       ``fun(w) -> np.ndarray``

        where ``w`` is a 2D np.ndarray with shape (nvars+1, nsamples) and the
        output is a 2D np.ndarray with shape (nsamples, 1). The first ``nvars``
        rows of ``w`` are realizations of the random variables. The last row
        is a model ID specifying a different numerical discretization. This is useful for testing
        multi-fidelity approximate control variate Monte Carlo estimators.

    Examples
    --------
    >>> from pyapprox_dev.benchmarks.benchmarks import setup_benchmark
    >>> benchmark = setup_benchmark('multi_index_advection_diffusion', nvars=2)
    >>> print(benchmark.keys())
    dict_keys(['fun', 'variable'])
    """
    base_model, variable, config_var_trans, model_ensemble = (
        _setup_multi_index_advection_diffusion_benchmark(
            kle_length_scale, kle_stdev, kle_nvars,
            time_scenario=time_scenario,
            functional=functional, config_values=config_values,
            source_loc=source_loc, source_scale=source_scale,
            source_amp=source_amp, vel_vec=vel_vec,
            kle_mean_field=kle_mean_field))
    timer_model = TimerModel(base_model, base_model)
    pool_model = PoolModel(
        timer_model, max_eval_concurrency, base_model=base_model)
    # enforce_timer_model must be False because pool is wrapping TimerModel
    model = WorkTrackingModel(pool_model, base_model,
                              base_model._nconfig_vars,
                              enforce_timer_model=False)
    model0 = base_model._model_ensemble.functions[0]
    attributes = {
        'fun': model, 'variable': variable,
        "get_num_degrees_of_freedom": model0.get_num_degrees_of_freedom_cost,
        "config_var_trans": config_var_trans,
        'model_ensemble': model_ensemble, "funs": model_ensemble.functions}
    return Benchmark(attributes)


def setup_advection_diffusion_kle_inversion_benchmark(
        source_loc=[0.25, 0.75], source_amp=100, source_width=0.1,
        kle_length_scale=0.5, kle_stdev=1, kle_nvars=2, true_sample=None,
        orders=[20, 20], noise_stdev=0.4, nobs=2, max_eval_concurrency=1,
        obs_indices=None):
    r"""
    A benchmark for testing maximum likelihood estimation and Bayesian inference algorithms that involves
    learning the uncertain parameters :math:`\rv` from synthteically generated observational data using
    the model

    .. math::

        \nabla u(x,t,\rv)-\nabla\cdot\left[k(x,\rv) \nabla u(x,t,\rv)\right] &=g(x,t) \qquad (x,t,\rv)\in D\times [0,1]\times\rvdom\\
       \mathcal{B}(x,t,\rv)&=0 \qquad\qquad (x,t,\rv)\in \partial D\times[0,1]\times\rvdom\\
       u(x,t,\rv)&=u_0(x,\rv) \qquad (x,t,\rv)\in D\times\{t=0\}\times\rvdom

    Following [MNRJCP2006]_, [LMSISC2014]_ we set

    .. math:: g(x,t)=\frac{s_\mathrm{src}}{2\pi h_\mathrm{src}^2}\exp\left(-\frac{\lvert x-x_\mathrm{src}\rvert^2}{2h_\mathrm{src}^2}\right)

    the initial condition as :math:`u(x,z)=0`, :math:`B(x,t,z)` to be zero Dirichlet boundary conditions, i.e.

    .. math:: u(x) = 0 \quad\mathrm{on} \quad\partial D

    and we model the diffusivity as a Karhunen Loeve Expansion (KLE)

    .. math:: k(x, \rv)=\exp\left(\sum_{d=1}^D \sqrt{\lambda_d}\psi_d(x)\rv_d\right).

    The observations are noisy  observations :math:`u(x_l)`
    at :math:`L` locations :math:`\{x_l\}_{l=1}^L` with additive independent Gaussian noise
    with mean zero and variance :math:`\sigma^2`.
    These observations can be used to define the posterior distribution

    .. math::  \pi_{\text{post}}(\rv)=\frac{\pi(\V{y}|\rv)\pi(\rv)}{\int_{\rvdom} \pi(\V{y}|\rv)\pi(\rv)d\rv}

    where the prior is the tensor product of independent and identically
    distributed Gaussian with zero mean and unit variance
    In this scenario the likelihood is given by

    .. math:: \pi(\V{y}|\rv)=\frac{1}{(2\pi)^{d/2}\sigma}\exp\left(-\frac{1}{2}\frac{(y-f(\rv))^T(y-f(\rv))}{\sigma^2}\right)

    which can be used for Bayesian inference and maximum likelihood estimation of the parameters
    :math:`\rv`.

    Parameters
    ----------
    source_loc : np.ndarray (2)
        The center of the source

    source_amp : float
        The source strength :math:`s`

    source_width : float
        The source width :math:`h`

    kle_length_scale : float
        The length scale of the KLE

    kle_stdev : float
        The standard deviation of the KLE covariance kernel

    kle_nvars : integer
        The number of KLE modes

    true_sample : np.ndarray (2)
        The true location of the source used to generate the observations
        used in the likelihood function

    orders : np.ndarray (2)
        The degrees of the collocation polynomials in each mesh dimension

    nobs : integer
         The number of observations :math:`L`

    obs_indices : np.ndarray (nobs)
         The indices of the collocation mesh at which observations are
         collected. If not specified the indices will be chosen randomly
         ensuring that no indices associated with boundary segments are
         selected.

    noise_stdev : float
        The standard deviation :math:`\sigma` of the observational noise

    max_eval_concurrency : integer
        The maximum number of simulations that can be run in parallel. Should
        be no more than the maximum number of cores on the computer being used


    Returns
    -------
    benchmark : :py:class:`~pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes documented below

    negloglike : callable

        The negative log likelihood :math:`\exp(\pi(\V{y}|\rv))` with signature

        ``negloglike(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars, nsamples) and the
        output is a 2D np.ndarray with shape (nsamples, 1).

    variable : :py:class:`~pyapprox.variables.joint.IndependentMarginalsVariable`
        Object containing information of the joint density of the inputs z
        which is the tensor product of independent and identically distributed
        uniform variables on :math:`[0,1]`.

    noiseless_obs : np.ndarray (nobs)
        The solution :math:`u(x_l)` at the :math:`L` locations
        :math:`\{x_l\}_{l=1}^L` determined by ``obs_indices``

    obs : np.ndarray (nobs)
        The noisy observations :math:`u(x_l)+\epsilon_l`

    true_sample : np.ndarray (nkle_vars)
        The KLE coefficients used to generate the noisy observations

    obs_indices : np.ndarray (nobs)
         The indices of the collocation mesh at which observations are
         collected. If not specified the indices will be chosen randomly
         ensuring that no indices associated with boundary segments are
         selected.

    obs_fun : callable

        The function used to generate the noisless observations with signature

        ``obs_fun(z) -> np.ndarray``

        where ``z`` is a 2D np.ndarray with shape (nvars, nsamples) and the
        output is a 2D np.ndarray with shape (nsamples, nobs).

    KLE : :py:class:`~pyapprox.pde.karhunen_loeve_expansion.MeshKLE`
        KLE object containing the attributes needed to evaluate the KLE

    Examples
    --------
    >>> from pyapprox_dev.benchmarks.benchmarks import setup_benchmark
    >>> benchmark = setup_benchmark('advection_diffusion_kle_inversion', nvars=2)
    >>> print(benchmark.keys())
    dict_keys(['fun', 'variable'])
    """

    (base_model, variable, true_sample, noiseless_obs, obs, obs_indices,
     obs_model, kle, mesh) = _setup_inverse_advection_diffusion_benchmark(
         source_amp, source_width, source_loc, nobs, noise_stdev,
         kle_length_scale, kle_stdev, kle_nvars, orders, obs_indices)
    # add wrapper to allow execution times to be captured
    timer_model = TimerModel(base_model, base_model)
    pool_model = PoolModel(
        timer_model, max_eval_concurrency, base_model=base_model)

    # add wrapper that tracks execution times.
    model = WorkTrackingModel(pool_model, base_model, enforce_timer_model=False)

    attributes = {'negloglike': model, 'variable': variable,
                  "noiseless_obs": noiseless_obs, "obs": obs,
                  "true_sample": true_sample, "obs_indices": obs_indices,
                  "obs_fun": obs_model, "KLE": kle, "mesh": mesh}
    return Benchmark(attributes)


_benchmarks = {
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
    'multioutput_model_ensemble': setup_multioutput_model_ensemble,
    'short_column_ensemble': setup_short_column_ensemble,
    "parameterized_nonlinear_model": setup_parameterized_nonlinear_model}


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
    benchmark : :py:class:`~pyapprox.benchmarks.Benchmark`
       Object containing the benchmark attributes

    The benchmark object must contain at least the following two attributes

    fun : callable
        A function with signature

        fun(samples) -> np.ndarray(nsamples, nqoi)

        where samples : np.ndarray(nvars, nsamples)

    variable : :py:class:`~pyapprox.variables.JointVariable`
        Class containing information about each of the nvars inputs to fun

    """
    if name not in _benchmarks:
        msg = f'Benchmark "{name}" not found.\n Available benchmarks are:\n'
        for key in _benchmarks.keys():
            msg += f"\t{key}\n"
        raise ValueError(msg)

    return _benchmarks[name](**kwargs)


def list_benchmarks():
    """
    List the names of all available benchmarks

    Returns
    -------
    names : list
        A list of the name of each benchmark implemented in PyApprox
    """
    return list(_benchmarks.keys())
