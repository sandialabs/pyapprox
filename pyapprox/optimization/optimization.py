import numpy as np
from functools import partial
from scipy import stats
from pyapprox.util.utilities import get_all_sample_combinations
from pyapprox.interface.wrappers import ActiveSetVariableModel
from pyapprox.optimization.cvar_regression import smooth_max_function_first_derivative,\
    smooth_max_function_second_derivative


def eval_function_at_multiple_design_and_random_samples(function, uq_samples, design_samples):
    """
    for functions which only take 1d arrays for uq_samples and design_samples
    loop over all combinations and evaluate function at each combination

    design_samples vary slowest and uq_samples vary fastest

    Let design samples = [[1,2],[2,3]]
    uq_samples = [[0, 0, 0],[0, 1, 2]]
    Then samples will be

    ([1, 2], [0, 0, 0])
    ([1, 2], [0, 1, 2])
    ([3, 4], [0, 0, 0])
    ([3, 4], [0, 1, 2])

    function(uq_samples,design_samples)
    """
    vals = []
    # put design samples first so that samples iterates over uq_samples fastest
    samples = get_all_sample_combinations(design_samples, uq_samples)
    for xx, zz in zip(
            samples[:design_samples.shape[0]].T,
            samples[design_samples.shape[0]:].T):
        # flip xx,zz because functions assumed to take uq_samples then
        # design_samples
        vals.append(function(zz, xx))
    return np.asarray(vals)


def eval_mc_based_jacobian_at_multiple_design_samples(grad, stat_func,
                                                      uq_samples, design_samples):
    """
    Alternatively I could use
    jacobian = [np.mean([constraint_grad_single(z,x) for z in zz.T],axis=0) for x in xx.T]
    But I think this implementation will allow better use of concurent evaluations in the
    future. For example eval_function_at_multiple_design_and_random_samples could
    utilize an asynchronous call over all the sample combinations

    TODO combine uq_samples and design samples into one matrix and assume functions
    always take a single matrix and not two matrices
    """
    grads = eval_function_at_multiple_design_and_random_samples(
        grad, uq_samples, design_samples)

    ndesign_samples = design_samples.shape[1]
    nuq_samples = uq_samples.shape[1]
    jacobian = np.array(
        [stat_func(grads[ii*nuq_samples:(ii+1)*nuq_samples])
         for ii in range(ndesign_samples)])
    return jacobian


def check_inputs(uq_samples, design_samples):
    if design_samples.ndim == 1:
        design_samples = design_samples[:, np.newaxis]
    if uq_samples is not None and uq_samples.ndim == 1:
        uq_samples = design_samples[:, np.newaxis]
    if (uq_samples is not None and
            (design_samples.shape[1] > 1 and uq_samples.shape[1] > 1)):
        assert design_samples.shape[1] == uq_samples.shape[1]
    return uq_samples, design_samples


def deterministic_lower_bound_constraint(constraint_function, lower_bound,
                                         uq_samples, design_samples):
    uq_samples, design_samples = check_inputs(uq_samples, design_samples)
    assert design_samples.shape[1] == 1
    val = lower_bound-constraint_function(uq_samples, design_samples)
    # scipy minimize enforces constraints are non-negative so use negative here
    # to enforce upper bound
    return -val


def variance_lower_bound_constraint(constraint_function, lower_bound, uq_samples,
                                    design_samples):
    uq_samples, design_samples = check_inputs(uq_samples, design_samples)
    assert design_samples.shape[1] == 1
    # scipy minimize enforces constraints are non-negative
    vals = constraint_function(uq_samples, design_samples)
    val = lower_bound-np.std(vals)**2
    # scipy minimize enforces constraints are non-negative so use negative here
    # to enforce upper bound
    return -val


def mean_lower_bound_constraint(constraint_function, lower_bound, uq_samples,
                                design_samples):
    uq_samples, design_samples = check_inputs(uq_samples, design_samples)
    assert design_samples.shape[1] == 1
    # scipy minimize enforces constraints are non-negative
    vals = constraint_function(uq_samples, design_samples)
    val = lower_bound-np.mean(vals)**2
    # scipy minimize enforces constraints are non-negative so use negative here
    # to enforce upper bound
    return -val


def mean_lower_bound_constraint_jacobian(constraint_function_jacobian, uq_samples,
                                         design_samples):
    uq_samples, design_samples = check_inputs(uq_samples, design_samples)
    assert design_samples.shape[1] == 1
    # scipy minimize enforces constraints are non-negative
    vals = constraint_function_jacobian(uq_samples, design_samples)
    val = -np.mean(vals)**2
    # scipy minimize enforces constraints are non-negative so use negative here
    # to enforce upper bound
    return -val


def quantile_lower_bound_constraint(constraint_function, quantile, lower_bound,
                                    uq_samples, design_samples):
    uq_samples, design_samples = check_inputs(uq_samples, design_samples)
    assert design_samples.shape[1] == 1
    vals = constraint_function(uq_samples, design_samples)
    val = (lower_bound-stats.mstats.mquantiles(vals, prob=[quantile]))
    # scipy minimize enforces constraints are non-negative so use negative here
    # to enforce lower bound
    return -val

# from pyapprox.optimization.cvar_regression import smooth_conditional_value_at_risk, \
#     conditional_value_at_risk
# def cvar_lower_bound_constraint(constraint_function,quantile,lower_bound,eps,
#                                 uq_samples,design_samples):
#     uq_samples,design_samples = check_inputs(uq_samples,design_samples)
#     assert design_samples.shape[1]==1
#     vals = constraint_function(uq_samples,design_samples)
#     # -vals because we want to minimize lower tail
#     val = (lower_bound-smooth_conditional_value_at_risk(0,eps,quantile,-vals))
#     #val = (lower_bound-conditional_value_at_risk(-vals,quantile))
#     return val


class MultipleConstraints(object):
    def __init__(self, constraints):
        self.constraints = constraints

    def __call__(self, design_sample, constraint_idx=None):
        if constraint_idx is None:
            constraint_idx = np.arange(len(self.constraints))
        nconstraints = len(constraint_idx)
        vals = np.empty(nconstraints)
        for ii, jj in enumerate(constraint_idx):
            vals[ii] = self.constraints[jj](design_sample)
        return vals


class MCStatisticConstraint(object):
    def __init__(self, constraint_function, generate_samples, info):
        self.constraint_function = constraint_function
        self.generate_samples = generate_samples
        self.info = info

    def __call__(self, design_samples):
        uq_samples = self.generate_samples()
        constraint_type = self.info['type']
        if constraint_type == 'quantile':
            quantile = self.info['quantile']
            lower_bound = self.info['lower_bound']
            return quantile_lower_bound_constraint(
                self.constraint_function, quantile, lower_bound,
                uq_samples, design_samples)
        elif constraint_type == 'cvar':
            quantile = self.info['quantile']
            lower_bound = self.info['lower_bound']
            eps = self.info['smoothing_eps']
            return cvar_lower_bound_constraint(
                constraint_functions[ii], quantile, lower_bound, eps,
                uq_samples, design_samples)
        elif constraint_type == 'var':
            var_lower_bound = self.info['lower_bound']
            return variance_lower_bound_constraint(
                constraint_functions[ii], lower_bound, uq_samples, design_samples)
        else:
            raise Exception(
                'constraint type (%s) not implemented' % constraint_type[ii])


class DeterministicConstraint(object):
    def __init__(self, constraint_function, info):
        self.constraint_function = constraint_function
        self.info = info

    def __call__(self, design_samples):
        lower_bound = self.info['lower_bound']
        uq_nominal_sample = self.info['uq_nominal_sample']
        return deterministic_lower_bound_constraint(
            self.constraint_function, lower_bound, uq_nominal_sample,
            design_samples)


def setup_inequality_constraints(constraint_functions, constraints_info,
                                 uq_samples):
    constraints = []
    for ii in range(len(constraint_functions)):
        info = constraints_info[ii]
        constraint_type = info['type']
        if constraint_type == 'quantile':
            quantile = info['quantile']
            quantile_lower_bound = info['quantile_lower_bound']
            ineq_cons_fun = partial(
                quantile_lower_bound_constraint, constraint_functions[ii],
                quantile, quantile_lower_bound, uq_samples)
        elif constraint_type == 'cvar':
            quantile = info['quantile']
            quantile_lower_bound = info['cvar_lower_bound']
            eps = info['smoothing_eps']
            ineq_cons_fun = partial(
                cvar_lower_bound_constraint, constraint_functions[ii],
                quantile, quantile_lower_bound, eps, uq_samples)
        elif constraint_type == 'var':
            var_lower_bound = info['var_lower_bound']
            ineq_cons_fun = partial(
                variance_lower_bound_constraint, constraint_functions[ii],
                var_lower_bound, uq_samples)
        elif constraint_type == 'deterministic':
            lower_bound = info['lower_bound']
            ineq_cons_fun = partial(
                deterministic_lower_bound_constraint, constraint_functions[ii],
                lower_bound, uq_samples)
        else:
            raise Exception(
                'constraint type (%s) not implemented' % constraint_type[ii])
        ineq_cons = {'type': 'ineq', 'fun': ineq_cons_fun}
        constraints.append(ineq_cons)
    return constraints


def run_design(objective, init_design_sample,
               constraints, bounds, optim_options):

    opt_history = [init_design_sample[:, 0]]

    def callback(xk):
        opt_history.append(xk)
        # print(objective(xk))
        #print([constraints[ii]['fun'](xk) for ii in [0,1]])

    # opt_method = 'SLSQP'
    # res = minimize(
    #     objective, init_design_sample[:,0], method=opt_method, jac=None,
    #     constraints=constraints,
    #     options=optim_options,bounds=bounds,callback=callback)

    from scipy.optimize import fmin_slsqp
    res = fmin_slsqp(objective, init_design_sample[:, 0], f_ieqcons=constraints,
                     bounds=bounds, callback=callback, full_output=True)  # , **optim_options)

    class result():
        def __init__(self, x, fun):
            self.x = np.atleast_1d(x)
            self.fun = fun
    res = result(res[0], res[1])

    opt_history = (np.array(opt_history)).T
    return res, opt_history


def plot_optimization_history(obj_function, constraints, uq_samples, opt_history,
                              plot_limits):

    # fig,axs=plot_optimization_objective_and_constraints_2D(
    #     [constraints[ii]['fun'] for ii in range(len(constraints))],
    #     partial(obj_function,uq_samples[:,0]),plot_limits)

    fig, axs = plot_optimization_objective_and_constraints_2D(
        constraints, partial(obj_function, uq_samples[:, 0]), plot_limits)
    # objective can only be evaluated at one uq_sample thus use of
    # uq_samples[:,0]

    for ii in range(len(axs)):
        axs[ii].plot(opt_history[0, :], opt_history[1, :], 'ko')
        for jj, txt in enumerate(range(opt_history.shape[1])):
            axs[ii].annotate(
                '%d' % txt, (opt_history[0, jj], opt_history[1, jj]))
    return fig, axs

# def plot_optimization_objective_and_constraints_2D(
#        constraint_functions,objective,plot_limits):


def plot_optimization_objective_and_constraints_2D(
        constraints, objective, plot_limits):
    from pyapprox.util.visualization import get_meshgrid_function_data
    num_pts_1d = 100
    num_contour_levels = 30
    fig, axs = plt.subplots(1, 3, figsize=(3*8, 6))
    # for ii in range(len(constraint_functions)+1):
    for ii in range(len(constraints.constraints)+1):

        # if ii==len(constraint_functions):
        if ii == len(constraints.constraints):
            function = objective
        else:
            # def function(design_samples):
            #     vals = np.empty((design_samples.shape[1]))
            #     for jj in range(design_samples.shape[1]):
            #         vals[jj]=constraint_functions[ii](design_samples[:,jj])
            #     return vals
            def function(design_samples):
                vals = np.empty((design_samples.shape[1]))
                for jj in range(design_samples.shape[1]):
                    vals[jj] = constraints(design_samples[:, jj], [ii])
                return vals

        X, Y, Z = get_meshgrid_function_data(
            function, plot_limits, num_pts_1d)
        norm = None
        cset = axs[ii].contourf(
            X, Y, Z, levels=np.linspace(Z.min(), Z.max(), num_contour_levels),
            cmap=mpl.cm.coolwarm,
            norm=norm)
        # for kk in range(len(constraint_functions)):
        for kk in range(len(constraints.constraints)):
            if ii == kk:
                ls = '-'
            else:
                ls = '--'
            axs[kk].contour(X, Y, Z, levels=[0], colors='k', linestyles=ls)
        plt.colorbar(cset, ax=axs[ii])

    return fig, axs


def plot_constraint_pdfs(constraint_functions, uq_samples, design_sample,
                         fig_pdf=None, axs_pdf=None, label=None, color=None):
    colors = ['b', 'gray']
    nconstraints = len(constraint_functions)
    if axs_pdf is None:
        fig_pdf, axs_pdf = plt.subplots(
            1, nconstraints, figsize=(nconstraints*8, 6))
    for ii in range(nconstraints):
        # evaluate constraint function at each of the uq samples
        constraint_function_vals = constraint_functions[ii](
            uq_samples, design_sample)

        constraint_kde = stats.gaussian_kde(constraint_function_vals)
        yy = np.linspace(constraint_function_vals.min(),
                         constraint_function_vals.max(), 101)

        axs_pdf[ii].fill_between(yy, 0, constraint_kde(yy), alpha=0.5, label=label,
                                 color=color)
        axs_pdf[ii].axvline(0, color='k')
        # axs_pdf[ii].axvline(constraints[ii]['fun'](design_sample),color='r')
    return fig_pdf, axs_pdf


def plot_constraint_cdfs(constraints, constraint_functions, uq_samples,
                         design_sample, quantile, fig_cdf, axs_cdf=None, label=None,
                         color=None):
    nconstraints = len(constraint_functions)
    if axs_cdf is None:
        fig_cdf, axs_cdf = plt.subplots(
            1, nconstraints, figsize=(nconstraints*8, 6))

    for ii in range(nconstraints):
        constraint_function_vals = constraint_functions[ii](
            uq_samples, design_sample)

        cvar = (conditional_value_at_risk(-constraint_function_vals, 0.9))
        cvars = (smooth_conditional_value_at_risk(
            0, 1e-3, 0.9, -constraint_function_vals))
        print('cvar', cvar)
        print('cvars', cvars)
        #constraint_val = constraints[ii]['fun'](design_sample)
        constraint_val = constraints(design_sample, [ii])
        constraint_function_vals.sort()
        cdf_vals = np.linspace(0, 1, constraint_function_vals.shape[0]+1)[1:]
        axs_cdf[ii].plot(constraint_function_vals, cdf_vals, label=label,
                         color=color)
        #I = np.where(constraint_function_vals<=constraint_val)[0]
        I = np.where(constraint_function_vals <= 0)[0]
        axs_cdf[ii].fill_between(
            constraint_function_vals[I], 0, cdf_vals[I], alpha=0.5, color=color)
        axs_cdf[ii].axvline(0, color='k')
        J = np.where(constraint_function_vals <= 0)[0]
        #print (J.shape[0]/float(constraint_function_vals.shape[0]),'p failure',constraint_val,J.shape[0])
        # Compute the constraint value. This combines constraint_function_vals
        # into a scalar value
        # axs_cdf[ii].axvline(constraint_val,color='r')
        # axs_cdf[ii].plot(
        #    np.linspace(constraint_function_vals[0],constraint_val,101),
        #    quantile*np.ones(101),'-r')
        #axs_cdf[ii].set_yticks(list(axs_cdf[ii].get_yticks()) + [quantile])
        axs_cdf[ii].set_ylim(0, 1.05)
        axs_cdf[ii].set_xlim(
            constraint_function_vals[0], constraint_function_vals[-1])
    return fig_cdf, axs_cdf


def expectation_fun(values, weights):
    assert values.shape[0] % weights.shape[0] == 0
    nqoi = values.shape[0]//weights.shape[0]
    nsamples = values.shape[0]//nqoi
    assert nqoi == 1
    fun_vals = (values.T.dot(weights)).T
    return fun_vals


def expectation_jac(jac_values, weights):
    assert jac_values.shape[0] % weights.shape[0] == 0
    nqoi = jac_values.shape[0]//weights.shape[0]
    nsamples = jac_values.shape[0]//nqoi
    num_vars = jac_values.shape[1]
    assert nqoi == 1
    jac = (jac_values.T.dot(weights)).T
    return jac


def smooth_prob_failure_fun(smoother_type, eps, tol, values, weights):
    assert values.shape[0] % weights.shape[0] == 0
    nqoi = values.shape[0]//weights.shape[0]
    assert nqoi == 1
    # nsamples = values.shape[0]//nqoi
    heaviside_vals = smooth_max_function_first_derivative(
        smoother_type, eps, values-tol)
    fun_vals = (heaviside_vals.dot(weights)).T
    # print(fun_vals.shape)
    return fun_vals


def smooth_prob_failure_jac(smoother_type, eps, tol, jac_values, weights):
    assert jac_values.shape[0] % weights.shape[0] == 0
    nqoi = jac_values.shape[0]//weights.shape[0]
    assert nqoi == 1
    # nsamples = jac_values.shape[0]//nqoi
    # num_vars = jac_values.shape[1]
    grad_heaviside_vals = smooth_max_function_second_derivative(
        smoother_type, eps, jac_values-tol)
    jac = (grad_heaviside_vals*jac_values).T.dot(weights)[np.newaxis, :]
    print(jac_values.max(axis=0), 'm', eps)

    return jac


def generate_monte_carlo_quadrature_data(
        generate_random_samples, num_vars, design_var_indices, fun, seed=None):
    if seed is not None:
        np.random.seed(seed)
    samples = generate_random_samples()
    weights = np.ones(samples.shape[1])/samples.shape[1]
    values = fun(samples)
    return samples, weights, values


class StatisticalConstraint(object):
    """
    Notes
    -----
    TODO ensure the following.

    This class unifies the jac=True and callable(jac)=True interfaces.
    The interface is used for passing to optimizers that need the fun and jac functions
    to be separate. This is often good practice as it avoids computing
    jac when only fun is required.
    If jac=True the jacobian is stored and returned when self.jac is called
    """

    def __init__(self, fun, jac, stats_fun, stats_jac, num_vars,
                 design_var_indices, generate_sample_data, bound=None,
                 upper_bound=True, isobjective=False):
        self.fun, self.jac, self.stats_fun = fun, jac, stats_fun
        self.stats_jac = stats_jac
        self.num_vars = num_vars
        self.design_var_indices = design_var_indices
        self.random_var_indices = np.delete(
            np.arange(self.num_vars), self.design_var_indices)
        self.generate_sample_data = generate_sample_data
        self.bound = bound
        self.upper_bound = upper_bound
        self.isobjective = isobjective

        self.design_sample = None
        self.jac_values = None
        self.samples = None

        if self.stats_jac is not None and self.jac is None:
            msg = 'stats_jac requries jac to be defined'
            raise Exception(msg)
        if self.jac is not None and self.stats_jac is None:
            msg = 'jac will be ignored because stats_jac was not defined'
            raise Exception(msg)

    def generate_shared_data(self, design_sample):
        self.design_sample = design_sample.copy()

        fun = ActiveSetVariableModel(self.fun, self.num_vars, design_sample,
                                     self.random_var_indices)
        data = self.generate_sample_data(fun)
        self.samples, self.weights, self.fun_values = data[:3]
        assert self.samples.shape[0] ==\
            self.num_vars-self.design_var_indices.shape[0]
        assert self.samples.shape[1] == self.weights.shape[0]
        #assert self.samples.shape[1]==self.fun_values.shape[0]
        if not callable(self.jac) and self.jac:
            # consider whether to support self.jac=True. It seems appealing
            # if using gradients from adjoint PDE simulation which requires
            # data used to compute function values and thus better to do at the
            # time the function values are obtained. Challenge is defining the
            # correct output interface and only computing gradients if self.jac
            # has been called and not if self.__call__ is called.
            raise Exception("Not yet implemented")
            self.jac_values = data[3]

    def __call__(self, design_sample):
        if design_sample.ndim == 1:
            design_sample = design_sample[:, np.newaxis]
        self.generate_shared_data(design_sample)
        # nsamples = self.weights.shape[0]
        nqoi = self.fun_values.shape[1]
        # print(self.fun_values)
        values = np.empty((nqoi))
        for ii in range(nqoi):
            values[ii] = self.stats_fun(
                self.fun_values[:, ii:ii+1], self.weights)

            # print('b',np.where(self.fun_values[:,ii:ii+1]>0)[0].shape[0]/nsamples)
            # print('c',values[ii])
        # print(self.fun_values.min(),self.fun_values.max())
        if self.bound is not None:
            values = values-self.bound
            if self.upper_bound:
                values *= -1
        if self.isobjective:
            values = values[0]
        return values

    def jacobian(self, design_sample):
        if design_sample.ndim == 1:
            design_sample = design_sample[:, np.newaxis]
        if (np.array_equal(design_sample, self.design_sample) and
                self.jac_values is not None):
            jac_values = self.jac_values
        else:
            jac = ActiveSetVariableModel(
                self.jac, self.num_vars, self.samples, self.design_var_indices)
            jac_values = jac(design_sample)
        nsamples = self.weights.shape[0]
        nqoi = self.fun_values.shape[1]
        nvars = jac_values.shape[1]
        constraint_jac = np.empty((nqoi, nvars))
        for ii in range(nqoi):
            constraint_jac[ii] = self.stats_jac(
                jac_values[ii*nsamples:(ii+1)*nsamples, :], self.weights)
        if self.bound is not None and self.upper_bound:
            constraint_jac *= -1
        return constraint_jac.squeeze()


class PyapproxFunctionAsScipyMinimizeObjective(object):
    def __init__(self, fun):
        self.fun = fun

    def __call__(self, scipy_sample):
        assert scipy_sample.ndim == 1
        data = self.fun(scipy_sample[:, np.newaxis])
        if not np.isscalar(data):
            assert len(data) == 2
            val = data[0]
            assert np.isscalar(val)
            assert data[1].ndim == 2 and data[1].shape[0] == 1
            jac = data[1][0, :]
            return val, jac
        return data


class ScipyMinimizeObjectiveAsPyapproxFunction(object):
    def __init__(self, fun):
        self.fun = fun

    def __call__(self, pyapprox_sample):
        assert pyapprox_sample.ndim == 2 and pyapprox_sample.shape[1] == 1
        data = self.fun(pyapprox_sample[:, 0])
        if not np.isscalar(data):
            assert len(data) == 2
            val = data[0]
            assert np.isscalar(val)
            assert data[1].ndim == 2 and data[1].shape[0] == 1
            jac = data[1][0, :]
            return val, jac
        return data


class ScipyMinimizeObjectiveJacAsPyapproxJac(object):
    def __init__(self, jac):
        self.jac = jac

    def __call__(self, pyapprox_sample):
        assert pyapprox_sample.ndim == 2 and pyapprox_sample.shape[1] == 1
        grad = self.jac(pyapprox_sample[:, 0])
        return grad[np.newaxis, :]
