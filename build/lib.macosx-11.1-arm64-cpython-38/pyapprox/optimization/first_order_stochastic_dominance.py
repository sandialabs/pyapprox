import numpy as np
from functools import partial
from scipy.optimize import NonlinearConstraint, Bounds

from pyapprox.optimization.pya_minimize import pyapprox_minimize, has_ROL
from pyapprox.util.pya_numba import njit


def smooth_max_function_log(eps, shift, x):
    x = x+shift
    x_div_eps = x/eps
    # vals = (x + eps*np.log(1+np.exp(-x_div_eps)))
    # avoid overflow
    vals = np.zeros_like(x)
    II = np.where((x_div_eps < 1e2) & (x_div_eps > -1e2))
    vals[II] = x[II]+eps*np.log(1+np.exp(-x_div_eps[II]))
    J = np.where(x_div_eps >= 1e2)
    vals[J] = x[J]
    assert np.all(np.isfinite(vals))
    return vals


def smooth_max_function_first_derivative_log(eps, shift, x):
    x = x+shift
    x_div_eps = x/eps
    # vals = 1./(1+np.exp(-x_div_eps+shift))
    # Avoid overflow.
    II = np.where((x_div_eps < 1e2) & (x_div_eps > -1e2))
    vals = np.zeros(x.shape)
    vals[II] = 1./(1+np.exp(-x_div_eps[II]-shift/eps))
    vals[x_div_eps >= 1e2] = 1.
    assert np.all(np.isfinite(vals))
    return vals


def smooth_max_function_second_derivative_log(eps, shift, x):
    x = x+shift
    x_div_eps = x/eps
    vals = np.zeros(x.shape)
    # Avoid overflow.
    II = np.where((x_div_eps < 1e2) & (x_div_eps > -1e2))
    vals[II] = np.exp(x_div_eps[II]+shift/eps)/(
        eps*(np.exp(x_div_eps[II]+shift/eps)+1)**2)
    assert np.all(np.isfinite(vals))
    return vals


def smooth_max_function_third_derivative_log(eps, shift, x):
    x = x+shift
    x_div_eps = x/eps
    vals = np.zeros_like(x)
    # Avoid overflow.
    II = np.where((x_div_eps < 1e2) & (x_div_eps > -1e2))
    # vals[II] = np.exp(-x_div_eps[II])*(np.exp(-x_div_eps[II])-1)/(
    #    eps**2*(1+np.exp(-x_div_eps[II]))**3)
    vals[II] = np.exp(x_div_eps[II]+shift/eps)/(
        eps**2*(1+np.exp(x_div_eps[II]+shift/eps))**2)
    vals[II] -= 2*np.exp(x_div_eps[II]+shift/eps)**2/(
        eps**2*(1+np.exp(x_div_eps[II]+shift/eps))**3)
    return vals


def smooth_left_heaviside_function_log(eps, shift, x):
    return smooth_max_function_first_derivative_log(eps, -shift, -x)


def smooth_left_heaviside_function_first_derivative_log(eps, shift, x):
    return -smooth_max_function_second_derivative_log(eps, -shift, -x)


def smooth_left_heaviside_function_second_derivative_log(eps, shift, x):
    return smooth_max_function_third_derivative_log(eps, -shift, -x)


@njit(cache=True)
def numba_smooth_left_heaviside_function_quartic(eps, shift, x):
    assert shift == 0  # need to employ chain rule to accound for shift
    x = x+shift
    vals = np.ones_like(x)
    for ii in range(x.shape[0]):
        for jj in range(x.shape[1]):
            if (x[ii, jj] < 0 and x[ii, jj] > -eps):
                vals[ii, jj] = 6*(-x[ii, jj]/eps)**2-8*(-x[ii, jj]/eps)**3 +\
                    3*(-x[ii, jj]/eps)**4
            elif (x[ii, jj] >= 0):
                vals[ii, jj] = 0
    return vals


@njit(cache=True)
def numba_smooth_left_heaviside_function_first_derivative_quartic(
        eps, shift, x):
    assert shift == 0  # need to employ chain rule to accound for shift
    x = x+shift
    vals = np.zeros_like(x)
    for ii in range(x.shape[0]):
        for jj in range(x.shape[1]):
            if (x[ii, jj] < 0 and x[ii, jj] > -eps):
                vals[ii, jj] = 12*x[ii, jj]*(eps+x[ii, jj])**2/eps**4
    return vals


@njit(cache=True)
def numba_smooth_left_heaviside_function_second_derivative_quartic(
        eps, shift, x):
    assert shift == 0  # need to employ chain rule to accound for shift
    x = x+shift
    vals = np.zeros_like(x)
    for ii in range(x.shape[0]):
        for jj in range(x.shape[1]):
            if (x[ii, jj] < 0 and x[ii, jj] > -eps):
                vals[ii, jj] = 12*(eps**2+4*eps*x[ii, jj] +
                                   3*x[ii, jj]**2)/eps**4
    return vals


@njit(cache=True)
def numba_smooth_left_heaviside_function_quintic(eps, shift, x):
    assert shift == 0  # need to employ chain rule to accound for shift
    x = x+shift
    vals = np.ones_like(x)
    c3, c4, c5 = 10, -15, 6
    for ii in range(x.shape[0]):
        for jj in range(x.shape[1]):
            if (x[ii, jj] < 0 and x[ii, jj] > -eps):
                xe = -x[ii, jj]/eps
                vals[ii, jj] = c3*xe**3 + c4*xe**4 + c5*xe**5
            elif (x[ii, jj] >= 0):
                vals[ii, jj] = 0
    return vals


@njit(cache=True)
def numba_smooth_left_heaviside_function_first_derivative_quintic(
        eps, shift, x):
    assert shift == 0  # need to employ chain rule to accound for shift
    x = x+shift
    vals = np.zeros_like(x)
    c3, c4, c5 = 10, -15, 6
    for ii in range(x.shape[0]):
        for jj in range(x.shape[1]):
            if (x[ii, jj] < 0 and x[ii, jj] > -eps):
                xe = -x[ii, jj]/eps
                vals[ii, jj] = -(3*c3*xe**2 + 4*c4*xe**3 + 5*c5*xe**4)/eps
    return vals


@njit(cache=True)
def numba_smooth_left_heaviside_function_second_derivative_quintic(
        eps, shift, x):
    assert shift == 0  # need to employ chain rule to accound for shift
    x = x+shift
    vals = np.zeros_like(x)
    c3, c4, c5 = 10, -15, 6
    for ii in range(x.shape[0]):
        for jj in range(x.shape[1]):
            if (x[ii, jj] < 0 and x[ii, jj] > -eps):
                xe = -x[ii, jj]/eps
                vals[ii, jj] = (6*c3*xe + 12*c4*xe**2 + 20*c5*xe**3)/eps**2
    return vals


def compute_quartic_spline_of_right_heaviside_function():
    r"""
    Get spline approximation of step function enforcing all derivatives 
    at 0 and 1 are zero
    """
    from scipy.interpolate import BPoly
    poly = BPoly.from_derivatives(
        [0, 1], [[0, 0, 0, 0], [1, 0, 0, 0]], orders=[4])
    poly = BPoly.from_derivatives([0, 1], [[0, 0, 0], [1, 0, 0]], orders=[3])
    def basis(x, p): return x[:, np.newaxis]**np.arange(p+1)[np.newaxis, :]

    interp_nodes = (-np.cos(np.linspace(0, np.pi, 5))+1)/2
    basis_mat = basis(interp_nodes, 4)
    coef = np.linalg.inv(basis_mat).dot(poly(interp_nodes))
    print(coef)
    xx = np.linspace(0, 1, 101)
    print(np.absolute(basis(xx, 4).dot(coef)-poly(xx)).max())
    # plt.plot(xx,basis(xx,4).dot(coef))
    # plt.plot(xx,poly(xx))

    eps = 0.1
    a, b = 0, eps
    xx = np.linspace(a, b, 101)
    plt.plot(xx, basis((xx-a)/(b-a), 4).dot(coef))
    def f(x): return 6*((xx)/eps)**2-8*((xx)/eps)**3+3*((xx)/eps)**4
    plt.plot(xx, f(xx))
    plt.show()


def linear_model_fun(basis_matrix, coef):
    return basis_matrix.dot(coef).squeeze()


def linear_model_jac(basis_matrix, coef):
    return basis_matrix


class FSDOptProblem(object):
    r"""
    Optimization problem used to solve least squares regression with first
    order dominance constraints.

    Parameters
    ----------
    values : np.ndarray (nvalues)
        The values at the training data

    fun : callable
        Function with signature

        `fun(coef) -> np.ndarray(nvalues)`

        The approximation evaluated at the training data with the coefficient 
        values `coef np.ndarray (ncoef)`

    jac : callable
        Function with signature

        `jac(coef) -> np.ndarray(ncoef)`

        The Jacobian of the approximation evaluated at the training data with 
        the coefficient values `coef np.ndarray (ncoef)`

    hessp : callable
        Function with signature

        `hessp(coef, v) -> np.ndarray(ncoef)`

        The Hessian of the approximation applied to a vector 
        `v np.ndarray (ncoef)`

    eta_indices : np.ndarray (neta)
        Indices of the training data at which constraints are enforced
        `neta <= nsamples`

    probabilities : np.ndarray(nvalues)
        The probability weight assigned to each training data. When
        sampling randomly from a probability measure the probabilities
        are all 1/nsamples

    smoother_type : string
        The name of the function used to smooth the heaviside function
        Supported types are `[quartic, quintic, log]`

    eps : float
        A parameter which controls the amount that the heaviside function
        is smoothed. As eps decreases the smooth approximation converges to
        the heaviside function but the derivatives of the approximation
        become more difficult to compute

    ncoef : integer
        The number of unknowns of the approximation `fun`
    """

    def __init__(self, values, fun, jac, hessp, eta_indices, probabilities,
                 smoother_type, eps, ncoef):
        self.values = values
        self.values = self.values.squeeze()
        assert self.values.ndim == 1
        self.ncoef = ncoef
        self.fun = fun
        self.jac = jac
        self.hessp = hessp
        self.eta_indices = eta_indices
        if self.eta_indices is None:
            self.eta_indices = np.arange(self.values.shape[0])
        self.probabilities = probabilities
        self.set_smoother(smoother_type, eps)

    def set_smoother(self, smoother_type, eps):
        self.smoother_type = smoother_type
        self.eps = eps

        smoothers = {}
        smoothers['quartic'] = [
            numba_smooth_left_heaviside_function_quartic,
            numba_smooth_left_heaviside_function_first_derivative_quartic,
            numba_smooth_left_heaviside_function_second_derivative_quartic]
        smoothers['quintic'] = [
            numba_smooth_left_heaviside_function_quintic,
            numba_smooth_left_heaviside_function_first_derivative_quintic,
            numba_smooth_left_heaviside_function_second_derivative_quintic]
        smoothers['log'] = [
            smooth_left_heaviside_function_log,
            smooth_left_heaviside_function_first_derivative_log,
            smooth_left_heaviside_function_second_derivative_log]

        if smoother_type not in smoothers:
            raise Exception(f'Smoother {smoother_type} not found')

        self.smooth_fun, self.smooth_jac, self.smooth_hess = \
            [partial(f, self.eps, -self.eps) for f in smoothers[smoother_type]]

    def objective_fun(self, x):
        __approx_values = self.fun(x)
        assert __approx_values.shape == self.values.shape
        val = 0.5*np.sum(self.probabilities*(self.values-__approx_values)**2)
        assert np.isfinite(val)
        return val

    def objective_jac(self, x):
        coef = x[:self.ncoef]
        grad = np.zeros((x.shape[0]))
        # TODO reuse __approx_values from objective_fun
        __approx_values = self.fun(coef)
        __jac = self.jac(coef)
        assert __jac.shape == (self.values.shape[0], self.ncoef)
        grad[:self.ncoef] = - \
            __jac.T.dot(self.probabilities*(self.values-__approx_values))
        assert np.all(np.isfinite(grad))
        return grad

    def objective_hessp(self, x, v):
        # TODO reuse __jac from objective_grad
        coef = x[:self.ncoef]
        res = np.zeros((x.shape[0]))
        __jac = self.jac(coef)
        res = __jac.T.dot((self.probabilities[:, None]*__jac).dot(v))
        if self.hessp is None:
            return res
        else:
            msg = 'support for nonzero self.hess is not currently supported'
            raise Exception(msg)

    def constraint_fun(self, x):
        r"""
        Compute the constraints. The nth row of the Jacobian is
        the derivative of the nth constraint :math:`c_n(x)`.
        Let :math:`h(z)` be the smooth heaviside function and :math:`f(x)` the
        function approximation evaluated
        at the training samples and coeficients :math:`x`, then

        .. math::

           c_n(x) =
           \sum_{m=1}^M h(f(x_m)-f(x_n))- h(y_m-f(x_n))\le 0

        Parameters
        ----------
        x : np.ndarray (ncoef)
            The unknowns

        Returns
        -------
        jac : np.ndarray(nconstraints, ncoef)
            The Jacobian of the constraints
        """

        coef = x[:self.ncoef]
        __approx_values = self.fun(coef)
        assert __approx_values.shape == self.values.shape
        tmp1 = self.smooth_fun(
            __approx_values[:, None]-__approx_values[None, self.eta_indices])
        tmp2 = self.smooth_fun(
            self.values[:, None]-__approx_values[None, self.eta_indices])
        val = self.probabilities.dot(tmp1-tmp2)
        assert np.all(np.isfinite(val))
        return val

    def constraint_jac(self, x):
        r"""
        Compute the Jacobian of the constraints. The nth row of the Jacobian is
        the derivative of the nth constraint :math:`c_n(x)`.
        Let :math:`h(z)` be the smooth heaviside function and :math:`f(x)` the
        function approximation evaluated
        at the training samples and coeficients :math:`x`, then

        .. math::

           \frac{\partial c_n}{\partial x} =
           \sum_{m=1}^M h^\prime(f(x_m)-f(x_n))
              \left(\nabla_x f(x_m)-\nabla_x f(x_n))\right) -
              h^\prime(y_m-f(x_n))\left(-\nabla_x f(x_n))\right)

        Parameters
        ----------
        x : np.ndarray (ncoef)
            The unknowns

        Returns
        -------
        jac : np.ndarray(nconstraints, ncoef)
            The Jacobian of the constraints
        """
        coef = x[:self.ncoef]
        __approx_values = self.fun(coef)
        __jac = self.jac(coef)
        hder1 = self.smooth_jac(
            (__approx_values[None, :] -
             __approx_values[self.eta_indices, None]))*self.probabilities
        fder1 = (__jac[None, :, :]-__jac[self.eta_indices, None, :])
        con_jac = np.sum(hder1[:, :, None]*fder1, axis=1)
        hder2 = self.smooth_jac(
            (self.values[None, :]-__approx_values[self.eta_indices, None])) *\
            self.probabilities
        fder2 = 0*__jac[None, :, :]-__jac[self.eta_indices, None, :]
        con_jac -= np.sum(hder2[:, :, None]*fder2, axis=1)

        # con_jac2 = np.zeros((self.values.shape[0],self.ncoef))
        # for ii in range(self.values.shape[0]):
        #     tmp1 = self.smooth_jac(
        #         ((__approx_values-__approx_values[None, ii]))[:, None])
        #     tmp2 = (__jac[:, :] - __jac[ii, :])
        #     con_jac2[ii] = (
        #         self.probabilities[:,None]*tmp1).T.dot(tmp2).squeeze()
        # for ii in range(self.values.shape[0]):
        #     tmp3 = self.smooth_jac(
        #         ((self.values-__approx_values[None, ii]))[:, None])
        #     tmp4 = (__jac[:, :]*0 - __jac[ii, :])
        #     con_jac2[ii] -= (
        #         self.probabilities[:,None]*tmp3).T.dot(tmp4).squeeze()
        # assert np.allclose(con_jac,con_jac2)
        return con_jac

    def constraint_hess(self, x, lmult):
        r"""
        Compute the Hessian of the constraints applied to the Lagrange
        multipliers.

        We need to compute

        .. math:: d^2/dx^2 f(g(x))=g'(x)^2 f''(g(x))+g''(x)f'(g(x))

        and assume that  :math:`g''(x)=0 \forall x`. I.e. only linear
        approximations g(x) are implemented

        Parameters
        ----------
        x : np.ndarray (ncoef)
            The unknowns

        lmult : np.ndarray (nconstraints)
            vector of N Lagrange multipliers with

        Returns
        -------
        hess : np.ndarray(ncoef, ncoef)
            The weighted sum of the individual constraint Hessians

            .. math:: \sum_{n=1}^N H_n(x)
        """
        # only linear approximations currently supported
        assert self.hessp is None
        coef = x[:self.ncoef]
        # Todo store the next two values in and reuse
        __approx_values = self.fun(coef)
        __jac = self.jac(coef)
        hder1 = self.smooth_hess(
            (__approx_values[None, :] -
             __approx_values[self.eta_indices, None]))*self.probabilities
        hder2 = self.smooth_hess(
            (self.values[None, :]-__approx_values[self.eta_indices, None])) *\
            self.probabilities
        # Todo fder1 and fder2 can be stored when computing Jacobian and
        # reused
        fder1 = (__jac[None, :, :]-__jac[self.eta_indices, None, :])
        fder2 = 0*__jac[None, :, :]-__jac[self.eta_indices, None, :]
        hessian = np.zeros((self.ncoef, self.ncoef))
        for ii in range(lmult.shape[0]):
            hessian += lmult[ii]*(
                hder1[ii, :, None]*fder1[ii, :]).T.dot(fder1[ii, :])
            hessian -= lmult[ii]*(
                hder2[ii, :, None]*fder2[ii, :]).T.dot(fder2[ii, :])
            # for jj in range(self.ncoef):
            #     for kk in range(jj, self.ncoef):
            #         hessian[jj, kk] += lmult[ii]*np.sum(
            #             hder1[ii, :]*(fder1[ii, :, jj]*fder1[ii, :, kk]),
            #             axis=0)
            #         hessian[jj, kk] -= lmult[ii]*np.sum(
            #             hder2[ii, :]*(fder2[ii, :, jj]*fder2[ii, :, kk]),
            #             axis=0)
            #         hessian[kk, jj] = hessian[jj, kk]
        return hessian

    def get_unknowns_bounds(self):
        lb = -np.inf*np.ones(self.ncoef)
        ub = np.inf*np.ones(self.ncoef)
        bounds = Bounds(lb, ub)
        return bounds

    def get_constraint_bounds(self):
        nconstraints = self.eta_indices.shape[0]
        lb = -np.inf*np.ones(nconstraints)
        ub = np.zeros(nconstraints)
        return lb, ub

    def solve(self, x0, optim_options={}, method=None):
        r"""
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a OptimizeResult object.
            Important attributes are: x the solution array, success a Boolean
            flag indicating if the optimizer exited successfully and message
            which describes the cause of the termination.
        """
        x_grad = x0  # use rol to check_gradients
        if method is None:
            if has_ROL:
                method = 'rol-trust-constr'
            else:
                method = 'trust-constr'
        if method == 'trust_constr':
            x_grad = None

        bounds = self.get_unknowns_bounds()

        keep_feasible = True
        constr_lb, constr_ub = self.get_constraint_bounds()
        constraint = NonlinearConstraint(
            self.constraint_fun, constr_lb, constr_ub,
            jac=self.constraint_jac, hess=self.constraint_hess,
            keep_feasible=keep_feasible)
        res = pyapprox_minimize(
            self.objective_fun, x0, method=method,
            jac=self.objective_jac, hessp=self.objective_hessp,
            constraints=[constraint], bounds=bounds, options=optim_options,
            x_grad=x_grad)

        return res

    def debug_plot(self, coef, samples):
        approx_values = self.fun(coef)
        import matplotlib.pyplot as plt
        if samples.shape[0] == 1:
            fig, axs = plt.subplots(1, 2, figsize=(2*8, 6))
            xx = samples
            axs[1].plot(xx, approx_values, 'o-')
            axs[1].plot(xx, self.values, 's-')
        else:
            fig, axs = plt.subplots(1, 1, figsize=(8, 6))
            axs = [axs]
        yy = np.linspace(min(self.values.min(), approx_values.min()),
                         max(self.values.max(), approx_values.max()), 101)

        def cdf1(zz): return self.probabilities.dot(
            self.smooth_fun(approx_values[:, None]-zz[None, :]))

        def cdf2(zz): return self.probabilities.dot(
            self.smooth_fun(self.values[:, None]-zz[None, :]))

        # from pyapprox.variables.density import EmpiricalCDF
        # cdf3 = EmpiricalCDF(self.values)
        def cdf3(zz): return self.probabilities.dot(
            np.heaviside(-(self.values[:, None]-zz[None, :]), 1))

        color = next(axs[0]._get_lines.prop_cycler)['color']
        axs[0].plot(yy, cdf1(yy), '-', c=color, label='approx-approx')
        axs[0].plot(approx_values, cdf1(approx_values), 'o', c=color)
        color = next(axs[0]._get_lines.prop_cycler)['color']
        axs[0].plot(yy, cdf2(yy), '-', c=color, label='values-approx')
        axs[0].plot(approx_values, cdf2(approx_values), 's', c=color)
        #color = next(axs[0]._get_lines.prop_cycler)['color']
        #axs[0].plot(yy, cdf3(yy), '--', c=color)
        # print(approx_values[np.where((cdf2(approx_values[self.eta_indices])-cdf3(approx_values[self.eta_indices]))>0)])
        # print(approx_values[np.where((cdf1(approx_values[self.eta_indices])-cdf3(approx_values[self.eta_indices]))>0)])
        plt.legend()
        plt.show()
        return fig, axs


def solve_FSD_constrained_least_squares_smooth(
        samples, values, eval_basis_matrix, eta_indices=None,
        probabilities=None, eps=1e-1, optim_options={}, return_full=False,
        method='trust-constr', smoother_type='log', scale_data=False):
    r"""
    Solve first order stochastic dominance (FSD) constrained least squares

    Parameters
    ----------
    samples : np.ndarary (nvars, nsamples)
        The training samples

    values : np.ndarary (nsamples, 1)
        The function values at the training samples

    eval_basis_matrix : callable
        A function returning the basis evaluated at the set of samples
        with signature
        ``eval_basis_matrix(samples) -> np.ndarray (nsamples, nbasis)``

    eta_indices : np.ndarray (nconstraint_samples)
        Indices of the training data at which constraints are enforced
        `neta <= nsamples`

    probabilities : np.ndarray(nvalues)
        The probability weight assigned to each training data. When
        sampling randomly from a probability measure the probabilities
        are all 1/nsamples

    eps : float
        A parameter which controls the amount that the heaviside function
        is smoothed. As eps decreases the smooth approximation converges to
        the heaviside function but the derivatives of the approximation
        become more difficult to compute

    optim_options : dict
        The keyword arguments passed to the non-linear optimization used
        to solve the regression problem

    smoother_type : string
        The name of the function used to smooth the heaviside function
        Supported types are `[quartic, quintic, log]`

    return_full : boolean
        False - return regression solution
        True - return regression solution and regression optimizer object

    method : string
        The name of the non-linear solver used to solve the regresion problem

    scale_data : boolean
        False - use raw training values
        True - scale training values to have unit standard deviation

    Returns
    -------
    coef : np.narray (nbasis, 1)
        The solution to the regression problem

    opt_problem : :py:class:`FSDOptProblem`
        The object used to solve the regression problem
    """
    # only useful for 1D plotting
    # I = np.argsort(samples)[0]
    # samples = samples[:, I]
    # values = values[I]

    num_samples = samples.shape[1]
    if probabilities is None:
        probabilities = np.ones((num_samples))/num_samples

    if eta_indices is None:
        eta_indices = np.arange(num_samples)

    basis_matrix = eval_basis_matrix(samples)

    fun = partial(linear_model_fun, basis_matrix)
    jac = partial(linear_model_jac, basis_matrix)
    probabilities = np.ones((num_samples))/num_samples
    ncoef = basis_matrix.shape[1]

    if scale_data is True:
        values_std = values.std()
    else:
        values_std = 1
    scaled_values = values.copy()/values_std

    x0 = np.linalg.lstsq(basis_matrix, scaled_values, rcond=None)[0]
    residual = scaled_values-basis_matrix.dot(x0)
    x0[0] += max(0, residual.max()+eps)

    fsd_opt_problem = FSDOptProblem(
        scaled_values, fun, jac, None, eta_indices, probabilities,
        smoother_type, eps, ncoef)
    # print(fsd_opt_problem.constraint_fun(x0))

    # fsd_opt_problem.debug_plot(x0, samples[0, :])

    result = fsd_opt_problem.solve(x0, optim_options, method)
    assert result.success is True

    # fsd_opt_problem.debug_plot(result.x, samples[0, :])

    coef = result.x*values_std

    # xx = np.linspace(-1.5,2,100)
    # import matplotlib.pyplot as plt
    # print(samples)
    # print(fsd_opt_problem.init_guess.shape)
    # plt.plot(xx,eval_basis_matrix(xx[None, :]).dot(coef))
    # plt.plot(xx,eval_basis_matrix(xx[None, :]).dot(fsd_opt_problem.init_guess[:coef.shape[0]]),'--')
    # plt.show()
    # assert False

    if return_full:
        return coef, fsd_opt_problem
    else:
        return coef
