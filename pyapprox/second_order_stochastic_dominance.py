from pyapprox.first_order_stochastic_dominance import *
from pyapprox.risk_measures import compute_conditional_expectations


class SSDOptProblem(FSDOptProblem):
    r"""
    Solve disutility stochastic dominance

    -Y \ge -Y^\prime
    """
    
    def set_smoother(self, smoother_type, eps):
        self.smoother_type = smoother_type
        self.eps = eps

        smoothers = {}
        smoothers['log'] = [
            smooth_max_function_log,
            smooth_max_function_first_derivative_log,
            smooth_max_function_second_derivative_log]

        if smoother_type not in smoothers:
            raise Exception(f'Smoother {smoother_type} not found')

        self.smooth_fun, self.smooth_jac, self.smooth_hess = \
            [partial(f, self.eps, 0) for f in smoothers[smoother_type]]

    def constraint_fun(self, x):
        r"""
        Compute the constraints. The nth row of the Jacobian is
        the derivative of the nth constraint :math:`c_n(x)`. 
        Let :math:`h(z)` be the smooth max function and :math:`f(x)` the 
        function approximation evaluated
        at the training samples and coeficients :math:`x`, then

        .. math:: 

           c_n(x) = 
           \sum_{m=1}^M h(f(x_m)-f(x_n))- h(y_m-f(x_n))\ge 0

        Parameters
        ----------
        x : np.ndarray (ncoef)
            The unknowns

        Returns
        -------
        jac : np.ndarray(nconstraints, ncoef)
            The Jacobian of the constraints
        """
        # We can use the same function but smoothers and bounds
        # are different. FSD bounds are (-oo, 0) and SSD are (0, oo)
        return super().constraint_fun(x)

    def get_constraint_bounds(self):
        nconstraints = self.eta_indices.shape[0]
        lb = np.zeros(nconstraints)
        ub = np.inf*np.ones(nconstraints)
        return lb, ub

    def debug_plot(self, x):
        pce_vals = self.fun(x)
        pce_econds = compute_conditional_expectations(pce_vals, pce_vals, True)
        train_econds = compute_conditional_expectations(
            pce_vals, self.values, True)
        import matplotlib.pyplot as plt
        plt.plot(pce_vals, pce_econds, 'o-k', label=r'$\mathrm{Approx}$')
        plt.plot(pce_vals, train_econds, 's-r', label=r'$\mathrm{Train}$')
        plt.legend()
        plt.show()
        

def solve_SSD_constrained_least_squares_smooth(
        samples, values, eval_basis_matrix, eta_indices=None,
        probabilities=None, eps=1e-1, optim_options={}, return_full=False,
        method='trust-constr', smoother_type='log', scale_data=False):
    """
    Second order stochastic dominance SSD
    """
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
    scaled_values = values/values_std

    x0 = np.linalg.lstsq(basis_matrix, scaled_values, rcond=None)[0]
    residual = scaled_values-basis_matrix.dot(x0)
    x0[0] += max(0, residual.max()+eps)

    ssd_opt_problem = SSDOptProblem(
        scaled_values, fun, jac, None, eta_indices, probabilities,
        smoother_type, eps, ncoef)
    
    result = ssd_opt_problem.solve(x0, optim_options, method)
    assert result.success is True
    coef = result.x*values_std
    # ssd_opt_problem.debug_plot(coef)

    if return_full:
        return coef, fsd_opt_problem
    else:
        return coef
