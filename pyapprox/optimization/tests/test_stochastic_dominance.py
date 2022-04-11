import unittest
from functools import partial
import numpy as np

from pyapprox.optimization.first_order_stochastic_dominance import (
    linear_model_fun, linear_model_jac, FSDOptProblem
)
from pyapprox.optimization.second_order_stochastic_dominance import (
    solve_SSD_constrained_least_squares_smooth
)
from pyapprox.util.utilities import check_gradients, check_hessian
from pyapprox.optimization.pya_minimize import has_ROL
from pyapprox.variables.risk import compute_conditional_expectations


skiptest_rol = unittest.skipIf(
    not has_ROL, reason="rol package not found")


class TestFirstOrderStochasticDominance(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def setup_linear_regression_problem(self, nsamples, degree, delta):
        # samples = np.random.normal(0, 1, (1, nsamples-1))
        # samples = np.hstack([samples, samples[:, 0:1]+delta])
        samples = np.linspace(1e-3, 2+1e-3, nsamples)[None, :]
        # have two samples close together so gradient of smooth heaviside
        # function evaluated at approx_values - approx_values[jj] is
        # small and so gradient will be non-zero
        values = np.exp(samples).T

        basis_matrix = samples.T**np.arange(degree+1)
        fun = partial(linear_model_fun, basis_matrix)
        jac = partial(linear_model_jac, basis_matrix)
        probabilities = np.ones((nsamples))/nsamples
        ncoef = degree+1
        x0 = np.linalg.lstsq(basis_matrix, values, rcond=None)[0]
        shift = (values-basis_matrix.dot(x0)).max()
        x0[0] += shift

        return samples, values, fun, jac, probabilities, ncoef, x0

    def test_objective_derivatives(self):
        smoother_type, eps = 'log', 5e-1
        nsamples, degree = 10, 1

        samples, values, fun, jac, probabilities, ncoef, x0 = \
            self.setup_linear_regression_problem(nsamples, degree, eps)

        x0 -= eps/3

        eta = np.arange(nsamples//2, nsamples)
        problem = FSDOptProblem(
            values, fun, jac, None, eta, probabilities, smoother_type, eps,
            ncoef)

        # assert smooth function is shifted correctly (local methods only)
        # assert problem.smooth_fun(np.array([[0.]])) == 1.0

        err = check_gradients(
            problem.objective_fun, problem.objective_jac, x0, rel=False)
        # fd difference error should exhibit V-cycle. These values
        # test this for this specific problem
        assert err.min() < 1e-6 and err.max() > 0.1
        err = check_hessian(
            problem.objective_jac, problem.objective_hessp, x0, rel=False)
        # fd hessian  error should decay linearly (assuming first order fd)
        # because hessian is constant (indendent of x)
        assert err[0] < 1e-14 and err[10] > 1e-9
        # fd difference error should exhibit V-cycle. These values
        # test this for this specific problem
        err = check_gradients(
            problem.constraint_fun, problem.constraint_jac, x0, rel=False)
        assert err.min() < 1e-7 and err.max() > 0.03

        lmult = np.random.normal(0, 1, (eta.shape[0]))

        def constr_jac(x):
            jl = problem.constraint_jac(x).T.dot(lmult)
            return jl

        def constr_hessp(x, v):
            hl = problem.constraint_hess(x, lmult).dot(v)
            return hl
        err = check_hessian(constr_jac, constr_hessp, x0)
        assert err.min() < 1e-5 and err.max() > 0.1

    # @skiptest
    def test_1d_monomial_regression(self):
        # smoother_type, eps = 'quartic', 1e-2
        # smoother_type, eps = 'quintic', 5e-1
        smoother_type, eps = 'log', 5e-2
        nsamples, degree = 10, 3
        # nsamples, degree, eps = 10, 1, 5e-2 converges but it takes a long time
        # convergence seems to get easier (I can decrease eps) as degree
        # increases. I think because size of residuals gets smaller

        # import matplotlib.pyplot as plt
        # xx = np.linspace(-1, 1, 101)
        # yy = smooth_left_heaviside_function_log(eps, 0, xx)
        # plt.plot(xx, yy)
        # yy = numba_smooth_left_heaviside_function_quartic(
        #     eps, 0, xx[:, None])
        # plt.plot(xx, yy)
        # plt.show()

        samples, values, fun, jac, probabilities, ncoef, x0 = \
            self.setup_linear_regression_problem(nsamples, degree, eps)
        values_std = values.std()
        scaled_values = values/values_std

        eta = np.arange(nsamples)
        problem = FSDOptProblem(
            scaled_values, fun, jac, None, eta, probabilities, smoother_type,
            eps, ncoef)

        # import matplotlib.pyplot as plt
        # xx = np.linspace(-1, 1 , 101)
        # print(problem.smooth_fun(np.array([[0]])))
        # plt.plot(xx, problem.smooth_fun(xx[:, None]))
        # plt.show()

        method, maxiter = 'rol-trust-constr', 100
        # method, maxiter = 'trust-constr', 1000
        optim_options = {'verbose': 2, 'maxiter': maxiter}
        coef = problem.solve(x0, optim_options, method).x
        coef *= values_std

    def test_second_order_stochastic_dominance_constraints(self):
        # import matplotlib.pyplot as plt
        # eps = 1e-1
        # xx = np.linspace(-1, 1, 101)
        # yy = smooth_max_function_log(eps, 0, xx)
        # plt.plot(xx, yy)
        # plt.show()

        nbasis = 2

        def func(x):
            return (1+x-x**2+x**3).T
        samples = np.random.uniform(-1, 1, (1, 20))
        samples = np.sort(samples)
        values = func(samples)

        def eval_basis_matrix(x):
            return (x**np.arange(nbasis)[:, None]).T
        # tau = 0.75
        tol = 1e-14
        eps = 1e-3
        optim_options = {'verbose': 3, 'maxiter': 2000,
                         'gtol': tol, 'xtol': tol, 'barrier_tol': tol}
        ssd_coef = solve_SSD_constrained_least_squares_smooth(
            samples, values, eval_basis_matrix, optim_options=optim_options,
            eps=eps, method='trust-constr', scale_data=True)
        # true_coef = np.zeros((nbasis))
        approx_vals = eval_basis_matrix(samples).dot(ssd_coef)
        assert approx_vals.max() >= values.max()
        assert approx_vals.mean() >= values.mean()

        pce_vals = eval_basis_matrix(samples).dot(ssd_coef)
        pce_econds = compute_conditional_expectations(pce_vals, pce_vals, True)
        train_econds = compute_conditional_expectations(
            pce_vals, values[:, 0], True)
        print(train_econds-pce_econds)
        assert np.all(train_econds <= pce_econds+4e-5)

        # import matplotlib.pyplot as plt
        # xx = np.linspace(-1, 1, 101)
        # yy = eval_basis_matrix(xx[None, :]).dot(ssd_coef)
        # plt.plot(xx, yy)
        # plt.plot(samples[0, :], values, 'o')
        # plt.show()


if __name__ == "__main__":
    fsd_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestFirstOrderStochasticDominance)
    unittest.TextTestRunner(verbosity=2).run(fsd_test_suite)
