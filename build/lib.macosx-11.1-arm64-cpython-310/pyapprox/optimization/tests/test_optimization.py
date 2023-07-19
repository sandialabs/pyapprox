import unittest
import numpy as np

from pyapprox.util.utilities import approx_jacobian
from pyapprox.optimization.optimization import (
    eval_function_at_multiple_design_and_random_samples,
    eval_mc_based_jacobian_at_multiple_design_samples,
    smooth_prob_failure_fun, smooth_prob_failure_jac
)


class TestOptimization(unittest.TestCase):
    def setUp(self):
        np.random.seed(2)

    def test_approx_jacobian(self):
        def constraint_function(x): return np.array([1-x[0]**2-x[1]])
        def constraint_grad(x): return [-2*x[0], -1]

        x0 = np.random.uniform(0, 1, (2, 1))
        true_jacobian = constraint_grad(x0[:, 0])
        assert np.allclose(true_jacobian, approx_jacobian(
            constraint_function, x0[:, 0]))

        def constraint_function(x): return np.array(
            [1 - x[0] - 2*x[1], 1 - x[0]**2 - x[1], 1 - x[0]**2 + x[1]])

        def constraint_grad(x): return np.array([[-1.0, -2.0],
                                                 [-2*x[0], -1.0],
                                                 [-2*x[0], 1.0]])

        x0 = np.random.uniform(0, 1, (2, 1))
        true_jacobian = constraint_grad(x0[:, 0])
        # print(true_jacobian,'\n',approx_jacobian(
        #    constraint_function,x0[:,0]))
        assert np.allclose(true_jacobian, approx_jacobian(
            constraint_function, x0[:, 0]))

    def test_eval_mc_based_jacobian_at_multiple_design_samples(self):
        def constraint_function_single(
            z, x): return np.array([z[0]*(1-x[0]**2-x[1])])

        def constraint_grad_single(z, x): return [-2*z[0]*x[0], -z[0]]

        x0 = np.random.uniform(0, 1, (2, 2))
        zz = np.arange(0, 6, 2)[np.newaxis, :]

        vals = eval_function_at_multiple_design_and_random_samples(
            constraint_function_single, zz, x0)

        def stat_func(vals): return np.mean(vals, axis=0)
        jacobian = eval_mc_based_jacobian_at_multiple_design_samples(
            constraint_grad_single, stat_func, zz, x0)

        true_jacobian = [np.mean([constraint_grad_single(z, x) for z in zz.T],
                                 axis=0)
                         for x in x0.T]
        assert np.allclose(true_jacobian, jacobian)

        def constraint_function_single(z, x): return z[0]*np.array(
            [1 - x[0] - 2*x[1], 1 - x[0]**2 - x[1], 1 - x[0]**2 + x[1]])

        def constraint_grad_single(z, x):
            return z[0]*np.array([[-1.0, -2.0],
                                  [-2*x[0], -1.0],
                                  [-2*x[0], 1.0]])

        x0 = np.random.uniform(0, 1, (2, 2))
        zz = np.arange(0, 6, 2)[np.newaxis, :]

        def stat_func(vals): return np.mean(vals, axis=0)
        jacobian = eval_mc_based_jacobian_at_multiple_design_samples(
            constraint_grad_single, stat_func, zz, x0)

        true_jacobian = [
            np.mean([constraint_grad_single(z, x) for z in zz.T], axis=0)
            for x in x0.T]
        assert np.allclose(true_jacobian, jacobian)

        # lower_bound=0
        # nsamples = 100
        # uq_samples = np.random.uniform(0,1,(2,nsamples))
        # func = partial(mean_lower_bound_constraint,constraint_function,lower_bound,uq_samples)
        # grad = partial(mean_lower_bound_constraint_jacobian,constraint_grad,uq_samples)

    def test_prob_failure_fun(self):
        smoother_type, eps = 0, 1e-3
        nsamples = 1000
        weights = np.ones(nsamples)/nsamples
        samples = np.random.uniform(0, 0.25, (1, nsamples))
        a = np.array([2])
        values = a**2*(samples).sum(axis=0)
        # print(values.mean())
        tol = 0.5
        prob_vals = smooth_prob_failure_fun(
            smoother_type, eps, tol, values, weights)
        jac_values = (2*a*samples).T
        fd_eps = 1e-7
        values_pert = (a+fd_eps)**2*(samples).sum(axis=0)
        prob_vals_pert = smooth_prob_failure_fun(smoother_type, eps, tol,
                                                 values_pert, weights)
        grad = (smooth_prob_failure_jac(
            smoother_type, eps, tol, jac_values, weights))
        fd_grad = ((prob_vals_pert-prob_vals)/fd_eps)
        assert np.allclose(grad, fd_grad)


if __name__ == '__main__':
    optimization_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestOptimization)
    unittest.TextTestRunner(verbosity=2).run(optimization_test_suite)
