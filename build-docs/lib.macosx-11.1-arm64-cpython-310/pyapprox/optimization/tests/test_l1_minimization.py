import unittest
import numpy as np
from functools import partial
import scipy.sparse as sp
from scipy.optimize import minimize

from pyapprox.optimization.l1_minimization import (
    basis_pursuit, kouri_smooth_l1_norm,
    nonlinear_basis_pursuit, kouri_smooth_l1_norm_gradient,
    kouri_smooth_l1_norm_hessian, basis_pursuit_denoising, lasso,
    iterative_hard_thresholding, s_sparse_projection,
    orthogonal_matching_pursuit
)
from pyapprox.util.utilities import check_gradients, approx_jacobian
from pyapprox.optimization.optimization import (
    PyapproxFunctionAsScipyMinimizeObjective,
    ScipyMinimizeObjectiveAsPyapproxFunction,
    ScipyMinimizeObjectiveJacAsPyapproxJac
)


class TestL1Minimization(unittest.TestCase):
    def test_basis_pursuit(self):
        np.random.seed(1)
        nsamples, degree, sparsity = 6, 7, 2
        samples = np.random.uniform(0, 1, (1, nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis, :]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]] = 1.
        vals = basis_matrix.dot(true_coef)

        options = {'presolve': True, 'autoscale': True, 'disp': True}
        coef = basis_pursuit(basis_matrix, vals, options)
        assert np.allclose(coef, true_coef)

    def test_least_squares(self):
        """for tutorial purposes. Perhaps move to a tutorial"""
        np.random.seed(1)
        tol = 1e-14
        nsamples, degree, sparsity = 20, 7, 2
        samples = np.random.uniform(0, 1, (1, nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis, :]

        true_coef = np.zeros((basis_matrix.shape[1], 1))
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]] = 1.
        vals = basis_matrix.dot(true_coef)

        def objective(x, return_grad=True):
            residual = basis_matrix.dot(x)-vals
            obj = 0.5*residual.T.dot(residual)[0, 0]
            jac = basis_matrix.T.dot(residual).T
            if return_grad:
                return obj, jac
            return obj

        def hessian(x):
            return basis_matrix.T.dot(basis_matrix)

        # lstsq_coef = np.linalg.lstsq(basis_matrix, vals, rcond=0)[0]

        init_guess = np.random.normal(0, 0.1, (true_coef.shape[0], 1))
        # init_guess = lstsq_coef+np.random.normal(0,1e-3,(true_coef.shape[0]))

        errors = check_gradients(objective, True, init_guess, disp=True)
        print(errors.min())
        assert errors.min() < 2e-7

        method = 'trust-constr'
        func = partial(objective, return_grad=True)
        jac = True
        hess = hessian
        options = {'gtol': tol, 'verbose': 0,
                   'disp': True, 'xtol': tol, 'maxiter': 10000}
        constraints = []

        fun = PyapproxFunctionAsScipyMinimizeObjective(func)
        res = minimize(
            fun, init_guess[:, 0], method=method, jac=jac, hess=hess,
            options=options, constraints=constraints)

        # print(lstsq_coef)
        # print(res.x,true_coef)
        assert np.allclose(res.x[:, np.newaxis], true_coef, atol=5e-4)

    def test_nonlinear_basis_pursuit_with_linear_model(self):
        np.random.seed(1)
        # nsamples, degree, sparsity = 100, 7, 2
        # samples = np.random.uniform(0,1,(1,nsamples))
        # basis_matrix = samples.T**np.arange(degree+1)[np.newaxis,:]

        # true_coef = np.zeros(basis_matrix.shape[1])
        # true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]]=1.
        # vals = basis_matrix.dot(true_coef)

        basis_matrix, true_coef, vals = self.SPARCO_problem_7(
            1024//4, 256//4, 32//4)

        def func(x):
            return basis_matrix.dot(x)-vals, basis_matrix
        jac = True

        def hess(x):
            return sp.lil_matrix((x.shape[0], x.shape[0]), dtype=float)

        tol = 1e-12
        options = {'gtol': tol, 'verbose': 2,
                   'disp': True, 'xtol': tol, 'maxiter': 10000}
        # options = {'tol':tol,'maxiter':1000,'print_level':3,
        #           'method':'ipopt'}
        init_guess = true_coef+np.random.normal(0, 1, true_coef.shape[0])
        # fd_jac = approx_jacobian(lambda x: func(x)[0],init_guess,epsilon=1e-7)
        # exact_jac = func(init_guess)[1]
        l1_coef = nonlinear_basis_pursuit(func, jac, hess, init_guess, options)
        assert np.allclose(l1_coef, true_coef, atol=1e-5)

    def test_nonlinear_basis_pursuit_denoising_with_linear_model(self):
        np.random.seed(1)
        nsamples, degree, sparsity = 20, 7, 2
        samples = np.random.uniform(0, 1, (1, nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis, :]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]] = 1.
        vals = basis_matrix.dot(true_coef)

        # basis_matrix,true_coef,vals = self.SPARCO_problem_7(1024//4,256//4,32//4)

        eps = 1e-3

        def func(x, return_grad=True):
            residual = basis_matrix.dot(x)-vals
            obj = residual.dot(residual)
            grad = 2*basis_matrix.T.dot(residual)
            if return_grad:
                return obj, grad
            return obj
        jac = True
        hess = None

        tol = 1e-6
        # options = {'gtol':tol,'verbose':2,'disp':True,'xtol':tol,'maxiter':1000,
        #           'method':'trust-constr'}
        options = {'ftol':tol,'disp':True,'maxiter':1000,'iprint':3,
                   'method':'slsqp'}
        # options = {'tol': tol, 'maxiter': 1000, 'print_level': 3,
        #            'method': 'ipopt'}
        init_guess = np.random.normal(0, 1, true_coef.shape[0])
        # fd_jac = approx_jacobian(lambda x: func(x)[0],init_guess,epsilon=1e-7)
        # exact_jac = func(init_guess)[1]
        l1_coef = nonlinear_basis_pursuit(
            func, jac, hess, init_guess, options, eps**2)
        print(np.linalg.norm(l1_coef-true_coef))
        print(l1_coef-true_coef)
        print(l1_coef, true_coef)
        assert np.allclose(l1_coef, true_coef, atol=6e-3)

    def test_nonlinear_basis_pursuit(self):
        np.random.seed(1)
        nsamples, degree, sparsity = 7, 7, 2
        samples = np.random.uniform(0, 1, (1, nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis, :]

        def model(x):
            val = basis_matrix.dot(x[:-1])*np.exp(samples[0, :]*x[-1])
            grad = np.hstack(
                [basis_matrix*np.exp(samples[0, :]*x[-1])[:, np.newaxis],
                 (samples[0, :]*val)[:, np.newaxis]])
            return val, grad

        true_coef = np.zeros(basis_matrix.shape[1]+1)
        true_coef[np.random.permutation(
            true_coef.shape[0]-1)[:sparsity-1]] = 1.
        true_coef[-1] = 1
        vals = model(true_coef)[0]

        def func(x):
            model_vals, grad = model(x)
            return model_vals-vals, grad
        jac = True
        hess = None

        init_guess = true_coef+np.random.normal(0, 1, true_coef.shape[0])
        fd_jac = approx_jacobian(lambda x: model(
            x)[0], init_guess, epsilon=1e-7)
        analytical_jac = model(init_guess)[1]
        # print(analytical_jac-fd_jac)
        assert np.allclose(analytical_jac, fd_jac, atol=1e-8)

        tol = 1e-12
        options = {'ftol': tol, 'verbose': 2,
                   'disp': True, 'xtol': tol, 'maxiter': 1000}
        init_guess = true_coef+np.random.normal(0, 1, true_coef.shape[0])
        l1_coef = nonlinear_basis_pursuit(func, jac, hess, init_guess, options)
        print(np.linalg.norm(true_coef-l1_coef))
        assert np.allclose(l1_coef, true_coef, atol=2e-6)

    def test_smooth_l1_norm_gradients(self):
        # x = np.linspace(-1,1,101)
        # t = np.ones_like(x)
        # r = 1e1
        # plt.plot(x,kouri_smooth_absolute_value(t,r,x))
        # plt.show()

        t = np.ones(5)
        r = 1
        init_guess = np.random.normal(0, 1, (t.shape[0], 1))
        func = ScipyMinimizeObjectiveAsPyapproxFunction(
            partial(kouri_smooth_l1_norm, t, r))
        jac = partial(kouri_smooth_l1_norm_gradient, t, r)
        pya_jac = ScipyMinimizeObjectiveJacAsPyapproxJac(jac)
        errors = check_gradients(func, pya_jac, init_guess, disp=False)
        assert errors.min() < 3e-7

        fd_hess = approx_jacobian(jac, init_guess[:, 0])
        assert np.allclose(
            fd_hess, kouri_smooth_l1_norm_hessian(t, r, init_guess))

    def SPARCO_problem_7(self, basis_length, num_samples, sparsity):
        sparsity_matrix = np.eye(basis_length)
        measurement_matrix = np.random.normal(
            0, 1, (num_samples, basis_length))
        signal = np.zeros((basis_length), float)
        non_zero_indices = np.random.permutation(basis_length)[:sparsity]
        signal[non_zero_indices] = np.sign(np.random.normal(0, 1, (sparsity)))
        measurements = np.dot(measurement_matrix, signal)
        A_matrix = np.dot(measurement_matrix, sparsity_matrix)
        true_solution = signal
        return A_matrix, true_solution, measurements

    def test_basis_pursuit_smooth_l1_norm(self):
        np.random.seed(1)
        # nsamples, degree, sparsity = 6, 7, 2
        nsamples, degree, sparsity = 15, 20, 3
        samples = np.random.uniform(0, 1, (1, nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis, :]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]] = 1.
        vals = basis_matrix.dot(true_coef)

        # basis_matrix,true_coef,vals = self.SPARCO_problem_7(1024//4,256//4,32//4)

        def func(x):
            return basis_matrix.dot(x)-vals, basis_matrix
        jac = True

        def hess(x):
            return sp.lil_matrix((x.shape[0], x.shape[0]), dtype=float)

        tol = 1e-10
        eps = 0
        init_guess = np.random.normal(0, 1, (true_coef.shape[0]))*0
        method = 'slsqp'
        # init_guess = true_coef
        options = {'gtol': tol, 'verbose': 2, 'disp': True, 'xtol': 1e-10,
                   'maxiter': 20, 'method': method, 'ftol': 1e-10}
        res = basis_pursuit_denoising(
            func, jac, hess, init_guess, eps, options)
        coef = res.x

        # print(true_coef,coef)
        print(np.absolute(coef-true_coef), 'c')
        assert np.allclose(true_coef, coef, atol=3e-5)

    def test_basis_pursuit_denoising_smooth_l1_norm(self):
        np.random.seed(1)
        nsamples, degree, sparsity = 20, 7, 2
        # nsamples, degree, sparsity = 15, 20, 3
        samples = np.random.uniform(0, 1, (1, nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis, :]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]] = 1.
        vals = basis_matrix.dot(true_coef)

        # basis_matrix,true_coef,vals = self.SPARCO_problem_7(1024//4,256//4,32//4)

        def func(x, return_grad=True):
            residual = basis_matrix.dot(x)-vals
            obj = residual.dot(residual)
            grad = 2*basis_matrix.T.dot(residual)
            if return_grad:
                return obj, grad
            return obj

        def hess(x):
            return 2*basis_matrix.T.dot(basis_matrix)
        jac = True

        init_guess = np.random.normal(0, 1, (true_coef.shape[0]))
        assert np.allclose(
            func(init_guess)[0],
            np.linalg.norm(basis_matrix.dot(init_guess)-vals)**2)

        print(true_coef)
        eps = 1e-3
        method = 'slsqp'
        # method = 'ipopt'
        init_guess = np.random.normal(0, 1, (true_coef.shape[0]))*0
        # init_guess = true_coef
        options = {'gtol': 1e-8, 'verbose': 2, 'disp': True, 'dualtol': 1e-6,
                   'maxiter_inner': 3e3, 'r0': 1e4, 'maxiter': 1e2,
                   'ftol': 1e-10,
                   'method': method}
        res = basis_pursuit_denoising(
            func, jac, hess, init_guess, eps, options)
        coef = res.x

        print(np.linalg.norm(true_coef-coef))
        assert np.allclose(true_coef, coef, atol=6e-3)

    @unittest.skip(reason="test incomplete")
    def test_lasso(self):
        np.random.seed(1)
        nsamples, degree, sparsity = 20, 7, 2
        # nsamples, degree, sparsity = 15, 20, 3
        samples = np.random.uniform(0, 1, (1, nsamples))
        basis_matrix = samples.T**np.arange(degree+1)[np.newaxis, :]

        true_coef = np.zeros(basis_matrix.shape[1])
        true_coef[np.random.permutation(true_coef.shape[0])[:sparsity]] = 1.
        vals = basis_matrix.dot(true_coef)
        lamdas = np.logspace(-6, 1, 10)

        # basis_matrix,true_coef,vals = self.SPARCO_problem_7(1024//4,256//4,32//4)
        # lamdas = np.logspace(0, 2, 20)

        def func(x, return_grad=True):
            residual = basis_matrix.dot(x)-vals
            obj = 0.5*residual.dot(residual)
            grad = basis_matrix.T.dot(residual)
            if return_grad:
                return obj, grad
            return obj
        jac = True

        AtA = basis_matrix.T.dot(basis_matrix)

        def hess(x):
            return AtA
        # hess=None

        init_guess = np.random.normal(0, 1, (true_coef.shape[0]))*0
        #init_guess = true_coef
        for i, lamda in enumerate(lamdas):
            # options = {'gtol':1e-12,'disp':True,
            #           'maxiter':1e3, 'method':'trust-constr',
            #           'sparse_jacobian':True,
            #           'barrier_tol':1e-12}
            # options = {'ftol':1e-12,'disp':False,
            #           'maxiter':1e3, 'method':'slsqp'};hess=None
            options = {'tol': 1e-8, 'maxiter': int(1e4), 'print_level': 0,
                       'method': 'ipopt', 'mu_strategy': 'adaptive',
                       'jac_d_constant': 'yes', 'hessian_constant': 'yes',
                       'obj_scaling_factor': float(1/lamda)}
            coef, res = lasso(func, jac, hess, init_guess, lamda, options)
            # initial_guess = coef
            print(lamda, np.absolute(coef).sum(), func(
                coef, False), np.linalg.norm(true_coef-coef))
            assert res.success == True
        assert False, 'test not finished'

    def test_iterative_hard_thresholding_gaussian_matrix(self):
        np.random.seed(3)
        num_samples = 30
        sparsity = 3
        num_terms = 30
        Amatrix = np.random.normal(
            0., 1., (num_samples, num_terms))/np.sqrt(num_samples)
        true_sol = np.zeros((num_terms))
        II = np.random.permutation(num_terms)[:sparsity]
        true_sol[II] = np.random.normal(0., 1., (sparsity))
        true_sol /= np.linalg.norm(true_sol)
        obs = np.dot(Amatrix, true_sol)

        def approx_eval(x): return np.dot(Amatrix, x)
        def apply_approx_adjoint_jacobian(x, y): return -np.dot(Amatrix.T, y)
        project = partial(s_sparse_projection, sparsity=sparsity)

        initial_guess = np.zeros_like(true_sol)
        tol = 1e-5
        max_iter = 100
        result = iterative_hard_thresholding(
            approx_eval, apply_approx_adjoint_jacobian, project,
            obs, initial_guess, tol, max_iter)
        sol = result[0]
        assert np.allclose(true_sol, sol, atol=10*tol)

    def test_omp_gaussian_matrix(self):
        num_samples = 30
        sparsity = 5
        num_terms = 30
        Amatrix = np.random.normal(
            0., 1., (num_samples, num_terms))/np.sqrt(num_samples)
        true_sol = np.zeros((num_terms))
        II = np.random.permutation(num_terms)[:sparsity]
        true_sol[II] = np.random.normal(0., 1., (sparsity))
        true_sol /= np.linalg.norm(true_sol)
        obs = np.dot(Amatrix, true_sol)

        def least_squares_regression(indices, initial_guess):
            return np.linalg.lstsq(
                Amatrix[:, indices], obs, rcond=None)[0]

        def approx_eval(x): return np.dot(Amatrix, x)
        def apply_approx_adjoint_jacobian(x, y): return -np.dot(Amatrix.T, y)

        tol = 1e-5
        active_indices = None

        result = orthogonal_matching_pursuit(
            approx_eval, apply_approx_adjoint_jacobian,
            least_squares_regression,
            obs, active_indices, num_terms, tol, sparsity)
        sol = result[0]
        assert np.allclose(true_sol, sol, atol=10*tol)

    def test_omp_gaussian_matrix_with_initial_active_indices(self):
        num_samples = 30
        sparsity = 5
        num_terms = 30
        Amatrix = np.random.normal(
            0., 1., (num_samples, num_terms))/np.sqrt(num_samples)
        true_sol = np.zeros((num_terms))
        II = np.random.permutation(num_terms)[:sparsity]
        true_sol[II] = np.random.normal(0., 1., (sparsity))
        true_sol /= np.linalg.norm(true_sol)
        obs = np.dot(Amatrix, true_sol)

        def approx_eval(x): return np.dot(Amatrix, x)
        def apply_approx_adjoint_jacobian(x, y): return -np.dot(Amatrix.T, y)

        def least_squares_regression(indices, initial_guess):
            return np.linalg.lstsq(
                Amatrix[:, indices], obs, rcond=None)[0]

        tol = 1e-5
        # use first three sparse terms
        active_indices = II[:3]

        result = orthogonal_matching_pursuit(
            approx_eval, apply_approx_adjoint_jacobian,
            least_squares_regression,
            obs, active_indices, num_terms, tol, sparsity)
        sol = result[0]
        assert np.allclose(true_sol, sol, atol=10*tol)


if __name__ == '__main__':
    l1_minimization_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestL1Minimization)
    unittest.TextTestRunner(verbosity=2).run(l1_minimization_test_suite)
