import unittest
from pyapprox.first_order_stochastic_dominance import *
from pyapprox.optimization import check_gradients, check_hessian
from pyapprox.rol_minimize import has_ROL
from functools import partial


skiptest_rol = unittest.skipIf(
    not has_ROL, reason="rol package not found")
skiptest = unittest.skip(reason="test incomplete")
class TestFirstOrderStochasticDominance(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def setup_linear_regression_problem(self, nsamples, degree, delta):
        #samples = np.random.normal(0, 1, (1, nsamples-1))
        #samples = np.hstack([samples, samples[:, 0:1]+delta])
        samples = np.linspace(1e-3,2+1e-3,nsamples)[None, :]
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
        smoother_type, eps = 'quintic', 5e-1
        nsamples, degree = 10, 1

        samples, values, fun, jac, probabilities, ncoef, x0 = \
            self.setup_linear_regression_problem(nsamples, degree, eps)

        x0 -= eps/3

        eta = np.arange(nsamples//2, nsamples)
        problem = FSDOptProblem(
            values, fun, jac, None, eta, probabilities, smoother_type, eps,
            ncoef)

        # assert smooth function is shifted correctly
        assert problem.smooth_fun(np.array([[0.]])) == 1.0

        err = check_gradients(
            problem.objective_fun, problem.objective_jac, x0, rel=False)
        # fd difference error should exhibit V-cycle. These values
        # test this for this specific problem
        assert err.min()<1e-6 and err.max()>0.1
        err = check_hessian(
            problem.objective_jac, problem.objective_hessp, x0, rel=False)
        # fd hessian  error should decay linearly (assuming first order fd)
        # because hessian is constant (indendent of x)
        assert err[0]<1e-14 and err[10]>1e-9
        # fd difference error should exhibit V-cycle. These values
        # test this for this specific problem
        err = check_gradients(
            problem.constraint_fun, problem.constraint_jac, x0, rel=False)
        assert err.min()<1e-7 and err.max()>0.1

        lmult = np.random.normal(0, 1, (eta.shape[0]))
        def constr_jac(x):
            jl = problem.constraint_jac(x).T.dot(lmult)
            return jl
        
        def constr_hessp(x, v):
            hl = problem.constraint_hess(x, lmult).dot(v)
            return hl
        err = check_hessian(constr_jac, constr_hessp, x0)
        assert err.min()<1e-5 and err.max()>0.1

    @skiptest
    def test_1d_monomial_regression(self):
        smoother_type, eps = 'quartic', 1e-2
        nsamples, degree = 10, 1

        samples, values, fun, jac, probabilities, ncoef, x0 = \
            self.setup_linear_regression_problem(nsamples, degree, eps)

        eta = np.arange(nsamples)
        problem = FSDOptProblem(
            values, fun, jac, None, eta, probabilities, smoother_type, eps,
            ncoef)

        optim_options = {'verbose': 2, 'maxiter':2}
        problem.solve(x0, optim_options)
        


if __name__ == "__main__":
    fsd_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestFirstOrderStochasticDominance)
    unittest.TextTestRunner(verbosity=2).run(fsd_test_suite)
