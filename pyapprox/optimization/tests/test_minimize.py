import unittest

import numpy as np
from scipy import stats

from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer, Bounds, Constraint,
    SampleAverageMean, SampleAverageVariance, SampleAverageStdev,
    SampleAverageMeanPlusStdev, SampleAverageEntropicRisk,
    SampleAverageConstraint,
    SampleAverageConditionalValueAtRisk, CVaRSampleAverageConstraint,
    ObjectiveWithCVaRConstraints)
from pyapprox.benchmarks import setup_benchmark
from pyapprox.interface.model import ModelFromCallable
from pyapprox.variables.risk import gaussian_cvar
from pyapprox.interface.model import Model, ModelFromCallable


class TestMinimize(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_constrained_scipy(self):
        # check that no bounds is handled correctly
        nvars = 2
        benchmark = setup_benchmark("rosenbrock", nvars=nvars)
        optimizer = ScipyConstrainedOptimizer(benchmark.fun)
        result = optimizer.minimize(benchmark.variable.get_statistics("mean"))
        assert np.allclose(result.x, np.full(nvars, 1))

        # check that constraints are handled correctly
        nvars = 2
        bounds = Bounds(np.full((nvars,), -2), np.full((nvars,), 2))
        benchmark = setup_benchmark("rosenbrock", nvars=nvars)
        optimizer = ScipyConstrainedOptimizer(benchmark.fun, bounds=bounds)
        result = optimizer.minimize(benchmark.variable.get_statistics("mean"))
        assert np.allclose(result.x, np.full(nvars, 1))

        # check apply_jacobian and apply_hessian with 1D samples
        objective = ModelFromCallable(
            lambda x: ((x[0] - 1)**2 + (x[1] - 2.5)**2),
            jacobian=lambda x: np.array([2*(x[0] - 1), 2*(x[1] - 2.5)]),
            apply_jacobian=lambda x, v: 2*(x[0] - 1)*v[0]+2*(x[1] - 2.5)*v[1],
            # apply_hessian=lambda x, v: np.array(np.diag([2, 2])) @ v,
            hessian=lambda x: np.array(np.diag([2, 2])),
            sample_ndim=1, values_ndim=0)
        sample = np.array([2, 0])[:, None]
        errors = objective.check_apply_jacobian(sample, disp=True)
        assert errors.min()/errors.max() < 1e-6
        errors = objective.check_apply_hessian(sample, disp=True)
        assert errors[0] < 1e-15

        constraint_model = ModelFromCallable(
            lambda x:  np.array(
                [x[0]-2*x[1]+2, -x[0]-2*x[1]+6, -x[0]+2*x[1]+2]),
            lambda x: np.array(
                [[1., -2.], [-1., -2], [-1, 2.]]),
            # if there are m constraints with n inputs
            # then apply hessian must return a matrix of shape (n, n)
            # given a vector of shape (m)
            apply_hessian=lambda x, v: np.zeros((2, 2)),
            sample_ndim=1, values_ndim=1)
        errors = constraint_model.check_apply_jacobian(sample, disp=True)
        # jacobian is constant so check first finite difference is exact
        assert errors[0] < 1e-15

        constraint_bounds = np.hstack(
            [np.full((3, 1), 0), np.full((3, 1), np.inf)])
        print(constraint_bounds.shape)
        constraint = Constraint(constraint_model, constraint_bounds)

        bounds = Bounds(np.full((nvars,), 0), np.full((nvars,), np.inf))
        optimizer = ScipyConstrainedOptimizer(
            objective, bounds=bounds, constraints=[constraint],
            opts={"gtol": 1e-10, "verbose": 3})
        result = optimizer.minimize(np.array([2, 0])[:, None])
        assert np.allclose(result.x, np.array([1.4, 1.7]))

    def test_sample_average_constraints(self):
        benchmark = setup_benchmark('cantilever_beam')
        constraint_model = benchmark.funs[1]

        # test jacobian
        nsamples = 10000
        samples = benchmark.variable.rvs(nsamples)
        weights = np.full((nsamples, 1), 1/nsamples)
        for stat in [SampleAverageMean(), SampleAverageVariance(),
                     SampleAverageStdev(), SampleAverageMeanPlusStdev(2),
                     SampleAverageEntropicRisk(0.5)]:
            constraint_bounds = np.hstack(
                [np.zeros((2, 1)), np.full((2, 1), np.inf)])
            constraint = SampleAverageConstraint(
                constraint_model, samples, weights, stat, constraint_bounds,
                benchmark.variable.num_vars() +
                benchmark.design_variable.num_vars(),
                benchmark.design_var_indices)
            design_sample = np.array([3, 3])[:, None]
            assert constraint(design_sample).shape == (1, 2)
            errors = constraint.check_apply_jacobian(design_sample)
            # print(errors.min()/errors.max())
            assert errors.min()/errors.max() < 1.3e-6

        # test apply_jacobian
        constraint_model._apply_jacobian_implemented = True
        constraint_model._apply_jacobian = (
            lambda x, v: constraint_model.jacobian(x) @ v)

        nsamples = 1000
        samples = benchmark.variable.rvs(nsamples)
        weights = np.full((nsamples, 1), 1/nsamples)
        for stat in [SampleAverageMean(), SampleAverageVariance(),
                     SampleAverageStdev(), SampleAverageMeanPlusStdev(2),
                     SampleAverageEntropicRisk(0.5)]:
            constraint_bounds = np.hstack(
                [np.zeros((2, 1)), np.full((2, 1), np.inf)])
            constraint = SampleAverageConstraint(
                constraint_model, samples, weights, stat, constraint_bounds,
                benchmark.variable.num_vars() +
                benchmark.design_variable.num_vars(),
                benchmark.design_var_indices)
            design_sample = np.array([3, 3])[:, None]
            errors = constraint.check_apply_jacobian(design_sample)
            # print(errors.min()/errors.max(), stat)
            assert errors.min()/errors.max() < 1.3e-6

    def test_conditional_value_at_risk_gradients(self):
        benchmark = setup_benchmark('cantilever_beam')
        constraint_model = benchmark.funs[1]

        # test jacobian
        nsamples = 1000
        samples = benchmark.variable.rvs(nsamples)
        weights = np.full((nsamples, 1), 1/nsamples)
        for stat in [SampleAverageConditionalValueAtRisk([0.5, 0.85]),
                     SampleAverageConditionalValueAtRisk([0.85, 0.9])]:
            # first two are parameters second two are VaR
            # (one for each constraint)
            constraint_bounds = np.hstack(
                [np.zeros((4, 1)), np.full((4, 1), np.inf)])
            constraint = CVaRSampleAverageConstraint(
                constraint_model, samples, weights, stat, constraint_bounds,
                benchmark.variable.num_vars() +
                benchmark.design_variable.num_vars(),
                benchmark.design_var_indices)
            design_sample = np.array([3, 3, 1, 1])[:, None]
            assert constraint(design_sample).shape == (1, 2)
            errors = constraint.check_apply_jacobian(design_sample, disp=True)
            # print(errors.min()/errors.max())
            assert errors.min()/errors.max() < 1.3e-6

    def test_conditional_value_at_risk_optimization(self):
        # min -(s1) s.t CVaR[m1+s1*f1(z1)] <= c1 and CVaR[f2(z2)] <= c2
        mu1, sigma1 = 0, 1.1
        mu2, sigma2 = -0.5, 1.5

        # 2 random variables 2 design variables
        nrandom_vars, ndesign_vars = 2, 1
        # Model assumes ith contraint depends on ith variable
        nconstraints = nrandom_vars

        class CustomModel(Model):
            def __init__(self, nrandom_vars):
                super().__init__()
                self._jacobian_implemented = True
                self._nrandom_vars = nrandom_vars
                self._ndesign_vars = 1
                self._ident = np.eye(self._nrandom_vars-1)
                self._jac = np.zeros(
                    (self._nrandom_vars,
                     self._nrandom_vars+self._ndesign_vars))
                self._jac[1, 1:self._nrandom_vars] = self._ident

            def __call__(self, x):
                return np.hstack(
                    (x[:1].T*x[self._nrandom_vars:self._nrandom_vars+1].T,
                     x[1:self._nrandom_vars].T))

            def _jacobian(self, x):
                # only change the parts of jacobian that depend on x
                self._jac[0, 0] = x[self._nrandom_vars:self._nrandom_vars+1, 0]
                self._jac[0, -1] = x[:1, 0]
                return self._jac

        constraint_model = CustomModel(nrandom_vars)
        constraint_x0 = np.arange(2, nrandom_vars+ndesign_vars+2)[:, None]
        errors = constraint_model.check_apply_jacobian(
            constraint_x0, disp=True)
        assert errors.min()/errors.max() < 1e-6

        # objective model is just a function of design variables
        # so it is called as objective_model(design_sample)
        # no random variables are passed
        design_x0 = constraint_x0[nrandom_vars:nrandom_vars+1]
        objective_model = ModelFromCallable(
            lambda x: -x.T, lambda x: -np.ones((1, 1)))
        errors = objective_model.check_apply_jacobian(
            design_x0, disp=True)
        assert errors.min()/errors.max() < 1e-6

        assert mu1 == 0
        nsamples = int(1e4)
        # set mu, sigma of first random samples to (0, 1) as we are learning
        # scaling so that stdev of samlpes is equal to sigma1
        samples = np.vstack((np.random.normal(0, 1, (1, nsamples)),
                             np.random.normal(mu2, sigma2, (1, nsamples))))
        weights = np.full((nsamples, 1), 1/nsamples)
        # from pyapprox.surrogates.orthopoly.quadrature import (
        #     gauss_hermite_pts_wts_1D)

        # nsamples = 1000
        # samples = np.vstack(
        #     [gauss_hermite_pts_wts_1D(nsamples)[0],
        #      gauss_hermite_pts_wts_1D(nsamples)[0]*sigma2+mu2])
        # weights = gauss_hermite_pts_wts_1D(nsamples)[1][:, None]
        nsamples = int(1e3)+1
        from pyapprox.surrogates.interp.tensorprod import (
            UnivariatePiecewiseQuadraticBasis)
        basis = UnivariatePiecewiseQuadraticBasis()
        nodes = np.linspace(*stats.norm(0, 1).interval(1-1e-6), nsamples)
        print(nodes)
        weights = basis._quadrature_rule_from_nodes(nodes[None, :])[1][:, 0]
        weights = (weights*stats.norm(0, 1).pdf(nodes))[:, None]
        samples = np.vstack([nodes[None, :], nodes[None, :]*sigma2+mu2])
        stat = SampleAverageConditionalValueAtRisk([0.5, 0.85], eps=1e-3)

        CVaR1 = gaussian_cvar(mu1, sigma1, stat._alpha[0])
        CVaR2 = gaussian_cvar(mu2, sigma2, stat._alpha[1])
        VaR1 = stats.norm(mu1, sigma1).ppf(stat._alpha[0])
        VaR2 = stats.norm(mu2, sigma2).ppf(stat._alpha[1])
        constraint_bounds = np.hstack(
            [np.zeros((2, 1)), np.hstack([CVaR1, CVaR2])[:, None]])
        constraint = CVaRSampleAverageConstraint(
            constraint_model, samples, weights, stat, constraint_bounds,
            nrandom_vars+ndesign_vars,
            np.arange(nrandom_vars, nrandom_vars+ndesign_vars))
        objective = ObjectiveWithCVaRConstraints(
            objective_model, nconstraints)
        opt_x0 = np.vstack((design_x0, np.full((nconstraints, 1), 0.5)))
        errors = objective.check_apply_jacobian(opt_x0, disp=True)
        assert errors.min()/errors.max() < 1e-6
        errors = constraint.check_apply_jacobian(opt_x0, disp=True)
        assert errors.min()/errors.max() < 1e-6

        exact_opt_x = np.array([sigma1, VaR1, VaR2])[:, None]
        # Gauss Quadrature cannot easily get accruate estimate of CVaR
        # because of discontinuity (or highly nonlinear component) of
        # (smoothed) max function
        print(constraint(exact_opt_x)[0, :]-np.array([CVaR1, CVaR2]))
        # assert np.allclose(
        #    constraint(exact_opt_x)[0, :], [CVaR1, CVaR2], rtol=5e-3)

        bounds = Bounds(
            np.hstack(([0], np.full((nconstraints,), -np.inf))),
            np.full((ndesign_vars+nconstraints,), np.inf))
        optimizer = ScipyConstrainedOptimizer(
            objective, bounds=bounds, constraints=[constraint],
            opts={"gtol": 3e-6, "verbose": 3, "maxiter": 500})
        result = optimizer.minimize(opt_x0)

        # errors in sample based estimate of CVaR will cause
        # optimal solution to be biased.
        assert np.allclose(
            constraint(result.x[:, None]), [CVaR1, CVaR2], rtol=1e-2)
        # print(constraint(exact_opt_x), [CVaR1, CVaR2])
        # print(result.x-exact_opt_x[:, 0], exact_opt_x[:, 0])

        # TODO: on ubuntu reducing gtol causes minimize not to converge
        # ideally find reason and dencrease rtol and atol below
        assert np.allclose(result.x, exact_opt_x[:, 0], rtol=2e-3, atol=1e-5)
        assert np.allclose(-sigma1, result.fun, rtol=1e-4)


if __name__ == '__main__':
    minimize_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestMinimize)
    unittest.TextTestRunner(verbosity=2).run(minimize_test_suite)
