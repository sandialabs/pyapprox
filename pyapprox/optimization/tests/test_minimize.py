import unittest
import math

import numpy as np
from scipy import stats

from pyapprox.optimization.scipy import ScipyConstrainedOptimizer
from pyapprox.optimization.minimize import (
    SampleAverageMean,
    SampleAverageVariance,
    SampleAverageStdev,
    SampleAverageMeanPlusStdev,
    SampleAverageEntropicRisk,
    SampleAverageConstraint,
    SampleAverageConditionalValueAtRisk,
    CVaRSampleAverageConstraint,
    ObjectiveWithCVaRConstraints,
    LinearConstraint,
    MiniMaxOptimizer,
    EmpiricalAVaRSlackBasedOptimizer,
    AVaRSlackBasedOptimizer,
    ADAMOptimizer,
    StochasticGradientDescentOptimizer,
    SmoothLogBasedMaxFunction,
    SmoothLogBasedLeftHeavisideFunction,
    SmoothQuarticBasedLeftHeavisideFunction,
    SmoothQuinticBasedLeftHeavisideFunction,
    SampleSmoothedConditionalValueAtRisk,
    SampleSmoothedConditionalValueAtRiskDeviation,
    ChainRuleArrays,
    ChainRuleFunctions,
)
from pyapprox.optimization.risk import GaussianAnalyticalRiskMeasures
from pyapprox.benchmarks import (
    RosenbrockUnconstrainedOptimizationBenchmark,
    RosenbrockConstrainedOptimizationBenchmark,
    CantileverBeamUncertainOptimizationBenchmark,
    EvtushenkoConstrainedOptimizationBenchmark,
)
from pyapprox.interface.model import ModelFromSingleSampleCallable
from pyapprox.interface.model import Model
from pyapprox.optimization.risk import AverageValueAtRisk

from pyapprox.util.sys_utilities import package_available
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin

# from pyapprox.util.print_wrapper import *

if package_available("pyrol"):
    has_pyrol = True
    from pyapprox.optimization.rol import ROLConstrainedOptimizer
else:
    has_pyrol = False


class TestMinimize:
    def setUp(self):
        np.random.seed(1)

    def get_backend(self):
        return NumpyMixin

    def test_unconstrained_scipy_rosenbrock(self):
        # check that no bounds is handled correctly
        bkd = self.get_backend()
        for nvars in range(2, 4):
            benchmark = RosenbrockUnconstrainedOptimizationBenchmark(
                nvars=nvars, backend=bkd
            )
            optimizer = ScipyConstrainedOptimizer(benchmark.objective())
            result = optimizer.minimize(benchmark.init_iterate())
            assert bkd.allclose(result.x, benchmark.optimal_iterate())

    def test_constrained_scipy_evtushenko(self):
        bkd = self.get_backend()
        benchmark = EvtushenkoConstrainedOptimizationBenchmark(backend=bkd)
        optimizer = ScipyConstrainedOptimizer(
            benchmark.objective(),
            constraints=benchmark.constraints(),
            bounds=benchmark.design_variable().bounds(),
            opts={"gtol": 1e-15},
        )
        benchmark.objective().work_tracker().set_active(True)
        benchmark.constraints()[0].work_tracker().set_active(True)
        init_iterate = benchmark.init_iterate()
        assert bkd.allclose(
            benchmark.objective()(init_iterate), bkd.asarray(7.2)
        )
        errors = benchmark.objective().check_apply_jacobian(init_iterate)
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.objective().check_apply_hessian(init_iterate)
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.constraints()[0].check_apply_jacobian(init_iterate)
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.constraints()[0].check_apply_hessian(
            init_iterate, weights=bkd.ones((1, 1))
        )
        assert (
            benchmark.constraints()[0].work_tracker().nevaluations("whvp") == 1
        )
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.constraints()[0].check_apply_hessian(
            init_iterate, weights=bkd.ones((1, 1))
        )
        assert errors.min() / errors.max() < 1e-6

        # turn of apply_weighed_hessian and make sure weighted_hessian
        # function is used
        benchmark.constraints()[
            0
        ].apply_weighted_hessian_implemented = lambda: False
        assert (
            benchmark.constraints()[0].work_tracker().nevaluations("whess")
            == 0
        )
        errors = benchmark.constraints()[0].check_apply_hessian(
            init_iterate, weights=bkd.ones((1, 1))
        )
        # print(benchmark.constraints()[0].work_tracker())
        assert (
            benchmark.constraints()[0].work_tracker().nevaluations("whess")
            == 1
        )
        # turn apply_weighed_hessian back on
        benchmark.constraints()[
            0
        ].apply_weighted_hessian_implemented = lambda: False

        optimizer.set_verbosity(0)
        result = optimizer.minimize(init_iterate)
        assert result.nhev > 0
        assert bkd.any(bkd.asarray(result.constr_nhev) > 0)
        assert bkd.allclose(bkd.asarray(result.fun), bkd.asarray(1.0))
        assert bkd.allclose(
            result.x, bkd.array([0.0, 0.0, 1.0])[:, None], atol=1e-5
        )

    def test_constrained_scipy_rosenbrock(self):
        # check that constraints are handled correctly
        bkd = self.get_backend()
        benchmark = RosenbrockConstrainedOptimizationBenchmark(backend=bkd)
        errors = benchmark.objective().check_apply_jacobian(
            benchmark.init_iterate()
        )
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.objective().check_apply_hessian(
            benchmark.init_iterate()
        )
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.constraints()[0].check_apply_jacobian(
            benchmark.init_iterate()
        )
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.constraints()[0].check_apply_hessian(
            benchmark.init_iterate(), weights=bkd.ones((2, 1))
        )
        assert errors.min() / errors.max() < 1e-6
        optimizer = ScipyConstrainedOptimizer(
            benchmark.objective(),
            constraints=benchmark.constraints(),
            bounds=benchmark.design_variable().bounds(),
            opts={"gtol": 1e-16},
        )
        result = optimizer.minimize(benchmark.init_iterate())
        assert bkd.allclose(result.x, benchmark.optimal_iterate())

    def test_sample_average_constraints(self):
        bkd = self.get_backend()
        benchmark = CantileverBeamUncertainOptimizationBenchmark(backend=bkd)
        constraint_model = benchmark.constraints()[0]._model

        # test jacobian and hessian
        nsamples = 1000
        samples = benchmark.variable().rvs(nsamples)
        weights = bkd.full((nsamples, 1), 1 / nsamples)
        for stat in [
            SampleAverageMean(backend=bkd),
            SampleAverageVariance(backend=bkd),
            SampleAverageStdev(backend=bkd),
            SampleAverageMeanPlusStdev(2, backend=bkd),
            SampleAverageEntropicRisk(0.5, backend=bkd),
        ]:
            constraint_bounds = bkd.hstack(
                [bkd.zeros((2, 1)), bkd.full((2, 1), np.inf)]
            )
            constraint = SampleAverageConstraint(
                constraint_model,
                samples,
                weights,
                stat,
                constraint_bounds,
                benchmark.variable().nvars()
                + benchmark.design_variable().nvars(),
                benchmark.design_var_indices(),
                backend=bkd,
            )
            design_sample = bkd.array([3.0, 3.0])[:, None]
            assert constraint(design_sample).shape == (1, 2)
            errors = constraint.check_apply_jacobian(design_sample)
            # print(errors.min()/errors.max())
            assert errors.min() / errors.max() < 1.3e-6 and errors.max() > 0.2

            if not stat.hessian_implemented():
                continue
            # assert False
            errors = constraint.check_apply_hessian(
                design_sample, weights=bkd.ones((constraint.nqoi(), 1))
            )
            assert errors.min() / errors.max() < 1.3e-6 and errors.max() > 0.2

    def test_conditional_value_at_risk_gradients(self):
        bkd = self.get_backend()
        benchmark = CantileverBeamUncertainOptimizationBenchmark(backend=bkd)
        constraint_model = benchmark.constraints()[0]._model

        # test jacobian
        nsamples = 1000
        samples = benchmark.variable().rvs(nsamples)
        weights = bkd.full((nsamples, 1), 1 / nsamples)
        for stat in [
            SampleAverageConditionalValueAtRisk([0.5, 0.85], backend=bkd),
            SampleAverageConditionalValueAtRisk([0.85, 0.9], backend=bkd),
        ]:
            # first two are parameters second two are VaR
            # (one for each constraint)
            constraint_bounds = bkd.hstack(
                [bkd.zeros((4, 1)), bkd.full((4, 1), np.inf)]
            )
            constraint = CVaRSampleAverageConstraint(
                constraint_model,
                samples,
                weights,
                stat,
                constraint_bounds,
                benchmark.variable().nvars()
                + benchmark.design_variable().nvars(),
                benchmark.design_var_indices(),
                backend=bkd,
            )
            design_sample = bkd.array([3.0, 3, 1, 1])[:, None]
            assert constraint(design_sample).shape == (1, 2)
            errors = constraint.check_apply_jacobian(design_sample)
            # print(errors.min()/errors.max())
            assert errors.min() / errors.max() < 1.3e-6

    def test_conditional_value_at_risk_optimization(self):
        bkd = self.get_backend()
        # min -(s1) s.t CVaR[m1+s1*f1(z1)] <= c1 and CVaR[f2(z2)] <= c2
        mu1, sigma1 = 0, 1.1
        mu2, sigma2 = -0.5, 1.5

        # 2 random variables 2 design variables
        nrandom_vars, ndesign_vars = 2, 1
        # Model assumes ith constraint depends on ith variable
        nconstraints = nrandom_vars

        class CustomModel(Model):
            def __init__(self, nrandom_vars, backend):
                super().__init__(backend)
                self._nrandom_vars = nrandom_vars
                self._ndesign_vars = 1
                self._ident = bkd.eye(self._nrandom_vars - 1)
                self._jac = bkd.zeros(
                    (
                        self._nrandom_vars,
                        self.nvars(),
                    )
                )
                self._jac[1, 1 : self._nrandom_vars] = self._ident

            def jacobian_implemented(self):
                return True

            def hessian_implemented(self):
                return True

            def nvars(self):
                return self._nrandom_vars + self._ndesign_vars

            def nqoi(self):
                return self._nrandom_vars

            def _values(self, x):
                # x stores random vars, design_vars
                # for nvars == 2
                # f(x) = [x[0]*x[2], [x1]]
                return bkd.hstack(
                    (
                        x[:1].T
                        * x[self._nrandom_vars : self._nrandom_vars + 1].T,
                        x[1 : self._nrandom_vars].T,
                    )
                )

            def _jacobian(self, x):
                # only change the parts of jacobian that depend on x
                # must compute gradient with respect to each entry of x
                # not just design vars
                self._jac[0, 0] = x[self._nrandom_vars, 0]
                self._jac[0, -1] = x[0, 0]
                return self._jac

            def _hessian(self, x):
                hess = bkd.zeros((self.nqoi(), x.shape[0], x.shape[0]))
                hess[0, 0, 2] = 1.0
                hess[0, 2, 0] = 1.0
                return hess

        constraint_model = CustomModel(nrandom_vars, bkd)
        constraint_x0 = bkd.arange(2, nrandom_vars + ndesign_vars + 2)[:, None]
        errors = constraint_model.check_apply_jacobian(constraint_x0)
        assert errors.min() / errors.max() < 1e-6
        errors = constraint_model.check_apply_hessian(
            constraint_x0, weights=bkd.ones((2, 1))
        )
        assert errors.min() / errors.max() < 1e-6

        # objective model is just a function of design variables
        # so it is called as objective_model(design_sample)
        # no random variables are passed
        design_x0 = constraint_x0[nrandom_vars : nrandom_vars + 1]
        objective_model = ModelFromSingleSampleCallable(
            1,
            1,
            lambda x: -x.T,
            jacobian=lambda x: -bkd.ones((1, 1)),
            hessian=lambda x: bkd.zeros((1, 1, 1)),
            backend=bkd,
        )

        errors = objective_model.check_apply_jacobian(design_x0)
        assert errors.min() / errors.max() < 1e-6

        errors = objective_model.check_apply_hessian(design_x0, relative=False)
        assert errors.max() < 1e-15

        assert mu1 == 0
        nsamples = int(1e4)
        # set mu, sigma of first random samples to (0, 1) as we are learning
        # scaling so that stdev of samlpes is equal to sigma1
        samples = bkd.vstack(
            (
                bkd.asarray(np.random.normal(0, 1, (1, nsamples))),
                bkd.asarray(np.random.normal(mu2, sigma2, (1, nsamples))),
            )
        )
        weights = bkd.full((nsamples, 1), 1 / nsamples)
        # from pyapprox.surrogates.orthopoly.quadrature import (
        #     gauss_hermite_pts_wts_1D)

        # nsamples = 1000
        # samples = bkd.vstack(
        #     [gauss_hermite_pts_wts_1D(nsamples)[0],
        #      gauss_hermite_pts_wts_1D(nsamples)[0]*sigma2+mu2])
        # weights = gauss_hermite_pts_wts_1D(nsamples)[1][:, None]
        nsamples = int(1e3) + 1
        from pyapprox.surrogates.univariate.local import (
            UnivariatePiecewiseQuadraticBasis,
            UnivariateEquidistantNodeGenerator,
        )

        basis = UnivariatePiecewiseQuadraticBasis(
            stats.norm(0, 1).interval(1 - 1e-6),
            UnivariateEquidistantNodeGenerator(backend=bkd),
            backend=bkd,
        )
        basis.set_nterms(nsamples)
        nodes, weights = basis.quadrature_rule()
        nodes = nodes[0]
        # nodes = bkd.linspace(*stats.norm(0, 1).interval(1 - 1e-6), nsamples)
        # weights = basis._quadrature_rule_from_nodes(nodes[None, :])[1][:, 0]
        weights = weights * bkd.asarray(stats.norm(0, 1).pdf(nodes)[:, None])
        samples = bkd.vstack([nodes[None, :], nodes[None, :] * sigma2 + mu2])
        stat = SampleAverageConditionalValueAtRisk(
            [0.5, 0.85], eps=1e-3, backend=bkd
        )

        risks1 = GaussianAnalyticalRiskMeasures(mu1, sigma1)
        risks2 = GaussianAnalyticalRiskMeasures(mu2, sigma2)
        AVaR1 = bkd.asarray(risks1.AVaR(stat._alpha[0]))
        AVaR2 = bkd.asarray(risks2.AVaR(stat._alpha[1]))
        VaR1 = bkd.asarray(stats.norm(mu1, sigma1).ppf(stat._alpha[0]))
        VaR2 = bkd.asarray(stats.norm(mu2, sigma2).ppf(stat._alpha[1]))
        constraint_bounds = bkd.hstack(
            [bkd.zeros((2, 1)), bkd.hstack([AVaR1, AVaR2])[:, None]]
        )
        constraint = CVaRSampleAverageConstraint(
            constraint_model,
            samples,
            weights,
            stat,
            constraint_bounds,
            nrandom_vars + ndesign_vars,
            bkd.arange(nrandom_vars, nrandom_vars + ndesign_vars),
            backend=bkd,
        )
        objective = ObjectiveWithCVaRConstraints(
            objective_model, nconstraints, ndesign_vars
        )
        opt_x0 = bkd.vstack((design_x0, bkd.full((nconstraints, 1), 0.5)))
        errors = objective.check_apply_jacobian(opt_x0)
        assert errors.min() / errors.max() < 1e-6
        # rectivate when  sampleaveragecvar.hessian is implemented
        # errors = objective.check_apply_hessian(
        #     opt_x0, disp=False, relative=False
        # )
        # assert errors.max() < 1e-15
        errors = constraint.check_apply_jacobian(opt_x0)
        assert errors.min() / errors.max() < 1e-6

        exact_opt_x = bkd.array([sigma1, VaR1, VaR2])[:, None]
        # Gauss Quadrature cannot easily get accruate estimate of CVaR
        # because of discontinuity (or highly nonlinear component) of
        # (smoothed) max function
        # print(constraint(exact_opt_x)[0, :] - bkd.array([CVaR1, CVaR2]))
        assert bkd.allclose(
            constraint(exact_opt_x)[0, :],
            bkd.hstack([AVaR1, AVaR2]),
            rtol=5e-3,
        )

        bounds = bkd.stack(
            (
                bkd.hstack(
                    (bkd.zeros((1,)), bkd.full((nconstraints,), -np.inf))
                ),
                bkd.full((ndesign_vars + nconstraints,), np.inf),
            ),
            axis=1,
        )
        optimizer = ScipyConstrainedOptimizer(
            objective,
            bounds=bounds,
            constraints=[constraint],
            # tighten opt and allclose tolerances once hessian is implemented
            opts={"gtol": 6e-5, "maxiter": 5000},
        )
        optimizer.set_verbosity(0)
        result = optimizer.minimize(opt_x0)

        # errors in sample based estimate of CVaR will cause
        # optimal solution to be biased.
        assert bkd.allclose(
            constraint(result.x), bkd.hstack([AVaR1, AVaR2]), rtol=1e-2
        )
        # print(constraint(exact_opt_x), [CVaR1, CVaR2])
        # print(result.x-exact_opt_x[:, 0], exact_opt_x[:, 0])

        # TODO: on ubuntu reducing gtol causes minimize not to converge
        # ideally find reason and dencrease rtol and atol below
        # print(result.x-exact_opt_x)
        assert bkd.allclose(result.x, exact_opt_x, rtol=2e-3, atol=6e-3)
        # print(-sigma1-result.fun)
        assert bkd.allclose(
            -bkd.asarray(sigma1), bkd.asarray(result.fun), rtol=7e-3
        )

    @unittest.skipIf(not has_pyrol, "pyrol cannot be imported")
    def test_rol_minimize_constrained_rosenbrock(self):
        bkd = self.get_backend()
        nvars = 2
        benchmark = RosenbrockConstrainedOptimizationBenchmark(backend=bkd)
        optimizer = ROLConstrainedOptimizer(
            benchmark.objective(),
            constraints=benchmark.constraints(),
            bounds=benchmark.design_variable().bounds(),
        )
        result = optimizer.minimize(benchmark.init_iterate())
        assert bkd.allclose(result.x, bkd.full((nvars,), 1.0))

    def test_minimax_optimizer(self):
        bkd = self.get_backend()
        model = ModelFromSingleSampleCallable(
            3,
            3,
            lambda x: x.T**3,
            lambda x: 3 * bkd.diag(x[:, 0] ** 2),
            weighted_hessian=lambda x, w: 6 * bkd.diag(x[:, 0] * w[:, 0]),
            backend=bkd,
        )
        optimizer = ScipyConstrainedOptimizer(opts={"gtol": 1e-15})
        minimax = MiniMaxOptimizer(optimizer, backend=bkd)
        minimax.set_objective_function(model)
        minimax.set_constraints(
            [LinearConstraint(bkd.ones((3,)), 15, 15, keep_feasible=True)]
        )
        # Set bounds on model. Slack bounds are set separately
        minimax.set_bounds(
            bkd.stack(
                (bkd.full((3,), -np.inf), bkd.full((3,), np.inf)),
                axis=1,
            )
        )

        iterate = bkd.array([10000, 4, 10, 1])[:, None]
        # if get warning and optimization does not converge
        # actual_reduction = merit_function - merit_function_next
        # it is likely because _constraint_from_objective.keep_feasible=True
        # and initial guess is infeasiable
        weights = bkd.ones((3, 1))
        errors = minimax._constraint_from_objective.check_apply_jacobian(
            iterate
        )
        assert errors.min() / errors.max() < 1e-6
        errors = minimax._constraint_from_objective.check_apply_hessian(
            iterate, weights=weights
        )
        assert errors.min() / errors.max() < 1e-6

        res = minimax.minimize(iterate)
        # Constraint is active and max is found when all original variables = 5
        # print(res.x)
        assert bkd.allclose(res.x, bkd.full((3,), 5.0))
        assert bkd.allclose(
            bkd.asarray(res.fun), bkd.full((1,), 125.0), rtol=1e-2
        )

    def test_compute_avar_from_samples(self):
        bkd = self.get_backend()
        nsamples = 6
        optimizer = ScipyConstrainedOptimizer(opts={"gtol": 1e-15})
        # sub eps tp avoid numerical issue with beta falling exactly at sample
        eps = 1e-8
        beta = 4 / 6 - eps

        mu, sigma = 0, 2
        rv = stats.norm(mu, sigma)
        samples = bkd.asarray(rv.rvs(nsamples)[None, :])
        samples = bkd.sort(samples)  # hack
        quadw = bkd.full((nsamples,), 1 / nsamples)

        minimax = EmpiricalAVaRSlackBasedOptimizer(
            optimizer, beta, samples, quadw, backend=bkd
        )

        AVaR = AverageValueAtRisk(beta, backend=bkd)
        AVaR.set_samples(samples)
        # check objective function returns correct value when
        # correct value of slack variables is passed
        iterate = bkd.empty((nsamples + 1, 1))
        iterate[0] = AVaR()[1]
        iterate[1:, 0] = bkd.maximum(samples[0] - iterate[0], bkd.zeros((1,)))
        assert bkd.allclose(
            minimax._optimizer._objective(iterate), bkd.asarray(AVaR()[0])
        )

        # check gradients
        errors = minimax._constraint_from_objective.check_apply_jacobian(
            iterate, disp=False
        )
        assert errors.min() / errors.max() < 1e-6
        # weights = bkd.ones((nsamples, 1))
        # errors = minimax._constraint_from_objective.check_apply_hessian(
        #     iterate, weights=weights, disp=False
        # )
        # assert errors.min() / errors.max() < 1e-6

        iterate = bkd.full((nsamples + 1, 1), bkd.max(samples))
        # print(minimax._constraint_from_objective(iterate))
        minimax.set_bounds(None)
        optimizer.set_verbosity(0)
        res = minimax.minimize(iterate)
        opt_avar = res.fun
        opt_var = res.slack[0]

        lin_avar, lin_var = AVaR.optimize()

        # print(opt_var, lin_var, AVaR()[1], "VAR")
        # print(opt_avar, lin_avar, AVaR()[0], "AVAR")

        assert bkd.allclose(bkd.asarray(lin_avar), bkd.asarray(AVaR()[0]))
        assert bkd.allclose(bkd.asarray(lin_var), bkd.asarray(AVaR()[1]))
        assert bkd.allclose(bkd.asarray(opt_avar), bkd.asarray(AVaR()[0]))
        assert bkd.allclose(bkd.asarray(opt_var), bkd.asarray(AVaR()[1]))

    def test_avar_optimizer(self):
        bkd = self.get_backend()
        nsamples = 3
        mesh = bkd.arange(1, nsamples + 1)
        model = ModelFromSingleSampleCallable(
            nsamples,
            nsamples,
            lambda x: mesh * x**3,
            lambda x: 3 * mesh * bkd.diag(x**2),
            weighted_hessian=lambda x, w: 6 * mesh * bkd.diag(x * w[:, 0]),
            sample_ndim=1,
            values_ndim=1,
            backend=bkd,
        )
        optimizer = ScipyConstrainedOptimizer(opts={"gtol": 1e-15})
        quadw = bkd.full((nsamples,), 1 / nsamples)
        beta = 0.5
        minimax = AVaRSlackBasedOptimizer(optimizer, beta, quadw, backend=bkd)
        minimax.set_objective_function(model)
        minimax.set_constraints(
            [
                LinearConstraint(
                    bkd.ones((nsamples,)), 15, 15, keep_feasible=True
                )
            ]
        )
        minimax.set_bounds(
            bkd.stack(
                (
                    bkd.full((nsamples,), -np.inf),
                    bkd.full((nsamples,), np.inf),
                ),
                axis=1,
            )
        )

        iterate = bkd.ones((nsamples * 2 + 1, 1))
        weights = bkd.ones((nsamples, 1))
        errors = minimax._constraint_from_objective.check_apply_jacobian(
            iterate
        )
        assert errors.min() / errors.max() < 1e-6
        errors = minimax._constraint_from_objective.check_apply_hessian(
            iterate, weights=weights, disp=False
        )
        assert errors.min() / errors.max() < 1e-6

        iterate[0] = 1e6
        res = minimax.minimize(iterate)
        # Constraint is active and max is found when all original variables = 5
        # print(res.x)
        # print(res.slack)
        # print(res.fun, "f")
        # print(bkd.mean(model(res.x)))
        # print(bkd.median(model(res.x)))
        raise NotImplementedError

    def _adam_objective(self):
        bkd = self.get_backend()

        def objective(x):
            return bkd.sum(x**2, axis=0)[:, None]

        def jacobian(x):
            return 2 * x.T

        return ModelFromSingleSampleCallable(
            1, 3, objective, jacobian, backend=bkd
        )

    def test_adam_update(self):
        bkd = self.get_backend()

        # Test update function without clipping
        opt = ADAMOptimizer()
        opt.set_options(
            learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8
        )
        # set iter so iterate is not based on raising moments to zeroth power
        # which will make answer the same whether clipping is used or not
        opt._iter = 1
        opt.set_objective_function(self._adam_objective())
        iterate = bkd.array([1.0, 2.0, 3.0])[:, None]
        jacobian = opt._objective.jacobian(iterate)
        new_iterate = opt.update(iterate, jacobian)
        assert not bkd.allclose(iterate, new_iterate)

        # Test update function with clipping
        opt_clip = ADAMOptimizer()
        opt_clip.set_options(
            learning_rate=0.1,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            clip=(3.0, 5.0),
        )
        # set iter so iterate is not based on raising moments to zeroth power
        # which will make answer the same whether clipping is used or not
        opt._iter = 1
        opt_clip.set_objective_function(self._adam_objective())
        iterate = bkd.array([1.0, 2.0, 3.0])[:, None]
        jacobian = opt_clip._objective.jacobian(iterate)
        new_clip_iterate = opt_clip.update(iterate, jacobian)
        assert not bkd.allclose(opt._first_mom, opt_clip._first_mom)
        assert not bkd.allclose(opt._second_mom, opt_clip._second_mom)
        assert not bkd.allclose(new_iterate, new_clip_iterate)

    def test_adam_zero_grad(self):
        bkd = self.get_backend()
        opt = ADAMOptimizer()
        opt.set_objective_function(self._adam_objective())
        opt.set_options(
            learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8
        )
        opt.zero_grad()
        self.assertIsNone(opt._first_mom)
        self.assertIsNone(opt._second_mom)

    def test_adam_minimize(self):
        bkd = self.get_backend()
        opt = ADAMOptimizer()
        opt.set_options(
            learning_rate=0.1,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            maxiters=1000,
        )
        opt.set_objective_function(self._adam_objective())
        iterate = bkd.array([1.0, 2.0, 3.0])[:, None]
        result = opt.minimize(iterate)
        assert result.fun < 1e-4
        assert result.message == "gtol reached"

    def test_adam_set_options(self):
        bkd = self.get_backend()
        opt = ADAMOptimizer()
        opt.set_options(
            learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8
        )
        self.assertEqual(opt._learning_rate, 0.1)
        self.assertEqual(opt._beta1, 0.9)
        self.assertEqual(opt._beta2, 0.999)
        self.assertEqual(opt._epsilon, 1e-8)

    def test_init_gradient_descent(self):
        bkd = self.get_backend()
        opt = StochasticGradientDescentOptimizer()
        opt.set_objective_function(self._adam_objective())
        opt.set_options(maxiters=10, learning_rate=1e-3)
        self.assertEqual(opt._maxiters, 10)
        self.assertEqual(opt._learning_rate, 1e-3)
        self.assertEqual(opt._bkd, bkd)

    def test_step_from_objective_gradient_descent(self):
        bkd = self.get_backend()
        opt = StochasticGradientDescentOptimizer()
        opt.set_objective_function(self._adam_objective())
        iterate = bkd.array([4.0, 5.0, 6.0])[:, None]
        (
            val,
            grad,
            new_iterate,
        ) = StochasticGradientDescentOptimizer()._step_from_objective(
            opt._objective, iterate
        )
        self.assertEqual(val, opt._objective(iterate))
        assert opt._bkd.allclose(
            new_iterate, iterate - 1e-3 * opt._objective.jacobian(iterate).T
        )

    def test_minimize_gradient_descent(self):
        bkd = self.get_backend()
        iterate = bkd.array([1.0, 1.0, 2.0])[:, None]
        opt = StochasticGradientDescentOptimizer()
        opt.set_options(maxiters=1000, learning_rate=1e-2)
        opt.set_verbosity(0)
        opt.set_objective_function(self._adam_objective())
        result = opt.minimize(iterate)
        assert result["fun"] < 1e-4
        assert result["gnorm"] < 1e-4

    def test_smooth_log_based_max_function(self):
        bkd = self.get_backend()
        nsamples = 5
        samples = bkd.linspace(-1, 1, nsamples)[None, :]
        fun = SmoothLogBasedMaxFunction(2, 1e-1, 1e2, 1e-1, backend=bkd)
        errors = fun.check_first_derivative(samples, disp=False)
        assert errors.min() / errors.max() < 1e-6
        errors = fun.check_second_derivative(samples, disp=False)
        assert errors.min() / errors.max() < 1e-6
        errors = fun.check_third_derivative(samples, disp=False)
        assert errors.min() / errors.max() < 1e-6

    def test_smooth_log_based_left_heaviside_function(self):
        bkd = self.get_backend()
        nsamples = 5
        samples = bkd.linspace(-1, 1, nsamples)[None, :]
        fun = SmoothLogBasedLeftHeavisideFunction(
            2, 1e-2, 1e2, 1e-2, backend=bkd
        )
        errors = fun.check_first_derivative(samples, disp=False)
        assert errors.min() / errors.max() < 1e-6
        errors = fun.check_second_derivative(samples, disp=False)
        assert errors.min() / errors.max() < 1e-6

    def test_smooth_quartic_based_left_heaviside_function(self):
        bkd = self.get_backend()
        nsamples = 101
        eps = 1e-2
        samples = bkd.linspace(-3e-1, 3e-1, nsamples)[None, :]
        fun = SmoothQuarticBasedLeftHeavisideFunction(2, eps, backend=bkd)
        errors = fun.check_first_derivative(samples, disp=False)
        assert errors.min() / errors.max() < 1e-6
        errors = fun.check_second_derivative(samples, disp=False)
        print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 1e-6

    def test_smooth_quintic_based_left_heaviside_function(self):
        bkd = self.get_backend()
        nsamples = 101
        eps = 1e-2
        samples = bkd.linspace(-3e-1, 3e-1, nsamples)[None, :]
        fun = SmoothQuinticBasedLeftHeavisideFunction(2, eps, backend=bkd)
        errors = fun.check_first_derivative(samples, disp=False)
        assert errors.min() / errors.max() < 1e-6
        errors = fun.check_second_derivative(samples, disp=False)
        assert errors.min() / errors.max() < 1e-6

    def test_smoothed_conditional_value_at_risk(self):
        bkd = self.get_backend()
        mu, sigma, beta = 0, 1, 0.5
        risks = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_avar = bkd.asarray(risks.AVaR(beta))
        AVaR = AverageValueAtRisk(beta, backend=bkd)
        rv = stats.norm(mu, sigma)
        nsamples = int(1e5)
        samples = bkd.asarray(rv.rvs(nsamples)[None, :])
        AVaR.set_samples(samples)
        assert bkd.allclose(AVaR()[0], exact_avar, rtol=1e-2)

        smooth_avar = SampleSmoothedConditionalValueAtRisk(
            alpha=beta, eps=1000, backend=bkd
        )
        nsamples = int(1e6)
        # samples = bkd.asarray(rv.rvs(nsamples)[None, :])
        # a more accurate ansswer can be obtained by using 1D
        # halton like sequence, i.e. equidistant points
        samples = bkd.asarray(rv.ppf(bkd.linspace(1e-6, 1 - 1e-6, nsamples)))[
            None, :
        ]
        weights = bkd.full((nsamples, 1), 1 / nsamples)
        # print(smooth_avar(samples.T, weights) - exact_avar, exact_avar)
        assert bkd.allclose(
            smooth_avar(samples.T, weights), exact_avar, rtol=1e-5
        )

        # Test avar with importance sampling
        smooth_avar = SampleSmoothedConditionalValueAtRisk(
            alpha=beta, eps=1000, backend=bkd
        )
        nsamples = int(1e6)
        # samples = bkd.asarray(rv.rvs(nsamples)[None, :])
        # a more accurate ansswer can be obtained by using 1D
        # halton like sequence, i.e. equidistant points
        dominating_rv = stats.norm(mu, sigma * 2)
        samples = bkd.asarray(
            dominating_rv.ppf(bkd.linspace(1e-6, 1 - 1e-6, nsamples))
        )[None, :]
        weights = bkd.full((nsamples, 1), 1 / nsamples)
        weights *= bkd.asarray(
            rv.pdf(samples.T) / dominating_rv.pdf(samples.T)
        )
        # print(smooth_avar(samples.T, weights) - exact_avar, exact_avar)
        assert bkd.allclose(
            smooth_avar(samples.T, weights), exact_avar, rtol=1e-5
        )

        # check Jacobian
        def fun(samples, params):
            return params[0] + bkd.sqrt(params[1]) * samples.T

        def fjac(samples, params):
            return bkd.stack(
                (
                    bkd.ones((samples.shape[1],)),
                    samples[0] / (2 * bkd.sqrt(params[1])),
                ),
                axis=1,
            )

        model = ModelFromSingleSampleCallable(
            1,
            2,
            lambda p: bkd.atleast2d(smooth_avar(fun(samples, p), weights)),
            lambda p: smooth_avar.jacobian(
                fun(samples, p), fjac(samples, p), weights
            ),
            backend=bkd,
        )
        params = bkd.asarray([[1.0, 2.0]]).T
        # params defines a new distribution with mean 1.0 and variance 2.0
        # assuming rv is a standard normal
        risks = GaussianAnalyticalRiskMeasures(1.0, math.sqrt(2.0))
        exact_avar = bkd.asarray(risks.AVaR(beta))
        assert bkd.allclose(exact_avar, model(params), rtol=1e-2)
        errors = model.check_apply_jacobian(params, disp=False)
        assert errors.min() / errors.max() < 1e-6

    def test_smoothed_conditional_value_at_risk_deviation(self):
        # test smooth cvar deviation using importance sampling
        bkd = self.get_backend()
        mu, sigma, beta = 0.5, 1, 0.5
        risks = GaussianAnalyticalRiskMeasures(mu, sigma)
        exact_avardev = bkd.array(risks.AVaR(beta) - mu)
        rv = stats.norm(mu, sigma)
        smooth_avardev = SampleSmoothedConditionalValueAtRiskDeviation(
            alpha=beta, eps=100000, backend=bkd
        )
        nsamples = int(1e6)
        # samples = bkd.asarray(rv.rvs(nsamples)[None, :])
        # a more accurate ansswer can be obtained by using 1D
        # halton like sequence, i.e. equidistant points
        dominating_rv = stats.norm(mu, sigma * 2)
        np_samples = (
            dominating_rv.ppf(np.linspace(1e-6, 1 - 1e-6, nsamples))
        )[None, :]
        weights = bkd.full((nsamples, 1), 1 / nsamples)
        weights *= bkd.asarray(
            rv.pdf(np_samples.T) / dominating_rv.pdf(np_samples.T)
        )
        smooth_avardev.set_mean(mu)
        print(
            smooth_avardev(bkd.asarray(np_samples).T, weights), exact_avardev
        )
        print(
            smooth_avardev(bkd.asarray(np_samples).T, weights) - exact_avardev
        )
        assert bkd.allclose(
            smooth_avardev(bkd.asarray(np_samples).T, weights),
            exact_avardev,
            rtol=1e-5,
        )


class TestChainRule:
    def _uncompress_jacobian(self, jac_compressed):
        bkd = self.get_backend()
        N, n_o, n_p = jac_compressed.shape
        jac_uncompressed = bkd.zeros((N, n_o, N, n_p))

        # Populate the diagonal blocks
        for n in range(N):
            jac_uncompressed[n, :, n, :] = jac_compressed[n]

        return jac_uncompressed

    def _define_jacobians(self):
        bkd = self.get_backend()
        self._x_jac_compressed = (
            lambda p: bkd.cos((p @ self._A.T) + self._b)[:, :, None] * self._A
        )
        self._u_jac_compressed = (
            lambda x: -bkd.sin((x @ self._B.T) + self._c)[:, :, None] * self._B
        )
        self._x_jac_uncompressed = lambda p: self._uncompress_jacobian(
            self._x_jac_compressed(p)
        )
        self._u_jac_uncompressed = lambda x: self._uncompress_jacobian(
            self._u_jac_compressed(x)
        )

    def _precompute_arrays(self):
        bkd = self.get_backend()
        self._x = self._x_function(self._p)
        self._u = self._u_function(self._x)
        self._dx_dp_uncompressed = self._x_jac_uncompressed(self._p)
        self._du_dx_uncompressed = self._u_jac_uncompressed(self._x)
        self._dx_dp_compressed = self._x_jac_compressed(self._p)
        self._du_dx_compressed = self._u_jac_compressed(self._x)
        self._du_dp = bkd.einsum(
            "noi,nip->nop", self._du_dx_compressed, self._dx_dp_compressed
        )

    def setUp(self):
        bkd = self.get_backend()
        np.random.seed(1)
        # Define dimensions
        self._N = 2
        self._n_p = 3
        self._n_i = 4
        self._n_o = 5

        # Define random matrices and biases for affine transformations
        self._A = bkd.array(np.random.randn(self._n_i, self._n_p))
        self._b = bkd.array(np.random.randn(self._n_i))
        self._B = bkd.array(np.random.randn(self._n_o, self._n_i))
        self._c = bkd.array(np.random.randn(self._n_o))

        # Define p
        self._p = bkd.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Define functions
        self._x_function = lambda p: bkd.sin((p @ self._A.T) + self._b)
        self._u_function = lambda x: bkd.cos((x @ self._B.T) + self._c)

        # Define Jacobians
        self._define_jacobians()

        # Precompute arrays
        self._precompute_arrays()

    def test_jacobian_compression(self):
        bkd = self.get_backend()
        chain_rule = ChainRuleFunctions(
            self._x_function,
            self._u_function,
            self._x_jac_compressed,
            self._u_jac_compressed,
            True,
            bkd,
        )
        assert bkd.allclose(
            chain_rule._uncompress_jacobian(self._dx_dp_compressed),
            self._dx_dp_uncompressed,
        )
        assert bkd.allclose(
            chain_rule._compress_jacobian(self._dx_dp_uncompressed),
            self._dx_dp_compressed,
        )

    def test_chain_rule_functions_no_compression(self):
        bkd = self.get_backend()
        chain_rule = ChainRuleFunctions(
            self._x_function,
            self._u_function,
            self._x_jac_compressed,
            self._u_jac_compressed,
            False,
            bkd,
        )
        du_dp = chain_rule(self._p)
        self.assertEqual(du_dp.shape, (self._N, self._n_o, self._n_p))
        assert bkd.allclose(du_dp, self._du_dp)

    def test_chain_rule_functions_with_compression(self):
        bkd = self.get_backend()
        chain_rule = ChainRuleFunctions(
            self._x_function,
            self._u_function,
            self._x_jac_uncompressed,
            self._u_jac_uncompressed,
            True,
            bkd,
        )
        du_dp = chain_rule(self._p)
        self.assertEqual(du_dp.shape, (self._N, self._n_o, self._n_p))
        assert bkd.allclose(du_dp, self._du_dp)

    def test_chain_rule_arrays_no_compression(self):
        bkd = self.get_backend()
        chain_rule = ChainRuleArrays(
            self._x,
            self._u,
            self._dx_dp_compressed,
            self._du_dx_compressed,
            False,
            bkd,
        )
        du_dp = chain_rule(self._p)
        self.assertEqual(du_dp.shape, (self._N, self._n_o, self._n_p))
        assert bkd.allclose(du_dp, self._du_dp)

    def test_chain_rule_arrays_with_compression(self):
        bkd = self.get_backend()
        chain_rule = ChainRuleArrays(
            self._x,
            self._u,
            self._dx_dp_uncompressed,
            self._du_dx_uncompressed,
            True,
            bkd,
        )
        du_dp = chain_rule(self._p)
        self.assertEqual(du_dp.shape, (self._N, self._n_o, self._n_p))
        assert bkd.allclose(du_dp, self._du_dp)


class TestNumpyMinimize(TestMinimize, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchMinimize(TestMinimize, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


class TestNumpyChainRule(TestChainRule, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchChainRule(TestChainRule, unittest.TestCase):
    def get_backend(self):
        return TorchMixin

    def _define_jacobians(self):
        bkd = self.get_backend()
        super()._define_jacobians()
        # show how to use autograd to compute components of chain rule
        # self._x_jac_uncompressed = lambda p: bkd.jacobian(self._x_function, p)
        # self._u_jac_uncompressed = lambda x: bkd.jacobian(self._u_function, x)

    def _precompute_arrays(self):
        bkd = self.get_backend()
        super()._precompute_arrays()
        self._du_dp = bkd.sum(
            bkd.jacobian(
                lambda p: self._u_function(self._x_function(p)), self._p
            ),
            axis=2,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
