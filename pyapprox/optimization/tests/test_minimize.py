import unittest

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

# from pyapprox.util.print_wrapper import *

if package_available("pyrol"):
    has_pyrol = True
    from pyapprox.optimization.rol import ROLConstrainedOptimizer
else:
    has_pyrol = False


class TestMinimize(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def get_backend(self):
        return NumpyMixin

    def test_unconstrained_scipy_rosenbrock(self):
        # check that no bounds is handled correctly
        for nvars in range(2, 4):
            benchmark = RosenbrockUnconstrainedOptimizationBenchmark(
                nvars=nvars
            )
            optimizer = ScipyConstrainedOptimizer(benchmark.objective())
            result = optimizer.minimize(benchmark.init_iterate())
            assert np.allclose(result.x, benchmark.optimal_iterate())

    def test_constrained_scipy_evtushenko(self):
        benchmark = EvtushenkoConstrainedOptimizationBenchmark()
        optimizer = ScipyConstrainedOptimizer(
            benchmark.objective(),
            constraints=benchmark.constraints(),
            bounds=benchmark.design_variable().bounds(),
            opts={"gtol": 1e-15},
        )
        benchmark.objective().work_tracker().set_active(True)
        benchmark.constraints()[0].work_tracker().set_active(True)
        init_iterate = benchmark.init_iterate()
        assert np.allclose(benchmark.objective()(init_iterate), 7.2)
        errors = benchmark.objective().check_apply_jacobian(init_iterate)
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.objective().check_apply_hessian(init_iterate)
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.constraints()[0].check_apply_jacobian(init_iterate)
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.constraints()[0].check_apply_hessian(
            init_iterate, weights=np.ones((1, 1))
        )
        assert (
            benchmark.constraints()[0].work_tracker().nevaluations("whvp") == 1
        )
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.constraints()[0].check_apply_hessian(
            init_iterate, weights=np.ones((1, 1))
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
            init_iterate, weights=np.ones((1, 1))
        )
        print(benchmark.constraints()[0].work_tracker())
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
        assert np.any(np.asarray(result.constr_nhev) > 0)
        assert np.allclose(result.fun, 1.0)
        assert np.allclose(
            result.x, np.array([0.0, 0.0, 1.0])[:, None], atol=1e-5
        )

    def test_constrained_scipy_rosenbrock(self):
        # check that constraints are handled correctly
        benchmark = RosenbrockConstrainedOptimizationBenchmark()
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
            benchmark.init_iterate(), weights=np.ones((2, 1))
        )
        assert errors.min() / errors.max() < 1e-6
        optimizer = ScipyConstrainedOptimizer(
            benchmark.objective(),
            constraints=benchmark.constraints(),
            bounds=benchmark.design_variable().bounds(),
            opts={"gtol": 1e-16},
        )
        result = optimizer.minimize(benchmark.init_iterate())
        assert np.allclose(result.x, benchmark.optimal_iterate())

    def test_sample_average_constraints(self):
        benchmark = CantileverBeamUncertainOptimizationBenchmark()
        constraint_model = benchmark.constraints()[0]._model

        # test jacobian and hessian
        nsamples = 1000
        samples = benchmark.variable().rvs(nsamples)
        weights = np.full((nsamples, 1), 1 / nsamples)
        for stat in [
            SampleAverageMean(),
            SampleAverageVariance(),
            SampleAverageStdev(),
            SampleAverageMeanPlusStdev(2),
            SampleAverageEntropicRisk(0.5),
        ]:
            constraint_bounds = np.hstack(
                [np.zeros((2, 1)), np.full((2, 1), np.inf)]
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
            )
            design_sample = np.array([3, 3])[:, None]
            assert constraint(design_sample).shape == (1, 2)
            errors = constraint.check_apply_jacobian(design_sample)
            # print(errors.min()/errors.max())
            assert errors.min() / errors.max() < 1.3e-6 and errors.max() > 0.2

            if not stat.hessian_implemented():
                continue
            # assert False
            errors = constraint.check_apply_hessian(
                design_sample, weights=np.ones((constraint.nqoi(), 1))
            )
            assert errors.min() / errors.max() < 1.3e-6 and errors.max() > 0.2

    def test_conditional_value_at_risk_gradients(self):
        benchmark = CantileverBeamUncertainOptimizationBenchmark()
        constraint_model = benchmark.constraints()[0]._model

        # test jacobian
        nsamples = 1000
        samples = benchmark.variable().rvs(nsamples)
        weights = np.full((nsamples, 1), 1 / nsamples)
        for stat in [
            SampleAverageConditionalValueAtRisk([0.5, 0.85]),
            SampleAverageConditionalValueAtRisk([0.85, 0.9]),
        ]:
            # first two are parameters second two are VaR
            # (one for each constraint)
            constraint_bounds = np.hstack(
                [np.zeros((4, 1)), np.full((4, 1), np.inf)]
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
            )
            design_sample = np.array([3, 3, 1, 1])[:, None]
            assert constraint(design_sample).shape == (1, 2)
            errors = constraint.check_apply_jacobian(design_sample)
            # print(errors.min()/errors.max())
            assert errors.min() / errors.max() < 1.3e-6

    def test_conditional_value_at_risk_optimization(self):
        # min -(s1) s.t CVaR[m1+s1*f1(z1)] <= c1 and CVaR[f2(z2)] <= c2
        mu1, sigma1 = 0, 1.1
        mu2, sigma2 = -0.5, 1.5

        # 2 random variables 2 design variables
        nrandom_vars, ndesign_vars = 2, 1
        # Model assumes ith constraint depends on ith variable
        nconstraints = nrandom_vars

        class CustomModel(Model):
            def __init__(self, nrandom_vars):
                super().__init__()
                self._nrandom_vars = nrandom_vars
                self._ndesign_vars = 1
                self._ident = np.eye(self._nrandom_vars - 1)
                self._jac = np.zeros(
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
                return np.hstack(
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
                self._jac[0, 0] = x[
                    self._nrandom_vars : self._nrandom_vars + 1, 0
                ]
                self._jac[0, -1] = x[:1, 0]
                return self._jac

            def _hessian(self, x):
                hess = np.zeros((self.nqoi(), x.shape[0], x.shape[0]))
                hess[0, 0, 2] = 1.0
                hess[0, 2, 0] = 1.0
                return hess

        constraint_model = CustomModel(nrandom_vars)
        constraint_x0 = np.arange(2, nrandom_vars + ndesign_vars + 2)[:, None]
        errors = constraint_model.check_apply_jacobian(constraint_x0)
        assert errors.min() / errors.max() < 1e-6
        errors = constraint_model.check_apply_hessian(
            constraint_x0, weights=np.ones((2, 1))
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
            jacobian=lambda x: -np.ones((1, 1)),
            hessian=lambda x: np.zeros((1, 1, 1)),
        )
        errors = objective_model.check_apply_jacobian(design_x0)
        assert errors.min() / errors.max() < 1e-6

        errors = objective_model.check_apply_hessian(design_x0, relative=False)
        assert errors.max() < 1e-15

        assert mu1 == 0
        nsamples = int(1e4)
        # set mu, sigma of first random samples to (0, 1) as we are learning
        # scaling so that stdev of samlpes is equal to sigma1
        samples = np.vstack(
            (
                np.random.normal(0, 1, (1, nsamples)),
                np.random.normal(mu2, sigma2, (1, nsamples)),
            )
        )
        weights = np.full((nsamples, 1), 1 / nsamples)
        # from pyapprox.surrogates.orthopoly.quadrature import (
        #     gauss_hermite_pts_wts_1D)

        # nsamples = 1000
        # samples = np.vstack(
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
            UnivariateEquidistantNodeGenerator(),
        )
        basis.set_nterms(nsamples)
        nodes, weights = basis.quadrature_rule()
        nodes = nodes[0]
        # nodes = np.linspace(*stats.norm(0, 1).interval(1 - 1e-6), nsamples)
        # weights = basis._quadrature_rule_from_nodes(nodes[None, :])[1][:, 0]
        weights = weights * stats.norm(0, 1).pdf(nodes)[:, None]
        samples = np.vstack([nodes[None, :], nodes[None, :] * sigma2 + mu2])
        stat = SampleAverageConditionalValueAtRisk([0.5, 0.85], eps=1e-3)

        risks1 = GaussianAnalyticalRiskMeasures(mu1, sigma1)
        risks2 = GaussianAnalyticalRiskMeasures(mu2, sigma2)
        AVaR1 = risks1.AVaR(stat._alpha[0])
        AVaR2 = risks2.AVaR(stat._alpha[1])
        VaR1 = stats.norm(mu1, sigma1).ppf(stat._alpha[0])
        VaR2 = stats.norm(mu2, sigma2).ppf(stat._alpha[1])
        constraint_bounds = np.hstack(
            [np.zeros((2, 1)), np.hstack([AVaR1, AVaR2])[:, None]]
        )
        constraint = CVaRSampleAverageConstraint(
            constraint_model,
            samples,
            weights,
            stat,
            constraint_bounds,
            nrandom_vars + ndesign_vars,
            np.arange(nrandom_vars, nrandom_vars + ndesign_vars),
        )
        objective = ObjectiveWithCVaRConstraints(
            objective_model, nconstraints, ndesign_vars
        )
        opt_x0 = np.vstack((design_x0, np.full((nconstraints, 1), 0.5)))
        errors = objective.check_apply_jacobian(opt_x0)
        assert errors.min() / errors.max() < 1e-6
        # rectivate when  sampleaveragecvar.hessian is implemented
        # errors = objective.check_apply_hessian(
        #     opt_x0, disp=True, relative=False
        # )
        # assert errors.max() < 1e-15
        errors = constraint.check_apply_jacobian(opt_x0)
        assert errors.min() / errors.max() < 1e-6

        exact_opt_x = np.array([sigma1, VaR1, VaR2])[:, None]
        # Gauss Quadrature cannot easily get accruate estimate of CVaR
        # because of discontinuity (or highly nonlinear component) of
        # (smoothed) max function
        # print(constraint(exact_opt_x)[0, :] - np.array([CVaR1, CVaR2]))
        assert np.allclose(
            constraint(exact_opt_x)[0, :], [AVaR1, AVaR2], rtol=5e-3
        )

        bounds = np.stack(
            (
                np.hstack(([0], np.full((nconstraints,), -np.inf))),
                np.full((ndesign_vars + nconstraints,), np.inf),
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
        optimizer.set_verbosity(1)
        result = optimizer.minimize(opt_x0)

        # errors in sample based estimate of CVaR will cause
        # optimal solution to be biased.
        assert np.allclose(constraint(result.x), [AVaR1, AVaR2], rtol=1e-2)
        # print(constraint(exact_opt_x), [CVaR1, CVaR2])
        # print(result.x-exact_opt_x[:, 0], exact_opt_x[:, 0])

        # TODO: on ubuntu reducing gtol causes minimize not to converge
        # ideally find reason and dencrease rtol and atol below
        # print(result.x-exact_opt_x)
        assert np.allclose(result.x, exact_opt_x, rtol=2e-3, atol=6e-3)
        # print(-sigma1-result.fun)
        assert np.allclose(-sigma1, result.fun, rtol=7e-3)

    @unittest.skipIf(not has_pyrol, "pyrol cannot be imported")
    def test_rol_minimize_constrained_rosenbrock(self):
        nvars = 2
        benchmark = RosenbrockConstrainedOptimizationBenchmark()
        optimizer = ROLConstrainedOptimizer(
            benchmark.objective(),
            constraints=benchmark.constraints(),
            bounds=benchmark.design_variable().bounds(),
        )
        result = optimizer.minimize(benchmark.init_iterate())
        assert np.allclose(result.x, np.full(nvars, 1))

    def test_minimax_optimizer(self):
        bkd = NumpyMixin
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
        print(res.x)
        assert bkd.allclose(res.x, bkd.full((3,), 5.0))
        assert bkd.allclose(res.fun, bkd.full((1,), 125.0), rtol=1e-2)

    def test_compute_avar_from_samples(self):
        bkd = NumpyMixin
        nsamples = 6
        optimizer = ScipyConstrainedOptimizer(opts={"gtol": 1e-15})
        # sub eps tp avoid numerical issue with beta falling exactly at sample
        eps = 1e-8
        beta = 4 / 6 - eps

        mu, sigma = 0, 2
        rv = stats.norm(mu, sigma)
        samples = rv.rvs(nsamples)[None, :]
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
        iterate[1:, 0] = bkd.maximum(samples[0] - iterate[0], 0)
        assert bkd.allclose(minimax._optimizer._objective(iterate), AVaR()[0])

        # check gradients
        errors = minimax._constraint_from_objective.check_apply_jacobian(
            iterate, disp=True
        )
        assert errors.min() / errors.max() < 1e-6
        # weights = bkd.ones((nsamples, 1))
        # errors = minimax._constraint_from_objective.check_apply_hessian(
        #     iterate, weights=weights, disp=True
        # )
        # assert errors.min() / errors.max() < 1e-6

        iterate = bkd.full((nsamples + 1, 1), bkd.max(samples))
        print(minimax._constraint_from_objective(iterate))
        minimax.set_bounds(None)
        optimizer.set_verbosity(3)
        res = minimax.minimize(iterate)
        opt_avar = res.fun
        opt_var = res.slack[0]

        lin_avar, lin_var = AVaR.optimize()

        print(opt_var, lin_var, AVaR()[1], "VAR")
        print(opt_avar, lin_avar, AVaR()[0], "AVAR")

        assert bkd.allclose(lin_avar, AVaR()[0])
        assert bkd.allclose(lin_var, AVaR()[1])
        assert bkd.allclose(opt_avar, AVaR()[0])
        assert bkd.allclose(opt_var, AVaR()[1])

    def test_avar_optimizer(self):
        bkd = NumpyMixin
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
            iterate, weights=weights, disp=True
        )
        assert errors.min() / errors.max() < 1e-6

        iterate[0] = 1e6
        res = minimax.minimize(iterate)
        # Constraint is active and max is found when all original variables = 5
        print(res.x)
        print(res.slack)
        print(res.fun, "f")
        print(np.mean(model(res.x)))
        print(np.median(model(res.x)))
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
        errors = fun.check_first_derivative(samples, disp=True)
        assert errors.min() / errors.max() < 1e-6
        errors = fun.check_second_derivative(samples, disp=True)
        assert errors.min() / errors.max() < 1e-6
        errors = fun.check_third_derivative(samples, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_smooth_log_based_left_heaviside_function(self):
        bkd = self.get_backend()
        nsamples = 5
        samples = bkd.linspace(-1, 1, nsamples)[None, :]
        fun = SmoothLogBasedLeftHeavisideFunction(
            2, 1e-2, 1e2, 1e-2, backend=bkd
        )
        errors = fun.check_first_derivative(samples, disp=True)
        assert errors.min() / errors.max() < 1e-6
        errors = fun.check_second_derivative(samples, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_smooth_quartic_based_left_heaviside_function(self):
        bkd = self.get_backend()
        nsamples = 101
        eps = 1e-2
        samples = bkd.linspace(-3e-1, 3e-1, nsamples)[None, :]
        fun = SmoothQuarticBasedLeftHeavisideFunction(2, eps, backend=bkd)
        errors = fun.check_first_derivative(samples, disp=True)
        assert errors.min() / errors.max() < 1e-6
        errors = fun.check_second_derivative(samples, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_smooth_quintic_based_left_heaviside_function(self):
        bkd = self.get_backend()
        nsamples = 101
        eps = 1e-2
        samples = bkd.linspace(-3e-1, 3e-1, nsamples)[None, :]
        fun = SmoothQuinticBasedLeftHeavisideFunction(2, eps, backend=bkd)
        errors = fun.check_first_derivative(samples, disp=True)
        assert errors.min() / errors.max() < 1e-6
        errors = fun.check_second_derivative(samples, disp=True)
        assert errors.min() / errors.max() < 1e-6


if __name__ == "__main__":
    unittest.main(verbosity=2)
