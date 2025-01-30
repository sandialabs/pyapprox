import unittest

import numpy as np
from scipy import stats

from pyapprox.optimization.pya_minimize import (
    ScipyConstrainedOptimizer,
    Bounds,
    ConstraintFromModel,
    SampleAverageMean,
    SampleAverageVariance,
    SampleAverageStdev,
    SampleAverageMeanPlusStdev,
    SampleAverageEntropicRisk,
    SampleAverageConstraint,
    SampleAverageConditionalValueAtRisk,
    CVaRSampleAverageConstraint,
    ObjectiveWithCVaRConstraints,
)
from pyapprox.benchmarks import (
    RosenbrockUnconstrainedOptimizationBenchmark,
    RosenbrockConstrainedOptimizationBenchmark,
    CantileverBeamUncertainOptimizationBenchmark,
    EvtushenkoConstrainedOptimizationBenchmark,
)
from pyapprox.interface.model import ModelFromSingleSampleCallable
from pyapprox.variables.risk import gaussian_cvar
from pyapprox.interface.model import Model

from pyapprox.util.sys_utilities import package_available

if package_available("pyrol"):
    has_pyrol = True
    from pyapprox.optimization.rol_minimize import ROLConstrainedOptimizer
else:
    has_pyrol = False


class TestMinimize(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

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
        benchmark.constraints()[0]._apply_weighted_hessian_implemented = False
        assert (
            benchmark.constraints()[0].work_tracker().nevaluations("whess")
            == 0
        )
        errors = benchmark.constraints()[0].check_apply_hessian(
            init_iterate, weights=np.ones((1, 1))
        )
        assert (
            benchmark.constraints()[0].work_tracker().nevaluations("whess")
            == 1
        )
        # turn apply_weighed_hessian back on
        benchmark.constraints()[0]._apply_weighted_hessian_implemented = False

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
        errors = benchmark.constraints()[0].check_apply_jacobian(
            benchmark.init_iterate()
        )
        assert errors.min() / errors.max() < 1e-6
        errors = benchmark.constraints()[0].check_apply_hessian(
            benchmark.init_iterate(), weights=np.ones(2)
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
                benchmark.variable().num_vars()
                + benchmark.design_variable().nvars(),
                benchmark.design_var_indices(),
            )
            design_sample = np.array([3, 3])[:, None]
            assert constraint(design_sample).shape == (1, 2)
            errors = constraint.check_apply_jacobian(design_sample)
            # print(errors.min()/errors.max())
            assert errors.min() / errors.max() < 1.3e-6 and errors.max() > 0.2

            if not stat._hessian_implemented:
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
                benchmark.variable().num_vars()
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
        # Model assumes ith contraint depends on ith variable
        nconstraints = nrandom_vars

        class CustomModel(Model):
            def __init__(self, nrandom_vars):
                super().__init__()
                self._jacobian_implemented = True
                self._hessian_implemented = True
                self._nrandom_vars = nrandom_vars
                self._ndesign_vars = 1
                self._ident = np.eye(self._nrandom_vars - 1)
                self._jac = np.zeros(
                    (
                        self._nrandom_vars,
                        self._nrandom_vars + self._ndesign_vars,
                    )
                )
                self._jac[1, 1 : self._nrandom_vars] = self._ident

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
        from pyapprox.surrogates.interp.tensorprod import (
            UnivariatePiecewiseQuadraticBasis,
        )

        basis = UnivariatePiecewiseQuadraticBasis()
        nodes = np.linspace(*stats.norm(0, 1).interval(1 - 1e-6), nsamples)
        weights = basis._quadrature_rule_from_nodes(nodes[None, :])[1][:, 0]
        weights = (weights * stats.norm(0, 1).pdf(nodes))[:, None]
        samples = np.vstack([nodes[None, :], nodes[None, :] * sigma2 + mu2])
        stat = SampleAverageConditionalValueAtRisk([0.5, 0.85], eps=1e-3)

        CVaR1 = gaussian_cvar(mu1, sigma1, stat._alpha[0])
        CVaR2 = gaussian_cvar(mu2, sigma2, stat._alpha[1])
        VaR1 = stats.norm(mu1, sigma1).ppf(stat._alpha[0])
        VaR2 = stats.norm(mu2, sigma2).ppf(stat._alpha[1])
        constraint_bounds = np.hstack(
            [np.zeros((2, 1)), np.hstack([CVaR1, CVaR2])[:, None]]
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
        objective = ObjectiveWithCVaRConstraints(objective_model, nconstraints)
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
            constraint(exact_opt_x)[0, :], [CVaR1, CVaR2], rtol=5e-3
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
        assert np.allclose(constraint(result.x), [CVaR1, CVaR2], rtol=1e-2)
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
        optimizer = ROLConstrainedOptimizer(benchmark.objective())
        result = optimizer.minimize(benchmark.init_iterate())
        assert np.allclose(result.x, np.full(nvars, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
