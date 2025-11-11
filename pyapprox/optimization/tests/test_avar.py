import unittest

import numpy as np
from scipy import stats

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.benchmarks import CantileverBeamUncertainOptimizationBenchmark
from pyapprox.optimization.avar import (
    AVaRSampleAverageConstraint,
    SampleAverageAverageValueAtRisk,
    ObjectiveWithAVaRConstraints,
    AVaRSlackBasedOptimizer,
    EmpiricalAVaRSlackBasedOptimizer,
)
from pyapprox.interface.model import Model, ModelFromSingleSampleCallable
from pyapprox.optimization.risk import (
    GaussianAnalyticalRiskMeasures,
    AverageValueAtRisk,
)
from pyapprox.optimization.scipy import (
    ScipyConstrainedOptimizer,
    LinearConstraint,
)


class TestAVaR:
    def setUp(self):
        np.random.seed(1)

    def test_conditional_value_at_risk_gradients(self):
        bkd = self.get_backend()
        benchmark = CantileverBeamUncertainOptimizationBenchmark(backend=bkd)
        constraint_model = benchmark.constraints()[0]._model

        # test jacobian
        nsamples = 1000
        samples = benchmark.prior().rvs(nsamples)
        weights = bkd.full((nsamples, 1), 1 / nsamples)
        for stat in [
            SampleAverageAverageValueAtRisk([0.5, 0.85], backend=bkd),
            SampleAverageAverageValueAtRisk([0.85, 0.9], backend=bkd),
        ]:
            # first two are parameters second two are VaR
            # (one for each constraint)
            constraint_bounds = bkd.hstack(
                [bkd.zeros((4, 1)), bkd.full((4, 1), np.inf)]
            )
            constraint = AVaRSampleAverageConstraint(
                constraint_model,
                samples,
                weights,
                stat,
                constraint_bounds,
                benchmark.prior().nvars()
                + benchmark.design_variable().nvars(),
                benchmark.design_var_indices(),
                backend=bkd,
            )
            design_sample = bkd.array([3.0, 3, 1, 1])[:, None]
            assert constraint(design_sample).shape == (1, 2)
            errors = constraint.check_apply_jacobian(design_sample)
            # print(errors.min()/errors.max())
            assert errors.min() / errors.max() < 1.3e-6

    def test_average_value_at_risk_optimization(self):
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
        stat = SampleAverageAverageValueAtRisk(
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
        constraint = AVaRSampleAverageConstraint(
            constraint_model,
            samples,
            weights,
            stat,
            constraint_bounds,
            nrandom_vars + ndesign_vars,
            bkd.arange(nrandom_vars, nrandom_vars + ndesign_vars),
            backend=bkd,
        )
        objective = ObjectiveWithAVaRConstraints(
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
        # Gauss Quadrature cannot easily get accruate estimate of AVaR
        # because of discontinuity (or highly nonlinear component) of
        # (smoothed) max function
        # print(constraint(exact_opt_x)[0, :] - bkd.array([AVaR1, AVaR2]))
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

        # errors in sample based estimate of AVaR will cause
        # optimal solution to be biased.
        assert bkd.allclose(
            constraint(result.x), bkd.hstack([AVaR1, AVaR2]), rtol=1e-2
        )
        # print(constraint(exact_opt_x), [AVaR1, AVaR2])
        # print(result.x-exact_opt_x[:, 0], exact_opt_x[:, 0])

        # TODO: on ubuntu reducing gtol causes minimize not to converge
        # ideally find reason and dencrease rtol and atol below
        # print(result.x-exact_opt_x)
        assert bkd.allclose(result.x, exact_opt_x, rtol=2e-3, atol=6e-3)
        # print(-sigma1-result.fun)
        assert bkd.allclose(
            -bkd.asarray(sigma1), bkd.asarray(result.fun), rtol=7e-3
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

    @unittest.skip
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
        optimizer = AVaRSlackBasedOptimizer(
            optimizer, beta, quadw, backend=bkd
        )
        optimizer.set_objective_function(model)
        optimizer.set_constraints(
            [
                LinearConstraint(
                    bkd.ones((nsamples,)), 15, 15, keep_feasible=True
                )
            ]
        )
        optimizer.set_bounds(
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
        errors = optimizer._constraint_from_objective.check_apply_jacobian(
            iterate
        )
        assert errors.min() / errors.max() < 1e-6
        errors = optimizer._constraint_from_objective.check_apply_hessian(
            iterate, weights=weights, disp=False
        )
        assert errors.min() / errors.max() < 1e-6

        iterate[0] = 1e6
        res = optimizer.minimize(iterate)
        # Constraint is active and max is found when all original variables = 5
        # print(res.x)
        # print(res.slack)
        # print(res.fun, "f")
        # print(bkd.mean(model(res.x)))
        # print(bkd.median(model(res.x)))
        raise NotImplementedError(
            "Need to find a analytical solution or compare with smoothed "
            "avar based optimization"
        )


class TestNumpyAVaR(TestAVaR, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchAVaR(TestAVaR, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
