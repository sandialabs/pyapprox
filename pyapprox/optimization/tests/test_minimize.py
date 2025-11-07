import unittest

import numpy as np
from scipy import stats

from pyapprox.optimization.scipy import (
    ScipyConstrainedOptimizer,
    ScipyConstrainedDifferentialEvolutionOptimizer,
)
from pyapprox.optimization.minimize import (
    LinearConstraint,
    MiniMaxOptimizer,
    ADAMOptimizer,
    StochasticGradientDescentOptimizer,
    SmoothLogBasedMaxFunction,
    SmoothLogBasedLeftHeavisideFunction,
    SmoothQuarticBasedLeftHeavisideFunction,
    SmoothQuinticBasedLeftHeavisideFunction,
    ChainRuleArrays,
    ChainRuleFunctions,
)
from pyapprox.benchmarks import (
    RosenbrockUnconstrainedOptimizationBenchmark,
    RosenbrockConstrainedOptimizationBenchmark,
    EvtushenkoConstrainedOptimizationBenchmark,
)
from pyapprox.interface.model import ModelFromSingleSampleCallable
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

    @unittest.skipIf(not has_pyrol, "pyrol cannot be imported")
    def test_rol_minimize_constrained_evtushenko(self):
        bkd = self.get_backend()
        benchmark = EvtushenkoConstrainedOptimizationBenchmark(backend=bkd)
        optimizer = ROLConstrainedOptimizer(
            benchmark.objective(),
            constraints=benchmark.constraints(),
            bounds=benchmark.design_variable().bounds(),
        )
        init_iterate = benchmark.init_iterate()
        result = optimizer.minimize(init_iterate)
        assert bkd.allclose(result.x, bkd.array([0.0, 0.0, 1.0])[:, None])

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
        fun = SmoothLogBasedMaxFunction(2, 1e-1, bkd, 1e2, 1e-1)
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
        fun = SmoothLogBasedLeftHeavisideFunction(2, 1e-2, bkd, 1e2, 1e-2)
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

    def test_differential_evolution(self):
        bkd = self.get_backend()
        benchmark = EvtushenkoConstrainedOptimizationBenchmark(backend=bkd)
        bounds = benchmark.design_variable().bounds()
        # bounds must be finite for differential_evolution
        bounds[:, 1] = 1e3
        optimizer = ScipyConstrainedDifferentialEvolutionOptimizer(
            benchmark.objective(),
            constraints=benchmark.constraints(),
            bounds=bounds,
            opts={"tol": 1e-7},
        )
        init_iterate = benchmark.init_iterate()
        result = optimizer.minimize(init_iterate)
        assert bkd.allclose(
            result.x, bkd.array([0.0, 0.0, 1.0])[:, None], atol=1e-5
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
            True,
            bkd,
        )
        assert bkd.allclose(
            chain_rule._uncompress_jacobian(self._dx_dp_compressed),
            self._dx_dp_uncompressed,
        )
        assert bkd.allclose(
            chain_rule._compress_jacobian(self._dx_dp_uncompressed, "dx_dp"),
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
            True,
            bkd,
        )
        du_dp = chain_rule(self._p)
        self.assertEqual(du_dp.shape, (self._N, self._n_o, self._n_p))
        assert bkd.allclose(du_dp, self._du_dp)

    def test_chain_rule_arrays_no_compression(self):
        bkd = self.get_backend()
        chain_rule = ChainRuleArrays(False, False, bkd)
        chain_rule.set_arrays(
            self._x.shape,
            self._u.shape,
            self._dx_dp_compressed,
            self._du_dx_compressed,
        )
        du_dp = chain_rule(self._p.shape)
        self.assertEqual(du_dp.shape, (self._N, self._n_o, self._n_p))
        assert bkd.allclose(du_dp, self._du_dp)

    def test_chain_rule_arrays_with_compression(self):
        bkd = self.get_backend()
        chain_rule = ChainRuleArrays(True, True, bkd)
        chain_rule.set_arrays(
            self._x.shape,
            self._u.shape,
            self._dx_dp_uncompressed,
            self._du_dx_uncompressed,
        )
        du_dp = chain_rule(self._p.shape)
        self.assertEqual(du_dp.shape, (self._N, self._n_o, self._n_p))
        assert bkd.allclose(du_dp, self._du_dp)

    def test_chain_rule_arrays_with_mixed_compression(self):
        bkd = self.get_backend()
        chain_rule = ChainRuleArrays(False, True, bkd)
        chain_rule.set_arrays(
            self._x.shape,
            self._u.shape,
            self._dx_dp_compressed,
            self._du_dx_uncompressed,
        )
        du_dp = chain_rule(self._p.shape)
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
        self._x_jac_uncompressed = lambda p: bkd.jacobian(self._x_function, p)
        self._u_jac_uncompressed = lambda x: bkd.jacobian(self._u_function, x)

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
