import unittest

import numpy as np

from pyapprox.util.backends.template import Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.util.newton import (
    NewtonSolver,
    AdjointFunctional,
    AdjointSolver,
)
from pyapprox.pde.collocation.adjoint import (
    SteadyAdjointModelFixedInitialIterate,
)
from pyapprox.pde.collocation.parameterized_pdes import (
    NonLinearCoupledEquationsResidual,
    NonLinearCoupledEquationsResidualAuto,
    NonLinearCoupledEquationsAffineParamResidual,
)


class ScalarSumFunctionalAuto(AdjointFunctional):
    def nstates(self) -> int:
        return 2

    def nparams(self) -> int:
        return 2

    def _value(self, sol: Array) -> Array:
        return self._bkd.hstack((self._bkd.sum(sol**3),))

    def nqoi(self) -> int:
        return 1

    def nunique_functional_params(self) -> int:
        return 0


class ScalarSumFunctional(ScalarSumFunctionalAuto):
    def _qoi_state_jacobian(self, sol: Array) -> Array:
        dqdu = 3 * sol**2
        return dqdu[None, :]

    def _qoi_param_jacobian(self, sol: Array) -> Array:
        # return super()._qoi_param_jacobian(sol)
        return self._bkd.zeros((self.nparams(),))

    def _qoi_param_param_hvp(self, sol: Array, vvec: Array) -> Array:
        # return super()._qoi_param_param_hvp(sol, vvec)
        return self._bkd.zeros((sol.shape[0],))

    def _qoi_state_state_hvp(self, sol: Array, wvec: Array) -> Array:
        # return super()._qoi_state_state_hvp(sol, wvec)
        return self._bkd.array([[6 * sol[0], 0], [0, 6 * sol[1]]]) @ wvec

    def _qoi_state_param_hvp(self, sol: Array, vvec: Array) -> Array:
        # return super()._qoi_state_param_hvp(sol, vvec)
        return self._bkd.zeros((2, 2)) @ vvec

    def _qoi_param_state_hvp(self, sol: Array, wvec: Array) -> Array:
        # return super()._qoi_param_state_hvp(sol, wvec)
        return self._bkd.zeros((2, 2)) @ wvec


class VectorSumFunctionalAuto(AdjointFunctional):
    def nstates(self) -> int:
        return 2

    def nparams(self) -> int:
        return 2

    def _value(self, sol: Array) -> Array:
        return self._bkd.hstack((self._bkd.sum(sol**3), self._bkd.sum(sol**2)))

    def nqoi(self) -> int:
        return 2

    def nunique_functional_params(self) -> int:
        return 0


class TestNewton:
    def setUp(self):
        np.random.seed(1)

    def _check_nonlinear_coupled_residual(self, res, functional):
        bkd = self.get_backend()
        sample = bkd.array([0.8, 1.1])[:, None]
        res.set_param(sample[:, 0])
        solver = NewtonSolver()
        solver.set_residual(res)
        init_iterate = bkd.array([-1, -1])
        sol = solver.solve(init_iterate)

        a, b = sample[:, 0]
        exact_sol = bkd.array(
            [
                -bkd.sqrt((b + 1) * (b**2 - b + 1) / (a**2 * b**3 + 1)),
                -bkd.sqrt(-(a - 1) * (a + 1) / (a**2 * b**3 + 1)),
            ]
        )
        assert bkd.allclose(sol, exact_sol)

        adjoint_solver = AdjointSolver(solver, functional)
        adjoint_solver.set_param(sample[:, 0])
        adjoint_solver.set_initial_iterate(init_iterate)
        adj_sol = adjoint_solver.solve_adjoint()
        exact_adj_sol = bkd.array(
            [
                3
                * (-(b**3) * exact_sol[0] - exact_sol[1])
                / (2 * (a**2 * b**3 + 1)),
                3
                * (a**2 * exact_sol[1] - exact_sol[0])
                / (2 * (a**2 * b**3 + 1)),
            ]
        )
        assert bkd.allclose(adj_sol, exact_adj_sol)

        vec = bkd.array([1, 2])
        hess_fwd_sol = adjoint_solver.forward_hessian_solve(vec)
        exact_hess_fwd_sol = bkd.array(
            [
                b**2
                * (
                    a * b * vec[0] * exact_sol[0] ** 2
                    - (3 * vec[1] * exact_sol[1] ** 2) / 2
                )
                / (exact_sol[0] * (a**2 * b**3 + 1)),
                a
                * (
                    3 * a * b**2 * vec[1] * exact_sol[1] ** 2
                    + 2 * vec[0] * exact_sol[0] ** 2
                )
                / (2 * exact_sol[1] * (a**2 * b**3 + 1)),
            ]
        )
        assert bkd.allclose(hess_fwd_sol, exact_hess_fwd_sol)

        adj_hess_sol = adjoint_solver.adjoint_hessian_solve(
            exact_hess_fwd_sol, vec
        )
        # exact answer computed using mathematica
        exact_adj_hess_sol = bkd.array([-0.911372311958973, -2.38166422148656])
        assert bkd.allclose(adj_hess_sol, exact_adj_hess_sol)

        model = SteadyAdjointModelFixedInitialIterate(
            res,
            init_iterate,
            2,
            functional,
            jacobian_implemented=True,
            apply_hessian_implemented=True,
        )
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        errors = model.check_apply_jacobian(sample, fd_eps=fd_eps, disp=True)
        assert errors.min() / errors.max() < 1e-6

        # exact answer computed using mathematica
        hvp_exact = bkd.array([0.390549158575547, 3.82494342170840])
        assert bkd.allclose(
            model.apply_hessian(sample, vec[:, None]), hvp_exact
        )
        errors = model.check_apply_hessian(sample, fd_eps=fd_eps, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_nonlinear_coupled_residual(self):
        bkd = self.get_backend()
        res = NonLinearCoupledEquationsResidual(bkd)
        functional = ScalarSumFunctional(backend=bkd)
        self._check_nonlinear_coupled_residual(res, functional)

        if (
            not bkd.hvp_implemented()
            or not bkd.jvp_implemented()
            or not bkd.jacobian_implemented()
        ):
            return
        res = NonLinearCoupledEquationsResidualAuto(bkd)
        functional = ScalarSumFunctionalAuto(backend=bkd)
        self._check_nonlinear_coupled_residual(res, functional)

    def test_nonlinear_coupled_residual_affine_params(self):
        bkd = self.get_backend()
        init_iterate = bkd.array([-1, -1])
        sample = bkd.array([0.8, 1.1])[:, None]
        res = NonLinearCoupledEquationsAffineParamResidual(bkd)
        functional = ScalarSumFunctional(backend=bkd)
        model = SteadyAdjointModelFixedInitialIterate(
            res,
            init_iterate,
            2,
            functional,
            jacobian_implemented=True,
            apply_hessian_implemented=True,
        )
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        errors = model.check_apply_jacobian(sample, fd_eps=fd_eps, disp=True)
        assert errors.min() / errors.max() < 1e-6
        errors = model.check_apply_hessian(sample, fd_eps=fd_eps, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_forward_parameter_jacobian(self):
        bkd = self.get_backend()
        res = NonLinearCoupledEquationsResidual(bkd)
        init_iterate = bkd.array([-1, -1])
        functional = ScalarSumFunctional(backend=bkd)
        model = SteadyAdjointModelFixedInitialIterate(
            res,
            init_iterate,
            2,
            functional,
            jacobian_implemented=True,
            apply_hessian_implemented=True,
            jacobian_mode="forward",
        )
        sample = bkd.array([0.8, 1.1])[:, None]
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        errors = model.check_apply_jacobian(sample, fd_eps=fd_eps, disp=True)
        assert errors.min() / errors.max() < 1e-6

        if not bkd.jacobian_implemented():
            return
        functional = VectorSumFunctionalAuto(backend=bkd)
        model.set_functional(functional)
        errors = model.check_apply_jacobian(sample, fd_eps=fd_eps, disp=True)
        assert errors.min() / errors.max() < 1e-6


class TestNumpyNewton(TestNewton, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchNewton(TestNewton, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main()
