import unittest

import numpy as np

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin

# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.pde.collocation.newton import (
    NewtonResidual,
    NewtonSolver,
    AdjointFunctional,
    AdjointSolver,
)
from pyapprox.pde.collocation.adjoint_models import (
    SteadyAdjointModelFixedInitialIterate,
)


class NonLinearCoupledResidual(NewtonResidual):
    def __call__(self, iterate: Array) -> Array:
        return self._bkd.array(
            [
                self._a**2 * iterate[0] ** 2 + iterate[1] ** 2 - 1,
                iterate[0] ** 2 - self._b**3 * iterate[1] ** 2 - 1,
            ]
        )

    def jacobian(self, iterate: Array) -> Array:
        return self._bkd.array(
            [
                [2 * self._a**2 * iterate[0], 2 * iterate[1]],
                [2 * iterate[0], -2 * self._b**3 * iterate[1]],
            ]
        )

    def param_jacobian(self, iterate: Array) -> Array:
        return self._bkd.array(
            [
                [2 * self._a * iterate[0] ** 2, 0.0],
                [0.0, -3 * self._b**2 * iterate[1] ** 2],
            ]
        )

    def set_param(self, param: Array):
        self._param = param
        self._a, self._b = self._param

    def __repr__(self):
        return "{0}(a={1}, b={2})".format(
            self.__class__.__name__, self._a, self._b
        )

    def param_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        return self._bkd.array(
            [
                [2 * adj_sol[0] * fwd_sol[0] ** 2, 0],
                [0, -6 * adj_sol[1] * self._b * fwd_sol[1] ** 2],
            ]
            @ vvec
        )

    def state_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        return self._bkd.array(
            [
                [2 * adj_sol[0] * self._a**2 + 2 * adj_sol[1], 0],
                [0, 2 * adj_sol[0] - 2 * adj_sol[1] * self._b**3],
            ]
            @ wvec
        )

    def state_param_hvp(
        self, fwd_sol: Array, adj_sol: Array, vvec: Array
    ) -> Array:
        return (
            self._bkd.array(
                [
                    [4 * adj_sol[0] * self._a * fwd_sol[0], 0],
                    [0, -6 * adj_sol[1] * self._b**2 * fwd_sol[1]],
                ]
            )
            @ vvec
        )

    def param_state_hvp(
        self, fwd_sol: Array, adj_sol: Array, wvec: Array
    ) -> Array:
        return (
            self._bkd.array(
                [
                    [4 * adj_sol[0] * self._a * fwd_sol[0], 0],
                    [0, -6 * adj_sol[1] * self._b**2 * fwd_sol[1]],
                ]
            )
            @ wvec
        )


class SumFunctional(AdjointFunctional):
    def nstates(self):
        return 2

    def nparams(self):
        return 2

    def _value(self, sol: Array) -> Array:
        return self._bkd.atleast1d(self._bkd.sum(sol**3))

    def _qoi_sol_jacobian(self, sol: Array) -> Array:
        dqdu = 3 * sol**2
        return dqdu

    def _qoi_param_jacobian(self, sol: Array) -> Array:
        return self._bkd.zeros((self.nparams(),))

    def _qoi_param_param_hvp(self, sol: Array, vvec: Array) -> Array:
        return self._bkd.zeros((sol.shape[0],))

    def _qoi_state_state_hvp(self, sol: Array, wvec: Array) -> Array:
        return self._bkd.array([[6 * sol[0], 0], [0, 6 * sol[1]]]) @ wvec

    def _qoi_state_param_hvp(self, sol: Array, vvec: Array) -> Array:
        return self._bkd.zeros((2, 2)) @ vvec

    def _qoi_param_state_hvp(self, sol: Array, wvec: Array) -> Array:
        return self._bkd.zeros((2, 2)) @ wvec


class TestNewton:
    def setUp(self):
        np.random.seed(1)

    def test_nonlinear_coupled_residual(self):
        bkd = self.get_backend()
        res = NonLinearCoupledResidual(bkd)
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

        functional = SumFunctional()
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
            res, functional, init_iterate, apply_hessian_implemented=True
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

        model(sample)
        model.jacobian(sample)
        model.apply_hessian(sample, vec[:, None])


class TestNumpyNewton(TestNewton, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# class TestTorchNewton(TestNewton, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
