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
    SteadyAdjointModelFixedInitialIterate
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
                [2 * self._a * iterate[0] ** 2, 0.],
                [0., -3 * self._b**2 * iterate[1] ** 2],
            ]
        )

    def set_param(self, param: Array):
        self._param = param
        self._a, self._b = self._param

    def __repr__(self):
        return "{0}(a={1}, b={2})".format(
            self.__class__.__name__, self._a, self._b
        )


class SumFunctional(AdjointFunctional):
    def nstates(self):
        return 2

    def nparams(self):
        return 2

    def _value(self, sol: Array) -> Array:
        return self._bkd.atleast1d(self._bkd.sum(sol**3))

    def _qoi_sol_jacobian(self, sol: Array) -> Array:
        dqdu = 3 * sol ** 2
        return dqdu

    def _qoi_param_jacobian(self, sol: Array) -> Array:
        return self._bkd.zeros((self.nparams(),))





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
        adj_sol = adjoint_solver.solve_adjoint(sol)
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

        model = SteadyAdjointModelFixedInitialIterate(
            res, functional, init_iterate)
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        errors = model.check_apply_jacobian(
            sample, fd_eps=fd_eps, disp=True
        )
        assert errors.min() / errors.max() < 1e-6


class TestNumpyNewton(TestNewton, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# class TestTorchNewton(TestNewton, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
