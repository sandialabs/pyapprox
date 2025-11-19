import unittest

import numpy as np

from pyapprox.util.backends.template import Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.util.newnewton import (
    NewtonSolver,
    NewtonResidual,
    BisectionSearch,
    BoundedNewtonResidual,
)
from pyapprox.pde.adjoint import (
    SteadyAdjointModelFixedInitialIterate,
)
from pyapprox.pde.collocation.parameterized_pdes import (
    NonLinearCoupledEquationsResidual,
    NonLinearCoupledEquationsResidualAuto,
    NonLinearCoupledEquationsAffineParamResidual,
)


class TestNewton:
    def setUp(self):
        np.random.seed(1)

    def _check_nonlinear_coupled_residual(self, res):
        bkd = self.get_backend()
        res = NonLinearCoupledEquationsResidual(bkd)
        sample = bkd.array([0.8, 1.1])[:, None]
        res.set_param(sample[:, 0])
        solver = NewtonSolver()
        solver.set_residual(res)
        init_iterate = bkd.array([-1, -1.0])
        sol = solver.solve(init_iterate)

        a, b = sample[:, 0]
        exact_sol = bkd.array(
            [
                -bkd.sqrt((b + 1) * (b**2 - b + 1) / (a**2 * b**3 + 1)),
                -bkd.sqrt(-(a - 1) * (a + 1) / (a**2 * b**3 + 1)),
            ]
        )
        assert bkd.allclose(sol, exact_sol)

    def test_bisection_search(self):
        bkd = self.get_backend()

        class Residual(NewtonResidual):
            def __call__(self, iterate: Array) -> Array:
                self._rhs = self._bkd.array([0.1, 0.3, 0.6])
                return iterate**2 - self._rhs

        bisearch = BisectionSearch()
        residual = Residual(backend=bkd)
        bisearch.set_residual(residual)
        bounds = bkd.array([[0.0, 0.5], [0.1, 1], [0.5, 1.0]])
        roots = bisearch.solve(bounds)
        assert bkd.allclose(roots, bkd.sqrt(residual._rhs))

    def test_univariate_bounded_newton(self):
        bkd = self.get_backend()

        class Residual(NewtonResidual):
            def __init__(self, backend):
                super().__init__(backend)
                self._rhs = self._bkd.array([1e-3, 0.3, 0.6])

            def __call__(self, iterate: Array) -> Array:
                return iterate**2 - self._rhs

            def _jacobian(self, iterate: Array) -> Array:
                return self._bkd.diag(2 * iterate)

        residual = Residual(backend=bkd)
        bounds = (0, 1)
        bounded_residual = BoundedNewtonResidual(residual, bounds)
        iterate = bkd.array([0.01, 0.5, 0.8])
        can_iterate = bounded_residual._to_canonical(iterate)

        # Using sigmoid transformation to enforce bounds changes topology
        # of residual. The code below plots the new residual which
        # was original quadratic
        # import matplotlib.pyplot as plt
        # can_xx = bkd.linspace(-10, 10, 101)
        # vals = bkd.empty(can_xx.shape[0])
        # for ii in range(can_xx.shape[0]):
        #     cit = bkd.copy(can_iterate)
        #     cit[0] = can_xx[ii]
        #     vals[ii] = bounded_residual(cit)[0]
        # plt.plot(can_xx, vals)
        # plt.show()
        assert bkd.allclose(
            bounded_residual._from_canonical(can_iterate), iterate
        )
        if bkd.jacobian_implemented():
            assert bkd.allclose(
                super(BoundedNewtonResidual, bounded_residual)._jacobian(
                    can_iterate
                ),
                bounded_residual.jacobian(can_iterate),
            )
        newton = NewtonSolver(verbosity=0, rtol=1e-10, atol=1e-10)
        newton.set_residual(bounded_residual)
        # iterate = bkd.sqrt(residual._rhs) + 0.1
        # can_iterate = bounded_residual._to_canonical(iterate)
        can_roots = newton.solve(can_iterate)
        roots = bounded_residual._from_canonical(can_roots)
        assert bkd.allclose(roots, bkd.sqrt(residual._rhs))


class TestNumpyNewton(TestNewton, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchNewton(TestNewton, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
