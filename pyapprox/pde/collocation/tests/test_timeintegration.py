import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.collocation.timeintegration import (
    TransientNewtonResidual,
    ImplicitTimeIntegrator,
    BackwardEulerResidual,
    CrankNicholsonResidual,
)


class TestCollocation:
    def setUp(self):
        np.random.seed(1)

    def test_decoupled_ode(self):
        bkd = self.get_backend()

        class DecoupledODE(TransientNewtonResidual):
            def __init__(self, param: float, backend):
                super().__init__(backend)
                self.set_param(param)

            def set_time(self, time: float):
                self._time = time

            def set_param(self, param):
                self._param = param

            def __call__(self, sol):
                b = self._param**2*self._bkd.arange(1, sol.shape[0]+1)
                return -b*sol

            def jacobian(self, sol):
                b = self._param**2*self._bkd.arange(1, sol.shape[0]+1)
                return -self._bkd.diag(b)

        param, nqoi = 3., 2
        init_time, final_time = 0, 0.25
        deltat = 0.001
        residual = BackwardEulerResidual(DecoupledODE(param, bkd))
        # residual = CrankNicholsonResidual(DecoupledODE(param, bkd))
        time_int = ImplicitTimeIntegrator(
            residual, init_time, final_time, deltat, verbosity=0
        )
        init_sol = bkd.ones((nqoi,))
        sols, times = time_int.solve(init_sol)
        assert bkd.allclose(times, np.arange(init_time, final_time+deltat, deltat))

        exact_sols = init_sol[:, None] * bkd.exp(
            -times[None, :]*(param**2*bkd.arange(1, nqoi+1))[:, None]
        )
        ax = plt.figure().gca()
        print(bkd.abs(exact_sols-sols).max())
        ax.plot(times, exact_sols.T)
        ax.plot(times, sols.T, '--')
        # plt.show()


class TestNumpyCollocation(TestCollocation, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


if __name__ == "__main__":
    unittest.main()
