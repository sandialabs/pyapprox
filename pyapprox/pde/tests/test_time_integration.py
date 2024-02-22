import unittest
import numpy as np
from functools import partial

from pyapprox.pde.time_integration import (
    CustomTimeIntegratorResidual, StateTimeIntegrator)


class LinearODEExample():
    def dfdp_fun(self, fwd_state):
        return np.zeros((1, 2))

    def dhdp_fun(self, params, fwd_state, time):
        aparam, bparam = params
        if time == 0:
            dhda = -2*aparam*np.ones_like(fwd_state)[:, None]
            dhdb = np.zeros_like(fwd_state)[:, None]
        else:
            dhda = np.zeros_like(fwd_state)[:, None]
            dhdb = -2*bparam*fwd_state[:, None]*deltat
        return np.hstack((dhda, dhdb))


class TestTimeIntegration(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _check_decoupled_linear_ode(self, tableau_name, power):
        # solve dy/dt = a*power*(t+c+1)**(power-1)
        # with solution y = a*(t+c+1)**(power)+(init_sol-a*(1+c)**power)
        # can be used to test collocation implicit rules, i.e.
        # backward euler and trapezoid ruels
        def fun(a, power, y, t):
            nstates = y.shape[0]
            return np.array(
                [a*power*(t+1+ii)**(power-1) for ii in range(nstates)])

        def jac(a, power, y, t):
            nstates = y.shape[0]
            return np.zeros((nstates, nstates))

        def exact_sol(init_sol, a, power, t):
            nstates = init_sol.shape[0]
            return np.array([a*(t+1+ii)**(power)+(init_sol[ii]-a*(1+ii)**power)
                             for ii in range(nstates)]).T

        a = 2.0
        residual = CustomTimeIntegratorResidual(
            partial(fun, a, power), partial(jac, a, power))
        init_sol = np.array([1, 2])
        init_time, final_time, deltat = 0, 1, 0.5
        integrator = StateTimeIntegrator(residual, tableau_name)
        sols, times = integrator.solve(init_sol, init_time, final_time, deltat)
        assert np.allclose(sols, exact_sol(init_sol, a, power, times))

        qoi = integrator.get_update().integrate(times, sols)
        print(qoi)

    def test_decoupled_linear_ode(self):
        test_cases = [["imeuler1", 1], ["imtrap2", 2]]
        for test_case in test_cases:
            self._check_decoupled_linear_ode(*test_case)


if __name__ == "__main__":
    timeintegration_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestTimeIntegration)
    unittest.TextTestRunner(verbosity=2).run(timeintegration_test_suite)
