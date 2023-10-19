import torch
import unittest
import numpy as np
from functools import partial

from pyapprox.pde.autopde.time_integration import (
    explicit_runge_kutta_update, implicit_runge_kutta_residual,
    implicit_runge_kutta_stage_solution_trad,
    ImplicitRungeKutta, create_butcher_tableau
)


class TestTimeIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        np.random.seed(1)

    def test_implicit_runge_kutta_residual(self):
        degree, ndof = 2, 2

        def exact_sol(time):
            return np.array([(ii+1+time)**degree for ii in range(ndof)])

        def rhs(sol, time):
            # dy/dt = y0*p*(t+1)**(p-1) = y0*(t+1)**p*p/(t+1)
            return torch.cat(
                [sol[ii:ii+1]*degree/(ii+1+time) for ii in range(ndof)])

        tableau_name = "ex_rk4"
        # tableau_name = "ex_feuler1"
        deltat = 0.1
        time = 0
        sol = torch.tensor(exact_sol(0), dtype=torch.double)
        ex_butcher_tableau = create_butcher_tableau(
            tableau_name, return_tensors=False)
        im_butcher_tableau = create_butcher_tableau(
            tableau_name, return_tensors=True)

        tmp1, tmp2, tmp3 = explicit_runge_kutta_update(
            sol, deltat, time, rhs, ex_butcher_tableau)
        tmp2 = torch.cat([torch.tensor(t) for t in tmp2])
        tmp3 = torch.cat([torch.tensor(t) for t in tmp3])
        tmp4, tmp5 = implicit_runge_kutta_stage_solution_trad(
            sol, deltat, time, rhs, im_butcher_tableau, tmp2)
        res = implicit_runge_kutta_residual(
            implicit_runge_kutta_stage_solution_trad, None,
            sol, deltat, time, rhs, im_butcher_tableau, tmp2, None)[0]
        assert np.allclose(res, 0)
        assert np.allclose(tmp2, tmp4)
        assert np.allclose(tmp3, tmp5)

        # Note the stage unknowns of the Wildey formulation are different
        # to the traditional form and so will not produce a zero residual
        # using the stage solutions from the explicit update

    def check_implicit_runge_kutta_update(
            self, degree, tableau_name, deltat):

        print(degree, tableau_name, deltat)
        ndof = 2

        def exact_sol(time):
            return np.array([(ii+1+time)**degree for ii in range(ndof)])

        def rhs(sol, time):
            # dy/dt = y0*p*(t+1)**(p-1) = y0*(t+1)**p*p/(t+1)
            vals = torch.cat(
                [sol[ii:ii+1]*degree/(ii+1+time) for ii in range(ndof)])
            return vals, None

        init_time = 0
        final_time = init_time + deltat
        init_sol = torch.tensor(exact_sol(0), dtype=torch.double)
        time_integrator = ImplicitRungeKutta(deltat, rhs, tableau_name)
        sols = time_integrator.integrate(
            init_sol, init_time, final_time,
            newton_kwargs={"verbosity": 2})[0]
        exact_sols = exact_sol(np.arange(sols.shape[1])*deltat)
        # print(sols)
        # print(exact_sols)
        # print(sols-exact_sols)
        assert np.allclose(exact_sols, sols, atol=1e-14)

    def test_implicit_runge_kutta_update(self):
        # Do not test im_beuler with degree=1 an deltat=1.
        # This is a pathalogical
        # case that causes jacobian to be zero
        # Warning gauss-p rules do not recover degree-p solution exactly.
        # They do recover degree-(n) solutions exactly (
        # where n is the number of stages). I think this makes sense as
        # the number of stages are like interpolation points and so
        # we can interpolate exactly. But order is just integration order
        # These rules will be highly accurate but not interpolants
        test_cases = [
            [1, "im_beuler1", 0.5], [2, "im_crank2", 1],
            [2, "im_gauss4", 1], [3, "im_gauss6", 1]]
        for test_case in test_cases:
            self.check_implicit_runge_kutta_update(*test_case)


if __name__ == "__main__":
    time_integration_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestTimeIntegration)
    unittest.TextTestRunner(verbosity=2).run(time_integration_test_suite)
