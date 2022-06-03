import torch
import unittest
import numpy as np
from functools import partial

from pyapprox.pde.autopde.time_integration import (
    explicit_runge_kutta_update, implicit_runge_kutta_residual,
    implicit_runge_kutta_stage_solution,
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

        tableau_name = "ex_mid2"
        # tableau_name = "ex_feuler1"
        deltat = 0.1
        time = 0
        sol = torch.tensor(exact_sol(0), dtype=torch.double)
        sols = [sol.detach().numpy()]
        ex_butcher_tableau = create_butcher_tableau(tableau_name, return_tensors=False)
        im_butcher_tableau = create_butcher_tableau(tableau_name, return_tensors=True)

        tmp1, tmp2, tmp3 = explicit_runge_kutta_update(
            sol, deltat, time, rhs, ex_butcher_tableau)
        tmp2 = torch.cat([torch.tensor(t) for t in tmp2])
        tmp3 = torch.cat([torch.tensor(t) for t in tmp3])
        tmp4, tmp5 = implicit_runge_kutta_stage_solution(
            sol, deltat, time, rhs, im_butcher_tableau, tmp2, None)
        res = implicit_runge_kutta_residual(
            sol, deltat, time, rhs, im_butcher_tableau, tmp2, None)
        assert np.allclose(res, 0)
        assert np.allclose(tmp2, tmp4)
        assert np.allclose(tmp3, tmp5)
        
        # implicit_runge_kutta_update(
        #     sol, deltat, time, rhs, butcher_tableau,
        #     [sol.clone()]*butcher_tableau[0].shape[0], None)
        

    def test_implicit_runge_kutta_update(self):
        degree, ndof = 2, 1

        def exact_sol(time):
            return np.array([(ii+1+time)**degree for ii in range(ndof)])

        def rhs(sol, time):
            # dy/dt = y0*p*(t+1)**(p-1) = y0*(t+1)**p*p/(t+1)
            print(sol, time, 'rhs')
            return torch.cat(
                [sol[ii:ii+1]*degree/(ii+1+time) for ii in range(ndof)])

        # tableau_name = "ex_heun2"
        # tableau_name = "ex_rk4"
        tableau_name = "im_crank2"
        # tableau_name = "im_beuler1" # test with degree=1 will fail because of pathalogical choices of degree and deltat if degree=2 and deltat=1
        deltat = 1
        init_time = 0
        final_time = init_time + deltat
        print(exact_sol(0).shape)
        init_sol = torch.tensor(exact_sol(0), dtype=torch.double)
        print(init_sol.shape)
        time_integrator = ImplicitRungeKutta(deltat, rhs, tableau_name)
        sols = time_integrator.integrate(init_sol, init_time, final_time)

        exact_sols = exact_sol(np.arange(sols.shape[1])*deltat)
        print(sols)
        print(exact_sols)
        # print(sols-exact_sols)
        assert np.allclose(exact_sols, sols)


    

if __name__ == "__main__":
    time_integration_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestTimeIntegration)
    unittest.TextTestRunner(verbosity=2).run(time_integration_test_suite)
