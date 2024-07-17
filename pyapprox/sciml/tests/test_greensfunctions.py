import unittest
from functools import partial

import numpy as np

from pyapprox.sciml.greensfunctions import (
    GreensFunctionSolver, DrivenHarmonicOscillatorGreensKernel,
    Helmholtz1DGreensKernel, HeatEquation1DGreensKernel,
    WaveEquation1DGreensKernel, ActiveGreensKernel,
    HomogeneousLaplace1DGreensKernel)
from pyapprox.sciml.quadrature import (
    Fixed1DTrapezoidIOQuadRule, TensorProduct2DQuadRule,
    Transformed1DQuadRule, OnePointRule1D)
from pyapprox.sciml.util._torch_wrappers import (to_numpy)

from pyapprox.util.visualization import get_meshgrid_samples


class TestGreensFunction(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_driven_harmonic_oscillator(self):
        nquad = 10000
        omega = 3
        final_time = 3
        kernel = DrivenHarmonicOscillatorGreensKernel(omega, [1e-8, 10])
        quad_rule = Transformed1DQuadRule(
            Fixed1DTrapezoidIOQuadRule(nquad), [0, final_time])
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(tt):
            f0 = 1
            return f0/omega**2*(omega*tt-np.sin(omega*tt)).T

        def forcing_function(omega, tt):
            f0 = 1
            return f0*omega*tt.T

        plot_tt = np.linspace(0, final_time, 101)[None, :]
        green_sol = to_numpy(solver(partial(forcing_function, omega), plot_tt))
        # print(exact_solution(plot_tt)-green_sol)
        assert np.allclose(exact_solution(plot_tt), green_sol)

    def test_laplace_1d(self):
        nquad = 10000
        kappa = 0.1
        kernel = HomogeneousLaplace1DGreensKernel(kappa, [1e-3, 1])
        quad_rule = Transformed1DQuadRule(
            Fixed1DTrapezoidIOQuadRule(nquad), [0, 1])
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(xx):
            return (16*xx**4*(1 - xx)**4).T

        def forcing_function(xx):
            return (-192*xx**4*(1 - xx)**2 + 512*xx**3*(1 - xx)**3 -
                    192*xx**2*(1 - xx)**4).T*kappa

        plot_xx = np.linspace(0, 1, 101)[None, :]
        green_sol = to_numpy(solver(forcing_function, plot_xx))
        assert np.allclose(exact_solution(plot_xx), green_sol)

    def test_helmholtz_1d(self):
        nquad = 10000
        # x_freq must be a integer multiple of np.pi otherwise BC will
        # be violated in exact_solution
        x_freq = 2*np.pi
        wavenum = 10
        kernel = Helmholtz1DGreensKernel(wavenum, [1e-3, 100])
        quad_rule = Transformed1DQuadRule(
            Fixed1DTrapezoidIOQuadRule(nquad), [0, 1])
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(xx):
            return np.sin(x_freq*xx.T)

        def forcing_function(xx):
            return (wavenum**2-x_freq**2)*np.sin(x_freq*xx.T)

        plot_xx = np.linspace(0, 1, 101)[None, :]
        green_sol = to_numpy(solver(forcing_function, plot_xx))
        assert np.allclose(exact_solution(plot_xx), green_sol)

        # test that multiple solutions can be computed at once
        forcing_vals = np.hstack(
            [forcing_function(solver._quad_rule.get_samples_weights()[0]),
             2*forcing_function(solver._quad_rule.get_samples_weights()[0])])
        assert np.allclose(
            solver._eval(forcing_vals, plot_xx),
            np.hstack([to_numpy(solver._eval(fvals[:, None], plot_xx))
                      for fvals in forcing_vals.T]))
        assert np.allclose(
            solver._eval(forcing_vals[:, 1:2], plot_xx),
            2*solver._eval(forcing_vals[:, :1], plot_xx))

        # import matplotlib.pyplot as plt
        # ax = plt.figure().gca()
        # ax.plot(plot_xx[0], exact_solution(plot_xx), label=r"$u(x)$")
        # ax.plot(plot_xx[0], green_sol, '--', label=r"$u_G(x)$")
        # ax.legend()

        # # Now plot the greens function
        # ax = plt.figure().gca()
        # X, Y = np.meshgrid(plot_xx[0], plot_xx[0])
        # G = kernel(plot_xx, plot_xx)
        # ax.imshow(G, origin="lower", extent=[0, 1, 0, 1], cmap="jet")
        # plt.show()

    def test_heat_equation_1d_no_forcing(self):
        kappa, L, final_time = 10.0, 10, 0.1
        kernel = HeatEquation1DGreensKernel(
            kappa, [1e-3, 100], L=L, nterms=100)
        nquad = 10000
        quad_rule1 = Transformed1DQuadRule(
            Fixed1DTrapezoidIOQuadRule(nquad), [0, L])

        quad_rule2 = OnePointRule1D(0, 1)
        quad_rule = TensorProduct2DQuadRule(quad_rule1, quad_rule2)
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(xx):
            x = xx[0]
            t = xx[1]
            # return (
            #     6*np.sin(np.pi*x/L)*np.exp(-kappa*(np.pi/L)**2*t))[:, None]
            return (
                12*np.sin(9*np.pi*x/L)*np.exp(-kappa*(9*np.pi/L)**2*t) -
                7*np.sin(4*np.pi*x/L)*np.exp(-kappa*(4*np.pi/L)**2*t))[:, None]

        def initial_condition_function(xx):
            x = xx[0]
            # return 6*np.sin(np.pi*x/L)[:, None]
            return (12*np.sin(9*np.pi*x/L)-7*np.sin(4*np.pi*x/L))[:, None]

        assert np.allclose(
            exact_solution(quad_rule.get_samples_weights()[0]),
            initial_condition_function(quad_rule.get_samples_weights()[0]))

        from pyapprox.util.visualization import get_meshgrid_samples
        X, Y, plot_xx = get_meshgrid_samples([0, L, 0, final_time], 51)
        green_sol = solver(initial_condition_function, plot_xx).numpy()
        assert np.allclose(exact_solution(plot_xx), green_sol)

        kernel = ActiveGreensKernel(
            HeatEquation1DGreensKernel(
                kappa, [1e-3, 100], L, nterms=100), [final_time], [0.])
        solver = GreensFunctionSolver(kernel, quad_rule1)
        plot_xx = np.vstack((
            np.linspace(0, 1, 101)[None, :], np.full((101,), final_time)))
        green_sol = solver(initial_condition_function, plot_xx[:1]).numpy()
        assert np.allclose(exact_solution(plot_xx), green_sol)

    def test_heat_equation_1d_with_forcing(self):
        kappa, L, final_time = 10.0, 10, np.pi*2
        kernel = HeatEquation1DGreensKernel(
            kappa, [1e-3, 100], L=L, nterms=10)
        nquad = 200
        quad_rule1 = Transformed1DQuadRule(
            Fixed1DTrapezoidIOQuadRule(nquad), [0, L])
        quad_rule2 = Transformed1DQuadRule(
            Fixed1DTrapezoidIOQuadRule(nquad), [0, final_time])
        quad_rule = TensorProduct2DQuadRule(quad_rule1, quad_rule2)
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(xx):
            x = xx[0]
            t = xx[1]
            return (np.sin(np.pi*x/L)*np.sin(t))[:, None]

        def forcing_function(xx):
            x = xx[0]
            t = xx[1]
            return (np.sin(np.pi*x/L)*np.cos(t) +
                    kappa*(np.pi/L)**2*np.sin(np.pi*x/L)*np.sin(t))[:, None]

        assert np.allclose(
            exact_solution(np.array([[0, L], [0.1, 0.1]])),
            np.zeros(2)[:, None])

        X, Y, plot_xx = get_meshgrid_samples([0, L, 0, final_time], 51)
        green_sol = to_numpy(solver(forcing_function, plot_xx))
        rel_error = (np.linalg.norm(exact_solution(plot_xx)-green_sol) /
                     np.linalg.norm(exact_solution(plot_xx)))
        assert rel_error < 1.3e-2

        # import matplotlib.pyplot as plt
        # axs = plt.subplots(1, 2, figsize=(2*8, 6), sharey=True)[1]
        # im = axs[0].contourf(
        #     # X, Y, (exact_solution(plot_xx)-green_sol).reshape(X.shape),
        #     X, Y, exact_solution(plot_xx).reshape(X.shape),
        #     levels=40)
        # plt.colorbar(im, ax=axs[0])
        # axs[0].set_xlabel("space")
        # axs[0].set_ylabel("time")
        # im = axs[1].contourf(X, Y, green_sol.reshape(X.shape), levels=40)
        # plt.colorbar(im, ax=axs[1])
        # plt.show()

    def test_wave_equation_1d_with_forcing(self):
        L = 1
        omega, k = 2*np.pi/L, 5*np.pi/L
        final_time = 10
        coeff = omega/k
        kernel_pos = WaveEquation1DGreensKernel(
            coeff, [1e-3, 100], L=L, nterms=10, pos=True)
        kernel_vel = WaveEquation1DGreensKernel(
            coeff, [1e-3, 100], L=L, nterms=10, pos=False)
        # as k increase nquad must increase
        nquad = 100
        quad_rule1 = Transformed1DQuadRule(
            Fixed1DTrapezoidIOQuadRule(nquad), [0, L])
        quad_rule2 = OnePointRule1D(0, 1)
        quad_rule = TensorProduct2DQuadRule(quad_rule1, quad_rule2)
        solver_pos = GreensFunctionSolver(kernel_pos, quad_rule)
        solver_vel = GreensFunctionSolver(kernel_vel, quad_rule)

        def exact_solution(xx):
            x = xx[0]
            t = xx[1]
            return (np.cos(omega*t+0.25)*np.sin(k*x))[:, None]

        def initial_pos_function(xx):
            xx = np.vstack([xx, np.zeros(xx.shape)])
            return exact_solution(xx)

        def initial_vel_function(xx):
            x = xx[0]
            t = 0
            return -omega*(np.sin(omega*t+0.25)*np.sin(k*x))[:, None]

        assert np.allclose(
            exact_solution(np.array([[0, L], [0.1, 0.1]])),
            np.zeros(2)[:, None])

        X, Y, plot_xx = get_meshgrid_samples([0, L, 0, final_time], 51)
        green_sol = (solver_pos(initial_pos_function, plot_xx).numpy() +
                     solver_vel(initial_vel_function, plot_xx).numpy())
        assert np.allclose(green_sol, exact_solution(plot_xx))


if __name__ == '__main__':
    greensfunction_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGreensFunction)
    unittest.TextTestRunner(verbosity=2).run(greensfunction_test_suite)
