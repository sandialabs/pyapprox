import unittest
from functools import partial

import numpy as np

from pyapprox.sciml.greensfunctions import (
    GreensFunctionSolver, DrivenHarmonicOscillatorGreensKernel,
    Helmholtz1DGreensKernel, HeatEquation1DGreensKernel)
from pyapprox.sciml.kernels import HomogeneousLaplace1DGreensKernel
from pyapprox.sciml.quadrature import (
    IntegralOperatorQuadratureRule, Fixed1DTrapezoidIOQuadRule,
    Fixed1DGaussLegendreIOQuadRule, TensorProduct2DQuadRule,
    TransformedQuadRule, OnePointRule1D)
from pyapprox.sciml.util._torch_wrappers import asarray


class TransformedUnitIntervalQuadRule(TransformedQuadRule):
    def __init__(self, quad_rule, bounds):
        self._quad_rule = quad_rule
        self._bounds = bounds

    def _transform(self, points, weights):
        length = self._bounds[1]-self._bounds[0]
        return points*length+self._bounds[0], weights*length


class TestGreensFunction(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_driven_harmonic_oscillator(self):
        nquad = 10000
        omega = 3
        final_time = 3
        kernel = DrivenHarmonicOscillatorGreensKernel(omega, [1e-8, 10])
        quad_rule = TransformedUnitIntervalQuadRule(
            Fixed1DTrapezoidIOQuadRule(nquad), [0, final_time])
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(tt):
            f0 = 1
            return f0/omega**2*(omega*tt-np.sin(omega*tt)).T

        def forcing_function(omega, tt):
            f0 = 1
            return f0*omega*tt.T

        plot_tt = np.linspace(0, final_time, 101)[None, :]
        green_sol = solver(partial(forcing_function, omega), plot_tt).numpy()
        # print(exact_solution(plot_tt)-green_sol)
        assert np.allclose(exact_solution(plot_tt), green_sol)

    def test_laplace_1d(self):
        nquad = 10000
        kappa = 0.1
        kernel = HomogeneousLaplace1DGreensKernel(kappa, [1e-3, 1])
        quad_rule = TransformedUnitIntervalQuadRule(
            Fixed1DTrapezoidIOQuadRule(nquad), [0, 1])
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(xx):
            return (16*xx**4*(1 - xx)**4).T

        def forcing_function(xx):
            return (-192*xx**4*(1 - xx)**2 + 512*xx**3*(1 - xx)**3 -
                    192*xx**2*(1 - xx)**4).T*kappa

        plot_xx = np.linspace(0, 1, 101)[None, :]
        green_sol = solver(forcing_function, plot_xx).numpy()
        assert np.allclose(exact_solution(plot_xx), green_sol)

    def test_helmholtz_1d(self):
        nquad = 10000
        # x_freq must be a integer multiple of np.pi otherwise BC will
        # be violated in exact_solution
        x_freq = 2*np.pi
        wavenum = 10
        kernel = Helmholtz1DGreensKernel(wavenum, [1e-3, 100])
        quad_rule = TransformedUnitIntervalQuadRule(
            Fixed1DTrapezoidIOQuadRule(nquad), [0, 1])
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(xx):
            return np.sin(x_freq*xx.T)

        def forcing_function(xx):
            return (wavenum**2-x_freq**2)*np.sin(x_freq*xx.T)

        plot_xx = np.linspace(0, 1, 101)[None, :]
        green_sol = solver(forcing_function, plot_xx).numpy()
        assert np.allclose(exact_solution(plot_xx), green_sol)

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

    def test_heat_quation_1d_no_forcing(self):
        kappa, L, final_time = 10.0, 10, 0.1
        kernel = HeatEquation1DGreensKernel(
            kappa, [1e-3, 100], L=L, nterms=100)
        nquad = 10000
        quad_rule1 = TransformedUnitIntervalQuadRule(
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
        # plot_xx = np.linspace(0, 1, 101)[None, :]
        green_sol = solver(initial_condition_function, plot_xx).numpy()
        assert np.allclose(exact_solution(plot_xx), green_sol)

        # import matplotlib.pyplot as plt
        # axs = plt.subplots(1, 2, figsize=(2*8, 6))[1]
        # axs[0].contourf(
        #     X, Y, exact_solution(plot_xx).reshape(X.shape), levels=40)
        # axs[0].set_xlabel("space")
        # axs[0].set_ylabel("time")
        # axs[1].contourf(X, Y, green_sol.reshape(X.shape), levels=40)

        # # Now plot the greens function for a fixed time
        # ax = plt.figure().gca()
        # time = 0.05
        # nx = 51
        # x = np.linspace(0, L, nx)
        # from pyapprox.util.utilities import cartesian_product
        # plot_xx = cartesian_product([x, np.array([time])]).copy()
        # print(plot_xx.shape)
        # G = kernel(plot_xx, plot_xx)
        # ax.imshow(G, origin="lower", extent=[0, 1, 0, 1], cmap="jet")

        # # Now plot the greens function for a spatial location
        # ax = plt.figure().gca()
        # space = L/2
        # nt = 51
        # t = np.linspace(0, final_time, nt)
        # from pyapprox.util.utilities import cartesian_product
        # plot_tt = cartesian_product([np.array([space]), t]).copy()
        # G = kernel(plot_tt, plot_tt)
        # ax.imshow(G, origin="lower", extent=[0, 1, 0, 1], cmap="jet")
        # # X, Y = np.meshgrid(x, x)
        # # ax.contourf(X, Y, G.reshape(X.shape), cmap="jet", levels=40)
        # plt.show()

    def test_heat_quation_1d_with_forcing(self):
        kappa, L, final_time = 10.0, 10, np.pi*2
        kernel = HeatEquation1DGreensKernel(
            kappa, [1e-3, 100], L=L, nterms=10)
        nquad = 200
        quad_rule1 = TransformedUnitIntervalQuadRule(
            Fixed1DTrapezoidIOQuadRule(nquad), [0, L])
        quad_rule2 = TransformedUnitIntervalQuadRule(
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

        from pyapprox.util.visualization import get_meshgrid_samples
        X, Y, plot_xx = get_meshgrid_samples([0, L, 0, final_time], 51)
        green_sol = solver(forcing_function, plot_xx).numpy()
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


if __name__ == '__main__':
    greensfunction_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestGreensFunction)
    unittest.TextTestRunner(verbosity=2).run(greensfunction_test_suite)
