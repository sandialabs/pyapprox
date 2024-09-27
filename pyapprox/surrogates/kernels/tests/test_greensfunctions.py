import unittest
from functools import partial
import math
import numpy as np

from pyapprox.util.sys_utilities import package_available
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.univariate import (
    UnivariatePiecewisePolynomialQuadratureRule
)
from pyapprox.surrogates.kernels.greensfunctions import (
    GreensFunctionSolver,
    DrivenHarmonicOscillatorGreensKernel,
    Helmholtz1DGreensKernel,
    HeatEquation1DGreensKernel,
    WaveEquation1DGreensKernel,
    ActiveGreensKernel,
    HomogeneousLaplace1DGreensKernel,
)
from pyapprox.surrogates.bases.basis import FixedTensorProductQuadratureRule
from pyapprox.util.visualization import get_meshgrid_samples
from pyapprox.util.print_wrapper import *


class TestGreensFunctions:
    def setUp(self):
        np.random.seed(1)

    def test_driven_harmonic_oscillator(self):
        bkd = self.get_backend()
        nquad = 1001
        omega = 3
        final_time = 3
        kernel = DrivenHarmonicOscillatorGreensKernel(
            omega, [1e-8, 10], backend=bkd
        )
        quad_rule = UnivariatePiecewisePolynomialQuadratureRule(
            "quadratic", [0, final_time], backend=bkd
        )
        quad_rule.set_nnodes(nquad)
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(tt):
            f0 = 1
            return f0 / omega**2 * (omega * tt - bkd.sin(omega * tt)).T

        def forcing_function(omega, tt):
            f0 = 1
            return f0 * omega * tt.T

        plot_tt = bkd.linspace(0, final_time, 101)[None, :]
        green_sol = solver(partial(forcing_function, omega), plot_tt)
        assert bkd.allclose(exact_solution(plot_tt), green_sol)

    def test_laplace_1d(self):
        bkd = self.get_backend()
        nquad = 1001
        kappa = 0.1
        kernel = HomogeneousLaplace1DGreensKernel(
            kappa, [1e-3, 1], backend=bkd
        )
        quad_rule = UnivariatePiecewisePolynomialQuadratureRule(
            "quadratic", [0, 1], backend=bkd
        )
        quad_rule.set_nnodes(nquad)
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(xx):
            return (16 * xx**4 * (1 - xx) ** 4).T

        def forcing_function(xx):
            return (
                -192 * xx**4 * (1 - xx) ** 2
                + 512 * xx**3 * (1 - xx) ** 3
                - 192 * xx**2 * (1 - xx) ** 4
            ).T * kappa

        plot_xx = bkd.linspace(0, 1, 101)[None, :]
        green_sol = solver(forcing_function, plot_xx)
        assert bkd.allclose(exact_solution(plot_xx), green_sol)

    def test_helmholtz_1d(self):
        bkd = self.get_backend()
        nquad = 1001
        # x_freq must be a integer multiple of np.pi otherwise BC will
        # be violated in exact_solution
        x_freq = 2 * np.pi
        wavenum = 10
        kernel = Helmholtz1DGreensKernel(wavenum, [1e-3, 100], backend=bkd)
        quad_rule = UnivariatePiecewisePolynomialQuadratureRule(
            "quadratic", [0, 1], backend=bkd
        )
        quad_rule.set_nnodes(nquad)
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(xx):
            return bkd.sin(x_freq * xx.T)

        def forcing_function(xx):
            return (wavenum**2 - x_freq**2) * bkd.sin(x_freq * xx.T)

        plot_xx = bkd.linspace(0, 1, 101)[None, :]
        green_sol = solver(forcing_function, plot_xx)
        assert bkd.allclose(exact_solution(plot_xx), green_sol)

        # test that multiple solutions can be computed at once
        forcing_vals = bkd.hstack(
            [
                forcing_function(solver._quad_rule()[0]),
                2 * forcing_function(solver._quad_rule()[0]),
            ]
        )
        assert bkd.allclose(
            solver._eval(forcing_vals, plot_xx),
            bkd.hstack(
                [
                    solver._eval(fvals[:, None], plot_xx)
                    for fvals in forcing_vals.T
                ]
            ),
        )
        assert bkd.allclose(
            solver._eval(forcing_vals[:, 1:2], plot_xx),
            2 * solver._eval(forcing_vals[:, :1], plot_xx),
        )

        # import matplotlib.pyplot as plt
        # ax = plt.figure().gca()
        # ax.plot(plot_xx[0], exact_solution(plot_xx), label=r"$u(x)$")
        # ax.plot(plot_xx[0], green_sol, '--', label=r"$u_G(x)$")
        # ax.legend()

        # # Now plot the greens function
        # ax = plt.figure().gca()
        # X, Y = bkd.meshgrid(plot_xx[0], plot_xx[0])
        # G = kernel(plot_xx, plot_xx)
        # ax.imshow(G, origin="lower", extent=[0, 1, 0, 1], cmap="jet")
        # plt.show()

    def test_heat_equation_1d_no_forcing(self):
        bkd = self.get_backend()
        kappa, L, final_time = 10.0, 10, 0.1
        kernel = HeatEquation1DGreensKernel(
            kappa, [1e-3, 100], L=L, nterms=100, backend=bkd
        )
        quad_rule1 = UnivariatePiecewisePolynomialQuadratureRule(
            "quadratic", [0, L], backend=bkd
        )
        quad_rule2 = UnivariatePiecewisePolynomialQuadratureRule(
            "quadratic", [0, 1], backend=bkd
        )

        quad_rule = FixedTensorProductQuadratureRule(
            2, [quad_rule1, quad_rule2], [1001, 1]
        )
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(xx):
            x = xx[0]
            t = xx[1]
            # return (
            #     6*bkd.sin(math.pi*x/L)*bkd.exp(-kappa*(math.pi/L)**2*t))[:, None]
            return (
                12
                * bkd.sin(9 * math.pi * x / L)
                * bkd.exp(-kappa * (9 * math.pi / L) ** 2 * t)
                - 7
                * bkd.sin(4 * math.pi * x / L)
                * bkd.exp(-kappa * (4 * math.pi / L) ** 2 * t)
            )[:, None]

        def initial_condition_function(xx):
            x = xx[0]
            # return 6*bkd.sin(math.pi*x/L)[:, None]
            return (
                12 * bkd.sin(9 * math.pi * x / L)
                - 7 * bkd.sin(4 * math.pi * x / L)
            )[:, None]

        assert bkd.allclose(
            exact_solution(quad_rule()[0]),
            initial_condition_function(quad_rule()[0]),
        )

        X, Y, plot_xx = get_meshgrid_samples(
            [0, L, 0, final_time], 51, bkd=bkd
        )
        green_sol = solver(initial_condition_function, plot_xx)
        assert bkd.allclose(exact_solution(plot_xx), green_sol)

        kernel = ActiveGreensKernel(
            HeatEquation1DGreensKernel(
                kappa, [1e-3, 100], L, nterms=100, backend=bkd
            ),
            [final_time],
            [0.0],
        )
        quad_rule1.set_nnodes(1001)
        solver = GreensFunctionSolver(kernel, quad_rule1)
        plot_xx = bkd.vstack(
            (bkd.linspace(0, 1, 101)[None, :], bkd.full((101,), final_time))
        )
        green_sol = solver(initial_condition_function, plot_xx[:1])
        assert bkd.allclose(exact_solution(plot_xx), green_sol)

    def test_heat_equation_1d_with_forcing(self):
        bkd = self.get_backend()
        kappa, L, final_time = 10.0, 10, math.pi * 2
        kernel = HeatEquation1DGreensKernel(
            kappa, [1e-3, 100], L=L, nterms=10, backend=bkd
        )
        quad_rule1 = UnivariatePiecewisePolynomialQuadratureRule(
            "quadratic", [0, L], backend=bkd
        )
        quad_rule2 = UnivariatePiecewisePolynomialQuadratureRule(
            "quadratic", [0, final_time], backend=bkd
        )

        quad_rule = FixedTensorProductQuadratureRule(
            2, [quad_rule1, quad_rule2], [201] * 2
        )
        solver = GreensFunctionSolver(kernel, quad_rule)

        def exact_solution(xx):
            x = xx[0]
            t = xx[1]
            return (bkd.sin(math.pi * x / L) * bkd.sin(t))[:, None]

        def forcing_function(xx):
            x = xx[0]
            t = xx[1]
            return (
                bkd.sin(math.pi * x / L) * bkd.cos(t)
                + kappa
                * (math.pi / L) ** 2
                * bkd.sin(math.pi * x / L)
                * bkd.sin(t)
            )[:, None]

        assert bkd.allclose(
            exact_solution(bkd.array([[0, L], [0.1, 0.1]])),
            bkd.zeros(2)[:, None],
        )

        X, Y, plot_xx = get_meshgrid_samples(
            [0, L, 0, final_time], 51, bkd=bkd
        )
        green_sol = solver(forcing_function, plot_xx)
        rel_error = bkd.norm(exact_solution(plot_xx) - green_sol) / bkd.norm(
            exact_solution(plot_xx)
        )
        assert rel_error < 1.5e-2

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
        bkd = self.get_backend()
        L = 1
        omega, k = 2 * math.pi / L, 5 * math.pi / L
        final_time = 10
        coeff = omega / k
        kernel_pos = WaveEquation1DGreensKernel(
            coeff, [1e-3, 100], L=L, nterms=10, pos=True, backend=bkd
        )
        kernel_vel = WaveEquation1DGreensKernel(
            coeff, [1e-3, 100], L=L, nterms=10, pos=False, backend=bkd
        )
        # as k increase nquad must increase
        quad_rule1 = UnivariatePiecewisePolynomialQuadratureRule(
            "quadratic", [0, L], backend=bkd
        )
        quad_rule2 = UnivariatePiecewisePolynomialQuadratureRule(
            "quadratic", [0, 1], backend=bkd
        )

        quad_rule = FixedTensorProductQuadratureRule(
            2, [quad_rule1, quad_rule2], [1001, 1]
        )
        solver_pos = GreensFunctionSolver(kernel_pos, quad_rule)
        solver_vel = GreensFunctionSolver(kernel_vel, quad_rule)

        def exact_solution(xx):
            x = xx[0]
            t = xx[1]
            return (bkd.cos(omega * t + 0.25) * bkd.sin(k * x))[:, None]

        def initial_pos_function(xx):
            xx = bkd.vstack([xx, bkd.zeros(xx.shape)])
            return exact_solution(xx)

        def initial_vel_function(xx):
            x = xx[0]
            t = 0
            return (
                -omega * (math.sin(omega * t + 0.25) * bkd.sin(k * x))[:, None]
            )

        assert bkd.allclose(
            exact_solution(bkd.array([[0, L], [0.1, 0.1]])),
            bkd.zeros(2)[:, None],
        )

        X, Y, plot_xx = get_meshgrid_samples(
            [0, L, 0, final_time], 51, bkd=bkd
        )
        green_sol = solver_pos(initial_pos_function, plot_xx) + solver_vel(
            initial_vel_function, plot_xx
        )
        assert bkd.allclose(green_sol, exact_solution(plot_xx))


class TestNumpyGreensFunctions(TestGreensFunctions, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchGreensFunctions(TestGreensFunctions, unittest.TestCase):
    def setUp(self):
        if not package_available("torch"):
            self.skipTest("torch not available")
        TestGreensFunctions.setUp(self)

    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
