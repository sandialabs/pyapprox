import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.pde.collocation.parameterized_pdes import (
    TransientDiffusionAdvectionModel,
    SteadyDarcy2DKLEModel,
    FitzHughNagumoModel,
    SteadyShallowShelfModel2D,
)
from pyapprox.pde.collocation.timeintegration import (
    BackwardEulerResidual,
    CrankNicholsonResidual,
    ForwardEulerResidual,
    HeunResidual,
    TransientMSEAdjointFunctional,
)
from pyapprox.util.newton import (
    NewtonSolver,
    AdjointFunctional,
    Array,
)
from pyapprox.pde.collocation.solvers import CollocationModelMixin
from pyapprox.pde.collocation.functions import (
    animate_transient_2d_scalar_solution,
    animate_transient_2d_vector_solution,
)

# from pyapprox.util.print_wrapper import *


class SumFunctional(AdjointFunctional):
    def __init__(self, model):
        self._model = model
        if not isinstance(model, CollocationModelMixin):
            raise ValueError(
                "model must be an instance of CollocationModelMixin"
            )
        super().__init__(model._bkd)

    def nqoi(self) -> int:
        return 1

    def nstates(self) -> int:
        return (
            self._model.basis().mesh().nmesh_pts()
            * self._model.physics().ncomponents()
        )

    def nparams(self) -> int:
        return self._model.nvars()

    def _value(self, sol: Array) -> Array:
        return self._bkd.hstack((sol.sum(),))

    def nunique_functional_params(self) -> int:
        return 0

    def _qoi_state_jacobian(self, sol: Array) -> Array:
        return self._bkd.ones((1, sol.shape[0]))

    def _qoi_param_jacobian(self, sol: Array) -> Array:
        return self._bkd.zeros((1, self.nparams()))


class TestParameterizedModels:
    def setUp(self):
        np.random.seed(1)

    def test_steady_parameterized_diffusion(self):
        bkd = self.get_backend()
        newton_solver = NewtonSolver(verbosity=2, rtol=1e-8, atol=1e-8)
        model = SteadyDarcy2DKLEModel(
            10,
            1.0,
            0.1,
            0.0,
            newton_solver=newton_solver,
            backend=bkd,
        )

        sample = bkd.array(np.random.normal(0, 1, (model.nvars(), 1)))

        # sol = model.forward_solve(sample)
        # axs = plt.subplots(1, 2, figsize=(2*8, 6))[1]
        # velocity = model.velocity_field(sol)
        # sol.plot(axs[0])
        # velocity.plot_vector_field(axs[0])
        # model.physics()._diffusion.plot(axs[1])
        # plt.show()

        if not bkd.jacobian_implemented():
            # TODO implement analytical expressions for the componets needed
            # by adjoint solve
            return

        functional = SumFunctional(model)
        newton_solver._verbosity = 0
        model.set_functional(functional)
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        errors = model.check_apply_jacobian(sample, fd_eps, disp=True)
        print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 1.4e-6

        errors = model.check_apply_hessian(sample, fd_eps, disp=True)
        print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 1.3e-6

    def test_transient_parameterized_diffusion_with_fixed_advection(self):
        bkd = self.get_backend()
        time_residual_cls = BackwardEulerResidual
        newton_solver = NewtonSolver(verbosity=2, rtol=1e-8, atol=1e-8)
        model = TransientDiffusionAdvectionModel(
            0,
            1,
            0.05,
            time_residual_cls,
            newton_solver=newton_solver,
            backend=bkd,
        )

        # ax = plt.figure().gca()
        # model._vel_field.plot_vector_field(ax)
        # plt.show()

        sample = bkd.array(np.random.normal(0, 1, (model.nvars(), 1)))
        sol, times = model.forward_solve(sample)

        ani = animate_transient_2d_scalar_solution(
            model._basis, sol, times, plot_surface=False
        )
        ani.save("diffusion.gif", dpi=100)

    def test_fitzhugonagumo(self):
        # TODO consider making diffusion coefficient also a parameter
        bkd = self.get_backend()
        time_residual_cls = BackwardEulerResidual
        newton_solver = NewtonSolver(verbosity=2, rtol=1e-6, atol=1e-6)
        model = FitzHughNagumoModel(
            0,
            50,
            1,
            time_residual_cls,
            newton_solver=newton_solver,
            backend=bkd,
        )
        sample = bkd.array([0.1, 0.01, 0.5, 1.0])[:, None]
        sols, times = model.forward_solve(sample)
        ani = animate_transient_2d_vector_solution(
            model.basis(),
            sols,
            times,
            model.physics().ncomponents(),
            [0, 1],
            [0, 1],
            51,
        )
        ani.save("fitzhugnagumo.gif", dpi=100)

    def test_steady_shallow_shelf_equation_2d(self):
        bkd = self.get_backend()
        newton_solver = NewtonSolver(verbosity=0, rtol=1e-12, atol=1e-6)
        model = SteadyShallowShelfModel2D(newton_solver, backend=bkd)
        sample = bkd.asarray(np.random.normal(0.0, 1.0, (model.nvars(), 1)))

        # Todo most of the following can be moved to an code example page
        # model.physics().set_param(sample[:, 0])
        # from pyapprox.pde.collocation.functions import plot_vector_function
        # axs = plt.subplots(1, 4, figsize=(4*8, 6))[1]
        # im = model._depth.plot(axs[0])
        # plt.colorbar(im, ax=axs[0])
        # im = model._surface.plot(axs[1], levels=51)
        # plt.colorbar(im, ax=axs[1])
        # im = model._bed.plot(axs[2])
        # plt.colorbar(im, ax=axs[2])
        # im = model.physics()._friction.plot(axs[3])
        # plt.colorbar(im, ax=axs[3])

        # axs = plt.subplots(1, 3, figsize=(4*8, 6))[1]
        # from pyapprox.pde.collocation.functions import ScalarFunction
        # for ii in range(3):
        #     kle_mode = ScalarFunction(
        #         model.basis(), model.physics()._friction._kle._eig_vecs[:, ii]
        #     )
        #     im = kle_mode.plot(axs[ii])
        #     plt.colorbar(im, ax=axs[ii])

        # sol = model.forward_solve(sample)
        # res_array = model._adjoint_solver._newton_solver._residual(
        #     sol.get_flattened_values()
        # )
        # res = model.physics().solution_from_array(res_array)
        # plot_vector_function(sol)
        # plot_vector_function(res)
        # fig = plt.figure(figsize=(2*8, 6))
        # ax0 = fig.add_subplot(121, projection="3d")
        # ax1 = fig.add_subplot(122, projection="3d")
        # model._depth.plot(ax0)
        # model._surface.plot(ax1)
        # #plt.show()
        # #assert False

        # warning test assumes mesh_npts_1d = [15, 15]
        # checks of apply jacobian and hessian can not converge for certain
        # finite difference sizes, if initial guess is not tuned correctly, .e.g
        # by modifying the number of picard iterations
        functional = SumFunctional(model)
        model.set_functional(functional)
        # newton_solver._verbosity = 0
        # compare times of lagrange based apply_hessian and bkd.hvp by timing
        # using mesh_npts_1d = [25, 25]
        # Spoliler hvp is very slow with torch
        # vec = bkd.array(np.random.normal(0, 1, (model.nvars(), 1)))
        # model.apply_hessian(sample, vec)
        # bkd.hvp(lambda x: model(x[:, None]), sample[:, 0], vec[:, 0])

        errors = model.check_apply_jacobian(sample, disp=True)
        print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 2e-7

        # from pyapprox.util.utilities import approx_jacobian

        # def gfun(param):
        #     return model.forward_solve(param[:, None]).get_values().flatten()

        # assert bkd.allclose(
        #     approx_jacobian(gfun, sample[:, 0], bkd=bkd),
        #     bkd.jacobian(gfun, sample[:, 0]),
        #     atol=1e-6,
        # )
        # assert False
        if not bkd.hessian_implemented():
            return

        from pyapprox.optimization.minimize import approx_hessian

        # print(
        #     approx_hessian(
        #         lambda x: model.jacobian(x[:, None])[0], sample[:, 0], bkd=bkd
        #     )
        #     - bkd.hessian(lambda x: model(x[:, None]), sample[:, 0]),
        # )
        # assert bkd.allclose(
        #     approx_hessian(
        #         lambda x: model.jacobian(x[:, None])[0], sample[:, 0], bkd=bkd
        #     ),
        #     bkd.hessian(lambda x: model(x[:, None]), sample[:, 0]),
        #     atol=1e-6,
        # )
        # vec = bkd.ones((sample.shape[0], 1))
        # print(model.apply_hessian(sample, vec))
        # print(bkd.hvp(lambda x: model(x[:, None]), sample[:, 0], vec[:, 0]))
        # assert False

        errors = model.check_apply_hessian(sample, None, disp=True)
        print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 4e-7


class TestNumpyParameterizedModels(TestParameterizedModels, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchParameterizedModels(TestParameterizedModels, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main()
