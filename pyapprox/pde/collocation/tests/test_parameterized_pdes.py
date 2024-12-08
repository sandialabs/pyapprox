import unittest

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.pde.collocation.parameterized_pdes import (
    LotkaVolterraModel,
    TransientAdvectionDiffusionReactionModel,
    FitzHughNagumoModel,
)
from pyapprox.pde.collocation.timeintegration import (
    BackwardEulerResidual,
    CrankNicholsonResidual,
    ForwardEulerResidual,
    HeunResidual,
    TransientMSEAdjointFunctional,
)
from pyapprox.pde.collocation.newton import NewtonSolver
from pyapprox.pde.collocation.functions import (
    animate_transient_2d_scalar_solution,
    animate_transient_2d_vector_solution,
)


class TestParameterizedModels:
    def setUp(self):
        np.random.seed(1)

    def _check_lotka_volterra(self, time_residual_cls):
        bkd = self.get_backend()
        newton_solver = NewtonSolver(verbosity=0, rtol=1e-12, atol=1e-12)
        model = LotkaVolterraModel(
            0,
            10,
            1,
            time_residual_cls,
            newton_solver=newton_solver,
            backend=bkd,
        )
        obs_sample = bkd.array(np.random.uniform(0.3, 0.7, model.nvars()))[
            :, None
        ]
        model_obs_sol, model_obs_times = model.forward_solve(obs_sample)
        obs_time_indices = bkd.arange(model_obs_times.shape[0], dtype=int)
        # observe the 0th and at all time points, 2nd state at every second time point
        # do not observe 1st state
        obs_time_tuples = [
            (0, obs_time_indices),
            (2, obs_time_indices[::2]),
        ]
        functional = TransientMSEAdjointFunctional(
            3, model.nvars(), obs_time_tuples, backend=bkd
        )
        obs = functional.observations_from_solution(model_obs_sol)
        functional.set_observations(obs)
        # The error in check apply jacobian depend on newton tolerance
        # because finite difference is only accurate to that tolerance
        model.set_functional(functional)
        sample = bkd.array(np.random.uniform(0.3, 0.7, model.nvars()))[:, None]
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        errors = model.check_apply_jacobian(sample, fd_eps, disp=True)
        print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 1.3e-6

    def test_lotka_volterra(self):
        test_cases = [
            [BackwardEulerResidual],
            [CrankNicholsonResidual],
            [ForwardEulerResidual],
            [HeunResidual],
        ]
        for test_case in test_cases:
            self._check_lotka_volterra(*test_case)

    def test_advection_diffusion_reaction(self):
        bkd = self.get_backend()
        time_residual_cls = BackwardEulerResidual
        newton_solver = NewtonSolver(verbosity=2, rtol=1e-8, atol=1e-8)
        model = TransientAdvectionDiffusionReactionModel(
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

        from pyapprox.pde.collocation.functions import nabla

        ax = plt.figure().gca()
        velocity = nabla(
            -model._diffusion * model.physics().solution_from_array(sol[:, -1])
        )
        velocity.plot_vector_field(ax)
        plt.show()

        print(sol.max(axis=0))

        ani = animate_transient_2d_scalar_solution(
            model._basis, sol, times, plot_surface=False
        )
        ani.save("diffusion.gif", dpi=100)

    def test_fitzhugonagumo(self):
        #TODO consider making diffusion coefficient also a parameter
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
            51
        )
        ani.save("fitzhugnagumo.gif", dpi=100)


class TestNumpyParameterizedModels(TestParameterizedModels, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchParameterizedModels(TestParameterizedModels, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
