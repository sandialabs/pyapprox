import unittest
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.pde.collocation.timeintegration import (
    TransientNewtonResidual,
    ImplicitTimeIntegrator,
    BackwardEulerResidual,
    CrankNicholsonResidual,
    ForwardEulerResidual,
    HeunResidual,
    RK4,
    AdjointFunctional,
    TransientAdjointModel,
)


class LinearDecoupledODE(TransientNewtonResidual):
    def __init__(self, backend):
        super().__init__(backend)

    def set_time(self, time: float):
        self._time = time

    def set_param(self, param: Array):
        self._param = param
        self._coef = param[0]
        self._init_cond = param[1]

    def __call__(self, sol: Array) -> Array:
        b = self._coef**2 * self._bkd.arange(
            1, sol.shape[0] + 1, dtype=self._bkd.double_type()
        )
        return -b * sol

    def jacobian(self, sol: Array) -> Array:
        b = self._coef**2 * self._bkd.arange(
            1, sol.shape[0] + 1, dtype=self._bkd.double_type()
        )
        return -self._bkd.diag(b)

    def _initial_residual_param_jacobian(self) -> Array:
        nstates = 2
        return self._bkd.stack(
            [
                self._bkd.full((nstates,), 0),
                self._bkd.full((nstates,), -1),
            ],
            axis=1,
        )

    def _residual_param_jacobian(self, sol: Array) -> Array:
        nstates = sol.shape[0]
        if self._time == 0:
            raise RuntimeError("self._time must be > 0")
        return self._bkd.stack(
            (
                -2
                * self._coef
                * self._bkd.arange(
                    1, nstates + 1, dtype=self._bkd.double_type()
                )
                * sol,
                self._bkd.full((nstates,), 0),
            ),
            axis=1,
        )


class NonlinearDecoupledODE(TransientNewtonResidual):
    def __init__(self, backend):
        super().__init__(backend)

    def set_time(self, time: float):
        self._time = time

    def set_param(self, param):
        self._param = param[0]

    def __call__(self, sol):
        b = self._param**2 * self._bkd.arange(
            1, sol.shape[0] + 1, dtype=self._bkd.double_type()
        )
        return -b * sol**2

    def jacobian(self, sol):
        b = self._param**2 * self._bkd.arange(
            1, sol.shape[0] + 1, dtype=self._bkd.double_type()
        )
        return self._bkd.diag(-2 * b * sol)


class Functional(AdjointFunctional):
    # TODO: for now assume fixed timestep but allow for variable timestep
    # e.g. by passing in list of times
    def __init__(self, deltat, backend=NumpyLinAlgMixin):
        self._bkd = backend
        self._deltat = deltat

    def nstates(self):
        return 2

    def nparams(self):
        return 2

    def _value(self, sol: Array) -> Array:
        # todo extract deltat from times to allow different lengths
        # print(self._param)
        # print(self._bkd.atleast1d(self._bkd.sum(sol[0, 1:])*deltat))
        return self._bkd.atleast1d(self._bkd.sum(sol[0, 1:]) * self._deltat)

    def _qoi_sol_jacobian(self, sol: Array) -> Array:
        e1 = self._bkd.zeros((self.nstates(),))
        e1[0] = 1.0
        return self._deltat * self._bkd.stack([e1] * sol.shape[1], axis=1)

    def _qoi_param_jacobian(self, sol: Array) -> Array:
        return self._bkd.zeros((self.nparams(),))


class LinearDecoupledODEModel(TransientAdjointModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        backend=NumpyLinAlgMixin,
    ):
        self._init_time = init_time
        self._final_time = final_time
        self._deltat = deltat
        super().__init__(backend)

    def _setup_residual(self):
        self._residual = BackwardEulerResidual(LinearDecoupledODE(self._bkd))

    def _setup_time_integrator(self):
        self._time_int = ImplicitTimeIntegrator(
            self._residual,
            self._init_time,
            self._final_time,
            self._deltat,
            verbosity=0,
        )

    def _setup_functional(self):
        self.functional = Functional(self._deltat, self._bkd)

    def set_param(self, sample: Array):
        if sample.ndim != 2 or sample.shape != (2, 1):
            raise ValueError(
                "sample has shape {0} but must have shape (2, 1)".format(
                    sample.shape
                )
            )
        self._residual._residual.set_param(sample[:, 0])
        self.functional.set_param(sample[:, 0])

    def get_initial_solution(self):
        # do not use bkd.full as it will mess up torch autograd
        return self.functional._param[1] * self._bkd.ones(
            (self.functional.nstates(),)
        )


class TestCollocation:
    def setUp(self):
        np.random.seed(1)

    def test_decoupled_linear_ode(self):
        bkd = self.get_backend()

        param, nstates = bkd.array([4., 3.0]), 2
        init_time, final_time = 0, 0.25
        deltat = 0.125
        residual = BackwardEulerResidual(LinearDecoupledODE(bkd))
        # init_time, final_time = 0, 0.25
        # deltat = 0.001
        # residual = CrankNicholsonResidual(LinearDecoupledODE(bkd))
        residual._residual.set_param(param)
        time_int = ImplicitTimeIntegrator(
            residual, init_time, final_time, deltat, verbosity=0
        )
        init_sol = param[1] * bkd.ones((nstates,))
        sols, times = time_int.solve(init_sol)
        assert bkd.allclose(
            times,
            bkd.arange(
                init_time, final_time + deltat, deltat, dtype=bkd.double_type()
            ),
        )

        # exact_sols = init_sol[:, None] * bkd.exp(
        #     -times[None, :]
        #     * (
        #         param[0] ** 2
        #         * bkd.arange(1, nstates + 1, dtype=bkd.double_type())
        #     )[:, None]
        # )
        # ax = plt.figure().gca()
        # ax.plot(times, exact_sols.T)
        # ax.plot(times, sols.T, "--")
        # print(bkd.abs(exact_sols - sols).max())
        # true for final_time = 0.25, dt = 0.001 crank nic
        # assert bkd.abs(exact_sols - sols).max() < 1e-5

        model = LinearDecoupledODEModel(init_time, final_time, deltat, bkd)
        sample = param[:, None]
        model(sample)  # needed so that model._sols is created
        # check forward solution
        scale = bkd.arange(
            1, model.functional.nstates() + 1, dtype=bkd.double_type()
        )
        exact_sols = bkd.stack(
            [
                bkd.full((model.functional.nstates(),), param[1]),
                sols[:, 0] / (1 + deltat * scale * param[0] ** 2),
                sols[:, 1] / (1 + deltat * scale * param[0] ** 2),
            ],
            axis=1,
        )
        assert bkd.allclose(sols, exact_sols)
        print(exact_sols, "exact sols")

        residual = model._time_int.newton_solver._residual
        res_param_jac0 = bkd.stack(
            [
                bkd.zeros((model._sols.shape[0],)),
                bkd.full((model._sols.shape[0],), -1.0),
            ],
            axis=1,
        )
        assert bkd.allclose(
            res_param_jac0,
            residual.initial_residual_param_jacobian(),
        )
        drdp = [res_param_jac0]
        for ii, time in enumerate(times[1:], start=1):
            residual.set_time(time, deltat, model._sols[:, ii - 1])
            drdp.append(residual.residual_param_jacobian(model._sols[:, ii]))
        drdp = bkd.vstack(drdp)

        deltat1, deltat2 = times[1:] - times[:-1]
        res_param_jac1 = bkd.stack(
            [
                (2 * deltat1 * scale * param[0] * param[1])
                / ((deltat1 * scale * param[0] ** 2 + 1)),
                bkd.zeros((scale.shape[0],)),
            ],
            axis=1,
        )
        res_param_jac2 = bkd.stack(
            [
                (2 * deltat2 * scale * param[0] * param[1])
                / (
                    (deltat1 * scale * param[0] ** 2 + 1)
                    * (deltat2 * scale * param[0] ** 2 + 1)
                ),
                bkd.zeros((scale.shape[0],)),
            ],
            axis=1,
        )

        exact_drdp = bkd.vstack(
            [res_param_jac0, res_param_jac1, res_param_jac2]
        )
        #print(exact_drdp, "exacxt drdp")
        #print(drdp)
        assert bkd.allclose(drdp, exact_drdp)

        if bkd.jacobian_implemented():
            # check jacobian of residual with respect to parameters using autograd
            # this is redundant when using derived solutions like tested here
            # but this test is useful when such expressions for drdp_exact
            # do not exist
            def fun(time, prev_sol, sol, param):
                residual = model._time_int.newton_solver._residual
                residual._residual.set_param(param)
                residual.set_time(time, deltat, prev_sol)
                return residual(sol)

            auto_drdp = [res_param_jac0]
            for ii, time in enumerate(times[1:], start=1):
                partial_fun = partial(
                    fun, time, model._sols[:, ii - 1], model._sols[:, ii]
                )
                auto_drdp.append(bkd.jacobian(partial_fun, param))
            auto_drdp = bkd.vstack(auto_drdp)
            assert bkd.allclose(auto_drdp, exact_drdp)

        adj_sols = model._time_int.solve_adjoint(model._sols, model._times)
        deltat1, deltat2 = times[1:] - times[:-1]
        exact_adj_sols = bkd.stack(
            [
                -(deltat1 * (1 + deltat2 * scale * param[0] ** 2) + deltat2)
                / (
                    (1 + deltat1 * scale * param[0] ** 2)
                    * (1 + deltat2 * scale * param[0] ** 2)
                ),
                -(deltat1 * (1 + deltat2 * scale * param[0] ** 2) + deltat2)
                / (
                    (1 + deltat1 * scale * param[0] ** 2)
                    * (1 + deltat2 * scale * param[0] ** 2)
                ),
                -deltat2 / (1+deltat2 * scale * param[0] ** 2),
            ],
            axis=1,
        )
        # The qoi only depends on the first state at each time step
        exact_adj_sols[1:] = 0.0
        assert bkd.allclose(adj_sols, exact_adj_sols)

        # print(model.jacobian(sample), "grad")
        # print(bkd.grad(model._evaluate, sample)[1], "grad_auto")
        if bkd.jacobian_implemented():
            assert bkd.allclose(
                model.jacobian(sample), bkd.grad(model._evaluate, sample)[1].T
            )

        errors = model.check_apply_jacobian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def _check_decoupled_nonlinear_ode(self, residual_class, deltat, tol):
        bkd = self.get_backend()

        param, nstates = bkd.array([3.0]), 2
        init_time, final_time = 0, 0.25
        residual = residual_class(NonlinearDecoupledODE(bkd))
        residual._residual.set_param(param)
        time_int = ImplicitTimeIntegrator(
            residual, init_time, final_time, deltat, verbosity=0
        )
        init_sol = bkd.full((nstates,), 2)
        sols, times = time_int.solve(init_sol)
        assert bkd.allclose(
            times,
            bkd.arange(
                init_time, final_time + deltat, deltat, dtype=bkd.double_type()
            ),
        )
        exact_sols = 1 / (
            times[None, :]
            * (
                (
                    param**2
                    * bkd.arange(1, nstates + 1, dtype=bkd.double_type())[
                        :, None
                    ]
                )
            )
            + 1 / init_sol[:, None]
        )
        ax = plt.figure().gca()
        ax.plot(times, exact_sols.T)
        ax.plot(times, sols.T, "--")
        # plt.show()
        print(bkd.abs(exact_sols - sols).max())
        assert bkd.abs(exact_sols - sols).max() < tol

    def test_decoupled_nonlinear_ode(self):
        test_cases = [
            [BackwardEulerResidual, 0.0001, 2e-3],
            [CrankNicholsonResidual, 0.0001, 3e-5],
            [ForwardEulerResidual, 0.0001, 2e-3],
            [HeunResidual, 0.0001, 3e-5],
            [RK4, 0.0001, 2e-12],
        ]
        for test_case in test_cases:
            self._check_decoupled_nonlinear_ode(*test_case)


class TestNumpyCollocation(TestCollocation, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchCollocation(TestCollocation, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
