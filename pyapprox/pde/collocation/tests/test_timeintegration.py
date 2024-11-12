import unittest
from functools import partial

import numpy as np

# from pyapprox.util.print_wrapper import *
from pyapprox.util.linearalgebra.linalgbase import Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.pde.collocation.timeintegration import (
    TransientNewtonResidual,
    ImplicitTimeIntegrator,
    BackwardEulerResidual,
    SymplecticMidpointResidual,
    CrankNicholsonResidual,
    ForwardEulerResidual,
    HeunResidual,
    RK4,
    TransientAdjointFunctional,
    TimeIntegratorNewtonResidual,
    NewtonSolver,
)
from pyapprox.pde.collocation.adjoint_models import TransientAdjointModel


class LinearDecoupledODE(TransientNewtonResidual):
    def __init__(self, nstates, transient_coef, backend):
        super().__init__(backend)
        self._nstates = nstates
        self._transient_coef = transient_coef

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
        if self._transient_coef:
            b = b * (2 + self._time)
        return -b * sol

    def jacobian(self, sol: Array) -> Array:
        b = self._coef**2 * self._bkd.arange(
            1, sol.shape[0] + 1, dtype=self._bkd.double_type()
        )
        if self._transient_coef:
            b = b * (2 + self._time)
        return -self._bkd.diag(b)

    def _initial_param_jacobian(self) -> Array:
        return self._bkd.stack(
            [
                self._bkd.full((self._nstates,), 0),
                self._bkd.full((self._nstates,), -1),
            ],
            axis=1,
        )

    def _param_jacobian(self, sol: Array) -> Array:
        nstates = sol.shape[0]
        if self._transient_coef:
            coef = 2 + self._time
        else:
            coef = 1
        return self._bkd.stack(
            (
                -2
                * coef
                * self._coef
                * self._bkd.arange(
                    1, nstates + 1, dtype=self._bkd.double_type()
                )
                * sol,
                self._bkd.full((nstates,), 0),
            ),
            axis=1,
        )


class NonLinearDecoupledODE(TransientNewtonResidual):
    def __init__(self, nstates, transient_coef, backend):
        self._nstates = nstates
        self._transient_coef = transient_coef
        super().__init__(backend)

    def set_time(self, time: float):
        self._time = time

    def set_param(self, param):
        self._param = param
        self._coef = param[0]
        self._init_cond = param[1]

    def __call__(self, sol):
        b = self._coef**2 * self._bkd.arange(
            1, sol.shape[0] + 1, dtype=self._bkd.double_type()
        )
        if self._transient_coef:
            b *= 2 + self._time
        return -b * sol**2

    def jacobian(self, sol):
        b = self._coef**2 * self._bkd.arange(
            1, sol.shape[0] + 1, dtype=self._bkd.double_type()
        )
        if self._transient_coef:
            b *= 2 + self._time
        return self._bkd.diag(-2 * b * sol)

    def _initial_param_jacobian(self) -> Array:
        nstates = self._nstates
        return self._bkd.stack(
            [
                self._bkd.full((nstates,), 0),
                self._bkd.full((nstates,), -1),
            ],
            axis=1,
        )

    def _param_jacobian(self, sol: Array) -> Array:
        nstates = sol.shape[0]
        if self._transient_coef:
            coef = 2 + self._time
        else:
            coef = 1
        return self._bkd.stack(
            (
                -2
                * coef
                * self._coef
                * self._bkd.arange(
                    1, nstates + 1, dtype=self._bkd.double_type()
                )
                * sol**2,
                self._bkd.full((nstates,), 0),
            ),
            axis=1,
        )


class TransientSingleStateLinearFunctional(TransientAdjointFunctional):
    def __init__(self, nstates, backend=NumpyLinAlgMixin):
        self._nstates = nstates
        self._bkd = backend

    def nstates(self):
        return self._nstates

    def nparams(self):
        return 2

    def _value(self, sol: Array) -> Array:
        return self._bkd.atleast1d(self._bkd.sum(sol[0, :] * self._quadw))

    def _qoi_sol_jacobian(self, sol: Array) -> Array:
        e1 = self._bkd.zeros((self.nstates(),))
        e1[0] = 1.0
        dqdu = self._bkd.stack([e1] * (sol.shape[1]), axis=1) * self._quadw
        return dqdu

    def _qoi_param_jacobian(self, sol: Array) -> Array:
        return self._bkd.zeros((self.nparams(),))


class TransientSingleStateNonLinearFunctional(
    TransientSingleStateLinearFunctional
):
    def _value(self, sol: Array) -> Array:
        return self._bkd.atleast1d(self._bkd.sum(sol[0, :] ** 2 * self._quadw))

    def _qoi_sol_jacobian(self, sol: Array) -> Array:
        e1 = self._bkd.zeros((self.nstates(),))
        # print(sol.shape, e1.shape, self._quadw.shape)
        e1[0] = 1.0
        dqdu = e1[:, None] * 2 * sol * self._quadw
        return dqdu


class LinearDecoupledODEModel(TransientAdjointModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        time_residual_cls: TimeIntegratorNewtonResidual,
        nstates: int = 2,
        transient_coef: bool = False,
        backend=NumpyLinAlgMixin,
    ):
        time_residual = self._setup_residual(
            time_residual_cls, nstates, transient_coef, backend
        )
        functional = self._setup_functional(nstates, backend)
        super().__init__(
            init_time,
            final_time,
            deltat,
            time_residual,
            functional,
            None,
            backend,
        )

    def _setup_residual(self, time_residual_cls, nstates, transient_coef, bkd):
        return time_residual_cls(
            LinearDecoupledODE(nstates, transient_coef, bkd)
        )

    def _setup_functional(self, nstates, bkd):
        return TransientSingleStateLinearFunctional(nstates, bkd)

    def get_initial_condition(self):
        # do not use bkd.full as it will mess up torch autograd
        return self._functional._param[1] * self._bkd.ones(
            (self._functional.nstates(),)
        )


class NonLinearDecoupledODEModel(LinearDecoupledODEModel):
    def _setup_residual(self, time_residual_cls, nstates, transient_coef, bkd):
        return time_residual_cls(
            NonLinearDecoupledODE(nstates, transient_coef, bkd)
        )

    def _setup_functional(self, nstates, bkd):
        return TransientSingleStateNonLinearFunctional(nstates, bkd)


class TestTimeIntegration:
    def setUp(self):
        np.random.seed(1)

    def test_decoupled_linear_ode_forward_euler(self):
        bkd = self.get_backend()

        nstates = 3
        param = bkd.array([4.0, 3.0])
        init_time, final_time = 0, 0.25
        deltat = 0.13  # intentionally create smaller last time step
        model = LinearDecoupledODEModel(
            init_time,
            final_time,
            deltat,
            ForwardEulerResidual,
            nstates,
            False,
            bkd,
        )
        sample = param[:, None]
        model(sample)  # needed so that model._sols is created
        scale = bkd.arange(
            1, model._functional.nstates() + 1, dtype=bkd.double_type()
        )
        times = model._times
        sols = model._sols
        deltat1, deltat2 = times[1:] - times[:-1]
        exact_sols = bkd.stack(
            [
                bkd.full((model._functional.nstates(),), param[1]),
                (1 - deltat1 * scale * param[0] ** 2) * sols[:, 0],
                (1 - deltat2 * scale * param[0] ** 2) * sols[:, 1],
            ],
            axis=1,
        )
        assert bkd.allclose(sols, exact_sols, atol=1e-15, rtol=1e-15)

        time_residual = model._time_int.time_residual
        res_param_jac0 = bkd.stack(
            [
                bkd.zeros((model._sols.shape[0],)),
                bkd.full((model._sols.shape[0],), -1.0),
            ],
            axis=1,
        )
        assert bkd.allclose(
            res_param_jac0,
            time_residual.initial_param_jacobian(),
        )
        drdp = [res_param_jac0]
        for ii, time in enumerate(times[:-1], start=0):
            print(time)
            time_residual.set_time(
                time, times[ii + 1] - times[ii], model._sols[:, ii]
            )
            drdp.append(
                time_residual.param_jacobian(
                    model._sols[:, ii], model._sols[:, ii + 1]
                )
            )
        drdp = bkd.vstack(drdp)

        res_param_jac1 = bkd.stack(
            [
                (2 * deltat1 * scale * param[0]) * exact_sols[:, 0],
                bkd.zeros((scale.shape[0],)),
            ],
            axis=1,
        )
        res_param_jac2 = bkd.stack(
            [
                (2 * deltat2 * scale * param[0]) * (exact_sols[:, 1]),
                bkd.zeros((scale.shape[0],)),
            ],
            axis=1,
        )

        exact_drdp = bkd.vstack(
            [res_param_jac0, res_param_jac1, res_param_jac2]
        )
        assert bkd.allclose(drdp, exact_drdp)

        adj_sols = model._time_int.solve_adjoint(model._sols, model._times)
        deltat1, deltat2 = times[1:] - times[:-1]
        exact_adj_sols = bkd.stack(
            [
                deltat2 * (-1 + deltat1 * scale * param[0] ** 2) - deltat1,
                -deltat2 + scale * 0,
                scale * 0,
            ],
            axis=1,
        )
        # The qoi only depends on the first state at each time step
        exact_adj_sols[1:] = 0.0
        # print(adj_sols)
        # print(exact_adj_sols, "exact adj sols")
        assert bkd.allclose(adj_sols, exact_adj_sols, atol=1e-15, rtol=1e-15)

        if bkd.jacobian_implemented():
            assert bkd.allclose(
                model.jacobian(sample),
                bkd.grad(model._evaluate, sample)[1].T,
                atol=1e-15,
                rtol=1e-15,
            )

        errors = model.check_apply_jacobian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_decoupled_nonlinear_ode_forward_euler(self):
        bkd = self.get_backend()

        nstates = 3
        param = bkd.array([4.0, 3.0])
        init_time, final_time = 0, 0.25
        deltat = 0.12  # 0.13  # intentionally create smaller last time step
        model = NonLinearDecoupledODEModel(
            init_time,
            final_time,
            deltat,
            ForwardEulerResidual,
            nstates,
            True,
            bkd,
        )
        sample = param[:, None]
        model(sample)  # needed so that model._sols is created
        scale = bkd.arange(
            1, model._functional.nstates() + 1, dtype=bkd.double_type()
        )
        times = model._times
        sols = model._sols
        deltat1, deltat2, deltat3 = times[1:] - times[:-1]
        t0, t1, t2 = times[:3]
        y0 = bkd.full((model._functional.nstates(),), param[1])
        y1 = y0 - deltat1 * scale * param[0] ** 2 * (t0 + 2) * y0**2
        y2 = y1 - deltat2 * scale * param[0] ** 2 * (t1 + 2) * y1**2
        y3 = y2 - deltat3 * scale * param[0] ** 2 * (t2 + 2) * y2**2
        exact_sols = bkd.stack([y0, y1, y2, y3], axis=1)
        print(sols)
        print(exact_sols)
        assert bkd.allclose(sols, exact_sols, atol=1e-15, rtol=1e-15)

        time_residual = model._time_int.time_residual
        res_param_jac0 = bkd.stack(
            [
                bkd.zeros((model._sols.shape[0],)),
                bkd.full((model._sols.shape[0],), -1.0),
            ],
            axis=1,
        )
        assert bkd.allclose(
            res_param_jac0,
            time_residual.initial_param_jacobian(),
        )
        drdp = [res_param_jac0]
        for ii, time in enumerate(times[:-1], start=0):
            print(time)
            time_residual.set_time(
                time, times[ii + 1] - times[ii], model._sols[:, ii]
            )
            drdp.append(
                time_residual.param_jacobian(
                    model._sols[:, ii], model._sols[:, ii + 1]
                )
            )
        drdp = bkd.vstack(drdp)

        deltats = times[1:] - times[:-1]
        exact_drdp = bkd.vstack(
            [res_param_jac0]
            + [
                bkd.stack(
                    [
                        (2 * deltats[ii] * scale * param[0])
                        * exact_sols[:, ii] ** 2
                        * (times[ii] + 2),
                        bkd.zeros((scale.shape[0],)),
                    ],
                    axis=1,
                )
                for ii in range(3)
            ]
        )
        assert bkd.allclose(drdp, exact_drdp)

        adj_sols = model._time_int.solve_adjoint(model._sols, model._times)
        exact_adj_sols = bkd.stack(
            [
                2
                * deltat2
                * exact_sols[:, 1]
                * (
                    -1
                    + 2
                    * deltat1
                    * scale
                    * param[0] ** 2
                    * exact_sols[:, 0]
                    * (t0 + 2)
                )
                - 2 * deltat1 * exact_sols[:, 0]
                - 2
                * deltat3
                * exact_sols[:, 2]
                * (
                    4
                    * deltat1
                    * deltat2
                    * scale**2
                    * param[0] ** 4
                    * exact_sols[:, 0]
                    * exact_sols[:, 1]
                    * (t0 + 2)
                    * (t1 + 2)
                    - 2
                    * deltat1
                    * scale
                    * param[0] ** 2
                    * exact_sols[:, 0]
                    * (t0 + 2)
                    - 2
                    * deltat2
                    * scale
                    * param[0] ** 2
                    * exact_sols[:, 1]
                    * (t1 + 2)
                    + 1
                ),
                2
                * deltat3
                * exact_sols[:, 2]
                * (
                    -1
                    + 2
                    * deltat2
                    * scale
                    * param[0] ** 2
                    * exact_sols[:, 1]
                    * (t1 + 2)
                )
                - 2 * deltat2 * exact_sols[:, 1],
                -2 * deltat3 * exact_sols[:, 2],
                scale * 0,
            ],
            axis=1,
        )
        # The qoi only depends on the first state at each time step
        exact_adj_sols[1:] = 0.0
        # print(adj_sols)
        # print(exact_adj_sols, "exact adj")
        assert bkd.allclose(adj_sols, exact_adj_sols, atol=1e-15, rtol=1e-15)

        if bkd.jacobian_implemented():
            assert bkd.allclose(
                model.jacobian(sample),
                bkd.grad(model._evaluate, sample)[1].T,
                atol=1e-15,
                rtol=1e-15,
            )

        errors = model.check_apply_jacobian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_decoupled_linear_ode_heun(self):
        bkd = self.get_backend()
        nstates = 3
        param = bkd.array([4.0, 3.0])
        init_time, final_time = 0, 0.25
        deltat = 0.13  # intentionally create smaller last time step
        model = LinearDecoupledODEModel(
            init_time, final_time, deltat, HeunResidual, nstates, False, bkd
        )
        sample = param[:, None]
        model(sample)  # needed so that model._sols is created
        scale = bkd.arange(
            1, model._functional.nstates() + 1, dtype=bkd.double_type()
        )
        times = model._times
        sols = model._sols
        deltat1, deltat2 = times[1:] - times[:-1]
        b, a = param
        exact_sols = [
            bkd.full((model._functional.nstates(),), param[1]),
            0.5
            * deltat1
            * (
                -scale * b**2 * a
                - scale * b**2 * (-deltat1 * scale * b**2 * a + a)
            )
            + a,
        ]
        exact_sols.append(
            0.5
            * deltat2
            * (
                -scale * b**2 * exact_sols[1]
                - scale
                * b**2
                * (-deltat2 * scale * b**2 * exact_sols[1] + exact_sols[1])
            )
            + exact_sols[1],
        )
        exact_sols = bkd.stack(exact_sols, axis=1)
        assert bkd.allclose(sols, exact_sols, atol=1e-15, rtol=1e-15)

        # time_residual = model._time_int.time_residual
        time_residual = model._time_int.time_residual
        res_param_jac0 = bkd.stack(
            [
                bkd.zeros((model._sols.shape[0],)),
                bkd.full((model._sols.shape[0],), -1.0),
            ],
            axis=1,
        )
        assert bkd.allclose(
            res_param_jac0,
            time_residual.initial_param_jacobian(),
        )
        drdp = [res_param_jac0]
        for ii, time in enumerate(times[:-1], start=0):
            print(time)
            time_residual.set_time(
                time, times[ii + 1] - times[ii], model._sols[:, ii]
            )
            drdp.append(
                time_residual.param_jacobian(
                    model._sols[:, ii], model._sols[:, ii + 1]
                )
            )
        drdp = bkd.vstack(drdp)

        res_param_jac1 = bkd.stack(
            [
                -0.5
                * deltat1
                * (
                    2 * deltat1 * scale**2 * b**3 * exact_sols[:, 0]
                    - 2 * scale * b * exact_sols[:, 0]
                    - 2
                    * scale
                    * b
                    * (
                        -deltat1 * scale * b**2 * exact_sols[:, 0]
                        + exact_sols[:, 0]
                    )
                ),
                bkd.zeros((scale.shape[0],)),
            ],
            axis=1,
        )
        res_param_jac2 = bkd.stack(
            [
                -0.5
                * deltat2
                * (
                    2 * deltat2 * scale**2 * b**3 * exact_sols[:, 1]
                    - 2 * scale * b * exact_sols[:, 1]
                    - 2
                    * scale
                    * b
                    * (
                        -deltat2 * scale * b**2 * exact_sols[:, 1]
                        + exact_sols[:, 1]
                    )
                ),
                bkd.zeros((scale.shape[0],)),
            ],
            axis=1,
        )
        exact_drdp = bkd.vstack(
            [res_param_jac0, res_param_jac1, res_param_jac2]
        )
        print(exact_drdp)
        print(drdp)
        assert bkd.allclose(drdp, exact_drdp)

        adj_sols = model._time_int.solve_adjoint(model._sols, model._times)
        # deltat1, deltat2 = times[1:] - times[:-1]
        # exact_adj_sols = bkd.stack(
        #     [
        #         scale,
        #         -deltat1 / 2 - deltat2 - deltat2**3*scale**2*b**4 / 4 + deltat2**2*scale*b**2 / 2,
        #         bkd.full((model._functional.nstates(),), -deltat2 / 2)
        #     ],
        #     axis=1,
        # )
        # # The qoi only depends on the first state at each time step
        # exact_adj_sols[1:] = 0.0
        # #print(exact_sols[:, 2])
        # print(adj_sols)
        # print(exact_adj_sols, "exact adj sols")
        print(deltat2 * (-(b**2) * (-deltat2 * b**2 + 1) - b**2) / 2 + 1, "1")
        print(deltat1 * (-(b**2) * (-deltat1 * b**2 + 1) - b**2) / 2 + 1, "2")
        # assert bkd.allclose(adj_sols, exact_adj_sols, atol=1e-15, rtol=1e-15)

        if bkd.jacobian_implemented():
            print(model.jacobian(sample), "j")
            print(bkd.grad(model._evaluate, sample)[1].T, "k")
            assert bkd.allclose(
                model.jacobian(sample),
                bkd.grad(model._evaluate, sample)[1].T,
                atol=1e-15,
                rtol=1e-15,
            )

        errors = model.check_apply_jacobian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_decoupled_nonlinear_ode_backward_euler(self):
        bkd = self.get_backend()

        nstates = 3
        param = bkd.array([4.0, 3.0])
        init_time, final_time = 0, 0.25
        deltat = 0.13  # intentionally create smaller last time step
        model = NonLinearDecoupledODEModel(
            init_time,
            final_time,
            deltat,
            BackwardEulerResidual,
            nstates,
            True,
            bkd,
        )
        sample = param[:, None]
        # tighten tolerances of newton solver from default
        # to allow exact solution comparison to use tol of 1e-15
        # Also for some reason autograd with torch produces a slightly different
        # grad than that computed with adjoints here as newton tolerances
        # are relaxed
        model._time_int.newton_solver._atol = 1e-10
        model._time_int.newton_solver._rtol = 1e-10
        model(sample)  # needed so that model._sols is created
        scale = bkd.arange(
            1, model._functional.nstates() + 1, dtype=bkd.double_type()
        )
        times = model._times
        sols = model._sols
        deltat1, deltat2 = times[1:] - times[:-1]
        t0, t1, t2 = times
        exact_sols = [bkd.full((model._functional.nstates(),), param[1])]
        exact_sols.append(
            (
                bkd.sqrt(
                    4
                    * deltat1
                    * scale
                    * param[0] ** 2
                    * exact_sols[-1]
                    * (t1 + 2)
                    + 1
                )
                - 1
            )
            / (2 * deltat1 * scale * param[0] ** 2 * (t1 + 2))
        )
        exact_sols.append(
            (
                bkd.sqrt(
                    4
                    * deltat2
                    * scale
                    * param[0] ** 2
                    * exact_sols[-1]
                    * (t2 + 2)
                    + 1
                )
                - 1
            )
            / (2 * deltat2 * scale * param[0] ** 2 * (t2 + 2))
        )
        exact_sols = bkd.stack(exact_sols, axis=1)
        print(exact_sols)
        print(sols)
        assert bkd.allclose(sols, exact_sols, atol=1e-7, rtol=1e-7)

        time_residual = model._time_int.time_residual
        res_param_jac0 = bkd.stack(
            [
                bkd.zeros((model._sols.shape[0],)),
                bkd.full((model._sols.shape[0],), -1.0),
            ],
            axis=1,
        )
        assert bkd.allclose(
            res_param_jac0,
            time_residual.initial_param_jacobian(),
        )
        drdp = [res_param_jac0]
        for ii, time in enumerate(times[:-1], start=0):
            time_residual.set_time(
                time, times[ii + 1] - times[ii], model._sols[:, ii]
            )
            drdp.append(
                time_residual.param_jacobian(
                    model._sols[:, ii], model._sols[:, ii + 1]
                )
            )
        drdp = bkd.vstack(drdp)

        res_param_jac1 = bkd.stack(
            [
                (2 * deltat1 * scale * param[0])
                * exact_sols[:, 1] ** 2
                * (t1 + 2),
                bkd.zeros((scale.shape[0],)),
            ],
            axis=1,
        )
        res_param_jac2 = bkd.stack(
            [
                (2 * deltat2 * scale * param[0])
                * exact_sols[:, 2] ** 2
                * (t2 + 2),
                bkd.zeros((scale.shape[0],)),
            ],
            axis=1,
        )

        exact_drdp = bkd.vstack(
            [res_param_jac0, res_param_jac1, res_param_jac2]
        )
        assert bkd.allclose(drdp, exact_drdp)

        adj_sols = model._time_int.solve_adjoint(model._sols, model._times)
        deltat1, deltat2 = times[1:] - times[:-1]
        exact_adj_sols1 = (
            2
            * (
                -2
                * deltat1
                * deltat2
                * param[0] ** 2
                * (t2 + 2)
                * exact_sols[:, 1]
                * exact_sols[:, 2]
                - deltat1 * exact_sols[:, 1]
                - deltat2 * exact_sols[:, 2]
            )
            / (
                4
                * deltat1
                * deltat2
                * param[0] ** 4
                * (t1 + 2)
                * (t2 + 2)
                * exact_sols[:, 1]
                * exact_sols[:, 2]
                + 2 * deltat1 * param[0] ** 2 * exact_sols[:, 1] * (t1 + 2)
                + 2 * deltat2 * param[0] ** 2 * exact_sols[:, 2] * (t2 + 2)
                + 1
            )
        )
        exact_adj_sols = bkd.stack(
            [
                exact_adj_sols1,
                exact_adj_sols1,
                -2
                * deltat2
                * exact_sols[:, 2]
                / (
                    2 * deltat2 * param[0] ** 2 * exact_sols[:, 2] * (t2 + 2)
                    + 1
                ),
            ],
            axis=1,
        )
        # The qoi only depends on the first state at each time step
        exact_adj_sols[1:] = 0.0
        assert bkd.allclose(adj_sols, exact_adj_sols, atol=1e-15, rtol=1e-15)

        if bkd.jacobian_implemented():
            assert bkd.allclose(
                model.jacobian(sample),
                bkd.grad(model._evaluate, sample)[1].T,
                atol=1e-15,
                rtol=1e-15,
            )

        errors = model.check_apply_jacobian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_decoupled_linear_ode_backward_euler(self):
        bkd = self.get_backend()

        nstates = 3
        param = bkd.array([4.0, 3.0])
        init_time, final_time = 0, 0.25
        deltat = 0.13  # intentionally create smaller last time step
        model = LinearDecoupledODEModel(
            init_time,
            final_time,
            deltat,
            BackwardEulerResidual,
            nstates,
            False,
            bkd,
        )
        sample = param[:, None]
        model(sample)  # needed so that model._sols is created
        # check forward solution
        scale = bkd.arange(
            1, model._functional.nstates() + 1, dtype=bkd.double_type()
        )
        times = model._times
        sols = model._sols
        deltat1, deltat2 = times[1:] - times[:-1]
        exact_sols = bkd.stack(
            [
                bkd.full((model._functional.nstates(),), param[1]),
                sols[:, 0] / (1 + deltat1 * scale * param[0] ** 2),
                sols[:, 1] / (1 + deltat2 * scale * param[0] ** 2),
            ],
            axis=1,
        )
        assert bkd.allclose(sols, exact_sols, atol=1e-15, rtol=1e-15)

        time_residual = model._time_int.time_residual
        res_param_jac0 = bkd.stack(
            [
                bkd.zeros((model._sols.shape[0],)),
                bkd.full((model._sols.shape[0],), -1.0),
            ],
            axis=1,
        )
        assert bkd.allclose(
            res_param_jac0,
            time_residual.initial_param_jacobian(),
        )
        drdp = [res_param_jac0]
        for ii, time in enumerate(times[:-1], start=0):
            print(time)
            time_residual.set_time(
                time, times[ii + 1] - times[ii], model._sols[:, ii]
            )
            drdp.append(
                time_residual.param_jacobian(
                    model._sols[:, ii], model._sols[:, ii + 1]
                )
            )
        drdp = bkd.vstack(drdp)

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
        # print(exact_drdp, "exacxt drdp")
        # print(drdp)
        assert bkd.allclose(drdp, exact_drdp)

        if bkd.jacobian_implemented():
            # check jacobian of residual w.r.t to parameters using autograd
            # this is redundant when using derived solutions like tested here
            # but this test is useful when such expressions for drdp_exact
            # do not exist
            def fun(time, deltat, prev_sol, sol, param):
                time_residual = model._time_int.time_residual
                time_residual.native_residual.set_param(param)
                time_residual.set_time(time, deltat, prev_sol)
                return time_residual(sol)

            auto_drdp = [res_param_jac0]
            for ii, time in enumerate(times[:-1], start=0):
                partial_fun = partial(
                    fun,
                    time,
                    times[ii + 1] - time,
                    model._sols[:, ii],
                    model._sols[:, ii + 1],
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
                -deltat2 / (1 + deltat2 * scale * param[0] ** 2),
            ],
            axis=1,
        )
        # The qoi only depends on the first state at each time step
        exact_adj_sols[1:] = 0.0
        assert bkd.allclose(adj_sols, exact_adj_sols, atol=1e-15, rtol=1e-15)

        # print(model.jacobian(sample), "grad")
        # print(bkd.grad(model._evaluate, sample)[1], "grad_auto")
        if bkd.jacobian_implemented():
            print(
                model.jacobian(sample)
                - bkd.grad(model._evaluate, sample)[1].T,
            )
            assert bkd.allclose(
                model.jacobian(sample),
                bkd.grad(model._evaluate, sample)[1].T,
                atol=1e-15,
                rtol=1e-15,
            )

        errors = model.check_apply_jacobian(sample, disp=True)
        assert errors.min() / errors.max() < 1e-6

    def test_decoupled_linear_ode_implicit_midpoint(self):
        bkd = self.get_backend()

        nstates = 3
        param = bkd.array([4.0, 3.0])
        init_time, final_time = 0, 0.25
        deltat = 0.13  # intentionally create smaller last time step
        model = LinearDecoupledODEModel(
            init_time,
            final_time,
            deltat,
            SymplecticMidpointResidual,
            nstates,
            True,
            bkd,
        )
        sample = param[:, None]
        model(sample)  # needed so that model._sols is created
        # check forward solution
        scale = bkd.arange(
            1, model._functional.nstates() + 1, dtype=bkd.double_type()
        )
        times = model._times
        sols = model._sols
        deltat1, deltat2 = times[1:] - times[:-1]
        t0, t1 = times[:2]
        y1 = (
            param[1]
            * (
                4
                - deltat1**2 * scale * param[0] ** 2
                - 2 * deltat1 * scale * param[0] ** 2 * (t0 + 2)
            )
            / (
                4
                + deltat1**2 * scale * param[0] ** 2
                + 2 * deltat1 * scale * param[0] ** 2 * (t0 + 2)
            )
        )
        exact_sols = bkd.stack(
            [
                bkd.full((model._functional.nstates(),), param[1]),
                y1,
                y1
                * (
                    4
                    - deltat2**2 * scale * param[0] ** 2
                    - 2 * deltat2 * scale * param[0] ** 2 * (t1 + 2)
                )
                / (
                    4
                    + deltat2**2 * scale * param[0] ** 2
                    + 2 * deltat2 * scale * param[0] ** 2 * (t1 + 2)
                ),
            ],
            axis=1,
        )
        assert bkd.allclose(sols, exact_sols, atol=1e-15, rtol=1e-15)

    def _check_decoupled_nonlinear_ode(self, time_residual_cls, deltat, tol):
        bkd = self.get_backend()

        nstates = 2
        param = bkd.array([4.0, 3.0])
        init_time, final_time = 0, 0.25

        nstates = 3
        transient_coef = True
        model = NonLinearDecoupledODEModel(
            init_time,
            final_time,
            deltat,
            time_residual_cls,
            nstates,
            transient_coef,
            bkd,
        )
        model._time_int.newton_solver._atol = 1e-8
        model._time_int.newton_solver._rtol = 1e-8
        sample = param[:, None]
        model(sample)  # needed so that model._sols is created

        if transient_coef:
            c0, c1 = 2, 1
        else:
            c0, c1 = 1, 0
        scale = bkd.arange(1, nstates + 1, dtype=bkd.double_type())
        exact_sols = (
            2
            * param[1]
            / (
                (
                    2
                    + (
                        param[1]
                        * scale[:, None]
                        * param[0] ** 2
                        * model._times[None, :]
                    )
                    * (2 * c0 + c1 * model._times)
                )
            )
        )
        print(bkd.abs(exact_sols - model._sols).max())
        assert bkd.abs(exact_sols - model._sols).max() < tol

        if not model._time_int.time_residual.adjoint_implemented():
            return False

        if bkd.jacobian_implemented():
            print(model.jacobian(sample))
            print(bkd.grad(model._evaluate, sample)[1].T)
            print(
                model.jacobian(sample)
                - bkd.grad(model._evaluate, sample)[1].T,
                time_residual_cls,
            )
            # For some reason autograd with torch produces a slightly different
            # grad than that computed with adjoints here as newton tolerances
            # are relaxed
            assert bkd.allclose(
                model.jacobian(sample),
                bkd.grad(model._evaluate, sample)[1].T,
            )

        errors = model.check_apply_jacobian(sample, disp=True)
        # print(errors.min() / errors.max())
        assert errors.min() / errors.max() < 1e-6

    def test_decoupled_nonlinear_ode(self):
        test_cases = [
            [ForwardEulerResidual, 1e-4, 2e-2],
            [BackwardEulerResidual, 1e-4, 2e-2],
            [HeunResidual, 1e-4, 3e-4],
            [CrankNicholsonResidual, 1e-4, 3e-4],
            [SymplecticMidpointResidual, 0.0001, 9e-3],
            [RK4, 0.0001, 1e-8],
        ]
        for test_case in test_cases:
            self._check_decoupled_nonlinear_ode(*test_case)


class TestNumpyTimeIntegration(TestTimeIntegration, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


class TestTorchTimeIntegration(TestTimeIntegration, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()


# TODO add test with coefficient entering functional
