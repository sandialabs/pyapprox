from scipy import stats
import numpy as np

from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.benchmarks.base import SingleModelBenchmark
from pyapprox.pde.collocation.newton import (
    ParameterizedNewtonResidualMixin,
    NewtonSolver,
)
from pyapprox.pde.collocation.timeintegration import (
    TransientNewtonResidual,
    TransientSingleStateFinalTimeFunctional,
    Functional,
    BackwardEulerResidual,
    TransientAdjointFunctional,
    TransientMSEAdjointFunctional,
)
from pyapprox.pde.collocation.adjoint_models import (
    TransientAdjointModel,
    TimeIntegratorNewtonResidual,
)


class ParameterizedChemicalReactionResidual(
    TransientNewtonResidual, ParameterizedNewtonResidualMixin
):
    def set_time(self, time: float):
        self._time = time

    def nstates(self) -> int:
        return 3

    def set_param(self, param: Array):
        if param.shape[0] != self.nvars():
            raise ValueError("param has the wrong shape")
        self._param = param

    def __call__(self, sol: Array) -> Array:
        a, b, c, d, e, f = self._param
        z = 1.0 - sol[0] - sol[1] - sol[2]
        val = self._bkd.hstack(
            (
                a * z - c * sol[0] - 4 * d * sol[0] * sol[1],
                2 * b * z**2 - 4 * d * sol[0] * sol[1],
                e * z - f * sol[2],
            )
        )
        # print(val.shape)
        return val

    def _jacobian(self, sol: Array) -> Array:
        a, b, c, d, e, f = self._param
        z = 1.0 - sol[0] - sol[1] - sol[2]
        jac = self._bkd.stack(
            [
                self._bkd.hstack(
                    [-a - c - 4 * d * sol[1], -a - 4 * d * sol[0], -a]
                ),
                self._bkd.hstack(
                    [
                        -4 * d * sol[1] - 4 * b * z,
                        -4 * d * sol[0] - 4 * b * z,
                        -4 * b * z,
                    ]
                ),
                self._bkd.hstack([-e, -e, -e - f]),
            ],
            axis=0,
        )
        # auto_jac = self._bkd.jacobian(self, sol)
        # # print(jac)
        # # print(auto_jac)
        # assert self._bkd.allclose(auto_jac, jac)
        return jac

    # def _param_jacobian(self, sol: Array) -> Array:
    #     a, b, c, d, e, f = self._param
    #     z = 1.0 - sol[0] - sol[1] - sol[2]

    #     return self._bkd.stack(
    #         [
    #             self._bkd.asarray([z, 0.0, -sol[0], -4 * sol[0] * sol[1], 0.0, 0.0]),
    #             self._bkd.asarray([0.0, 2 * z**2, 0.0, -4 * sol[0] * sol[1], 0.0, 0.0]),
    #             self._bkd.asarray([0.0, 0.0, 0.0, 0.0, z, -sol[2]]),
    #         ],
    #         axis=0,
    #     )

    def nvars(self) -> int:
        return 6

    def _initial_param_jacobian(self) -> Array:
        return self._bkd.zeros((self.nstates(), self.nvars()))


class ChemicalReactionModel(TransientAdjointModel):
    """
    Model of species absorbing onto a surface out of gas phase
    # u = y[0] = monomer species
    # v = y[1] = dimer species
    # w = y[2] = inert species

    Vigil et al., Phys. Rev. E., 1996; Makeev et al., J. Chem. Phys., 2002
    Bert dubescere used this example 2014 talk
    """

    def __init__(
        self,
        time_residual_cls: TimeIntegratorNewtonResidual,
        functional: Functional = None,
        final_time: float = 100.0,
        deltat: float = 0.1,
        newton_solver: NewtonSolver = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._init_time = 0
        self._residual = ParameterizedChemicalReactionResidual(backend)
        if functional is None:
            functional = TransientSingleStateFinalTimeFunctional(
                2, self._residual.nstates(), self.nvars(), backend=backend
            )
        super().__init__(
            self._init_time,
            final_time,
            deltat,
            time_residual_cls(self._residual),
            functional,
            newton_solver,
            backend=backend,
        )
        self._jacobian_implemented = True

    def get_initial_condition(self) -> Array:
        return self._bkd.zeros(self._residual.nstates())

    def nvars(self) -> int:
        if not hasattr(self, "_functional"):
            return self._residual.nvars()
        return (
            self._functional.nunique_functional_params()
            + self._residual.nvars()
        )


class ChemicalReactionBenchmark(SingleModelBenchmark):
    def _variable_ranges(self):
        nominal_vals = self._bkd.array(
            [1.6, 20.75, 0.04, 1.0, 0.36, 0.016], 
        )
        ranges = self._bkd.empty(2 * nominal_vals.shape[0])
        ranges[:4] = [0.0, 4, 5.0, 35.0]
        ranges[4::2] = nominal_vals[2:] * 0.9
        ranges[5::2] = nominal_vals[2:] * 1.1
        return ranges

    def _set_variable(self):
        ranges = self._variable_ranges()
        marginals = [
            stats.uniform(ranges[2 * ii], ranges[2 * ii + 1] - ranges[2 * ii])
            for ii in range(len(ranges) // 2)
        ]
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_model(self):
        self._model = ChemicalReactionModel(
            BackwardEulerResidual, None, backend=self._bkd
        )


class ParameterizedLotkaVolterraResidual(
    TransientNewtonResidual, ParameterizedNewtonResidualMixin
):
    def set_time(self, time: float):
        self._time = time

    def nstates(self) -> int:
        return 3

    def set_param(self, param: Array):
        if param.shape[0] != self.nvars():
            raise ValueError("param has the wrong shape")
        self._param = param
        self._rcoefs = param[: self.nstates()]
        self._acoefs = self._bkd.reshape(
            param[self.nstates() :], (self.nstates(), self.nstates())
        )

    def __call__(self, sol: Array) -> Array:
        return self._rcoefs * sol * (1.0 - self._acoefs @ sol)

    def _jacobian(self, sol: Array) -> Array:
        return (
            self._bkd.diag(self._rcoefs)
            - self._rcoefs * self._bkd.diag(self._acoefs @ sol)
            - (self._rcoefs * sol) * self._acoefs.T
        ).T

    def _param_jacobian(self, sol: Array) -> Array:
        jac_r = self._bkd.diag(sol) - sol * self._bkd.diag(self._acoefs @ sol)
        jac_a_rows = -(self._rcoefs * sol)[:, None] * sol[None, :]
        jac_a = self._bkd.zeros((3, 9))
        for ii in range(3):
            jac_a[ii, 3 * ii : 3 * (ii + 1)] = jac_a_rows[ii]
        jac = self._bkd.hstack((jac_r, jac_a))
        return jac

    def nvars(self) -> int:
        return (self.nstates() + 1) * self.nstates()

    def _initial_param_jacobian(self) -> Array:
        return self._bkd.zeros((self.nstates(), self.nvars()))


class LotkaVolterraModel(TransientAdjointModel):
    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        time_residual_cls: TimeIntegratorNewtonResidual,
        functional: TransientAdjointFunctional = None,
        newton_solver: NewtonSolver = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._residual = ParameterizedLotkaVolterraResidual(backend)
        super().__init__(
            init_time,
            final_time,
            deltat,
            time_residual_cls(self._residual),
            functional,
            newton_solver,
            backend=backend,
        )
        self._jacobian_implemented = True

    def get_initial_condition(self) -> Array:
        return self._bkd.array([0.3, 0.4, 0.3])

    def nvars(self) -> int:
        if not hasattr(self, "_functional"):
            return self._residual.nvars()
        return (
            self._functional.nunique_functional_params()
            + self._residual.nvars()
        )


class LotkaVolterraBenchmark(SingleModelBenchmark):
    def __init__(
        self,
        time_residual_cls: TimeIntegratorNewtonResidual = BackwardEulerResidual,
        newton_solver: NewtonSolver = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._time_residual_cls = time_residual_cls
        self._newton_solver = newton_solver
        super().__init__(backend)

    def _set_variable(self):
        marginals = [stats.uniform(0.3, 0.4) for ii in range(12)]
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_model(self):
        self._model = LotkaVolterraModel(
            0,
            10,
            1,
            self._time_residual_cls,
            None,
            self._newton_solver,
            backend=self._bkd,
        )

        obs_sample = self.variable().get_statistics("mean")
        model_obs_sol, model_obs_times = self._model.forward_solve(obs_sample)
        obs_time_indices = self._bkd.arange(
            model_obs_times.shape[0], dtype=int
        )
        obs_time_tuples = [
            (0, obs_time_indices),
            (2, obs_time_indices[::2]),
        ]
        functional = TransientMSEAdjointFunctional(
            3, self._model.nvars(), obs_time_tuples, backend=self._bkd
        )
        obs = functional.observations_from_solution(model_obs_sol)
        functional.set_observations(obs)
        self._model.set_functional(functional)


class ParameterizedCoupledSpringsResidual(
    TransientNewtonResidual, ParameterizedNewtonResidualMixin
):
    def set_time(self, time: float):
        self._time = time

    def nstates(self) -> int:
        return 4

    def set_param(self, param: Array):
        if param.shape[0] != self.nvars():
            raise ValueError("param has the wrong shape")
        self._param = param

    def __call__(self, sol: Array) -> Array:
        """
        Defines the differential equations for the coupled spring-mass system.

        Arguments:
        w :  vector of the state variables:
        w = [x1, y1, x2, y2]
        t :  time
        p :  vector of the parameters:
        p = [m1, m2, k1, k2, L1, L2, b1, b2

        m1 and m2 are the masses
        k1 and k2 are the spring constants
        L1 and L2 are the natural lengths
        b1 abd b2 are the friction coefficients
        x1 and x2 are the initial displacements;
        y1 and y2 are the initial velocities
        """
        x1, y1, x2, y2 = sol
        m1, m2, k1, k2, L1, L2, b1, b2 = self._param[:8]
        return self._bkd.hstack(
            [
                y1,
                (-b1 * y1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1,
                y2,
                (-b2 * y2 - k2 * (x2 - x1 - L2)) / m2,
            ]
        )

    def _jacobian(self, sol: Array) -> Array:
        x1, y1, x2, y2 = sol
        m1, m2, k1, k2, L1, L2, b1, b2 = self._param[:8]
        jac = self._bkd.stack(
            [
                self._bkd.asarray([0.0, 1.0, 0.0, 0.0]),
                self._bkd.asarray([-k1 - k2, -b1, k2, 0.0]) / m1,
                self._bkd.asarray([0.0, 0.0, 0.0, 1.0]),
                self._bkd.asarray([k2, 0.0, -k2, -b2]) / m2,
            ],
            axis=0,
        )
        return jac

    def _param_jacobian(self, sol: Array) -> Array:
        x1, y1, x2, y2 = sol
        m1, m2, k1, k2, L1, L2, b1, b2 = self._param[:8]
        numer1 = -b1 * y1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)
        numer2 = -b2 * y2 - k2 * (x2 - x1 - L2)
        row0 = self._bkd.zeros((self.nvars(),))
        row1 = self._bkd.asarray(
            [
                -numer1 / m1**2,
                0.0,
                -(x1 - L1) / m1,
                (x2 - x1 - L2) / m1,
                k1 / m1,
                -k2 / m1,
                -y1 / m1,
                0.0,
                0.,
                0.,
                0.,
                0.
            ]
            )
        row2 = self._bkd.zeros((self.nvars(),))
        row3 = self._bkd.asarray(
            [
                0.0,
                -numer2 / m2**2,
                0.0,
                -(x2 - x1 - L2) / m2,
                0.0,
                k2 / m2,
                0.0,
                -y2 / m2,
                0.,
                0.,
                0.,
                0.,
            ]
        )
        jac = self._bkd.stack([row0, row1, row2, row3], axis=0)
        return jac

    def nvars(self) -> int:
        return 12

    def _initial_param_jacobian(self) -> Array:
        return self._bkd.hstack(
            (
                self._bkd.zeros((self.nstates(), 8)),
                -self._bkd.eye(4),
            )
        )


class CoupledSpringsModel(TransientAdjointModel):
    """
    Two objects with masses m1 and m2 are coupled through springs with
    spring constants k1 and k2. The left end of the left spring is fixed.
    We assume that the lengths of the springs, when subjected to no external
    forces, are L1 and L2.

    The masses are sliding on a surface that creates friction,
    so there are two friction coefficients, b1 and b2.
    http://www.scipy.org/Cookbook/CoupledSpringMassSystem
    """

    def __init__(
        self,
        time_residual_cls: TimeIntegratorNewtonResidual,
        functional: Functional = None,
        final_time: float = 10.0,
        deltat: float = 0.1,
        newton_solver: NewtonSolver = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._init_time = 0
        self._residual = ParameterizedCoupledSpringsResidual(backend)
        if functional is None:
            functional = TransientSingleStateFinalTimeFunctional(
                2, self._residual.nstates(), self.nvars(), backend=backend
            )
        super().__init__(
            self._init_time,
            final_time,
            deltat,
            time_residual_cls(self._residual),
            functional,
            newton_solver,
            backend=backend,
        )
        self._jacobian_implemented = True

    def get_initial_condition(self) -> Array:
        return self._residual._param[8:]

    def nvars(self) -> int:
        if not hasattr(self, "_functional"):
            return self._residual.nvars()
        return (
            self._functional.nunique_functional_params()
            + self._residual.nvars()
        )


class CoupledSpringsBenchmark(SingleModelBenchmark):
    def _variable_ranges(self) -> Array:
        return self._bkd.asarray(
            [
                0.9,
                1.1,
                1.4,
                1.6,
                7.0,
                9.0,
                39.0,
                41.0,
                0.4,
                0.6,
                0.9,
                1.1,
                0.7,
                0.9,
                0.4,
                0.6,
                0.4,
                0.6,
                -0.1,
                0.1,
                2.2,
                2.3,
                -0.1,
                0.1,
            ],
        )

    def _set_variable(self):
        ranges = self._variable_ranges()
        marginals = [
            stats.uniform(ranges[2 * ii], ranges[2 * ii + 1] - ranges[2 * ii])
            for ii in range(len(ranges) // 2)
        ]
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_model(self):
        self._model = CoupledSpringsModel(
            BackwardEulerResidual, None, backend=self._bkd
        )


def get_nondim_hastings_ecology_nominal_values():
    # return np.array([5.0, 4.1, 0.1, 2.0, 0.4, 0.01, 0.75, 0.15, 10.0],
    #                 np.double)
    return np.array([5.0, 3, 0.1, 2.0, 0.4, 0.01, 0.75, 0.15, 10.0], np.double)


def define_nondim_hastings_ecology_random_variables():
    nominal_sample = get_nondim_hastings_ecology_nominal_values()
    ranges = np.zeros((2 * len(nominal_sample)), np.double)
    ranges[::2] = nominal_sample * 0.95
    ranges[1::2] = nominal_sample * 1.05
    # ranges[:2] = 4.9, 5.1
    # ranges[12:14] = 0, 1
    # ranges[14:16] = 0, 1
    # ranges[16:18] = 5, 12
    univariate_variables = [
        stats.uniform(ranges[2 * ii], ranges[2 * ii + 1] - ranges[2 * ii])
        for ii in range(len(ranges) // 2)
    ]
    variable = IndependentMarginalsVariable(univariate_variables)
    return variable


# @njit(cache=True)
def nondim_hastings_ecology_rhs(y, t, z):
    """ """
    y1, y2, y3 = y
    a1, b1, a2, b2, d1, d2 = z[:6]
    return [
        y1 * (1 - y1) - a1 * y1 * y2 / (1 + b1 * y1),
        a1 * y1 * y2 / (1.0 + b1 * y1)
        - a2 * y2 * y3 / (1.0 + b2 * y2)
        - d1 * y2,
        a2 * y2 * y3 / (1.0 + b2 * y2) - d2 * y3,
    ]


def non_dimensionalize_hastings_ecology_variables(z):
    R0, K0, C1, C2, D1, D2, A1, A2, B1, B2 = z
    a1 = K0 * A1 / (R0 * B1)
    b1 = K0 / B1
    a2 = C2 * A2 * K0 / (C1 * R0 * B2)
    b2 = K0 / (C1 * B2)
    d1 = D1 / R0
    d2 = D2 / R0
    return np.array([a1, b1, a2, b2, d1, d2])


# @njit(cache=True)
def dim_hastings_ecology_rhs(y, t, z):
    """ """
    y1, y2, y3 = y
    nondim_z = non_dimensionalize_hastings_ecology_variables(z[:10])
    nondim_z = np.vstack((nondim_z, z[10:]))
    return nondim_hastings_ecology_rhs(y, t, z)


class HastingsEcology(object):
    """
    http://www.jstor.org/stable/1940591

    Original model

    dY1/dT = R_0 Y_1(1-Y_1/K_0)-C_1 F_1(Y_1) Y_2
    dY1/dT = F_1(Y_1) Y_2 - F_2(Y_2) Y_3 - D_1 Y_2
    dY1/dT = C_2 F_2(Y_2) Y_3 - D_2 Y_3

    F_i(U) = A_iU/(B_i+U) i=1,2

    T is time
    R0 intrinsic growth rate
    K0 carry capacity
    C1 conversion rate to prey for species y2
    C2 conversion rate to prey for species y3
    D1 constant death rate for species y2
    D2 constant death rate for species y3
    A1 saturating functional response for f1
    A2 saturating functional response for f2
    B1 prey population level where the predator rate per unit prey
       is half its maximum value for f1
    B1 prey population level where the predator rate per unit prey
       is half its maximum value for f2

    Model is non-dimensionalized
    a_1 = K_0 A_1/(R_0 B_1)
    b_1 = K_0/B_1
    a_2 = C_2 A_2 K_0/(C_1 R_0 B_2)
    b_2 = K_0/(C_1 B_2)
    d_1 = D_1/R_0
    d_2 = D_2/R_0
    """

    def __init__(self, qoi_functional=None, nondim=True, time=None):
        self.num_vars = 9
        self.opts = {"rtol": 1e-10, "atol": 1e-10}
        if time is None:
            self.t = np.linspace(0.0, 100, 101)
        else:
            self.t = time
        self.nondim = nondim
        if self.nondim:
            self.run = self.nondim_run
        else:
            self.run = self.dim_run
        if qoi_functional is None:
            self.qoi_functional = lambda sol: np.array([sol[-1, 2]])
        else:
            self.qoi_functional = qoi_functional
        self.name = "hastings-ecology-9"

    def nondim_run(self, z):
        assert z.ndim == 1
        y0 = z[6:]
        return integrate.odeint(
            nondim_hastings_ecology_rhs, y0, self.t, args=(z,), **self.opts
        )

    def dim_run(self, z):
        assert z.ndim == 1
        y0 = z[10:]
        return integrate.odeint(
            nondim_hastings_ecology_rhs, y0, self.t, args=(z,), **self.opts
        )

    def value(self, z):
        assert z.ndim == 1
        sol = self.run(z)
        return self.qoi_functional(sol)

    def __call__(self, samples):
        return evaluate_1darray_function_on_2d_array(self.value, samples, None)


class ParameterizedNonlinearModel(object):
    def __init__(self):
        self.qoi = 1
        self.ranges = np.array(
            [0.79, 0.99, 1 - 4.5 * np.sqrt(0.1), 1 + 4.5 * np.sqrt(0.1)],
            np.double,
        )

    def num_qoi(self):
        if np.isscalar(self.qoi):
            return 1
        else:
            return len(self.qoi)

    def evaluate(self, samples):
        assert samples.ndim == 1
        assert samples.ndim == 1

        sol = np.ones((2), float)

        x1 = samples[0]
        x2 = samples[1]
        u1 = sol[0]
        u2 = sol[1]

        res1 = 1.0 - (x1 * u1 * u1 + u2 * u2)
        res2 = 1.0 - (u1 * u1 - x2 * u2 * u2)

        norm_res = np.sqrt(res1 * res1 + res2 * res2)

        it = 0
        max_iters = 20
        while (norm_res > 1e-10) and (it < max_iters):
            det = -4 * u1 * u2 * (x1 * x2 + 1.0)
            j11i = -2.0 * x2 * u2 / det
            j12i = -2.0 * u2 / det
            j21i = -2.0 * u1 / det
            j22i = 2 * x1 * u1 / det

            du1 = j11i * res1 + j12i * res2
            du2 = j21i * res1 + j22i * res2

            u1 += du1
            u2 += du2

            res1 = 1.0 - (x1 * u1 * u1 + u2 * u2)
            res2 = 1.0 - (u1 * u1 - x2 * u2 * u2)

            norm_res = np.sqrt(res1 * res1 + res2 * res2)
            it += 1

        sol[0] = u1
        sol[1] = u2

        if np.isscalar(self.qoi):
            values = np.array([sol[self.qoi]])
        else:
            values = sol[self.qoi]
        return values

    def __call__(self, samples):
        num_samples = samples.shape[1]
        values = np.empty((num_samples, self.num_qoi()), float)
        for i in range(samples.shape[1]):
            values[i, :] = self.evaluate(samples[:, i])
        return values
