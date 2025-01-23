from scipy import stats

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

    def _param_jacobian(self, sol: Array) -> Array:
        a, b, c, d, e, f = self._param
        z = 1.0 - sol[0] - sol[1] - sol[2]
        zero = sol[0] * 0.  # needed to be compatable with torch hstack
        # if asarray is used instead, autograd graph will be wrong
        return self._bkd.stack(
            [
                self._bkd.hstack([z, zero, -sol[0], -4 * sol[0] * sol[1], zero, zero]),
                self._bkd.hstack([zero, 2 * z**2, zero, -4 * sol[0] * sol[1], zero, zero]),
                self._bkd.hstack([zero, zero, zero, zero, z, -sol[2]]),
            ],
            axis=0,
        )

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
        ranges[:4] = self._bkd.array([0.0, 4, 5.0, 35.0])
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
        zero = x1 * 0.  # needed to be compatable with torch hstack
        one = x1 * 0 + 1.
        # if asarray is used instead, autograd graph will be wrong
        jac = self._bkd.stack(
            [
                self._bkd.hstack([zero, one, zero, zero]),
                self._bkd.hstack([-k1 - k2, -b1, k2, zero]) / m1,
                self._bkd.hstack([zero, zero, zero, one]),
                self._bkd.hstack([k2, zero, -k2, -b2]) / m2,
            ],
            axis=0,
        )
        return jac

    def _param_jacobian(self, sol: Array) -> Array:
        x1, y1, x2, y2 = sol
        m1, m2, k1, k2, L1, L2, b1, b2 = self._param[:8]
        zero = x1 * 0.  # needed to be compatable with torch hstack
        # if asarray is used instead, autograd graph will be wrong
        numer1 = -b1 * y1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)
        numer2 = -b2 * y2 - k2 * (x2 - x1 - L2)
        row0 = self._bkd.zeros((self.nvars(),))
        row1 = self._bkd.hstack(
            [
                -numer1 / m1**2,
                zero,
                -(x1 - L1) / m1,
                (x2 - x1 - L2) / m1,
                k1 / m1,
                -k2 / m1,
                -y1 / m1,
                zero,
                zero,
                zero,
                zero,
                zero,
            ]
            )
        row2 = self._bkd.zeros((self.nvars(),))
        row3 = self._bkd.hstack(
            [
                zero,
                -numer2 / m2**2,
                zero,
                -(x2 - x1 - L2) / m2,
                zero,
                k2 / m2,
                zero,
                -y2 / m2,
                zero,
                zero,
                zero,
                zero,
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


class ParameterizedHastingsEcologyResidual(
        TransientNewtonResidual, ParameterizedNewtonResidualMixin
):
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
    def set_time(self, time: float):
        self._time = time

    def nstates(self) -> int:
        return 3

    def set_param(self, param: Array):
        if param.shape[0] != self.nvars():
            raise ValueError("param has the wrong shape")
        self._param = param

    def nvars(self) -> int:
        return 9

    def __call__(self, sol: Array) -> Array:
        y1, y2, y3 = sol
        a1, b1, a2, b2, d1, d2 = self._param[:6]
        return self._bkd.stack(
            [
                y1 * (1 - y1) - a1 * y1 * y2 / (1 + b1 * y1),
                a1 * y1 * y2 / (1.0 + b1 * y1)
                - a2 * y2 * y3 / (1.0 + b2 * y2)
                - d1 * y2,
                a2 * y2 * y3 / (1.0 + b2 * y2) - d2 * y3,
            ], axis=0
        )

    def _jacobian(self, sol: Array) -> Array:
        y1, y2, y3 = sol
        a1, b1, a2, b2, d1, d2 = self._param[:6]
        zero = y1 * 0.  # needed to be compatable with torch hstack
        # if asarray is used instead, autograd graph will be wrong
        jac = self._bkd.stack(
            [
                self._bkd.hstack(
                    [
                        1 - (a1 * y2)/(1 + b1 * y1) + y1 * (-2 + (a1*b1*y2)/(1 + b1*y1)**2),
                        -((a1*y1)/(1 + b1*y1)),
                        zero,
                    ]
                ),
                self._bkd.hstack(
                    [
                        (a1*y2)/(1. + b1*y1)**2,
                        -d1 + (a1*y1)/(1. + b1*y1) - (a2*y3)/(1. + b2 * y2) ** 2,
                        (-a2*y2)/(1. + b2 * y2),
                    ]
                ),
                self._bkd.hstack(
                    [
                        zero,
                        (1.*a2*y3)/(1. + b2 * y2)**2,
                        -d2 + (a2*y2)/(1. + b2*y2)
                    ]
                ),
            ], axis=0
        )
        return jac

    def _param_jacobian(self, sol: Array) -> Array:
        y1, y2, y3 = sol
        a1, b1, a2, b2, d1, d2 = self._param[:6]
        zero = y1 * 0.  # needed to be compatable with torch hstack
        row0 = self._bkd.hstack(
            [
                -y1 * y2 / (1 + b1 * y1),
                (a1 * y1 ** 2 * y2) / (1 + b1 * y1) ** 2,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                
            ]
        )
        row1 = self._bkd.hstack(
            [
                y1*y2/(b1*y1+1),
                -(a1 * y1 ** 2 * y2)/(1 + b1 * y1) ** 2,
                -y2*y3/(b2*y2+1),
                (a2 * y2 ** 2 * y3)/(1 + b2 * y2) ** 2,
                -y2,
                zero,
                zero,
                zero,
                zero,
            ]
        )
        row2 = self._bkd.hstack(
            [
                zero,
                zero,
                y2*y3/(b2*y2+1),
                -(a2 * y2 ** 2 * y3)/(1 + b2 * y2) ** 2,
                zero,
                -y3,
                zero,
                zero,
                zero,
            ]
        )
        return self._bkd.stack((row0, row1, row2), axis=0)

    def _initial_param_jacobian(self) -> Array:
        return self._bkd.hstack(
            (
                self._bkd.zeros((self.nstates(), 6)),
                -self._bkd.eye(3),
            )
        )


class HastingsEcologyModel(TransientAdjointModel):
    def __init__(
        self,
        time_residual_cls: TimeIntegratorNewtonResidual,
        functional: Functional = None,
        final_time: float = 100.0,
        deltat: float = 2.5,
        newton_solver: NewtonSolver = None,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._init_time = 0
        self._residual = ParameterizedHastingsEcologyResidual(backend)
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
        return self._residual._param[6:]

    def nvars(self) -> int:
        if not hasattr(self, "_functional"):
            return self._residual.nvars()
        return (
            self._functional.nunique_functional_params()
            + self._residual.nvars()
        )


class HastingsEcologyBenchmark(SingleModelBenchmark):
    def _variable_ranges(self):
        nominal_values = self._bkd.array([5.0, 3, 0.1, 2.0, 0.4, 0.01, 0.75, 0.15, 10.0])
        ranges = self._bkd.zeros((2 * len(nominal_values)))
        ranges[::2] = nominal_values * 0.95
        ranges[1::2] = nominal_values * 1.05
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
        newton_solver = NewtonSolver(verbosity=0, rtol=1e-12, atol=1e-12)
        self._model = HastingsEcologyModel(
            BackwardEulerResidual, None, newton_solver=newton_solver,
            backend=self._bkd
        )
