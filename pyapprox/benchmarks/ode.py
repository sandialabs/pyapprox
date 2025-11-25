from typing import Tuple

from scipy import stats

from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.benchmarks.base import (
    SingleModelBenchmark,
    SingleModelBayesianGoalOrientedOEDBenchmark,
)
from pyapprox.util.newton import (
    ParameterizedNewtonResidualMixin,
    NewtonSolver,
)
from pyapprox.pde.timeintegration import (
    TransientNewtonResidual,
    TransientSingleStateFinalTimeFunctional,
    Functional,
    BackwardEulerResidual,
    TransientAdjointFunctional,
    TransientMSEAdjointFunctional,
    TransientObservationFunctional,
)
from pyapprox.pde.adjoint import (
    TransientAdjointModel,
    TimeIntegratorNewtonResidual,
)


class ParameterizedLotkaVolterraResidual(
    TransientNewtonResidual, ParameterizedNewtonResidualMixin
):
    r"""
    Parameterized Lotka-Volterra residual.

    This class implements the residual equations for the Lotka-Volterra model,
    which describes predator-prey dynamics in a three-species system. The model
    supports computation of residuals, Jacobian matrices, and parameter
    Jacobians.

    The system is governed by the following differential equations:

    .. math::
        \frac{dx_i}{dt} = r_i x_i \left(1 - \sum_{j=1}^n a_{ij} x_j \right), \quad i = 1, 2, 3

    where:

    - :math:`x_i`: Population of species :math:`i`.
    - :math:`r_i`: Growth rate of species :math:`i`.
    - :math:`a_{ij}`: Interaction coefficient between species :math:`i` and :math:`j`.

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def set_time(self, time: float) -> None:
        """
        Set the current simulation time.

        Parameters
        ----------
        time : float
            Current simulation time.
        """
        self._time = time

    def nstates(self) -> int:
        """
        Return the number of states in the model.

        Returns
        -------
        nstates : int
            Number of states in the model. For this model, it is always 3.
        """
        return 3

    def set_param(self, param: Array) -> None:
        """
        Set the model parameters.

        Parameters
        ----------
        param : Array
            Array containing the model parameters.

        Raises
        ------
        ValueError
            If the shape of `param` is incorrect.
        """
        if param.shape[0] != self.nvars():
            raise ValueError("param has the wrong shape")
        self._param = param
        self._rcoefs = param[: self.nstates()]
        self._acoefs = self._bkd.reshape(
            param[self.nstates() :], (self.nstates(), self.nstates())
        )

    def __call__(self, sol: Array) -> Array:
        r"""
        Compute the residuals for the Lotka-Volterra model.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        residuals : Array
            Array of shape (nstates,) containing the residuals.

        Notes
        -----
        The solution vector is defined as:

        .. math::
            x = [x_1, x_2, x_3]

        The parameter vector is defined as:

        .. math::
            p = [r_1, r_2, r_3, a_{11}, a_{12}, a_{13}, a_{21}, a_{22}, a_{23}, a_{31}, a_{32}, a_{33}]
        """
        return self._rcoefs * sol * (1.0 - self._acoefs @ sol)

    def _jacobian(self, sol: Array) -> Array:
        """
        Compute the Jacobian of the residuals with respect to the states.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        jacobian : Array
            Array of shape (nstates, nstates) containing the Jacobian matrix.
        """
        return (
            self._bkd.diag(self._rcoefs)
            - self._rcoefs * self._bkd.diag(self._acoefs @ sol)
            - (self._rcoefs * sol) * self._acoefs.T
        ).T

    def _param_jacobian(self, sol: Array) -> Array:
        """
        Compute the Jacobian of the residuals with respect to the parameters.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        param_jacobian : Array
            Array of shape (nstates, nvars) containing the parameter Jacobian
            matrix.
        """
        jac_r = self._bkd.diag(sol) - sol * self._bkd.diag(self._acoefs @ sol)
        jac_a_rows = -(self._rcoefs * sol)[:, None] * sol[None, :]
        jac_a = self._bkd.zeros((3, 9))
        for ii in range(3):
            jac_a[ii, 3 * ii : 3 * (ii + 1)] = jac_a_rows[ii]
        jac = self._bkd.hstack((jac_r, jac_a))
        return jac

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the model.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the model.
        """
        return (self.nstates() + 1) * self.nstates()

    def _initial_param_jacobian(self) -> Array:
        """
        Compute the initial parameter Jacobian.

        Returns
        -------
        initial_param_jacobian : Array
            Array of shape (nstates, nvars) containing the initial parameter
            Jacobian matrix.
        """
        return self._bkd.zeros((self.nstates(), self.nvars()))


class LotkaVolterraModel(TransientAdjointModel):
    """
    Lotka-Volterra model.

    This class implements a transient adjoint model for simulating predator-prey
    dynamics in a three-species system based on the Lotka-Volterra equations.

    Parameters
    ----------
    init_time : float
        Initial simulation time.
    final_time : float
        Final simulation time.
    deltat : float
        Time step size.
    time_residual_cls : TimeIntegratorNewtonResidual
        Class for computing the time residuals using Newton's method.
    functional : TransientAdjointFunctional, optional
        Functional to evaluate at the final time. Default is None.
    newton_solver : NewtonSolver, optional
        Solver for Newton's method. Default is None.
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(
        self,
        init_time: float,
        final_time: float,
        deltat: float,
        backend: BackendMixin,
        time_residual_cls: TimeIntegratorNewtonResidual,
        functional: TransientAdjointFunctional = None,
        newton_solver: NewtonSolver = None,
    ):
        """
        Initialize the Lotka-Volterra model.

        Parameters
        ----------
        init_time : float
            Initial simulation time.
        final_time : float
            Final simulation time.
        deltat : float
            Time step size.
        time_residual_cls : TimeIntegratorNewtonResidual
            Class for computing the time residuals using Newton's method.
        functional : TransientAdjointFunctional, optional
            Functional to evaluate at the final time. Default is None.
        newton_solver : NewtonSolver, optional
            Solver for Newton's method. Default is None.
        backend : BackendMixin
            Backend for numerical computations.
        """
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

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def get_initial_condition(self) -> Array:
        """
        Return the initial condition for the simulation.

        Returns
        -------
        initial_condition : Array
            Array containing the initial condition for the simulation.
        """
        return self._bkd.array([0.3, 0.4, 0.3])

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the model.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the model.
        """
        if not hasattr(self, "_functional"):
            return self._residual.nvars()
        return (
            self._functional.nunique_functional_params()
            + self._residual.nvars()
        )


class LotkaVolterraBenchmark(SingleModelBenchmark):
    r"""
    Lotka-Volterra benchmark.

    This class implements a benchmark for simulating predator-prey dynamics in
    a three-species system based on the Lotka-Volterra model. The benchmark defines
    a prior distribution for the uncertain variables and sets up the Lotka-Volterra model.

    The system is governed by the following differential equations:

    .. math::
        \frac{dx_i}{dt} = r_i x_i \left(1 - \sum_{j=1}^n a_{ij} x_j \right), \quad i = 1, 2, 3

    where:

    - :math:`x_i`: Population of species :math:`i`.
    - :math:`r_i`: Growth rate of species :math:`i`.
    - :math:`a_{ij}`: Interaction coefficient between species :math:`i` and :math:`j`.

    The prior distribution for the Lotka-Volterra benchmark is defined as follows:

    .. math::    z_i \sim \mathcal{U}[0.3, 0.4], \quad i = 1, \dots, 12

    where:

    - :math:`z_i` represents the uncertain variables in the model.
    - :math:`\mathcal{U}[0.3, 0.4]` denotes a uniform distribution over the interval [0.3, 0.4].

    Parameters
    ----------
    time_residual_cls : TimeIntegratorNewtonResidual, optional
        Class for computing the time residuals using Newton's method. Default is `BackwardEulerResidual`.
    newton_solver : NewtonSolver, optional
        Solver for Newton's method. Default is None.
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(
        self,
        backend: BackendMixin,
        time_residual_cls: TimeIntegratorNewtonResidual = BackwardEulerResidual,
        newton_solver: NewtonSolver = None,
    ):
        """
        Initialize the Lotka-Volterra benchmark.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations
        time_residual_cls : TimeIntegratorNewtonResidual, optional
            Class for computing the time residuals using Newton's method.
            Default is `BackwardEulerResidual`.
        newton_solver : NewtonSolver, optional
            Solver for Newton's method. Default is None.
        """
        self._noise_std = None
        self._time_residual_cls = time_residual_cls
        self._newton_solver = newton_solver
        self._init_time = 0.0
        self._final_time, self._timestep = self._define_time()
        super().__init__(backend)

    def set_noise_std(self, noise_std: float) -> None:
        """
        Set the standard deviation of the noise in the observations.

        Parameters
        ----------
        noise_std : float
            Standard deviation of the noise.

        Returns
        -------
        None
        """
        self._noise_std = noise_std

    def _define_time(self) -> Tuple[int, int]:
        """
        Define the simulation time and timestep.

        Returns
        -------
        final_time : int
            Final simulation time.
        timestep : int
            Time step size.
        """
        return 10.0, 1.0

    def _set_prior(self) -> None:
        """
        Define the prior distribution for the uncertain variables.

        Returns
        -------
        None
        """
        marginals = [stats.uniform(0.3, 0.4) for ii in range(12)]
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _obs_time_tuples(self, ntimes: int) -> Tuple[Array, Array]:
        """
        Define observation time tuples for the model.

        Parameters
        ----------
        ntimes : int
            Number of observation times.

        Returns
        -------
        obs_time_tuples : Tuple[Array, Array]
            Observation time tuples.
        """
        obs_time_indices = self._bkd.arange(ntimes, dtype=int)
        obs_time_tuples = [
            (0, obs_time_indices),
            (2, obs_time_indices[::2]),
        ]
        return obs_time_tuples

    def _set_model(self) -> None:
        """
        Set up the Lotka-Volterra model for the benchmark.

        Returns
        -------
        None
        """
        self._model = LotkaVolterraModel(
            0,
            self._final_time,
            self._timestep,
            self._bkd,
            self._time_residual_cls,
            None,
            self._newton_solver,
        )
        nominal_sample = self.prior().mean()
        model_obs_sol, model_obs_times = self._model.forward_solve(
            nominal_sample
        )
        self._times = model_obs_times
        functional = TransientMSEAdjointFunctional(
            3,
            self._model.nvars(),
            self._obs_time_tuples(model_obs_times.shape[0]),
            self._noise_std,
            backend=self._bkd,
        )
        obs = functional.observations_from_solution(model_obs_sol)
        functional.set_observations(obs)
        self._model.set_functional(functional)

    def solution_times(self) -> Array:
        """
        Return the solution times for the simulation.

        Returns
        -------
        solution_times : Array
            Array containing the solution times.
        """
        return self._times

    def observation_times(self) -> Array:
        """
        Return the observation times for the simulation.

        Returns
        -------
        observation_times : Array
            Array containing the observation times.
        """
        obs_times = []
        for time_idx in zip(self._model._functional._obs_time_indices):
            obs_times.append(self._times[time_idx])
        return self._bkd.stack(obs_times, axis=0)


class LotkaVolterraOEDBenchmark(SingleModelBayesianGoalOrientedOEDBenchmark):
    r"""
    Lotka-Volterra Goal-Oriented Optimal Experimental Design (OED) benchmark.

    This class implements a benchmark for Bayesian goal-oriented optimal experimental
    design (OED) based on the Lotka-Volterra model. The benchmark defines prior
    distributions, observation models, and prediction models for the three-species
    predator-prey system:

    .. math::
        \frac{dx_i}{dt} = r_i x_i \left(1 - \sum_{j=1}^n a_{ij} x_j \right), \quad i = 1, 2, 3

    where:

    - :math:`x_i`: Population of species :math:`i`.
    - :math:`r_i`: Growth rate of species :math:`i`.
    - :math:`a_{ij}`: Interaction coefficient between species :math:`i` and :math:`j`.

    The prior distribution for the Lotka-Volterra benchmark is defined as follows:

    .. math::    z_i \sim \mathcal{U}[0.3, 0.4], \quad i = 1, \dots, 12

    where:

    - :math:`z_i` represents the uncertain variables in the model.
    - :math:`\mathcal{U}[0.3, 0.4]` denotes a uniform distribution over the interval [0.3, 0.4].

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(self, backend: BackendMixin):
        """
        Initialize the Lotka-Volterra OED benchmark.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations.
        """
        self._time_residual_cls = BackwardEulerResidual
        self._newton_solver = None
        self._init_time = 0.0
        self._final_time, self._timestep = self._define_time()
        super().__init__(backend)

    def _set_prior(self) -> None:
        """
        Define the prior distribution for the uncertain variables.

        The prior distribution is uniform over the interval [0.3, 0.4] for each
        of the 12 uncertain variables.

        Returns
        -------
        None
        """
        marginals = [stats.uniform(0.3, 0.4) for ii in range(12)]
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _define_time(self) -> Tuple[int, int]:
        """
        Define the simulation time and timestep.

        Returns
        -------
        final_time : int
            Final simulation time.
        timestep : int
            Time step size.
        """
        return 50.0, 2.0

    def _obs_time_tuples(self, ntimes: int) -> Tuple[Array, Array]:
        """
        Define observation time tuples for the observation model.

        Parameters
        ----------
        ntimes : int
            Number of observation times.

        Returns
        -------
        obs_time_tuples : Tuple[Array, Array]
            Observation time tuples.
        """
        obs_time_indices = self._bkd.arange(ntimes, dtype=int)
        obs_time_tuples = [
            (0, obs_time_indices),
            (2, obs_time_indices),
        ]
        return obs_time_tuples

    def _pred_time_tuples(self, ntimes: int) -> Tuple[Array, Array]:
        """
        Define prediction time tuples for the prediction model.

        Parameters
        ----------
        ntimes : int
            Number of prediction times.

        Returns
        -------
        pred_time_tuples : Tuple[Array, Array]
            Prediction time tuples.
        """
        pred_time_indices = self._bkd.arange(ntimes, dtype=int)[::2]
        pred_time_tuples = [(1, pred_time_indices)]
        return pred_time_tuples

    def solution_times(self) -> Array:
        """
        Return the solution times for the simulation.

        Returns
        -------
        solution_times : Array
            Array containing the solution times.
        """
        return self._times

    def observation_times(self) -> Array:
        """
        Return the observation times for the simulation.

        Returns
        -------
        observation_times : Array
            Array containing the observation times.
        """
        obs_times = []
        for time_idx in zip(self._obs_model._functional._obs_time_indices):
            obs_times.append(self._times[time_idx])
        return self._bkd.stack(obs_times, axis=0)

    def prediction_times(self) -> Array:
        """
        Return the prediction times for the simulation.

        Returns
        -------
        prediction_times : Array
            Array containing the prediction times.
        """
        pred_times = []
        for time_idx in zip(self._pred_model._functional._obs_time_indices):
            pred_times.append(self._times[time_idx])
        return self._bkd.stack(pred_times, axis=0)

    def _set_obs_model(self) -> None:
        """
        Set up the observation model for the benchmark.
        """
        self._obs_model = LotkaVolterraModel(
            0,
            self._final_time,
            self._timestep,
            self._bkd,
            self._time_residual_cls,
            None,
            self._newton_solver,
        )
        self._times = self._bkd.linspace(
            0,
            self._final_time,
            int(self._final_time / self._timestep) + 1,
        )
        obs_functional = TransientObservationFunctional(
            3,
            self._obs_model.nvars(),
            self._obs_time_tuples(self._times.shape[0]),
            backend=self._bkd,
        )
        self._obs_model.set_functional(obs_functional)

    def _set_pred_model(self) -> None:
        """
        Set up the prediction model for the benchmark.
        """
        self._pred_model = LotkaVolterraModel(
            0,
            self._final_time,
            self._timestep,
            self._bkd,
            self._time_residual_cls,
            None,
            self._newton_solver,
        )
        pred_functional = TransientObservationFunctional(
            3,
            self._pred_model.nvars(),
            self._pred_time_tuples(self._times.shape[0]),
            backend=self._bkd,
        )
        self._pred_model.set_functional(pred_functional)

    def prediction_model(self) -> LotkaVolterraModel:
        """
        Return the prediction model for the benchmark.

        Returns
        -------
        prediction_model : LotkaVolterraModel
            The prediction model for the benchmark.
        """
        return self._pred_model


class ParameterizedCoupledSpringsResidual(
    TransientNewtonResidual, ParameterizedNewtonResidualMixin
):
    r"""
    Parameterized coupled springs residual.

    This class implements the residual equations for a coupled spring-mass system
    involving two masses connected by springs. The model supports computation of
    residuals, Jacobian matrices, and parameter Jacobians.

    The system is governed by the following differential equations:

    .. math::

        x'_1 = y_1 \\
        y'_1 = \frac{-b_1 y_1 - k_1 (x_1 - L_1) + k_2 (x_2 - x_1 - L_2)}{m_1} \\
        x'_2 = y_2 \\
        y'_2 = \frac{-b_2 y_2 - k_2 (x_2 - x_1 - L_2)}{m_2}

    where:

    - :math:`x_1, x_2`: Positions of the masses.
    - :math:`y_1, y_2`: Velocities of the masses.
    - :math:`m_1, m_2`: Masses of the objects.
    - :math:`k_1, k_2`: Spring constants.
    - :math:`L_1, L_2`: Natural lengths of the springs.
    - :math:`b_1, b_2`: Friction coefficients.

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def set_time(self, time: float) -> None:
        """
        Set the current simulation time.

        Parameters
        ----------
        time : float
            Current simulation time.
        """
        self._time = time

    def nstates(self) -> int:
        """
        Return the number of states in the model.

        Returns
        -------
        nstates : int
            Number of states in the model. For this model, it is always 4.
        """
        return 4

    def set_param(self, param: Array) -> None:
        """
        Set the model parameters.

        Parameters
        ----------
        param : Array
            Array containing the model parameters.

        Raises
        ------
        ValueError
            If the shape of `param` is incorrect.
        """
        if param.shape[0] != self.nvars():
            raise ValueError("param has the wrong shape")
        self._param = param

    def __call__(self, sol: Array) -> Array:
        """
        Compute the residuals for the coupled spring-mass system.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        residuals : Array
            Array of shape (nstates,) containing the residuals.

        Notes
        -----
        The solution vector is defined as:

        .. math::
            w = [x_1, y_1, x_2, y_2]

        The parameter vector is defined as:

        .. math::
            p = [m_1, m_2, k_1, k_2, L_1, L_2, b_1, b_2]
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
        """
        Compute the Jacobian of the residuals with respect to the states.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        jacobian : Array
            Array of shape (nstates, nstates) containing the Jacobian matrix.
        """
        x1, y1, x2, y2 = sol
        m1, m2, k1, k2, L1, L2, b1, b2 = self._param[:8]
        zero = x1 * 0.0  # Needed for compatibility with torch hstack
        one = x1 * 0 + 1.0
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
        """
        Compute the Jacobian of the residuals with respect to the parameters.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        param_jacobian : Array
            Array of shape (nstates, nvars) containing the parameter Jacobian matrix.
        """
        x1, y1, x2, y2 = sol
        m1, m2, k1, k2, L1, L2, b1, b2 = self._param[:8]
        zero = x1 * 0.0  # Needed for compatibility with torch hstack
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
        """
        Return the number of uncertain variables in the model.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the model. For this model, it is always 12.
        """
        return 12

    def _initial_param_jacobian(self) -> Array:
        """
        Compute the initial parameter Jacobian.

        Returns
        -------
        initial_param_jacobian : Array
            Array of shape (nstates, nvars) containing the initial parameter Jacobian matrix.
        """
        return self._bkd.hstack(
            (
                self._bkd.zeros((self.nstates(), 8)),
                -self._bkd.eye(4),
            )
        )


class CoupledSpringsModel(TransientAdjointModel):
    """
    Coupled springs model.

    This class implements a transient adjoint model for simulating the dynamics
    of two masses coupled through springs. The left end of the left spring is fixed,
    and the masses slide on a surface with friction. The model uses a time integrator
    for solving the residual equations and supports functional evaluation at the final time.

    Parameters
    ----------
    time_residual_cls : TimeIntegratorNewtonResidual
        Class for computing the time residuals using Newton's method.
    functional : Functional, optional
        Functional to evaluate at the final time. If None, a default functional
        is used. Default is None.
    final_time : float, optional
        Final simulation time. Default is 10.0.
    deltat : float, optional
        Time step size. Default is 0.1.
    newton_solver : NewtonSolver, optional
        Solver for Newton's method. Default is None.
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.

    Notes
    -----
    The system consists of:
    - Two masses :math:`m_1` and :math:`m_2`.
    - Two springs with spring constants :math:`k_1` and :math:`k_2`.
    - Spring lengths :math:`L_1` and :math:`L_2` when subjected to no external forces.
    - Friction coefficients :math:`b_1` and :math:`b_2`.

    The dynamics of the system are governed by the coupled spring-mass equations,
    which account for spring forces and frictional forces.

    Methods
    -------
    jacobian_implemented()
        Check if the Jacobian is implemented.
    get_initial_condition()
        Return the initial condition for the simulation.
    nvars()
        Return the number of uncertain variables in the model.
    """

    def __init__(
        self,
        time_residual_cls: TimeIntegratorNewtonResidual,
        functional: Functional = None,
        final_time: float = 10.0,
        deltat: float = 0.1,
        newton_solver: NewtonSolver = None,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the coupled springs model.

        Parameters
        ----------
        time_residual_cls : TimeIntegratorNewtonResidual
            Class for computing the time residuals using Newton's method.
        functional : Functional, optional
            Functional to evaluate at the final time. If None, a default functional
            is used. Default is None.
        final_time : float, optional
            Final simulation time. Default is 10.0.
        deltat : float, optional
            Time step size. Default is 0.1.
        newton_solver : NewtonSolver, optional
            Solver for Newton's method. Default is None.
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
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

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def get_initial_condition(self) -> Array:
        """
        Return the initial condition for the simulation.

        Returns
        -------
        initial_condition : Array
            Array containing the initial condition for the simulation.
        """
        return self._residual._param[8:]

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the model.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the model.
        """
        if not hasattr(self, "_functional"):
            return self._residual.nvars()
        return (
            self._functional.nunique_functional_params()
            + self._residual.nvars()
        )


class CoupledSpringsBenchmark(SingleModelBenchmark):
    r"""
    Coupled springs benchmark.

    This class implements a benchmark for simulating the dynamics of two masses
    coupled through springs. The benchmark defines a prior distribution for the
    uncertain variables and sets up the coupled springs model.

    The Coupled ODEs are:

    .. math::

    x'_1 = y_1

    .. math::

        y'_1 = \frac{-b_1 y_1 - k_1 (x_1 - L_1) + k_2 (x_2 - x_1 - L_2)}{m_1}

    .. math::

        x'_2 = y_2

    .. math::

        y'_2 = \frac{-b_2 y_2 - k_2 (x_2 - x_1 - L_2)}{m_2}

    The prior distribution is defined as uniform distributions over the ranges:

    .. math::
        z_i \sim \mathcal{U}[z_i^\text{min}, z_i^\text{max}]

    where :math:`z_i^\text{min}` and :math:`z_i^\text{max}` are the lower and upper
    bounds for each uncertain variable.

    """

    def _prior_ranges(self) -> Array:
        """
        Compute the ranges for the prior distribution.

        The ranges are predefined for each uncertain variable.

        Returns
        -------
        ranges : Array
            Array of shape (2 * nvars,) containing the lower and upper bounds
            for each uncertain variable.
        """
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

    def _set_prior(self) -> None:
        """
        Define the prior distribution for the uncertain variables.

        The prior distribution is uniform over the ranges computed by `_prior_ranges`.
        """
        ranges = self._prior_ranges()
        marginals = [
            stats.uniform(ranges[2 * ii], ranges[2 * ii + 1] - ranges[2 * ii])
            for ii in range(len(ranges) // 2)
        ]
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_model(self) -> None:
        """
        Set up the coupled springs model for the benchmark.

        Returns
        -------
        None
        """
        self._model = CoupledSpringsModel(
            BackwardEulerResidual, None, backend=self._bkd
        )


class ParameterizedHastingsEcologyResidual(
    TransientNewtonResidual, ParameterizedNewtonResidualMixin
):
    r"""
    Parameterized Hastings ecology residual.

    This class implements the residual equations for the Hastings ecology model,
    which simulates ecological dynamics involving three species. The model is
    parameterized and supports computation of the residuals, Jacobian, and parameter
    Jacobian.

    The original model is defined as:

    .. math::
        \frac{dY_1}{dT} = R_0 Y_1(1 - Y_1 / K_0) - C_1 F_1(Y_1) Y_2 \\
        \frac{dY_2}{dT} = F_1(Y_1) Y_2 - F_2(Y_2) Y_3 - D_1 Y_2 \\
        \frac{dY_3}{dT} = C_2 F_2(Y_2) Y_3 - D_2 Y_3

    where:

    .. math::
        F_i(U) = \frac{A_i U}{B_i + U}, \quad i = 1, 2

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def set_time(self, time: float) -> Array:
        """
        Set the current simulation time.

        Parameters
        ----------
        time : float
            Current simulation time.
        """
        self._time = time

    def nstates(self) -> int:
        """
        Return the number of states in the model.

        Returns
        -------
        nstates : int
            Number of states in the model. For this model, it is always 3.
        """
        return 3

    def set_param(self, param: Array) -> None:
        """
        Set the model parameters.

        Parameters
        ----------
        param : Array
            Array containing the model parameters.

        Raises
        ------
        ValueError
            If the shape of `param` is incorrect.
        """
        if param.shape[0] != self.nvars():
            raise ValueError("param has the wrong shape")
        self._param = param

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the model.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the model. For this model, it is always 9.
        """
        return 9

    def __call__(self, sol: Array) -> Array:
        """
        Compute the residuals for the given solution.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        residuals : Array
            Array of shape (nstates,) containing the residuals.
        """
        y1, y2, y3 = sol
        a1, b1, a2, b2, d1, d2 = self._param[:6]
        return self._bkd.stack(
            [
                y1 * (1 - y1) - a1 * y1 * y2 / (1 + b1 * y1),
                a1 * y1 * y2 / (1.0 + b1 * y1)
                - a2 * y2 * y3 / (1.0 + b2 * y2)
                - d1 * y2,
                a2 * y2 * y3 / (1.0 + b2 * y2) - d2 * y3,
            ],
            axis=0,
        )

    def _jacobian(self, sol: Array) -> Array:
        """
        Compute the Jacobian of the residuals with respect to the states.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        jacobian : Array
            Array of shape (nstates, nstates) containing the Jacobian matrix.
        """
        y1, y2, y3 = sol
        a1, b1, a2, b2, d1, d2 = self._param[:6]
        zero = y1 * 0.0  # Needed for compatibility with torch hstack
        jac = self._bkd.stack(
            [
                self._bkd.hstack(
                    [
                        1
                        - (a1 * y2) / (1 + b1 * y1)
                        + y1 * (-2 + (a1 * b1 * y2) / (1 + b1 * y1) ** 2),
                        -((a1 * y1) / (1 + b1 * y1)),
                        zero,
                    ]
                ),
                self._bkd.hstack(
                    [
                        (a1 * y2) / (1.0 + b1 * y1) ** 2,
                        -d1
                        + (a1 * y1) / (1.0 + b1 * y1)
                        - (a2 * y3) / (1.0 + b2 * y2) ** 2,
                        (-a2 * y2) / (1.0 + b2 * y2),
                    ]
                ),
                self._bkd.hstack(
                    [
                        zero,
                        (1.0 * a2 * y3) / (1.0 + b2 * y2) ** 2,
                        -d2 + (a2 * y2) / (1.0 + b2 * y2),
                    ]
                ),
            ],
            axis=0,
        )
        return jac

    def _param_jacobian(self, sol: Array) -> Array:
        """
        Compute the Jacobian of the residuals with respect to the parameters.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        param_jacobian : Array
            Array of shape (nstates, nvars) containing the parameter Jacobian matrix.
        """
        y1, y2, y3 = sol
        a1, b1, a2, b2, d1, d2 = self._param[:6]
        zero = y1 * 0.0  # Needed for compatibility with torch hstack
        row0 = self._bkd.hstack(
            [
                -y1 * y2 / (1 + b1 * y1),
                (a1 * y1**2 * y2) / (1 + b1 * y1) ** 2,
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
                y1 * y2 / (b1 * y1 + 1),
                -(a1 * y1**2 * y2) / (1 + b1 * y1) ** 2,
                -y2 * y3 / (b2 * y2 + 1),
                (a2 * y2**2 * y3) / (1 + b2 * y2) ** 2,
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
                y2 * y3 / (b2 * y2 + 1),
                -(a2 * y2**2 * y3) / (1 + b2 * y2) ** 2,
                zero,
                -y3,
                zero,
                zero,
                zero,
            ]
        )
        return self._bkd.stack((row0, row1, row2), axis=0)

    def _initial_param_jacobian(self) -> Array:
        """
        Compute the initial parameter Jacobian.

        Returns
        -------
        initial_param_jacobian : Array
            Array of shape (nstates, nvars) containing the initial parameter Jacobian matrix.
        """
        return self._bkd.hstack(
            (
                self._bkd.zeros((self.nstates(), 6)),
                -self._bkd.eye(3),
            )
        )


class HastingsEcologyModel(TransientAdjointModel):
    r"""
    Hastings ecology model.

    This class implements a transient adjoint model for simulating ecological dynamics
    based on the Hastings model. The model uses a time integrator for solving the
    residual equations and supports functional evaluation at the final time.

    The original model is defined as:

    .. math::
        \frac{dY_1}{dT} = R_0 Y_1(1 - Y_1 / K_0) - C_1 F_1(Y_1) Y_2 \\
        \frac{dY_2}{dT} = F_1(Y_1) Y_2 - F_2(Y_2) Y_3 - D_1 Y_2 \\
        \frac{dY_3}{dT} = C_2 F_2(Y_2) Y_3 - D_2 Y_3

    where:

    .. math::
        F_i(U) = \frac{A_i U}{B_i + U}, \quad i = 1, 2

    Parameters
    ----------
    time_residual_cls : TimeIntegratorNewtonResidual
        Class for computing the time residuals using Newton's method.
    functional : Functional, optional
        Functional to evaluate at the final time. If None, a default functional
        is used. Default is None.
    final_time : float, optional
        Final simulation time. Default is 100.0.
    deltat : float, optional
        Time step size. Default is 2.5.
    newton_solver : NewtonSolver, optional
        Solver for Newton's method. Default is None.
    backend : BackendMixin
        Backend for numerical computations.
    """

    def __init__(
        self,
        time_residual_cls: TimeIntegratorNewtonResidual,
        backend: BackendMixin,
        functional: Functional = None,
        final_time: float = 100.0,
        deltat: float = 2.5,
        newton_solver: NewtonSolver = None,
    ):
        """
        Initialize the Hastings ecology model.

        Parameters
        ----------
        time_residual_cls : TimeIntegratorNewtonResidual
            Class for computing the time residuals using Newton's method.
        backend : BackendMixin
            Backend for numerical computations`.
        functional : Functional, optional
            Functional to evaluate at the final time. If None, a default functional
            is used. Default is None.
        final_time : float, optional
            Final simulation time. Default is 100.0.
        deltat : float, optional
            Time step size. Default is 2.5.
        newton_solver : NewtonSolver, optional
            Solver for Newton's method. Default is None.
        """
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

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def get_initial_condition(self) -> Array:
        """
        Return the initial condition for the simulation.

        Returns
        -------
        initial_condition : Array
            Array containing the initial condition for the simulation.
        """
        return self._residual._param[6:]

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the model.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the model.
        """
        if not hasattr(self, "_functional"):
            return self._residual.nvars()
        return (
            self._functional.nunique_functional_params()
            + self._residual.nvars()
        )


class HastingsEcologyBenchmark(SingleModelBenchmark):
    r"""
    Hastings ecology benchmark.

    This class implements a benchmark for simulating ecological dynamics based
    on the Hastings model. The benchmark defines a prior distribution for the
    uncertain variables and sets up the Hastings ecology model.

    The original model is defined as:

    .. math::
        \frac{dY_1}{dT} = R_0 Y_1(1 - Y_1 / K_0) - C_1 F_1(Y_1) Y_2 \\
        \frac{dY_2}{dT} = F_1(Y_1) Y_2 - F_2(Y_2) Y_3 - D_1 Y_2 \\
        \frac{dY_3}{dT} = C_2 F_2(Y_2) Y_3 - D_2 Y_3

    where:

    .. math::
        F_i(U) = \frac{A_i U}{B_i + U}, \quad i = 1, 2

    Notes
    -----
    The parameters of the model are defined as follows:

    - :math:`T`: Time.
    - :math:`R_0`: Intrinsic growth rate.
    - :math:`K_0`: Carrying capacity.
    - :math:`C_1`: Conversion rate to prey for species :math:`Y_2`.
    - :math:`C_2`: Conversion rate to prey for species :math:`Y_3`.
    - :math:`D_1`: Constant death rate for species :math:`Y_2`.
    - :math:`D_2`: Constant death rate for species :math:`Y_3`.
    - :math:`A_1`: Saturating functional response for :math:`F_1`.
    - :math:`A_2`: Saturating functional response for :math:`F_2`.
    - :math:`B_1`: Prey population level where the predator rate per unit prey
      is half its maximum value for :math:`F_1`.
    - :math:`B_2`: Prey population level where the predator rate per unit prey
      is half its maximum value for :math:`F_2`.

    The model is non-dimensionalized as follows:

    .. math::
        a_1 = \frac{K_0 A_1}{R_0 B_1}, \quad b_1 = \frac{K_0}{B_1}, \\
        a_2 = \frac{C_2 A_2 K_0}{C_1 R_0 B_2}, \quad b_2 = \frac{K_0}{C_1 B_2}, \\
        d_1 = \frac{D_1}{R_0}, \quad d_2 = \frac{D_2}{R_0}.


    The prior distribution is defined as:

    .. math::
        z_i \sim \mathcal{U}[0.95 \cdot z_i^\text{nominal}, 1.05 \cdot z_i^\text{nominal}]

    where :math:`z_i^\text{nominal}` are the nominal values of the uncertain variables.

    References
    ----------
    .. [Hastings1991] `Hastings, Alan, and Thomas Powell. "Chaos in a Three-Species Food Chain." Ecology 72, no. 3 (1991): 896–903. <https://doi.org/10.2307/1940591>`_
    """

    def _prior_ranges(self) -> Array:
        """
        Compute the ranges for the prior distribution.

        The ranges are computed as 95% to 105% of the nominal values.

        Returns
        -------
        ranges : Array
            Array of shape (2 * nvars,) containing the lower and upper bounds
            for each uncertain variable.
        """
        nominal_values = self._bkd.array(
            [5.0, 3, 0.1, 2.0, 0.4, 0.01, 0.75, 0.15, 10.0]
        )
        ranges = self._bkd.zeros((2 * len(nominal_values)))
        ranges[::2] = nominal_values * 0.95
        ranges[1::2] = nominal_values * 1.05
        return ranges

    def _set_prior(self) -> None:
        """
        Define the prior distribution for the uncertain variables.

        The prior distribution is uniform over the ranges computed by `_prior_ranges`.
        """
        ranges = self._prior_ranges()
        marginals = [
            stats.uniform(ranges[2 * ii], ranges[2 * ii + 1] - ranges[2 * ii])
            for ii in range(len(ranges) // 2)
        ]
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_model(self) -> None:
        """
        Set up the Hastings ecology model for the benchmark.
        """
        newton_solver = NewtonSolver(verbosity=0, rtol=1e-12, atol=1e-12)
        self._model = HastingsEcologyModel(
            BackwardEulerResidual, self._bkd, None, newton_solver=newton_solver
        )


class ParameterizedChemicalReactionResidual(
    TransientNewtonResidual, ParameterizedNewtonResidualMixin
):
    r"""
    Parameterized chemical reaction residual.

    This class implements the residual equations for a chemical reaction model
    describing species absorbing onto a surface out of the gas phase. The model
    supports computation of residuals, Jacobian matrices, and parameter Jacobians.

    The system is governed by the following differential equations:

    .. math::
        \frac{du}{dt} = a z - c u - 4 d u v \\
        \frac{dv}{dt} = 2 b z^2 - 4 d u v \\
        \frac{dw}{dt} = e z - f w

    where:
 
    - :math:`u`: Monomer species.
    - :math:`v`: Dimer species.
    - :math:`w`: Inert species.
    - :math:`z = 1 - u - v - w`: Fraction of unoccupied surface.
    - :math:`a, b, c, d, e, f`: Model parameters.

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.
    """

    def set_time(self, time: float) -> None:
        """
        Set the current simulation time.

        Parameters
        ----------
        time : float
            Current simulation time.
        """
        self._time = time

    def nstates(self) -> int:
        """
        Return the number of states in the model.

        Returns
        -------
        nstates : int
            Number of states in the model. For this model, it is always 3.
        """
        return 3

    def set_param(self, param: Array) -> None:
        """
        Set the model parameters.

        Parameters
        ----------
        param : Array
            Array containing the model parameters.

        Raises
        ------
        ValueError
            If the shape of `param` is incorrect.
        """
        if param.shape[0] != self.nvars():
            raise ValueError("param has the wrong shape")
        self._param = param

    def __call__(self, sol: Array) -> Array:
        """
        Compute the residuals for the chemical reaction model.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        residuals : Array
            Array of shape (nstates,) containing the residuals.

        Notes
        -----
        The solution vector is defined as:

        .. math::
            w = [u, v, w]

        The parameter vector is defined as:

        .. math::
            p = [a, b, c, d, e, f]
        """
        a, b, c, d, e, f = self._param
        z = 1.0 - sol[0] - sol[1] - sol[2]
        val = self._bkd.hstack(
            (
                a * z - c * sol[0] - 4 * d * sol[0] * sol[1],
                2 * b * z**2 - 4 * d * sol[0] * sol[1],
                e * z - f * sol[2],
            )
        )
        return val

    def _jacobian(self, sol: Array) -> Array:
        """
        Compute the Jacobian of the residuals with respect to the states.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        jacobian : Array
            Array of shape (nstates, nstates) containing the Jacobian matrix.
        """
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
        return jac

    def _param_jacobian(self, sol: Array) -> Array:
        """
        Compute the Jacobian of the residuals with respect to the parameters.

        Parameters
        ----------
        sol : Array
            Array of shape (nstates,) containing the solution.

        Returns
        -------
        param_jacobian : Array
            Array of shape (nstates, nvars) containing the parameter Jacobian matrix.
        """
        a, b, c, d, e, f = self._param
        z = 1.0 - sol[0] - sol[1] - sol[2]
        zero = sol[0] * 0.0  # Needed for compatibility with torch hstack
        return self._bkd.stack(
            [
                self._bkd.hstack(
                    [z, zero, -sol[0], -4 * sol[0] * sol[1], zero, zero]
                ),
                self._bkd.hstack(
                    [zero, 2 * z**2, zero, -4 * sol[0] * sol[1], zero, zero]
                ),
                self._bkd.hstack([zero, zero, zero, zero, z, -sol[2]]),
            ],
            axis=0,
        )

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the model.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the model. For this model, it is always 6.
        """
        return 6

    def _initial_param_jacobian(self) -> Array:
        """
        Compute the initial parameter Jacobian.

        Returns
        -------
        initial_param_jacobian : Array
            Array of shape (nstates, nvars) containing the initial parameter Jacobian matrix.
        """
        return self._bkd.zeros((self.nstates(), self.nvars()))


class ChemicalReactionModel(TransientAdjointModel):
    r"""
    Chemical reaction model.

    This class implements a transient adjoint model for simulating species absorbing
    onto a surface out of the gas phase. The model describes the dynamics of three
    species: monomer, dimer, and inert species.

    Species:
    - :math:`u`: Monomer species y[0]
    - :math:`v`: Dimer species y[1]
    - :math:`w`: Inert species y[2]

    References
    ----------
    - Vigil et al., Phys. Rev. E., 1996.
    - Makeev et al., J. Chem. Phys., 2002.
    - Bert Dubescere, 2014 talk.

    Parameters
    ----------
    time_residual_cls : TimeIntegratorNewtonResidual
        Class for computing the time residuals using Newton's method.
    functional : Functional, optional
        Functional to evaluate at the final time. If None, a default functional
        is used. Default is None.
    final_time : float, optional
        Final simulation time. Default is 100.0.
    deltat : float, optional
        Time step size. Default is 0.1.
    newton_solver : NewtonSolver, optional
        Solver for Newton's method. Default is None.
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(
        self,
        time_residual_cls: TimeIntegratorNewtonResidual,
        functional: Functional = None,
        final_time: float = 100.0,
        deltat: float = 0.1,
        newton_solver: NewtonSolver = None,
        backend: BackendMixin = NumpyMixin,
    ):
        """
        Initialize the chemical reaction model.

        Parameters
        ----------
        time_residual_cls : TimeIntegratorNewtonResidual
            Class for computing the time residuals using Newton's method.
        functional : Functional, optional
            Functional to evaluate at the final time. If None, a default functional
            is used. Default is None.
        final_time : float, optional
            Final simulation time. Default is 100.0.
        deltat : float, optional
            Time step size. Default is 0.1.
        newton_solver : NewtonSolver, optional
            Solver for Newton's method. Default is None.
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
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

    def jacobian_implemented(self) -> bool:
        """
        Check if the Jacobian is implemented.

        Returns
        -------
        jacobian_implemented : bool
            True if the Jacobian is implemented, False otherwise.
        """
        return True

    def get_initial_condition(self) -> Array:
        """
        Return the initial condition for the simulation.

        Returns
        -------
        initial_condition : Array
            Array containing the initial condition for the simulation.
        """
        return self._bkd.zeros(self._residual.nstates())

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the model.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the model.
        """
        if not hasattr(self, "_functional"):
            return self._residual.nvars()
        return (
            self._functional.nunique_functional_params()
            + self._residual.nvars()
        )


class ChemicalReactionBenchmark(SingleModelBenchmark):
    r"""
    Chemical reaction benchmark.

    This class implements a benchmark for simulating species absorbing onto a surface
    out of the gas phase. The benchmark defines a prior distribution for the uncertain
    variables and sets up the chemical reaction model.

    The system is governed by the following differential equations:

    .. math::
        \frac{du}{dt} = a z - c u - 4 d u v \\
        \frac{dv}{dt} = 2 b z^2 - 4 d u v \\
        \frac{dw}{dt} = e z - f w

    where:
    
    - :math:`u`: Monomer species.
    - :math:`v`: Dimer species.
    - :math:`w`: Inert species.
    - :math:`z = 1 - u - v - w`: Fraction of unoccupied surface.
    - :math:`a, b, c, d, e, f`: Model parameters.

    The prior distribution is defined as uniform distributions over the ranges:

    .. math::
        z_i \sim \mathcal{U}[z_i^\text{min}, z_i^\text{max}]

    where :math:`z_i^\text{min}` and :math:`z_i^\text{max}` are the lower and upper
    bounds for each uncertain variable.
    """

    def _prior_ranges(self) -> Array:
        """
        Compute the ranges for the prior distribution.

        The ranges are predefined for each uncertain variable.

        Returns
        -------
        ranges : Array
            Array of shape (2 * nvars,) containing the lower and upper bounds
            for each uncertain variable.
        """
        nominal_vals = self._bkd.array(
            [1.6, 20.75, 0.04, 1.0, 0.36, 0.016],
        )
        ranges = self._bkd.empty(2 * nominal_vals.shape[0])
        ranges[:4] = self._bkd.array([0.0, 4, 5.0, 35.0])
        ranges[4::2] = nominal_vals[2:] * 0.9
        ranges[5::2] = nominal_vals[2:] * 1.1
        return ranges

    def _set_prior(self) -> None:
        """
        Define the prior distribution for the uncertain variables.

        The prior distribution is uniform over the ranges computed by `_prior_ranges`.
        """
        ranges = self._prior_ranges()
        marginals = [
            stats.uniform(ranges[2 * ii], ranges[2 * ii + 1] - ranges[2 * ii])
            for ii in range(len(ranges) // 2)
        ]
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_model(self) -> None:
        """
        Set up the chemical reaction model for the benchmark.
        """
        self._model = ChemicalReactionModel(
            BackwardEulerResidual, None, backend=self._bkd
        )
