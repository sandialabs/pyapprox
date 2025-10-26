import math
import itertools
from typing import List, Tuple

import numpy as np
from scipy import stats

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.util.backends.template import BackendMixin, Array
from pyapprox.benchmarks.base import (
    SingleModelBayesianInferenceBenchmark,
    OperatorBenchmark,
    SingleModelBenchmark,
    SingleModelBayesianGoalOrientedOEDBenchmark,
)
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.interface.wrappers import ChangeModelSignWrapper
from pyapprox.interface.model import Model
from pyapprox.pde.adjoint import (
    AdjointFunctional,
    TransientAdjointFunctional,
)
from pyapprox.pde.collocation.parameterized_pdes import (
    PyApproxPaperAdvectionDiffusionKLEInversionModel,
    PyApproxPaperAdvectionDiffusionKLEForwardModel,
    ScalarFunction,
    TransientViscousBurgers1DModel,
    SteadyDarcy2DKLEModel,
    TransientSolutionTimeSnapshotFunctional,
    SteadySolutionFunctional,
    SteadySingleStateFunctional,
    NonlinearSystemOfEquationsModel,
)
from pyapprox.pde.collocation.functions import (
    CollocationSubdomainFunction,
    ZeroScalarFunction,
    ScalarKLEFunctionOnDifferentMesh,
    ScalarKLEFunction,
    ConstantScalarFunction,
)
from pyapprox.pde.collocation.mesh import ChebyshevCollocationMesh2D
from pyapprox.pde.collocation.mesh_transforms import (
    ScaleAndTranslationTransform2D,
)
from pyapprox.pde.collocation.basis import ChebyshevCollocationBasis2D
from pyapprox.util.newton import NewtonSolver
from pyapprox.pde.timeintegration import (
    CrankNicholsonResidual,
    BackwardEulerResidual,
    TransientObservationFunctional,
)
from pyapprox.inference.likelihood import LogLikelihoodFromModel, LogLikelihood
from pyapprox.pde.galerkin.parameterized import (
    ObstructedAdvectionDiffusion,
    KLEHyperParameters,
    FETransientOutputModel,
    FETransientSubdomainIntegralFunctional,
)


class PointwiseObservationFunctional(AdjointFunctional):
    # Note this is not truly an Adjoint Functional because nqoi > 1
    # However, it allows for easier modification of adjoint models
    # that return solutions at a set of points
    def __init__(
        self,
        nstates: int,
        nresidual_params: int,
        obs_state_indices: List[Tuple[int, Array]],
        backend: BackendMixin = NumpyMixin,
    ):
        self._nstates = nstates
        self._nparams = nresidual_params
        self._obs_state_indices = obs_state_indices
        super().__init__(backend)

    def _value(self, sol: Array) -> Array:
        if sol.ndim != 1:
            raise ValueError("sol must be a 1D array")
        return sol[self._obs_state_indices]

    def nqoi(self) -> int:
        return self._obs_state_indices.shape[0]

    def nunique_functional_params(self) -> int:
        return 0

    def nparams(self) -> int:
        return self._nparams

    def nstates(self) -> int:
        return self._nstates


class SteadyMSEAdjointFunctional(AdjointFunctional):
    def __init__(
        self,
        nstates: int,
        nresidual_params: int,
        obs_state_indices: List[Tuple[int, Array]],
        sigma: float = 1.0,
        backend: BackendMixin = NumpyMixin,
    ):
        self._nstates = nstates
        self._nparams = nresidual_params
        self._obs_state_indices = obs_state_indices
        self._sigma = sigma
        super().__init__(backend)

    def nunique_functional_params(self) -> int:
        return 0

    def set_observations(self, obs: Array):
        if obs.ndim != 1:
            raise ValueError("obs must be 1D array")
        self._obs = obs

    def noiseless_observations_from_solution(self, sol: Array) -> Array:
        if sol.ndim != 1:
            raise ValueError("sol must be a 1D array")
        return sol[self._obs_state_indices]

    def set_param(self, param: Array):
        self._param = param

    def nqoi(self) -> int:
        return 1

    def _value(self, sol: Array) -> Array:
        pred_obs = self.noiseless_observations_from_solution(sol)
        return (
            self._bkd.sum(
                (pred_obs[:, None] - self._obs[:, None]) ** 2,
                axis=0,
            )
            / self._sigma**2
        )

    def _qoi_state_jacobian(self, sol: Array) -> Array:
        dqdu = self._bkd.zeros(sol.shape)
        dqdu[self._obs_state_indices] = (
            2 * (sol[self._obs_state_indices] - self._obs) / self._sigma**2
        )
        return dqdu[None, :]

    def _qoi_param_jacobian(self, sol: Array) -> Array:
        return self._bkd.zeros((self.nparams(),))

    def nstates(self) -> int:
        return self._nstates

    def nparams(self) -> int:
        return self._nparams

    def nobservations(self) -> int:
        return self._obs.shape[0]


class FinalTimeSubdomainIntegralFunctional(TransientAdjointFunctional):
    def __init__(
        self,
        sol: ScalarFunction,
        subdomain_npts_1d: ScalarFunction = None,
    ):
        # currently this functional does not support gradient computation
        # sol must be on full domain
        # subdomain_sol contains subdomain basis which may use a different
        # resolution mesh
        self._sol = sol
        self._subdomain_sol = CollocationSubdomainFunction(
            [0.5, 1.0, -1.0, -0.5], self._sol, subdomain_npts_1d
        )
        super().__init__(self._sol._bkd)

    def _value(self, sol: Array) -> Array:
        if sol.ndim != 2:
            raise ValueError("sol must be a 2D array")
        self._sol.set_values(sol[:, -1])
        self._subdomain_sol._set_values(self._sol)
        return self._subdomain_sol.integrate()

    def nqoi(self) -> int:
        return 1

    def nunique_functional_params(self) -> int:
        return 0

    def nparams(self) -> int:
        # currently this functional does not support gradient computation
        return 0

    def nstates(self) -> int:
        return self._sol.basis().mesh().nmesh_pts()


class SteadyGaussianNegLogLikelihoodAdjointFunctional(
    SteadyMSEAdjointFunctional
):
    """
    The negative log likelihood associated with I.I.D. Gaussian noise
    """

    def _value(self, sol: Array) -> Array:
        return super()._value(sol) / 2.0 + self.nobservations() / 2 * math.log(
            2 * self._sigma**2 * math.pi
        )

    def _qoi_state_jacobian(self, sol: Array) -> Array:
        return super()._qoi_state_jacobian(sol) / 2


class TransientViscousBurgers1DOperatorBenchmark(OperatorBenchmark):
    r"""
    Transient viscous Burgers' equation benchmark.

    This class implements a benchmark for solving the transient viscous Burgers'
    equation in one dimension. The benchmark defines a prior distribution for the
    uncertain variables and sets up the model and functional for the operator-based
    analysis.

    The transient viscous Burgers' equation is given by:

    .. math::
        \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}

    where:

    - :math:`u`: State variable (e.g., velocity).
    - :math:`\nu`: Viscosity coefficient.

    Parameters
    ----------
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(self, backend: BackendMixin = NumpyMixin):
        """
        Initialize the transient viscous Burgers' equation benchmark.

        Parameters
        ----------
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        super().__init__(backend)

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the benchmark.

        The number of uncertain variables is determined by the number of eigenvalues
        used in the Karhunen-Loève expansion.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the benchmark.
        """
        neigs = 1024 // 2
        return neigs * 2

    def prior(self) -> IndependentMarginalsVariable:
        """
        Define the prior distribution for the uncertain variables.

        The prior distribution is defined as independent standard normal random variables.

        Returns
        -------
        prior : IndependentMarginalsVariable
            Prior distribution for the uncertain variables.
        """
        return IndependentMarginalsVariable(
            [stats.norm(0, 1)] * self.nvars(), backend=self._bkd
        )

    def _set_model(self):
        """
        Set up the model and functional for the benchmark.

        The model solves the transient viscous Burgers' equation using the Crank-Nicholson
        time integration scheme. The functional evaluates the solution at the final time.

        Returns
        -------
        None
        """
        self._model = TransientViscousBurgers1DModel(
            0.0,  # Initial time
            1.0,  # Final time
            1 / 200,  # Time step size
            self.nvars() // 2 + 1,  # Number of spatial points
            0.1,  # Viscosity coefficient
            self.nvars() // 2,  # Number of eigenvalues
            49,  # Number of snapshots
            7,  # Number of modes
            2.5,  # Scaling factor
            CrankNicholsonResidual,  # Residual type
            backend=self._bkd,
        )
        self._model.set_functional(
            TransientSolutionTimeSnapshotFunctional(self._model, -1)
        )


class SteadyDarcy2DOperatorBenchmark(OperatorBenchmark):
    r"""
    Steady Darcy 2D operator benchmark.

    This class implements a benchmark for solving the steady-state Darcy equation
    in two dimensions. The benchmark defines a prior distribution for the uncertain
    variables and sets up the model and functional for the operator-based analysis.

    The steady-state Darcy equation is given by:

    .. math::
        -\nabla \cdot (\kappa \nabla u) = f

    where:

    - :math:`u`: State variable (e.g., pressure or hydraulic head).
    - :math:`\kappa`: Permeability field.
    - :math:`f`: Source term.

    The permeability field :math:`\kappa` is parameterized using a Karhunen-Loève expansion (KLE):

    .. math::
        \log(\kappa(x)) = \kappa_0 + \sum_{i=1}^n \sigma_i \phi_i(x)

    where:

    - :math:`\kappa_0`: Mean permeability field.
    - :math:`\sigma_i`: Standard deviation of the :math:`i`-th mode.
    - :math:`\phi_i(x)`: Spatial basis function for the :math:`i`-th mode.

    Parameters
    ----------
    backend : BackendMixin, optional
        Backend for numerical computations. Default is `NumpyMixin`.
    """

    def __init__(self, backend: BackendMixin = NumpyMixin):
        """
        Initialize the steady Darcy 2D operator benchmark.

        Parameters
        ----------
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        super().__init__(backend)

    def nvars(self) -> int:
        """
        Return the number of uncertain variables in the benchmark.

        The number of uncertain variables is determined by the number of modes
        used in the Karhunen-Loève expansion.

        Returns
        -------
        nvars : int
            Number of uncertain variables in the benchmark. For this benchmark, it is always 100.
        """
        return 100

    def prior(self) -> IndependentMarginalsVariable:
        """
        Define the prior distribution for the uncertain variables.

        The prior distribution is defined as independent standard normal random variables.

        Returns
        -------
        prior : IndependentMarginalsVariable
            Prior distribution for the uncertain variables.
        """
        return IndependentMarginalsVariable(
            [stats.norm(0, 1)] * self.nvars(), backend=self._bkd
        )

    def _setup_basis(self, nmesh_pts_1d: Tuple[int, int]):
        r"""
        Set up the basis for the spatial domain.

        Parameters
        ----------
        nmesh_pts_1d : Tuple[int, int]
            Number of mesh points in the x and y directions.

        Returns
        -------
        basis : ChebyshevCollocationBasis2D
            Basis for the spatial domain.

        Notes
        -----
        The spatial domain is defined as a unit square :math:`[0, 1] \times [0, 1]`.
        """
        Lx, Ly = 1, 1
        bounds = self._bkd.array([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], bounds, self._bkd
        )
        mesh = ChebyshevCollocationMesh2D(nmesh_pts_1d, transform)
        basis = ChebyshevCollocationBasis2D(mesh)
        return basis

    def _setup_kle(self):
        r"""
        Set up the Karhunen-Loève expansion (KLE) for the permeability field.

        Returns
        -------
        None

        Notes
        -----
        The permeability field :math:`\kappa` is parameterized using a KLE with
        100 modes and a correlation length of 0.25.
        """
        basis = self._setup_basis(self._bkd.array([20, 20], dtype=int))
        self._kle = ScalarKLEFunction(
            basis,
            0.25,  # Correlation length
            self.nvars(),  # Number of modes
            sigma=1.0,  # Standard deviation
            mean_field=ConstantScalarFunction(basis, 0.0, 1),  # Mean field
            ninput_funs=1,
            use_log=True,
        )

    def _set_model(self) -> SteadyDarcy2DKLEModel:
        """
        Set up the model and functional for the benchmark.

        The model solves the steady-state Darcy equation using the KLE parameterization
        of the permeability field. The functional evaluates the solution.

        Returns
        -------
        None
        """
        self._setup_kle()
        self._model = SteadyDarcy2DKLEModel(self._kle)
        self._model.set_functional(SteadySolutionFunctional(self._model))


class PyApproxPaperAdvectionDiffusionKLEInversionBenchmark(
    SingleModelBayesianInferenceBenchmark
):
    r"""
    Advection-diffusion KLE inversion benchmark.

    This class implements a benchmark for Bayesian inference based on the
    advection-diffusion model with Karhunen-Loève expansion (KLE) parameterization.
    The benchmark defines prior distributions, observation models, and quantities
    of interest (QoI) models.

    The advection-diffusion equation is given by:

    .. math::
        \frac{\partial u}{\partial t} - \nabla \cdot (\kappa \nabla u) + \mathbf{v} \cdot \nabla u = f

    where:

    - :math:`u`: State variable (e.g., concentration or temperature).
    - :math:`\kappa`: Diffusion coefficient.
    - :math:`\mathbf{v}`: Advection velocity.
    - :math:`f`: Source term.

    The diffusion coefficient :math:`\kappa` is parameterized using a Karhunen-Loève expansion (KLE):

    .. math::
        \log(\kappa(x)) = \kappa_0 + \sum_{i=1}^n \sigma_i \phi_i(x)

    where:

    - :math:`\kappa_0`: Mean diffusion coefficient.
    - :math:`\sigma_i`: Standard deviation of the :math:`i`-th mode.
    - :math:`\phi_i(x)`: Spatial basis function for the :math:`i`-th mode.

    References
    ----------
    The example is based on the examle in:

    .. [Jakeman2023] `Jakeman, J.D. "PyApprox: A software package for sensitivity analysis, Bayesian inference, optimal experimental design, and multi-fidelity uncertainty quantification and surrogate modeling." Environmental Modelling & Software, 170, 105825, 2023. <https://doi.org/10.1016/j.envsoft.2023.105825>`_

    However, in this version of PyApprox the benchmark differs slightly. Unlike, when generating the paper results, We solve the conservative form of the equations.
    """

    def __init__(self, backend: BackendMixin = NumpyMixin):
        """
        Initialize the advection-diffusion KLE inversion benchmark.

        Parameters
        ----------
        backend : BackendMixin, optional
            Backend for numerical computations. Default is `NumpyMixin`.
        """
        self._source_loc = backend.array([0.25, 0.75])
        self._source_amp = 100.0
        self._source_scale = 0.1
        self._nobs = 5
        self._noise_stdev = 1.0
        super().__init__(backend)

    def _setup_basis(self, nmesh_pts_1d: Tuple[int, int]):
        """
        Set up the basis for the spatial domain.

        Parameters
        ----------
        nmesh_pts_1d : Tuple[int, int]
            Number of mesh points in the x and y directions.

        Returns
        -------
        basis : ChebyshevCollocationBasis2D
            Basis for the spatial domain.
        """
        Lx, Ly = 1, 1
        bounds = self._bkd.array([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], bounds, self._bkd
        )
        mesh = ChebyshevCollocationMesh2D(nmesh_pts_1d, transform)
        basis = ChebyshevCollocationBasis2D(mesh)
        return basis

    def _setup_kle(self):
        """
        Set up the Karhunen-Loève expansion (KLE) for the diffusion coefficient.
        """
        basis = self._setup_basis(self._bkd.array([20, 20], dtype=int))
        self._kle = ScalarKLEFunction(
            basis,
            0.5,
            3,
            sigma=1.0,
            mean_field=ConstantScalarFunction(basis, 0.0, 1),
            ninput_funs=1,
            use_log=True,
        )

    def observation_design(self) -> Array:
        """
        Return the spatial locations where the observations are collected.

        Returns
        -------
        observation_design : Array
            Array of shape (2, nobs) containing the observation locations.
        """
        return self._obs_model._basis.mesh().mesh_pts()[:, self._obs_indices]

    def observation_generating_parameters(self) -> Array:
        """
        Return the true parameters used to generate the observations.

        Returns
        -------
        observation_generating_parameters : Array
            Array containing the true parameters for the KLE.
        """
        return self._true_kle_params

    def observations(self) -> Array:
        """
        Return the collected observations.

        Returns
        -------
        observations : Array
            Array containing the collected observations.
        """
        return self._obs

    def nobservations(self) -> Array:
        """
        Return the number of observations.

        Returns
        -------
        nobservations : int
            Number of observations.
        """
        return self._obs.shape[0]

    def nqoi(self) -> int:
        """
        Return the number of quantities of interest (QoI).

        Returns
        -------
        nqoi : int
            Number of QoI.
        """
        return self._functional.nqoi()

    def _set_prior(self):
        """
        Define the prior distribution for the uncertain variables.
        """
        self._setup_kle()
        marginals = [stats.norm(0, 1)] * self._kle.kle().nvars()
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_obs_model(self):
        """
        Set up the observation model for the benchmark.
        """
        self._obs_model = PyApproxPaperAdvectionDiffusionKLEInversionModel(
            self._kle, self._source_amp, self._source_loc, self._source_scale
        )
        self._obs_model._adjoint_solver.set_initial_iterate(
            self._obs_model.get_initial_iterate()
        )
        self._set_negloglike_model()

    def _set_negloglike_model(self):
        """
        Set up the negative log-likelihood model for the benchmark.
        """
        self._negloglike_model = (
            PyApproxPaperAdvectionDiffusionKLEInversionModel(
                self._kle,
                self._source_amp,
                self._source_loc,
                self._source_scale,
            )
        )
        npts_1d = self._negloglike_model.basis().mesh()._npts_1d
        ndof = self._bkd.prod(npts_1d)
        bndry_indices = self._bkd.hstack(
            [
                self._bkd.arange(0, npts_1d[0]),
                self._bkd.arange(ndof - npts_1d[0], ndof),
            ]
            + [jj * (npts_1d[0]) for jj in range(1, npts_1d[1] - 1)]
            + [
                jj * (npts_1d[0]) + npts_1d[0] - 1
                for jj in range(1, npts_1d[1] - 1)
            ]
        )
        self._obs_indices = self._bkd.asarray(
            np.random.permutation(
                self._bkd.delete(self._bkd.arange(ndof), bndry_indices)
            ),
            dtype=int,
        )[: self._nobs]
        self._obs_functional = PointwiseObservationFunctional(
            ndof, self._kle.kle().nvars(), self._obs_indices, backend=self._bkd
        )
        self._obs_model.set_functional(self._obs_functional)

        self._negloglike_functional = (
            SteadyGaussianNegLogLikelihoodAdjointFunctional(
                ndof,
                self._kle.kle().nvars(),
                self._obs_indices,
                self._noise_stdev,
                backend=self._bkd,
            )
        )
        self._negloglike_model._adjoint_solver.set_initial_iterate(
            self._negloglike_model.get_initial_iterate()
        )
        self._true_kle_params = self._prior.rvs(1)
        noiseless_obs = self._obs_model(self._true_kle_params)[0]
        noise = self._bkd.asarray(
            np.random.normal(
                0, self._noise_stdev, (self._obs_indices.shape[0])
            )
        )
        self._obs = noiseless_obs + noise
        self._negloglike_functional.set_observations(self._obs)
        self._negloglike_model.set_functional(self._negloglike_functional)

    def setup_qoi_model(
        self, nmesh_pts_x: int, nmesh_pts_y: int, deltat: float
    ):
        """
        Set up a QoI model for the benchmark.

        Parameters
        ----------
        nmesh_pts_x : int
            Number of mesh points in the x direction.
        nmesh_pts_y : int
            Number of mesh points in the y direction.
        deltat : float
            Time step size.

        Returns
        -------
        qoi_model : PyApproxPaperAdvectionDiffusionKLEForwardModel
            QoI model for the benchmark.
        """
        basis = self._setup_basis((nmesh_pts_x, nmesh_pts_y))
        kle = ScalarKLEFunctionOnDifferentMesh(
            self._obs_model.physics()._diffusion, basis
        )
        qoi_model = PyApproxPaperAdvectionDiffusionKLEForwardModel(
            self._obs_model, kle, deltat=deltat, final_time=0.2
        )
        sol = ZeroScalarFunction(qoi_model._basis)
        qoi_functional = FinalTimeSubdomainIntegralFunctional(
            sol, self._obs_model.basis().mesh()._npts_1d
        )
        qoi_model.set_functional(qoi_functional)
        return qoi_model

    def _qoi_models(
        self, model_config: List[Tuple[int, int, float]]
    ) -> List[Model]:
        """
        Define a list of QoI models based on the given configuration.

        Parameters
        ----------
        model_config : List[Tuple[int, int, float]]
            List of configurations for the QoI models, where each configuration
            specifies the number of mesh points in the x and y directions and the
            time step size.

        Returns
        -------
        models : List[Model]
            List of QoI models.
        """
        models = [self.setup_qoi_model(*config) for config in model_config]
        return models

    def qoi_models(self) -> List[Model]:
        """
        Return the QoI models for the benchmark.

        Returns
        -------
        models : List[Model]
            List of QoI models.
        model_names : List[str]
            Names of the QoI models.
        """
        npts_x = [4, int(self._obs_model.basis().mesh()._npts_1d[0])]
        npts_y = npts_x
        ntsteps = [2, 8]
        deltat = [0.2 / nt for nt in ntsteps]
        model_config = itertools.product(npts_x, npts_y, deltat)
        model_names = list(itertools.product(npts_x, npts_y, ntsteps))
        models = self._qoi_models(model_config)
        models = models[-1:] + models[:-1]  # High-fidelity model first
        model_names = model_names[-1:] + model_names[:-1]
        return models, [str(name) for name in model_names]

    def obs_model(self) -> Model:
        """
        Return the observation model for the benchmark.

        Returns
        -------
        obs_model : Model
            Observation model for the benchmark.
        """
        return self._obs_model

    def diffusion_function(self) -> ScalarFunction:
        """
        Return the diffusion function parameterized by the KLE.

        Returns
        -------
        diffusion_function : ScalarFunction
            Diffusion function parameterized by the KLE.
        """
        return self._obs_model._physics._diffusion

    def loglike(self) -> LogLikelihood:
        """
        Return the log-likelihood model for the benchmark.

        Returns
        -------
        loglike : LogLikelihood
            Log-likelihood model for the benchmark.
        """
        return LogLikelihoodFromModel(
            ChangeModelSignWrapper(self._negloglike_model)
        )

    def sobol_interaction_indices(self) -> Array:
        """
        Return the Sobol interaction indices for the benchmark.

        Returns
        -------
        sobol_interaction_indices : Array
            Array containing the Sobol interaction indices.
        """
        sobol_interaction_indices = self._bkd.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ],
            dtype=int,
        ).T
        return sobol_interaction_indices


class NonlinearSystemOfEquationsBenchmark(SingleModelBenchmark):
    r"""
    Nonlinear system of equations benchmark.

    This class implements a benchmark for solving a nonlinear system of equations
    using a steady adjoint model. The benchmark defines a prior distribution for
    the uncertain variables and sets up the nonlinear system of equations model.

    The system is governed by the following equations:

    .. math::
        f_1(x_1, x_2) = a \cdot x_1^2 + x_2^2 - 1 \\
        f_2(x_1, x_2) = x_1^2 - b \cdot x_2^2 - 1

    where:
    - :math:`x_1, x_2`: State variables.
    - :math:`a, b`: Parameters.

    Parameters
    ----------
    backend : BackendMixin
        Backend for numerical computations.

    Notes
    -----
    The parameters of the model are defined as follows:

    - :math:`a`: Coefficient controlling the influence of :math:`x_1` in :math:`f_1`.
    - :math:`b`: Coefficient controlling the influence of :math:`x_2` in :math:`f_2`.

    The prior distribution is defined as:

    .. math::
        a \sim \mathcal{U}[0.79, 0.99] \\
        b \sim \mathcal{U}[1 - 4.5 \sqrt{0.1}, 1 + 4.5 \sqrt{0.1}]

    References
    ----------
    .. [Butler2018] `Butler, T., Jakeman, J., & Wildey, T. "Combining Push-Forward Measures and Bayes' Rule to Construct Consistent Solutions to Stochastic Inverse Problems." SIAM Journal on Scientific Computing, 40(2), A984-A1011, 2018. <https://doi.org/10.1137/16M1087229>`_
    """

    def _set_prior(self):
        r"""
        Define the input prior for the benchmark.

        The input prior is defined as two independent uniform random variables.

        Returns
        -------
        None

        Notes
        -----
        The prior distribution is defined as:

        .. math::
            a \sim \mathcal{U}[0.79, 0.99] \\
            b \sim \mathcal{U}[1 - 4.5 \sqrt{0.1}, 1 + 4.5 \sqrt{0.1}]
        """
        marginals = [
            stats.uniform(0.79, 0.2),
            stats.uniform(1 - 4.5 * np.sqrt(0.1), 2 * 4.5 * np.sqrt(0.1)),
        ]
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_model(self):
        r"""
        Set up the nonlinear system of equations model for the benchmark.

        Returns
        -------
        None

        Notes
        -----
        The system is governed by the following equations:

        .. math::
            f_1(x_1, x_2) = a^p \cdot x_1^2 + x_2^2 - 1 \\
            f_2(x_1, x_2) = x_1^2 - b^q \cdot x_2^2 - 1

        where:
  
        - :math:`x_1, x_2`: State variables.
        - :math:`a, b`: Parameters.
        - :math:`p, q`: Powers of the parameters.
        """
        functional = SteadySingleStateFunctional(1, 2, 2, backend=self._bkd)
        newton_solver = NewtonSolver(atol=1e-10, rtol=1e-10)
        self._model = NonlinearSystemOfEquationsModel(
            newton_solver, functional=functional, backend=self._bkd
        )


class ObstructedAdvectionDiffusionOEDBenchmark(
    SingleModelBayesianGoalOrientedOEDBenchmark
):
    def __init__(self, backend: BackendMixin):
        """
        Initialize the Obstructed Advection-Diffusion OED benchmark.

        Parameters
        ----------
        backend : BackendMixin
            Backend for numerical computations.
        """
        self._time_residual_cls = BackwardEulerResidual
        self._newton_solver = None
        self._init_time = 0.0
        self._final_time, self._timestep = self._define_time()
        self._kle_hyperparams = KLEHyperParameters(0.5, 1.0, np.inf, 10)
        self._nobs = 200
        super().__init__(backend)

    def _set_prior(self):
        """
        Define the prior distribution for the uncertain variables.

        The prior distribution is comprised of independent standard normals
        for the KLE coefficients, an independent uniform distribution on [5, 15]
        for the Reynolds number and independent uniform distributions on
        [2, 3] dicating the inlet velocity
        """
        kle_marginals = [
            stats.norm(0.0, 1.0) for ii in range(self._kle_hyperparams.nterms)
        ]
        inlet_marginals = [stats.uniform(2.0, 1.0) for ii in range(2)]
        reynolds_marginal = [stats.uniform(5.0, 15.0)]
        marginals = kle_marginals + inlet_marginals + reynolds_marginal
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
        return 1.5, 0.25

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
        # Take 3 measurements at each location (nodes on the mesh)
        obs_time_indices = self._bkd.arange(ntimes, dtype=int)[[2, 4, 6]]
        node_indices = self._bkd.asarray(
            np.random.randint(
                0, self._obs_model.fe_model().nmesh_pts(), self._nobs
            )
        )
        obs_time_tuples = [(idx, obs_time_indices) for idx in node_indices]
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

    def observation_locations(self) -> Array:
        """
        Return the observation spatial locations for the simulation.

        Returns
        -------
        locs : Array
            Array containing the observation locations.
        """
        locs = (
            self._obs_model.fe_model()
            .mesh()
            .p[:, self._obs_model._functional._obs_state_indices]
        )
        return self._bkd.asarray(locs)

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

    def _set_obs_model(self):
        """
        Set up the observation model for the benchmark.
        """
        obs_fe_model = ObstructedAdvectionDiffusion(
            3, 3, self._timestep, self._final_time, self._kle_hyperparams, True
        )
        self._obs_model = FETransientOutputModel(obs_fe_model, self._bkd)
        self._times = self._bkd.linspace(
            0,
            self._final_time,
            int(self._final_time / self._timestep) + 1,
        )
        obs_functional = TransientObservationFunctional(
            self._obs_model.fe_model().nmesh_pts(),
            self._obs_model.nvars(),
            self._obs_time_tuples(self._times.shape[0]),
            backend=self._bkd,
        )
        self._obs_model.set_functional(obs_functional)

    def _integrate_snapshot_on_target_subdomain(
        self, sol: np.ndarray
    ) -> float:
        return self._solver.integrate_on_subdomain(sol, "target_subdomain")

    def _set_pred_model(self):
        """
        Set up the prediction model for the benchmark.

        Returns
        -------
        None
        """
        pred_fe_model = ObstructedAdvectionDiffusion(
            3, 3, self._timestep, self._final_time, self._kle_hyperparams, True
        )

        self._pred_model = FETransientOutputModel(pred_fe_model, self._bkd)
        pred_functional = FETransientSubdomainIntegralFunctional(
            pred_fe_model.nmesh_pts(),
            pred_fe_model.nparams(),
            pred_fe_model._physics.subdomain_basis("target_subdomain"),
            self._bkd,
        )
        self._pred_model.set_functional(pred_functional)

    def prediction_model(self) -> FETransientOutputModel:
        """
        Return the prediction model for the benchmark.

        Returns
        -------
        prediction_model : LotkaVolterraModel
            The prediction model for the benchmark.
        """
        return self._pred_model
