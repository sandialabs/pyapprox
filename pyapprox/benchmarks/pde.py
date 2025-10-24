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
from pyapprox.pde.timeintegration import CrankNicholsonResidual
from pyapprox.bayes.likelihood import LogLikelihoodFromModel, LogLikelihood


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
    def __init__(self, backend: BackendMixin = NumpyMixin):
        super().__init__(backend)

    def nvars(self) -> int:
        neigs = 1024 // 2
        return neigs * 2

    def variable(self) -> IndependentMarginalsVariable:
        return IndependentMarginalsVariable(
            [stats.norm(0, 1)] * self.nvars(), backend=self._bkd
        )

    def _set_model(self):
        self._model = TransientViscousBurgers1DModel(
            0.0,
            1.0,
            1 / 200,
            self.nvars() // 2 + 1,  # 1025 ~6 times faster than 1024 + 1
            0.1,
            self.nvars() // 2,
            49,
            7,
            2.5,
            CrankNicholsonResidual,
            backend=self._bkd,
        )
        self._model.set_functional(
            TransientSolutionTimeSnapshotFunctional(self._model, -1)
        )


class SteadyDarcy2DOperatorBenchmark(OperatorBenchmark):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        super().__init__(backend)

    def nvars(self) -> int:
        return 100

    def variable(self) -> IndependentMarginalsVariable:
        return IndependentMarginalsVariable(
            [stats.norm(0, 1)] * self.nvars(), backend=self._bkd
        )

    def _setup_basis(self, nmesh_pts_1d: Tuple[int, int]):
        Lx, Ly = 1, 1
        bounds = self._bkd.array([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], bounds, self._bkd
        )
        mesh = ChebyshevCollocationMesh2D(nmesh_pts_1d, transform)
        basis = ChebyshevCollocationBasis2D(mesh)
        return basis

    def _setup_kle(self):
        basis = self._setup_basis(self._bkd.array([20, 20], dtype=int))
        self._kle = ScalarKLEFunction(
            basis,
            0.25,
            self.nvars(),
            sigma=1.0,
            mean_field=ConstantScalarFunction(basis, 0.0, 1),
            ninput_funs=1,
            use_log=True,
        )

    def _set_model(self) -> SteadyDarcy2DKLEModel:
        self._setup_kle()
        self._model = SteadyDarcy2DKLEModel(self._kle)
        self._model.set_functional(SteadySolutionFunctional(self._model))


class PyApproxPaperAdvectionDiffusionKLEInversionBenchmark(
    SingleModelBayesianInferenceBenchmark
):
    def __init__(self, backend: BackendMixin = NumpyMixin):
        self._source_loc = backend.array([0.25, 0.75])
        self._source_amp = 100.0
        self._source_scale = 0.1
        self._nobs = 5
        self._noise_stdev = 1.0
        super().__init__(backend)

    def _setup_basis(self, nmesh_pts_1d: Tuple[int, int]):
        Lx, Ly = 1, 1
        bounds = self._bkd.array([0, Lx, 0, Ly])
        transform = ScaleAndTranslationTransform2D(
            [-1, 1, -1, 1], bounds, self._bkd
        )
        mesh = ChebyshevCollocationMesh2D(nmesh_pts_1d, transform)
        basis = ChebyshevCollocationBasis2D(mesh)
        return basis

    def _setup_kle(self):
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
        Spatial locations where the observations are collected
        """
        return self._obs_model._basis.mesh().mesh_pts()[:, self._obs_indices]

    def observation_generating_parameters(self) -> Array:
        return self._true_kle_params

    def observations(self) -> Array:
        return self._obs

    def nobservations(self) -> Array:
        return self._obs.shape[0]

    def nqoi(self) -> int:
        return self._functional.nqoi()

    def _set_prior(self):
        self._setup_kle()
        marginals = [stats.norm(0, 1)] * self._kle.kle().nvars()
        self._prior = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_obs_model(self):
        self._obs_model = PyApproxPaperAdvectionDiffusionKLEInversionModel(
            self._kle, self._source_amp, self._source_loc, self._source_scale
        )
        # must set initial iterate so that forward solver runs
        # since these equations are linear we can always use the same initial
        # guess
        self._obs_model._adjoint_solver.set_initial_iterate(
            self._obs_model.get_initial_iterate()
        )

        self._set_negloglike_model()

    def _set_negloglike_model(self):
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
        # print(bndry_indices.shape)
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
        # must set initial iterate so that forward solver runs
        # since these equations are linear we can always use the same initial
        # guess
        self._negloglike_model._adjoint_solver.set_initial_iterate(
            self._negloglike_model.get_initial_iterate()
        )

        # run model at true param
        self._true_kle_params = self._prior.rvs(1)
        # sol = self._negloglike_model.forward_solve(self._true_kle_params)
        # noiseless_obs = self._obs_functional(sol)
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
        models = [self.setup_qoi_model(*config) for config in model_config]
        return models

    def qoi_models(self) -> List[Model]:
        npts_x = [4, int(self._obs_model.basis().mesh()._npts_1d[0])]
        npts_y = npts_x
        ntsteps = [2, 8]
        deltat = [0.2 / nt for nt in ntsteps]
        model_config = itertools.product(npts_x, npts_y, deltat)
        model_names = list(itertools.product(npts_x, npts_y, ntsteps))
        models = self._qoi_models(model_config)
        # multifidelity algorithms require hf model to be first in list
        models = models[-1:] + models[:-1]
        model_names = model_names[-1:] + model_names[:-1]
        return models, [str(name) for name in (model_names)]

    def obs_model(self) -> Model:
        return self._obs_model

    def diffusion_function(self) -> ScalarFunction:
        return self._obs_model._physics._diffusion

    def loglike(self) -> LogLikelihood:
        return LogLikelihoodFromModel(
            ChangeModelSignWrapper(self._negloglike_model)
        )

    def sobol_interaction_indices(self) -> Array:
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
    def _set_variable(self):
        marginals = [
            stats.uniform(0.79, 0.2),
            stats.uniform(1 - 4.5 * np.sqrt(0.1), 2 * 4.5 * np.sqrt(0.1)),
        ]
        self._variable = IndependentMarginalsVariable(marginals)

    def _set_model(self):
        functional = SteadySingleStateFunctional(1, 2, 2, backend=self._bkd)
        newton_solver = NewtonSolver(atol=1e-10, rtol=1e-10)
        self._model = NonlinearSystemOfEquationsModel(
            newton_solver, functional=functional, backend=self._bkd
        )
