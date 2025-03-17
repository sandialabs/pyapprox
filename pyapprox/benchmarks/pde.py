import math
from typing import List, Tuple

import numpy as np
from scipy import stats

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.benchmarks.base import (
    SingleModelBayesianInferenceBenchmark,
    OperatorBenchmark,
    SingleModelBenchmark,
)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.interface.model import Model
from pyapprox.pde.collocation.adjoint_models import AdjointFunctional
from pyapprox.pde.collocation.parameterized_pdes import (
    PyApproxPaperAdvectionDiffusionKLEInversionModel,
    ScalarFunction,
    TransientViscousBurgers1DModel,
    SteadyDarcy2DKLEModel,
    TransientSolutionTimeSnapshotFunctional,
    SteadySolutionFunctional,
    SteadySingleStateFunctional,
)
from pyapprox.pde.collocation.newton import NewtonSolver
from pyapprox.pde.collocation.timeintegration import CrankNicholsonResidual
from pyapprox.pde.collocation.parameterized_pdes import (
    NonlinearSystemOfEquationsModel,
)


class SteadyMSEAdjointFunctional(AdjointFunctional):
    def __init__(
        self,
        nstates: int,
        nresidual_params: int,
        obs_state_indices: List[Tuple[int, Array]],
        sigma: float = 1.0,
        backend: LinAlgMixin = NumpyLinAlgMixin,
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

    def nobservations(self):
        return self._obs.shape[0]


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
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
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
    def __init__(self, backend: LinAlgMixin = NumpyLinAlgMixin):
        super().__init__(backend)

    def nvars(self) -> int:
        return 100

    def variable(self) -> IndependentMarginalsVariable:
        return IndependentMarginalsVariable(
            [stats.norm(0, 1)] * self.nvars(), backend=self._bkd
        )

    def _set_model(self) -> SteadyDarcy2DKLEModel:
        self._model = SteadyDarcy2DKLEModel(
            self.nvars(),
            1.0,
            0.25,
            0.0,
            backend=self._bkd,
        )
        self._model.set_functional(SteadySolutionFunctional(self._model))


class PyApproxPaperAdvectionDiffusionKLEInversionBenchmark(
    SingleModelBayesianInferenceBenchmark
):
    def __init__(
        self, nmodels: int = 5, backend: LinAlgMixin = NumpyLinAlgMixin
    ):
        self._nmodels = nmodels
        self._mesh_npts_1d = backend.array([21, 21], dtype=int)
        self._kle_nvars = 3
        self._kle_std = 1.0
        self._kle_lenscale = 0.5
        self._kle_mean_field = 0.0
        self._source_loc = backend.array([0.25, 0.75])
        self._source_amp = 100.0
        self._source_scale = 0.1
        self._nobs = 5
        self._noise_stdev = 1.0
        super().__init__(backend)
        self._set_model_functional()

    def _set_model_functional(self):
        ndof = self._bkd.prod(self._mesh_npts_1d)
        bndry_indices = self._bkd.hstack(
            [
                self._bkd.arange(0, self._mesh_npts_1d[0]),
                self._bkd.arange(ndof - self._mesh_npts_1d[0], ndof),
            ]
            + [
                jj * (self._mesh_npts_1d[0])
                for jj in range(1, self._mesh_npts_1d[1] - 1)
            ]
            + [
                jj * (self._mesh_npts_1d[0]) + self._mesh_npts_1d[0] - 1
                for jj in range(1, self._mesh_npts_1d[1] - 1)
            ]
        )
        obs_indices = self._bkd.asarray(
            np.random.permutation(
                self._bkd.delete(self._bkd.arange(ndof), bndry_indices)
            ),
            dtype=int,
        )[: self._nobs]
        # print(bndry_indices.shape)
        self._functional = SteadyGaussianNegLogLikelihoodAdjointFunctional(
            ndof,
            self._kle_nvars,
            obs_indices,
            self._noise_stdev,
            backend=self._bkd,
        )
        # run model at true param
        self._true_kle_params = self._variable.rvs(1)
        sol = self._model.forward_solve(self._true_kle_params)
        noiseless_obs = self._functional.noiseless_observations_from_solution(
            sol
        )
        noise = self._bkd.asarray(
            np.random.normal(0, self._noise_stdev, (obs_indices.shape[0]))
        )
        obs = noiseless_obs + noise
        self._functional.set_observations(obs)
        self._model.set_functional(self._functional)

    def true_params(self) -> Array:
        return self._true_kle_params

    def nmodels(self) -> int:
        return self._nmodels

    def nqoi(self) -> int:
        return self._functional.nqoi()

    def _set_variable(self):
        marginals = [stats.norm(0, 1)] * self._kle_nvars
        self._variable = IndependentMarginalsVariable(
            marginals, backend=self._bkd
        )

    def _set_model(self):
        self._model = PyApproxPaperAdvectionDiffusionKLEInversionModel(
            self._mesh_npts_1d,
            self._kle_nvars,
            self._kle_std,
            self._kle_lenscale,
            self._kle_mean_field,
            self._source_amp,
            self._source_loc,
            self._source_scale,
            backend=self._bkd,
        )

    def diffusion_function(self) -> ScalarFunction:
        return self._model._physics._diffusion

    def negloglike(self) -> Model:
        return self._model


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
