from typing import List, Tuple

from scipy import stats

from pyapprox.variables.joint import IndependentMarginalsVariable
from pyapprox.interface.model import Model
from pyapprox.util.linearalgebra.linalgbase import LinAlgMixin, Array
from pyapprox.benchmarks.base import MultiModelBenchmark
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.pde.collocation.adjoint_models import AdjointFunctional


class SteadyMSEAdjointFunctional(AdjointFunctional):
    def __init__(
        self,
        nstates: int,
        nresidual_params: int,
        obs_state_indices: List[Tuple[int, Array]],
        sigma: float = 1.,
        backend: LinAlgMixin = NumpyLinAlgMixin,
    ):
        self._nstates = nstates
        self._nparams = nresidual_params
        self._obs_state_indices = obs_state_indices
        self._sigma = sigma
        super().__init__(backend)

    def nunique_functional_params(self) -> int:
        return 0

    def observations_from_solution(self, sol: Array) -> Array:
        return sol[self._obs_state_indices]

    def set_param(self, param: Array):
        self._param = param
        self._sigma = self._unique_functional_params(self._param)[0]

    def nqoi(self) -> int:
        return 1

    def _value(self, sol: Array) -> Array:
        pred_obs = self.observations_from_solution(sol)
        return self._bkd.atleast1d(
            self._bkd.sum((pred_obs - self._obs) ** 2) / self._sigma**2
        )

    def _qoi_state_jacobian(self, sol: Array) -> Array:
        dqdu = self._bkd.zeros(sol.shape)
        dqdu[self._obs_state_indices] = (
            2 * (sol[self._obs_state_indices] - self._obs) / self._sigma**2
        )
        return dqdu

    def _qoi_param_jacobian(self, sol: Array) -> Array:
        return self._bkd.zeros((self.nparams(),))


class PyApproxPaperAdvectionDiffusionKLEInversionBenchmark(
    MultiModelBenchmark
):
    def __init__(
        self, nmodels: int = 5, backend: LinAlgMixin = NumpyLinAlgMixin
    ):
        self._nmodels = nmodels
        self._source_loc = self._bkd.array([0.25, 0.75])
        self._source_amp = 100
        self._source_width = 0.1
        self._kle_lscale = 0.5
        self._kle_std = 1.0
        self._kle_nvars = 3
        self._nobs = 3
        self._noise_stdev = 1.0
        self._mesh_nterms = self._bkd.array([20, 20])
        super().__init__(backend)
        self._set_model_functional()

    def _set_model_functional(self):
        ndof = self._bkd.prod(self._mesh_nterms)
        bndry_indices = self._bkd.hstack(
            [
                self._bkd.arange(0, self._mesh_nterms[0]),
                self._bkd.arange(ndof - self._mesh_nterms[0] - 2, ndof),
            ]
            + [
                jj * (self._mesh_nterms[0])
                for jj in range(1, self._mesh_nterms[1] - 1)
            ]
            + [
                jj * (self._mesh_nterms[0]) + self._mesh_nterms[0] - 1
                for jj in range(1, self._mesh_nterms[1] - 1)
            ]
        )
        obs_indices = self._bkd.random.permutation(
            self._bkd.delete(self._bkd.arange(ndof), bndry_indices)
        )[: self._nobs]
        self._functional = SteadyMSEAdjointFunctional(
            ndof, self._kle_nvars, obs_indices, self._noise_stdev
        )
        # run model at true param
        true_kle_params = self._variable.rvs(1)
        self._model.set_param(true_kle_params)
        sol = self._model.forward_solve()
        obs = self._functional.observations_from_solution(sol)
        self._functional.set_observations(obs)

    def nmodels(self) -> int:
        return self._nmodels

    def nqoi(self) -> int:
        return self._functional.nqoi()

    def _set_variable(self):
        marginals = stats.normal([stats.norm(0, 1)]*self._kle_nvars)
        self._variable = IndependentMarginalsVariable(marginals)
