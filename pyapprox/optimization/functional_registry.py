from pyapprox.optimization.adjoint import (
    ScalarAdjointFunctionalWithHessian,
    VectorAdjointFunctional,
)
from pyapprox.util.backends.template import Array, BackendMixin


class MSEFunctional(ScalarAdjointFunctionalWithHessian):
    def __init__(self, nstates: int, nvars: int, backend: BackendMixin):
        self._nstates = nstates
        self._nvars = nvars
        super().__init__(backend)

    def set_observations(self, obs: Array):
        if obs.shape != (self.nstates(),):
            raise ValueError(
                f"obs has shape {obs.shape} but must have "
                f"shape {(self.nstates(),)}"
            )
        self._obs = obs

    def nstates(self) -> int:
        return self._nstates

    def nvars(self) -> int:
        return self._nvars

    def nunique_vars(self) -> int:
        return 0

    def _value(self, state: Array, param: Array) -> Array:
        return self._bkd.sum((self._obs - state) ** 2) / 2.0

    def _param_jacobian(self, state: Array, param: Array) -> Array:
        return self._bkd.zeros((1, self.nvars()))

    def _state_jacobian(self, state: Array, param: Array) -> Array:
        return (state - self._obs)[None, :]

    def _param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(),))

    def _state_state_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return vvec

    def _param_state_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(),))

    def _state_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(),))


class TikhinovMSEFunctional(MSEFunctional):
    def _value(self, state: Array, param: Array) -> Array:
        return super()._value(state, param) + self._bkd.sum(param**2) / 2.0

    def _param_jacobian(self, state: Array, param: Array) -> Array:
        return param[None, :]

    def _param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return vvec


class WeightedSumFunctional(ScalarAdjointFunctionalWithHessian):
    def __init__(self, weights: Array, nvars: int, backend: BackendMixin):
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        self._nstates = weights.shape[0]
        self._nvars = nvars
        self._weights = weights
        super().__init__(backend)

    def nstates(self) -> int:
        return self._nstates

    def nvars(self) -> int:
        return self._nvars

    def nunique_vars(self) -> int:
        return 0

    def _value(self, state: Array, param: Array) -> float:
        return self._bkd.sum(state * self._weights)

    def _param_jacobian(self, state: Array, param: Array) -> Array:
        return self._bkd.zeros((1, self.nvars()))

    def _state_jacobian(self, state: Array, param: Array) -> Array:
        return self._weights[None, :]

    def _param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(),))

    def _state_state_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(),))

    def _param_state_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(),))

    def _state_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(),))


class SubsetVectorAdjointFunctional(VectorAdjointFunctional):
    def __init__(
        self, nstates: int, nvars: int, subset: Array, backend: BackendMixin
    ):
        self._nstates = nstates
        self._nvars = nvars
        if subset.shape[0] > self.nstates():
            raise ValueError(
                "subset index must be smaller than self.nstates()"
            )
        self._subset = subset
        super().__init__(backend)

    def nstates(self) -> int:
        return self._nstates

    def nvars(self) -> int:
        return self._nvars

    def nunique_vars(self) -> int:
        return 0

    def _value(self, state: Array, param: Array) -> Array:
        return state[self._subset]

    def nqoi(self) -> int:
        return self._subset.shape[0]

    def _param_jacobian(self, state: Array, param: Array) -> Array:
        return self._bkd.zeros((self.nqoi(), self.nvars()))

    def _state_jacobian(self, state: Array, param: Array) -> Array:
        jac = self._bkd.zeros((self.nqoi(), self.nvars()))
        jac[self._subset, self._subset] = 1.0
        return jac
