from pyapprox.optimization.adjoint import ScalarAdjointFunctionalWithHessian
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

    def _value(self, state: Array, param: Array) -> float:
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
    def _value(self, state: Array, param: Array) -> float:
        return super()._value(state, param) + self._bkd.sum(param**2) / 2.0

    def _param_jacobian(self, state: Array, param: Array) -> Array:
        return param[None, :]

    def _param_param_hvp(
        self, state: Array, param: Array, vvec: Array
    ) -> Array:
        return vvec
