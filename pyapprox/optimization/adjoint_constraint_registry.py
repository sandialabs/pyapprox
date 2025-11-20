from pyapprox.util.backends.template import Array, BackendMixin
from pyapprox.optimization.adjoint import AdjointConstraintEquationWithHessian


class LinearConstraintEquation(AdjointConstraintEquationWithHessian):
    def __init__(self, Amat: Array, bvec: Array, backend: BackendMixin):
        super().__init__(backend)
        if bvec.ndim != 1:
            raise ValueError(
                f"bvec must be a 1D array but has shape {bvec.shape}"
            )
        if Amat.shape[0] != bvec.shape[0]:
            raise ValueError(
                "Amat and bvec must have the same number of rows"
                f"but had shapes {Amat.shape} and {bvec.shape}"
            )
        self._Amat = Amat
        self._bvec = bvec

    def nstates(self) -> int:
        return self._Amat.shape[0]

    def nvars(self) -> int:
        return self._Amat.shape[1]

    def _value(self, state: Array, param: Array) -> Array:
        return state - self._Amat @ param

    def _solve(self, init_state: Array, param: Array):
        # init_state is ignored for this linear problem
        return self._Amat @ param

    def _param_jacobian(self, state: Array, param: Array):
        return -self._Amat

    def _state_jacobian(self, state: Array, param: Array):
        return self._bkd.eye(self.nstates())

    def param_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(),))

    def _state_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(),))

    def _param_state_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, wvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nvars(),))

    def _state_param_hvp(
        self, fwd_state: Array, param: Array, adj_state: Array, vvec: Array
    ) -> Array:
        return self._bkd.zeros((self.nstates(),))
        pass
