import unittest

import numpy as np

from pyapprox.util.backends.template import Array, BackendMixin
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.optimization.adjoint import (
    AdjointFunctional,
    AdjointInterfaceConstraintEquation,
    AdjointInterface,
)


class MSEFunctional(AdjointFunctional):
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

    def _qoi_param_jacobian(self, state: Array, param: Array) -> Array:
        return self._bkd.zeros((1, self.nparams()))

    def _qoi_state_jacobian(self, state: Array, param: Array) -> Array:
        return (state - self._obs)[None, :]


class LinearConstraintEquation(AdjointInterfaceConstraintEquation):
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

    def _solve(self, init_state: Array, param: Array):
        # init_state is ignored for this linear problem
        return self._bkd.lstsq(self._Amat, self._bvec)

    def _param_jacobian(self, state: Array, param: Array):
        return -self._Amat

    def _state_jacobian(self, state: Array, param: Array):
        return self._bkd.eye(self.nstates())


class TestAdjoint:
    def setUp(self):
        np.random.seed(1)

    def _check_linear_least_squares(self, nobs: int, nvars: int):
        bkd = self.get_backend()
        Amat = bkd.asarray(np.random.normal(0, 1, (nobs, nvars)))
        generating_params = bkd.asarray(np.random.normal(0, 1, (nvars, 1)))
        print(Amat.shape, generating_params.shape)
        obs = (Amat @ generating_params).T
        functional = MSEFunctional(nobs, nvars, bkd)
        functional.set_observations(obs[0])
        constraint_eq = LinearConstraintEquation(Amat, obs[0], bkd)
        adjointinterface = AdjointInterface(constraint_eq, functional)
        params = generating_params + 0.5
        init_state = bkd.zeros((nobs,))
        jac = adjointinterface.jacobian(init_state, params[:, 0])
        print(jac)

    def test_linear_least_squares(self):
        self._check_linear_least_squares(2, 3)
        self._check_linear_least_squares(1, 4)


class TestNumpyAdjoint(TestAdjoint, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchAdjoint(TestAdjoint, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
