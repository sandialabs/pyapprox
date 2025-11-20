import unittest
from typing import Tuple, Type

import numpy as np

from pyapprox.util.backends.template import Array, BackendMixin
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.optimization.adjoint import (
    ScalarAdjointFunctionalWithHessian,
    VectorAdjointFunctional,
    AdjointConstraintEquationWithHessian,
    ScalarAdjointOperator,
)


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


class TestAdjoint:
    def setUp(self):
        np.random.seed(1)

    def _setup_linear_least_squares(
        self, nobs: int, nvars: int, constraint_eq_type, mse_functional_type
    ):
        bkd = self.get_backend()
        Amat = bkd.asarray(np.random.normal(0, 1, (nobs, nvars)))
        generating_params = bkd.asarray(np.random.normal(0, 1, (nvars,)))
        obs = Amat @ generating_params
        constraint_eq = constraint_eq_type(Amat, obs, bkd)
        functional = mse_functional_type(nobs, nvars, bkd)
        functional.set_observations(obs)
        params = generating_params + 0.5
        init_state = bkd.zeros((nobs,))
        return constraint_eq, functional, params, init_state

    def _check_linear_least_squares(
        self,
        nobs: int,
        nvars: int,
        constraint_eq_type: Type[LinearConstraintEquation],
        mse_functional_type: Type[MSEFunctional],
    ):
        constraint_eq, functional, params, init_state = (
            self._setup_linear_least_squares(
                nobs, nvars, constraint_eq_type, mse_functional_type
            )
        )
        adjoint_op = ScalarAdjointOperator(constraint_eq, functional)

        # check first order derivatives of contraint equation
        errors = constraint_eq.check_state_jacobian(init_state, params)
        self.assertLessEqual(errors.min() / errors.max(), 1e-8)
        errors = constraint_eq.check_param_jacobian(init_state, params)
        self.assertLessEqual(errors.min() / errors.max(), 1e-8)

        # check first order derivatives of functional
        errors = functional.check_state_jacobian(init_state, params)
        self.assertLessEqual(errors.min() / errors.max(), 2e-7)
        # exact jac is zero so set relative to zero to avoid divide by zero
        errors = functional.check_param_jacobian(
            init_state, params, relative=False
        )
        self.assertEqual(errors.min(), errors.max())
        self.assertEqual(errors.min(), 0.0)

        # check Jacobian
        errors = adjoint_op.check_jacobian(init_state, params)
        self.assertLessEqual(errors.min() / errors.max(), 2e-7)

        # check second order derivatives of contraint equation
        adj_state = adjoint_op.adjoint_data().get_adjoint_state()
        errors = constraint_eq.check_param_param_hvp(
            init_state, params, adj_state, relative=False
        )
        self.assertEqual(errors.min(), errors.max())
        self.assertEqual(errors.min(), 0.0)
        errors = constraint_eq.check_state_state_hvp(
            init_state, params, adj_state, relative=False
        )
        self.assertEqual(errors.min(), errors.max())
        self.assertEqual(errors.min(), 0.0)
        errors = constraint_eq.check_param_state_hvp(
            init_state, params, adj_state, relative=False
        )
        self.assertEqual(errors.min(), errors.max())
        self.assertEqual(errors.min(), 0.0)

        errors = constraint_eq.check_state_param_hvp(
            init_state, params, adj_state, relative=False
        )
        self.assertEqual(errors.min(), errors.max())
        self.assertEqual(errors.min(), 0.0)

        # check second order derivaties of functional
        errors = functional.check_param_param_hvp(
            init_state, params, relative=False
        )
        self.assertEqual(errors.min(), errors.max())
        self.assertEqual(errors.min(), 0.0)
        errors = functional.check_state_state_hvp(
            init_state, params, relative=False, disp=True
        )
        self.assertLessEqual(errors.min() / errors.max(), 2e-7)
        errors = functional.check_state_param_hvp(
            init_state, params, relative=False
        )
        self.assertEqual(errors.min(), errors.max())
        self.assertEqual(errors.min(), 0.0)

        # check Hessian
        # errors = adjoint_op.check_apply_hessian(init_state, params)
        # self.assertLessEqual(errors.min() / errors.max(), 2e-7)

    def test_linear_least_squares(self):
        self._check_linear_least_squares(
            2, 3, LinearConstraintEquation, MSEFunctional
        )
        self._check_linear_least_squares(
            1, 4, LinearConstraintEquation, MSEFunctional
        )

    def test_vector_functional_jacobian(self):
        bkd = self.get_backend()
        raise NotImplementedError


class TestNumpyAdjoint(TestAdjoint, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TorchAutogradMSEFunctional(ScalarAdjointFunctionalWithHessian):
    """
    Call default autograd functions
    """

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


class TorchAutogradLinearConstraintEquation(
    AdjointConstraintEquationWithHessian
):
    """
    Call default autograd functions
    """

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


class TestTorchAdjoint(TestAdjoint, unittest.TestCase):
    def get_backend(self):
        return TorchMixin

    def test_linear_least_squares_with_autograd(self):
        self._check_linear_least_squares(
            2,
            3,
            TorchAutogradLinearConstraintEquation,
            TorchAutogradMSEFunctional,
        )
        self._check_linear_least_squares(
            1,
            4,
            TorchAutogradLinearConstraintEquation,
            TorchAutogradMSEFunctional,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
