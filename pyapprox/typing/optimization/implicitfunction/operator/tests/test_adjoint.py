import unittest
from typing import Type

import numpy as np

from pyapprox.util.backends.template import Array, BackendMixin
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.optimization.adjoint import (
    ScalarAdjointFunctionalWithHessian,
    AdjointResidualEquationWithHessian,
    ScalarAdjointOperator,
    ScalarAdjointOperatorWithHessian,
    VectorAdjointOperator,
)
from pyapprox.optimization.implicitfunction.benchmarks import (
    LinearStateEquations,
    NonLinearCoupledStateEquations,
)
from pyapprox.optimization.functionals import (
    MSEFunctional,
    TikhinovMSEFunctional,
    WeightedSumFunctional,
    SubsetOfStatesAdjointFunctional,
)


class TestAdjoint:
    def get_backend(self):
        raise NotImplementedError

    def setUp(self):
        np.random.seed(1)

    def test_nonlinear_coupled_residual_scalar_functional(self):
        bkd = self.get_backend()
        res = NonLinearCoupledEquationsResidual(bkd)
        res._apow = 2
        res._bpow = 3
        sample = bkd.array([0.8, 1.1])
        init_iterate = bkd.array([-1.0, -1.0])
        sol = res.solve(init_iterate, sample)

        a, b = sample
        exact_sol = bkd.array(
            [
                -bkd.sqrt((b + 1) * (b**2 - b + 1) / (a**2 * b**3 + 1)),
                -bkd.sqrt(-(a - 1) * (a + 1) / (a**2 * b**3 + 1)),
            ]
        )
        bkd.assert_allclose(sol, exact_sol)

        weights = bkd.ones((2,))
        functional = WeightedSumFunctional(weights, 2, bkd)
        adjoint_op = ScalarAdjointOperator(res, functional)
        tols = adjoint_op.get_derivative_tolerances(1e-6)
        # need to reduce fd_eps so newton converges for the largest
        # finite different steps sizes
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        adjoint_op.check_derivatives(init_iterate, sample, tols, fd_eps)

    def _setup_linear_least_squares(
        self, nobs: int, nvars: int, res_type, mse_functional_type
    ):
        bkd = self.get_backend()
        Amat = bkd.asarray(np.random.normal(0, 1, (nobs, nvars)))
        generating_param = bkd.asarray(np.random.normal(0, 1, (nvars,)))
        obs = Amat @ generating_param
        res = res_type(Amat, obs, bkd)
        functional = mse_functional_type(nobs, nvars, bkd)
        functional.set_observations(obs)
        param = generating_param + 0.5
        init_state = bkd.zeros((nobs,))
        return res, functional, param, init_state

    def _check_linear_least_squares(
        self,
        nobs: int,
        nvars: int,
        res_type: Type[LinearResidualEquation],
        mse_functional_type: Type[MSEFunctional],
    ):
        # check constraint and functional derivatives for
        # linear least squares loss
        res, functional, param, init_state = self._setup_linear_least_squares(
            nobs, nvars, res_type, mse_functional_type
        )
        adjoint_op = ScalarAdjointOperatorWithHessian(res, functional)
        tols = adjoint_op.get_derivative_tolerances(1e-8)
        tols[[2, 4]] = 2.0e-7
        adjoint_op.check_derivatives(init_state, param, tols)

    def test_linear_least_squares(self):
        self._check_linear_least_squares(
            2, 3, LinearResidualEquation, MSEFunctional
        )
        self._check_linear_least_squares(
            3, 2, LinearResidualEquation, MSEFunctional
        )

    def test_tikhinov_linear_least_squares(
        self,
    ):
        nobs, nvars = 3, 2
        res, functional, param, init_state = self._setup_linear_least_squares(
            nobs, nvars, LinearResidualEquation, TikhinovMSEFunctional
        )
        adjoint_op = ScalarAdjointOperator(res, functional)
        tols = adjoint_op.get_derivative_tolerances(1e-8)
        tols[[2, 3, 4]] = 2.0e-7
        adjoint_op.check_derivatives(init_state, param, tols)

    def test_modified_tikhinov_least_squares(self):
        # modify functional to have non zero state_param and param_state hvp
        # define here, instead of in registry,
        # because it is not something that people will use.
        class ModifiedTikhinovFunctional(TikhinovMSEFunctional):
            def _Bmat(self):
                Bmat = self._bkd.zeros((self.nvars(), self.nstates()))
                Bmat[0, 0] = 1.0
                Bmat[1, 1] = 1.0
                return Bmat

            def _value(self, state: Array, param: Array) -> Array:
                return (
                    super()._value(state, param)
                    + self._bkd.sum((param - self._Bmat() @ state) ** 2) / 2.0
                )

            def _state_jacobian(self, state: Array, param: Array) -> Array:
                return (
                    state
                    - self._obs
                    - self._Bmat().T @ (param - self._Bmat() @ state)
                )[None, :]

            def _param_jacobian(self, state: Array, param: Array) -> Array:
                return (2.0 * param - self._Bmat() @ state)[None, :]

            def _param_param_hvp(
                self, state: Array, param: Array, vvec: Array
            ) -> Array:
                return 2.0 * vvec

            def _state_state_hvp(
                self, state: Array, param: Array, vvec: Array
            ) -> Array:
                Bmat = self._Bmat()
                return Bmat.T @ (Bmat @ vvec) + vvec

            def _param_state_hvp(
                self, state: Array, param: Array, vvec: Array
            ) -> Array:
                return -self._Bmat() @ vvec

            def _state_param_hvp(
                self, state: Array, param: Array, vvec: Array
            ) -> Array:
                return -self._Bmat().T @ vvec

        nobs, nvars = 3, 2
        res, tikhinov_functional, param, init_state = (
            self._setup_linear_least_squares(
                nobs, nvars, LinearResidualEquation, TikhinovMSEFunctional
            )
        )
        functional = ModifiedTikhinovFunctional(
            nobs, nvars, self.get_backend()
        )
        functional.set_observations(tikhinov_functional._obs)
        adjoint_op = ScalarAdjointOperator(res, functional)
        tols = adjoint_op.get_derivative_tolerances(1e-8)
        tols[[2, 3, 4]] = 2.0e-7
        adjoint_op.check_derivatives(init_state, param, tols)

    def test_nonlinear_coupled_residual_vector_functional(self):
        bkd = self.get_backend()
        res = NonLinearCoupledEquationsResidual(bkd)
        sample = bkd.array([0.8, 1.1])
        init_state = bkd.array([-1.0, -1.0])
        functional = SubsetVectorAdjointFunctional(2, 2, bkd.arange(2), bkd)
        adjoint_op = VectorAdjointOperator(res, functional)
        # need to reduce fd_eps so newton converges for the largest
        # finite different steps sizes
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        tols = adjoint_op.get_derivative_tolerances(3e-7)
        adjoint_op.check_derivatives(
            init_state, sample, tols, fd_eps=fd_eps, disp=False
        )


class TestNumpyAdjoint(TestAdjoint, unittest.TestCase):
    def get_backend(self) -> Type[NumpyMixin]:
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

    def _value(self, state: Array, param: Array) -> Array:
        return self._bkd.sum((self._obs - state) ** 2) / 2.0

    def use_auto_differentiation(self) -> bool:
        return True


class TorchAutogradLinearResidualEquation(AdjointResidualEquationWithHessian):
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

    def _solve(self, init_state: Array):
        # init_state is ignored for this linear problem
        return self._Amat @ self._param

    def _set_parameters(self, param: Array) -> None:
        pass

    def use_auto_differentiation(self) -> bool:
        return True


class TestTorchAdjoint(TestAdjoint, unittest.TestCase):
    def get_backend(self) -> Type[TorchMixin]:
        return TorchMixin

    def test_linear_least_squares_with_autograd(self):
        self._check_linear_least_squares(
            2,
            3,
            TorchAutogradLinearResidualEquation,
            TorchAutogradMSEFunctional,
        )
        self._check_linear_least_squares(
            1,
            4,
            TorchAutogradLinearResidualEquation,
            TorchAutogradMSEFunctional,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
