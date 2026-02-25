"""Tests for FlowODEResidual and integrate_flow."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.pde.time.explicit_steppers.forward_euler import (
    ForwardEulerResidual,
)
from pyapprox.pde.time.explicit_steppers.heun import HeunResidual
from pyapprox.pde.time.protocols.ode_residual import (
    ODEResidualProtocol,
)
from pyapprox.surrogates.flowmatching.ode_adapter import (
    FlowODEResidual,
    integrate_flow,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd


class TestFlowODEResidual(Generic[Array], unittest.TestCase):
    """Tests for FlowODEResidual."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_satisfies_protocol(self) -> None:
        bkd = self._bkd

        def dummy_vf(x):
            return bkd.zeros((1, x.shape[1]))

        res = FlowODEResidual(dummy_vf, bkd)
        self.assertIsInstance(res, ODEResidualProtocol)

    def test_constant_vf(self) -> None:
        """v(t,x) = c => dx/dt = c at any state."""
        bkd = self._bkd
        c_val = bkd.array([[2.0], [-1.0]])  # (2, 1)

        def const_vf(vf_input):
            vf_input.shape[1]
            return c_val * bkd.ones_like(vf_input[0:1, :])  # (2, ns)

        res = FlowODEResidual(const_vf, bkd)
        res.set_time(0.5)
        state = bkd.array([10.0, 20.0])
        result = res(state)
        bkd.assert_allclose(result, bkd.array([2.0, -1.0]), rtol=1e-12)

    def test_output_shape(self) -> None:
        bkd = self._bkd
        d = 3

        def vf(x):
            return bkd.zeros((d, x.shape[1]))

        res = FlowODEResidual(vf, bkd)
        res.set_time(0.0)
        state = bkd.zeros((d,))
        result = res(state)
        self.assertEqual(result.shape, (d,))

    def test_with_conditioning(self) -> None:
        """VF that uses conditioning: v(t,x,c) = c (broadcast)."""
        bkd = self._bkd

        def cond_vf(vf_input):
            # vf_input: (1 + d + m, 1)
            # Return last m rows as output (d=1, m=1: row index 2)
            d = 1
            c_part = vf_input[1 + d :, :]  # (m, 1)
            return c_part  # (1, 1) since d=m=1

        c_val = bkd.array([[3.0]])  # (1, 1)
        res = FlowODEResidual(cond_vf, bkd, conditioning=c_val)
        res.set_time(0.0)
        state = bkd.array([0.0])
        result = res(state)
        bkd.assert_allclose(result, bkd.array([3.0]), rtol=1e-12)

    def test_jacobian_returns_zeros(self) -> None:
        bkd = self._bkd
        d = 2

        def vf(x):
            return bkd.zeros((d, x.shape[1]))

        res = FlowODEResidual(vf, bkd)
        state = bkd.zeros((d,))
        jac = res.jacobian(state)
        bkd.assert_allclose(jac, bkd.zeros((d, d)), rtol=1e-12)

    def test_mass_matrix_is_identity(self) -> None:
        bkd = self._bkd

        def vf(x):
            return bkd.zeros((2, x.shape[1]))

        res = FlowODEResidual(vf, bkd)
        M = res.mass_matrix(2)
        bkd.assert_allclose(M, bkd.eye(2), rtol=1e-12)

    def test_apply_mass_matrix_is_identity(self) -> None:
        bkd = self._bkd

        def vf(x):
            return bkd.zeros((2, x.shape[1]))

        res = FlowODEResidual(vf, bkd)
        vec = bkd.array([1.0, 2.0])
        bkd.assert_allclose(res.apply_mass_matrix(vec), vec, rtol=1e-12)


class TestIntegrateFlow(Generic[Array], unittest.TestCase):
    """Tests for integrate_flow helper."""

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_constant_vf_euler_exact(self) -> None:
        """v=c => x(T) = x0 + c*T. Euler is exact for constant VF."""
        bkd = self._bkd
        c_val = bkd.array([[1.0], [-0.5]])  # (2, 1)

        def const_vf(vf_input):
            return c_val * bkd.ones_like(vf_input[0:1, :])  # (2, ns)

        x0 = bkd.array([[0.0, 1.0], [0.0, 2.0]])  # (2, 2)
        T = 1.0
        result = integrate_flow(
            const_vf,
            x0,
            0.0,
            T,
            n_steps=10,
            bkd=bkd,
            stepper_cls=ForwardEulerResidual,
        )

        # x(T) = x0 + c * T
        expected = bkd.array([[0.0 + 1.0, 1.0 + 1.0], [0.0 - 0.5, 2.0 - 0.5]])
        bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_constant_vf_with_conditioning(self) -> None:
        """v(t,x,c) returns c as the velocity."""
        bkd = self._bkd
        d, _m = 1, 1

        def cond_vf(vf_input):
            # vf_input: (1+d+m, ns) = (3, ns)
            return vf_input[1 + d :, :]  # c part, shape (1, ns)

        x0 = bkd.array([[0.0, 0.0]])  # (1, 2)
        c = bkd.array([[2.0, -1.0]])  # (1, 2)
        T = 1.0
        result = integrate_flow(
            cond_vf,
            x0,
            0.0,
            T,
            n_steps=10,
            bkd=bkd,
            c=c,
            stepper_cls=ForwardEulerResidual,
        )
        expected = bkd.array([[2.0, -1.0]])
        bkd.assert_allclose(result, expected, rtol=1e-10)

    def test_linear_vf_euler_convergence(self) -> None:
        """v=x => x(t)=x0*exp(t). Verify Euler O(h) convergence."""
        bkd = self._bkd

        def linear_vf(vf_input):
            # vf_input: (1+d, ns), return x part
            return vf_input[1:, :]

        x0 = bkd.array([[1.0]])  # (1, 1)
        T = 1.0
        exact = np.exp(1.0)

        errors = []
        steps_list = [10, 20, 40, 80]
        for ns in steps_list:
            result = integrate_flow(
                linear_vf,
                x0,
                0.0,
                T,
                n_steps=ns,
                bkd=bkd,
                stepper_cls=ForwardEulerResidual,
            )
            err = abs(float(bkd.to_numpy(result[0, 0])) - exact)
            errors.append(err)

        # Check O(h) convergence: error ratio should be ~2 for halving h
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            self.assertGreater(ratio, 1.8, f"Euler convergence failed: ratio={ratio}")

    def test_linear_vf_heun_convergence(self) -> None:
        """v=x => x(t)=x0*exp(t). Verify Heun O(h^2) convergence."""
        bkd = self._bkd

        def linear_vf(vf_input):
            return vf_input[1:, :]

        x0 = bkd.array([[1.0]])
        T = 1.0
        exact = np.exp(1.0)

        errors = []
        steps_list = [10, 20, 40, 80]
        for ns in steps_list:
            result = integrate_flow(
                linear_vf,
                x0,
                0.0,
                T,
                n_steps=ns,
                bkd=bkd,
                stepper_cls=HeunResidual,
            )
            err = abs(float(bkd.to_numpy(result[0, 0])) - exact)
            errors.append(err)

        # Check O(h^2) convergence: error ratio should be ~4 for halving h
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            self.assertGreater(ratio, 3.5, f"Heun convergence failed: ratio={ratio}")

    def test_output_shape(self) -> None:
        bkd = self._bkd
        d, nsamples = 3, 5

        def vf(x):
            return bkd.zeros((d, x.shape[1]))

        x0 = bkd.zeros((d, nsamples))
        result = integrate_flow(vf, x0, 0.0, 1.0, n_steps=5, bkd=bkd)
        self.assertEqual(result.shape, (d, nsamples))

    def test_batch_samples(self) -> None:
        """Multiple samples with different initial conditions."""
        bkd = self._bkd
        c_val = 1.0

        def const_vf(vf_input):
            return bkd.ones_like(vf_input[1:, :]) * c_val

        x0 = bkd.array([[0.0, 1.0, 2.0]])  # (1, 3)
        T = 0.5
        result = integrate_flow(
            const_vf,
            x0,
            0.0,
            T,
            n_steps=10,
            bkd=bkd,
            stepper_cls=ForwardEulerResidual,
        )
        expected = bkd.array([[0.5, 1.5, 2.5]])
        bkd.assert_allclose(result, expected, rtol=1e-10)


class TestFlowODEResidualNumpy(TestFlowODEResidual[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFlowODEResidualTorch(TestFlowODEResidual[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestIntegrateFlowNumpy(TestIntegrateFlow[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIntegrateFlowTorch(TestIntegrateFlow[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


from pyapprox.util.test_utils import load_tests  # noqa: F401
