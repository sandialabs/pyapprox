"""Tests for CFMLoss, UniformWeight, and FlowMatchingObjective."""

import unittest
from typing import Any, Generic

import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd

from pyapprox.probability import UniformMarginal, GaussianMarginal
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion

from pyapprox.surrogates.flowmatching.linear_path import LinearPath
from pyapprox.surrogates.flowmatching.cfm_loss import (
    CFMLoss,
    FlowMatchingObjective,
    UniformWeight,
)
from pyapprox.surrogates.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.surrogates.flowmatching.protocols import (
    TimeWeightProtocol,
    CFMLossProtocol,
)


def _make_vf(bkd: Backend[Array], d: int, degree: int, m: int = 0):
    """Create a BasisExpansion VF with input_dim = 1+d+m, nqoi = d."""
    marginals = [UniformMarginal(0.0, 1.0, bkd)]
    marginals += [GaussianMarginal(0.0, 1.0, bkd)] * d
    marginals += [GaussianMarginal(0.0, 1.0, bkd)] * m
    bases_1d = create_bases_1d(marginals, bkd)
    nvars = 1 + d + m
    indices = compute_hyperbolic_indices(nvars, degree, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=d)


def _make_quad_data(bkd, d, ns, m=0):
    """Create simple FlowMatchingQuadData for testing."""
    t = bkd.array([[i / (ns - 1) for i in range(ns)]])  # (1, ns)
    x0 = bkd.zeros((d, ns))
    x1 = bkd.ones_like(x0)
    weights = bkd.ones_like(t[0, :]) / ns  # (ns,)
    c = bkd.zeros((m, ns)) if m > 0 else None
    return FlowMatchingQuadData(t, x0, x1, weights, bkd, c)


class TestUniformWeight(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_satisfies_protocol(self) -> None:
        w = UniformWeight(self._bkd)
        self.assertIsInstance(w, TimeWeightProtocol)

    def test_returns_ones(self) -> None:
        w = UniformWeight(self._bkd)
        t = self._bkd.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
        result = w(t)
        self._bkd.assert_allclose(result, self._bkd.ones_like(t), rtol=1e-12)

    def test_output_shape(self) -> None:
        w = UniformWeight(self._bkd)
        t = self._bkd.array([[0.1, 0.5, 0.9]])
        result = w(t)
        self.assertEqual(result.shape, (1, 3))


class TestCFMLoss(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_satisfies_protocol(self) -> None:
        loss = CFMLoss(self._bkd)
        self.assertIsInstance(loss, CFMLossProtocol)

    def test_zero_loss_when_vf_matches_target(self) -> None:
        """When VF exactly equals the target field, loss should be zero."""
        bkd = self._bkd
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)

        d = 2
        ns = 5
        t = bkd.array([[0.1, 0.3, 0.5, 0.7, 0.9]])
        x0 = bkd.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                         [0.5, 1.5, 2.5, 3.5, 4.5]])
        x1 = bkd.array([[6.0, 7.0, 8.0, 9.0, 10.0],
                         [5.5, 6.5, 7.5, 8.5, 9.5]])
        weights = bkd.ones_like(t[0, :]) / ns

        # Create a "perfect" VF that returns exactly x1 - x0
        u_t = path.target_field(t, x0, x1)

        def perfect_vf(vf_input):
            return u_t

        result = loss(perfect_vf, path, t, x0, x1, weights)
        self._bkd.assert_allclose(
            bkd.reshape(result, (1,)),
            bkd.array([0.0]),
            atol=1e-14,
        )

    def test_integrand_shape(self) -> None:
        bkd = self._bkd
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        d, ns = 2, 4
        vf = _make_vf(bkd, d, degree=1)

        t = bkd.array([[0.1, 0.3, 0.5, 0.7]])
        x0 = bkd.zeros((d, ns))
        x1 = bkd.ones_like(x0)

        result = loss.integrand(vf, path, t, x0, x1)
        self.assertEqual(result.shape, (ns,))

    def test_integrand_shape_with_conditioning(self) -> None:
        bkd = self._bkd
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        d, ns, m = 2, 4, 1
        vf = _make_vf(bkd, d, degree=1, m=m)

        t = bkd.array([[0.1, 0.3, 0.5, 0.7]])
        x0 = bkd.zeros((d, ns))
        x1 = bkd.ones_like(x0)
        c = bkd.zeros((m, ns))

        result = loss.integrand(vf, path, t, x0, x1, c=c)
        self.assertEqual(result.shape, (ns,))

    def test_loss_scalar_output(self) -> None:
        bkd = self._bkd
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        d, ns = 1, 3
        vf = _make_vf(bkd, d, degree=1)

        t = bkd.array([[0.2, 0.5, 0.8]])
        x0 = bkd.zeros((d, ns))
        x1 = bkd.ones_like(x0)
        weights = bkd.ones_like(t[0, :]) / ns

        result = loss(vf, path, t, x0, x1, weights)
        # Should be a scalar (0-d or 1-element)
        self.assertEqual(result.shape, ())

    def test_weight_accessor(self) -> None:
        bkd = self._bkd
        loss = CFMLoss(bkd)
        w = loss.weight()
        self.assertIsInstance(w, TimeWeightProtocol)


class TestFlowMatchingObjective(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def _make_objective(self, d=1, m=0, degree=1):
        bkd = self._bkd
        vf = _make_vf(bkd, d, degree, m)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        ns = 10
        qd = _make_quad_data(bkd, d, ns, m)
        return FlowMatchingObjective(vf, path, loss, qd, bkd), vf

    def test_call_returns_1x1(self) -> None:
        obj, vf = self._make_objective()
        params = vf.hyp_list().get_active_values()
        result = obj(params)
        self.assertEqual(result.shape, (1, 1))

    def test_call_with_2d_params(self) -> None:
        """Test that (nactive, 1) shaped params work too."""
        bkd = self._bkd
        obj, vf = self._make_objective()
        params = vf.hyp_list().get_active_values()
        params_2d = bkd.reshape(params, (-1, 1))
        result = obj(params_2d)
        self.assertEqual(result.shape, (1, 1))

    def test_jacobian_returns_1xnactive(self) -> None:
        obj, vf = self._make_objective()
        params = vf.hyp_list().get_active_values()
        grad = obj.jacobian(params)
        nactive = vf.hyp_list().nactive_params()
        self.assertEqual(grad.shape, (1, nactive))

    def test_jacobian_vs_finite_differences(self) -> None:
        """Verify analytical gradient matches finite differences."""
        bkd = self._bkd
        obj, vf = self._make_objective(d=1, degree=2)

        # Set some nonzero coefficients
        nparams = vf.hyp_list().nactive_params()
        params = bkd.array([0.1 * (i + 1) for i in range(nparams)])

        grad = obj.jacobian(params)

        # Finite difference
        eps = 1e-6
        fd_grad = bkd.zeros((nparams,))
        for i in range(nparams):
            params_plus = bkd.array(
                [p + (eps if j == i else 0.0)
                 for j, p in enumerate(bkd.to_numpy(params))]
            )
            params_minus = bkd.array(
                [p - (eps if j == i else 0.0)
                 for j, p in enumerate(bkd.to_numpy(params))]
            )
            f_plus = obj(params_plus)
            f_minus = obj(params_minus)
            fd_val = (f_plus[0, 0] - f_minus[0, 0]) / (2.0 * eps)
            fd_grad = _set_element(bkd, fd_grad, i, fd_val)

        self._bkd.assert_allclose(
            grad[0, :], fd_grad, rtol=1e-5, atol=1e-8
        )

    def test_jacobian_vs_fd_multidim(self) -> None:
        """Verify gradient for d=2 VF."""
        bkd = self._bkd
        obj, vf = self._make_objective(d=2, degree=1)

        nparams = vf.hyp_list().nactive_params()
        params = bkd.array([0.05 * (i + 1) for i in range(nparams)])

        grad = obj.jacobian(params)

        eps = 1e-6
        fd_grad = bkd.zeros((nparams,))
        for i in range(nparams):
            params_plus = bkd.array(
                [p + (eps if j == i else 0.0)
                 for j, p in enumerate(bkd.to_numpy(params))]
            )
            params_minus = bkd.array(
                [p - (eps if j == i else 0.0)
                 for j, p in enumerate(bkd.to_numpy(params))]
            )
            f_plus = obj(params_plus)
            f_minus = obj(params_minus)
            fd_val = (f_plus[0, 0] - f_minus[0, 0]) / (2.0 * eps)
            fd_grad = _set_element(bkd, fd_grad, i, fd_val)

        self._bkd.assert_allclose(
            grad[0, :], fd_grad, rtol=1e-5, atol=1e-8
        )

    def test_jacobian_with_partial_active_params(self) -> None:
        """Verify gradient with some params fixed (inactive)."""
        bkd = self._bkd
        d, degree = 1, 2
        vf = _make_vf(bkd, d, degree)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        ns = 10
        qd = _make_quad_data(bkd, d, ns)

        # Fix the first 2 parameters
        nparams_total = vf.hyp_list().nparams()
        active_idx = bkd.array(
            list(range(2, nparams_total)), dtype=int
        )
        vf.hyp_list().set_active_indices(active_idx)

        obj = FlowMatchingObjective(vf, path, loss, qd, bkd)
        nactive = vf.hyp_list().nactive_params()
        self.assertLess(nactive, nparams_total)

        params = bkd.array([0.1 * (i + 1) for i in range(nactive)])
        grad = obj.jacobian(params)
        self.assertEqual(grad.shape, (1, nactive))

        # FD check
        eps = 1e-6
        fd_grad = bkd.zeros((nactive,))
        for i in range(nactive):
            params_plus = bkd.array(
                [p + (eps if j == i else 0.0)
                 for j, p in enumerate(bkd.to_numpy(params))]
            )
            params_minus = bkd.array(
                [p - (eps if j == i else 0.0)
                 for j, p in enumerate(bkd.to_numpy(params))]
            )
            f_plus = obj(params_plus)
            f_minus = obj(params_minus)
            fd_val = (f_plus[0, 0] - f_minus[0, 0]) / (2.0 * eps)
            fd_grad = _set_element(bkd, fd_grad, i, fd_val)

        self._bkd.assert_allclose(
            grad[0, :], fd_grad, rtol=1e-5, atol=1e-8
        )

    def test_jacobian_with_conditioning(self) -> None:
        """Verify gradient with conditioning variables."""
        bkd = self._bkd
        obj, vf = self._make_objective(d=1, m=1, degree=1)

        nparams = vf.hyp_list().nactive_params()
        params = bkd.array([0.1 * (i + 1) for i in range(nparams)])

        grad = obj.jacobian(params)
        self.assertEqual(grad.shape, (1, nparams))

        eps = 1e-6
        fd_grad = bkd.zeros((nparams,))
        for i in range(nparams):
            params_plus = bkd.array(
                [p + (eps if j == i else 0.0)
                 for j, p in enumerate(bkd.to_numpy(params))]
            )
            params_minus = bkd.array(
                [p - (eps if j == i else 0.0)
                 for j, p in enumerate(bkd.to_numpy(params))]
            )
            f_plus = obj(params_plus)
            f_minus = obj(params_minus)
            fd_val = (f_plus[0, 0] - f_minus[0, 0]) / (2.0 * eps)
            fd_grad = _set_element(bkd, fd_grad, i, fd_val)

        self._bkd.assert_allclose(
            grad[0, :], fd_grad, rtol=1e-5, atol=1e-8
        )

    def test_nvars_and_nqoi(self) -> None:
        obj, vf = self._make_objective()
        self.assertEqual(obj.nvars(), vf.hyp_list().nactive_params())
        self.assertEqual(obj.nqoi(), 1)


def _set_element(bkd, arr, idx, val):
    """Backend-agnostic element assignment helper."""
    import numpy as np
    arr_np = bkd.to_numpy(arr).copy()
    arr_np[idx] = float(bkd.to_numpy(bkd.reshape(val, (1,)))[0])
    return bkd.array(arr_np.tolist())


class TestUniformWeightNumpy(TestUniformWeight[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestUniformWeightTorch(TestUniformWeight[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestCFMLossNumpy(TestCFMLoss[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCFMLossTorch(TestCFMLoss[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestFlowMatchingObjectiveNumpy(TestFlowMatchingObjective[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFlowMatchingObjectiveTorch(TestFlowMatchingObjective[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


from pyapprox.util.test_utils import load_tests  # noqa: F401
