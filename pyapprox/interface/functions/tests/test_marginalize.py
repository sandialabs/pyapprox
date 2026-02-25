"""Tests for function dimension reduction.

Tests FunctionMarginalizer (quadrature integration) and
CrossSectionReducer (fixing variables at nominal values).
Pure function tests only (no probability imports).
"""

import unittest
from typing import Any, Generic, List, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.interface.functions.marginalize import (
    CrossSectionReducer,
    FunctionMarginalizer,
)
from pyapprox.surrogates.affine.univariate.globalpoly.jacobi import (
    JacobiPolynomial1D,
)
from pyapprox.surrogates.affine.univariate.globalpoly.quadrature import (
    GaussQuadratureRule,
)
from pyapprox.surrogates.quadrature.protocols import (
    MultivariateQuadratureRuleProtocol,
)
from pyapprox.surrogates.quadrature.tensor_product import (
    TensorProductQuadratureRule,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401


class _AffineRule(Generic[Array]):
    """Wraps a [-1,1] tensor product rule with affine mapping to [lb, ub].

    Used only for testing. Satisfies MultivariateQuadratureRuleProtocol.
    """

    def __init__(
        self,
        tp_rule: TensorProductQuadratureRule[Array],
        lb: Array,
        ub: Array,
        bkd: Backend[Array],
    ):
        self._bkd = bkd
        raw_samples, raw_weights = tp_rule()
        nvars = raw_samples.shape[0]
        # Affine map each dimension: x = (ub-lb)/2 * t + (ub+lb)/2
        half_width = (ub - lb) / 2.0
        center = (ub + lb) / 2.0
        self._samples = half_width[:, None] * raw_samples + center[:, None]
        # Legendre weights integrate probability measure (1/2)dx on [-1,1],
        # so sum(w_i) = 1. For Lebesgue integral on [lb, ub]:
        # int_lb^ub f(x) dx = (ub-lb) * sum(w_i * f(map(t_i)))
        width = ub - lb
        weight_scale = bkd.prod(width)
        self._weights = raw_weights * weight_scale
        self._nvars = nvars

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return self._nvars

    def nsamples(self) -> int:
        return self._samples.shape[1]

    def __call__(self) -> Tuple[Array, Array]:
        return self._bkd.copy(self._samples), self._bkd.copy(self._weights)

    def integrate(self, func):
        values = func(self._samples)
        return self._bkd.sum(self._weights[:, None] * values, axis=0)


class _TestQuadratureFactory(Generic[Array]):
    """Test quadrature factory using Gauss-Legendre + affine mapping."""

    def __init__(
        self,
        npoints_1d: List[int],
        domain: Array,
        bkd: Backend[Array],
    ):
        self._npoints_1d = npoints_1d
        self._domain = domain
        self._bkd = bkd
        legendre = JacobiPolynomial1D(0.0, 0.0, bkd)
        self._quad_rule = GaussQuadratureRule(legendre)

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def __call__(
        self, integrate_indices: List[int]
    ) -> MultivariateQuadratureRuleProtocol[Array]:
        rules = [self._quad_rule] * len(integrate_indices)
        npts = [self._npoints_1d[i] for i in integrate_indices]
        tp = TensorProductQuadratureRule(self._bkd, rules, npts)
        lb = self._bkd.asarray([float(self._domain[i, 0]) for i in integrate_indices])
        ub = self._bkd.asarray([float(self._domain[i, 1]) for i in integrate_indices])
        return _AffineRule(tp, lb, ub, self._bkd)


class _PolynomialFunction3D(Generic[Array]):
    """f(x,y,z) = x*y*z. Satisfies FunctionProtocol."""

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 3

    def nqoi(self) -> int:
        return 1

    def __call__(self, samples: Array) -> Array:
        x, y, z = samples[0:1, :], samples[1:2, :], samples[2:3, :]
        return x * y * z


class _MultiQoIFunction2D(Generic[Array]):
    """f(x,y) = [[x*y], [x+y]]. nqoi=2."""

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        return self._bkd

    def nvars(self) -> int:
        return 2

    def nqoi(self) -> int:
        return 2

    def __call__(self, samples: Array) -> Array:
        x, y = samples[0:1, :], samples[1:2, :]
        return self._bkd.concatenate([x * y, x + y], axis=0)


class TestFunctionMarginalizer(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_marginalize_3d_to_2d(self) -> None:
        """f(x,y,z) = x*y*z on [0,1]^3, integrate out z -> 0.5*x*y."""
        bkd = self._bkd
        func = _PolynomialFunction3D(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _TestQuadratureFactory([5, 5, 5], domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)

        marg_2d = marginalizer.marginalize([0, 1])
        self.assertEqual(marg_2d.nvars(), 2)
        self.assertEqual(marg_2d.nqoi(), 1)

        test_pts = bkd.asarray([[0.2, 0.5, 0.8], [0.3, 0.7, 0.9]])
        result = marg_2d(test_pts)
        expected = 0.5 * test_pts[0:1, :] * test_pts[1:2, :]
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_marginalize_3d_to_1d(self) -> None:
        """f(x,y,z) = x*y*z on [0,1]^3, integrate out y,z -> 0.25*x."""
        bkd = self._bkd
        func = _PolynomialFunction3D(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _TestQuadratureFactory([5, 5, 5], domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)

        marg_1d = marginalizer.marginalize([0])
        self.assertEqual(marg_1d.nvars(), 1)
        self.assertEqual(marg_1d.nqoi(), 1)

        test_pts = bkd.asarray([[0.1, 0.3, 0.5, 0.7, 0.9]])
        result = marg_1d(test_pts)
        expected = 0.25 * test_pts
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_marginalize_non_contiguous_indices(self) -> None:
        """f(x,y,z) = x*y*z on [0,1]^3, keep [0,2] integrate out y -> 0.5*x*z."""
        bkd = self._bkd
        func = _PolynomialFunction3D(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _TestQuadratureFactory([5, 5, 5], domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)

        marg_xz = marginalizer.marginalize([0, 2])
        self.assertEqual(marg_xz.nvars(), 2)

        # Input: (2, nsamples) where dim 0 = x, dim 1 = z
        test_pts = bkd.asarray([[0.2, 0.6], [0.4, 0.8]])
        result = marg_xz(test_pts)
        expected = 0.5 * test_pts[0:1, :] * test_pts[1:2, :]
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_marginalize_multi_qoi(self) -> None:
        """f(x,y) = [[x*y], [x+y]] on [0,1]^2, integrate out y."""
        bkd = self._bkd
        func = _MultiQoIFunction2D(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0]])
        factory = _TestQuadratureFactory([5, 5], domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)

        marg_1d = marginalizer.marginalize([0])
        self.assertEqual(marg_1d.nvars(), 1)
        self.assertEqual(marg_1d.nqoi(), 2)

        test_pts = bkd.asarray([[0.2, 0.5, 0.8]])
        result = marg_1d(test_pts)
        # int_0^1 x*y dy = 0.5*x, int_0^1 (x+y) dy = x + 0.5
        expected_q0 = 0.5 * test_pts
        expected_q1 = test_pts + 0.5
        expected = bkd.concatenate([expected_q0, expected_q1], axis=0)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_keep_all_variables(self) -> None:
        """Marginalizing with keep_indices = all recovers original."""
        bkd = self._bkd
        func = _PolynomialFunction3D(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _TestQuadratureFactory([5, 5, 5], domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)

        marg_all = marginalizer.marginalize([0, 1, 2])
        self.assertEqual(marg_all.nvars(), 3)

        test_pts = bkd.asarray([[0.2, 0.5], [0.3, 0.7], [0.4, 0.9]])
        result = marg_all(test_pts)
        expected = func(test_pts)
        bkd.assert_allclose(result, expected, rtol=1e-14)

    def test_output_shape(self) -> None:
        """Verify output shapes for various nsamples."""
        bkd = self._bkd
        func = _PolynomialFunction3D(bkd)
        domain = bkd.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        factory = _TestQuadratureFactory([3, 3, 3], domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)

        for nsamples in [1, 5, 10]:
            marg_1d = marginalizer.marginalize([1])
            test_pts = bkd.asarray(np.random.uniform(0, 1, (1, nsamples)))
            result = marg_1d(test_pts)
            self.assertEqual(result.shape, (1, nsamples))

    def test_non_unit_domain(self) -> None:
        """Test marginalization on non-[0,1] domain."""
        bkd = self._bkd
        func = _PolynomialFunction3D(bkd)
        domain = bkd.asarray([[1.0, 3.0], [2.0, 4.0], [0.0, 1.0]])
        factory = _TestQuadratureFactory([5, 5, 5], domain, bkd)
        marginalizer = FunctionMarginalizer(func, factory, bkd)

        marg_1d = marginalizer.marginalize([0])
        # int_{2}^{4} int_{0}^{1} x*y*z dz dy = x * (4^2-2^2)/2 * 0.5 = x * 3.0
        test_pts = bkd.asarray([[1.5, 2.0, 2.5]])
        result = marg_1d(test_pts)
        expected = 3.0 * test_pts
        bkd.assert_allclose(result, expected, rtol=1e-12)


class TestCrossSectionReducer(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()

    def test_cross_section_3d_to_1d(self) -> None:
        """f(x,y,z)=x*y*z, fix y=0.4 z=0.6 -> f(x)=0.24*x."""
        bkd = self._bkd
        func = _PolynomialFunction3D(bkd)
        nominal = bkd.asarray([0.0, 0.4, 0.6])
        reducer = CrossSectionReducer(func, nominal, bkd)

        reduced = reducer.reduce([0])
        self.assertEqual(reduced.nvars(), 1)
        self.assertEqual(reduced.nqoi(), 1)

        test_pts = bkd.asarray([[0.1, 0.5, 0.9]])
        result = reduced(test_pts)
        expected = 0.24 * test_pts
        bkd.assert_allclose(result, expected, rtol=1e-14)

    def test_cross_section_3d_to_2d(self) -> None:
        """f(x,y,z)=x*y*z, fix z=0.5 -> f(x,y)=0.5*x*y."""
        bkd = self._bkd
        func = _PolynomialFunction3D(bkd)
        nominal = bkd.asarray([0.0, 0.0, 0.5])
        reducer = CrossSectionReducer(func, nominal, bkd)

        reduced = reducer.reduce([0, 1])
        self.assertEqual(reduced.nvars(), 2)

        test_pts = bkd.asarray([[0.2, 0.8], [0.3, 0.7]])
        result = reduced(test_pts)
        expected = 0.5 * test_pts[0:1] * test_pts[1:2]
        bkd.assert_allclose(result, expected, rtol=1e-14)

    def test_cross_section_non_contiguous(self) -> None:
        """f(x,y,z)=x*y*z, keep [0,2] fix y=0.3 -> f(x,z)=0.3*x*z."""
        bkd = self._bkd
        func = _PolynomialFunction3D(bkd)
        nominal = bkd.asarray([0.0, 0.3, 0.0])
        reducer = CrossSectionReducer(func, nominal, bkd)

        reduced = reducer.reduce([0, 2])
        test_pts = bkd.asarray([[0.4, 0.6], [0.5, 0.8]])
        result = reduced(test_pts)
        expected = 0.3 * test_pts[0:1] * test_pts[1:2]
        bkd.assert_allclose(result, expected, rtol=1e-14)

    def test_cross_section_keep_all(self) -> None:
        """Keeping all variables recovers original function."""
        bkd = self._bkd
        func = _PolynomialFunction3D(bkd)
        nominal = bkd.asarray([0.5, 0.5, 0.5])
        reducer = CrossSectionReducer(func, nominal, bkd)

        reduced = reducer.reduce([0, 1, 2])
        test_pts = bkd.asarray([[0.2, 0.5], [0.3, 0.7], [0.4, 0.9]])
        result = reduced(test_pts)
        expected = func(test_pts)
        bkd.assert_allclose(result, expected, rtol=1e-14)

    def test_cross_section_multi_qoi(self) -> None:
        """f(x,y) = [[x*y], [x+y]], fix y=0.6 -> [[0.6*x], [x+0.6]]."""
        bkd = self._bkd
        func = _MultiQoIFunction2D(bkd)
        nominal = bkd.asarray([0.0, 0.6])
        reducer = CrossSectionReducer(func, nominal, bkd)

        reduced = reducer.reduce([0])
        self.assertEqual(reduced.nvars(), 1)
        self.assertEqual(reduced.nqoi(), 2)

        test_pts = bkd.asarray([[0.2, 0.5, 0.8]])
        result = reduced(test_pts)
        expected_q0 = 0.6 * test_pts
        expected_q1 = test_pts + 0.6
        expected = bkd.concatenate([expected_q0, expected_q1], axis=0)
        bkd.assert_allclose(result, expected, rtol=1e-14)


class TestFunctionMarginalizerNumpy(TestFunctionMarginalizer[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestFunctionMarginalizerTorch(TestFunctionMarginalizer[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


class TestCrossSectionReducerNumpy(TestCrossSectionReducer[NDArray[Any]]):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestCrossSectionReducerTorch(TestCrossSectionReducer[torch.Tensor]):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()
