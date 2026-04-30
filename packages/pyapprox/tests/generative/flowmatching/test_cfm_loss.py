"""Tests for UniformWeight and FlowMatchingObjective."""

import pytest

from pyapprox.probability import GaussianMarginal, UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.generative.flowmatching.linear_path import LinearPath
from pyapprox.generative.flowmatching.objective import FlowMatchingObjective
from pyapprox.generative.flowmatching.protocols import (
    TimeWeightProtocol,
)
from pyapprox.generative.flowmatching.quad_data import (
    FlowMatchingQuadData,
)
from pyapprox.generative.flowmatching.time_weight import UniformWeight


def _make_vf(bkd, d: int, degree: int, m: int = 0):
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


class TestUniformWeight:
    def test_satisfies_protocol(self, bkd) -> None:
        w = UniformWeight(bkd)
        assert isinstance(w, TimeWeightProtocol)

    def test_returns_ones(self, bkd) -> None:
        w = UniformWeight(bkd)
        t = bkd.array([[0.0, 0.25, 0.5, 0.75, 1.0]])
        result = w(t)
        bkd.assert_allclose(result, bkd.ones_like(t), rtol=1e-12)

    def test_output_shape(self, bkd) -> None:
        w = UniformWeight(bkd)
        t = bkd.array([[0.1, 0.5, 0.9]])
        result = w(t)
        assert result.shape == (1, 3)


class TestFlowMatchingObjective:
    def _make_objective(self, bkd, d=1, m=0, degree=1):
        vf = _make_vf(bkd, d, degree, m)
        path = LinearPath(bkd)
        ns = 10
        qd = _make_quad_data(bkd, d, ns, m)
        return FlowMatchingObjective(vf, path, qd, bkd), vf

    def test_call_returns_1x1(self, bkd) -> None:
        obj, vf = self._make_objective(bkd)
        params = vf.hyp_list().get_active_values()
        result = obj(params)
        assert result.shape == (1, 1)

    def test_call_with_2d_params(self, bkd) -> None:
        """Test that (nactive, 1) shaped params work too."""
        obj, vf = self._make_objective(bkd)
        params = vf.hyp_list().get_active_values()
        params_2d = bkd.reshape(params, (-1, 1))
        result = obj(params_2d)
        assert result.shape == (1, 1)

    def test_jacobian_returns_1xnactive(self, bkd) -> None:
        obj, vf = self._make_objective(bkd)
        params = vf.hyp_list().get_active_values()
        grad = obj.jacobian(params)
        nactive = vf.hyp_list().nactive_params()
        assert grad.shape == (1, nactive)

    def test_jacobian_vs_finite_differences(self, bkd) -> None:
        """Verify analytical gradient matches finite differences."""
        obj, vf = self._make_objective(bkd, d=1, degree=2)

        nparams = vf.hyp_list().nactive_params()
        params = bkd.array([0.1 * (i + 1) for i in range(nparams)])

        grad = obj.jacobian(params)

        eps = 1e-6
        fd_grad = bkd.zeros((nparams,))
        for i in range(nparams):
            params_plus = bkd.array(
                [
                    p + (eps if j == i else 0.0)
                    for j, p in enumerate(bkd.to_numpy(params))
                ]
            )
            params_minus = bkd.array(
                [
                    p - (eps if j == i else 0.0)
                    for j, p in enumerate(bkd.to_numpy(params))
                ]
            )
            f_plus = obj(params_plus)
            f_minus = obj(params_minus)
            fd_val = (f_plus[0, 0] - f_minus[0, 0]) / (2.0 * eps)
            fd_grad = _set_element(bkd, fd_grad, i, fd_val)

        bkd.assert_allclose(grad[0, :], fd_grad, rtol=1e-5, atol=1e-8)

    @pytest.mark.slow_on("TorchBkd")
    def test_jacobian_vs_fd_multidim(self, bkd) -> None:
        """Verify gradient for d=2 VF."""
        obj, vf = self._make_objective(bkd, d=2, degree=1)

        nparams = vf.hyp_list().nactive_params()
        params = bkd.array([0.05 * (i + 1) for i in range(nparams)])

        grad = obj.jacobian(params)

        eps = 1e-6
        fd_grad = bkd.zeros((nparams,))
        for i in range(nparams):
            params_plus = bkd.array(
                [
                    p + (eps if j == i else 0.0)
                    for j, p in enumerate(bkd.to_numpy(params))
                ]
            )
            params_minus = bkd.array(
                [
                    p - (eps if j == i else 0.0)
                    for j, p in enumerate(bkd.to_numpy(params))
                ]
            )
            f_plus = obj(params_plus)
            f_minus = obj(params_minus)
            fd_val = (f_plus[0, 0] - f_minus[0, 0]) / (2.0 * eps)
            fd_grad = _set_element(bkd, fd_grad, i, fd_val)

        bkd.assert_allclose(grad[0, :], fd_grad, rtol=1e-5, atol=1e-8)

    def test_jacobian_with_partial_active_params(self, bkd) -> None:
        """Verify gradient with some params fixed (inactive)."""
        d, degree = 1, 2
        vf = _make_vf(bkd, d, degree)
        path = LinearPath(bkd)
        ns = 10
        qd = _make_quad_data(bkd, d, ns)

        nparams_total = vf.hyp_list().nparams()
        active_idx = bkd.array(list(range(2, nparams_total)), dtype=int)
        vf.hyp_list().set_active_indices(active_idx)

        obj = FlowMatchingObjective(vf, path, qd, bkd)
        nactive = vf.hyp_list().nactive_params()
        assert nactive < nparams_total

        params = bkd.array([0.1 * (i + 1) for i in range(nactive)])
        grad = obj.jacobian(params)
        assert grad.shape == (1, nactive)

        eps = 1e-6
        fd_grad = bkd.zeros((nactive,))
        for i in range(nactive):
            params_plus = bkd.array(
                [
                    p + (eps if j == i else 0.0)
                    for j, p in enumerate(bkd.to_numpy(params))
                ]
            )
            params_minus = bkd.array(
                [
                    p - (eps if j == i else 0.0)
                    for j, p in enumerate(bkd.to_numpy(params))
                ]
            )
            f_plus = obj(params_plus)
            f_minus = obj(params_minus)
            fd_val = (f_plus[0, 0] - f_minus[0, 0]) / (2.0 * eps)
            fd_grad = _set_element(bkd, fd_grad, i, fd_val)

        bkd.assert_allclose(grad[0, :], fd_grad, rtol=1e-5, atol=1e-8)

    def test_jacobian_with_conditioning(self, bkd) -> None:
        """Verify gradient with conditioning variables."""
        obj, vf = self._make_objective(bkd, d=1, m=1, degree=1)

        nparams = vf.hyp_list().nactive_params()
        params = bkd.array([0.1 * (i + 1) for i in range(nparams)])

        grad = obj.jacobian(params)
        assert grad.shape == (1, nparams)

        eps = 1e-6
        fd_grad = bkd.zeros((nparams,))
        for i in range(nparams):
            params_plus = bkd.array(
                [
                    p + (eps if j == i else 0.0)
                    for j, p in enumerate(bkd.to_numpy(params))
                ]
            )
            params_minus = bkd.array(
                [
                    p - (eps if j == i else 0.0)
                    for j, p in enumerate(bkd.to_numpy(params))
                ]
            )
            f_plus = obj(params_plus)
            f_minus = obj(params_minus)
            fd_val = (f_plus[0, 0] - f_minus[0, 0]) / (2.0 * eps)
            fd_grad = _set_element(bkd, fd_grad, i, fd_val)

        bkd.assert_allclose(grad[0, :], fd_grad, rtol=1e-5, atol=1e-8)

    def test_nvars_and_nqoi(self, bkd) -> None:
        obj, vf = self._make_objective(bkd)
        assert obj.nvars() == vf.hyp_list().nactive_params()
        assert obj.nqoi() == 1


def _set_element(bkd, arr, idx, val):
    """Backend-agnostic element assignment helper."""
    arr_np = bkd.to_numpy(arr).copy()
    arr_np[idx] = float(bkd.to_numpy(bkd.reshape(val, (1,)))[0])
    return bkd.array(arr_np.tolist())
