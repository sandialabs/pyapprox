"""Tests for CFMLoss, UniformWeight, and FlowMatchingObjective."""

import pytest

from pyapprox.probability import GaussianMarginal, UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.generative.flowmatching.cfm_loss import (
    CFMLoss,
    FlowMatchingObjective,
    UniformWeight,
)
from pyapprox.generative.flowmatching.linear_path import LinearPath
from pyapprox.generative.flowmatching.protocols import (
    CFMLossProtocol,
    TimeWeightProtocol,
)
from pyapprox.generative.flowmatching.quad_data import (
    FlowMatchingQuadData,
)


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


class TestCFMLoss:
    def test_satisfies_protocol(self, bkd) -> None:
        loss = CFMLoss(bkd)
        assert isinstance(loss, CFMLossProtocol)

    def test_zero_loss_when_vf_matches_target(self, bkd) -> None:
        """When VF exactly equals the target field, loss should be zero."""
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)

        ns = 5
        t = bkd.array([[0.1, 0.3, 0.5, 0.7, 0.9]])
        x0 = bkd.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 1.5, 2.5, 3.5, 4.5]])
        x1 = bkd.array([[6.0, 7.0, 8.0, 9.0, 10.0], [5.5, 6.5, 7.5, 8.5, 9.5]])
        weights = bkd.ones_like(t[0, :]) / ns

        # Create a "perfect" VF that returns exactly x1 - x0
        u_t = path.target_field(t, x0, x1)

        def perfect_vf(vf_input):
            return u_t

        result = loss(perfect_vf, path, t, x0, x1, weights)
        bkd.assert_allclose(
            bkd.reshape(result, (1,)),
            bkd.array([0.0]),
            atol=1e-14,
        )

    @pytest.mark.slow_on("TorchBkd")
    def test_integrand_shape(self, bkd) -> None:
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        d, ns = 2, 4
        vf = _make_vf(bkd, d, degree=1)

        t = bkd.array([[0.1, 0.3, 0.5, 0.7]])
        x0 = bkd.zeros((d, ns))
        x1 = bkd.ones_like(x0)

        result = loss.integrand(vf, path, t, x0, x1)
        assert result.shape == (ns,)

    @pytest.mark.slow_on("TorchBkd")
    def test_integrand_shape_with_conditioning(self, bkd) -> None:
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        d, ns, m = 2, 4, 1
        vf = _make_vf(bkd, d, degree=1, m=m)

        t = bkd.array([[0.1, 0.3, 0.5, 0.7]])
        x0 = bkd.zeros((d, ns))
        x1 = bkd.ones_like(x0)
        c = bkd.zeros((m, ns))

        result = loss.integrand(vf, path, t, x0, x1, c=c)
        assert result.shape == (ns,)

    def test_loss_scalar_output(self, bkd) -> None:
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
        assert result.shape == ()

    def test_weight_accessor(self, bkd) -> None:
        loss = CFMLoss(bkd)
        w = loss.weight()
        assert isinstance(w, TimeWeightProtocol)


class TestFlowMatchingObjective:
    def _make_objective(self, bkd, d=1, m=0, degree=1):
        vf = _make_vf(bkd, d, degree, m)
        path = LinearPath(bkd)
        loss = CFMLoss(bkd)
        ns = 10
        qd = _make_quad_data(bkd, d, ns, m)
        return FlowMatchingObjective(vf, path, loss, qd, bkd), vf

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

        # Set some nonzero coefficients
        nparams = vf.hyp_list().nactive_params()
        params = bkd.array([0.1 * (i + 1) for i in range(nparams)])

        grad = obj.jacobian(params)

        # Finite difference
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
        loss = CFMLoss(bkd)
        ns = 10
        qd = _make_quad_data(bkd, d, ns)

        # Fix the first 2 parameters
        nparams_total = vf.hyp_list().nparams()
        active_idx = bkd.array(list(range(2, nparams_total)), dtype=int)
        vf.hyp_list().set_active_indices(active_idx)

        obj = FlowMatchingObjective(vf, path, loss, qd, bkd)
        nactive = vf.hyp_list().nactive_params()
        assert nactive < nparams_total

        params = bkd.array([0.1 * (i + 1) for i in range(nactive)])
        grad = obj.jacobian(params)
        assert grad.shape == (1, nactive)

        # FD check
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
