"""Unit tests for VariablePoissonFixedHamiltonianSurrogate."""

import numpy as np
import pytest

from pyapprox.surrogates.dynamical_systems.protocols import (
    LearnedFunctionProtocol,
)
from pyapprox.surrogates.dynamical_systems.surrogates.variable_poisson_fixed_hamiltonian import (  # noqa: E501
    VariablePoissonFixedHamiltonianSurrogate,
)


def _make_surrogate(bkd, n_dynamic=3, n_aux=0, eta=None):
    def grad_H(samples):
        return samples[:n_dynamic, :]

    def grad_H_jac(samples):
        nsamples = samples.shape[1]
        eye = bkd.eye(n_dynamic)
        return bkd.tile(
            bkd.reshape(eye, (1, n_dynamic, n_dynamic)),
            (nsamples, 1, 1),
        )

    surrogate = VariablePoissonFixedHamiltonianSurrogate(
        grad_hamiltonian=grad_H,
        grad_hamiltonian_jacobian=grad_H_jac,
        n_dynamic=n_dynamic,
        bkd=bkd,
        n_aux=n_aux,
    )
    if eta is not None:
        surrogate.hyp_list().set_active_values(bkd.array(eta))
    return surrogate


class TestProtocolConformance:
    def test_isinstance_learned_function(self, bkd):
        surrogate = _make_surrogate(bkd)
        assert isinstance(surrogate, LearnedFunctionProtocol)

    def test_has_required_methods(self, bkd):
        surrogate = _make_surrogate(bkd)
        assert callable(surrogate.bkd)
        assert callable(surrogate.nvars)
        assert callable(surrogate.nqoi)
        assert callable(surrogate.hyp_list)
        assert callable(surrogate)
        assert callable(surrogate.jacobian_batch)
        assert callable(surrogate.jacobian_wrt_params)
        assert callable(surrogate.with_params)


class TestDimensions:
    def test_nvars_no_aux(self, bkd):
        surrogate = _make_surrogate(bkd, n_dynamic=4)
        assert surrogate.nvars() == 4

    def test_nvars_with_aux(self, bkd):
        surrogate = _make_surrogate(bkd, n_dynamic=4, n_aux=2)
        assert surrogate.nvars() == 6

    def test_nqoi_equals_n_dynamic(self, bkd):
        surrogate = _make_surrogate(bkd, n_dynamic=4)
        assert surrogate.nqoi() == 4
        assert surrogate.n_dynamic() == 4

    def test_n_skew_params(self, bkd):
        surrogate = _make_surrogate(bkd, n_dynamic=4)
        n_skew = 4 * 3 // 2
        assert surrogate.hyp_list().get_values().shape[0] == n_skew


class TestShapes:
    def test_call_output_shape(self, bkd):
        n_dynamic = 3
        eta = [1.0, -0.5, 0.3]
        surrogate = _make_surrogate(bkd, n_dynamic=n_dynamic, eta=eta)
        nsamples = 10
        samples = bkd.array(np.random.RandomState(0).randn(3, nsamples))
        result = surrogate(samples)
        assert result.shape == (3, nsamples)

    def test_jacobian_batch_shape(self, bkd):
        n_dynamic = 3
        eta = [1.0, -0.5, 0.3]
        surrogate = _make_surrogate(bkd, n_dynamic=n_dynamic, eta=eta)
        nsamples = 10
        samples = bkd.array(np.random.RandomState(0).randn(3, nsamples))
        jac = surrogate.jacobian_batch(samples)
        assert jac.shape == (nsamples, 3, 3)

    def test_jacobian_batch_shape_with_aux(self, bkd):
        n_dynamic, n_aux = 3, 1
        eta = [1.0, -0.5, 0.3]
        surrogate = _make_surrogate(
            bkd, n_dynamic=n_dynamic, n_aux=n_aux, eta=eta
        )
        nsamples = 10
        samples = bkd.array(np.random.RandomState(0).randn(4, nsamples))
        jac = surrogate.jacobian_batch(samples)
        assert jac.shape == (nsamples, 3, 4)

    def test_jacobian_wrt_params_shape(self, bkd):
        n_dynamic = 4
        surrogate = _make_surrogate(bkd, n_dynamic=n_dynamic)
        nsamples = 10
        samples = bkd.array(np.random.RandomState(0).randn(4, nsamples))
        jac_p = surrogate.jacobian_wrt_params(samples)
        n_skew = 4 * 3 // 2
        assert jac_p.shape == (nsamples, 4, n_skew)


class TestSkewSymmetry:
    def test_build_L_is_skew(self, bkd):
        eta = [1.0, -2.0, 0.5]
        surrogate = _make_surrogate(bkd, n_dynamic=3, eta=eta)
        L = surrogate._build_L()
        bkd.assert_allclose(
            L + bkd.transpose(L),
            bkd.zeros((3, 3)),
            atol=1e-15,
        )

    def test_build_L_entries(self, bkd):
        eta = [1.0, -2.0, 0.5]
        surrogate = _make_surrogate(bkd, n_dynamic=3, eta=eta)
        L = surrogate._build_L()
        expected = bkd.array([
            [0.0, 1.0, -2.0],
            [-1.0, 0.0, 0.5],
            [2.0, -0.5, 0.0],
        ])
        bkd.assert_allclose(L, expected, atol=1e-15)

    def test_zero_params_gives_zero_output(self, bkd):
        surrogate = _make_surrogate(bkd, n_dynamic=3)
        nsamples = 5
        samples = bkd.array(np.random.RandomState(0).randn(3, nsamples))
        result = surrogate(samples)
        bkd.assert_allclose(result, bkd.zeros((3, nsamples)), atol=1e-15)


class TestWithParams:
    def test_immutability(self, bkd):
        surrogate = _make_surrogate(bkd, n_dynamic=3)
        original = surrogate.hyp_list().get_active_values()

        new_eta = bkd.array([1.0, 2.0, 3.0])
        new_surrogate = surrogate.with_params(new_eta)

        bkd.assert_allclose(
            surrogate.hyp_list().get_active_values(),
            original,
        )
        assert new_surrogate is not surrogate

    def test_new_surrogate_uses_new_params(self, bkd):
        surrogate = _make_surrogate(bkd, n_dynamic=3)
        nsamples = 5
        samples = bkd.array(np.random.RandomState(0).randn(3, nsamples))

        new_eta = bkd.array([1.0, 2.0, 3.0])
        new_surrogate = surrogate.with_params(new_eta)

        out_old = surrogate(samples)
        out_new = new_surrogate(samples)
        diff = float(bkd.max(bkd.abs(out_new)))
        assert diff > 1e-10
        bkd.assert_allclose(out_old, bkd.zeros((3, nsamples)), atol=1e-15)


class TestValidation:
    def test_n_dynamic_one_raises(self, bkd):
        def grad_H(s):
            return s

        def grad_H_jac(s):
            return bkd.ones((s.shape[1], 1, 1))

        with pytest.raises(ValueError, match="n_dynamic must be >= 2"):
            VariablePoissonFixedHamiltonianSurrogate(
                grad_hamiltonian=grad_H,
                grad_hamiltonian_jacobian=grad_H_jac,
                n_dynamic=1,
                bkd=bkd,
            )
