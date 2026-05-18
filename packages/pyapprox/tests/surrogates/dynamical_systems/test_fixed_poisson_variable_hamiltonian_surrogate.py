"""Unit tests for FixedPoissonVariableHamiltonianSurrogate."""

import numpy as np
import pytest

from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.protocols import (
    LearnedFunctionProtocol,
)
from pyapprox.surrogates.dynamical_systems.surrogates.fixed_poisson_variable_hamiltonian import (  # noqa: E501
    FixedPoissonVariableHamiltonianSurrogate,
)


def _make_expansion(bkd, nvars, max_level=3, nqoi=1):
    marginals = [UniformMarginal(-3.0, 3.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    indices = indices[:, 1:]
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    expansion = BasisExpansion(basis, bkd, nqoi=nqoi)
    rng = np.random.RandomState(99)
    coef = bkd.array(rng.randn(expansion.nterms(), nqoi) * 0.1)
    expansion.set_coefficients(coef)
    return expansion


def _make_canonical_surrogate(bkd, nvars=2, max_level=3, n_params=0):
    has_time = False
    expansion = _make_expansion(bkd, nvars, max_level)
    return FixedPoissonVariableHamiltonianSurrogate.canonical(
        expansion, has_time_input=has_time, n_params=n_params
    )


def _make_skew_matrix(bkd, n):
    rng = np.random.RandomState(7)
    A = rng.randn(n, n)
    S = A - A.T
    return bkd.array(S)


class TestProtocolConformance:
    def test_isinstance_learned_function(self, bkd):
        surrogate = _make_canonical_surrogate(bkd)
        assert isinstance(surrogate, LearnedFunctionProtocol)

    def test_has_required_methods(self, bkd):
        surrogate = _make_canonical_surrogate(bkd)
        assert callable(surrogate.bkd)
        assert callable(surrogate.nvars)
        assert callable(surrogate.nqoi)
        assert callable(surrogate.hyp_list)
        assert callable(surrogate)
        assert callable(surrogate.jacobian_batch)
        assert callable(surrogate.jacobian_wrt_params)
        assert callable(surrogate.with_params)


class TestShapes:
    def test_call_output_shape(self, bkd):
        surrogate = _make_canonical_surrogate(bkd)
        nsamples = 15
        samples = bkd.array(np.random.RandomState(0).randn(2, nsamples))
        result = surrogate(samples)
        assert result.shape == (2, nsamples)

    def test_jacobian_batch_shape(self, bkd):
        surrogate = _make_canonical_surrogate(bkd)
        nsamples = 10
        samples = bkd.array(np.random.RandomState(0).randn(2, nsamples))
        jac = surrogate.jacobian_batch(samples)
        assert jac.shape == (nsamples, 2, 2)

    def test_jacobian_wrt_params_shape(self, bkd):
        surrogate = _make_canonical_surrogate(bkd)
        nsamples = 10
        samples = bkd.array(np.random.RandomState(0).randn(2, nsamples))
        jac_p = surrogate.jacobian_wrt_params(samples)
        nactive = surrogate.hyp_list().get_active_values().shape[0]
        assert jac_p.shape == (nsamples, 2, nactive)

    def test_basis_jacobian_batch_shape(self, bkd):
        surrogate = _make_canonical_surrogate(bkd)
        nsamples = 10
        samples = bkd.array(np.random.RandomState(0).randn(2, nsamples))
        basis_jac = surrogate.basis_jacobian_batch(samples)
        nterms = surrogate.hyp_list().get_values().shape[0]
        assert basis_jac.shape == (nsamples, nterms, 2)

    def test_parametric_jacobian_batch_shape(self, bkd):
        surrogate = _make_canonical_surrogate(bkd, nvars=3, n_params=1)
        nsamples = 10
        samples = bkd.array(np.random.RandomState(0).randn(3, nsamples))
        jac = surrogate.jacobian_batch(samples)
        assert jac.shape == (nsamples, 2, 3)


class TestDimensions:
    def test_nvars(self, bkd):
        surrogate = _make_canonical_surrogate(bkd)
        assert surrogate.nvars() == 2

    def test_nqoi_equals_n_dynamic(self, bkd):
        surrogate = _make_canonical_surrogate(bkd)
        assert surrogate.nqoi() == 2
        assert surrogate.n_dynamic() == 2

    def test_nvars_parametric(self, bkd):
        surrogate = _make_canonical_surrogate(bkd, nvars=3, n_params=1)
        assert surrogate.nvars() == 3
        assert surrogate.nqoi() == 2
        assert surrogate.n_dynamic() == 2


class TestWithParams:
    def test_immutability(self, bkd):
        surrogate = _make_canonical_surrogate(bkd)
        original_params = surrogate.hyp_list().get_active_values()

        nterms = original_params.shape[0]
        new_params = bkd.array(np.ones((nterms, 1)))
        new_surrogate = surrogate.with_params(new_params)

        bkd.assert_allclose(
            surrogate.hyp_list().get_active_values(),
            original_params,
        )
        assert new_surrogate is not surrogate

    def test_new_surrogate_uses_new_params(self, bkd):
        surrogate = _make_canonical_surrogate(bkd)
        nsamples = 5
        rng = np.random.RandomState(0)
        samples = bkd.array(rng.randn(2, nsamples))

        nterms = surrogate.hyp_list().get_active_values().shape[0]
        new_params = bkd.array(rng.randn(nterms, 1))
        new_surrogate = surrogate.with_params(new_params)

        out_old = surrogate(samples)
        out_new = new_surrogate(samples)
        diff = float(bkd.max(bkd.abs(out_old - out_new)))
        assert diff > 1e-10


class TestCanonicalFactory:
    def test_poisson_matrix_is_canonical_J(self, bkd):
        expansion = _make_expansion(bkd, nvars=4, max_level=2)
        surrogate = FixedPoissonVariableHamiltonianSurrogate.canonical(
            expansion
        )
        L = surrogate.poisson_matrix()
        n = 2
        expected = bkd.vstack([
            bkd.hstack([bkd.zeros((n, n)), bkd.eye(n)]),
            bkd.hstack([-bkd.eye(n), bkd.zeros((n, n))]),
        ])
        bkd.assert_allclose(L, expected, atol=1e-15)

    def test_odd_dim_raises(self, bkd):
        expansion = _make_expansion(bkd, nvars=3, max_level=2)
        with pytest.raises(ValueError, match="even n_dynamic"):
            FixedPoissonVariableHamiltonianSurrogate.canonical(expansion)

    def test_canonical_with_time_input(self, bkd):
        expansion = _make_expansion(bkd, nvars=3, max_level=2)
        surrogate = FixedPoissonVariableHamiltonianSurrogate.canonical(
            expansion, has_time_input=True
        )
        assert surrogate.nvars() == 3
        assert surrogate.n_dynamic() == 2


class TestCustomPoissonMatrix:
    def test_non_canonical_skew(self, bkd):
        expansion = _make_expansion(bkd, nvars=3, max_level=2)
        L = _make_skew_matrix(bkd, 3)
        surrogate = FixedPoissonVariableHamiltonianSurrogate(expansion, L)
        assert surrogate.n_dynamic() == 3
        bkd.assert_allclose(surrogate.poisson_matrix(), L)

    def test_poisson_matrix_accessor(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, max_level=2)
        L = _make_skew_matrix(bkd, 2)
        surrogate = FixedPoissonVariableHamiltonianSurrogate(expansion, L)
        bkd.assert_allclose(surrogate.poisson_matrix(), L)


class TestValidation:
    def test_nqoi_not_one_raises(self, bkd):
        marginals = [UniformMarginal(-3.0, 3.0, bkd) for _ in range(2)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(2, 2, 1.0, bkd)[:, 1:]
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        expansion = BasisExpansion(basis, bkd, nqoi=2)
        coef = bkd.array(np.random.RandomState(0).randn(expansion.nterms(), 2))
        expansion.set_coefficients(coef)

        L = _make_skew_matrix(bkd, 2)
        with pytest.raises(ValueError, match="nqoi=1"):
            FixedPoissonVariableHamiltonianSurrogate(expansion, L)

    def test_non_skew_symmetric_raises(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, max_level=2)
        L = bkd.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="skew-symmetric"):
            FixedPoissonVariableHamiltonianSurrogate(expansion, L)

    def test_non_square_poisson_raises(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, max_level=2)
        L = bkd.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="square"):
            FixedPoissonVariableHamiltonianSurrogate(expansion, L)

    def test_nvars_mismatch_raises(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, max_level=2)
        L = _make_skew_matrix(bkd, 3)
        with pytest.raises(ValueError, match="hamiltonian.nvars"):
            FixedPoissonVariableHamiltonianSurrogate(expansion, L)

    def test_1d_poisson_raises(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, max_level=2)
        L = bkd.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="2D"):
            FixedPoissonVariableHamiltonianSurrogate(expansion, L)


class TestSkewSymmetryPreserved:
    def test_L_plus_LT_is_zero(self, bkd):
        expansion = _make_expansion(bkd, nvars=4, max_level=2)
        surrogate = FixedPoissonVariableHamiltonianSurrogate.canonical(
            expansion
        )
        L = surrogate.poisson_matrix()
        bkd.assert_allclose(
            L + bkd.transpose(L),
            bkd.zeros((4, 4)),
            atol=1e-15,
        )
