"""Tests for BasisExpansionVectorField and AdditiveVectorField."""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.protocols import (
    ParametricVectorFieldProtocol,
)
from pyapprox.surrogates.dynamical_systems.vector_fields import (
    AdditiveVectorField,
    BasisExpansionVectorField,
)


def _make_expansion(bkd, nvars, max_level, nqoi):
    marginals = [UniformMarginal(-2.0, 2.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=nqoi)


class _VectorFieldAsFunction:
    """Wraps VF __call__ as a FunctionWithJacobianProtocol for DerivativeChecker.

    Maps x -> F_eta(x), with jacobian = dF/dx (state jacobian).
    """

    def __init__(self, vf, bkd):
        self._vf = vf
        self._bkd = bkd

    def bkd(self):
        return self._bkd

    def nvars(self):
        return self._vf.nstates()

    def nqoi(self):
        return self._vf.nstates()

    def __call__(self, samples):
        return self._vf(samples)

    def jacobian(self, sample):
        jac_batch = self._vf.state_jacobian(sample)
        return jac_batch[0]


class _VectorFieldParamFunction:
    """Wraps VF as a function of params for DerivativeChecker.

    Maps eta -> F_eta(x_fixed), with jacobian = dF/d_eta.
    """

    def __init__(self, vf, fixed_states, bkd):
        self._vf = vf
        self._fixed_states = fixed_states
        self._bkd = bkd
        self._nstates = vf.nstates()

    def bkd(self):
        return self._bkd

    def nvars(self):
        return self._vf.hyp_list().nactive_params()

    def nqoi(self):
        nsamples = self._fixed_states.shape[1]
        return self._nstates * nsamples

    def __call__(self, params):
        p = params[:, 0]
        self._vf.hyp_list().set_active_values(p)
        result = self._vf(self._fixed_states)
        # Transpose to (nsamples, nstates) before flatten so row order
        # matches jacobian's (nsamples, nstates, nactive) reshape.
        return self._bkd.reshape(
            self._bkd.flatten(result.T), (-1, 1)
        )

    def jacobian(self, params):
        p = params[:, 0]
        self._vf.hyp_list().set_active_values(p)
        jac_batch = self._vf.param_jacobian(self._fixed_states)
        nsamples = self._fixed_states.shape[1]
        nactive = jac_batch.shape[2]
        return self._bkd.reshape(jac_batch, (nsamples * self._nstates, nactive))


class TestBasisExpansionVectorField:
    def test_protocol_conformance(self, bkd):
        exp = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        vf = BasisExpansionVectorField(exp)
        assert isinstance(vf, ParametricVectorFieldProtocol)

    def test_nstates(self, bkd):
        exp = _make_expansion(bkd, nvars=3, max_level=2, nqoi=3)
        vf = BasisExpansionVectorField(exp)
        assert vf.nstates() == 3

    def test_call_shape(self, bkd):
        exp = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        rng = np.random.RandomState(0)
        coef = bkd.array(rng.randn(exp.nterms(), 2))
        exp.set_coefficients(coef)
        vf = BasisExpansionVectorField(exp)
        states = bkd.array(rng.uniform(-1, 1, (2, 15)))
        result = vf(states)
        assert result.shape == (2, 15)

    def test_call_matches_expansion(self, bkd):
        exp = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        rng = np.random.RandomState(0)
        coef = bkd.array(rng.randn(exp.nterms(), 2))
        exp.set_coefficients(coef)
        vf = BasisExpansionVectorField(exp)
        states = bkd.array(rng.uniform(-1, 1, (2, 10)))
        bkd.assert_allclose(vf(states), exp(states))

    def test_state_jacobian_shape(self, bkd):
        exp = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        vf = BasisExpansionVectorField(exp)
        states = bkd.array(np.random.RandomState(0).uniform(-1, 1, (2, 5)))
        jac = vf.state_jacobian(states)
        assert jac.shape == (5, 2, 2)

    def test_param_jacobian_shape(self, bkd):
        exp = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        vf = BasisExpansionVectorField(exp)
        states = bkd.array(np.random.RandomState(0).uniform(-1, 1, (2, 5)))
        jac = vf.param_jacobian(states)
        nactive = vf.hyp_list().nactive_params()
        assert jac.shape == (5, 2, nactive)

    def test_state_jacobian_derivative_check(self, bkd):
        exp = _make_expansion(bkd, nvars=2, max_level=3, nqoi=2)
        rng = np.random.RandomState(42)
        coef = bkd.array(rng.randn(exp.nterms(), 2))
        exp.set_coefficients(coef)
        vf = BasisExpansionVectorField(exp)
        wrapper = _VectorFieldAsFunction(vf, bkd)
        checker = DerivativeChecker(wrapper)
        sample = bkd.array(rng.uniform(-1, 1, (2, 1)))
        errors = checker.check_derivatives(sample)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_param_jacobian_derivative_check(self, bkd):
        exp = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        rng = np.random.RandomState(42)
        coef = bkd.array(rng.randn(exp.nterms(), 2))
        exp.set_coefficients(coef)
        vf = BasisExpansionVectorField(exp)
        fixed_states = bkd.array(rng.uniform(-1, 1, (2, 3)))
        wrapper = _VectorFieldParamFunction(vf, fixed_states, bkd)
        checker = DerivativeChecker(wrapper)
        params = bkd.array(rng.randn(vf.hyp_list().nactive_params(), 1))
        errors = checker.check_derivatives(params, relative=False)
        assert bkd.min(errors[0]) < 1e-10

    def test_nvars_nqoi_mismatch_raises(self, bkd):
        exp = _make_expansion(bkd, nvars=2, max_level=2, nqoi=3)
        with pytest.raises(ValueError, match="nvars == nqoi"):
            BasisExpansionVectorField(exp)

    def test_has_state_jacobian(self, bkd):
        exp = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        vf = BasisExpansionVectorField(exp)
        assert hasattr(vf, "state_jacobian")

    def test_has_param_jacobian(self, bkd):
        exp = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        vf = BasisExpansionVectorField(exp)
        assert hasattr(vf, "param_jacobian")

    def test_hyp_list_sync(self, bkd):
        exp = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        rng = np.random.RandomState(0)
        coef = bkd.array(rng.randn(exp.nterms(), 2))
        exp.set_coefficients(coef)
        vf = BasisExpansionVectorField(exp)
        new_params = bkd.array(rng.randn(vf.hyp_list().nactive_params()))
        vf.hyp_list().set_active_values(new_params)
        states = bkd.array(rng.uniform(-1, 1, (2, 5)))
        result_vf = vf(states)
        result_exp = exp(states)
        bkd.assert_allclose(result_vf, result_exp)


class TestAdditiveVectorField:
    def test_additive_evaluation(self, bkd):
        exp1 = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        exp2 = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        rng = np.random.RandomState(0)
        exp1.set_coefficients(bkd.array(rng.randn(exp1.nterms(), 2)))
        exp2.set_coefficients(bkd.array(rng.randn(exp2.nterms(), 2)))
        vf1 = BasisExpansionVectorField(exp1)
        vf2 = BasisExpansionVectorField(exp2)
        additive = AdditiveVectorField([vf1, vf2])

        states = bkd.array(rng.uniform(-1, 1, (2, 10)))
        expected = vf1(states) + vf2(states)
        bkd.assert_allclose(additive(states), expected)

    def test_additive_nstates(self, bkd):
        exp1 = _make_expansion(bkd, nvars=3, max_level=2, nqoi=3)
        exp2 = _make_expansion(bkd, nvars=3, max_level=2, nqoi=3)
        vf1 = BasisExpansionVectorField(exp1)
        vf2 = BasisExpansionVectorField(exp2)
        additive = AdditiveVectorField([vf1, vf2])
        assert additive.nstates() == 3

    def test_additive_state_jacobian(self, bkd):
        exp1 = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        exp2 = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        rng = np.random.RandomState(0)
        exp1.set_coefficients(bkd.array(rng.randn(exp1.nterms(), 2)))
        exp2.set_coefficients(bkd.array(rng.randn(exp2.nterms(), 2)))
        vf1 = BasisExpansionVectorField(exp1)
        vf2 = BasisExpansionVectorField(exp2)
        additive = AdditiveVectorField([vf1, vf2])

        states = bkd.array(rng.uniform(-1, 1, (2, 5)))
        expected = vf1.state_jacobian(states) + vf2.state_jacobian(states)
        bkd.assert_allclose(additive.state_jacobian(states), expected)

    def test_additive_param_jacobian_block_concatenated(self, bkd):
        exp1 = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        exp2 = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        rng = np.random.RandomState(0)
        exp1.set_coefficients(bkd.array(rng.randn(exp1.nterms(), 2)))
        exp2.set_coefficients(bkd.array(rng.randn(exp2.nterms(), 2)))
        vf1 = BasisExpansionVectorField(exp1)
        vf2 = BasisExpansionVectorField(exp2)
        additive = AdditiveVectorField([vf1, vf2])

        states = bkd.array(rng.uniform(-1, 1, (2, 5)))
        jac = additive.param_jacobian(states)
        n1 = vf1.hyp_list().nactive_params()
        n2 = vf2.hyp_list().nactive_params()
        assert jac.shape == (5, 2, n1 + n2)

    def test_nstates_mismatch_raises(self, bkd):
        exp1 = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        exp2 = _make_expansion(bkd, nvars=3, max_level=2, nqoi=3)
        vf1 = BasisExpansionVectorField(exp1)
        vf2 = BasisExpansionVectorField(exp2)
        with pytest.raises(ValueError, match="nstates"):
            AdditiveVectorField([vf1, vf2])

    def test_empty_components_raises(self, bkd):
        with pytest.raises(ValueError, match="At least one"):
            AdditiveVectorField([])

    def test_hyp_list_concatenated(self, bkd):
        exp1 = _make_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        exp2 = _make_expansion(bkd, nvars=2, max_level=3, nqoi=2)
        vf1 = BasisExpansionVectorField(exp1)
        vf2 = BasisExpansionVectorField(exp2)
        additive = AdditiveVectorField([vf1, vf2])
        n1 = vf1.hyp_list().nactive_params()
        n2 = vf2.hyp_list().nactive_params()
        assert additive.hyp_list().nactive_params() == n1 + n2
