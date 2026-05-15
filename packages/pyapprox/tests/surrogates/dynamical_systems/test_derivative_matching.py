"""Tests for DerivativeMatchingLoss."""

import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.protocols.jacobian import (
    FunctionWithJacobianProtocol,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.dataset import SnapshotDataset
from pyapprox.surrogates.dynamical_systems.losses.derivative_matching import (
    DerivativeMatchingLoss,
)
from pyapprox.surrogates.dynamical_systems.vector_fields import (
    BasisExpansionVectorField,
)


def _make_vf_and_data(bkd, nvars=2, max_level=2, nsamples=50, seed=0):
    marginals = [UniformMarginal(-2.0, 2.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    exp = BasisExpansion(basis, bkd, nqoi=nvars)
    rng = np.random.RandomState(seed)
    true_coef = bkd.array(rng.randn(exp.nterms(), nvars))
    exp.set_coefficients(bkd.copy(true_coef))
    vf = BasisExpansionVectorField(exp)
    states = bkd.array(rng.uniform(-1, 1, (nvars, nsamples)))
    derivs = bkd.copy(vf(states))
    dataset = SnapshotDataset(states, derivs, bkd)
    return vf, dataset, true_coef


class TestDerivativeMatchingLoss:
    def test_protocol_conformance(self, bkd):
        vf, dataset, _ = _make_vf_and_data(bkd)
        loss = DerivativeMatchingLoss(vf, dataset)
        assert isinstance(loss, FunctionWithJacobianProtocol)

    def test_nvars_nqoi(self, bkd):
        vf, dataset, _ = _make_vf_and_data(bkd)
        loss = DerivativeMatchingLoss(vf, dataset)
        assert loss.nvars() == vf.hyp_list().nactive_params()
        assert loss.nqoi() == 1

    def test_call_shape(self, bkd):
        vf, dataset, _ = _make_vf_and_data(bkd)
        loss = DerivativeMatchingLoss(vf, dataset)
        params = vf.hyp_list().get_active_values()
        result = loss(params[:, None])
        assert result.shape == (1, 1)

    def test_zero_at_true_params(self, bkd):
        vf, dataset, true_coef = _make_vf_and_data(bkd)
        loss = DerivativeMatchingLoss(vf, dataset)
        params = bkd.flatten(true_coef)
        result = loss(params[:, None])
        bkd.assert_allclose(result, bkd.zeros((1, 1)), atol=1e-12)

    def test_positive_at_wrong_params(self, bkd):
        vf, dataset, true_coef = _make_vf_and_data(bkd)
        loss = DerivativeMatchingLoss(vf, dataset)
        rng = np.random.RandomState(99)
        wrong_params = bkd.array(rng.randn(loss.nvars(), 1))
        result = loss(wrong_params)
        assert bkd.to_numpy(result[0, 0]) > 0.0

    def test_jacobian_shape(self, bkd):
        vf, dataset, _ = _make_vf_and_data(bkd)
        loss = DerivativeMatchingLoss(vf, dataset)
        params = vf.hyp_list().get_active_values()
        jac = loss.jacobian(params[:, None])
        assert jac.shape == (1, loss.nvars())

    def test_jacobian_zero_at_true_params(self, bkd):
        vf, dataset, true_coef = _make_vf_and_data(bkd)
        loss = DerivativeMatchingLoss(vf, dataset)
        params = bkd.flatten(true_coef)
        jac = loss.jacobian(params[:, None])
        bkd.assert_allclose(jac, bkd.zeros((1, loss.nvars())), atol=1e-10)

    def test_jacobian_derivative_check(self, bkd):
        vf, dataset, _ = _make_vf_and_data(bkd, nsamples=20)
        loss = DerivativeMatchingLoss(vf, dataset)
        rng = np.random.RandomState(42)
        params = bkd.array(rng.randn(loss.nvars(), 1))
        checker = DerivativeChecker(loss)
        errors = checker.check_derivatives(params)
        assert checker.error_ratio(errors[0]) <= 1e-6

    def test_loss_decreases_toward_optimum(self, bkd):
        vf, dataset, true_coef = _make_vf_and_data(bkd)
        loss = DerivativeMatchingLoss(vf, dataset)
        rng = np.random.RandomState(42)
        far_params = bkd.array(rng.randn(loss.nvars()))
        true_params = bkd.flatten(true_coef)
        mid_params = 0.5 * far_params + 0.5 * true_params
        loss_far = bkd.to_numpy(loss(far_params[:, None])[0, 0])
        loss_mid = bkd.to_numpy(loss(mid_params[:, None])[0, 0])
        loss_true = bkd.to_numpy(loss(true_params[:, None])[0, 0])
        assert loss_far > loss_mid
        assert loss_mid > loss_true

    def test_invalid_vf_raises(self, bkd):
        states = bkd.array(np.zeros((2, 10)))
        derivs = bkd.array(np.zeros((2, 10)))
        dataset = SnapshotDataset(states, derivs, bkd)
        with pytest.raises(TypeError, match="ParametricVectorFieldProtocol"):
            DerivativeMatchingLoss("not a vf", dataset)
