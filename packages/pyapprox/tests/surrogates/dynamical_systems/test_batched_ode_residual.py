"""Tests for BatchedBoundODEResidual."""

import numpy as np
import pytest

from pyapprox.ode.linear_operator import (
    BlockDiagonalLinearOperator,
    LinearOperatorProtocol,
)
from pyapprox.ode.protocols.ode_residual import (
    ImplicitODEResidualProtocol,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.surrogates.dynamical_systems.batched_ode_residual import (
    BatchedBoundODEResidual,
)
from pyapprox.surrogates.dynamical_systems.protocols import (
    LearnedFunctionProtocol,
)


def _make_expansion(bkd, nvars, nqoi, max_level=3):
    marginals = [UniformMarginal(-3.0, 3.0, bkd) for _ in range(nvars)]
    bases_1d = create_bases_1d(marginals, bkd)
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
    expansion = BasisExpansion(basis, bkd, nqoi=nqoi)
    rng = np.random.RandomState(99)
    coef = bkd.array(rng.randn(expansion.nterms(), nqoi) * 0.1)
    expansion.set_coefficients(coef)
    return expansion


class TestBatchedBoundODEResidual:
    def test_protocol_conformance(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, nqoi=2)
        assert isinstance(expansion, LearnedFunctionProtocol)
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2)
        assert isinstance(wrapper, ImplicitODEResidualProtocol)

    def test_single_trajectory_nonparametric(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, nqoi=2)
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2)
        wrapper.set_time(0.5)

        state = bkd.array([1.0, -0.5])
        result = wrapper(state)
        assert result.shape == (2,)

        samples = bkd.array([[1.0], [-0.5]])
        expected = expansion(samples)
        bkd.assert_allclose(
            result, bkd.reshape(expected, (2,)), rtol=1e-12
        )

    def test_single_trajectory_parametric(self, bkd):
        expansion = _make_expansion(bkd, nvars=3, nqoi=2)
        mu_batch = bkd.array([[1.5]])
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2, mu_batch=mu_batch)
        wrapper.set_time(0.0)

        state = bkd.array([0.5, -1.0])
        result = wrapper(state)
        assert result.shape == (2,)

        augmented = bkd.array([[0.5], [-1.0], [1.5]])
        expected = expansion(augmented)
        bkd.assert_allclose(
            result, bkd.reshape(expected, (2,)), rtol=1e-12
        )

    def test_batched_trajectories_decouple(self, bkd):
        expansion = _make_expansion(bkd, nvars=3, nqoi=2)
        k = 4
        mu_vals = bkd.array([[0.5, 1.0, 1.5, 2.0]])
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2, mu_batch=mu_vals)
        wrapper.set_time(0.0)

        rng = np.random.RandomState(7)
        states_flat = bkd.array(rng.randn(2 * k))
        result = wrapper(states_flat)
        assert result.shape == (2 * k,)

        for i in range(k):
            single_state = states_flat[i * 2:(i + 1) * 2]
            single_mu = bkd.array([[float(mu_vals[0, i])]])
            single_wrapper = BatchedBoundODEResidual(
                expansion, n_dynamic=2, mu_batch=single_mu
            )
            single_wrapper.set_time(0.0)
            single_result = single_wrapper(single_state)
            bkd.assert_allclose(
                result[i * 2:(i + 1) * 2], single_result, rtol=1e-12
            )

    def test_newton_jacobian_block_diagonal(self, bkd):
        expansion = _make_expansion(bkd, nvars=3, nqoi=2)
        k = 3
        mu_vals = bkd.array([[0.5, 1.0, 1.5]])
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2, mu_batch=mu_vals)
        wrapper.set_time(0.0)

        rng = np.random.RandomState(11)
        state = bkd.array(rng.randn(2 * k))
        coeff = 0.05

        op = wrapper.newton_jacobian(state, coeff)
        assert isinstance(op, BlockDiagonalLinearOperator)
        assert isinstance(op, LinearOperatorProtocol)

        full_jac = wrapper.jacobian(state)
        eye = bkd.eye(2 * k)
        expected = eye - coeff * full_jac
        bkd.assert_allclose(op.as_matrix(), expected, rtol=1e-12)

    def test_param_jacobian_shape(self, bkd):
        expansion = _make_expansion(bkd, nvars=3, nqoi=2)
        k = 2
        mu_vals = bkd.array([[0.5, 1.5]])
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2, mu_batch=mu_vals)
        wrapper.set_time(0.0)

        state = bkd.array([1.0, 0.0, -1.0, 0.5])
        pjac = wrapper.param_jacobian(state)
        nactive = wrapper.nparams()
        assert pjac.shape == (2 * k, nactive)

    def test_invalid_nvars_raises(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, nqoi=2)
        mu_batch = bkd.array([[1.0]])
        with pytest.raises(ValueError, match="nvars"):
            BatchedBoundODEResidual(expansion, n_dynamic=2, mu_batch=mu_batch)

    def test_invalid_nqoi_raises(self, bkd):
        expansion = _make_expansion(bkd, nvars=2, nqoi=3)
        with pytest.raises(ValueError, match="nqoi"):
            BatchedBoundODEResidual(expansion, n_dynamic=2)
