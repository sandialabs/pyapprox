"""Tests for BatchedBoundODEResidual."""

import numpy as np
import pytest

from pyapprox.ode.linear_operator import (
    BlockDiagonalLinearOperator,
    LinearOperatorProtocol,
)
from pyapprox.ode.protocols import JacobianOperatorProtocol
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
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_block_diagonal import (
    BackendBlockDiagonalOperator,
    BlockDiagonalFormat,
    BlockDiagonalLU,
)
from pyapprox.util.linalg.backend_dense import BackendVectorSpace
from pyapprox.util.linalg.operators import (
    JacobianSnapshotProtocol,
    LinearizationPoint,
    ScaledSum,
)


def _make_expansion(
    bkd: Backend[Array], nvars: int, nqoi: int, max_level: int = 3
) -> BasisExpansion[Array]:
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
    def test_protocol_conformance(self, bkd: Backend[Array]) -> None:
        expansion = _make_expansion(bkd, nvars=2, nqoi=2)
        assert isinstance(expansion, LearnedFunctionProtocol)
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2)
        assert isinstance(wrapper, ImplicitODEResidualProtocol)

    def test_single_trajectory_nonparametric(self, bkd: Backend[Array]) -> None:
        expansion = _make_expansion(bkd, nvars=2, nqoi=2)
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2)

        state = bkd.array([1.0, -0.5])
        result = wrapper(state, 0.5)
        assert result.shape == (2,)

        samples = bkd.array([[1.0], [-0.5]])
        expected = expansion(samples)
        bkd.assert_allclose(result, bkd.reshape(expected, (2,)), rtol=1e-12)

    def test_single_trajectory_parametric(self, bkd: Backend[Array]) -> None:
        expansion = _make_expansion(bkd, nvars=3, nqoi=2)
        mu_batch = bkd.array([[1.5]])
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2, mu_batch=mu_batch)

        state = bkd.array([0.5, -1.0])
        result = wrapper(state, 0.0)
        assert result.shape == (2,)

        augmented = bkd.array([[0.5], [-1.0], [1.5]])
        expected = expansion(augmented)
        bkd.assert_allclose(result, bkd.reshape(expected, (2,)), rtol=1e-12)

    def test_batched_trajectories_decouple(self, bkd: Backend[Array]) -> None:
        expansion = _make_expansion(bkd, nvars=3, nqoi=2)
        k = 4
        mu_vals = bkd.array([[0.5, 1.0, 1.5, 2.0]])
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2, mu_batch=mu_vals)

        rng = np.random.RandomState(7)
        states_flat = bkd.array(rng.randn(2 * k))
        result = wrapper(states_flat, 0.0)
        assert result.shape == (2 * k,)

        for i in range(k):
            single_state = states_flat[i * 2 : (i + 1) * 2]
            single_mu = bkd.array([[float(mu_vals[0, i])]])
            single_wrapper = BatchedBoundODEResidual(
                expansion, n_dynamic=2, mu_batch=single_mu
            )
            single_result = single_wrapper(single_state, 0.0)
            bkd.assert_allclose(result[i * 2 : (i + 1) * 2], single_result, rtol=1e-12)

    def test_newton_jacobian_block_diagonal(self, bkd: Backend[Array]) -> None:
        expansion = _make_expansion(bkd, nvars=3, nqoi=2)
        k = 3
        mu_vals = bkd.array([[0.5, 1.0, 1.5]])
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2, mu_batch=mu_vals)

        rng = np.random.RandomState(11)
        state = bkd.array(rng.randn(2 * k))
        coeff = 0.05

        op = wrapper.newton_jacobian(state, 0.0, coeff)
        assert isinstance(op, BlockDiagonalLinearOperator)
        assert isinstance(op, LinearOperatorProtocol)

        full_jac = wrapper.jacobian(state, 0.0)
        eye = bkd.eye(2 * k)
        expected = eye - coeff * full_jac
        bkd.assert_allclose(op.as_matrix(), expected, rtol=1e-12)

    def test_param_jacobian_shape(self, bkd: Backend[Array]) -> None:
        expansion = _make_expansion(bkd, nvars=3, nqoi=2)
        k = 2
        mu_vals = bkd.array([[0.5, 1.5]])
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2, mu_batch=mu_vals)

        state = bkd.array([1.0, 0.0, -1.0, 0.5])
        pjac = wrapper.param_jacobian(state, 0.0)
        nactive = wrapper.nparams()
        assert pjac.shape == (2 * k, nactive)

    def test_set_param_bumps_the_parameter_version(self, bkd: Backend[Array]) -> None:
        """Every set_param increments the version, even with an unchanged
        value."""
        expansion = _make_expansion(bkd, nvars=2, nqoi=2)
        wrapper = BatchedBoundODEResidual(expansion, n_dynamic=2)
        assert wrapper.parameters_version() == 0
        coefficients = expansion.hyp_list().get_active_values()
        wrapper.set_param(coefficients)
        wrapper.set_param(coefficients)
        assert wrapper.parameters_version() == 2

    def test_invalid_nvars_raises(self, bkd: Backend[Array]) -> None:
        expansion = _make_expansion(bkd, nvars=2, nqoi=2)
        mu_batch = bkd.array([[1.0]])
        with pytest.raises(ValueError, match="nvars"):
            BatchedBoundODEResidual(expansion, n_dynamic=2, mu_batch=mu_batch)

    def test_invalid_nqoi_raises(self, bkd: Backend[Array]) -> None:
        expansion = _make_expansion(bkd, nvars=2, nqoi=3)
        with pytest.raises(ValueError, match="nqoi"):
            BatchedBoundODEResidual(expansion, n_dynamic=2)


def _four_trajectories(
    bkd: Backend[Array],
) -> tuple[BatchedBoundODEResidual[Array], Array]:
    """Four parametric trajectories of two states each, and a state."""
    expansion = _make_expansion(bkd, nvars=3, nqoi=2)
    residual = BatchedBoundODEResidual(
        expansion, n_dynamic=2, mu_batch=bkd.array([[0.5, 1.0, 1.5, 2.0]])
    )
    return residual, bkd.array(np.random.RandomState(13).randn(8))


class TestBatchedJacobianOperator:
    def test_satisfies_the_protocols(self, bkd: Backend[Array]) -> None:
        residual, state = _four_trajectories(bkd)
        producer: JacobianOperatorProtocol[Array, Array, BlockDiagonalFormat[Array]] = (
            residual
        )
        assert isinstance(producer, JacobianOperatorProtocol)
        point = LinearizationPoint(0, ("step", 2))
        op = producer.jacobian_operator(state, 0.0, point)
        assert isinstance(op, JacobianSnapshotProtocol)
        assert op.linearization() == point
        assert op.domain() == residual.state_descriptor()
        snapshot = residual.jacobian_operator(state, 0.0, point)
        assert (snapshot.nblocks(), snapshot.block_size()) == (4, 2)

    def test_matches_the_dense_jacobian(self, bkd: Backend[Array]) -> None:
        residual, state = _four_trajectories(bkd)
        op = residual.jacobian_operator(state, 0.0, LinearizationPoint(0))
        dense = residual.jacobian(state, 0.0)
        x = bkd.array(np.random.RandomState(14).randn(8, 3))
        bkd.assert_allclose(op.apply(x), bkd.dot(dense, x), rtol=1e-12)
        bkd.assert_allclose(op.apply_transpose(x), bkd.dot(dense.T, x), rtol=1e-12)

    def test_action_is_the_residual_derivative(self, bkd: Backend[Array]) -> None:
        residual, state = _four_trajectories(bkd)
        op = residual.jacobian_operator(state, 0.0, LinearizationPoint(0))
        direction = bkd.array(np.random.RandomState(15).randn(8))
        step = 1e-6
        difference = (
            residual(state + step * direction, 0.0)
            - residual(state - step * direction, 0.0)
        ) / (2 * step)
        bkd.assert_allclose(
            op.apply(direction[:, None])[:, 0], difference, rtol=1e-6, atol=1e-8
        )

    def test_newton_operator_factors_blockwise(self, bkd: Backend[Array]) -> None:
        """I - c J built from operators stays block-diagonal, and its batched
        LU solves as the existing block-diagonal Newton matrix does."""
        residual, state = _four_trajectories(bkd)
        descriptor = residual.state_descriptor()
        coefficient = 0.05
        jacobian = residual.jacobian_operator(state, 0.0, LinearizationPoint(0))
        identity = BackendBlockDiagonalOperator(
            bkd.stack([bkd.eye(2)] * 4), descriptor, bkd
        )
        newton = ScaledSum(
            [(1.0, identity), (-coefficient, jacobian)],
            BackendVectorSpace(descriptor, bkd),
        )
        handle = BlockDiagonalLU(bkd).setup(newton)
        rhs = bkd.array(np.random.RandomState(16).randn(8, 1))
        x, info = handle.solve(rhs)
        assert info.converged
        expected = residual.newton_jacobian(state, 0.0, coefficient).solve(rhs[:, 0])
        bkd.assert_allclose(x[:, 0], expected, rtol=1e-12)


class TestBatchedDerivativesBundle:
    def test_bundle_keeps_the_block_diagonal_jacobian(
        self, bkd: Backend[Array]
    ) -> None:
        """The bundle's Jacobian is the residual's own block-diagonal
        producer (so its format pairs with the batched LU), and its
        parameter and initial-state operators reproduce the matrix
        methods."""
        residual, state = _four_trajectories(bkd)
        bundle = residual.derivatives()
        assert bundle.jacobian is not None and bundle.param is not None
        assert bundle.initial_state is not None and bundle.state_state is None
        point = LinearizationPoint(0, ("step", 1))
        jacobian = bundle.jacobian(state, 0.0, point)
        assert isinstance(jacobian.format(), BlockDiagonalFormat)
        assert jacobian.linearization() == point
        rng = np.random.RandomState(17)
        nactive = residual.nparams()
        v = bkd.array(rng.randn(nactive, 2))
        w = bkd.array(rng.randn(8, 2))
        matrix = residual.param_jacobian(state, 0.0)
        param = bundle.param(state, 0.0)
        assert (param.domain().size, param.range().size) == (nactive, 8)
        bkd.assert_allclose(param.apply(v), bkd.dot(matrix, v), rtol=1e-12)
        bkd.assert_allclose(param.apply_transpose(w), bkd.dot(matrix.T, w), rtol=1e-12)
        initial = bundle.initial_state()
        assert initial.properties().zero

    def test_bundle_follows_the_active_coefficients(
        self, bkd: Backend[Array]
    ) -> None:
        """Kept between calls and across set_param (its fields read the
        current coefficients, so a step asking for it several times builds
        no spaces), and rebuilt when fixing a coefficient shrinks its
        parameter space."""
        residual, _ = _four_trajectories(bkd)
        hyp_list = residual.learned_function().hyp_list()
        bundle = residual.derivatives()
        assert residual.derivatives() is bundle
        residual.set_param(hyp_list.get_active_values())
        assert residual.derivatives() is bundle
        before = bundle.param_space
        assert before is not None
        hyp_list.set_active_indices(bkd.arange(1, hyp_list.nparams()))
        rebuilt = residual.derivatives()
        assert rebuilt is not bundle
        after = rebuilt.param_space
        assert after is not None
        assert after.descriptor().size == before.descriptor().size - 1
