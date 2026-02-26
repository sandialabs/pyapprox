"""Tests for BasisExpansionMSELoss."""

import numpy as np
import pytest

from pyapprox.optimization.minimize.scipy.trust_constr import (
    ScipyTrustConstrOptimizer,
)
from pyapprox.probability import UniformMarginal
from pyapprox.surrogates.affine.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.expansions.fitters import (
    LeastSquaresFitter,
)
from pyapprox.surrogates.affine.expansions.losses import (
    BasisExpansionMSELoss,
)
from pyapprox.surrogates.affine.indices import compute_hyperbolic_indices
from pyapprox.surrogates.affine.univariate import create_bases_1d
from pyapprox.util.test_utils import slow_test


class TestBasisExpansionMSELoss:

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _create_expansion(self, bkd, nvars: int, max_level: int, nqoi: int = 1):
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_loss_value_correct(self, bkd) -> None:
        """Loss computes correct MSE."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = bkd.asarray(np.random.randn(1, 20))

        loss = BasisExpansionMSELoss(expansion, samples, values, bkd)
        params = bkd.asarray(np.random.randn(loss.nvars(), 1))

        # Manual computation
        Phi = expansion.basis_matrix(samples)
        params_2d = bkd.reshape(params[:, 0], (-1, 1))
        residual = Phi @ params_2d - values.T
        expected = 0.5 * float(bkd.sum(residual**2)) / 20

        actual = float(loss(params)[0, 0])
        bkd.assert_allclose(
            bkd.asarray([actual]),
            bkd.asarray([expected]),
            rtol=1e-10,
        )

    def test_jacobian_matches_finite_difference(self, bkd) -> None:
        """Gradient matches finite difference."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = bkd.asarray(np.random.randn(1, 20))

        loss = BasisExpansionMSELoss(expansion, samples, values, bkd)
        params = bkd.asarray(np.random.randn(loss.nvars(), 1))

        analytical = bkd.flatten(loss.jacobian(params))

        # Finite difference
        eps = 1e-6
        fd_grad = []
        for i in range(loss.nvars()):
            p_plus = bkd.copy(params)
            p_minus = bkd.copy(params)
            p_plus[i, 0] = p_plus[i, 0] + eps
            p_minus[i, 0] = p_minus[i, 0] - eps
            fd_grad.append(
                (float(loss(p_plus)[0, 0]) - float(loss(p_minus)[0, 0])) / (2 * eps)
            )

        bkd.assert_allclose(analytical, bkd.asarray(fd_grad), rtol=1e-5)

    @pytest.mark.slow_on("TorchBkd")
    def test_hvp_matches_finite_difference(self, bkd) -> None:
        """HVP matches finite difference of gradient."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = bkd.asarray(np.random.randn(1, 20))

        loss = BasisExpansionMSELoss(expansion, samples, values, bkd)
        params = bkd.asarray(np.random.randn(loss.nvars(), 1))
        vec = bkd.asarray(np.random.randn(loss.nvars(), 1))

        analytical_hvp = bkd.flatten(loss.hvp(params, vec))

        # Finite difference: (grad(x + eps*v) - grad(x - eps*v)) / (2*eps)
        eps = 1e-6
        grad_plus = loss.jacobian(params + eps * vec)
        grad_minus = loss.jacobian(params - eps * vec)
        fd_hvp = bkd.flatten((grad_plus - grad_minus) / (2 * eps))

        bkd.assert_allclose(analytical_hvp, fd_hvp, rtol=1e-4)

    def test_scipy_trust_constr_recovers_lstsq_solution(self, bkd) -> None:
        """ScipyTrustConstr minimizing MSE gives same result as LeastSquaresFitter."""
        expansion_direct = self._create_expansion(bkd, nvars=2, max_level=3)
        expansion_iter = self._create_expansion(bkd, nvars=2, max_level=3)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = bkd.asarray(np.random.randn(1, 50))

        # Direct solver
        direct_result = LeastSquaresFitter(bkd).fit(
            expansion_direct, samples, values
        )

        # Iterative via ScipyTrustConstr
        loss = BasisExpansionMSELoss(expansion_iter, samples, values, bkd)

        optimizer = ScipyTrustConstrOptimizer(maxiter=1000, gtol=1e-10)
        nvars = loss.nvars()
        bounds = bkd.hstack(
            [
                bkd.full((nvars, 1), -1e10),
                bkd.full((nvars, 1), 1e10),
            ]
        )
        optimizer.bind(loss, bounds)

        init_params = bkd.zeros((nvars, 1))
        result = optimizer.minimize(init_params)

        iter_params = bkd.reshape(result.optima()[:, 0], (-1, 1))

        # Compare solutions
        bkd.assert_allclose(
            iter_params,
            direct_result.params(),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_multi_qoi(self, bkd) -> None:
        """Loss works with multiple QoIs."""
        expansion = self._create_expansion(bkd, nvars=2, max_level=2, nqoi=2)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = bkd.asarray(np.random.randn(2, 30))

        loss = BasisExpansionMSELoss(expansion, samples, values, bkd)

        # Verify nvars is correct
        assert loss.nvars() == expansion.nterms() * 2

        # Verify loss can be computed
        params = bkd.asarray(np.random.randn(loss.nvars(), 1))
        loss_val = loss(params)
        assert loss_val.shape == (1, 1)

        # Verify jacobian shape
        jac = loss.jacobian(params)
        assert jac.shape == (1, loss.nvars())

        # Verify hvp shape
        vec = bkd.asarray(np.random.randn(loss.nvars(), 1))
        hvp_result = loss.hvp(params, vec)
        assert hvp_result.shape == (loss.nvars(), 1)
