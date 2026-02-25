"""Tests for BasisExpansionMSELoss."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

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
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests, slow_test  # noqa: F401


class TestBasisExpansionMSELoss(Generic[Array], unittest.TestCase):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        np.random.seed(42)

    def _create_expansion(self, nvars: int, max_level: int, nqoi: int = 1):
        bkd = self._bkd
        marginals = [UniformMarginal(-1.0, 1.0, bkd) for _ in range(nvars)]
        bases_1d = create_bases_1d(marginals, bkd)
        indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
        basis = OrthonormalPolynomialBasis(bases_1d, bkd, indices)
        return BasisExpansion(basis, bkd, nqoi=nqoi)

    def test_loss_value_correct(self) -> None:
        """Loss computes correct MSE."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = self._bkd.asarray(np.random.randn(1, 20))

        loss = BasisExpansionMSELoss(expansion, samples, values, self._bkd)
        params = self._bkd.asarray(np.random.randn(loss.nvars(), 1))

        # Manual computation
        Phi = expansion.basis_matrix(samples)
        params_2d = self._bkd.reshape(params[:, 0], (-1, 1))
        residual = Phi @ params_2d - values.T
        expected = 0.5 * float(self._bkd.sum(residual**2)) / 20

        actual = float(loss(params)[0, 0])
        self._bkd.assert_allclose(
            self._bkd.asarray([actual]),
            self._bkd.asarray([expected]),
            rtol=1e-10,
        )

    def test_jacobian_matches_finite_difference(self) -> None:
        """Gradient matches finite difference."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = self._bkd.asarray(np.random.randn(1, 20))

        loss = BasisExpansionMSELoss(expansion, samples, values, self._bkd)
        params = self._bkd.asarray(np.random.randn(loss.nvars(), 1))

        analytical = self._bkd.flatten(loss.jacobian(params))

        # Finite difference
        eps = 1e-6
        fd_grad = []
        for i in range(loss.nvars()):
            p_plus = self._bkd.copy(params)
            p_minus = self._bkd.copy(params)
            p_plus[i, 0] = p_plus[i, 0] + eps
            p_minus[i, 0] = p_minus[i, 0] - eps
            fd_grad.append(
                (float(loss(p_plus)[0, 0]) - float(loss(p_minus)[0, 0])) / (2 * eps)
            )

        self._bkd.assert_allclose(analytical, self._bkd.asarray(fd_grad), rtol=1e-5)

    def test_hvp_matches_finite_difference(self) -> None:
        """HVP matches finite difference of gradient."""
        expansion = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 20)))
        values = self._bkd.asarray(np.random.randn(1, 20))

        loss = BasisExpansionMSELoss(expansion, samples, values, self._bkd)
        params = self._bkd.asarray(np.random.randn(loss.nvars(), 1))
        vec = self._bkd.asarray(np.random.randn(loss.nvars(), 1))

        analytical_hvp = self._bkd.flatten(loss.hvp(params, vec))

        # Finite difference: (grad(x + eps*v) - grad(x - eps*v)) / (2*eps)
        eps = 1e-6
        grad_plus = loss.jacobian(params + eps * vec)
        grad_minus = loss.jacobian(params - eps * vec)
        fd_hvp = self._bkd.flatten((grad_plus - grad_minus) / (2 * eps))

        self._bkd.assert_allclose(analytical_hvp, fd_hvp, rtol=1e-4)

    def test_scipy_trust_constr_recovers_lstsq_solution(self) -> None:
        """ScipyTrustConstr minimizing MSE gives same result as LeastSquaresFitter."""
        expansion_direct = self._create_expansion(nvars=2, max_level=3)
        expansion_iter = self._create_expansion(nvars=2, max_level=3)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 50)))
        values = self._bkd.asarray(np.random.randn(1, 50))

        # Direct solver
        direct_result = LeastSquaresFitter(self._bkd).fit(
            expansion_direct, samples, values
        )

        # Iterative via ScipyTrustConstr
        loss = BasisExpansionMSELoss(expansion_iter, samples, values, self._bkd)

        optimizer = ScipyTrustConstrOptimizer(maxiter=1000, gtol=1e-10)
        nvars = loss.nvars()
        bounds = self._bkd.hstack(
            [
                self._bkd.full((nvars, 1), -1e10),
                self._bkd.full((nvars, 1), 1e10),
            ]
        )
        optimizer.bind(loss, bounds)

        init_params = self._bkd.zeros((nvars, 1))
        result = optimizer.minimize(init_params)

        iter_params = self._bkd.reshape(result.optima()[:, 0], (-1, 1))

        # Compare solutions
        self._bkd.assert_allclose(
            iter_params,
            direct_result.params(),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_multi_qoi(self) -> None:
        """Loss works with multiple QoIs."""
        expansion = self._create_expansion(nvars=2, max_level=2, nqoi=2)
        samples = self._bkd.asarray(np.random.uniform(-1, 1, (2, 30)))
        values = self._bkd.asarray(np.random.randn(2, 30))

        loss = BasisExpansionMSELoss(expansion, samples, values, self._bkd)

        # Verify nvars is correct
        self.assertEqual(loss.nvars(), expansion.nterms() * 2)

        # Verify loss can be computed
        params = self._bkd.asarray(np.random.randn(loss.nvars(), 1))
        loss_val = loss(params)
        self.assertEqual(loss_val.shape, (1, 1))

        # Verify jacobian shape
        jac = loss.jacobian(params)
        self.assertEqual(jac.shape, (1, loss.nvars()))

        # Verify hvp shape
        vec = self._bkd.asarray(np.random.randn(loss.nvars(), 1))
        hvp_result = loss.hvp(params, vec)
        self.assertEqual(hvp_result.shape, (loss.nvars(), 1))


class TestBasisExpansionMSELossNumpy(TestBasisExpansionMSELoss[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestBasisExpansionMSELossTorch(TestBasisExpansionMSELoss[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        return TorchBkd()

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        super().setUp()

    @slow_test
    def test_hvp_matches_finite_difference(self) -> None:
        super().test_hvp_matches_finite_difference()


if __name__ == "__main__":
    unittest.main()
