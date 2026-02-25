"""Tests for MultiplicativeAdditiveDiscrepancy model."""

import unittest
from typing import Any, Generic

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import load_tests  # noqa: F401

from pyapprox.surrogates.affine.univariate import MonomialBasis1D
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.basis import MultiIndexBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)

from pyapprox.surrogates.mfnets.discrepancy import (
    MultiplicativeAdditiveDiscrepancy,
)


def _create_expansion(
    bkd: Backend[Array], nvars: int, nqoi: int, max_level: int = 2
) -> BasisExpansion[Array]:
    bases_1d = [MonomialBasis1D(bkd) for _ in range(nvars)]
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = MultiIndexBasis.__new__(MultiIndexBasis)
    MultiIndexBasis.__init__(basis, bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=nqoi)


class TestMultiplicativeAdditiveDiscrepancy(
    Generic[Array], unittest.TestCase
):
    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        np.random.seed(42)
        self._bkd = self.bkd()

    def _create_discrepancy(
        self,
        nvars_x: int = 1,
        nqoi: int = 1,
        nscaled_qoi: int = 1,
        scale_level: int = 1,
        delta_level: int = 2,
    ) -> MultiplicativeAdditiveDiscrepancy:
        bkd = self._bkd
        # Scaling models: one per output QoI
        scalings = []
        for ii in range(nqoi):
            sm = _create_expansion(bkd, nvars_x, nscaled_qoi, scale_level)
            np.random.seed(100 + ii)
            sm.set_coefficients(
                bkd.asarray(np.random.randn(sm.nterms(), nscaled_qoi))
            )
            scalings.append(sm)

        # Delta model
        delta = _create_expansion(bkd, nvars_x, nqoi, delta_level)
        np.random.seed(200)
        delta.set_coefficients(
            bkd.asarray(np.random.randn(delta.nterms(), nqoi))
        )

        return MultiplicativeAdditiveDiscrepancy(
            scalings, delta, nscaled_qoi, bkd
        )

    def test_construction(self) -> None:
        disc = self._create_discrepancy()
        self.assertEqual(disc.nvars(), 2)  # 1 (x) + 1 (q)
        self.assertEqual(disc.nqoi(), 1)

    def test_eval_against_manual(self) -> None:
        """Test __call__ matches manual computation."""
        bkd = self._bkd
        disc = self._create_discrepancy(nvars_x=1, nqoi=1, nscaled_qoi=1)

        np.random.seed(99)
        nsamples = 8
        # Augmented input: [x; q]
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, nsamples)))
        x = samples[:1]
        q = samples[1:]

        # Manual: scaling(x) * q + delta(x)
        delta_val = disc.delta_model()(x)  # (1, nsamples)
        scale_val = disc.scaling_models()[0](x)  # (1, nsamples)
        expected = scale_val * q + delta_val

        result = disc(samples)
        bkd.assert_allclose(result, expected, rtol=1e-12)

    def test_eval_multi_qoi(self) -> None:
        """Test with nqoi=2, nscaled_qoi=1."""
        bkd = self._bkd
        disc = self._create_discrepancy(
            nvars_x=1, nqoi=2, nscaled_qoi=1
        )
        self.assertEqual(disc.nvars(), 2)
        self.assertEqual(disc.nqoi(), 2)

        np.random.seed(99)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 6)))
        result = disc(samples)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 6)

    def test_hyp_list(self) -> None:
        disc = self._create_discrepancy()
        hyps = disc.hyp_list()
        # Total params = sum(scaling params) + delta params
        expected = sum(
            s.hyp_list().nparams() for s in disc.scaling_models()
        ) + disc.delta_model().hyp_list().nparams()
        self.assertEqual(hyps.nparams(), expected)

    def test_jacobian_wrt_params_with_derivative_checker(self) -> None:
        """Validate jacobian_wrt_params using DerivativeChecker."""
        bkd = self._bkd
        disc = self._create_discrepancy(
            nvars_x=1, nqoi=1, nscaled_qoi=1, scale_level=1, delta_level=2
        )

        # Wrap discrepancy as a function of params for DerivativeChecker
        np.random.seed(42)
        nsamples = 5
        fixed_samples = bkd.asarray(np.random.uniform(-0.8, 0.8, (2, nsamples)))

        nactive = disc.hyp_list().nactive_params()

        # Use nqoi=1 weighted sum for DerivativeChecker
        # Pick random weights to check jacobian in a single direction
        np.random.seed(77)
        nqoi_disc = disc.nqoi()
        weights = bkd.asarray(np.random.randn(nqoi_disc, 1))

        def eval_fn(params: Array) -> Array:
            """Weighted sum of disc output over QoI, stacked over samples."""
            p = bkd.flatten(params)
            disc.hyp_list().set_active_values(p)
            disc._sync_from_hyp_list()
            vals = disc(fixed_samples)  # (nqoi, nsamples)
            # Weighted sum over QoI: (1, nsamples)
            weighted = (weights.T @ vals)  # (1, nsamples)
            return weighted.T  # (nsamples, 1)

        def jac_fn(params: Array) -> Array:
            """Jacobian of weighted sum w.r.t. params."""
            p = bkd.flatten(params)
            disc.hyp_list().set_active_values(p)
            disc._sync_from_hyp_list()
            j = disc.jacobian_wrt_params(fixed_samples)  # (nsamples, nqoi, nactive)
            # Weighted sum over QoI: (nsamples, nactive)
            wj = bkd.einsum("snp,n->sp", j, bkd.flatten(weights))
            return wj  # (nsamples, nactive) = (nqoi_fn, nvars_fn)

        wrapped = FunctionWithJacobianFromCallable(
            nqoi=nsamples,
            nvars=nactive,
            fun=eval_fn,
            jacobian=jac_fn,
            bkd=bkd,
        )
        checker = DerivativeChecker(wrapped)

        current_params = disc.hyp_list().get_active_values()
        sample = bkd.reshape(current_params, (-1, 1))

        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        self.assertLess(float(bkd.to_numpy(ratio)), 1e-6)

    def test_jacobian_wrt_params_multi_qoi(self) -> None:
        """Validate jacobian_wrt_params for nqoi=2."""
        bkd = self._bkd
        disc = self._create_discrepancy(
            nvars_x=1, nqoi=2, nscaled_qoi=1
        )

        np.random.seed(42)
        nsamples = 4
        fixed_samples = bkd.asarray(np.random.uniform(-0.8, 0.8, (2, nsamples)))

        nactive = disc.hyp_list().nactive_params()
        nqoi_disc = disc.nqoi()

        # Use weighted sum to reduce to nqoi=nsamples for DerivativeChecker
        np.random.seed(88)
        weights = bkd.asarray(np.random.randn(nqoi_disc, 1))

        def eval_fn(params: Array) -> Array:
            p = bkd.flatten(params)
            disc.hyp_list().set_active_values(p)
            disc._sync_from_hyp_list()
            vals = disc(fixed_samples)  # (nqoi, nsamples)
            weighted = (weights.T @ vals)  # (1, nsamples)
            return weighted.T  # (nsamples, 1)

        def jac_fn(params: Array) -> Array:
            p = bkd.flatten(params)
            disc.hyp_list().set_active_values(p)
            disc._sync_from_hyp_list()
            j = disc.jacobian_wrt_params(fixed_samples)  # (nsamples, nqoi, nactive)
            wj = bkd.einsum("snp,n->sp", j, bkd.flatten(weights))
            return wj  # (nsamples, nactive)

        wrapped = FunctionWithJacobianFromCallable(
            nqoi=nsamples, nvars=nactive,
            fun=eval_fn, jacobian=jac_fn, bkd=bkd,
        )
        checker = DerivativeChecker(wrapped)

        current_params = disc.hyp_list().get_active_values()
        sample = bkd.reshape(current_params, (-1, 1))

        errors = checker.check_derivatives(sample)
        ratio = checker.error_ratio(errors[0])
        self.assertLess(float(bkd.to_numpy(ratio)), 1e-6)

    def test_basis_matrix_consistency(self) -> None:
        """basis_matrix @ coefficients should equal __call__ output."""
        bkd = self._bkd
        disc = self._create_discrepancy(
            nvars_x=1, nqoi=1, nscaled_qoi=1
        )

        np.random.seed(99)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))

        # The basis_matrix builds the augmented design matrix
        phi = disc.basis_matrix(samples)  # (nsamples, total_nterms)
        coef = disc.get_coefficients()  # (total_nterms,)
        coef_2d = bkd.reshape(coef, (-1, 1))

        predicted = (phi @ coef_2d).T  # (1, nsamples)
        expected = disc(samples)  # (1, nsamples)

        bkd.assert_allclose(predicted, expected, rtol=1e-12)


# --- Concrete backend test classes ---

class TestDiscrepancyNumpy(
    TestMultiplicativeAdditiveDiscrepancy[NDArray[Any]]
):
    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestDiscrepancyTorch(
    TestMultiplicativeAdditiveDiscrepancy[torch.Tensor]
):
    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main(verbosity=2)
