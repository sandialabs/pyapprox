"""Tests for MultiplicativeAdditiveDiscrepancy model."""

import numpy as np

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.surrogates.affine.basis import MultiIndexBasis
from pyapprox.surrogates.affine.expansions import BasisExpansion
from pyapprox.surrogates.affine.indices import (
    compute_hyperbolic_indices,
)
from pyapprox.surrogates.affine.univariate import MonomialBasis1D
from pyapprox.surrogates.mfnets.discrepancy import (
    MultiplicativeAdditiveDiscrepancy,
)


def _create_expansion(
    bkd, nvars: int, nqoi: int, max_level: int = 2
) -> BasisExpansion:
    bases_1d = [MonomialBasis1D(bkd) for _ in range(nvars)]
    indices = compute_hyperbolic_indices(nvars, max_level, 1.0, bkd)
    basis = MultiIndexBasis.__new__(MultiIndexBasis)
    MultiIndexBasis.__init__(basis, bases_1d, bkd, indices)
    return BasisExpansion(basis, bkd, nqoi=nqoi)


class TestMultiplicativeAdditiveDiscrepancy:

    def _create_discrepancy(
        self,
        bkd,
        nvars_x: int = 1,
        nqoi: int = 1,
        nscaled_qoi: int = 1,
        scale_level: int = 1,
        delta_level: int = 2,
    ) -> MultiplicativeAdditiveDiscrepancy:
        # Scaling models: one per output QoI
        scalings = []
        for ii in range(nqoi):
            sm = _create_expansion(bkd, nvars_x, nscaled_qoi, scale_level)
            np.random.seed(100 + ii)
            sm.set_coefficients(bkd.asarray(np.random.randn(sm.nterms(), nscaled_qoi)))
            scalings.append(sm)

        # Delta model
        delta = _create_expansion(bkd, nvars_x, nqoi, delta_level)
        np.random.seed(200)
        delta.set_coefficients(bkd.asarray(np.random.randn(delta.nterms(), nqoi)))

        return MultiplicativeAdditiveDiscrepancy(scalings, delta, nscaled_qoi, bkd)

    def test_construction(self, bkd) -> None:
        disc = self._create_discrepancy(bkd)
        assert disc.nvars() == 2  # 1 (x) + 1 (q)
        assert disc.nqoi() == 1

    def test_eval_against_manual(self, bkd) -> None:
        """Test __call__ matches manual computation."""
        disc = self._create_discrepancy(bkd, nvars_x=1, nqoi=1, nscaled_qoi=1)

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

    def test_eval_multi_qoi(self, bkd) -> None:
        """Test with nqoi=2, nscaled_qoi=1."""
        disc = self._create_discrepancy(bkd, nvars_x=1, nqoi=2, nscaled_qoi=1)
        assert disc.nvars() == 2
        assert disc.nqoi() == 2

        np.random.seed(99)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 6)))
        result = disc(samples)
        assert result.shape[0] == 2
        assert result.shape[1] == 6

    def test_hyp_list(self, bkd) -> None:
        disc = self._create_discrepancy(bkd)
        hyps = disc.hyp_list()
        # Total params = sum(scaling params) + delta params
        expected = (
            sum(s.hyp_list().nparams() for s in disc.scaling_models())
            + disc.delta_model().hyp_list().nparams()
        )
        assert hyps.nparams() == expected

    def test_jacobian_wrt_params_with_derivative_checker(self, bkd) -> None:
        """Validate jacobian_wrt_params using DerivativeChecker."""
        disc = self._create_discrepancy(
            bkd, nvars_x=1, nqoi=1, nscaled_qoi=1, scale_level=1, delta_level=2
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

        def eval_fn(params):
            """Weighted sum of disc output over QoI, stacked over samples."""
            p = bkd.flatten(params)
            disc.hyp_list().set_active_values(p)
            disc._sync_from_hyp_list()
            vals = disc(fixed_samples)  # (nqoi, nsamples)
            # Weighted sum over QoI: (1, nsamples)
            weighted = weights.T @ vals  # (1, nsamples)
            return weighted.T  # (nsamples, 1)

        def jac_fn(params):
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
        assert float(bkd.to_numpy(ratio)) < 1e-6

    def test_jacobian_wrt_params_multi_qoi(self, bkd) -> None:
        """Validate jacobian_wrt_params for nqoi=2."""
        disc = self._create_discrepancy(bkd, nvars_x=1, nqoi=2, nscaled_qoi=1)

        np.random.seed(42)
        nsamples = 4
        fixed_samples = bkd.asarray(np.random.uniform(-0.8, 0.8, (2, nsamples)))

        nactive = disc.hyp_list().nactive_params()
        nqoi_disc = disc.nqoi()

        # Use weighted sum to reduce to nqoi=nsamples for DerivativeChecker
        np.random.seed(88)
        weights = bkd.asarray(np.random.randn(nqoi_disc, 1))

        def eval_fn(params):
            p = bkd.flatten(params)
            disc.hyp_list().set_active_values(p)
            disc._sync_from_hyp_list()
            vals = disc(fixed_samples)  # (nqoi, nsamples)
            weighted = weights.T @ vals  # (1, nsamples)
            return weighted.T  # (nsamples, 1)

        def jac_fn(params):
            p = bkd.flatten(params)
            disc.hyp_list().set_active_values(p)
            disc._sync_from_hyp_list()
            j = disc.jacobian_wrt_params(fixed_samples)  # (nsamples, nqoi, nactive)
            wj = bkd.einsum("snp,n->sp", j, bkd.flatten(weights))
            return wj  # (nsamples, nactive)

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
        assert float(bkd.to_numpy(ratio)) < 1e-6

    def test_basis_matrix_consistency(self, bkd) -> None:
        """basis_matrix @ coefficients should equal __call__ output."""
        disc = self._create_discrepancy(bkd, nvars_x=1, nqoi=1, nscaled_qoi=1)

        np.random.seed(99)
        samples = bkd.asarray(np.random.uniform(-1, 1, (2, 10)))

        # The basis_matrix builds the augmented design matrix
        phi = disc.basis_matrix(samples)  # (nsamples, total_nterms)
        coef = disc.get_coefficients()  # (total_nterms,)
        coef_2d = bkd.reshape(coef, (-1, 1))

        predicted = (phi @ coef_2d).T  # (1, nsamples)
        expected = disc(samples)  # (1, nsamples)

        bkd.assert_allclose(predicted, expected, rtol=1e-12)
