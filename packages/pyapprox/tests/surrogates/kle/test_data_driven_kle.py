"""Tests for DataDrivenKLE."""

import numpy as np
import pytest
from pyapprox.surrogates.affine.univariate.globalpoly import (
    LegendrePolynomial1D,
)
from pyapprox.surrogates.affine.univariate.globalpoly.quadrature import (
    GaussQuadratureRule,
)
from pyapprox.surrogates.kernels.matern import ExponentialKernel
from pyapprox.surrogates.kle.data_driven_kle import DataDrivenKLE
from pyapprox.surrogates.kle.mesh_kle import MeshKLE
from pyapprox.surrogates.kle.snapshot_eigensolvers import (
    MethodOfSnapshotsSolver,
    SVDSnapshotSolver,
)
from pyapprox.util.linalg.inner_product import (
    DiagonalInnerProduct,
    MassInnerProduct,
)
from scipy.sparse import diags

from tests._helpers.markers import slow_test


def _gauss_legendre_quad(lb, ub, npts, bkd):
    """Gauss-Legendre quadrature on [lb, ub] for Lebesgue integration."""
    poly = LegendrePolynomial1D(bkd)
    poly.set_nterms(npts)
    quad_rule = GaussQuadratureRule(poly)
    pts, wts = quad_rule(npts)
    dom_len = ub - lb
    half_len = dom_len / 2.0
    mid = (lb + ub) / 2.0
    pts = pts * half_len + mid
    wts = wts * dom_len
    return pts, wts[:, 0]


class TestDataDrivenKLE:

    def test_data_driven_kle_vs_mesh_kle(self, bkd) -> None:
        """Build MeshKLE (no weights), generate 10k samples, build
        DataDrivenKLE from realizations, verify eigenvalues match.
        """
        nterms = 3
        level = 6
        len_scale, sigma = 1.0, 1.0
        lb, ub = 0.0, 2.0
        npts = 2**level + 1

        mesh_coords, quad_weights = _gauss_legendre_quad(lb, ub, npts, bkd)

        lenscale_arr = bkd.array([len_scale])
        kernel = ExponentialKernel(lenscale_arr, (0.01, 100.0), 1, bkd)

        kle = MeshKLE(
            mesh_coords,
            kernel,
            sigma=sigma,
            nterms=nterms,
            quad_weights=None,
            bkd=bkd,
        )

        nsamples = 10000
        samples = bkd.asarray(np.random.normal(0.0, 1.0, (nterms, nsamples)))
        kle_realizations = kle(samples)

        kle_data = DataDrivenKLE(
            kle_realizations,
            nterms=nterms,
            bkd=bkd,
        )
        bkd.assert_allclose(
            kle_data._sqrt_eig_vals,
            kle._sqrt_eig_vals,
            atol=2e-2,
            rtol=2e-2,
        )

    def test_data_driven_kle_with_weights(self, bkd) -> None:
        """MeshKLE with quad weights -> generate samples ->
        DataDrivenKLE with same weights -> eigenvalues match.
        """
        nterms = 3
        level = 6
        len_scale, sigma = 1.0, 1.0
        lb, ub = 0.0, 2.0
        npts = 2**level + 1

        mesh_coords, quad_weights = _gauss_legendre_quad(lb, ub, npts, bkd)

        lenscale_arr = bkd.array([len_scale])
        kernel = ExponentialKernel(lenscale_arr, (0.01, 100.0), 1, bkd)

        kle = MeshKLE(
            mesh_coords,
            kernel,
            sigma=sigma,
            nterms=nterms,
            quad_weights=quad_weights,
            bkd=bkd,
        )

        nsamples = 10000
        samples = bkd.asarray(np.random.normal(0.0, 1.0, (nterms, nsamples)))
        kle_realizations = kle(samples)

        kle_data = DataDrivenKLE(
            kle_realizations,
            nterms=nterms,
            quad_weights=quad_weights,
            bkd=bkd,
        )
        bkd.assert_allclose(
            kle_data._sqrt_eig_vals,
            kle._sqrt_eig_vals,
            atol=1e-2,
            rtol=1e-2,
        )


class TestTermValidation:
    """nterms must not exceed what the sample matrix can supply.

    Requesting more terms than the data has rank used to succeed and
    return zero columns: modes the caller asked for, scaled by a zero
    singular value, indistinguishable from a genuinely tiny mode.
    """

    def _data(self, bkd, ncoords=10, nsamples=6):
        return bkd.asarray(np.random.rand(ncoords, nsamples))

    def test_rejects_more_terms_than_samples(self, bkd) -> None:
        with pytest.raises(ValueError, match="rank of the sample matrix"):
            DataDrivenKLE(self._data(bkd), nterms=7, bkd=bkd)

    def test_rejects_more_terms_than_coords(self, bkd) -> None:
        with pytest.raises(ValueError, match="rank of the sample matrix"):
            DataDrivenKLE(
                self._data(bkd, ncoords=4, nsamples=20), nterms=5, bkd=bkd
            )

    def test_rejects_nonpositive_terms(self, bkd) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            DataDrivenKLE(self._data(bkd), nterms=0, bkd=bkd)

    def test_default_is_the_rank_not_ncoords(self, bkd) -> None:
        """ncoords was the old default and exceeds the rank whenever
        there are fewer samples than coordinates -- the common case."""
        kle = DataDrivenKLE(self._data(bkd, ncoords=10, nsamples=6), bkd=bkd)
        assert kle.nterms() == 6

    def test_rejects_centered_data_at_full_sample_count(self, bkd) -> None:
        """Centering costs one term, which only the spectrum reveals.

        Counting arguments cannot catch this: the class is handed a
        matrix without being told whether a mean was removed, so the
        rank drop from nsamples to nsamples - 1 shows up only in the
        eigenvalues. Rejected by the same shared check that guards the
        kernel-driven solvers.
        """
        data = self._data(bkd)
        centered = data - bkd.reshape(
            bkd.mean(data, axis=1), (data.shape[0], 1)
        )
        with pytest.raises(ValueError, match="rounding error"):
            DataDrivenKLE(centered, nterms=6, bkd=bkd)

    def test_accepts_centered_data_one_term_lower(self, bkd) -> None:
        data = self._data(bkd)
        centered = data - bkd.reshape(
            bkd.mean(data, axis=1), (data.shape[0], 1)
        )
        assert DataDrivenKLE(centered, nterms=5, bkd=bkd).nterms() == 5

    def test_rejects_constant_data(self, bkd) -> None:
        with pytest.raises(ValueError, match="no usable modes"):
            DataDrivenKLE(bkd.zeros((10, 6)), nterms=1, bkd=bkd)


class TestSampleConvergence:
    """The estimated spectrum must converge to the operator's own.

    A fixed-sample agreement check says the estimator was close once. It
    cannot tell an estimator that converges from one that is merely
    biased by a tolerable amount, which is what a rate measures.
    """

    @slow_test
    def test_eigenvalues_converge_at_monte_carlo_rate(self, bkd) -> None:
        r"""Error falls as :math:`n^{-1/2}`, the Monte Carlo rate.

        Sample eigenvalues are averages of random quantities, so the CLT
        fixes the error at :math:`O(\sigma/\sqrt{n})`. Note the rate is
        the square root of the familiar :math:`O(1/n)`, which is the
        *variance* of a Monte Carlo estimator rather than its error.

        The seeds are fixed, so the fitted rate is deterministic:
        -0.5008, reproducible bit-for-bit, which is why the tolerance
        can be tight. Averaging over 128 repeats per sample count is
        what buys that -- the rate estimate is itself noisy, and at 16
        repeats the same measurement gives -0.23. Do not reduce the
        repeat count without re-measuring; the band below has about
        0.02 of margin on each side.
        """
        nterms, npts, nrepeats = 3, 17, 128
        counts = (500, 2000, 8000)

        coords = bkd.array(np.linspace(0.0, 2.0, npts)[None, :])
        kernel = ExponentialKernel(bkd.array([1.0]), (0.01, 100.0), 1, bkd)
        exact = MeshKLE(
            coords, kernel, sigma=1.0, nterms=nterms,
            quad_weights=None, bkd=bkd,
        )
        truth = exact.eigenvalues()
        scale = bkd.to_float(bkd.max(bkd.abs(truth)))

        errors = []
        for nsamples in counts:
            total = 0.0
            for seed in range(nrepeats):
                rng = np.random.RandomState(seed)
                coef = bkd.asarray(
                    rng.normal(0.0, 1.0, (nterms, nsamples))
                )
                estimated = DataDrivenKLE(
                    exact(coef), 0.0, False, nterms, None, bkd=bkd
                ).eigenvalues()
                total += bkd.to_float(
                    bkd.max(bkd.abs(estimated - truth))
                ) / scale
            errors.append(total / nrepeats)

        rate = float(np.polyfit(np.log(counts), np.log(errors), 1)[0])
        assert -0.52 < rate < -0.48, f"rate {rate} is not ~-0.5: {errors}"


class TestEigensolverInjection:
    """Parity with MeshKLE: how the basis is computed is swappable.

    Without this seam a caller whose metric has no cheap square root --
    an assembled FEM mass matrix -- could not use this class at all,
    because the decomposition was written inline.
    """

    def _data(self, bkd, ncoords=12, nsamples=8):
        return bkd.asarray(np.random.rand(ncoords, nsamples))

    def test_injected_solver_matches_the_default(self, bkd) -> None:
        data = self._data(bkd)
        default = DataDrivenKLE(data, nterms=4, bkd=bkd)
        injected = DataDrivenKLE(
            data, nterms=4, bkd=bkd, eigensolver=SVDSnapshotSolver(bkd)
        )
        bkd.assert_allclose(
            injected.eigenvalues(), default.eigenvalues(), rtol=1e-12
        )

    def test_the_two_solvers_give_the_same_kle(self, bkd) -> None:
        """Swapping the algorithm must not change the field."""
        data = self._data(bkd)
        svd = DataDrivenKLE(
            data, nterms=4, bkd=bkd, eigensolver=SVDSnapshotSolver(bkd)
        )
        gram = DataDrivenKLE(
            data, nterms=4, bkd=bkd,
            eigensolver=MethodOfSnapshotsSolver(bkd),
        )
        bkd.assert_allclose(
            gram.eigenvalues(), svd.eigenvalues(), rtol=1e-8
        )
        coef = bkd.asarray(np.random.rand(4, 3))
        bkd.assert_allclose(gram(coef), svd(coef), rtol=1e-6, atol=1e-8)

    def test_metric_and_quad_weights_agree(self, bkd) -> None:
        """quad_weights is the diagonal special case of metric."""
        data = self._data(bkd)
        weights = bkd.asarray(np.linspace(0.5, 2.0, 12))
        by_weights = DataDrivenKLE(
            data, nterms=4, quad_weights=weights, bkd=bkd
        )
        by_metric = DataDrivenKLE(
            data, nterms=4, bkd=bkd,
            metric=DiagonalInnerProduct(weights, bkd),
        )
        bkd.assert_allclose(
            by_metric.eigenvectors(), by_weights.eigenvectors(), rtol=1e-12
        )

    def test_rejects_both_metric_and_quad_weights(self, bkd) -> None:
        weights = bkd.ones((12,))
        with pytest.raises(ValueError, match="not both"):
            DataDrivenKLE(
                self._data(bkd), nterms=4, quad_weights=weights, bkd=bkd,
                metric=DiagonalInnerProduct(weights, bkd),
            )

    def test_sparse_metric_works_through_the_gram_solver(self, bkd) -> None:
        """The case the inline SVD could not serve at all."""
        data = self._data(bkd)
        weights = np.linspace(0.5, 2.0, 12)
        kle = DataDrivenKLE(
            data, nterms=4, bkd=bkd,
            metric=MassInnerProduct(diags(weights), bkd),
        )
        by_diagonal = DataDrivenKLE(
            data, nterms=4, quad_weights=bkd.asarray(weights), bkd=bkd
        )
        bkd.assert_allclose(
            kle.eigenvalues(), by_diagonal.eigenvalues(), rtol=1e-8
        )


class TestSpectrumConventions:
    """Both scalings are reachable, and their relation is exact."""

    def test_eigenvalues_are_singular_values_scaled(self, bkd) -> None:
        nsamples = 6
        data = bkd.asarray(np.random.rand(10, nsamples))
        kle = DataDrivenKLE(data, nterms=4, bkd=bkd)
        bkd.assert_allclose(
            kle.eigenvalues(),
            kle.singular_values() ** 2 / (nsamples - 1),
            rtol=1e-12,
        )

    def test_singular_values_are_descending_and_positive(self, bkd) -> None:
        data = bkd.asarray(np.random.rand(10, 6))
        svals = DataDrivenKLE(data, nterms=4, bkd=bkd).singular_values()
        assert svals.shape == (4,)
        assert bool(bkd.all_bool(svals > 0.0))
        assert bool(bkd.all_bool(svals[:-1] >= svals[1:]))
