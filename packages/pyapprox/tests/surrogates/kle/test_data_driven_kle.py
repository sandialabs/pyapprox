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
    RandomizedSnapshotSolver,
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


class TestTheTwoBases:
    """The weighted basis is the unweighted one times ``sqrt(eigenvalues)``.

    Only the unweighted basis is stored; the weighted form is rebuilt on
    request and ``__call__`` avoids it entirely by scaling coefficients
    instead. The tests here pin the relationship those three paths must
    agree on, which nothing covered before.
    """

    def _kle(self, bkd, ncoords=30, nsamples=20, nterms=4):
        rng = np.random.RandomState(0)
        snapshots = bkd.array(
            rng.normal(size=(ncoords, 8)) @ rng.normal(size=(8, nsamples))
        )
        return DataDrivenKLE(
            snapshots, nterms=nterms, bkd=bkd, center=True
        )

    def test_weighted_is_the_unweighted_basis_scaled(self, bkd) -> None:
        kle = self._kle(bkd)
        bkd.assert_allclose(
            kle.weighted_eigenvectors(),
            kle.eigenvectors() * bkd.sqrt(kle.eigenvalues()),
            atol=1e-14,
        )

    def test_scaling_is_per_mode_not_per_coordinate(self, bkd) -> None:
        """The orientation a broadcast would get wrong without erroring.

        ``sqrt_eig_vals`` is ``(nterms,)`` and the coefficients are
        ``(nterms, nsamples)``, so scaling must multiply *rows*.  A
        ``(ncoords, nterms)`` basis and a ``(nterms,)`` vector broadcast
        along the last axis either way, so a transposed convention would
        still produce an array of the right shape and wrong content.
        Checked against an explicit per-mode construction rather than
        against another vectorized expression.
        """
        kle = self._kle(bkd, ncoords=30, nterms=4)
        basis = bkd.to_numpy(kle.eigenvectors())
        scales = bkd.to_numpy(bkd.sqrt(kle.eigenvalues()))
        expected = np.stack(
            [basis[:, a] * scales[a] for a in range(basis.shape[1])],
            axis=1,
        )
        bkd.assert_allclose(
            kle.weighted_eigenvectors(), bkd.array(expected), atol=1e-14
        )

    def test_a_single_mode_is_still_one_scale_per_mode(
        self, bkd
    ) -> None:
        """The case the shape guard must not reject.

        With one mode the scale vector is legitimately length one, which
        is also what a collapsed vector would look like. The guard
        distinguishes them by comparing against ``nterms`` rather than
        by rejecting length one, so this configuration has to keep
        working.
        """
        kle = self._kle(bkd, nterms=1)
        coef = bkd.array(np.ones((1, 3)))
        assert kle(coef).shape == (30, 3)
        bkd.assert_allclose(
            kle.weighted_eigenvectors(),
            kle.eigenvectors() * bkd.sqrt(kle.eigenvalues()),
            atol=1e-14,
        )

    def test_call_matches_the_weighted_basis(self, bkd) -> None:
        """``__call__`` folds the scaling into the coefficients.

        It never forms the weighted basis, so this is the check that the
        cheaper route computes the same field.
        """
        kle = self._kle(bkd, nterms=4)
        coef = bkd.array(np.random.RandomState(1).normal(size=(4, 7)))
        expected = (
            kle.mean_field()[:, None]
            + kle.weighted_eigenvectors() @ coef
        )
        bkd.assert_allclose(kle(coef), expected, atol=1e-13)


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

    Two guards catch that, and the tests below say which is which. A
    count above ``min(nstates, nsamples)`` is refused from the shape
    alone, before any work. A count within that bound but above the
    *numerical* rank is invisible to counting and is caught from the
    spectrum instead.
    """

    def _data(self, bkd, ncoords=10, nsamples=6):
        return bkd.asarray(np.random.rand(ncoords, nsamples))

    def test_rejects_more_terms_than_samples(self, bkd) -> None:
        """Refused from the shape, so the decomposition never runs."""
        with pytest.raises(ValueError, match="exceeds the rank"):
            DataDrivenKLE(self._data(bkd), nterms=7, bkd=bkd)

    def test_rejects_more_terms_than_coords(self, bkd) -> None:
        with pytest.raises(ValueError, match="exceeds the rank"):
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

        The case the cheap guard cannot see, and the reason both exist.
        Data centered by the caller arrives as an ordinary matrix whose
        shape says ``nsamples`` modes are available, so ``nterms=6``
        passes the counting check; the rank drop to ``nsamples - 1``
        appears only in the eigenvalues. Matched on the eigenvalue
        message to pin *which* guard fires, since a shape-based refusal
        here would mean the spectrum was never consulted.
        """
        data = self._data(bkd)
        centered = data - bkd.reshape(
            bkd.mean(data, axis=1), (data.shape[0], 1)
        )
        with pytest.raises(ValueError, match="eigenvalues at or below"):
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


class TestTruncationArguments:
    """The truncation policies, reachable from the constructor.

    A caller wanting a variance fraction asks for one here rather than
    computing a count themselves, and centering is an argument rather
    than an obligation to subtract the mean before calling.
    """

    def _data(self, bkd, ncoords=10, nsamples=8):
        rng = np.random.RandomState(0)
        return bkd.asarray(rng.standard_normal((ncoords, nsamples)))

    def test_an_explicit_count_reaches_the_solver(self, bkd) -> None:
        """Not merely applied afterwards, which two solvers cannot tell apart.

        An approximate solver never forms the whole spectrum, so it has
        no rank to report and refuses to be asked for one. Requesting
        the count up front is what makes it usable here, and using one
        as the eigensolver is how this test detects that the count was
        passed rather than the result sliced.
        """
        kle = DataDrivenKLE(
            self._data(bkd),
            nterms=3,
            bkd=bkd,
            center=True,
            eigensolver=RandomizedSnapshotSolver(bkd, seed=0),
        )
        assert kle.nterms() == 3
        assert kle.eigenvectors().shape == (10, 3)

    def test_a_variance_fraction_still_asks_for_everything(
        self, bkd
    ) -> None:
        """It is a question about the discarded modes too.

        The fraction cannot be resolved without the total variance, so
        this policy keeps requesting the full spectrum -- and is
        therefore available only from an exact solver.
        """
        kle = DataDrivenKLE(
            self._data(bkd),
            variance_fraction=0.9,
            bkd=bkd,
            center=True,
        )
        assert 0 < kle.nterms() <= 8

    def test_variance_fraction_keeps_fewer_modes_than_the_rank(
        self, bkd
    ) -> None:
        data = self._data(bkd)
        full = DataDrivenKLE(data, bkd=bkd).nterms()
        partial = DataDrivenKLE(
            data, variance_fraction=0.5, bkd=bkd
        ).nterms()
        assert 1 <= partial < full

    def test_variance_fraction_is_monotone(self, bkd) -> None:
        data = self._data(bkd)
        counts = [
            DataDrivenKLE(data, variance_fraction=f, bkd=bkd).nterms()
            for f in (0.3, 0.6, 0.9)
        ]
        assert counts[0] <= counts[1] <= counts[2]

    def test_rejects_nterms_with_variance_fraction(self, bkd) -> None:
        with pytest.raises(ValueError, match="not both"):
            DataDrivenKLE(
                self._data(bkd), nterms=3, variance_fraction=0.9, bkd=bkd
            )

    def test_center_subtracts_the_sample_mean(self, bkd) -> None:
        data = self._data(bkd)
        kle = DataDrivenKLE(data, center=True, bkd=bkd)
        bkd.assert_allclose(
            kle.mean_field(), bkd.mean(data, axis=1), rtol=1e-12
        )

    def test_center_matches_centering_by_hand(self, bkd) -> None:
        """The argument is a convenience, not a different computation."""
        data = self._data(bkd)
        mean = bkd.mean(data, axis=1)
        by_hand = DataDrivenKLE(
            data - mean[:, None], mean, nterms=4, bkd=bkd
        )
        by_arg = DataDrivenKLE(data, nterms=4, center=True, bkd=bkd)
        bkd.assert_allclose(
            by_arg.eigenvectors(), by_hand.eigenvectors(), rtol=1e-12
        )
        bkd.assert_allclose(
            by_arg.eigenvalues(), by_hand.eigenvalues(), rtol=1e-12
        )

    def test_center_costs_one_mode(self, bkd) -> None:
        """Subtracting the mean makes the columns linearly dependent."""
        data = self._data(bkd)
        assert (
            DataDrivenKLE(data, center=True, bkd=bkd).nterms()
            == DataDrivenKLE(data, bkd=bkd).nterms() - 1
        )

    def test_rejects_center_with_an_explicit_mean(self, bkd) -> None:
        with pytest.raises(ValueError, match="not both"):
            DataDrivenKLE(
                self._data(bkd), 3.0, center=True, bkd=bkd
            )

    def test_centered_realizations_are_about_the_mean(self, bkd) -> None:
        data = self._data(bkd)
        kle = DataDrivenKLE(data, nterms=3, center=True, bkd=bkd)
        zeros = bkd.zeros((3, 1))
        bkd.assert_allclose(
            kle(zeros)[:, 0], bkd.mean(data, axis=1), rtol=1e-12
        )

    def test_center_by_subtracts_the_field_given(self, bkd) -> None:
        data = self._data(bkd)
        field = bkd.array(np.linspace(0.5, 1.5, data.shape[0]))
        kle = DataDrivenKLE(data, center_by=field, bkd=bkd)
        bkd.assert_allclose(kle.mean_field(), field, rtol=1e-12)

    def test_center_by_matches_centering_by_hand(self, bkd) -> None:
        """Same computation, with the subtraction done for the caller."""
        data = self._data(bkd)
        field = bkd.array(np.linspace(0.5, 1.5, data.shape[0]))
        by_hand = DataDrivenKLE(
            data - field[:, None], field, nterms=4, bkd=bkd
        )
        by_arg = DataDrivenKLE(data, nterms=4, center_by=field, bkd=bkd)
        bkd.assert_allclose(
            by_arg.eigenvectors(), by_hand.eigenvectors(), rtol=1e-12
        )
        bkd.assert_allclose(
            by_arg.eigenvalues(), by_hand.eigenvalues(), rtol=1e-12
        )

    def test_center_by_accepts_a_column(self, bkd) -> None:
        data = self._data(bkd)
        field = bkd.array(np.linspace(0.5, 1.5, data.shape[0]))
        column = bkd.reshape(field, (data.shape[0], 1))
        bkd.assert_allclose(
            DataDrivenKLE(data, center_by=column, bkd=bkd).mean_field(),
            field,
            rtol=1e-12,
        )

    def test_center_by_realizations_are_about_that_field(self, bkd) -> None:
        data = self._data(bkd)
        field = bkd.array(np.linspace(0.5, 1.5, data.shape[0]))
        kle = DataDrivenKLE(data, nterms=3, center_by=field, bkd=bkd)
        bkd.assert_allclose(
            kle(bkd.zeros((3, 1)))[:, 0], field, rtol=1e-12
        )

    def test_center_by_differs_from_the_sample_mean(self, bkd) -> None:
        """The point of the argument: a mean the data does not supply."""
        data = self._data(bkd)
        field = bkd.array(np.linspace(0.5, 1.5, data.shape[0]))
        by_field = DataDrivenKLE(data, nterms=3, center_by=field, bkd=bkd)
        by_sample = DataDrivenKLE(data, nterms=3, center=True, bkd=bkd)
        difference = float(
            bkd.to_numpy(
                bkd.sum(bkd.abs(by_field.mean_field() - by_sample.mean_field()))
            )
        )
        assert difference > 1e-6

    def test_rejects_center_by_with_center(self, bkd) -> None:
        data = self._data(bkd)
        with pytest.raises(ValueError, match="not both"):
            DataDrivenKLE(
                data,
                center=True,
                center_by=bkd.zeros((data.shape[0],)),
                bkd=bkd,
            )

    def test_rejects_center_by_with_an_explicit_mean(self, bkd) -> None:
        data = self._data(bkd)
        with pytest.raises(ValueError, match="not both"):
            DataDrivenKLE(
                data,
                3.0,
                center_by=bkd.zeros((data.shape[0],)),
                bkd=bkd,
            )


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
