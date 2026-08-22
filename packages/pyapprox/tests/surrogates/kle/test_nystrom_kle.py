"""Tests for the Nystrom KLE.

The capability under test is evaluation at points the basis was *not*
built on, so the tests that matter here evaluate away from both the
landmark set and the collocation set. Agreement at the collocation
points is necessary but proves little: the textbook Nystrom formula,
which solves a different problem and measures 93% eigenvalue error,
also reproduces the basis exactly at the points it was built from.

Fixture choice is load-bearing. A landmark count only binds when the
operator's numerical rank exceeds it, and a squared exponential is rank
15 at lengthscale 0.3 on 100 points however many landmarks are asked
for -- a convergence test on that kernel would measure the rank ceiling
rather than the landmarks. Matern-3/2 and exponential kernels are full
rank at these sizes, so they are used wherever the landmark count is
the variable, and the precondition is asserted rather than assumed.
"""

import numpy as np
import pytest
from pyapprox.surrogates.kernels.matern import (
    Matern32Kernel,
    SquaredExponentialKernel,
)
from pyapprox.surrogates.kle.mesh_kle import MeshKLE
from pyapprox.surrogates.kle.nystrom_kle import (
    create_nystrom_kle,
)
from pyapprox.surrogates.kle.protocols import KLEProtocol


def _coords(bkd, npoints=100):
    return bkd.array(np.linspace(0.0, 1.0, npoints)[None, :])


def _out_of_sample(bkd, npoints=100):
    """Midpoints: in neither the collocation set nor any subset of it."""
    grid = np.linspace(0.0, 1.0, npoints)
    return bkd.array(((grid[:-1] + grid[1:]) / 2)[None, :])


def _full_rank_kernel(bkd, lenscale=0.2):
    """Matern-3/2: full numerical rank at these sizes.

    Required wherever the landmark count is the variable under test.
    """
    return Matern32Kernel(bkd.full((1,), lenscale), (0.01, 100.0), 1, bkd)


def _low_rank_kernel(bkd, lenscale=0.3):
    """Squared exponential: numerical rank 15 at 100 points."""
    return SquaredExponentialKernel(
        bkd.full((1,), lenscale), (0.01, 100.0), 1, bkd
    )


def _weights(bkd, npoints=100, seed=0):
    np.random.seed(seed)
    weights = np.random.uniform(0.5, 1.5, npoints)
    return bkd.array(weights / weights.sum())


def _numerical_rank(bkd, kernel, coords):
    kmat = bkd.to_numpy(kernel(coords, coords))
    eigvals = np.linalg.eigvalsh(kmat)[::-1]
    return int(
        (eigvals > eigvals.max() * kmat.shape[0] * np.finfo(float).eps).sum()
    )


class TestOutOfSampleEvaluation:
    """The reason this class exists, tested where it applies."""

    @pytest.mark.parametrize("weighted", [False, True])
    def test_mercer_reconstruction_away_from_the_basis(
        self, bkd, weighted
    ) -> None:
        r"""``sum_k lam_k phi_k(x) phi_k(y)`` must approach ``C(x, y)``.

        Checked at midpoints, so every evaluation point is outside both
        the landmark set and the collocation set. Compared against the
        kernel itself rather than against another implementation, so a
        misunderstanding shared by both code paths cannot pass.

        Both weightings are exercised because the weighted path has two
        failure modes the unweighted one cannot show: landmark weights
        being rescaled, and weights being applied at the evaluation
        points where they should cancel. Both yield plausible output.
        """
        coords, kernel = _coords(bkd), _low_rank_kernel(bkd)
        kwargs = {"quad_weights": _weights(bkd)} if weighted else {}
        kle = create_nystrom_kle(
            kernel, coords, 12, bkd, nlandmarks=60, **kwargs
        )
        query = _out_of_sample(bkd)
        basis = bkd.to_numpy(kle.eigenvectors_at(query))
        eigvals = bkd.to_numpy(kle.eigenvalues())
        reconstructed = (basis * eigvals[None, :]) @ basis.T
        exact = bkd.to_numpy(kernel(query, query))
        error = np.abs(reconstructed - exact).max() / np.abs(exact).max()
        assert error < 1e-8, (
            f"Mercer reconstruction at out-of-sample points is {error:.2e}; "
            "the extension is not reproducing the covariance"
        )

    def test_evaluate_at_returns_field_values(self, bkd) -> None:
        """Shape and mean handling at points outside the basis."""
        coords, kernel = _coords(bkd), _low_rank_kernel(bkd)
        kle = create_nystrom_kle(
            kernel, coords, 8, bkd, nlandmarks=40, mean_field=3.0
        )
        query = _out_of_sample(bkd)
        coef = bkd.array(np.random.RandomState(0).normal(size=(8, 4)))
        field = kle.evaluate_at(query, coef)
        assert field.shape == (query.shape[1], 4)
        # with zero coefficients the field is the mean everywhere
        zero = bkd.full((8, 1), 0.0)
        bkd.assert_allclose(
            kle.evaluate_at(query, zero),
            bkd.full((query.shape[1], 1), 3.0),
            rtol=1e-12,
        )

    def test_agrees_with_dense_kle_at_collocation(self, bkd) -> None:
        """Necessary but not sufficient, and labelled as such.

        The textbook Nystrom formula also passes this while being wrong
        by 93% on the eigenvalues, which is why the out-of-sample test
        above carries the weight.
        """
        coords, kernel = _coords(bkd), _low_rank_kernel(bkd)
        weights = _weights(bkd)
        reference = MeshKLE(
            coords, kernel, nterms=10, quad_weights=weights, bkd=bkd
        )
        kle = create_nystrom_kle(
            kernel, coords, 10, bkd, quad_weights=weights, nlandmarks=60
        )
        lhs = bkd.to_numpy(reference.eigenvectors())
        rhs = bkd.to_numpy(kle.eigenvectors_at(coords))
        lhs_proj = lhs @ np.linalg.pinv(lhs)
        rhs_proj = rhs @ np.linalg.pinv(rhs)
        assert np.abs(lhs_proj - rhs_proj).max() < 1e-6


class TestConvergesInLandmarks:
    """More landmarks, closer to the full operator."""

    def test_fixture_rank_exceeds_the_landmark_counts(self, numpy_bkd):
        """Guards the test below from measuring the wrong ceiling.

        A landmark count only binds while the operator has directions
        left to supply. If this fails, the convergence test underneath
        is measuring the kernel's numerical rank rather than the
        landmarks, and would pass or fail for reasons unrelated to the
        extension.
        """
        bkd = numpy_bkd
        coords, kernel = _coords(bkd), _full_rank_kernel(bkd)
        assert _numerical_rank(bkd, kernel, coords) > 80

    def test_error_decreases_with_more_landmarks(self, numpy_bkd) -> None:
        """Out-of-sample accuracy improves as landmarks are added.

        Two conditions have to hold for the landmark count to be the
        variable, and both were got wrong before being measured.

        The kernel must have rank to spare, hence Matern-3/2 rather than
        a squared exponential whose rank ceiling of 15 would bind first.

        And ``nterms`` must be large enough that truncation is not the
        dominant error. It is not a free parameter here: measured on
        this fixture, ``nterms=8`` gives 4.41e-02 at 40 landmarks and
        4.41e-02 at 80 -- flat, because reconstructing a full covariance
        from 8 modes of a rough kernel leaves a floor no number of
        landmarks can lower. At ``nterms=40`` the same comparison reads
        7.83e-04 against 2.27e-04, so the landmarks bind and the test
        measures what it claims to.
        """
        bkd = numpy_bkd
        coords, kernel = _coords(bkd), _full_rank_kernel(bkd)
        query = _out_of_sample(bkd)
        exact = bkd.to_numpy(kernel(query, query))
        errors = []
        for nlandmarks in (40, 60, 80):
            kle = create_nystrom_kle(
                kernel, coords, 40, bkd, nlandmarks=nlandmarks
            )
            basis = bkd.to_numpy(kle.eigenvectors_at(query))
            eigvals = bkd.to_numpy(kle.eigenvalues())
            reconstructed = (basis * eigvals[None, :]) @ basis.T
            errors.append(
                np.abs(reconstructed - exact).max() / np.abs(exact).max()
            )
        assert errors[0] > errors[-1], (
            f"out-of-sample errors {errors} did not improve between the "
            "smallest and largest landmark counts"
        )


class TestRankGuard:
    """Requesting more terms than the landmark block can supply."""

    def test_refuses_when_effective_rank_is_too_small(self, bkd) -> None:
        """The guard is on effective rank, not on the landmark count.

        A squared exponential at lengthscale 0.3 has numerical rank 15,
        so 40 landmarks cannot supply 30 terms however many points are
        offered. Comparing nterms against nlandmarks would let this
        through and return silently degenerate modes.
        """
        coords, kernel = _coords(bkd), _low_rank_kernel(bkd)
        with pytest.raises(ValueError, match="usable directions"):
            create_nystrom_kle(kernel, coords, 30, bkd, nlandmarks=40)

    def test_error_names_the_measured_rank(self, bkd) -> None:
        """The message must be actionable, not merely a refusal."""
        coords, kernel = _coords(bkd), _low_rank_kernel(bkd)
        with pytest.raises(ValueError) as excinfo:
            create_nystrom_kle(kernel, coords, 30, bkd, nlandmarks=40)
        message = str(excinfo.value)
        assert "effective rank" in message
        assert "reduce" in message.lower()


class TestMeanField:
    """An array mean has no value at a point outside its own set."""

    def test_accepts_a_callable(self, bkd) -> None:
        coords, kernel = _coords(bkd), _low_rank_kernel(bkd)

        def mean_field(pts):
            return bkd.flatten(pts[0:1, :]) * 2.0

        kle = create_nystrom_kle(
            kernel, coords, 6, bkd, nlandmarks=30, mean_field=mean_field
        )
        query = _out_of_sample(bkd)
        field = kle.evaluate_at(query, bkd.full((6, 1), 0.0))
        bkd.assert_allclose(
            bkd.flatten(field), bkd.flatten(query[0:1, :]) * 2.0, rtol=1e-12
        )

    def test_rejects_an_array(self, bkd) -> None:
        """The representation MeshKLE uses cannot work here."""
        coords, kernel = _coords(bkd), _low_rank_kernel(bkd)
        with pytest.raises(TypeError, match="scalar or a callable"):
            create_nystrom_kle(
                kernel,
                coords,
                6,
                bkd,
                nlandmarks=30,
                mean_field=bkd.full((100,), 1.0),
            )


class TestProtocol:
    """Satisfies the KLE protocol at the landmark set."""

    def test_satisfies_kle_protocol(self, bkd) -> None:
        coords, kernel = _coords(bkd), _low_rank_kernel(bkd)
        kle = create_nystrom_kle(kernel, coords, 8, bkd, nlandmarks=40)
        assert isinstance(kle, KLEProtocol)

    def test_eigenvalues_descending_and_nonnegative(self, bkd) -> None:
        coords, kernel = _coords(bkd), _low_rank_kernel(bkd)
        kle = create_nystrom_kle(kernel, coords, 8, bkd, nlandmarks=40)
        eigvals = bkd.to_numpy(kle.eigenvalues())
        assert np.all(np.diff(eigvals) <= 1e-12)
        assert np.all(eigvals >= 0.0)

    def test_landmark_count_may_fall_short_of_the_request(
        self, bkd
    ) -> None:
        """``nlandmarks`` is a request, not a guarantee.

        The pivoted factorization stops once the residual trace falls
        below its tolerance, so a low-rank kernel yields fewer pivots
        than asked for -- 26 of a requested 80 on a squared exponential.
        That is correct behaviour, and worth pinning so the accessor is
        not later "fixed" to report the request instead.
        """
        coords, kernel = _coords(bkd), _low_rank_kernel(bkd)
        kle = create_nystrom_kle(kernel, coords, 8, bkd, nlandmarks=80)
        assert kle.nvars() <= 80
        assert kle.landmark_coords().shape[1] == kle.nvars()
