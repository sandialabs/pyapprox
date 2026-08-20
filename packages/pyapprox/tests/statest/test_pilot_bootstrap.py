"""Tests for bootstrapping pilot uncertainty through allocation.

The substantive tests here check the bootstrap against closed-form
sampling theory rather than merely asserting that some spread exists.
A bootstrap that resampled incorrectly -- per model rather than with a
shared index, say -- would still produce a nonzero spread, so "the
numbers vary" is not evidence of correctness.

Two oracles carry most of the weight. For pilot values drawn from a
known covariance, the sample covariance has asymptotic variance
``(S_ii S_jj + S_ij^2) / npilot``; and for Monte Carlo mean estimation
the tolerance allocation is closed form, ``n* = sigma^2 / tol^2``, so
the delta method gives the budget's variance directly. Both were
checked numerically before being written down here, agreeing with
simulation to within 3%.

Tolerances are deliberately loose. These are asymptotic approximations
and the bootstrap adds its own Monte Carlo error, so a tight threshold
would buy flakiness rather than rigor.
"""

import numpy as np
import pytest
from pyapprox.statest.cv_estimator import CVEstimator
from pyapprox.statest.groupacv import GroupACVEstimatorIS
from pyapprox.statest.mc_estimator import MCEstimator
from pyapprox.statest.pilot_bootstrap import (
    BootstrapSamples,
    CovarianceEstimatorProtocol,
    PilotReplicateProtocol,
    ResampledPilotValues,
    SampleCountBudgetSolver,
    bootstrap_budget_from_pilot,
    bootstrap_covariance_from_pilot,
)
from pyapprox.statest.statistics import MultiOutputMean
from pyapprox.statest.tolerance import (
    CVToleranceAllocator,
    MaxMarginalStandardErrorConstraint,
    MCToleranceAllocator,
)

from tests._helpers.markers import slow_test


def _gaussian_pilot(bkd, cov, npilot, seed=0):
    """Pilot values drawn from a known covariance.

    Shared inputs through correlated models, which is what the pilot
    represents: one set of realizations seen by every model.
    """
    np.random.seed(seed)
    chol = np.linalg.cholesky(cov)
    nmodels = cov.shape[0]
    draws = chol @ np.random.normal(0.0, 1.0, (nmodels, npilot))
    return [bkd.array(draws[ii: ii + 1, :]) for ii in range(nmodels)]


def _pilot_values(bkd, nmodels=3, npilot=200, nqoi=1, seed=0):
    np.random.seed(seed)
    shared = np.random.normal(0.0, 1.0, (nqoi, npilot))
    return [
        bkd.array(
            shared + 0.1 * (ii + 1) * np.random.normal(0.0, 1.0, (nqoi, npilot))
        )
        for ii in range(nmodels)
    ]


def _mc_budget_samples(bkd, pilot_values, tolerance, nbootstraps):
    """Bootstrap the Monte Carlo budget for a standard-error tolerance."""
    costs = bkd.array([1.0])
    solver = SampleCountBudgetSolver(
        lambda stat: MCToleranceAllocator(MCEstimator(stat, costs)),
        MaxMarginalStandardErrorConstraint(tolerance, bkd),
    )
    return bootstrap_budget_from_pilot(
        ResampledPilotValues(pilot_values, bkd),
        lambda: MultiOutputMean(1, bkd),
        solver,
        bkd,
        nbootstraps=nbootstraps,
    )


class TestSamplingTheoryOracles:
    """The bootstrap must reproduce known sampling distributions."""

    def test_replicate_covariance_matches_asymptotic_variance(
        self, numpy_bkd
    ) -> None:
        """Var(S_ij) = (S_ii S_jj + S_ij^2) / npilot.

        The strongest available check on the resampling itself. The
        cross term ``S_ij^2`` is what fails if the resampling index is
        drawn per model instead of shared: independent indices would
        destroy the cross-model correlation and drive the off-diagonal
        variance to the wrong value.
        """
        bkd = numpy_bkd
        cov = np.array([[4.0, 1.6], [1.6, 1.0]])
        npilot = 1500
        pilot_values = _gaussian_pilot(bkd, cov, npilot)
        pilot = ResampledPilotValues(pilot_values, bkd)
        stat = MultiOutputMean(1, bkd)
        replicates = []
        for _ in range(400):
            replicates.append(
                bkd.to_numpy(stat.compute_pilot_quantities(pilot.draw())[0])
            )
        replicates = np.array(replicates)
        for ii, jj in ((0, 0), (0, 1), (1, 1)):
            empirical = replicates[:, ii, jj].var(ddof=1)
            predicted = (cov[ii, ii] * cov[jj, jj] + cov[ii, jj] ** 2) / npilot
            assert 0.6 < empirical / predicted < 1.6, (
                f"Sigma[{ii},{jj}]: bootstrap variance {empirical:.3e} "
                f"vs asymptotic {predicted:.3e}"
            )

    def test_budget_variance_matches_delta_method(self, numpy_bkd) -> None:
        """Var(n*) = (1/tol^4) (2 sigma^4 / npilot).

        End-to-end: validates that pilot uncertainty propagates through
        the allocation correctly, not merely that the resampling is
        right. For Monte Carlo mean estimation the allocation is the
        closed form n* = sigma^2 / tol^2, so the delta method applies
        with derivative 1/tol^2.
        """
        bkd = numpy_bkd
        sigma2 = 4.0
        npilot = 1200
        tolerance = 0.05
        pilot_values = _gaussian_pilot(
            bkd, np.array([[sigma2]]), npilot, seed=3
        )
        samples = _mc_budget_samples(bkd, pilot_values, tolerance, 400)
        assert samples.nfailures() == 0
        costs = bkd.to_numpy(samples.values()).ravel()
        empirical = costs.var(ddof=1)
        predicted = (1.0 / tolerance**2) ** 2 * (2 * sigma2**2 / npilot)
        assert 0.5 < empirical / predicted < 1.8, (
            f"budget variance {empirical:.4e} vs delta-method "
            f"{predicted:.4e}"
        )

    def test_spread_shrinks_as_one_over_sqrt_npilot(self, numpy_bkd) -> None:
        """Quadrupling the pilot halves the spread.

        Distribution-free, and it tests the invariant that replicates
        are drawn at a fixed width: a bootstrap whose replicate size
        drifted would not follow this law.
        """
        bkd = numpy_bkd
        sigma2 = 4.0
        tolerance = 0.05
        spreads = []
        for npilot in (150, 600, 2400):
            pilot_values = _gaussian_pilot(
                bkd, np.array([[sigma2]]), npilot, seed=7
            )
            samples = _mc_budget_samples(bkd, pilot_values, tolerance, 300)
            costs = bkd.to_numpy(samples.values()).ravel()
            spreads.append(costs.std(ddof=1))
        for coarse, fine in zip(spreads[:-1], spreads[1:]):
            assert 1.3 < coarse / fine < 3.0, (
                f"spread ratio {coarse / fine:.2f} across a 4x pilot "
                "increase; expected about 2"
            )

    def test_degenerate_pilot_produces_no_spread(self, numpy_bkd) -> None:
        """Identical pilot values must give identical replicates.

        Every resample of a constant is that constant, so any spread
        here would be variability the bootstrap invented -- the
        signature of an indexing bug.
        """
        bkd = numpy_bkd
        constant = [bkd.full((1, 50), 3.0) for _ in range(2)]
        pilot = ResampledPilotValues(constant, bkd)
        stat = MultiOutputMean(1, bkd)
        first = bkd.to_numpy(stat.compute_pilot_quantities(pilot.draw())[0])
        for _ in range(10):
            other = bkd.to_numpy(
                stat.compute_pilot_quantities(pilot.draw())[0]
            )
            assert np.array_equal(first, other)

    @slow_test
    def test_bootstrap_distribution_tracks_true_sampling_distribution(
        self, numpy_bkd
    ) -> None:
        """The defining property: bootstrap ~ true sampling distribution.

        Compares the spread the bootstrap infers from one pilot set
        against the spread of budgets computed from many independent
        pilot sets. Agreement is what makes the bootstrap usable at all;
        the other tests check pieces of the machinery, this checks the
        premise.
        """
        bkd = numpy_bkd
        sigma2 = 4.0
        npilot = 800
        tolerance = 0.05

        truth = []
        for trial in range(300):
            values = _gaussian_pilot(
                bkd, np.array([[sigma2]]), npilot, seed=1000 + trial
            )
            stat = MultiOutputMean(1, bkd)
            stat.set_pilot_quantities(*stat.compute_pilot_quantities(values))
            fitted = MCToleranceAllocator(
                MCEstimator(stat, bkd.array([1.0]))
            ).allocate_for_tolerance(
                MaxMarginalStandardErrorConstraint(tolerance, bkd)
            )
            truth.append(float(fitted.actual_cost()))
        true_spread = float(np.std(truth, ddof=1))

        pilot_values = _gaussian_pilot(
            bkd, np.array([[sigma2]]), npilot, seed=11
        )
        samples = _mc_budget_samples(bkd, pilot_values, tolerance, 400)
        boot_spread = float(
            bkd.to_numpy(samples.values()).ravel().std(ddof=1)
        )
        assert 0.5 < boot_spread / true_spread < 2.0, (
            f"bootstrap spread {boot_spread:.3e} vs true sampling "
            f"spread {true_spread:.3e}"
        )


class TestResampledPilotValues:
    """The replicate source: fixed width, shared indices, validation."""

    def test_replicate_width_matches_npilot(self, bkd) -> None:
        pilot = ResampledPilotValues(_pilot_values(bkd), bkd)
        for _ in range(5):
            replicate = pilot.draw()
            assert len(replicate) == pilot.nmodels()
            for vals in replicate:
                assert vals.shape[1] == pilot.npilot()

    def test_satisfies_the_protocol(self, bkd) -> None:
        pilot = ResampledPilotValues(_pilot_values(bkd), bkd)
        assert isinstance(pilot, PilotReplicateProtocol)

    def test_rejects_ragged_pilot_values(self, bkd) -> None:
        """A shared index requires a common number of pilot samples."""
        vals = _pilot_values(bkd, nmodels=2, npilot=50)
        vals[1] = vals[1][:, :30]
        with pytest.raises(ValueError, match="same number of pilot"):
            ResampledPilotValues(vals, bkd)

    def test_rejects_one_dimensional_values(self, bkd) -> None:
        with pytest.raises(ValueError, match="2D"):
            ResampledPilotValues([bkd.array([1.0, 2.0, 3.0])], bkd)

    def test_rejects_empty(self, bkd) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ResampledPilotValues([], bkd)


class TestBudgetBootstrap:
    """Budget variability, across estimator families."""

    def test_quantiles_are_ordered(self, numpy_bkd) -> None:
        """An upper quantile is the number a caller budgets against."""
        bkd = numpy_bkd
        samples = _mc_budget_samples(
            bkd, _gaussian_pilot(bkd, np.array([[4.0]]), 200), 0.05, 60
        )
        lo = float(bkd.to_numpy(samples.quantile(0.1)).ravel()[0])
        mid = float(bkd.to_numpy(samples.quantile(0.5)).ravel()[0])
        hi = float(bkd.to_numpy(samples.quantile(0.9)).ravel()[0])
        assert lo <= mid <= hi

    def test_cv_budget_bootstrap_runs(self, numpy_bkd) -> None:
        """One loop serves a second family, selected by the solver."""
        bkd = numpy_bkd
        costs = bkd.array([4.0, 2.0, 1.0])
        solver = SampleCountBudgetSolver(
            lambda stat: CVToleranceAllocator(CVEstimator(stat, costs)),
            MaxMarginalStandardErrorConstraint(0.05, bkd),
        )
        samples = bootstrap_budget_from_pilot(
            ResampledPilotValues(_pilot_values(bkd, npilot=150), bkd),
            lambda: MultiOutputMean(1, bkd),
            solver,
            bkd,
            nbootstraps=20,
        )
        assert samples.nsuccesses() > 0

    def test_rejects_nonpositive_nbootstraps(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        with pytest.raises(ValueError, match="nbootstraps"):
            _mc_budget_samples(
                bkd, _gaussian_pilot(bkd, np.array([[4.0]]), 50), 0.05, 0
            )


class TestCovarianceBootstrap:
    """Accuracy variability at a committed allocation."""

    def test_covariance_varies_at_fixed_allocation(self, bkd) -> None:
        """The complement question: allocation fixed, accuracy varies."""
        costs = bkd.array([4.0, 2.0, 1.0])
        samples = bootstrap_covariance_from_pilot(
            ResampledPilotValues(_pilot_values(bkd, npilot=150), bkd),
            lambda: MultiOutputMean(1, bkd),
            lambda stat: CVEstimator(stat, costs),
            bkd.full((3,), 20.0),
            bkd,
            nbootstraps=25,
        )
        assert samples.nsuccesses() > 0
        assert bkd.to_numpy(samples.values()).std() > 0.0

    def test_allocations_not_recorded_when_fixed(self, bkd) -> None:
        """Nothing varies about the allocation, so none is stored."""
        costs = bkd.array([4.0, 2.0, 1.0])
        samples = bootstrap_covariance_from_pilot(
            ResampledPilotValues(_pilot_values(bkd, npilot=100), bkd),
            lambda: MultiOutputMean(1, bkd),
            lambda stat: CVEstimator(stat, costs),
            bkd.full((3,), 20.0),
            bkd,
            nbootstraps=5,
        )
        assert samples.allocations() is None

    def test_all_families_satisfy_covariance_protocol(self, bkd) -> None:
        """One protocol reaches every family, which is what lets the
        loop serve them without naming any."""
        stat = MultiOutputMean(1, bkd)
        stat.set_pilot_quantities(bkd.eye(3))
        costs = bkd.array([4.0, 2.0, 1.0])
        for est in (
            MCEstimator(stat, costs[:1]),
            CVEstimator(stat, costs),
            GroupACVEstimatorIS(stat, costs),
        ):
            assert isinstance(est, CovarianceEstimatorProtocol)


class TestBootstrapSamples:
    """Failures are counted, and summaries refuse to invent data."""

    def test_failures_counted_not_dropped(self, numpy_bkd) -> None:
        """Dropping failures biases toward the pilots that were easy.

        A resampled covariance can make a tolerance unreachable. Those
        replicates are part of the answer: discarding them silently
        would report a tighter distribution than the pilot supports.
        """
        bkd = numpy_bkd

        class _AlwaysFails:
            def solve(self, statistic):
                raise RuntimeError("unreachable")

        samples = bootstrap_budget_from_pilot(
            ResampledPilotValues(_pilot_values(bkd, nmodels=1), bkd),
            lambda: MultiOutputMean(1, bkd),
            _AlwaysFails(),
            bkd,
            nbootstraps=7,
        )
        assert samples.nfailures() == 7
        assert samples.nsuccesses() == 0
        assert samples.nbootstraps() == 7

    def test_summaries_raise_when_everything_failed(self, numpy_bkd) -> None:
        """A quantile of nothing is not zero; it is undefined."""
        bkd = numpy_bkd
        empty = BootstrapSamples(bkd.array([]), 5, 5, bkd)
        with pytest.raises(ValueError, match="no successful"):
            empty.quantile(0.5)
        with pytest.raises(ValueError, match="no successful"):
            empty.mean()

    def test_quantile_level_validated(self, numpy_bkd) -> None:
        bkd = numpy_bkd
        samples = BootstrapSamples(bkd.array([[1.0], [2.0]]), 0, 2, bkd)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            samples.quantile(1.5)
