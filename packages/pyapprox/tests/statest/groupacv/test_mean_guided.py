"""Tests for MeanGuidedSubsetFitter with known_quantities."""

import numpy as np
from pyapprox.util.backends.numpy import NumpyBkd


def _setup_benchmark():
    """Build a PolynomialEnsembleBenchmark and return reusable objects."""
    from pyapprox_benchmarks.statest import PolynomialEnsembleBenchmark

    bkd = NumpyBkd()
    bench = PolynomialEnsembleBenchmark(bkd, nmodels=5)
    costs = bench.problem().costs()
    cov = bench.covariance_matrix()
    W = bench.covariance_of_centered_values_kronecker_product()
    B = bench.covariance_of_mean_and_variance_estimators()
    means = bench.ensemble_means()
    nqoi = bench.problem().models()[0].nqoi()
    nmodels = len(bkd.to_numpy(costs))
    return bkd, bench, costs, cov, W, B, means, nqoi, nmodels


def _make_fitter_args(bkd):
    from pyapprox.optimization.minimize.scipy.slsqp import ScipySLSQPOptimizer
    from pyapprox.statest.groupacv import GroupACVEstimatorIS
    from pyapprox.statest.groupacv.variable_space import AllocationProblemConfig

    return dict(
        estimator_class=GroupACVEstimatorIS,
        optimizer=ScipySLSQPOptimizer(maxiter=1000, ftol=1e-6),
        problem_config=AllocationProblemConfig(
            variable_scaling="log", budget_constraint_form="inequality",
        ),
    )


class TestMeanGuidedKnownQuantities:
    def test_mean_only_kq_raises_for_variance_stat(self, numpy_bkd):
        """Mean-only known_quantities should raise for variance stat."""
        import pytest
        from pyapprox.statest.groupacv import (
            MeanGuidedSubsetFitter,
            get_model_subsets,
        )
        from pyapprox.statest.statistics import MultiOutputVariance

        bkd, bench, costs, cov, W, B, means, nqoi, nmodels = (
            _setup_benchmark()
        )
        subsets = get_model_subsets(nmodels, bkd)
        fitter_args = _make_fitter_args(bkd)

        stat_var = MultiOutputVariance(nqoi, bkd)
        stat_var.set_pilot_quantities(cov, W)

        kq = {(4, "mean"): means[4, :]}

        with pytest.raises(ValueError, match="known mean but not known variance"):
            MeanGuidedSubsetFitter(
                stat_var, costs, candidate_subsets=subsets,
                known_quantities=kq, **fitter_args,
            )

    def test_known_mean_and_variance_improves_variance_stat(self, numpy_bkd):
        """Known mean+variance in stage 2 should give <= variance."""
        from pyapprox.statest.groupacv import (
            MeanGuidedSubsetFitter,
            get_model_subsets,
        )
        from pyapprox.statest.statistics import MultiOutputVariance

        bkd, bench, costs, cov, W, B, means, nqoi, nmodels = (
            _setup_benchmark()
        )
        subsets = get_model_subsets(nmodels, bkd)
        fitter_args = _make_fitter_args(bkd)

        stat_var = MultiOutputVariance(nqoi, bkd)
        stat_var.set_pilot_quantities(cov, W)

        variances = bkd.diag(cov)
        kq = {
            (4, "mean"): means[4, :],
            (4, "variance"): variances[4:5],
        }
        target_cost = 200.0

        result_none = MeanGuidedSubsetFitter(
            stat_var, costs, candidate_subsets=subsets,
            known_quantities=None, **fitter_args,
        ).fit(target_cost, min_nhf_samples=2)

        result_kq = MeanGuidedSubsetFitter(
            stat_var, costs, candidate_subsets=subsets,
            known_quantities=kq, **fitter_args,
        ).fit(target_cost, min_nhf_samples=2)

        var_none = float(bkd.to_numpy(
            result_none.best_estimator.covariance()[0, 0]
        ))
        var_kq = float(bkd.to_numpy(
            result_kq.best_estimator.covariance()[0, 0]
        ))

        assert var_kq <= var_none, (
            f"known mean+variance should not increase variance: "
            f"{var_kq:.6e} > {var_none:.6e}"
        )

    def test_known_quantities_change_screening(self, numpy_bkd):
        """Known mean+variance should change screening allocations."""
        from pyapprox.statest.groupacv import (
            MeanGuidedSubsetFitter,
            get_model_subsets,
        )
        from pyapprox.statest.statistics import MultiOutputVariance

        bkd, bench, costs, cov, W, B, means, nqoi, nmodels = (
            _setup_benchmark()
        )
        subsets = get_model_subsets(nmodels, bkd)
        fitter_args = _make_fitter_args(bkd)

        stat_var = MultiOutputVariance(nqoi, bkd)
        stat_var.set_pilot_quantities(cov, W)

        variances = bkd.diag(cov)
        kq_all = {}
        for i in range(1, nmodels):
            kq_all[(i, "mean")] = means[i, :]
            kq_all[(i, "variance")] = variances[i : i + 1]
        target_cost = 100.0

        result_none = MeanGuidedSubsetFitter(
            stat_var, costs, candidate_subsets=subsets,
            known_quantities=None, **fitter_args,
        ).fit(target_cost, min_nhf_samples=1)

        result_kq = MeanGuidedSubsetFitter(
            stat_var, costs, candidate_subsets=subsets,
            known_quantities=kq_all, **fitter_args,
        ).fit(target_cost, min_nhf_samples=1)

        # With many known quantities the screening solve has more freedom,
        # so the active partition set should differ.
        nps_none = bkd.to_numpy(result_none.mean_npartition_samples)
        nps_kq = bkd.to_numpy(result_kq.mean_npartition_samples)
        assert not np.allclose(nps_none, nps_kq, rtol=1e-3), (
            "screening allocations should differ when known quantities are "
            "provided"
        )

    def test_variance_only_kq_raises(self, numpy_bkd):
        """Variance-only known_quantities should raise ValueError."""
        import pytest
        from pyapprox.statest.groupacv import (
            MeanGuidedSubsetFitter,
            get_model_subsets,
        )
        from pyapprox.statest.statistics import MultiOutputVariance

        bkd, bench, costs, cov, W, B, means, nqoi, nmodels = (
            _setup_benchmark()
        )
        subsets = get_model_subsets(nmodels, bkd)
        fitter_args = _make_fitter_args(bkd)

        stat_var = MultiOutputVariance(nqoi, bkd)
        stat_var.set_pilot_quantities(cov, W)

        variances = bkd.diag(cov)
        kq_var_only = {(4, "variance"): variances[4:5]}

        with pytest.raises(ValueError, match="known variance but not known mean"):
            MeanGuidedSubsetFitter(
                stat_var, costs, candidate_subsets=subsets,
                known_quantities=kq_var_only, **fitter_args,
            )

    def test_joint_mean_variance_with_known(self, numpy_bkd):
        """MeanAndVariance stat with known quantities via the fitter."""
        from pyapprox.statest.groupacv import (
            MeanGuidedSubsetFitter,
            get_model_subsets,
        )
        from pyapprox.statest.statistics import MultiOutputMeanAndVariance

        bkd, bench, costs, cov, W, B, means, nqoi, nmodels = (
            _setup_benchmark()
        )
        subsets = get_model_subsets(nmodels, bkd)
        fitter_args = _make_fitter_args(bkd)

        stat_mv = MultiOutputMeanAndVariance(nqoi, bkd)
        stat_mv.set_pilot_quantities(cov, W, B)

        variances = bkd.diag(cov)
        kq = {
            (4, "mean"): means[4, :],
            (4, "variance"): variances[4:5],
        }
        target_cost = 500.0

        result_none = MeanGuidedSubsetFitter(
            stat_mv, costs, candidate_subsets=subsets,
            known_quantities=None, **fitter_args,
        ).fit(target_cost, min_nhf_samples=2)

        result_kq = MeanGuidedSubsetFitter(
            stat_mv, costs, candidate_subsets=subsets,
            known_quantities=kq, **fitter_args,
        ).fit(target_cost, min_nhf_samples=2)

        cov_none = bkd.to_numpy(result_none.best_estimator.covariance())
        cov_kq = bkd.to_numpy(result_kq.best_estimator.covariance())

        mean_idx = stat_mv.stat_slot_indices("mean")
        for mi in mean_idx:
            assert cov_kq[mi, mi] <= cov_none[mi, mi], (
                f"known quantities should not increase mean variance at "
                f"index {mi}: {cov_kq[mi, mi]:.6e} > {cov_none[mi, mi]:.6e}"
            )

    def test_beats_mc(self, numpy_bkd):
        """Fitter with known quantities should beat MC baseline."""
        from pyapprox.statest.groupacv import (
            MeanGuidedSubsetFitter,
            get_model_subsets,
        )
        from pyapprox.statest.statistics import MultiOutputVariance

        bkd, bench, costs, cov, W, B, means, nqoi, nmodels = (
            _setup_benchmark()
        )
        subsets = get_model_subsets(nmodels, bkd)
        fitter_args = _make_fitter_args(bkd)

        stat_var = MultiOutputVariance(nqoi, bkd)
        stat_var.set_pilot_quantities(cov, W)

        variances = bkd.diag(cov)
        kq = {
            (4, "mean"): means[4, :],
            (4, "variance"): variances[4:5],
        }
        target_cost = 200.0

        result = MeanGuidedSubsetFitter(
            stat_var, costs, candidate_subsets=subsets,
            known_quantities=kq, **fitter_args,
        ).fit(target_cost, min_nhf_samples=2)

        gacv_var = float(bkd.to_numpy(
            result.best_estimator.covariance()[0, 0]
        ))

        costs_np = bkd.to_numpy(costs)
        nhf = target_cost / costs_np[0]
        mc_var = float(bkd.to_numpy(
            stat_var.high_fidelity_estimator_covariance(
                bkd.array([nhf])
            )[0, 0]
        ))

        assert gacv_var < mc_var, (
            f"GACV with known quantities should beat MC: "
            f"{gacv_var:.6e} >= {mc_var:.6e}"
        )
