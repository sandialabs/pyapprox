"""
Test that the analytical LogNormalQoIAVaRDataMeanStdDev objective and the
MC/QMC double-loop estimator produce the same optimal design on the
lognormal benchmark, up to MC noise.
"""

import os
import warnings

import numpy as np
import pytest

from pyapprox.expdesign.analytical import (
    LogNormalDataMeanQoIAVaRStdDevObjective,
)
from pyapprox.expdesign.data import generate_oed_data
from pyapprox.expdesign.quadrature.oed import (
    OEDQuadratureSampler,
    build_oed_joint_distribution,
)
from pyapprox.expdesign.solver import (
    RelaxedOEDConfig,
    RelaxedOEDSolver,
    solve_prediction_oed,
)
from pyapprox.util.sampling.halton import HaltonSampler
from pyapprox_benchmarks.instances.oed.nonlinear_gaussian import (
    build_nonlinear_gaussian_pred_benchmark,
)


def _make_halton_sampler(benchmark, bkd, start_index=0):
    """Create Halton-based OEDQuadratureSampler from benchmark."""
    joint_dist = build_oed_joint_distribution(benchmark.problem(), bkd)
    ndim = benchmark.problem().nparams() + benchmark.problem().nobs()
    halton = HaltonSampler(ndim, bkd, distribution=joint_dist,
                           start_index=start_index)
    return OEDQuadratureSampler(halton, benchmark.problem().nparams(), bkd)


@pytest.mark.skipif(
    not os.environ.get("PYAPPROX_RUN_SLOW"),
    reason="Slow: set PYAPPROX_RUN_SLOW=1 to run",
)
class TestAnalyticalVsMCDesign:
    """Compare analytical and MC optimal designs on lognormal benchmark."""

    _nobs = 5
    _degree = 1
    _npred = 4
    _noise_std = 0.5
    _prior_std = 0.5
    _nouter = 5000
    _ninner = 500

    def _build_benchmark(self, bkd):
        return build_nonlinear_gaussian_pred_benchmark(
            self._nobs, self._degree, self._noise_std, self._prior_std,
            bkd, npred=self._npred,
        )

    def _solve_analytical(self, benchmark, alpha, bkd):
        """Optimise using the analytical objective."""
        problem = benchmark.problem()
        prior_mean = problem.prior_mean().reshape(
            problem.nparams(), 1
        )
        prior_cov = problem.prior_covariance()
        obs_mat = benchmark.design_matrix()
        qoi_mat = benchmark.qoi_matrix()
        noise_variances = problem.noise_variances()

        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat,
            noise_variances, alpha, bkd,
        )
        config = RelaxedOEDConfig(maxiter=500)
        solver = RelaxedOEDSolver(obj, config)
        weights, value = solver.solve()
        return weights, value, obj

    def _solve_mc(self, benchmark, alpha, bkd):
        """Optimise using MC double-loop estimator."""
        np.random.seed(42)
        outer_sampler = _make_halton_sampler(benchmark, bkd, start_index=0)
        inner_sampler = _make_halton_sampler(benchmark, bkd, start_index=10000)

        data = generate_oed_data(
            benchmark.problem(), outer_sampler, inner_sampler,
            self._nouter, self._ninner,
        )

        risk_kwargs = {"alpha": alpha} if alpha > 0.0 else {}
        config = RelaxedOEDConfig(maxiter=500)

        weights, value = solve_prediction_oed(
            benchmark.problem().noise_variances(),
            data.outer_shapes,
            data.inner_shapes,
            data.latent_samples,
            data.qoi_vals,
            bkd,
            deviation_type="stdev",
            risk_type="avar" if alpha > 0.0 else "mean",
            noise_stat_type="mean",
            risk_kwargs=risk_kwargs,
            config=config,
        )
        return weights, value

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 0.9])
    def test_analytical_vs_mc_design(self, numpy_bkd, alpha):
        """Analytical and MC designs produce near-equal utility."""
        bkd = numpy_bkd
        benchmark = self._build_benchmark(bkd)

        # Solve with both methods
        w_exact, _, obj = self._solve_analytical(benchmark, alpha, bkd)
        w_mc, _ = self._solve_mc(benchmark, alpha, bkd)

        # Utility agreement: evaluate analytical objective at both designs
        u_at_exact = obj.value(w_exact)
        u_at_mc = obj.value(w_mc)

        rtol = 0.05
        rel_diff = abs(u_at_mc - u_at_exact) / abs(u_at_exact)
        assert rel_diff < rtol, (
            f"alpha={alpha}: analytical utility at MC design ({u_at_mc:.6f}) "
            f"differs from optimal ({u_at_exact:.6f}) by {rel_diff:.3f} "
            f"(rtol={rtol})"
        )

        # Design agreement (soft): L1 distance between weight vectors
        l1_dist = float(bkd.to_numpy(bkd.sum(bkd.abs(w_exact - w_mc))))
        if l1_dist > 0.3:
            warnings.warn(
                f"alpha={alpha}: L1 distance between designs = {l1_dist:.3f} "
                f"> 0.3. This may indicate a flat utility landscape rather "
                f"than a bug.",
                stacklevel=1,
            )
