"""
Tests for LogNormalDataMeanQoIAVaRStdDevObjective (differentiable objective).

Verifies protocol satisfaction, shape correctness, finite differences vs
jacobian, autograd compatibility, and consistency with the diagnostics class.
"""

import numpy as np
import pytest

from pyapprox.expdesign.analytical import (
    ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev,
    LogNormalDataMeanQoIAVaRStdDevObjective,
)
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.util.backends.autodiff import AutodiffBackend
from tests._helpers.markers import slow_test


def _build_problem(bkd, nobs=3, npred=4):
    """Build degree-1 basis with general QoI locations."""
    np.random.seed(42)
    nvars = 2

    obs_mat = bkd.asarray(np.random.randn(nobs, nvars))
    prior_mean = bkd.zeros((nvars, 1))
    prior_cov = bkd.asarray(np.eye(nvars) * 0.5)
    noise_variances = bkd.asarray(np.random.uniform(0.1, 0.5, nobs))

    x_vals = np.linspace(-1.5, 1.5, npred)
    qoi_mat = bkd.asarray(np.column_stack([np.ones(npred), x_vals]))

    return obs_mat, prior_mean, prior_cov, qoi_mat, noise_variances


class TestLogNormalDataMeanQoIAVaRStdDevObjective:

    def test_call_returns_correct_shape(self, bkd):
        """Test __call__ returns (1, 1) array."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, 0.5, bkd
        )
        nobs = obs_mat.shape[0]
        weights = bkd.full((nobs, 1), 1.0 / nobs)
        result = obj(weights)
        assert result.shape == (1, 1)

    def test_positive_value(self, bkd):
        """Test objective returns positive value."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, 0.5, bkd
        )
        nobs = obs_mat.shape[0]
        weights = bkd.full((nobs, 1), 1.0 / nobs)
        result = obj(weights)
        assert float(bkd.to_numpy(result).flat[0]) > 0.0

    def test_derivatives_bundle_capability(self, bkd):
        """The objective owns its bundle: autograd jacobian on autodiff
        backends, empty bundle otherwise (optimizers then use their own
        finite differences on the value)."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, 0.5, bkd
        )
        jac_fn = obj.derivatives().jacobian
        if not isinstance(bkd, AutodiffBackend):
            assert jac_fn is None
            return
        weights = bkd.full((nobs, 1), 1.0 / nobs)
        jac = jac_fn(weights)
        assert jac.shape == (1, nobs)

    def test_derivatives_pass_derivative_checker(self, torch_bkd):
        """Validate the bundle jacobian against finite differences with
        the central DerivativeChecker (fd step-size sweep)."""
        bkd = torch_bkd
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, 0.5, bkd
        )
        checker = DerivativeChecker(obj)
        weights = bkd.full((nobs, 1), 1.0 / nobs)
        # steps capped at 1e-2 so perturbed weights stay strictly
        # positive (the objective is undefined for w <= 0)
        fd_eps = bkd.flip(bkd.logspace(-12, -2, 11))
        errors = checker.check_derivatives(
            weights, fd_eps=fd_eps, relative=True, verbosity=0
        )
        min_error = float(bkd.min(errors[0]))
        assert min_error < 1e-6, min_error

    def test_nvars_nqoi(self, bkd):
        """Test nvars and nqoi accessors."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, 0.5, bkd
        )
        assert obj.nvars() == obs_mat.shape[0]
        assert obj.nqoi() == 1

    def test_matches_diagnostics_class(self, bkd):
        """Test differentiable objective matches non-differentiable version."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        alpha = 0.5
        weights = bkd.full((nobs, 1), 1.0 / nobs)

        # Differentiable objective
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, alpha, bkd
        )
        obj_val = float(bkd.to_numpy(obj(weights)).flat[0])

        # Non-differentiable diagnostics class
        noise_cov = bkd.diag(noise_var / weights[:, 0])
        diag_utility = ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev(
            prior_mean, prior_cov, qoi_mat, alpha, bkd
        )
        diag_utility.set_observation_matrix(obs_mat)
        diag_utility.set_noise_covariance(noise_cov)
        diag_val = diag_utility.value()

        bkd.assert_allclose(
            bkd.asarray([obj_val]),
            bkd.asarray([diag_val]),
            rtol=1e-8,
        )

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75])
    def test_alpha_values(self, bkd, alpha):
        """Test objective computes for several alpha values."""
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        obj = LogNormalDataMeanQoIAVaRStdDevObjective(
            obs_mat, prior_mean, prior_cov, qoi_mat, noise_var, alpha, bkd
        )
        weights = bkd.full((nobs, 1), 1.0 / nobs)
        result = obj(weights)
        val = float(bkd.to_numpy(result).flat[0])
        assert np.isfinite(val) and val > 0.0

    def test_continuous_in_alpha(self, bkd):
        """Utility must be continuous in alpha, including fractional levels.

        The AVaR tail of a discrete distribution contains a fractional
        boundary atom whenever npred*(1-alpha) is not an integer. A tail
        rule that rounds to whole atoms (the old ceil bug) makes the value
        a step function of alpha with O(1e-2) jumps at multiples of
        1/npred; the correct rule is continuous.
        """
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        weights = bkd.full((nobs, 1), 1.0 / nobs)

        def sweep(alphas):
            vals = []
            for alpha in alphas:
                obj = LogNormalDataMeanQoIAVaRStdDevObjective(
                    obs_mat, prior_mean, prior_cov, qoi_mat, noise_var,
                    float(alpha), bkd,
                )
                vals.append(float(bkd.to_numpy(obj(weights)).flat[0]))
            return np.array(vals)

        # Scale-free continuity check: for a continuous function the max
        # step shrinks proportionally with the grid spacing, while a
        # staircase jump (the old ceil bug) stays O(1) at any resolution.
        # An absolute step bound would depend on the problem's Lipschitz
        # constant; the two-resolution ratio does not.
        coarse = sweep(np.linspace(0.0, 0.9, 46))    # h = 0.02
        fine = sweep(np.linspace(0.0, 0.9, 181))     # h = 0.005
        assert np.all(np.diff(coarse) >= -1e-12)     # nondecreasing
        assert np.all(np.diff(fine) >= -1e-12)
        max_step_coarse = np.abs(np.diff(coarse)).max()
        max_step_fine = np.abs(np.diff(fine)).max()
        assert max_step_fine < 0.5 * max_step_coarse, (
            max_step_fine, max_step_coarse,
        )

    @slow_test
    def test_matches_bruteforce_mc_at_fractional_alpha(self, bkd):
        """Compare with brute-force MC at alphas with fractional tails.

        The MC reference computes the discrete AVaR via the variational
        (Rockafellar-Uryasev) definition, an implementation independent of
        the cumulative tail rule, so a shared tail-selection bug cannot
        cancel. Covers uniform and non-uniform (Gauss-Legendre) weights.
        """
        obs_mat, prior_mean, prior_cov, qoi_mat, noise_var = (
            _build_problem(bkd)
        )
        nobs = obs_mat.shape[0]
        npred = qoi_mat.shape[0]
        weights = bkd.full((nobs, 1), 1.0 / nobs)

        def avar_variational(vals, p, alpha):
            # optimum of min_t t + E[(X - t)_+]/(1 - alpha) is at an atom
            return min(
                t + (p * np.maximum(vals - t, 0.0)).sum() / (1.0 - alpha)
                for t in vals
            )

        def bruteforce(p_np, alpha, nsamples=100_000):
            rng = np.random.default_rng(7)
            A = bkd.to_numpy(obs_mat)
            Sig0 = bkd.to_numpy(prior_cov)
            mu0 = bkd.to_numpy(prior_mean)
            B = bkd.to_numpy(qoi_mat)
            noise_cov = np.diag(
                bkd.to_numpy(noise_var) / bkd.to_numpy(weights)[:, 0]
            )
            y_cov = A @ Sig0 @ A.T + noise_cov
            y = rng.multivariate_normal(
                (A @ mu0).flatten(), y_cov, nsamples
            )
            noise_prec = np.linalg.inv(noise_cov)
            post_cov = np.linalg.inv(
                np.linalg.inv(Sig0) + A.T @ noise_prec @ A
            )
            mu_star = (post_cov @ A.T @ noise_prec @ y.T).T
            s2 = np.einsum("qi,ij,qj->q", B, post_cov, B)
            K = np.exp(s2 / 2) * np.sqrt(np.expm1(s2))
            D = K[None, :] * np.exp(mu_star @ B.T)
            return np.mean(
                [avar_variational(row, p_np, alpha) for row in D]
            )

        gl_nodes, gl_w = np.polynomial.legendre.leggauss(npred)
        cases = [
            (None, np.full(npred, 1.0 / npred), 0.1),
            (None, np.full(npred, 1.0 / npred), 0.643),
            (bkd.asarray(gl_w), gl_w / gl_w.sum(), 0.45),
            (bkd.asarray(gl_w), gl_w / gl_w.sum(), 0.643),
        ]
        for qoi_quad_weights, p_np, alpha in cases:
            obj = LogNormalDataMeanQoIAVaRStdDevObjective(
                obs_mat, prior_mean, prior_cov, qoi_mat, noise_var,
                alpha, bkd, qoi_quad_weights=qoi_quad_weights,
            )
            exact = float(bkd.to_numpy(obj(weights)).flat[0])
            mc = bruteforce(p_np, alpha)
            assert abs(exact - mc) / mc < 5e-3, (alpha, exact, mc)

    @slow_test
    def test_prediction_pipeline_mc_has_delta_floor(self, bkd):
        """exp2-style MC objective vs the exact analytical reference.

        The pipeline's AVaR risk over predictions is smoothed with
        regularization parameter delta, so at FIXED delta sample
        refinement converges to the smoothed objective, not the exact
        one: the error against the exact analytical value stabilizes at
        a delta-dependent floor. Asserting plain convergence to zero at
        fixed delta would be wrong; instead assert (i) the floor shrinks
        as delta grows and (ii) at the largest delta the residual error
        is within the MC-noise budget.
        """
        from pyapprox.expdesign.objective import (
            create_prediction_oed_objective,
        )
        from pyapprox.util.backends.numpy import NumpyBkd

        if not isinstance(bkd, NumpyBkd):
            pytest.skip(
                "certifies the numpy exp2 pipeline; torch adds only an "
                "inductor toolchain dependency"
            )

        nobs, npred, degree_basis = 5, 10, 2
        noise_std, prior_std = 0.5, 0.5
        alpha = 0.45  # fractional tail mass: npred*(1-alpha) = 5.5

        obs_x = np.linspace(-1, 1, nobs)
        A_np = np.column_stack([np.ones(nobs), obs_x])
        pred_x = np.linspace(-2 / 3, 2 / 3, npred)
        B_np = np.column_stack([np.ones(npred), pred_x])
        noise_var_np = np.full(nobs, noise_std**2)
        w = bkd.full((nobs, 1), 1.0 / nobs)

        # Exact analytical reference at the same design weights
        ref = ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev(
            bkd.zeros((degree_basis, 1)),
            bkd.asarray(prior_std**2 * np.eye(degree_basis)),
            bkd.asarray(B_np), alpha, bkd,
        )
        ref.set_observation_matrix(bkd.asarray(A_np))
        ref.set_noise_covariance(
            bkd.diag(bkd.asarray(noise_var_np) / w[:, 0])
        )
        exact = ref.value()

        def mc_value(nsamples, delta, seed=11):
            rng = np.random.default_rng(seed)
            th_out = prior_std * rng.standard_normal(
                (degree_basis, nsamples)
            )
            th_in = prior_std * rng.standard_normal(
                (degree_basis, nsamples)
            )
            objective = create_prediction_oed_objective(
                bkd.asarray(noise_var_np),
                bkd.asarray(A_np @ th_out),
                bkd.asarray(A_np @ th_in),
                bkd.asarray(rng.standard_normal((nobs, nsamples))),
                bkd.asarray(np.exp(B_np @ th_in).T),
                bkd,
                deviation_type="stdev",
                risk_type="avar",
                noise_stat_type="mean",
                risk_kwargs={"alpha": alpha, "delta": delta},
            )
            return float(bkd.to_numpy(objective(w)).flat[0])

        n_hi = 4000
        err_small_delta = abs(mc_value(n_hi, delta=20) - exact)
        err_large_delta = abs(mc_value(n_hi, delta=1e5) - exact)

        # (i) smoothing floor shrinks as delta grows
        assert err_small_delta > err_large_delta, (
            err_small_delta, err_large_delta,
        )
        # (ii) at the largest delta only MC/inner-loop error remains
        assert err_large_delta / exact < 2e-2, (err_large_delta, exact)

    @slow_test
    def test_quadrature_refinement_converges(self, bkd):
        """Refining the prediction quadrature gives a Cauchy sequence.

        The discrete Q-atom AVaR approximates the AVaR over a continuous
        prediction domain; uniform and Gauss-Legendre discretizations
        must approach the same limit. AVaR is not a smooth functional of
        the integrand (tail kink), so only modest algebraic convergence
        is asserted, not spectral.
        """
        nvars = 2
        alpha = 0.45
        prior_mean = bkd.zeros((nvars, 1))
        prior_cov = bkd.asarray(np.eye(nvars) * 0.25)
        nobs = 5
        obs_x = np.linspace(-1, 1, nobs)
        obs_mat = bkd.asarray(np.column_stack([np.ones(nobs), obs_x]))
        noise_cov = bkd.asarray(np.eye(nobs) * 0.25 * nobs)

        def value(x_np, p_np):
            npred = len(x_np)
            qoi_mat = bkd.asarray(
                np.column_stack([np.ones(npred), x_np])
            )
            obj = ConjugateGaussianOEDForLogNormalDataMeanQoIAVaRStdDev(
                prior_mean, prior_cov, qoi_mat, alpha, bkd,
                qoi_quad_weights=(
                    None if p_np is None else bkd.asarray(p_np)
                ),
            )
            obj.set_observation_matrix(obs_mat)
            obj.set_noise_covariance(noise_cov)
            return obj.value()

        qs = [8, 16, 32, 64]
        vals_unif = [
            value(np.linspace(-2 / 3, 2 / 3, q), None) for q in qs
        ]
        diffs = np.abs(np.diff(vals_unif))
        # Cauchy at the expected algebraic rate: the tail kink limits
        # convergence to O(1/Q), so each refinement should roughly halve
        # the difference (0.65 leaves noise headroom above the ideal
        # 0.5); a non-convergent discretization keeps diffs O(1).
        assert np.all(diffs[1:] < 0.65 * diffs[:-1]), vals_unif

        # Gauss-Legendre targets the same continuum limit: the gap to
        # the uniform rule must shrink under refinement (both are
        # O(1/Q)), which a wrong-measure GL path (e.g. unnormalized
        # weights) cannot satisfy.
        def gl_gap(q, val_unif):
            gl_nodes, gl_w = np.polynomial.legendre.leggauss(q)
            return abs(
                value(gl_nodes * (2 / 3), gl_w / gl_w.sum()) - val_unif
            )

        gap32 = gl_gap(32, vals_unif[qs.index(32)])
        gap64 = gl_gap(64, vals_unif[qs.index(64)])
        assert gap64 < 0.9 * gap32 + 1e-6, (gap32, gap64)
        assert gap64 < 4 * diffs[-1] + 1e-4, (gap64, diffs[-1])
