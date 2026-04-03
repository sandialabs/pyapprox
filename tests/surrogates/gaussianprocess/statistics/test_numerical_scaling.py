"""
Numerical tests for GP statistics scaling with kernel variance s**2.

Unlike the SymPy tests (which verify algebraic identities tautologically),
these tests verify that the actual code produces numerically correct results
by:

1. Verifying that the integral calculator decomposes scaled kernels and
   returns C-integrals (unit-variance) regardless of s**2.
2. Running the full statistics pipeline with different s**2 values and
   verifying that:
   - eta (mean of mean) is invariant to s**2
   - Var[mu] scales as s**2
   - E[gamma] is affine in s**2
   - alpha_K = s**-2 alpha_C
3. Comparing GP statistics against Monte Carlo estimates.

All tests use nugget=0 or nugget=1e-10 to avoid the nugget breaking
the exact s**2 cancellation (A_K = s**2*C + sigma**2*I != s**2*(C + sigma**2*I)
when sigma**2 != 0).
"""

import math
from itertools import product as iterproduct

import numpy as np

from pyapprox.probability.univariate.uniform import UniformMarginal
from pyapprox.surrogates.gaussianprocess import ExactGaussianProcess
from pyapprox.surrogates.gaussianprocess.statistics import (
    GaussianProcessStatistics,
    SeparableKernelIntegralCalculator,
)
from pyapprox.surrogates.gaussianprocess.statistics.sensitivity import (
    GaussianProcessSensitivity,
)
from pyapprox.surrogates.kernels import IIDGaussianNoise
from pyapprox.surrogates.kernels.composition import (
    SeparableProductKernel,
)
from pyapprox.surrogates.kernels.matern import (
    SquaredExponentialKernel,
)
from pyapprox.surrogates.kernels.scalings import PolynomialScaling
from pyapprox.surrogates.sparsegrids.basis_factory import (
    create_basis_factories,
)
from tests._helpers.markers import slow_test, slower_test

# ===================================================================
# Helpers
# ===================================================================

# Nugget small enough that sigma**2/s**2 approx 0, preserving exact scaling.
_NUGGET = 1e-10


def _create_quadrature_bases(
    marginals,
    nquad_points,
    bkd,
):
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


def _create_separable_kernel(
    length_scales,
    bkd,
):
    kernels_1d = [
        SquaredExponentialKernel([ls], (0.1, 10.0), 1, bkd) for ls in length_scales
    ]
    return SeparableProductKernel(kernels_1d, bkd)


def _create_scaled_kernel(
    s_value,
    length_scales,
    bkd,
):
    base = _create_separable_kernel(length_scales, bkd)
    nvars = len(length_scales)
    scaling = PolynomialScaling([s_value], (0.01, 100.0), bkd, nvars=nvars, fixed=False)
    return scaling * base


def _fit_gp(
    kernel,
    nvars,
    X_train,
    y_train,
    bkd,
    nugget=_NUGGET,
):
    gp = ExactGaussianProcess(kernel, nvars=nvars, bkd=bkd, nugget=nugget)
    gp.hyp_list().set_all_inactive()
    gp.fit(X_train, y_train)
    return gp


def _create_stats(
    gp,
    marginals,
    nquad,
    bkd,
):
    bases = _create_quadrature_bases(marginals, nquad, bkd)
    calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
    return GaussianProcessStatistics(gp, calc)


def _create_sensitivity(
    gp,
    marginals,
    nquad,
    bkd,
):
    bases = _create_quadrature_bases(marginals, nquad, bkd)
    calc = SeparableKernelIntegralCalculator(gp, bases, marginals, bkd=bkd)
    stats = GaussianProcessStatistics(gp, calc)
    return GaussianProcessSensitivity(stats)


def _make_training_data(
    bkd,
    nvars=2,
    n_train=5,
):
    """Create reproducible training data.

    Uses few training points so the kernel matrix is well-conditioned
    without a large nugget.
    """
    np.random.seed(42)
    X_train = bkd.array(np.random.rand(nvars, n_train) * 2 - 1)
    y_train = bkd.reshape(
        bkd.sin(math.pi * X_train[0, :])
        * bkd.cos(math.pi * X_train[1, :] if nvars > 1 else 1.0),
        (1, -1),
    )
    return X_train, y_train


# ===================================================================
# Test 1: Integral calculator decomposes scaled kernels correctly
# ===================================================================


class TestIntegralCalculatorDecomposition:
    """Verify that the integral calculator extracts the base kernel C
    from a scaled kernel K = s**2*C and computes identical C-integrals
    regardless of s**2.

    The calculator always computes integrals of C (not K).
    The s**2 factor is applied in the statistics formulas.
    """

    def _setup(self, bkd):
        X_train, y_train = _make_training_data(bkd)
        length_scales = [1.0, 0.5]
        nvars = 2
        nquad = 30

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        # Unscaled GP (C kernel)
        kernel_C = _create_separable_kernel(length_scales, bkd)
        gp_C = _fit_gp(kernel_C, nvars, X_train, y_train, bkd)
        bases_C = _create_quadrature_bases(marginals, nquad, bkd)
        calc_C = SeparableKernelIntegralCalculator(gp_C, bases_C, marginals, bkd=bkd)

        # Scaled GP (K = s**2*C kernel, s=2.5)
        kernel_K = _create_scaled_kernel(2.5, length_scales, bkd)
        gp_K = _fit_gp(kernel_K, nvars, X_train, y_train, bkd)
        bases_K = _create_quadrature_bases(marginals, nquad, bkd)
        calc_K = SeparableKernelIntegralCalculator(gp_K, bases_K, marginals, bkd=bkd)

        return calc_C, calc_K

    def test_tau_identical(self, bkd) -> None:
        """Calculator returns identical tau for C and K = s**2*C."""
        calc_C, calc_K = self._setup(bkd)
        bkd.assert_allclose(calc_K.tau_C(), calc_C.tau_C(), rtol=1e-12)

    def test_P_identical(self, bkd) -> None:
        """Calculator returns identical P for C and K = s**2*C."""
        calc_C, calc_K = self._setup(bkd)
        bkd.assert_allclose(calc_K.P(), calc_C.P(), rtol=1e-12)

    def test_u_identical(self, bkd) -> None:
        """Calculator returns identical u for C and K = s**2*C."""
        calc_C, calc_K = self._setup(bkd)
        bkd.assert_allclose(
            bkd.asarray([calc_K.u()]),
            bkd.asarray([calc_C.u()]),
            rtol=1e-12,
        )

    def test_nu_identical(self, bkd) -> None:
        """Calculator returns identical nu for C and K = s**2*C."""
        calc_C, calc_K = self._setup(bkd)
        bkd.assert_allclose(
            bkd.asarray([calc_K.nu()]),
            bkd.asarray([calc_C.nu()]),
            rtol=1e-12,
        )

    def test_Pi_identical(self, bkd) -> None:
        """Calculator returns identical Pi for C and K = s**2*C."""
        calc_C, calc_K = self._setup(bkd)
        bkd.assert_allclose(calc_K.Pi(), calc_C.Pi(), rtol=1e-12)

    def test_xi1_identical(self, bkd) -> None:
        """Calculator returns identical xi_1 for C and K = s**2*C."""
        calc_C, calc_K = self._setup(bkd)
        bkd.assert_allclose(
            bkd.asarray([calc_K.xi1()]),
            bkd.asarray([calc_C.xi1()]),
            rtol=1e-12,
        )

    def test_Gamma_identical(self, bkd) -> None:
        """Calculator returns identical Gamma for C and K = s**2*C."""
        calc_C, calc_K = self._setup(bkd)
        bkd.assert_allclose(calc_K.Gamma(), calc_C.Gamma(), rtol=1e-12)


# ===================================================================
# Test 2: End-to-end statistics scaling across s**2 values
# ===================================================================


class TestStatisticsScalingEndToEnd:
    """Verify that computed statistics scale correctly when s**2 changes.

    Uses nugget=1e-10 so sigma**2/s**2 approx 0 and the exact scaling holds.

    Checks:
    - eta (mean of mean) is INVARIANT to s**2
    - Var[mu] scales as s**2
    - E[gamma] is affine in s**2: E[gamma] = (zeta - eta**2) + s**2*(v**2 - varsigma**2)
    - Var[gamma] is non-negative for all s
    """

    def _setup(self, bkd):
        X_train, y_train = _make_training_data(bkd)
        length_scales = [1.0, 0.5]
        nvars = 2
        nquad = 30

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        s_values = [0.5, 1.0, 2.0, 5.0]
        stats_by_s = {}

        for s in s_values:
            kernel = _create_scaled_kernel(s, length_scales, bkd)
            gp = _fit_gp(kernel, nvars, X_train, y_train, bkd)
            stats = _create_stats(gp, marginals, nquad, bkd)
            stats_by_s[s] = stats

        return stats_by_s, s_values

    def test_eta_invariant_across_s_values(self, bkd) -> None:
        """eta = tau^T A^{-1} y should be invariant to s**2.

        With nugget approx 0:
        alpha_K = (s**2*C)^{-1} y = s**{-2} C^{-1} y = s**{-2} alpha_C.
        The formula computes eta = alpha_K @ tau_K where tau_K = s**2 tau_C (but
        the calculator returns tau_C and moments.py multiplies by s**2
        implicitly through alpha). Actually eta = alpha @ tau where alpha
        already includes s**{-2} and tau is C-integral. So eta = s**{-2} alpha_C @ tau_C
        ... wait, let's just check numerically.
        """
        stats_by_s, s_values = self._setup(bkd)
        ref_eta = stats_by_s[s_values[0]].input_mean_of_posterior_mean()
        for s in s_values[1:]:
            eta = stats_by_s[s].input_mean_of_posterior_mean()
            bkd.assert_allclose(
                bkd.asarray([eta]),
                bkd.asarray([ref_eta]),
                rtol=1e-6,
            )

    def test_gp_variance_of_posterior_mean_scales_as_s2(self, bkd) -> None:
        """Var[mu] = s**2 * varsigma**2 should scale proportionally to s**2."""
        stats_by_s, s_values = self._setup(bkd)
        ref_s = s_values[0]
        ref_s2 = ref_s * ref_s
        ref_var = stats_by_s[ref_s].gp_variance_of_posterior_mean()

        for s in s_values[1:]:
            s2 = s * s
            var_mu = stats_by_s[s].gp_variance_of_posterior_mean()
            expected_ratio = s2 / ref_s2
            actual_ratio = var_mu / ref_var
            bkd.assert_allclose(
                bkd.asarray([actual_ratio]),
                bkd.asarray([expected_ratio]),
                rtol=1e-6,
            )

    def test_input_mean_of_posterior_variance_affine_in_s2(self, bkd) -> None:
        """E[gamma] = (zeta - eta**2) + s**2*(v**2 - varsigma**2) is affine in s**2.

        Extract the affine coefficients from two s values, then verify
        the relationship holds at all other s values.
        """
        stats_by_s, s_values = self._setup(bkd)
        s1 = s_values[0]
        s2_val = s_values[2]
        s1_sq = s1 * s1
        s2_sq = s2_val * s2_val

        E_gamma_1 = stats_by_s[s1].input_mean_of_posterior_variance()
        E_gamma_2 = stats_by_s[s2_val].input_mean_of_posterior_variance()

        # E[gamma](s**2) = A + B*s**2
        B = (E_gamma_2 - E_gamma_1) / (s2_sq - s1_sq)
        A = E_gamma_1 - B * s1_sq

        for s in s_values:
            s_sq = s * s
            expected = A + B * s_sq
            actual = stats_by_s[s].input_mean_of_posterior_variance()
            bkd.assert_allclose(
                bkd.asarray([actual]),
                bkd.asarray([expected]),
                rtol=1e-6,
            )

    def test_gp_variance_of_posterior_variance_nonnegative(self, bkd) -> None:
        """Var[gamma] should be non-negative for all s."""
        stats_by_s, s_values = self._setup(bkd)
        for s in s_values:
            var_var = stats_by_s[s].gp_variance_of_posterior_variance()
            assert float(bkd.to_numpy(bkd.asarray([var_var]))[0]) >= 0.0, (
                f"Var[gamma] should be non-negative for s={s}"
            )


# ===================================================================
# Test 3: Alpha scaling verification
# ===================================================================


class TestAlphaScaling:
    """Verify alpha_K = s**{-2} alpha_C numerically.

    With nugget approx 0: A_K = s**2*C, so A_K^{-1} = s**{-2}*C^{-1},
    giving alpha_K = s**{-2}*alpha_C.
    """

    def _setup(self, bkd):
        X_train, y_train = _make_training_data(bkd)
        length_scales = [1.0, 0.5]
        nvars = 2

        s = 3.0
        s2 = s * s

        kernel_C = _create_separable_kernel(length_scales, bkd)
        gp_C = _fit_gp(kernel_C, nvars, X_train, y_train, bkd)

        kernel_K = _create_scaled_kernel(s, length_scales, bkd)
        gp_K = _fit_gp(kernel_K, nvars, X_train, y_train, bkd)

        return gp_C, gp_K, s2

    def test_alpha_K_equals_s_minus_2_alpha_C(self, bkd) -> None:
        """alpha_K = s**{-2} alpha_C (exact with nugget approx 0)."""
        gp_C, gp_K, s2 = self._setup(bkd)
        alpha_C = gp_C.alpha()
        alpha_K = gp_K.alpha()
        expected = alpha_C / s2
        bkd.assert_allclose(alpha_K, expected, rtol=1e-6)


# ===================================================================
# Test 4: Monte Carlo verification of statistics formulas
# ===================================================================


class TestStatisticsVsMonteCarlo:
    """Verify GP statistics formulas against Monte Carlo estimates.

    For a fitted GP, sample input points X from rho, evaluate the posterior
    mean mu(X) and variance gamma(X) at each, and compute sample statistics.
    Compare against the closed-form formulas.

    This is the strongest test: it verifies the formulas produce correct
    numerical results, not just correct scaling.
    """

    def _setup(self, bkd):
        X_train, y_train = _make_training_data(bkd)
        nvars = 2
        nquad = 30
        n_mc = 10000

        length_scales = [1.0, 0.5]
        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        # Test with s=2.0 to catch scaling bugs
        s = 2.0
        kernel = _create_scaled_kernel(s, length_scales, bkd)
        gp = _fit_gp(kernel, nvars, X_train, y_train, bkd)
        stats = _create_stats(gp, marginals, nquad, bkd)

        # Generate MC samples from Uniform[-1,1]^2
        np.random.seed(123)
        mc_np = np.random.rand(nvars, n_mc) * 2 - 1
        X_mc = bkd.array(mc_np)

        return stats, gp, marginals, X_mc, nvars, nquad

    def test_input_mean_of_posterior_mean_vs_mc(self, bkd) -> None:
        """E[mu(X)] approx (1/N) Sum mu(X_i) by Monte Carlo."""
        stats, gp, _, X_mc, _, _ = self._setup(bkd)
        mu_mc = gp.predict(X_mc)  # (nqoi, n_mc)
        mc_estimate = bkd.mean(mu_mc)
        formula_value = stats.input_mean_of_posterior_mean()
        bkd.assert_allclose(
            bkd.asarray([formula_value]),
            bkd.asarray([mc_estimate]),
            rtol=0.05,
        )

    def _sample_posterior_statistics(self, bkd, gp, marginals, nquad):
        """Sample GP posterior realizations and compute mu_r, gamma_r.

        For each realization f^(r) ~ GP(m*, C*):
        - mu_r = integral f^(r)(z) rho(z) dz  (integrated mean)
        - kappa_r = integral f^(r)(z)**2 rho(z) dz  (integrated second moment)
        - gamma_r = kappa_r - mu_r**2  (variance of realization over input space)

        Returns (mu_samples, gamma_samples) as numpy arrays.
        """
        np.random.seed(7777)

        # Build tensor product quadrature grid
        bases = _create_quadrature_bases(marginals, nquad, bkd)
        pts_1d = []
        wts_1d = []
        for b in bases:
            p, w = b.quadrature_rule()
            pts_1d.append(bkd.to_numpy(p).flatten())
            wts_1d.append(bkd.to_numpy(w).flatten())

        grid = list(iterproduct(*pts_1d))
        wgrid = list(iterproduct(*wts_1d))
        Z_np = np.array(grid).T  # (nvars, nquad_total)
        w_np = np.prod(np.array(wgrid), axis=1)  # (nquad_total,)
        Z = bkd.array(Z_np)
        w = bkd.array(w_np)
        nquad_total = Z.shape[1]

        # Posterior mean and covariance at Z
        X_train = gp.data().X()
        y_train = gp.data().y()
        K_zz = gp.kernel()(Z, Z)
        K_zt = gp.kernel()(Z, X_train)
        chol = gp.cholesky()
        alpha = chol.solve(y_train.T)  # (n_train, 1)
        m_star = K_zt @ alpha  # (nquad_total, 1)
        m_star = bkd.reshape(m_star, (-1,))

        A_inv_K_tz = chol.solve(K_zt.T)  # (n_train, nquad_total)
        C_star = K_zz - K_zt @ A_inv_K_tz
        C_star = C_star + 1e-8 * bkd.eye(nquad_total)
        L_star = bkd.cholesky(C_star)

        # Sample realizations
        n_realizations = 10000
        mu_samples = np.empty(n_realizations)
        gamma_samples = np.empty(n_realizations)
        for r in range(n_realizations):
            z_r = bkd.array(np.random.randn(nquad_total))
            f_r = m_star + L_star @ z_r
            mu_r = float(bkd.to_numpy(bkd.sum(w * f_r)))
            kappa_r = float(bkd.to_numpy(bkd.sum(w * f_r * f_r)))
            mu_samples[r] = mu_r
            gamma_samples[r] = kappa_r - mu_r * mu_r

        return mu_samples, gamma_samples

    @slow_test
    def test_gp_variance_of_posterior_mean_vs_mc(self, bkd) -> None:
        """Var[mu_f | y] = Var[integral f(z) rho(z) dz | y] via GP realizations."""
        stats, gp, marginals, _, _, nquad = self._setup(bkd)
        mu_samples, _ = self._sample_posterior_statistics(bkd, gp, marginals, nquad)
        mc_var = np.var(mu_samples)
        formula_value = stats.gp_variance_of_posterior_mean()
        bkd.assert_allclose(
            bkd.asarray([formula_value]),
            bkd.asarray([mc_var]),
            rtol=0.1,
        )

    @slow_test
    def test_input_mean_of_posterior_variance_vs_mc(self, bkd) -> None:
        """E[gamma_f] via GP realizations.

        E[gamma_f] = E[int f**2 rho dz - (int f rho dz)**2].
        """
        stats, gp, marginals, _, _, nquad = self._setup(bkd)
        _, gamma_samples = self._sample_posterior_statistics(bkd, gp, marginals, nquad)
        mc_mean_gamma = np.mean(gamma_samples)
        formula_value = stats.input_mean_of_posterior_variance()
        bkd.assert_allclose(
            bkd.asarray([formula_value]),
            bkd.asarray([mc_mean_gamma]),
            rtol=0.1,
        )


# ===================================================================
# Test 5: Sensitivity indices properties and Monte Carlo validation
# ===================================================================


def _mc_sobol_indices(
    gp,
    marginals,
    nquad,
    bkd,
    n_realizations=20000,
    seed=7777,
):
    """Estimate Sobol indices via Monte Carlo from GP realizations.

    Returns (main_effects, total_effects) as dicts mapping dim -> float.
    """
    np.random.seed(seed)
    bases = _create_quadrature_bases(marginals, nquad, bkd)
    nvars = len(marginals)

    pts_1d = []
    wts_1d = []
    for b in bases:
        p, w = b.quadrature_rule()
        pts_1d.append(bkd.to_numpy(p).flatten())
        wts_1d.append(bkd.to_numpy(w).flatten())

    grid = list(iterproduct(*pts_1d))
    wgrid = list(iterproduct(*wts_1d))
    Z_np = np.array(grid).T
    np.prod(np.array(wgrid), axis=1)
    Z = bkd.array(Z_np)
    nq = Z.shape[1]

    # Posterior mean and covariance at quadrature points
    X_train = gp.data().X()
    K_zz = gp.kernel()(Z, Z)
    K_zt = gp.kernel()(Z, X_train)
    chol = gp.cholesky()
    alpha = chol.solve(gp.data().y().T)
    m_star = bkd.reshape(K_zt @ alpha, (-1,))
    A_inv_K_tz = chol.solve(K_zt.T)
    C_star = K_zz - K_zt @ A_inv_K_tz
    C_star = C_star + 1e-8 * bkd.eye(nq)
    L_star = bkd.cholesky(C_star)

    # Grid dimensions per variable
    n_per_dim = [len(pts_1d[d]) for d in range(nvars)]
    w_list = [wts_1d[d] for d in range(nvars)]

    # Sample realizations and compute per-realization Sobol variances
    main_var = {d: np.empty(n_realizations) for d in range(nvars)}
    total_var_arr = np.empty(n_realizations)

    for r in range(n_realizations):
        z_r = bkd.array(np.random.randn(nq))
        f_r = bkd.to_numpy(m_star + L_star @ z_r)
        # Reshape to tensor product grid: (n0, n1, ...)
        F = f_r.reshape(n_per_dim)

        # Full weight tensor
        w_full = w_list[0]
        for d in range(1, nvars):
            w_full = np.multiply.outer(w_full, w_list[d])

        mu = np.sum(w_full * F)
        total_var_arr[r] = np.sum(w_full * F**2) - mu**2

        for dim in range(nvars):
            # E[f | z_dim]: integrate out all other dimensions
            # Sum over all axes except dim, weighted
            axes_to_sum = list(range(nvars))
            axes_to_sum.remove(dim)
            E_given_dim = F.copy()
            # Integrate out other dims one at a time (from highest to lowest
            # to keep axis indices stable)
            for ax in sorted(axes_to_sum, reverse=True):
                # Weight along axis ax
                shape = [1] * E_given_dim.ndim
                shape[ax] = n_per_dim[ax]
                w_ax = w_list[ax].reshape(shape)
                E_given_dim = np.sum(w_ax * E_given_dim, axis=ax)
            # E_given_dim has shape (n_dim,)
            main_var[dim][r] = np.sum(w_list[dim] * E_given_dim**2) - mu**2

    mc_E_gamma = np.mean(total_var_arr)
    mc_main = {}
    mc_total = {}
    for dim in range(nvars):
        mc_main[dim] = float(np.mean(main_var[dim]) / mc_E_gamma)
        # Total effect: T_i = 1 - V_{~i} / E[gamma]
        # V_{~i} = E[gamma] - V_{T,i}, but easier: T_i = E[Var[f|z_{~i}]] / E[gamma]
        # Var[f|z_{~i}] = integral f**2 rho_i dz_i - (integral f rho_i dz_i)**2
        # That's the variance integrating ONLY over dim i.

    # Recompute total effects: for T_i, integrate out only dim i
    for dim in range(nvars):
        total_var_dim = np.empty(n_realizations)
        np.random.seed(seed)
        for r in range(n_realizations):
            z_r = bkd.array(np.random.randn(nq))
            f_r = bkd.to_numpy(m_star + L_star @ z_r)
            F = f_r.reshape(n_per_dim)

            # E[f | z_{~i}]: integrate out dim i only
            shape = [1] * F.ndim
            shape[dim] = n_per_dim[dim]
            w_dim = w_list[dim].reshape(shape)
            E_no_i = np.sum(w_dim * F, axis=dim)  # shape without dim
            E2_no_i = np.sum(w_dim * F**2, axis=dim)  # E[f**2|z_{~i}]
            var_given_not_i = E2_no_i - E_no_i**2  # Var[f|z_{~i}]

            # Average Var[f|z_{~i}] over z_{~i}
            # Build weight for remaining dims
            remaining_axes = list(range(nvars))
            remaining_axes.remove(dim)
            w_remaining = np.array(1.0)
            for ax in remaining_axes:
                w_remaining = np.multiply.outer(w_remaining, w_list[ax])
            w_remaining = w_remaining.reshape(var_given_not_i.shape)
            total_var_dim[r] = np.sum(w_remaining * var_given_not_i)

        mc_total[dim] = float(np.mean(total_var_dim) / mc_E_gamma)

    return mc_main, mc_total


class TestSensitivityEndToEnd:
    """Verify sensitivity indices satisfy basic properties and match MC.

    Uses s=2.0 and nugget approx 0 to catch scaling bugs.
    """

    def _setup(self, bkd):
        X_train, y_train = _make_training_data(bkd)
        nvars = 2
        nquad = 30
        length_scales = [1.0, 0.5]

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        s = 2.0
        kernel = _create_scaled_kernel(s, length_scales, bkd)
        gp = _fit_gp(kernel, nvars, X_train, y_train, bkd)
        sens = _create_sensitivity(gp, marginals, nquad, bkd)

        return sens, gp, marginals, nvars, nquad

    def test_main_effects_nonnegative(self, bkd) -> None:
        """Each main effect index should be >= 0."""
        sens, _, _, _, _ = self._setup(bkd)
        indices = sens.main_effect_indices()
        for dim, val in indices.items():
            val_float = float(bkd.to_numpy(bkd.asarray([val]))[0])
            assert val_float >= -1e-6, (
                f"Main effect for dim {dim} negative: {val_float}"
            )

    def test_main_effects_sum_leq_one(self, bkd) -> None:
        """Sum of main effect indices should be <= 1."""
        sens, _, _, _, _ = self._setup(bkd)
        indices = sens.main_effect_indices()
        total = sum(indices.values())
        total_float = float(bkd.to_numpy(bkd.asarray([total]))[0])
        assert total_float <= 1.0 + 1e-6, f"Main effects sum > 1: {total_float}"

    def test_total_effects_geq_main_effects(self, bkd) -> None:
        """Each total effect should be >= its main effect."""
        sens, _, _, _, _ = self._setup(bkd)
        main = sens.main_effect_indices()
        total = sens.total_effect_indices()
        for dim in main:
            m = float(bkd.to_numpy(bkd.asarray([main[dim]]))[0])
            t = float(bkd.to_numpy(bkd.asarray([total[dim]]))[0])
            assert t >= m - 1e-6, (
                f"Total effect < main effect for dim {dim}: T={t}, S={m}"
            )

    def test_total_effects_sum_geq_one(self, bkd) -> None:
        """Sum of total effect indices should be >= 1."""
        sens, _, _, _, _ = self._setup(bkd)
        indices = sens.total_effect_indices()
        total = sum(indices.values())
        total_float = float(bkd.to_numpy(bkd.asarray([total]))[0])
        assert total_float >= 1.0 - 1e-6, f"Total effects sum < 1: {total_float}"

    @slower_test
    def test_sobol_indices_vs_mc(self, bkd) -> None:
        """Individual main and total Sobol indices match MC."""
        sens, gp, marginals, nvars, nquad = self._setup(bkd)
        mc_main, mc_total = _mc_sobol_indices(gp, marginals, nquad, bkd)
        formula_main = sens.main_effect_indices()
        formula_total = sens.total_effect_indices()
        for dim in range(nvars):
            bkd.assert_allclose(
                bkd.asarray([formula_main[dim]]),
                bkd.asarray([mc_main[dim]]),
                rtol=0.05,
                atol=0.01,
            )
            bkd.assert_allclose(
                bkd.asarray([formula_total[dim]]),
                bkd.asarray([mc_total[dim]]),
                rtol=0.05,
                atol=0.01,
            )


# ===================================================================
# Test 6: Sensitivity with nonzero nugget and s**2 != 1
# ===================================================================


class TestSensitivityWithNugget:
    """Verify sensitivity indices with nonzero nugget sigma**2 and s**2 != 1.

    When sigma**2 != 0, the exact s**2 scaling no longer simplifies
    (A = s**2*C + sigma**2*I).
    This test verifies correctness by comparing individual indices against MC.
    """

    def _setup(self, bkd):
        X_train, y_train = _make_training_data(bkd)
        nvars = 2
        nquad = 30
        length_scales = [1.0, 0.5]

        marginals = [
            UniformMarginal(-1.0, 1.0, bkd),
            UniformMarginal(-1.0, 1.0, bkd),
        ]

        # s=3.0 with IIDGaussianNoise(sigma**2=0.1): no simplification possible
        s = 3.0
        signal_kernel = _create_scaled_kernel(s, length_scales, bkd)
        noise = IIDGaussianNoise(0.1, (0.01, 1.0), bkd, fixed=True)
        kernel = signal_kernel + noise
        gp = _fit_gp(
            kernel,
            nvars,
            X_train,
            y_train,
            bkd,
        )
        sens = _create_sensitivity(gp, marginals, nquad, bkd)

        return sens, gp, marginals, nvars, nquad

    def test_properties(self, bkd) -> None:
        """Basic Sobol index properties hold with nonzero nugget."""
        sens, _, _, nvars, _ = self._setup(bkd)
        main = sens.main_effect_indices()
        total = sens.total_effect_indices()

        for dim in range(nvars):
            m = float(bkd.to_numpy(bkd.asarray([main[dim]]))[0])
            t = float(bkd.to_numpy(bkd.asarray([total[dim]]))[0])
            assert m >= -1e-6, f"S_{dim} negative: {m}"
            assert t >= -1e-6, f"T_{dim} negative: {t}"
            assert t >= m - 1e-6, f"T_{dim} < S_{dim}: T={t}, S={m}"

        main_sum = float(bkd.to_numpy(bkd.asarray([sum(main.values())]))[0])
        total_sum = float(bkd.to_numpy(bkd.asarray([sum(total.values())]))[0])
        assert main_sum <= 1.0 + 1e-6
        assert total_sum >= 1.0 - 1e-6

    @slower_test
    def test_sobol_indices_vs_mc(self, bkd) -> None:
        """Individual main and total Sobol indices match MC with nugget."""
        sens, gp, marginals, nvars, nquad = self._setup(bkd)
        mc_main, mc_total = _mc_sobol_indices(
            gp,
            marginals,
            nquad,
            bkd,
        )
        formula_main = sens.main_effect_indices()
        formula_total = sens.total_effect_indices()
        for dim in range(nvars):
            bkd.assert_allclose(
                bkd.asarray([formula_main[dim]]),
                bkd.asarray([mc_main[dim]]),
                rtol=0.05,
                atol=0.01,
            )
            bkd.assert_allclose(
                bkd.asarray([formula_total[dim]]),
                bkd.asarray([mc_total[dim]]),
                rtol=0.05,
                atol=0.01,
            )
