"""
Numerical tests for GP statistics scaling with kernel variance s².

Unlike the SymPy tests (which verify algebraic identities tautologically),
these tests verify that the actual code produces numerically correct results
by:

1. Verifying that the integral calculator decomposes scaled kernels and
   returns C-integrals (unit-variance) regardless of s².
2. Running the full statistics pipeline with different s² values and
   verifying that:
   - η (mean of mean) is invariant to s²
   - Var[μ] scales as s²
   - E[γ] is affine in s²
   - α_K = s⁻² α_C
3. Comparing GP statistics against Monte Carlo estimates.

All tests use nugget=0 or nugget=1e-10 to avoid the nugget breaking
the exact s² cancellation (A_K = s²C + σ²I ≠ s²(C + σ²I) when σ² ≠ 0).
"""

import math
import unittest
from itertools import product as iterproduct
from typing import Any, Generic, List

import numpy as np
import torch
from numpy.typing import NDArray

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
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.backends.torch import TorchBkd
from pyapprox.util.test_utils import (
    load_tests,  # noqa: F401
    slow_test,
    slower_test,
)

# ===================================================================
# Helpers
# ===================================================================

# Nugget small enough that σ²/s² ≈ 0, preserving exact scaling.
_NUGGET = 1e-10


def _create_quadrature_bases(
    marginals: List[Any], nquad_points: int, bkd: Backend[Array]
) -> List[Any]:
    factories = create_basis_factories(marginals, bkd, "gauss")
    bases = [f.create_basis() for f in factories]
    for b in bases:
        b.set_nterms(nquad_points)
    return bases


def _create_separable_kernel(
    length_scales: List[float], bkd: Backend[Array]
) -> SeparableProductKernel[Array]:
    kernels_1d = [
        SquaredExponentialKernel([ls], (0.1, 10.0), 1, bkd) for ls in length_scales
    ]
    return SeparableProductKernel(kernels_1d, bkd)


def _create_scaled_kernel(
    s_value: float,
    length_scales: List[float],
    bkd: Backend[Array],
) -> Any:
    base = _create_separable_kernel(length_scales, bkd)
    nvars = len(length_scales)
    scaling = PolynomialScaling([s_value], (0.01, 100.0), bkd, nvars=nvars, fixed=False)
    return scaling * base


def _fit_gp(
    kernel: Any,
    nvars: int,
    X_train: Any,
    y_train: Any,
    bkd: Backend[Array],
    nugget: float = _NUGGET,
) -> ExactGaussianProcess[Array]:
    gp: ExactGaussianProcess[Array] = ExactGaussianProcess(
        kernel, nvars=nvars, bkd=bkd, nugget=nugget
    )
    gp.hyp_list().set_all_inactive()
    gp.fit(X_train, y_train)
    return gp


def _create_stats(
    gp: Any, marginals: List[Any], nquad: int, bkd: Backend[Array]
) -> GaussianProcessStatistics[Array]:
    bases = _create_quadrature_bases(marginals, nquad, bkd)
    calc: SeparableKernelIntegralCalculator[Array] = SeparableKernelIntegralCalculator(
        gp, bases, marginals, bkd=bkd
    )
    return GaussianProcessStatistics(gp, calc)


def _create_sensitivity(
    gp: Any, marginals: List[Any], nquad: int, bkd: Backend[Array]
) -> GaussianProcessSensitivity[Array]:
    bases = _create_quadrature_bases(marginals, nquad, bkd)
    calc: SeparableKernelIntegralCalculator[Array] = SeparableKernelIntegralCalculator(
        gp, bases, marginals, bkd=bkd
    )
    stats = GaussianProcessStatistics(gp, calc)
    return GaussianProcessSensitivity(stats)


def _make_training_data(
    bkd: Backend[Array], nvars: int = 2, n_train: int = 5
) -> tuple[Any, Any]:
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


class TestIntegralCalculatorDecomposition(Generic[Array], unittest.TestCase):
    """Verify that the integral calculator extracts the base kernel C
    from a scaled kernel K = s²C and computes identical C-integrals
    regardless of s².

    The calculator always computes integrals of C (not K).
    The s² factor is applied in the statistics formulas.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        X_train, y_train = _make_training_data(self._bkd)
        length_scales = [1.0, 0.5]
        nvars = 2
        nquad = 30

        marginals: List[Any] = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        # Unscaled GP (C kernel)
        kernel_C = _create_separable_kernel(length_scales, self._bkd)
        gp_C = _fit_gp(kernel_C, nvars, X_train, y_train, self._bkd)
        bases_C = _create_quadrature_bases(marginals, nquad, self._bkd)
        self._calc_C: SeparableKernelIntegralCalculator[Array] = (
            SeparableKernelIntegralCalculator(gp_C, bases_C, marginals, bkd=self._bkd)
        )

        # Scaled GP (K = s²C kernel, s=2.5)
        kernel_K = _create_scaled_kernel(2.5, length_scales, self._bkd)
        gp_K = _fit_gp(kernel_K, nvars, X_train, y_train, self._bkd)
        bases_K = _create_quadrature_bases(marginals, nquad, self._bkd)
        self._calc_K: SeparableKernelIntegralCalculator[Array] = (
            SeparableKernelIntegralCalculator(gp_K, bases_K, marginals, bkd=self._bkd)
        )

    def test_tau_identical(self) -> None:
        """Calculator returns identical τ for C and K = s²C."""
        self._bkd.assert_allclose(
            self._calc_K.tau_C(), self._calc_C.tau_C(), rtol=1e-12
        )

    def test_P_identical(self) -> None:
        """Calculator returns identical P for C and K = s²C."""
        self._bkd.assert_allclose(self._calc_K.P(), self._calc_C.P(), rtol=1e-12)

    def test_u_identical(self) -> None:
        """Calculator returns identical u for C and K = s²C."""
        self._bkd.assert_allclose(
            self._bkd.asarray([self._calc_K.u()]),
            self._bkd.asarray([self._calc_C.u()]),
            rtol=1e-12,
        )

    def test_nu_identical(self) -> None:
        """Calculator returns identical ν for C and K = s²C."""
        self._bkd.assert_allclose(
            self._bkd.asarray([self._calc_K.nu()]),
            self._bkd.asarray([self._calc_C.nu()]),
            rtol=1e-12,
        )

    def test_Pi_identical(self) -> None:
        """Calculator returns identical Π for C and K = s²C."""
        self._bkd.assert_allclose(self._calc_K.Pi(), self._calc_C.Pi(), rtol=1e-12)

    def test_xi1_identical(self) -> None:
        """Calculator returns identical ξ₁ for C and K = s²C."""
        self._bkd.assert_allclose(
            self._bkd.asarray([self._calc_K.xi1()]),
            self._bkd.asarray([self._calc_C.xi1()]),
            rtol=1e-12,
        )

    def test_Gamma_identical(self) -> None:
        """Calculator returns identical Γ for C and K = s²C."""
        self._bkd.assert_allclose(
            self._calc_K.Gamma(), self._calc_C.Gamma(), rtol=1e-12
        )


class TestIntegralCalculatorDecompositionNumpy(
    TestIntegralCalculatorDecomposition[NDArray[Any]]
):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestIntegralCalculatorDecompositionTorch(
    TestIntegralCalculatorDecomposition[torch.Tensor]
):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# ===================================================================
# Test 2: End-to-end statistics scaling across s² values
# ===================================================================


class TestStatisticsScalingEndToEnd(Generic[Array], unittest.TestCase):
    """Verify that computed statistics scale correctly when s² changes.

    Uses nugget=1e-10 so σ²/s² ≈ 0 and the exact scaling holds.

    Checks:
    - η (mean of mean) is INVARIANT to s²
    - Var[μ] scales as s²
    - E[γ] is affine in s²: E[γ] = (ζ - η²) + s²(v² - ς²)
    - Var[γ] is non-negative for all s
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        X_train, y_train = _make_training_data(self._bkd)
        length_scales = [1.0, 0.5]
        nvars = 2
        nquad = 30

        self._marginals: List[Any] = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        self._s_values = [0.5, 1.0, 2.0, 5.0]
        self._stats_by_s: dict[float, GaussianProcessStatistics[Array]] = {}

        for s in self._s_values:
            kernel = _create_scaled_kernel(s, length_scales, self._bkd)
            gp = _fit_gp(kernel, nvars, X_train, y_train, self._bkd)
            stats = _create_stats(gp, self._marginals, nquad, self._bkd)
            self._stats_by_s[s] = stats

    def test_eta_invariant_across_s_values(self) -> None:
        """η = τᵀA⁻¹y should be invariant to s².

        With nugget ≈ 0: α_K = (s²C)⁻¹y = s⁻²C⁻¹y = s⁻²α_C.
        The formula computes η = α_K @ τ_K where τ_K = s²τ_C (but
        the calculator returns τ_C and moments.py multiplies by s²
        implicitly through α). Actually η = alpha @ tau where alpha
        already includes s⁻² and tau is C-integral. So η = s⁻²α_C @ τ_C
        ... wait, let's just check numerically.
        """
        ref_eta = self._stats_by_s[self._s_values[0]].mean_of_mean()
        for s in self._s_values[1:]:
            eta = self._stats_by_s[s].mean_of_mean()
            self._bkd.assert_allclose(
                self._bkd.asarray([eta]),
                self._bkd.asarray([ref_eta]),
                rtol=1e-6,
            )

    def test_variance_of_mean_scales_as_s2(self) -> None:
        """Var[μ] = s² · ς² should scale proportionally to s²."""
        ref_s = self._s_values[0]
        ref_s2 = ref_s * ref_s
        ref_var = self._stats_by_s[ref_s].variance_of_mean()

        for s in self._s_values[1:]:
            s2 = s * s
            var_mu = self._stats_by_s[s].variance_of_mean()
            expected_ratio = s2 / ref_s2
            actual_ratio = var_mu / ref_var
            self._bkd.assert_allclose(
                self._bkd.asarray([actual_ratio]),
                self._bkd.asarray([expected_ratio]),
                rtol=1e-6,
            )

    def test_mean_of_variance_affine_in_s2(self) -> None:
        """E[γ] = (ζ - η²) + s²(v² - ς²) is affine in s².

        Extract the affine coefficients from two s values, then verify
        the relationship holds at all other s values.
        """
        s1 = self._s_values[0]
        s2_val = self._s_values[2]
        s1_sq = s1 * s1
        s2_sq = s2_val * s2_val

        E_gamma_1 = self._stats_by_s[s1].mean_of_variance()
        E_gamma_2 = self._stats_by_s[s2_val].mean_of_variance()

        # E[γ](s²) = A + B·s²
        B = (E_gamma_2 - E_gamma_1) / (s2_sq - s1_sq)
        A = E_gamma_1 - B * s1_sq

        for s in self._s_values:
            s_sq = s * s
            expected = A + B * s_sq
            actual = self._stats_by_s[s].mean_of_variance()
            self._bkd.assert_allclose(
                self._bkd.asarray([actual]),
                self._bkd.asarray([expected]),
                rtol=1e-6,
            )

    def test_variance_of_variance_nonnegative(self) -> None:
        """Var[γ] should be non-negative for all s."""
        for s in self._s_values:
            var_var = self._stats_by_s[s].variance_of_variance()
            self.assertGreaterEqual(
                float(self._bkd.to_numpy(self._bkd.asarray([var_var]))[0]),
                0.0,
                f"Var[γ] should be non-negative for s={s}",
            )


class TestStatisticsScalingEndToEndNumpy(TestStatisticsScalingEndToEnd[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestStatisticsScalingEndToEndTorch(TestStatisticsScalingEndToEnd[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# ===================================================================
# Test 3: Alpha scaling verification
# ===================================================================


class TestAlphaScaling(Generic[Array], unittest.TestCase):
    """Verify α_K = s⁻² α_C numerically.

    With nugget ≈ 0: A_K = s²C, so A_K⁻¹ = s⁻²C⁻¹, giving α_K = s⁻²α_C.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        X_train, y_train = _make_training_data(self._bkd)
        length_scales = [1.0, 0.5]
        nvars = 2

        s = 3.0
        self._s2 = s * s

        kernel_C = _create_separable_kernel(length_scales, self._bkd)
        self._gp_C = _fit_gp(kernel_C, nvars, X_train, y_train, self._bkd)

        kernel_K = _create_scaled_kernel(s, length_scales, self._bkd)
        self._gp_K = _fit_gp(kernel_K, nvars, X_train, y_train, self._bkd)

    def test_alpha_K_equals_s_minus_2_alpha_C(self) -> None:
        """α_K = s⁻² α_C (exact with nugget ≈ 0)."""
        alpha_C = self._gp_C.alpha()
        alpha_K = self._gp_K.alpha()
        expected = alpha_C / self._s2
        self._bkd.assert_allclose(alpha_K, expected, rtol=1e-6)


class TestAlphaScalingNumpy(TestAlphaScaling[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestAlphaScalingTorch(TestAlphaScaling[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# ===================================================================
# Test 4: Monte Carlo verification of statistics formulas
# ===================================================================


class TestStatisticsVsMonteCarlo(Generic[Array], unittest.TestCase):
    """Verify GP statistics formulas against Monte Carlo estimates.

    For a fitted GP, sample input points X from ρ, evaluate the posterior
    mean μ(X) and variance γ(X) at each, and compute sample statistics.
    Compare against the closed-form formulas.

    This is the strongest test: it verifies the formulas produce correct
    numerical results, not just correct scaling.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        X_train, y_train = _make_training_data(self._bkd)
        self._nvars = 2
        self._nquad = 30
        self._n_mc = 10000

        length_scales = [1.0, 0.5]
        self._marginals: List[Any] = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        # Test with s=2.0 to catch scaling bugs
        s = 2.0
        kernel = _create_scaled_kernel(s, length_scales, self._bkd)
        self._gp = _fit_gp(kernel, self._nvars, X_train, y_train, self._bkd)
        self._stats = _create_stats(self._gp, self._marginals, self._nquad, self._bkd)

        # Generate MC samples from Uniform[-1,1]^2
        np.random.seed(123)
        mc_np = np.random.rand(self._nvars, self._n_mc) * 2 - 1
        self._X_mc = self._bkd.array(mc_np)

    def test_mean_of_mean_vs_mc(self) -> None:
        """E[μ(X)] ≈ (1/N) Σ μ(X_i) by Monte Carlo."""
        mu_mc = self._gp.predict(self._X_mc)  # (nqoi, n_mc)
        mc_estimate = self._bkd.mean(mu_mc)
        formula_value = self._stats.mean_of_mean()
        self._bkd.assert_allclose(
            self._bkd.asarray([formula_value]),
            self._bkd.asarray([mc_estimate]),
            rtol=0.05,
        )

    def _sample_posterior_statistics(self) -> tuple[Any, Any]:
        """Sample GP posterior realizations and compute μ_r, γ_r.

        For each realization f^(r) ~ GP(m*, C*):
        - μ_r = ∫ f^(r)(z) ρ(z) dz  (integrated mean)
        - κ_r = ∫ f^(r)(z)² ρ(z) dz  (integrated second moment)
        - γ_r = κ_r - μ_r²  (variance of realization over input space)

        Returns (mu_samples, gamma_samples) as numpy arrays.
        """
        np.random.seed(7777)

        # Build tensor product quadrature grid
        bases = _create_quadrature_bases(self._marginals, self._nquad, self._bkd)
        pts_1d = []
        wts_1d = []
        for b in bases:
            p, w = b.quadrature_rule()
            pts_1d.append(self._bkd.to_numpy(p).flatten())
            wts_1d.append(self._bkd.to_numpy(w).flatten())

        grid = list(iterproduct(*pts_1d))
        wgrid = list(iterproduct(*wts_1d))
        Z_np = np.array(grid).T  # (nvars, nquad_total)
        w_np = np.prod(np.array(wgrid), axis=1)  # (nquad_total,)
        Z = self._bkd.array(Z_np)
        w = self._bkd.array(w_np)
        nquad_total = Z.shape[1]

        # Posterior mean and covariance at Z
        X_train = self._gp.data().X()
        y_train = self._gp.data().y()
        K_zz = self._gp.kernel()(Z, Z)
        K_zt = self._gp.kernel()(Z, X_train)
        chol = self._gp.cholesky()
        alpha = chol.solve(y_train.T)  # (n_train, 1)
        m_star = K_zt @ alpha  # (nquad_total, 1)
        m_star = self._bkd.reshape(m_star, (-1,))

        A_inv_K_tz = chol.solve(K_zt.T)  # (n_train, nquad_total)
        C_star = K_zz - K_zt @ A_inv_K_tz
        C_star = C_star + 1e-8 * self._bkd.eye(nquad_total)
        L_star = self._bkd.cholesky(C_star)

        # Sample realizations
        n_realizations = 10000
        mu_samples = np.empty(n_realizations)
        gamma_samples = np.empty(n_realizations)
        for r in range(n_realizations):
            z_r = self._bkd.array(np.random.randn(nquad_total))
            f_r = m_star + L_star @ z_r
            mu_r = float(self._bkd.to_numpy(self._bkd.sum(w * f_r)))
            kappa_r = float(self._bkd.to_numpy(self._bkd.sum(w * f_r * f_r)))
            mu_samples[r] = mu_r
            gamma_samples[r] = kappa_r - mu_r * mu_r

        return mu_samples, gamma_samples

    @slow_test
    def test_variance_of_mean_vs_mc(self) -> None:
        """Var[μ_f | y] = Var[∫ f(z) ρ(z) dz | y] via GP realizations."""
        mu_samples, _ = self._sample_posterior_statistics()
        mc_var = np.var(mu_samples)
        formula_value = self._stats.variance_of_mean()
        self._bkd.assert_allclose(
            self._bkd.asarray([formula_value]),
            self._bkd.asarray([mc_var]),
            rtol=0.1,
        )

    @slow_test
    def test_mean_of_variance_vs_mc(self) -> None:
        """E[γ_f] = E[∫f²ρ dz - (∫fρ dz)²] via GP realizations."""
        _, gamma_samples = self._sample_posterior_statistics()
        mc_mean_gamma = np.mean(gamma_samples)
        formula_value = self._stats.mean_of_variance()
        self._bkd.assert_allclose(
            self._bkd.asarray([formula_value]),
            self._bkd.asarray([mc_mean_gamma]),
            rtol=0.1,
        )


class TestStatisticsVsMonteCarloNumpy(TestStatisticsVsMonteCarlo[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestStatisticsVsMonteCarloTorch(TestStatisticsVsMonteCarlo[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# ===================================================================
# Test 5: Sensitivity indices properties and Monte Carlo validation
# ===================================================================


def _mc_sobol_indices(
    gp: Any,
    marginals: List[Any],
    nquad: int,
    bkd: Backend[Array],
    n_realizations: int = 20000,
    seed: int = 7777,
) -> tuple[dict[int, float], dict[int, float]]:
    """Estimate Sobol indices via Monte Carlo from GP realizations.

    Returns (main_effects, total_effects) as dicts mapping dim -> float.
    """
    np.random.seed(seed)
    bases = _create_quadrature_bases(marginals, nquad, bkd)
    nvars = len(marginals)

    pts_1d: List[Any] = []
    wts_1d: List[Any] = []
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
    main_var: dict[int, Any] = {d: np.empty(n_realizations) for d in range(nvars)}
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
    mc_main: dict[int, float] = {}
    mc_total: dict[int, float] = {}
    for dim in range(nvars):
        mc_main[dim] = float(np.mean(main_var[dim]) / mc_E_gamma)
        # Total effect: T_i = 1 - V_{~i} / E[γ]
        # V_{~i} = E[γ] - V_{T,i}, but easier: T_i = E[Var[f|z_{~i}]] / E[γ]
        # Var[f|z_{~i}] = ∫ f² ρ_i dz_i - (∫ f ρ_i dz_i)²
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
            E2_no_i = np.sum(w_dim * F**2, axis=dim)  # E[f²|z_{~i}]
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


class TestSensitivityEndToEnd(Generic[Array], unittest.TestCase):
    """Verify sensitivity indices satisfy basic properties and match MC.

    Uses s=2.0 and nugget≈0 to catch scaling bugs.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        X_train, y_train = _make_training_data(self._bkd)
        self._nvars = 2
        self._nquad = 30
        length_scales = [1.0, 0.5]

        self._marginals: List[Any] = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        s = 2.0
        kernel = _create_scaled_kernel(s, length_scales, self._bkd)
        gp = _fit_gp(kernel, self._nvars, X_train, y_train, self._bkd)
        self._gp = gp
        self._sens = _create_sensitivity(gp, self._marginals, self._nquad, self._bkd)

    def test_main_effects_nonnegative(self) -> None:
        """Each main effect index should be ≥ 0."""
        indices = self._sens.main_effect_indices()
        for dim, val in indices.items():
            val_float = float(self._bkd.to_numpy(self._bkd.asarray([val]))[0])
            self.assertGreaterEqual(
                val_float,
                -1e-6,
                f"Main effect for dim {dim} negative: {val_float}",
            )

    def test_main_effects_sum_leq_one(self) -> None:
        """Sum of main effect indices should be ≤ 1."""
        indices = self._sens.main_effect_indices()
        total = sum(indices.values())
        total_float = float(self._bkd.to_numpy(self._bkd.asarray([total]))[0])
        self.assertLessEqual(
            total_float,
            1.0 + 1e-6,
            f"Main effects sum > 1: {total_float}",
        )

    def test_total_effects_geq_main_effects(self) -> None:
        """Each total effect should be ≥ its main effect."""
        main = self._sens.main_effect_indices()
        total = self._sens.total_effect_indices()
        for dim in main:
            m = float(self._bkd.to_numpy(self._bkd.asarray([main[dim]]))[0])
            t = float(self._bkd.to_numpy(self._bkd.asarray([total[dim]]))[0])
            self.assertGreaterEqual(
                t,
                m - 1e-6,
                f"Total effect < main effect for dim {dim}: T={t}, S={m}",
            )

    def test_total_effects_sum_geq_one(self) -> None:
        """Sum of total effect indices should be ≥ 1."""
        indices = self._sens.total_effect_indices()
        total = sum(indices.values())
        total_float = float(self._bkd.to_numpy(self._bkd.asarray([total]))[0])
        self.assertGreaterEqual(
            total_float,
            1.0 - 1e-6,
            f"Total effects sum < 1: {total_float}",
        )

    @slower_test
    def test_sobol_indices_vs_mc(self) -> None:
        """Individual main and total Sobol indices match MC."""
        mc_main, mc_total = _mc_sobol_indices(
            self._gp, self._marginals, self._nquad, self._bkd
        )
        formula_main = self._sens.main_effect_indices()
        formula_total = self._sens.total_effect_indices()
        for dim in range(self._nvars):
            self._bkd.assert_allclose(
                self._bkd.asarray([formula_main[dim]]),
                self._bkd.asarray([mc_main[dim]]),
                rtol=0.05,
                atol=0.01,
            )
            self._bkd.assert_allclose(
                self._bkd.asarray([formula_total[dim]]),
                self._bkd.asarray([mc_total[dim]]),
                rtol=0.05,
                atol=0.01,
            )


class TestSensitivityEndToEndNumpy(TestSensitivityEndToEnd[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSensitivityEndToEndTorch(TestSensitivityEndToEnd[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


# ===================================================================
# Test 6: Sensitivity with nonzero nugget and s² ≠ 1
# ===================================================================


class TestSensitivityWithNugget(Generic[Array], unittest.TestCase):
    """Verify sensitivity indices with nonzero nugget σ² and s² ≠ 1.

    When σ² ≠ 0, the exact s² scaling no longer simplifies (A = s²C + σ²I).
    This test verifies correctness by comparing individual indices against MC.
    """

    __test__ = False

    def bkd(self) -> Backend[Array]:
        raise NotImplementedError

    def setUp(self) -> None:
        self._bkd = self.bkd()
        X_train, y_train = _make_training_data(self._bkd)
        self._nvars = 2
        self._nquad = 30
        length_scales = [1.0, 0.5]

        self._marginals: List[Any] = [
            UniformMarginal(-1.0, 1.0, self._bkd),
            UniformMarginal(-1.0, 1.0, self._bkd),
        ]

        # s=3.0 with IIDGaussianNoise(σ²=0.1): no simplification possible
        s = 3.0
        signal_kernel = _create_scaled_kernel(s, length_scales, self._bkd)
        noise = IIDGaussianNoise(0.1, (0.01, 1.0), self._bkd, fixed=True)
        kernel = signal_kernel + noise
        gp = _fit_gp(
            kernel,
            self._nvars,
            X_train,
            y_train,
            self._bkd,
        )
        self._gp = gp
        self._sens = _create_sensitivity(gp, self._marginals, self._nquad, self._bkd)

    def test_properties(self) -> None:
        """Basic Sobol index properties hold with nonzero nugget."""
        main = self._sens.main_effect_indices()
        total = self._sens.total_effect_indices()

        for dim in range(self._nvars):
            m = float(self._bkd.to_numpy(self._bkd.asarray([main[dim]]))[0])
            t = float(self._bkd.to_numpy(self._bkd.asarray([total[dim]]))[0])
            self.assertGreaterEqual(m, -1e-6, f"S_{dim} negative: {m}")
            self.assertGreaterEqual(t, -1e-6, f"T_{dim} negative: {t}")
            self.assertGreaterEqual(t, m - 1e-6, f"T_{dim} < S_{dim}: T={t}, S={m}")

        main_sum = float(self._bkd.to_numpy(self._bkd.asarray([sum(main.values())]))[0])
        total_sum = float(
            self._bkd.to_numpy(self._bkd.asarray([sum(total.values())]))[0]
        )
        self.assertLessEqual(main_sum, 1.0 + 1e-6)
        self.assertGreaterEqual(total_sum, 1.0 - 1e-6)

    @slower_test
    def test_sobol_indices_vs_mc(self) -> None:
        """Individual main and total Sobol indices match MC with nugget."""
        mc_main, mc_total = _mc_sobol_indices(
            self._gp,
            self._marginals,
            self._nquad,
            self._bkd,
        )
        formula_main = self._sens.main_effect_indices()
        formula_total = self._sens.total_effect_indices()
        for dim in range(self._nvars):
            self._bkd.assert_allclose(
                self._bkd.asarray([formula_main[dim]]),
                self._bkd.asarray([mc_main[dim]]),
                rtol=0.05,
                atol=0.01,
            )
            self._bkd.assert_allclose(
                self._bkd.asarray([formula_total[dim]]),
                self._bkd.asarray([mc_total[dim]]),
                rtol=0.05,
                atol=0.01,
            )


class TestSensitivityWithNuggetNumpy(TestSensitivityWithNugget[NDArray[Any]]):
    __test__ = True

    def bkd(self) -> NumpyBkd:
        return NumpyBkd()


class TestSensitivityWithNuggetTorch(TestSensitivityWithNugget[torch.Tensor]):
    __test__ = True

    def bkd(self) -> TorchBkd:
        torch.set_default_dtype(torch.float64)
        return TorchBkd()


if __name__ == "__main__":
    unittest.main()
