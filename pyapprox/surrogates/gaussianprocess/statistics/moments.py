"""
Gaussian Process statistics module for computing moments.

This module provides the GaussianProcessStatistics class which computes
statistical quantities from fitted Gaussian Processes:

- E[mu_f]: Mean of the GP mean (expected value of posterior mean)
- Var[mu_f]: Variance of the GP mean (uncertainty in posterior mean)
- E[gamma_f]: Mean of the GP variance (expected posterior variance)
- Var[gamma_f]: Variance of the GP variance

These statistics quantify uncertainty in GP predictions integrated over
the input space according to the input distribution.

All public methods return values in the original (user) output space
when an output transform is set on the GP. Internal computations use
private ``_*_scaled()`` methods that operate in scaled (kernel) space.
"""

from typing import Generic, Optional
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.gaussianprocess.output_transform import (
    OutputAffineTransformProtocol,
)
from pyapprox.surrogates.gaussianprocess.protocols import (
    PredictiveGPProtocol,
)
from pyapprox.surrogates.gaussianprocess.statistics.protocols import (
    KernelIntegralCalculatorProtocol,
)
from pyapprox.surrogates.gaussianprocess.statistics.decompose import (
    _decompose_kernel,
)


class GaussianProcessStatistics(Generic[Array]):
    """
    Compute statistical quantities from a fitted Gaussian Process.

    This class provides methods to compute integrated statistics of GP
    predictions over the input space. These are useful for:

    - Uncertainty quantification: How much uncertainty remains?
    - Design of experiments: Which regions need more samples?
    - Sensitivity analysis: Which inputs matter most?

    Mathematical Background
    -----------------------
    For a GP f with posterior mean mu(x) and variance gamma(x), this class
    computes:

    1. Mean of mean: eta = E[mu(X)] = tau^T A^{-1} y
       where tau_i = int C(x, x^(i)) rho(x) dx

    2. Variance of mean: Var[mu(X)] = s^2 * varsigma^2
       where varsigma^2 = u - tau^T A^{-1} tau
       and u = int int C(x, z) rho(x) rho(z) dx dz

    3. Mean of variance: E[gamma(X)] = zeta + s^2 * v^2 - eta^2 - s^2 * varsigma^2
       where zeta = y^T A^{-1} P A^{-1} y
       and v^2 = 1 - Tr[P A^{-1}]

    Here:
    - A = K(X, X) + nugget * I is the noisy kernel matrix
    - s^2 is the kernel variance hyperparameter
    - rho(x) is the input probability density

    If an output transform is set on the GP, all statistics are returned
    in the original (unscaled) output space.

    Parameters
    ----------
    gp : PredictiveGPProtocol[Array]
        A fitted Gaussian Process.
    integral_calculator : KernelIntegralCalculatorProtocol[Array]
        Calculator for kernel integrals (tau, P, u, etc.).

    Raises
    ------
    RuntimeError
        If the GP has not been fitted.

    Examples
    --------
    >>> from pyapprox.surrogates.gaussianprocess.statistics import (
    ...     SeparableKernelIntegralCalculator,
    ...     GaussianProcessStatistics,
    ... )
    >>> # Assume gp is a fitted GP with separable kernel
    >>> # and marginals is a list of marginal distributions
    >>> calc = SeparableKernelIntegralCalculator(gp, marginals, nquad_points=50, bkd=bkd)
    >>> stats = GaussianProcessStatistics(gp, calc)
    >>> mean_of_mean = stats.mean_of_mean()
    >>> var_of_mean = stats.variance_of_mean()
    >>> mean_of_var = stats.mean_of_variance()
    """

    def __init__(
        self,
        gp: PredictiveGPProtocol[Array],
        integral_calculator: KernelIntegralCalculatorProtocol[Array],
    ):
        # Validate GP is fitted
        if not gp.is_fitted():
            raise RuntimeError(
                "GP must be fitted before computing statistics. "
                "Call gp.fit(X_train, y_train) first."
            )

        self._gp = gp
        self._calc = integral_calculator
        self._bkd = gp.bkd()

        # Get training data
        self._y = gp.data().y()  # Shape: (nqoi, n_train)
        self._n_train = gp.data().n_samples()

        # Get the Cholesky factor for efficient solves
        # A^{-1} v = L^{-T} L^{-1} v
        self._cholesky = gp.cholesky()

        # Cache computed quantities
        self._cache: dict[str, Array] = {}

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def _solve(self, b: Array) -> Array:
        """
        Solve A x = b using the precomputed Cholesky factor.

        Parameters
        ----------
        b : Array
            Right-hand side, shape (n_train,) or (n_train, k).

        Returns
        -------
        x : Array
            Solution x = A^{-1} b, same shape as b.
        """
        return self._cholesky.solve(b)

    def _get_kernel_variance(self) -> Array:
        """
        Get the kernel variance hyperparameter (s^2).

        For standard separable kernels, this is 1.0. For kernels composed as
        PolynomialScaling([s]) * SeparableKernel, this extracts s^2.

        Returns
        -------
        s2 : Array
            Kernel variance (scalar).
        """
        _, s2 = _decompose_kernel(self._gp.kernel(), self._bkd)
        return s2

    def _get_output_transform(self) -> Optional[OutputAffineTransformProtocol[Array]]:
        """Get output transform from GP, or None."""
        return self._gp.output_transform()

    # ===== Private methods: compute in scaled (kernel) space =====

    def _mean_of_mean_scaled(self) -> Array:
        """Compute η in scaled (internal) space.

        eta_scaled = tau_K^T A^{-1} y

        Returns
        -------
        Array
            Scalar η_scaled.
        """
        if 'eta_scaled' in self._cache:
            return self._cache['eta_scaled']

        tau_K = self._calc.tau_K()  # Shape: (n_train,), includes s²
        alpha = self._gp.alpha()  # Shape: (nqoi, n_train)

        eta = alpha @ tau_K  # Shape: (nqoi,)

        # Squeeze to scalar if nqoi=1
        if eta.shape[0] == 1:
            eta = eta[0]

        self._cache['eta_scaled'] = eta
        return eta

    def _variance_of_mean_scaled(self) -> Array:
        """Compute Var[μ_f] in scaled (internal) space.

        Var[μ]_scaled = u_K - τ_K^T A^{-1} τ_K

        Returns
        -------
        Array
            Scalar Var[μ]_scaled (non-negative).
        """
        if 'var_mu_scaled' in self._cache:
            return self._cache['var_mu_scaled']

        s2 = self._get_kernel_variance()
        tau_K = self._calc.tau_K()  # s² τ_C, shape: (n_train,)
        u_K = s2 * self._calc.u()  # s² u_C, scalar

        tau_K_col = self._bkd.reshape(tau_K, (-1, 1))
        A_inv_tau_K = self._solve(tau_K_col)  # Shape: (n_train, 1)
        A_inv_tau_K = self._bkd.reshape(A_inv_tau_K, (-1,))
        var_of_mean = u_K - tau_K @ A_inv_tau_K

        # Ensure non-negative (numerical stability)
        var_of_mean = var_of_mean * (var_of_mean >= 0.0)

        self._cache['var_mu_scaled'] = var_of_mean
        return var_of_mean

    def _mean_of_variance_scaled(self) -> Array:
        """Compute E[γ_f] in scaled (internal) space.

        E[γ]_scaled = ζ + integrated_post_var - η² - var_mu

        Returns
        -------
        Array
            Scalar E[γ]_scaled (non-negative).
        """
        if 'E_gamma_scaled' in self._cache:
            return self._cache['E_gamma_scaled']

        # Scale C-integrals to K-quantities
        s2 = self._get_kernel_variance()
        s4 = s2 * s2
        P_C = self._calc.P()        # Shape: (n_train, n_train)
        P_K = s4 * P_C              # s⁴ P_C
        tau_K = self._calc.tau_K()   # s² τ_C, shape: (n_train,)
        u_K = s2 * self._calc.u()   # s² u_C, scalar

        # Compute A^{-1} y (already have as alpha)
        alpha = self._gp.alpha()  # Shape: (nqoi, n_train)

        # For single output, squeeze to 1D
        if alpha.shape[0] == 1:
            alpha_1d = self._bkd.reshape(alpha, (-1,))  # Shape: (n_train,)
        else:
            raise NotImplementedError(
                "mean_of_variance currently only supports single-output GPs (nqoi=1)"
            )

        # zeta = αᵀ P_K α
        P_K_alpha = P_K @ alpha_1d
        zeta = alpha_1d @ P_K_alpha

        # integrated_post_var = s² - tr[P_K A⁻¹]
        A_inv_P_K = self._solve(P_K)
        trace_P_K_A_inv = self._bkd.trace(A_inv_P_K)
        integrated_post_var = s2 - trace_P_K_A_inv

        # Use scaled versions of eta and var_mu
        eta = self._mean_of_mean_scaled()
        var_mu = self._variance_of_mean_scaled()

        # E[γ] = ζ + integrated_post_var - η² - var_mu
        mean_of_var = zeta + integrated_post_var - eta * eta - var_mu

        # Ensure non-negative (numerical stability)
        mean_of_var = mean_of_var * (mean_of_var >= 0.0)

        self._cache['E_gamma_scaled'] = mean_of_var
        return mean_of_var

    def _variance_of_variance_scaled(self) -> Array:
        """Compute Var[γ_f] in scaled (internal) space.

        Var[γ_f] = ϑ₁ - 2ϑ₂ + ϑ₃ - E[γ_f]²

        Returns
        -------
        Array
            Scalar Var[γ]_scaled (non-negative).
        """
        if 'var_gamma_scaled' in self._cache:
            return self._cache['var_gamma_scaled']

        # Scale C-integrals to K-quantities
        s2 = self._get_kernel_variance()
        s4 = s2 * s2

        P_C = self._calc.P()
        P_K = s4 * P_C
        tau_K = self._calc.tau_K()        # s² τ_C
        u_K = s2 * self._calc.u()         # s² u_C
        nu_K = s4 * self._calc.nu()       # s⁴ ν_C
        Pi_K = s4 * s2 * self._calc.Pi()  # s⁶ Π_C
        xi1_K = s4 * self._calc.xi1()     # s⁴ ξ₁_C
        Gamma_K = s4 * self._calc.Gamma() # s⁴ Γ_C

        # === Prerequisite quantities (in scaled space) ===
        eta = self._mean_of_mean_scaled()
        var_mu = self._variance_of_mean_scaled()

        # A⁻¹y (already computed as alpha)
        alpha = self._gp.alpha()  # Shape: (nqoi, n_train)
        if alpha.shape[0] == 1:
            alpha_1d = self._bkd.reshape(alpha, (-1,))  # Shape: (n_train,)
        else:
            raise NotImplementedError(
                "variance_of_variance currently only supports single-output GPs (nqoi=1)"
            )

        # β_K = A⁻¹ τ_K
        tau_K_col = self._bkd.reshape(tau_K, (-1, 1))
        beta_K = self._bkd.reshape(self._solve(tau_K_col), (-1,))

        # ζ = αᵀ P_K α
        P_K_alpha = P_K @ alpha_1d
        zeta = alpha_1d @ P_K_alpha

        # v² = s² - tr[P_K A⁻¹]
        A_inv_P_K = self._solve(P_K)
        trace_P_K_A_inv = self._bkd.trace(A_inv_P_K)
        v_sq = s2 - trace_P_K_A_inv

        # === Intermediate quantities for ϑ₁ ===

        # φ = αᵀ Π_K α - αᵀ P_K A⁻¹ P_K α
        Pi_K_alpha = Pi_K @ alpha_1d
        phi_term1 = alpha_1d @ Pi_K_alpha
        P_K_A_inv_P_K_alpha = P_K @ (A_inv_P_K @ alpha_1d)
        phi_term2 = alpha_1d @ P_K_A_inv_P_K_alpha
        phi = phi_term1 - phi_term2

        # φ̃ = sum(P_K * (A⁻¹ P_K A⁻¹))
        A_inv = self._solve(self._bkd.eye(self._n_train))
        A_inv_P_K_A_inv = A_inv_P_K @ A_inv
        varphi = self._bkd.sum(P_K * A_inv_P_K_A_inv)

        # ψ = tr[A⁻¹ Π_K]
        A_inv_Pi_K = self._solve(Pi_K)
        psi = self._bkd.trace(A_inv_Pi_K)

        # χ = ν_K + φ̃ - 2ψ
        chi = nu_K + varphi - 2 * psi

        # ϑ₁ = 4φ + 2χ + (ζ + v²)²
        vartheta1 = 4 * phi + 2 * chi + (zeta + v_sq) ** 2

        # === ϑ₂ ===
        E_kappa = zeta + v_sq
        E_mu_sq = eta * eta + var_mu

        # Term_B = 4η · (αᵀ Γ_K - αᵀ P_K β_K)
        term_B = 4 * eta * (alpha_1d @ Gamma_K - alpha_1d @ (P_K @ beta_K))

        # Term_C = 2 · (ξ₁_K - 2 β_Kᵀ Γ_K + β_Kᵀ P_K β_K)
        term_C = 2 * (xi1_K - 2 * (beta_K @ Gamma_K) + (beta_K @ (P_K @ beta_K)))

        vartheta2 = E_kappa * E_mu_sq + term_B + term_C

        # ϑ₃ = η⁴ + 6η² var_mu + 3 var_mu²
        vartheta3 = eta ** 4 + 6 * eta * eta * var_mu + 3 * var_mu * var_mu

        # === Final result ===
        E_gamma = self._mean_of_variance_scaled()

        # Var[γ_f] = ϑ₁ - 2ϑ₂ + ϑ₃ - E[γ_f]²
        var_of_var = vartheta1 - 2 * vartheta2 + vartheta3 - E_gamma ** 2

        # Ensure non-negative (numerical stability)
        var_of_var = var_of_var * (var_of_var >= 0.0)

        self._cache['var_gamma_scaled'] = var_of_var
        return var_of_var

    # ===== Public methods: apply output transform =====

    def mean_of_mean(self) -> Array:
        """
        Compute the expected value of the GP posterior mean.

        eta = E[mu(X)] = tau^T A^{-1} y

        This is the mean prediction integrated over the input space.
        Returns values in the original output space if an output
        transform is set on the GP.

        Returns
        -------
        Array
            Scalar or shape (nqoi,) for multi-output GPs.
        """
        eta_scaled = self._mean_of_mean_scaled()

        transform = self._get_output_transform()
        if transform is None:
            return eta_scaled

        # η_orig = σ_y * η_scaled + μ_y
        sigma_y = transform.scale()
        mu_y = transform.shift()
        return sigma_y[0] * eta_scaled + mu_y[0]

    def variance_of_mean(self) -> Array:
        """
        Compute the variance of the GP posterior mean.

        Var[mu(X)] = s^2 * varsigma^2
        where varsigma^2 = u - tau^T A^{-1} tau

        This quantifies uncertainty in the mean prediction due to limited data.
        Returns values in the original output space if an output
        transform is set on the GP.

        Returns
        -------
        Array
            Scalar (non-negative).
        """
        var_mu_scaled = self._variance_of_mean_scaled()

        transform = self._get_output_transform()
        if transform is None:
            return var_mu_scaled

        # Var[μ]_orig = σ_y² * Var[μ]_scaled
        sigma_y_sq = transform.scale()[0] ** 2
        return sigma_y_sq * var_mu_scaled

    def mean_of_variance(self) -> Array:
        """
        Compute the expected value of the GP posterior variance.

        E[gamma(X)] = zeta + s^2 * v^2 - eta^2 - s^2 * varsigma^2

        This quantifies the expected prediction uncertainty over the input space.
        Returns values in the original output space if an output
        transform is set on the GP.

        Returns
        -------
        Array
            Scalar (non-negative).
        """
        E_gamma_scaled = self._mean_of_variance_scaled()

        transform = self._get_output_transform()
        if transform is None:
            return E_gamma_scaled

        # E[γ]_orig = σ_y² * E[γ]_scaled
        sigma_y_sq = transform.scale()[0] ** 2
        return sigma_y_sq * E_gamma_scaled

    def variance_of_variance(self) -> Array:
        """
        Compute the variance of the GP posterior variance.

        Var[γ_f] = ϑ₁ - 2ϑ₂ + ϑ₃ - E[γ_f]²

        This quantifies the uncertainty in the prediction variance.
        Returns values in the original output space if an output
        transform is set on the GP.

        Math Reference: docs/plans/gp_integration/02_2_2_variance_of_variance.qmd

        Returns
        -------
        Array
            Scalar (non-negative).
        """
        var_gamma_scaled = self._variance_of_variance_scaled()

        transform = self._get_output_transform()
        if transform is None:
            return var_gamma_scaled

        # Var[γ]_orig = σ_y⁴ * Var[γ]_scaled
        sigma_y_4 = transform.scale()[0] ** 4
        return sigma_y_4 * var_gamma_scaled
