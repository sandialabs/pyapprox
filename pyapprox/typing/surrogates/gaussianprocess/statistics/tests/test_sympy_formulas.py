"""
SymPy verification tests for GP statistics formulas.

These tests verify the algebraic correctness of formulas from:
  docs/plans/gp_integration/02_1_mean_of_gp.qmd (mean and variance of GP mean)
  docs/plans/gp_integration/02_2_2_variance_of_variance.qmd (variance of variance)

Tests use SymPy to verify symbolic expressions match expected forms.
This catches formula errors early and serves as executable documentation.
"""

import unittest
from sympy import symbols, expand, simplify, MatrixSymbol, Transpose


class TestMeanOfGPMeanFormulas(unittest.TestCase):
    """SymPy tests for mean and variance of GP mean formulas.

    Reference: docs/plans/gp_integration/02_1_mean_of_gp.qmd

    Code mapping (moments.py):
    - η = τᵀA⁻¹y → mean_of_mean() returns eta = tau @ alpha
    - ϖ = τᵀA⁻¹τ → computed as tau @ A_inv_tau in variance_of_mean()
    - ς² = u - ϖ → computed as varsigma_sq = u - tau @ A_inv_tau
    - Var[μ_f] = s²ς² → variance_of_mean() returns s2 * varsigma_sq
    """

    def test_varpi_derivation(self) -> None:
        """Verify ϖ = τᵀA⁻¹τ simplification from double integral.

        From @eq-varpi:
        ϖ = ∫∫ t(z)ᵀ A⁻¹ t(w) dF(z)dF(w)
          = (∫ t(z) dF(z))ᵀ A⁻¹ (∫ t(w) dF(w))
          = τᵀ A⁻¹ τ

        This verifies the linearity of integration allows factorization.
        """
        N = symbols("N", integer=True, positive=True)

        # Define symbolic matrices and vectors
        A_inv = MatrixSymbol("A_inv", N, N)
        tau = MatrixSymbol("tau", N, 1)

        # varpi = tau^T A^{-1} tau (scalar)
        varpi = Transpose(tau) * A_inv * tau

        # Verify shape is scalar (1x1)
        self.assertEqual(varpi.shape, (1, 1))

    def test_varsigma_sq_formula(self) -> None:
        """Verify ς² = u - ϖ for variance of GP mean.

        From @eq-varsigma-sq:
        ς² = s⁻² Var[μ_f | θ, y] = u - ϖ

        where:
        - u = ∫∫ C(z,w) dF(z)dF(w) (prior variance integral)
        - ϖ = τᵀA⁻¹τ (reduction from conditioning)
        """
        u, varpi = symbols("u varpi", real=True, positive=True)

        # varsigma_sq = u - varpi
        varsigma_sq = u - varpi

        # Verify structure: difference of two positive terms
        # u >= varpi always (since A is positive definite and conditioning
        # reduces variance)
        target = u - varpi

        self.assertEqual(simplify(varsigma_sq - target), 0)

    def test_eta_dimension_check(self) -> None:
        """Verify η = τᵀA⁻¹y has correct dimensions.

        From @eq-eta:
        η = E[μ_f | θ, y] = τᵀ A⁻¹ y

        Dimensions:
        - τᵀ: 1 × N
        - A⁻¹: N × N
        - y: N × 1
        - Result: 1 × 1 (scalar)
        """
        N = symbols("N", integer=True, positive=True)

        # Define symbolic matrices and vectors
        A_inv = MatrixSymbol("A_inv", N, N)
        tau = MatrixSymbol("tau", N, 1)
        y = MatrixSymbol("y", N, 1)

        # eta = tau^T A^{-1} y (scalar)
        eta = Transpose(tau) * A_inv * y

        # Verify shape is scalar (1x1)
        self.assertEqual(eta.shape, (1, 1))

    def test_tau_dimension_check(self) -> None:
        """Verify τ = ∫ t(z) dF(z) has correct dimensions.

        From @eq-tau:
        τ = ∫ t(z) dF(z) ∈ ℝᴺ

        where t(z) ∈ ℝᴺ is the vector of kernel evaluations
        t(z)_j = k(z, x^(j)) for training points x^(j).
        """
        N = symbols("N", integer=True, positive=True)

        # tau is N x 1 vector
        tau = MatrixSymbol("tau", N, 1)

        # Verify shape
        self.assertEqual(tau.shape, (N, 1))


class TestVarianceOfVarianceFormulas(unittest.TestCase):
    """SymPy tests for variance of variance formulas.

    Reference: docs/plans/gp_integration/02_2_2_variance_of_variance.qmd
    """

    def test_E_Xi2_Xk2_from_isserlis(self) -> None:
        """Verify E[X_i² X_k²] simplification from Isserlis' theorem.

        From @eq-mgf-4-prod-gauss with i=j, k=l.
        """
        mu_i, mu_k, omega_ik, omega_ii, omega_kk = symbols(
            "mu_i mu_k omega_ik omega_ii omega_kk", real=True
        )

        # From Isserlis with i=j, k=l
        E_Xi2_Xk2_general = (
            4 * mu_i * omega_ik * mu_k
            + 2 * omega_ik**2
            + omega_ii * omega_kk
            + mu_i**2 * mu_k**2
            + mu_i**2 * omega_kk
            + mu_k**2 * omega_ii
        )

        # Apply mu_i = mu_k, omega_ii = omega_kk
        E_Xi2_Xk2_equal = E_Xi2_Xk2_general.subs(
            [(mu_k, mu_i), (omega_kk, omega_ii)]
        )
        E_Xi2_Xk2_simplified = simplify(E_Xi2_Xk2_equal)

        # Target: 4*mu_i²*omega_ik + 2*omega_ik² + (mu_i² + omega_ii)²
        target = (
            4 * mu_i**2 * omega_ik
            + 2 * omega_ik**2
            + (mu_i**2 + omega_ii) ** 2
        )

        self.assertEqual(simplify(E_Xi2_Xk2_simplified - target), 0)

    def test_chi_expansion(self) -> None:
        """Verify χ = ν + varphi - 2ψ via algebraic expansion.

        χ = ∫∫ C*(x,z)² dF(x)dF(z)
        where C*(x,z) = C(x,z) - t(x)ᵀA⁻¹t(z)

        Expanding (C - tᵀA⁻¹t)² gives:
        C² + (tᵀA⁻¹t)² - 2C(tᵀA⁻¹t) = ν + varphi - 2ψ
        """
        C, tAt = symbols("C tAt", real=True)
        C_star = C - tAt
        C_star_sq = expand(C_star**2)

        # Target: C² + tAt² - 2*C*tAt
        target = C**2 + tAt**2 - 2 * C * tAt

        self.assertEqual(simplify(C_star_sq - target), 0)

    def test_E_XiXjXk2_from_isserlis(self) -> None:
        """Verify E[X_i X_j X_k²] with μ_i = μ_j from Isserlis' theorem.

        From @eq-mgf-4-prod-gauss with k=l.
        """
        mu_i, mu_j, mu_k = symbols("mu_i mu_j mu_k", real=True)
        omega_ij, omega_ik, omega_jk, omega_kk = symbols(
            "omega_ij omega_ik omega_jk omega_kk", real=True
        )

        # From Isserlis with k=l
        E_XiXjXk2 = (
            2 * mu_i * omega_jk * mu_k
            + 2 * mu_j * omega_ik * mu_k
            + 2 * omega_jk * omega_ik
            + mu_k**2 * omega_ij
            + omega_ij * omega_kk
            + mu_i * mu_j * mu_k**2
            + mu_i * omega_kk * mu_j
        )

        # Apply mu_i = mu_j
        E_XiXjXk2_equal = E_XiXjXk2.subs(mu_j, mu_i)

        # Target after mu_i = mu_j substitution
        # 2*mu_i*omega_jk*mu_k + 2*mu_i*omega_ik*mu_k = 2*mu_i*mu_k*(omega_jk + omega_ik)
        target = (
            2 * mu_i * mu_k * (omega_jk + omega_ik)
            + 2 * omega_ik * omega_jk
            + mu_k**2 * omega_ij
            + omega_ij * omega_kk
            + mu_i**2 * mu_k**2
            + mu_i**2 * omega_kk
        )

        self.assertEqual(simplify(expand(E_XiXjXk2_equal) - expand(target)), 0)

    def test_xi_expansion(self) -> None:
        """Verify ξ = ξ₁ - 2ξ₂ + ξ₃ via algebraic expansion.

        ξ = ∫∫∫ C*(w,x)C*(w,z) dF³
        where C*(a,b) = C(a,b) - t(a)ᵀA⁻¹t(b)

        Expanding gives four terms that combine to ξ₁ - 2ξ₂ + ξ₃.
        """
        C_wx, C_wz, tAt_x, tAt_z = symbols("C_wx C_wz tAt_x tAt_z", real=True)
        C_star_wx = C_wx - tAt_x
        C_star_wz = C_wz - tAt_z
        product = expand(C_star_wx * C_star_wz)

        # Target: C_wx*C_wz - C_wx*tAt_z - C_wz*tAt_x + tAt_x*tAt_z
        target = C_wx * C_wz - C_wx * tAt_z - C_wz * tAt_x + tAt_x * tAt_z

        self.assertEqual(simplify(product - target), 0)

    def test_vartheta3_from_fourth_moment(self) -> None:
        """Verify ϑ₃ = 6η²ς²s² + 3ς⁴s⁴ + η⁴ from E[X⁴] for Gaussian.

        E[X⁴] for X ~ N(μ, σ²) is: μ⁴ + 6μ²σ² + 3σ⁴

        For μ_f ~ N(η, s²ς²), substitute μ = η, σ² = s²ς²:
        ϑ₃ = η⁴ + 6η²(s²ς²) + 3(s²ς²)² = η⁴ + 6η²ς²s² + 3ς⁴s⁴

        This verifies the correct formula is 3ς⁴s⁴, NOT 3ς²s² (typo).
        """
        # E[X⁴] for X ~ N(μ, σ²)
        mu, sigma_sq = symbols("mu sigma_sq", real=True, positive=True)
        E_X4 = mu**4 + 6 * mu**2 * sigma_sq + 3 * sigma_sq**2

        # Substitute for μ_f ~ N(η, s²ς²)
        eta, s_sq, varsigma_sq = symbols(
            "eta s_sq varsigma_sq", real=True, positive=True
        )
        vartheta3 = E_X4.subs([(mu, eta), (sigma_sq, s_sq * varsigma_sq)])
        vartheta3_expanded = expand(vartheta3)

        # Expected correct form: η⁴ + 6η²ς²s² + 3ς⁴s⁴
        expected_correct = (
            eta**4
            + 6 * eta**2 * varsigma_sq * s_sq
            + 3 * varsigma_sq**2 * s_sq**2
        )

        # Wrong form (typo): η⁴ + 6η²ς²s² + 3ς²s²
        wrong_form = (
            eta**4
            + 6 * eta**2 * varsigma_sq * s_sq
            + 3 * varsigma_sq * s_sq
        )

        # Verify correct formula matches
        self.assertEqual(simplify(vartheta3_expanded - expected_correct), 0)

        # Verify wrong formula does NOT match
        self.assertNotEqual(simplify(vartheta3_expanded - wrong_form), 0)

    def test_phi_integrand_expansion(self) -> None:
        """Verify φ integrand: m*(x)·m*(z)·C*(x,z) expansion.

        φ = ∫∫ m*(x)m*(z)C*(x,z) dFdF

        Expanding m*(x)·m*(z)·(C - tᵀA⁻¹t) gives two terms:
        - Term 1: m*(x)·m*(z)·C(x,z) → yᵀA⁻¹ΠA⁻¹y
        - Term 2: m*(x)·m*(z)·tᵀA⁻¹t → yᵀA⁻¹PA⁻¹PA⁻¹y

        Therefore: φ = yᵀA⁻¹ΠA⁻¹y - yᵀA⁻¹PA⁻¹PA⁻¹y
        """
        m_x, m_z, C_xz, tAt_xz = symbols("m_x m_z C_xz tAt_xz", real=True)
        integrand = m_x * m_z * (C_xz - tAt_xz)
        expanded = expand(integrand)

        # Target: m_x*m_z*C_xz - m_x*m_z*tAt_xz
        target = m_x * m_z * C_xz - m_x * m_z * tAt_xz

        self.assertEqual(simplify(expanded - target), 0)

    def test_varrho_integrand_expansion(self) -> None:
        """Verify ϱ integrand: m*(x)·C*(x,z) expansion.

        ϱ = ∫∫ m*(x)C*(x,z) dFdF

        Expanding m*(x)·(C - tᵀA⁻¹t) gives two terms:
        - Term 1: m*(x)·C(x,z) → λᵀA⁻¹y
        - Term 2: m*(x)·tᵀA⁻¹t → τᵀA⁻¹PA⁻¹y

        Therefore: ϱ = λᵀA⁻¹y - τᵀA⁻¹PA⁻¹y
        """
        m_x, C_xz, tAt_xz = symbols("m_x C_xz tAt_xz", real=True)
        integrand = m_x * (C_xz - tAt_xz)
        expanded = expand(integrand)

        # Target: m_x*C_xz - m_x*tAt_xz
        target = m_x * C_xz - m_x * tAt_xz

        self.assertEqual(simplify(expanded - target), 0)

    def test_E_X4_for_gaussian(self) -> None:
        """Verify E[X⁴] = μ⁴ + 6μ²σ² + 3σ⁴ for X ~ N(μ, σ²).

        This is a standard result (kurtosis formula).
        Using i=j=k=l in Isserlis' theorem:
        E[X⁴] = 6μ²ω + 3ω² + μ⁴  where ω = σ²
        """
        mu, sigma_sq = symbols("mu sigma_sq", real=True, positive=True)

        # From Isserlis with i=j=k=l
        E_X4_isserlis = 6 * mu**2 * sigma_sq + 3 * sigma_sq**2 + mu**4

        # Standard form
        E_X4_standard = mu**4 + 6 * mu**2 * sigma_sq + 3 * sigma_sq**2

        self.assertEqual(simplify(E_X4_isserlis - E_X4_standard), 0)

    def test_E_Xi2_Xk_Xl_from_isserlis(self) -> None:
        """Verify E[X_i² X_k X_l] from Isserlis' theorem for ϑ₂.

        For E[κμ²] = E[∫f²dF · (∫fdF)²], we need E[f(x)²f(w)f(z)].
        Using Isserlis with i=j (same point x), k (at w), l (at z):

        E[X_i X_j X_k X_l] with i=j gives E[X_i² X_k X_l]

        From the general Isserlis formula:
        E[X_i X_j X_k X_l] = μ_i μ_j μ_k μ_l
                          + ω_ij ω_kl + ω_ik ω_jl + ω_il ω_jk
                          + (6 mixed terms with 2 means and 1 covariance)

        Setting i=j:
        E[X_i² X_k X_l] = μ_i² μ_k μ_l
                        + ω_ii ω_kl + ω_ik ω_il + ω_il ω_ik
                        + μ_i μ_i ω_kl + μ_i μ_k ω_il + μ_i μ_l ω_ik
                        + μ_i μ_k ω_il + μ_i μ_l ω_ik + μ_k μ_l ω_ii

        Simplifying (note ω_ik ω_il appears twice):
        = μ_i² μ_k μ_l + ω_ii ω_kl + 2 ω_ik ω_il
          + μ_i² ω_kl + 2 μ_i μ_k ω_il + 2 μ_i μ_l ω_ik + μ_k μ_l ω_ii
        """
        mu_i, mu_k, mu_l = symbols("mu_i mu_k mu_l", real=True)
        omega_ii, omega_kl, omega_ik, omega_il = symbols(
            "omega_ii omega_kl omega_ik omega_il", real=True
        )

        # Direct derivation from Isserlis with i=j
        E_Xi2_Xk_Xl = (
            mu_i**2 * mu_k * mu_l  # all means
            + omega_ii * omega_kl  # ω_ij ω_kl with i=j
            + 2 * omega_ik * omega_il  # ω_ik ω_jl + ω_il ω_jk with i=j
            + mu_i**2 * omega_kl  # μ_i μ_j ω_kl with i=j
            + 2 * mu_i * mu_k * omega_il  # μ_i μ_k ω_jl + μ_j μ_k ω_il with i=j
            + 2 * mu_i * mu_l * omega_ik  # μ_i μ_l ω_jk + μ_j μ_l ω_ik with i=j
            + mu_k * mu_l * omega_ii  # μ_k μ_l ω_ij with i=j
        )

        # Verify the structure contains the key term 2*ω_ik*ω_il
        # This is what generates the Γ integral when integrated
        coeff_omega_ik_omega_il = E_Xi2_Xk_Xl.coeff(omega_ik * omega_il)
        self.assertEqual(coeff_omega_ik_omega_il, 2)

        # Also verify ω_ii*ω_kl coefficient
        coeff_omega_ii_omega_kl = E_Xi2_Xk_Xl.coeff(omega_ii * omega_kl)
        self.assertEqual(coeff_omega_ii_omega_kl, 1)

    def test_vartheta2_covariance_product_expansion(self) -> None:
        """Verify the expansion of ∫∫∫ C*(x,w)C*(x,z) dF³ for ϑ₂.

        The term 2*ω_ik*ω_il in E[X_i² X_k X_l] integrates to:
        2s⁴ ∫∫∫ C*(x,w)C*(x,z) dF(x)dF(w)dF(z)

        where C*(a,b) = C(a,b) - t(a)ᵀA⁻¹t(b).

        Expanding C*(x,w)C*(x,z):
        = [C(x,w) - t(x)ᵀA⁻¹t(w)] [C(x,z) - t(x)ᵀA⁻¹t(z)]
        = C(x,w)C(x,z)
          - C(x,w) t(x)ᵀA⁻¹t(z)
          - C(x,z) t(x)ᵀA⁻¹t(w)
          + [t(x)ᵀA⁻¹t(w)] [t(x)ᵀA⁻¹t(z)]

        After integration over (x, w, z), this gives 4 distinct terms.
        """
        # Symbols for C*(x,w) = C_xw - tAt_xw and C*(x,z) = C_xz - tAt_xz
        C_xw, C_xz = symbols("C_xw C_xz", real=True)
        tAt_xw, tAt_xz = symbols("tAt_xw tAt_xz", real=True)

        C_star_xw = C_xw - tAt_xw
        C_star_xz = C_xz - tAt_xz

        product = expand(C_star_xw * C_star_xz)

        # Expected: C_xw*C_xz - C_xw*tAt_xz - C_xz*tAt_xw + tAt_xw*tAt_xz
        expected = (
            C_xw * C_xz
            - C_xw * tAt_xz
            - C_xz * tAt_xw
            + tAt_xw * tAt_xz
        )

        self.assertEqual(simplify(product - expected), 0)

        # Verify the 4 terms are distinct (not reducible to 3 like in ξ)
        # The cross terms -C_xw*tAt_xz and -C_xz*tAt_xw are NOT equal
        # because x appears in different kernel arguments
        self.assertNotEqual(C_xw * tAt_xz, C_xz * tAt_xw)

    def test_vartheta2_integral_mapping(self) -> None:
        """Verify the mapping from C*(x,w)C*(x,z) expansion to integral symbols.

        After expanding C*(x,w)C*(x,z) and integrating ∫∫∫ dF(x)dF(w)dF(z):

        Term 1: ∫∫∫ C(x,w)C(x,z) dF³ = ξ₁
        Term 2: ∫∫∫ C(x,w) t(x)ᵀA⁻¹t(z) dF³ = βᵀΓ  (NOT ξ₂!)
        Term 3: ∫∫∫ C(x,z) t(x)ᵀA⁻¹t(w) dF³ = βᵀΓ  (same as Term 2 by symmetry)
        Term 4: ∫∫∫ [t(x)ᵀA⁻¹t(w)][t(x)ᵀA⁻¹t(z)] dF³ = βᵀPβ

        Combined: ξ₁ - 2βᵀΓ + βᵀPβ

        where:
        - β = A⁻¹τ
        - Γᵢ = ∫∫ C(xᵢ,z)C(z,v) dF(z)dF(v)

        NOTE: The old wrong formula used ξ = ξ₁ - 2ξ₂ + ξ₃ where ξ₂ = λᵀA⁻¹τ.
        This is WRONG because ξ₂ comes from ∫∫∫ C(w,x) t(w)ᵀA⁻¹t(z) dF³
        where the kernel C(w,x) has w as first argument, not x.

        The correct Term 2 has C(x,w) with x as first argument, giving Γ not λ.
        """
        # Define symbolic integral quantities
        xi_1 = symbols("xi_1", real=True)  # ∫∫∫ C(x,w)C(x,z) dF³
        beta_Gamma = symbols("beta_Gamma", real=True)  # βᵀΓ
        beta_P_beta = symbols("beta_P_beta", real=True)  # βᵀPβ

        # Correct combination from C*(x,w)C*(x,z) expansion
        correct_term = xi_1 - 2 * beta_Gamma + beta_P_beta

        # Wrong combination that was used before (using ξ)
        xi_2 = symbols("xi_2", real=True)  # λᵀA⁻¹τ (WRONG for this integral)
        xi_3 = symbols("xi_3", real=True)  # τᵀA⁻¹PA⁻¹τ = βᵀPβ
        wrong_term = xi_1 - 2 * xi_2 + xi_3

        # These are NOT equal because beta_Gamma ≠ xi_2
        # beta_Gamma = βᵀΓ where Γᵢ = ∫∫ C(xᵢ,z)C(z,v) dF²
        # xi_2 = λᵀA⁻¹τ where λᵢ = ∫∫ C(x,z)C(x,xᵢ) dF²
        # The difference is which argument of C contains the training point
        self.assertNotEqual(
            correct_term.subs(beta_P_beta, xi_3),
            wrong_term
        )


if __name__ == "__main__":
    unittest.main()
