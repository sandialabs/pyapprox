import unittest
import numpy as np
from scipy import stats
import scipy.special as sp

from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_three_term_recurrence_polynomial_1d,
    convert_orthonormal_polynomials_to_monomials_1d,
    convert_orthonormal_expansion_to_monomial_expansion_1d,
)
from pyapprox.variables.marginals import float_rv_discrete
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.orthopoly import (
    LegendrePolynomial1D,
    JacobiPolynomial1D,
    HermitePolynomial1D,
    KrawtchoukPolynomial1D,
    HahnPolynomial1D,
    DiscreteChebyshevPolynomial1D,
    CharlierPolynomial1D,
    LaguerrePolynomial1D,
    Chebyshev1stKindPolynomial1D,
    Chebyshev2ndKindPolynomial1D,
    ContinuousNumericOrthonormalPolynomial1D,
    DiscreteNumericOrthonormalPolynomial1D,
    GaussQuadratureRule,
)
from pyapprox.surrogates.bases.univariate import (
    Monomial1D, UnivariateLagrangeBasis
)
from pyapprox.surrogates.bases.basis import MultiIndexBasis
from pyapprox.surrogates.bases.basisexp import MonomialExpansion


class TestOrthonormalPolynomials1D:
    def setUp(self):
        np.random.seed(1)

    def get_backend(self):
        return NumpyLinAlgMixin()

    def _check_orthonormal_poly(self, poly):
        bkd = self.get_backend()
        degree = 3
        poly.set_nterms(degree + 1)
        quad_x, quad_w = poly.gauss_quadrature_rule(degree + 1)
        vals = poly(quad_x)
        # test orthogonality
        exact_moments = bkd._la_full((degree + 1, 1), 0.0)
        exact_moments[0] = 1.0
        assert bkd._la_allclose(vals.T @ quad_w, exact_moments)
        # test orthonormality
        assert bkd._la_allclose(
            (vals.T * quad_w[:, 0]) @ vals, bkd._la_eye(degree + 1)
        )

    def test_chebyshev_poly_1d(self):
        degree = 4
        bkd = self.get_backend()
        poly = Chebyshev1stKindPolynomial1D(backend=bkd)
        poly.set_nterms(degree + 1)
        # polys are not orhtonormal only orthogonal so cannot use:
        # self._check_orthonormal_poly(poly)
        quad_x, quad_w = sp.roots_chebyt(degree+1, mu=False)
        quad_x = bkd._la_array(quad_x)
        quad_w = bkd._la_array(quad_w)
        pquad_x, pquad_w = poly.gauss_quadrature_rule(degree + 1)
        assert bkd._la_allclose(pquad_x, quad_x)
        assert bkd._la_allclose(pquad_w, quad_w)
        vals_exact = bkd._la_stack(
            [sp.eval_chebyt(ii, quad_x) for ii in range(degree+1)], axis=1
        )
        vals = poly(pquad_x)
        assert bkd._la_allclose(vals, vals_exact)

        poly = Chebyshev2ndKindPolynomial1D(backend=bkd)
        poly.set_nterms(degree + 1)
        # polys are not orhtonormal only orthogonal so cannot use:
        # self._check_orthonormal_poly(poly)
        quad_x, quad_w = sp.roots_chebyu(degree+1, mu=False)
        quad_x = bkd._la_array(quad_x)[None, :]
        quad_w = bkd._la_array(quad_w)[:, None]
        pquad_x, pquad_w = poly.gauss_quadrature_rule(degree + 1)
        assert bkd._la_allclose(pquad_x, quad_x)
        assert bkd._la_allclose(pquad_w, quad_w)
        vals_exact = bkd._la_stack(
            [sp.eval_chebyu(ii, quad_x[0]) for ii in range(degree+1)], axis=1
        )
        vals = poly(pquad_x)
        assert bkd._la_allclose(vals, vals_exact)

    def test_legendre_poly_1d(self):
        bkd = self.get_backend()
        poly = LegendrePolynomial1D(backend=bkd)
        self._check_orthonormal_poly(poly)

        # check quadrature rule is computed correctly
        degree = 3
        quad_x, quad_w = np.polynomial.legendre.leggauss(degree + 1)
        # make weights have probablity weight function w=1/2
        quad_x = bkd._la_array(quad_x)[None, :]
        quad_w = bkd._la_array(quad_w / 2.0)[:, None]
        pquad_x, pquad_w = poly.gauss_quadrature_rule(degree + 1)
        assert bkd._la_allclose(pquad_x, quad_x)
        assert bkd._la_allclose(pquad_w, quad_w)

        deriv_order = 2
        derivs = poly._derivatives(quad_x, deriv_order, return_all=True)
        # squeeze when computing derivaties exactly
        x = quad_x[0]
        derivs_exact = [
            bkd._la_stack(
                [
                    1 + 0.0 * x,
                    x,
                    0.5 * (3.0 * x**2 - 1),
                    0.5 * (5.0 * x**3 - 3.0 * x),
                ],
                axis=1,
            )
        ]
        derivs_exact.append(
            bkd._la_stack(
                [0.0 * x, 1.0 + 0.0 * x, 3.0 * x, 7.5 * x**2 - 1.5], axis=1
            )
        )
        derivs_exact.append(
            bkd._la_stack([0.0 * x, 0.0 * x, 3.0 + 0.0 * x, 15 * x], axis=1)
        )
        derivs_exact = bkd._la_stack(derivs_exact, axis=0) / bkd._la_sqrt(
            1.0 / (2 * bkd._la_arange(degree + 1) + 1)
        )
        for ii in range(deriv_order + 1):
            assert bkd._la_allclose(
                derivs[:, ii * (degree + 1) : (ii + 1) * (degree + 1)],
                derivs_exact[ii],
            )

    def test_continuous_askey_polys(self):
        bkd = self.get_backend()
        polys = [
            JacobiPolynomial1D(4, 1, backend=bkd),
            HermitePolynomial1D(backend=bkd),
            LaguerrePolynomial1D(2, backend=bkd),
        ]
        for poly in polys:
            self._check_orthonormal_poly(poly)

        # check hermite quadrature rule
        degree = 4
        quad_x, quad_w = np.polynomial.hermite.hermgauss(degree + 1)
        # transform rule to probablity weight
        # function w=1/sqrt(2*PI)exp(-x^2/2)
        quad_x = bkd._la_array(quad_x * np.sqrt(2.0))[None, :]
        quad_w = bkd._la_array(quad_w / np.sqrt(np.pi))[:, None]
        poly = HermitePolynomial1D(backend=bkd)
        poly.set_nterms(degree + 1)
        pquad_x, pquad_w = poly.gauss_quadrature_rule(degree + 1)
        assert bkd._la_allclose(pquad_x, quad_x)
        assert bkd._la_allclose(pquad_w, quad_w)
        assert bkd._la_allclose(
            (pquad_x**degree) @ pquad_w,
            bkd._la_array(sp.factorial2(degree - 1)),
        )

        # check hermite evaluation
        degree = 2
        poly.set_nterms(degree + 1)
        vals = poly(quad_x)
        # squeeze when computing values exactly
        x = quad_x[0]
        vals_exact = bkd._la_stack(
            [1 + 0.0 * x, x, x**2 - 1], axis=1
        ) / bkd._la_sqrt(bkd._la_asarray(sp.factorial(np.arange(degree + 1))))
        assert bkd._la_allclose(vals, vals_exact)

        # check jacobi quadrature rule
        degree = 4
        poly = JacobiPolynomial1D(4, 1, backend=bkd)
        poly.set_nterms(degree + 1)
        pquad_x, pquad_w = poly.gauss_quadrature_rule(degree + 1)
        true_moments = bkd._la_array(
            [1.0, -3.0 / 7.0, 2.0 / 7.0, -4.0 / 21.0, 1.0 / 7.0]
        )
        for ii in range(degree + 1):
            assert bkd._la_allclose(
                (pquad_x**ii) @ pquad_w, true_moments[ii]
            )

        degree = 2
        rho = 2
        poly = LaguerrePolynomial1D(rho, backend=bkd)
        poly.set_nterms(degree + 1)
        np_x = stats.gamma(rho + 1).rvs(int(1e3))
        x = bkd._la_array(np_x)
        vals = poly(x[None, :])

        vals_exact = bkd._la_stack(
            [
                1 + 0.0 * x,
                -(-x + rho + 1),
                0.5 * x**2 - x * (rho + 2) + (rho + 1) * (rho + 2) / 2,
            ],
            axis=1,
        )[:, : degree + 1]
        vals_exact /= bkd._la_sqrt(
            sp.gamma(bkd._la_arange(degree + 1) + rho + 1)
            / sp.factorial(bkd._la_arange(degree + 1))
            / sp.gamma(rho + 1)
        )
        assert bkd._la_allclose(vals, vals_exact)

    def test_physicists_hermite_polynomial(self):
        bkd = self.get_backend()
        degree = 2
        poly = HermitePolynomial1D(prob_meas=False, backend=bkd)
        poly.set_nterms(degree + 1)
        quad_x, quad_w = np.polynomial.hermite.hermgauss(degree + 1)
        quad_x = bkd._la_array(quad_x)[None, :]
        quad_w = bkd._la_array(quad_w)[:, None]
        pquad_x, pquad_w = poly.gauss_quadrature_rule(degree + 1)
        assert bkd._la_allclose(pquad_x, quad_x)
        assert bkd._la_allclose(pquad_w, quad_w)

        # squeeze when computing values exactly
        x = quad_x[0]
        vals = poly(quad_x)
        vals_exact = bkd._la_stack(
            [1 + 0.0 * x, 2 * x, 4.0 * x**2 - 2], axis=1
        ) / bkd._la_sqrt(bkd._la_asarray(
            sp.factorial(np.arange(degree + 1))
            * np.sqrt(np.pi)
            * 2 ** np.arange(degree + 1))
        )
        assert bkd._la_allclose(vals, vals_exact)

        # test orthogonality
        exact_moments = bkd._la_full((degree + 1, 1), 0.0)
        # basis is orthonormal so integration of constant basis will be
        # non-zero but will not integrate to 1.0
        exact_moments[0] = np.pi**0.25
        assert bkd._la_allclose(vals.T @ quad_w, exact_moments)
        # test orthonormality
        assert bkd._la_allclose(
            (vals.T * quad_w[:, 0]) @ vals, bkd._la_eye(degree + 1)
        )

    def test_krawtchouk_binomial(self):
        bkd = self.get_backend()
        degree = 4
        num_trials = 10
        prob_success = 0.5
        poly = KrawtchoukPolynomial1D(
            num_trials, prob_success, raisewarn=False, backend=bkd
        )
        poly.set_nterms(degree + 1)
        quad_x, quad_w = poly.gauss_quadrature_rule(degree + 1)

        probability_mesh = bkd._la_arange(
            0, num_trials + 1, dtype=float)[None, :]
        probability_masses = bkd._la_array(
            stats.binom.pmf(probability_mesh[0], num_trials, prob_success)
        )[:, None]

        basis_mat = poly(probability_mesh)
        assert bkd._la_allclose(
            (basis_mat * probability_masses).T @ basis_mat,
            bkd._la_eye(basis_mat.shape[1]),
        )

        coef = bkd._la_array(np.random.uniform(-1, 1, (degree + 1)))
        monomial = Monomial1D(backend=bkd)
        monomial.set_nterms(degree + 1)
        basis_matrix_at_pm = monomial(probability_mesh)
        vals_at_pm = basis_matrix_at_pm @ coef
        basis_matrix_at_gauss = monomial(quad_x)
        vals_at_gauss = basis_matrix_at_gauss @ coef

        true_mean = vals_at_pm @ probability_masses
        quadrature_mean = vals_at_gauss @ quad_w
        assert bkd._la_allclose(true_mean, quadrature_mean)

    def test_hahn_hypergeometric(self):
        """
        Given 20 animals, of which 7 are dogs. Then hypergeometric PDF gives
        the probability of finding a given number of dogs if we choose at
        random 12 of the 20 animals.
        """
        bkd = self.get_backend()
        degree = 4
        M, n, N = 20, 7, 12
        apoly, bpoly = -(n + 1), -M - 1 + n
        poly = HahnPolynomial1D(N, apoly, bpoly, raisewarn=False, backend=bkd)
        poly.set_nterms(degree + 1)
        quad_x, quad_w = poly.gauss_quadrature_rule(degree + 1)

        rv = stats.hypergeom(M, n, N)
        true_mean = bkd._la_array(rv.mean())
        quadrature_mean = quad_x @ quad_w
        assert bkd._la_allclose(true_mean, quadrature_mean)

        quad_x = bkd._la_arange(0, n + 1)[None, :]
        vals = poly(quad_x)
        quad_w = rv.pmf(quad_x[0])[:, None]
        assert bkd._la_allclose(
            (vals.T * quad_w[:, 0]) @ vals, bkd._la_eye(degree + 1)
        )

    def test_discrete_chebyshev(self):
        bkd = self.get_backend()
        N, degree = 100, 5
        np_xk, np_pk = np.arange(N), np.ones(N) / N
        rv = float_rv_discrete(
            name="discrete_chebyshev", values=(np_xk, np_pk)
        )
        poly = DiscreteChebyshevPolynomial1D(N, backend=bkd)
        poly.set_nterms(degree + 1)
        vals = poly(bkd._la_array(np_xk)[None, :])
        quad_w = rv.pmf(np_xk)[:, None]
        assert bkd._la_allclose(
            (vals.T * quad_w[:, 0]) @ vals, bkd._la_eye(degree + 1)
        )

    def test_charlier(self):
        # Note as rate gets smaller the number of terms that can be accurately
        # computed will decrease because the problem gets more ill conditioned.
        # This is caused because the number of masses with significant weights
        # gets smaller as rate does
        bkd = self.get_backend()
        degree, rate = 5, 2
        rv = stats.poisson(rate)
        poly = CharlierPolynomial1D(rate, backend=bkd)
        poly.set_nterms(degree + 1)
        lb, ub = rv.interval(1 - np.finfo(float).eps)
        np_xk = np.linspace(lb, ub, int(ub - lb + 1))
        vals = poly(bkd._la_array(np_xk)[None, :])
        quad_w = bkd._la_array(rv.pmf(np_xk))[:, None]
        assert bkd._la_allclose(
            bkd._la_dot(vals.T * quad_w[:, 0], vals),
            bkd._la_eye(degree + 1),
            atol=1e-7,
        )

    def test_convert_orthonormal_recurence_to_three_term_recurence(self):
        bkd = self.get_backend()
        degree = 2
        poly = HermitePolynomial1D(backend=bkd)
        poly.set_nterms(degree + 1)
        abc = poly._three_term_recurence()

        x = bkd._la_linspace(-3, 3, 101)[None, :]
        p_2term = poly(x)
        p_3term = evaluate_three_term_recurrence_polynomial_1d(
            abc, degree, x[0], bkd=bkd
        )
        assert bkd._la_allclose(p_2term, p_3term)

    def test_convert_orthonormal_polynomials_to_monomials_1d(self):
        """
        Example: orthonormal Hermite polynomials
        deg  monomial coeffs
        0    [1,0,0]
        1    [0,1,0]         1/1*((x-0)*1-1*0)=x
        2    [1/c,0,1/c]     1/c*((x-0)*x-1*1)=(x**2-1)/c,            c=sqrt(2)
        3    [0,-3/d,0,1/d]  1/d*((x-0)*(x**2-1)/c-c*x)=
                             1/(c*d)*(x**3-x-c**2*x)=(x**3-3*x)/(c*d),d=sqrt(3)
        """
        bkd = self.get_backend()
        degree = 10
        poly = HermitePolynomial1D(backend=bkd)
        poly.set_nterms(degree + 1)
        basis_mono_coefs = convert_orthonormal_polynomials_to_monomials_1d(
            poly._rcoefs, 4, bkd=bkd
        )

        true_basis_mono_coefs = bkd._la_zeros((5, 5))
        true_basis_mono_coefs[0, 0] = 1
        true_basis_mono_coefs[1, 1] = 1
        true_basis_mono_coefs[2, [0, 2]] = bkd._la_array(
            [-1 / np.sqrt(2), 1 / np.sqrt(2)]
        )
        true_basis_mono_coefs[3, [1, 3]] = bkd._la_array(
            [-3 / np.sqrt(6), 1 / np.sqrt(6)]
        )
        true_basis_mono_coefs[4, [0, 2, 4]] = bkd._la_array(
            [3, -6, 1]
        ) / np.sqrt(24)

        assert bkd._la_allclose(basis_mono_coefs, true_basis_mono_coefs)

        coefs = bkd._la_ones(degree + 1)
        basis_mono_coefs = convert_orthonormal_polynomials_to_monomials_1d(
            poly._rcoefs, degree, bkd=bkd
        )
        mono_coefs = bkd._la_sum(basis_mono_coefs * coefs, axis=0)[:, None]

        x = bkd._la_linspace(-3, 3, 5)[None, :]
        p_ortho = poly(x)
        ortho_vals = p_ortho @ coefs

        basis = MultiIndexBasis([Monomial1D(backend=bkd)])
        basis.set_indices(bkd._la_arange(degree + 1)[None, :])
        mono = MonomialExpansion(basis, nqoi=mono_coefs.shape[1])
        mono.set_coefficients(mono_coefs)
        mono_vals = mono(x)[:, 0]
        assert bkd._la_allclose(ortho_vals, mono_vals)

    def test_convert_monomials_to_orthonormal_polynomials_1d(self):
        bkd = self.get_backend()
        degree = 10
        poly = HermitePolynomial1D(backend=bkd)
        poly.set_nterms(degree + 1)
        basis_mono_coefs = convert_orthonormal_polynomials_to_monomials_1d(
            poly._rcoefs, degree, bkd=bkd
        )

        x = bkd._la_array(np.random.normal(0, 1, (1, 100)))
        basis_ortho_coefs = bkd._la_inv(basis_mono_coefs)
        ortho_basis_matrix = poly(x)
        mono_basis_matrix = x.T ** bkd._la_arange(degree + 1)[None, :]
        assert bkd._la_allclose(
            mono_basis_matrix, ortho_basis_matrix @ basis_ortho_coefs.T
        )

    def test_convert_orthonormal_expansion_to_monomial_expansion_1d(self):
        """
        Approximate function
        f1 = lambda x: ((x-mu)/sigma)**3 using hermite polynomials tailored for
        normal random variable with mean mu and variance sigma**2

        The function defined on canonical domain of the hermite polynomials,
        i.e. normal with mean zero and unit variance, is
        f2 = lambda x: x.T**3
        """
        bkd = self.get_backend()
        degree = 4
        mu, sigma = 1, 2
        ortho_coef = bkd._la_array([0, 3, 0, np.sqrt(6)])
        poly = HermitePolynomial1D(backend=bkd)
        poly.set_nterms(degree + 1)
        mono_coefs = convert_orthonormal_expansion_to_monomial_expansion_1d(
            ortho_coef, poly._rcoefs, mu, sigma, bkd=bkd
        )
        true_mono_coefs = (
            bkd._la_array([-(mu**3), 3 * mu**2, -3 * mu, 1]) / sigma**3
        )
        assert bkd._la_allclose(mono_coefs, true_mono_coefs)

    def test_continuous_numeric_orthonormal_polynomial(self):
        bkd = self.get_backend()
        a = 3
        rho = a-1
        degree = 5
        poly = LaguerrePolynomial1D(rho)
        poly.set_nterms(degree+1)
        marginal = stats.gamma(a)

        tol = 1e-8
        quad_opts = {"epsrel": tol, "epsabs": tol, "limit": 100}
        num_poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal, quad_opts
        )
        num_poly.set_nterms(degree+1)
        # The last coefficient of ab[:, 0] is never used so it is not computed
        # by the numerical routine
        assert bkd._la_allclose(num_poly._rcoefs, poly._rcoefs)

        rho = 0
        marginal = stats.expon()
        poly = LaguerrePolynomial1D(rho)
        poly.set_nterms(degree+1)
        num_poly = ContinuousNumericOrthonormalPolynomial1D(
            marginal, quad_opts
        )
        num_poly.set_nterms(degree+1)
        assert bkd._la_allclose(num_poly._rcoefs, poly._rcoefs)

    def test_discrete_numeric_orthonormal_polynomial(self):
        bkd = self.get_backend()
        degree, rate = 5, 2
        rv = stats.poisson(rate)
        lb, ub = rv.interval(1 - np.finfo(float).eps)
        nmasses = int(ub - lb + 1)
        np_xk = np.linspace(lb, ub, nmasses)
        xk = bkd._la_asarray(np_xk)[None, :]
        pk = bkd._la_array(rv.pmf(np_xk))[:, None]

        poly = CharlierPolynomial1D(rate, backend=bkd)
        poly.set_nterms(degree + 1)

        tol = 1e-8
        num_poly = DiscreteNumericOrthonormalPolynomial1D(
            xk[0, :], pk[:, 0], tol,
        )
        num_poly.set_nterms(degree+1)
        # The last coefficient of ab[-1, 0] is never used so it is not computed
        # by the numerical routine
        assert bkd._la_allclose(num_poly._rcoefs[:-1, :], poly._rcoefs[:-1, :])

    def test_univariate_lagrange_basis_gauss_quadrature(self):
        nterms = 5
        bkd = self.get_backend()
        marginal = stats.uniform(-1, 2)
        basis = UnivariateLagrangeBasis(
            GaussQuadratureRule(marginal, backend=bkd), nterms)

        def fun(degree, xx):
            return bkd._la_sum(xx**degree, axis=0)[:, None]

        def integral(degree):
            if degree == 2:
                return 1 / 3
            if degree == 3:
                return 0
            if degree == 4:
                return 1 / 5

        samples, weights = basis.quadrature_rule()
        degree = nterms-1
        assert np.allclose(
            fun(degree, samples).T @ weights, integral(degree)
        )


class TestNumpyOrthonormalPolynomials1D(
    TestOrthonormalPolynomials1D, unittest.TestCase
):
    def get_backend(self):
        return NumpyLinAlgMixin()


class TestTorchOrthonormalPolynomials1D(
    TestOrthonormalPolynomials1D, unittest.TestCase
):
    def get_backend(self):
        return TorchLinAlgMixin()


if __name__ == "__main__":
    unittest.main(verbosity=2)
