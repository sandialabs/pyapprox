import unittest
import numpy as np
from scipy import stats
import scipy.special as sp

from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_1d,
    evaluate_three_term_recurrence_polynomial_1d,
    convert_orthonormal_polynomials_to_monomials_1d,
    convert_orthonormal_expansion_to_monomial_expansion_1d)
from pyapprox.surrogates.interp.monomial import (
    univariate_monomial_basis_matrix, evaluate_monomial)
from pyapprox.variables.marginals import float_rv_discrete
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.orthopoly.poly import (
    LegendrePolynomial1D, JacobiPolynomial1D, HermitePolynomial1D,
    KrawtchoukPolynomial1D, HahnPolynomial1D, DiscreteChebyshevPolynomial1D,
    CharlierPolynomial1D, LaguerrePolynomial1D)


class TestOrthonormalPolynomials1D(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def get_backend(self):
        return NumpyLinAlgMixin()

    def _check_orthonormal_poly(self, poly):
        bkd = self.get_backend()
        degree = 3
        poly.set_recursion_coefficients(degree+1)
        quad_x, quad_w = poly.gauss_quadrature_rule(degree+1)
        vals = poly(quad_x, degree)
        # test orthogonality
        exact_moments = bkd._la_full((degree+1,), 0.)
        exact_moments[0] = 1.0
        assert bkd._la_allclose(vals.T @ quad_w, exact_moments)
        # test orthonormality
        assert bkd._la_allclose((vals.T*quad_w) @ vals, bkd._la_eye(degree+1))

    def test_legendre_poly_1d(self):
        bkd = self.get_backend()
        poly = LegendrePolynomial1D(backend=bkd)
        self._check_orthonormal_poly(poly)

        # check quadrature rule is computed correctly
        degree = 3
        quad_x, quad_w = np.polynomial.legendre.leggauss(degree+1)
        # make weights have probablity weight function w=1/2
        quad_x = bkd._la_array(quad_x)
        quad_w = bkd._la_array(quad_w/2.0)
        pquad_x, pquad_w = poly.gauss_quadrature_rule(degree+1)
        assert bkd._la_allclose(pquad_x, quad_x)
        assert bkd._la_allclose(pquad_w, quad_w)

        x = quad_x
        deriv_order = 2
        derivs = poly.derivatives(x, degree, deriv_order, return_all=True)
        derivs_exact = [bkd._la_array(
            [1+0.*x, x, 0.5*(3.*x**2-1), 0.5*(5.*x**3-3.*x)]).T]
        derivs_exact.append(
            bkd._la_array([0.*x, 1.0+0.*x, 3.*x, 7.5*x**2-1.5]).T)
        derivs_exact.append(bkd._la_array([0.*x, 0.*x, 3.+0.*x, 15*x]).T)
        derivs_exact = bkd._la_array(derivs_exact)/bkd._la_sqrt(
            1./(2*bkd._la_arange(degree+1)+1))
        for ii in range(deriv_order+1):
            assert bkd._la_allclose(
                derivs[:, ii*(degree+1):(ii+1)*(degree+1)], derivs_exact[ii])

    def test_continuous_askey_polys(self):
        bkd = self.get_backend()
        polys = [JacobiPolynomial1D(4, 1, backend=bkd),
                 HermitePolynomial1D(backend=bkd),
                 LaguerrePolynomial1D(2, backend=bkd)]
        for poly in polys:
            self._check_orthonormal_poly(poly)

        # check hermite quadrature rule
        degree = 4
        quad_x, quad_w = np.polynomial.hermite.hermgauss(degree+1)
        # transform rule to probablity weight
        # function w=1/sqrt(2*PI)exp(-x^2/2)
        quad_x = bkd._la_array(quad_x*bkd._la_sqrt(2.0))
        quad_w = bkd._la_array(quad_w/bkd._la_sqrt(np.pi))
        poly = HermitePolynomial1D(backend=bkd)
        poly.set_recursion_coefficients(degree+1)
        pquad_x, pquad_w = poly.gauss_quadrature_rule(degree+1)
        assert bkd._la_allclose(pquad_x, quad_x)
        assert bkd._la_allclose(pquad_w, quad_w)
        assert bkd._la_allclose(
            (pquad_x**degree) @ pquad_w, sp.factorial2(degree-1))

        # check hermite evaluation
        degree = 2
        x = quad_x
        vals = poly(quad_x, degree)
        vals_exact = bkd._la_array(
            [1+0.*x, x, x**2-1]).T/bkd._la_sqrt(
                sp.factorial(bkd._la_arange(degree+1)))
        assert bkd._la_allclose(vals, vals_exact)

        # check jacobi quadrature rule
        degree = 4
        poly = JacobiPolynomial1D(4, 1, backend=bkd)
        poly.set_recursion_coefficients(degree+1)
        pquad_x, pquad_w = poly.gauss_quadrature_rule(degree+1)
        true_moments = [1., -3./7., 2./7., -4./21., 1./7.]
        for ii in range(degree+1):
            assert bkd._la_allclose(
                (pquad_x**ii) @ pquad_w, true_moments[ii])

        degree = 2
        rho = 2
        poly = LaguerrePolynomial1D(rho, backend=bkd)
        poly.set_recursion_coefficients(degree+1)
        x = stats.gamma(rho+1).rvs(int(1e3))
        vals = poly(x, degree)

        vals_exact = bkd._la_array(
            [1+0.*x, -(-x+rho+1),
             0.5*x**2-x*(rho+2)+(rho+1)*(rho+2)/2]).T[:, :degree+1]
        vals_exact /= bkd._la_sqrt(
            sp.gamma(bkd._la_arange(degree+1)+rho+1)/sp.factorial(
                bkd._la_arange(degree+1))/sp.gamma(rho+1))
        assert bkd._la_allclose(vals, vals_exact)

    def test_physicists_hermite_polynomial(self):
        bkd = self.get_backend()
        degree = 2
        poly = HermitePolynomial1D(prob_meas=False, backend=bkd)
        poly.set_recursion_coefficients(degree+1)
        quad_x, quad_w = np.polynomial.hermite.hermgauss(degree+1)
        quad_x = bkd._la_array(quad_x)
        quad_w = bkd._la_array(quad_w)
        pquad_x, pquad_w = poly.gauss_quadrature_rule(degree+1)
        assert bkd._la_allclose(pquad_x, quad_x)
        assert bkd._la_allclose(pquad_w, quad_w)

        x = quad_x
        vals = poly(x, degree)
        vals_exact = bkd._la_array([1+0.*x, 2*x, 4.*x**2-2]).T/bkd._la_sqrt(
            sp.factorial(bkd._la_arange(degree+1))*bkd._la_sqrt(
                np.pi)*2**bkd._la_arange(degree+1))
        assert bkd._la_allclose(vals, vals_exact)

        # test orthogonality
        exact_moments = bkd._la_full((degree+1), 0.)
        # basis is orthonormal so integration of constant basis will be
        # non-zero but will not integrate to 1.0
        exact_moments[0] = np.pi**0.25
        assert bkd._la_allclose(vals.T @ quad_w, exact_moments)
        # test orthonormality
        assert bkd._la_allclose((vals.T*quad_w) @ vals, bkd._la_eye(degree+1))

    def test_krawtchouk_binomial(self):
        bkd = self.get_backend()
        degree = 4
        num_trials = 10
        prob_success = 0.5
        poly = KrawtchoukPolynomial1D(num_trials, prob_success, backend=bkd)
        poly.set_recursion_coefficients(degree+1)
        quad_x, quad_w = poly.gauss_quadrature_rule(degree+1)

        probability_mesh = bkd._la_arange(0, num_trials+1, dtype=float)
        probability_masses = bkd._la_array(stats.binom.pmf(
            probability_mesh, num_trials, prob_success))

        basis_mat = poly(probability_mesh, degree)
        assert bkd._la_allclose(
            (basis_mat*probability_masses[:, None]).T @ basis_mat,
            bkd._la_eye(basis_mat.shape[1]))

        coef = bkd._la_array(np.random.uniform(-1, 1, (degree+1)))
        basis_matrix_at_pm = univariate_monomial_basis_matrix(
            degree, probability_mesh, bkd=bkd)
        vals_at_pm = basis_matrix_at_pm @ coef
        basis_matrix_at_gauss = univariate_monomial_basis_matrix(
            degree, quad_x, bkd=bkd)
        vals_at_gauss = basis_matrix_at_gauss @ coef

        true_mean = vals_at_pm @ probability_masses
        quadrature_mean = vals_at_gauss @ quad_w
        # print (true_mean, quadrature_mean)
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
        apoly, bpoly = -(n+1), -M-1+n
        poly = HahnPolynomial1D(N, apoly, bpoly, backend=bkd)
        poly.set_recursion_coefficients(degree+1)
        quad_x, quad_w = poly.gauss_quadrature_rule(degree+1)

        rv = stats.hypergeom(M, n, N)
        true_mean = rv.mean()
        quadrature_mean = quad_x @ quad_w
        assert bkd._la_allclose(true_mean, quadrature_mean)

        quad_x = bkd._la_arange(0, n+1)
        vals = poly(quad_x, degree)
        quad_w = rv.pmf(quad_x)
        assert bkd._la_allclose((vals.T*quad_w) @ vals, bkd._la_eye(degree+1))

    def test_discrete_chebyshev(self):
        bkd = self.get_backend()
        N, degree = 100, 5
        xk, pk = bkd._la_arange(N), bkd._la_ones(N)/N
        rv = float_rv_discrete(name='discrete_chebyshev', values=(xk, pk))
        poly = DiscreteChebyshevPolynomial1D(N, backend=bkd)
        poly.set_recursion_coefficients(degree+1)
        vals = poly(xk, degree)
        quad_w = rv.pmf(xk)
        assert bkd._la_allclose((vals.T*quad_w) @ vals, bkd._la_eye(degree+1))

    def test_charlier(self):
        # Note as rate gets smaller the number of terms that can be accurately
        # computed will decrease because the problem gets more ill conditioned.
        # This is caused because the number of masses with significant weights
        # gets smaller as rate does
        bkd = self.get_backend()
        degree, rate = 5, 2
        rv = stats.poisson(rate)
        poly = CharlierPolynomial1D(rate)
        poly.set_recursion_coefficients(degree+1)
        lb, ub = rv.interval(1-np.finfo(float).eps)
        xk = bkd._la_linspace(lb, ub, int(ub-lb+1))
        vals = poly(xk, degree)
        quad_w = rv.pmf(xk)
        assert bkd._la_allclose(
            bkd._la_dot(vals.T*quad_w, vals), bkd._la_eye(degree+1), atol=1e-7)

    def test_convert_orthonormal_recurence_to_three_term_recurence(self):
        bkd = self.get_backend()
        degree = 2
        poly = HermitePolynomial1D()
        poly.set_recursion_coefficients(degree+1)
        abc = poly._three_term_recurence()

        x = bkd._la_linspace(-3, 3, 101)
        p_2term = poly(x, degree)
        p_3term = evaluate_three_term_recurrence_polynomial_1d(
            abc, degree, x, bkd=bkd)
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
        poly.set_recursion_coefficients(degree+1)
        basis_mono_coefs = convert_orthonormal_polynomials_to_monomials_1d(
            poly._rcoefs, 4, bkd=bkd)

        true_basis_mono_coefs = bkd._la_zeros((5, 5))
        true_basis_mono_coefs[0, 0] = 1
        true_basis_mono_coefs[1, 1] = 1
        true_basis_mono_coefs[2, [0, 2]] = (
            -1/bkd._la_sqrt(2), 1/bkd._la_sqrt(2))
        true_basis_mono_coefs[3, [1, 3]] = (
            -3/bkd._la_sqrt(6), 1/bkd._la_sqrt(6))
        true_basis_mono_coefs[4, [0, 2, 4]] = bkd._la_array(
            [3, -6, 1])/bkd._la_sqrt(24)

        assert bkd._la_allclose(basis_mono_coefs, true_basis_mono_coefs)

        coefs = bkd._la_ones(degree+1)
        basis_mono_coefs = convert_orthonormal_polynomials_to_monomials_1d(
            poly._rcoefs, degree)
        mono_coefs = bkd._la_sum(basis_mono_coefs*coefs, axis=0)

        x = bkd._la_linspace(-3, 3, 5)
        p_ortho = poly(x, degree)
        ortho_vals = p_ortho @ coefs

        mono_vals = evaluate_monomial(
            bkd._la_arange(degree+1)[None, :], mono_coefs,
            x[None, :])[:, 0]
        assert bkd._la_allclose(ortho_vals, mono_vals)

    def test_convert_monomials_to_orthonormal_polynomials_1d(self):
        bkd = self.get_backend()
        degree = 10
        poly = HermitePolynomial1D(backend=bkd)
        poly.set_recursion_coefficients(degree+1)
        basis_mono_coefs = convert_orthonormal_polynomials_to_monomials_1d(
            poly._rcoefs, degree)

        x = bkd._la_array(np.random.normal(0, 1, (100)))
        print('Cond number', bkd._la_cond(basis_mono_coefs))
        basis_ortho_coefs = bkd._la_inv(basis_mono_coefs)
        ortho_basis_matrix = poly(x, degree)
        mono_basis_matrix = x[:, None]**bkd._la_arange(degree+1)[None, :]
        assert bkd._la_allclose(
            mono_basis_matrix, ortho_basis_matrix.dot(basis_ortho_coefs.T))

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
        ortho_coef = bkd._la_array([0, 3, 0, bkd._la_sqrt(6)])
        poly = HermitePolynomial1D(backend=bkd)
        poly.set_recursion_coefficients(degree+1)
        mono_coefs = convert_orthonormal_expansion_to_monomial_expansion_1d(
            ortho_coef, poly._rcoefs, mu, sigma)
        true_mono_coefs = bkd._la_array([-mu**3, 3*mu**2, -3*mu, 1])/sigma**3
        assert bkd._la_allclose(mono_coefs, true_mono_coefs)


if __name__ == "__main__":
    orthonormal_poly_1d_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestOrthonormalPolynomials1D)
    unittest.TextTestRunner(verbosity=2).run(orthonormal_poly_1d_test_suite)
