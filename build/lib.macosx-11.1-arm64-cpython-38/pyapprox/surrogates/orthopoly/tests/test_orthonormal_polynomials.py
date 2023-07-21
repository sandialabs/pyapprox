import unittest
import numpy as np
from scipy.stats import binom, hypergeom, poisson
import scipy.special as sp

from pyapprox.surrogates.orthopoly.orthonormal_polynomials import (
    evaluate_orthonormal_polynomial_1d, gauss_quadrature,
    evaluate_orthonormal_polynomial_deriv_1d,
    evaluate_three_term_recurrence_polynomial_1d,
    convert_orthonormal_polynomials_to_monomials_1d,
    convert_orthonormal_expansion_to_monomial_expansion_1d
)
from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    jacobi_recurrence, hermite_recurrence, krawtchouk_recurrence,
    discrete_chebyshev_recurrence, hahn_recurrence, charlier_recurrence,
    laguerre_recurrence
)
from pyapprox.surrogates.orthopoly.orthonormal_recursions import (
    convert_orthonormal_recurence_to_three_term_recurence
)
from pyapprox.surrogates.interp.monomial import (
    univariate_monomial_basis_matrix, evaluate_monomial
)
from pyapprox.variables.marginals import float_rv_discrete


class TestOrthonormalPolynomials1D(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_orthonormality_legendre_polynomial(self):
        alpha = 0.
        beta = 0.
        degree = 3
        probability_measure = True

        ab = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=probability_measure)

        x, w = np.polynomial.legendre.leggauss(degree+1)
        # make weights have probablity weight function w=1/2
        w /= 2.0
        p = evaluate_orthonormal_polynomial_1d(x, degree, ab)
        # test orthogonality
        exact_moments = np.zeros((degree+1))
        exact_moments[0] = 1.0
        assert np.allclose(np.dot(p.T, w), exact_moments)
        # test orthonormality
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

        assert np.allclose(
            evaluate_orthonormal_polynomial_deriv_1d(x, degree, ab, 0), p)

    def test_orthonormality_asymetric_jacobi_polynomial(self):
        from scipy.stats import beta as beta_rv
        alpha = 4.
        beta = 1.
        degree = 3
        probability_measure = True

        ab = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=probability_measure)

        x, w = np.polynomial.legendre.leggauss(10*degree)
        p = evaluate_orthonormal_polynomial_1d(x, degree, ab)
        w *= beta_rv.pdf((x+1.)/2., a=beta+1, b=alpha+1)/2.

        # test orthogonality
        exact_moments = np.zeros((degree+1))
        exact_moments[0] = 1.0
        assert np.allclose(np.dot(p.T, w), exact_moments)
        # test orthonormality
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

        assert np.allclose(
            evaluate_orthonormal_polynomial_deriv_1d(x, degree, ab, 0), p)

    def test_derivatives_of_legendre_polynomial(self):
        alpha = 0.
        beta = 0.
        degree = 3
        probability_measure = True
        deriv_order = 2

        ab = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=probability_measure)
        x, w = np.polynomial.legendre.leggauss(degree+1)
        pd = evaluate_orthonormal_polynomial_deriv_1d(
            x, degree, ab, deriv_order)

        pd_exact = [np.asarray(
            [1+0.*x, x, 0.5*(3.*x**2-1), 0.5*(5.*x**3-3.*x)]).T]
        pd_exact.append(np.asarray([0.*x, 1.0+0.*x, 3.*x, 7.5*x**2-1.5]).T)
        pd_exact.append(np.asarray([0.*x, 0.*x, 3.+0.*x, 15*x]).T)
        pd_exact = np.asarray(pd_exact)/np.sqrt(1./(2*np.arange(degree+1)+1))
        for ii in range(deriv_order+1):
            assert np.allclose(
                pd[:, ii*(degree+1):(ii+1)*(degree+1)], pd_exact[ii])

    def test_orthonormality_physicists_hermite_polynomial(self):
        rho = 0.
        degree = 2
        probability_measure = False

        ab = hermite_recurrence(
            degree+1, rho, probability=probability_measure)
        x, w = np.polynomial.hermite.hermgauss(degree+1)

        p = evaluate_orthonormal_polynomial_1d(x, degree, ab)
        p_exact = np.asarray([1+0.*x, 2*x, 4.*x**2-2]).T/np.sqrt(
            sp.factorial(np.arange(degree+1))*np.sqrt(np.pi)*2**np.arange(
                degree+1))

        assert np.allclose(p, p_exact)

        # test orthogonality
        exact_moments = np.zeros((degree+1))
        # basis is orthonormal so integration of constant basis will be
        # non-zero but will not integrate to 1.0
        exact_moments[0] = np.pi**0.25
        assert np.allclose(np.dot(p.T, w), exact_moments)
        # test orthonormality
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

    def test_orthonormality_probabilists_hermite_polynomial(self):
        rho = 0.
        degree = 2
        probability_measure = True
        ab = hermite_recurrence(
            degree+1, rho, probability=probability_measure)

        x, w = np.polynomial.hermite.hermgauss(degree+1)
        # transform rule to probablity weight
        # function w=1/sqrt(2*PI)exp(-x^2/2)
        x *= np.sqrt(2.0)
        w /= np.sqrt(np.pi)
        p = evaluate_orthonormal_polynomial_1d(x, degree, ab)

        # Note if using pecos the following is done (i.e. ptFactpr=sqrt(2)),
        # but if I switch to using orthonormal recursion, used here, in Pecos
        # then I will need to set ptFactor=1.0 as done implicitly above
        p_exact = np.asarray(
            [1+0.*x, x, x**2-1]).T/np.sqrt(sp.factorial(np.arange(degree+1)))
        assert np.allclose(p, p_exact)

        # test orthogonality
        exact_moments = np.zeros((degree+1))
        exact_moments[0] = 1.0
        assert np.allclose(np.dot(p.T, w), exact_moments)
        # test orthonormality
        print(np.allclose(np.dot(p.T*w, p), np.eye(degree+1)))
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

    def test_gauss_quadrature(self):
        degree = 4
        alpha = 0.
        beta = 0.
        ab = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=True)

        x, w = gauss_quadrature(ab, degree+1)
        for ii in range(degree+1):
            if ii % 2 == 0:
                assert np.allclose(np.dot(x**ii, w), 1./(ii+1.))
            else:
                assert np.allclose(np.dot(x**ii, w), 0.)

        x_np, w_np = np.polynomial.legendre.leggauss(degree+1)
        assert np.allclose(x_np, x)
        assert np.allclose(w_np/2, w)

        degree = 4
        alpha = 4.
        beta = 1.
        ab = jacobi_recurrence(
            degree+1, alpha=alpha, beta=beta, probability=True)

        x, w = gauss_quadrature(ab, degree+1)

        true_moments = [1., -3./7., 2./7., -4./21., 1./7.]
        for ii in range(degree+1):
            assert np.allclose(np.dot(x**ii, w), true_moments[ii])

        degree = 4
        rho = 0.
        ab = hermite_recurrence(
            degree+1, rho, probability=True)
        x, w = gauss_quadrature(ab, degree+1)
        from scipy.special import factorial2
        assert np.allclose(np.dot(x**degree, w), factorial2(degree-1))

        x_sp, w_sp = sp.roots_hermitenorm(degree+1)
        w_sp /= np.sqrt(2*np.pi)
        assert np.allclose(x_sp, x)
        assert np.allclose(w_sp, w)

    def test_krawtchouk_binomial(self):
        degree = 4
        num_trials = 10
        prob_success = 0.5
        ab = krawtchouk_recurrence(
            degree+1, num_trials, prob_success)
        x, w = gauss_quadrature(ab, degree+1)

        probability_mesh = np.arange(0, num_trials+1, dtype=float)
        probability_masses = binom.pmf(
            probability_mesh, num_trials, prob_success)

        basis_mat = evaluate_orthonormal_polynomial_1d(
            probability_mesh, degree, ab)
        assert np.allclose(
            (basis_mat*probability_masses[:, None]).T.dot(basis_mat),
            np.eye(basis_mat.shape[1]))

        coef = np.random.uniform(-1, 1, (degree+1))
        basis_matrix_at_pm = univariate_monomial_basis_matrix(
            degree, probability_mesh)
        vals_at_pm = basis_matrix_at_pm.dot(coef)
        basis_matrix_at_gauss = univariate_monomial_basis_matrix(degree, x)
        vals_at_gauss = basis_matrix_at_gauss.dot(coef)

        true_mean = vals_at_pm.dot(probability_masses)
        quadrature_mean = vals_at_gauss.dot(w)
        # print (true_mean, quadrature_mean)
        assert np.allclose(true_mean, quadrature_mean)

    def test_hahn_hypergeometric(self):
        """
        Given 20 animals, of which 7 are dogs. Then hypergeometric PDF gives
        the probability of finding a given number of dogs if we choose at
        random 12 of the 20 animals.
        """
        degree = 4
        M, n, N = 20, 7, 12
        apoly, bpoly = -(n+1), -M-1+n
        ab = hahn_recurrence(
            degree+1, N, apoly, bpoly)
        x, w = gauss_quadrature(ab, degree+1)

        rv = hypergeom(M, n, N)
        true_mean = rv.mean()
        quadrature_mean = x.dot(w)
        assert np.allclose(true_mean, quadrature_mean)

        x = np.arange(0, n+1)
        p = evaluate_orthonormal_polynomial_1d(x, degree, ab)
        w = rv.pmf(x)
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

    def test_discrete_chebyshev(self):
        N, degree = 100, 5
        xk, pk = np.arange(N), np.ones(N)/N
        rv = float_rv_discrete(name='discrete_chebyshev', values=(xk, pk))
        ab = discrete_chebyshev_recurrence(degree+1, N)
        p = evaluate_orthonormal_polynomial_1d(xk, degree, ab)
        w = rv.pmf(xk)
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1))

    def test_charlier(self):
        # Note as rate gets smaller the number of terms that can be accurately
        # computed will decrease because the problem gets more ill conditioned.
        # This is caused because the number of masses with significant weights
        # gets smaller as rate does
        degree, rate = 5, 2
        rv = poisson(rate)
        ab = charlier_recurrence(degree+1, rate)
        lb, ub = rv.interval(1-np.finfo(float).eps)
        x = np.linspace(lb, ub, int(ub-lb+1))
        p = evaluate_orthonormal_polynomial_1d(x, degree, ab)
        w = rv.pmf(x)
        # print(np.absolute(np.dot(p.T*w,p)-np.eye(degree+1)).max())
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1), atol=1e-7)

    def test_convert_orthonormal_recurence_to_three_term_recurence(self):
        rho = 0.
        degree = 2
        probability_measure = True
        ab = hermite_recurrence(
            degree+1, rho, probability=probability_measure)
        abc = convert_orthonormal_recurence_to_three_term_recurence(ab)

        x = np.linspace(-3, 3, 101)
        p_2term = evaluate_orthonormal_polynomial_1d(x, degree, ab)
        p_3term = evaluate_three_term_recurrence_polynomial_1d(abc, degree, x)
        assert np.allclose(p_2term, p_3term)

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
        rho = 0.
        degree = 10
        probability_measure = True
        ab = hermite_recurrence(
            degree+1, rho, probability=probability_measure)

        basis_mono_coefs = convert_orthonormal_polynomials_to_monomials_1d(
            ab, 4)

        true_basis_mono_coefs = np.zeros((5, 5))
        true_basis_mono_coefs[0, 0] = 1
        true_basis_mono_coefs[1, 1] = 1
        true_basis_mono_coefs[2, [0, 2]] = -1/np.sqrt(2), 1/np.sqrt(2)
        true_basis_mono_coefs[3, [1, 3]] = -3/np.sqrt(6), 1/np.sqrt(6)
        true_basis_mono_coefs[4, [0, 2, 4]] = np.array([3, -6, 1])/np.sqrt(24)

        assert np.allclose(basis_mono_coefs, true_basis_mono_coefs)

        coefs = np.ones(degree+1)
        basis_mono_coefs = convert_orthonormal_polynomials_to_monomials_1d(
            ab, degree)
        mono_coefs = np.sum(basis_mono_coefs*coefs, axis=0)

        x = np.linspace(-3, 3, 5)
        p_ortho = evaluate_orthonormal_polynomial_1d(x, degree, ab)
        ortho_vals = p_ortho.dot(coefs)

        mono_vals = evaluate_monomial(
            np.arange(degree+1)[np.newaxis, :], mono_coefs,
            x[np.newaxis, :])[:, 0]
        assert np.allclose(ortho_vals, mono_vals)

    def test_convert_monomials_to_orthonormal_polynomials_1d(self):
        rho = 0.
        degree = 10
        probability_measure = True
        ab = hermite_recurrence(
            degree+1, rho, probability=probability_measure)

        basis_mono_coefs = convert_orthonormal_polynomials_to_monomials_1d(
            ab, degree)

        x = np.random.normal(0, 1, (100))
        print('Cond number', np.linalg.cond(basis_mono_coefs))
        basis_ortho_coefs = np.linalg.inv(basis_mono_coefs)
        ortho_basis_matrix = evaluate_orthonormal_polynomial_1d(x, degree, ab)
        mono_basis_matrix = x[:, None]**np.arange(degree+1)[None, :]
        assert np.allclose(
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
        degree = 4
        mu, sigma = 1, 2
        ortho_coef = np.array([0, 3, 0, np.sqrt(6)])
        ab = hermite_recurrence(degree+1, 0, True)
        mono_coefs = convert_orthonormal_expansion_to_monomial_expansion_1d(
            ortho_coef, ab, mu, sigma)
        true_mono_coefs = np.array([-mu**3, 3*mu**2, -3*mu, 1])/sigma**3
        assert np.allclose(mono_coefs, true_mono_coefs)

    def test_orthonormality_laguerre_polynomial(self):
        a = 3
        rho = a-1
        degree = 2
        probability_measure = True
        ab = laguerre_recurrence(
            rho, degree+1, probability=probability_measure)
        print(ab)
        from scipy import stats
        x = stats.gamma(a).rvs(int(1e6))
        w = np.ones(x.shape[0])/x.shape[0]
        p = evaluate_orthonormal_polynomial_1d(x, degree, ab)

        p_exact = np.asarray(
            [1+0.*x, -(-x+rho+1),
             0.5*x**2-x*(rho+2)+(rho+1)*(rho+2)/2]).T[:, :degree+1]
        p_exact /= np.sqrt(sp.gamma(np.arange(degree+1)+rho+1)/sp.factorial(
            np.arange(degree+1))/sp.gamma(rho+1))
        #print(p_exact.T.dot(w), 'p')
        #print(np.dot(p_exact.T*w, p_exact)-np.eye(degree+1))
        assert np.allclose(p, p_exact)

        # test orthogonality
        exact_moments = np.zeros((degree+1))
        exact_moments[0] = 1.0
        #print(np.dot(p.T, w)-exact_moments)
        assert np.allclose(np.dot(p.T, w), exact_moments, atol=2e-3)
        # test orthonormality
        # print(np.dot(p.T*w, p)-np.eye(degree+1))
        assert np.allclose(np.dot(p.T*w, p), np.eye(degree+1), atol=2e-2)


if __name__ == "__main__":
    orthonormal_poly_1d_test_suite = \
        unittest.TestLoader().loadTestsFromTestCase(
            TestOrthonormalPolynomials1D)
    unittest.TextTestRunner(verbosity=2).run(orthonormal_poly_1d_test_suite)
