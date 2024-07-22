import unittest
import copy

import numpy as np

from pyapprox.surrogates.orthopoly.recursion_factory import (
    LegendrePolynomial1D)
from pyapprox.surrogates.bases._basis import (
    MonomialBasis, OrthonormalPolynomialBasis)
from pyapprox.surrogates.bases._basisexp import (
    MonomialExpansion, PolynomialChaosExpansion)
from pyapprox.surrogates.bases._linearsystemsolvers import (
    LstSqSolver, OMPSolver)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin


class TestMonomialBasis:
    def _check_basis(self, nvars, nterms_1d):
        basis = MonomialBasis(backend=self.get_backend())
        basis.set_indices(self._la_cartesian_product(
            [self._la_arange(nterms_1d, dtype=int)]*nvars))
        samples = self._la_array(np.random.uniform(-1, 1, (nvars, 4)))
        basis_mat = basis(samples)
        for ii, index in enumerate(basis._indices.T):
            assert np.allclose(
                basis_mat[:, ii], self._la_prod(samples.T**index, axis=1))

    def test_basis(self):
        test_cases = [[1, 4], [2, 4], [3, 4]]
        for test_case in test_cases:
            self._check_basis(*test_case)

    def _check_jacobian(self, nvars, nterms_1d):
        basis = MonomialBasis(self)
        indices = self._la_cartesian_product(
            [self._la_arange(nterms_1d, dtype=int)]*nvars)
        basis.set_indices(indices)
        samples = self._la_array(np.random.uniform(-1, 1, (nvars, 4)))
        # samples = self._la_array([[-1, 1], [1, 0.5], [0.5, 0.5], [0.5, 1]]).T
        jac = basis.jacobian(samples)
        derivs = self._la_stack([samples*0, samples*0+1]+[
            ii*samples**(ii-1) for ii in range(2, nterms_1d)])
        for ii in range(indices.shape[1]):
            for dd in range(nvars):
                index = self._la_copy(indices[:, ii:ii+1])
                # evaluate basis that has constant in direction of derivative
                index[dd] = 0
                basis.set_indices(index)
                deriv_dd = derivs[indices[dd, ii], dd, :]*self._la_prod(
                    basis(samples), axis=1)
                if not self._la_allclose(deriv_dd, jac[:, ii, dd]):
                    print(dd, indices[:, ii])
                    print(derivs[indices[dd, ii], dd, :], "D")
                    print(basis(samples)[:, 0], "B")
                    print(deriv_dd)
                    print(jac[:, ii, dd])
                assert self._la_allclose(deriv_dd, jac[:, ii, dd])

    def test_jacobian(self):
        test_cases = [[1, 4], [2, 4], [3, 4]]
        for test_case in test_cases:
            # print(test_case)
            self._check_jacobian(*test_case)

    def _check_monomial_expansion(self, nvars, solver, nqoi):
        nterms_1d = 3
        basis = MonomialBasis(backend=self.get_backend())
        indices = self._la_cartesian_product(
            [self._la_arange(nterms_1d, dtype=int)]*nvars)
        basis.set_indices(indices)
        basisexp = MonomialExpansion(basis, solver=solver, nqoi=nqoi)
        ntrain_samples = 2*basis.nterms()
        train_samples = self._la_cos(
            self._la_array(
                np.random.uniform(0, np.pi, (nvars, ntrain_samples))))

        # Attempt to recover coefficients of additive function
        def fun(samples):
            values = self._la_sum(samples**2+samples, axis=0)[:, None]+1.0
            # Create 2 QoI
            return self._la_hstack([(ii+1)*values for ii in range(nqoi)])

        train_values = fun(train_samples)
        basisexp.fit(train_samples, train_values)
        coef = basisexp.get_coefficients()
        nonzero_indices = self._la_hstack(
            (self._la_where(
                self._la_count_nonzero(basis._indices, axis=0) == 0)[0],
             self._la_where(
                 self._la_count_nonzero(basis._indices, axis=0) == 1)[0]))
        true_coef = self._la_full((basis.nterms(), basisexp.nqoi()), 0)
        for ii in range(nqoi):
            true_coef[nonzero_indices, ii] = ii+1
        assert self._la_allclose(coef, true_coef)
        samples = self._la_atleast2d(np.random.uniform(-1, 1, (nvars, 1000)))
        assert self._la_allclose(basisexp(samples), fun(samples))

    def test_monomial_expansion(self):
        bkd = self.get_backend()
        test_cases = [[1, LstSqSolver(backend=bkd), 2],
                      [2, LstSqSolver(backend=bkd), 2],
                      [3, LstSqSolver(backend=bkd), 2],
                      [1, OMPSolver(max_nonzeros=3, backend=bkd), 1],
                      [2, OMPSolver(max_nonzeros=6, backend=bkd), 1],
                      [3, OMPSolver(max_nonzeros=9, backend=bkd), 1]]
        for test_case in test_cases:
            self._check_basis_expansion(*test_case)

    def test_legendre_basis(self):
        nvars, degree = 2, 2
        bases_1d = [LegendrePolynomial1D(backend=self.get_backend())]*nvars
        basis = OrthonormalPolynomialBasis(bases_1d)
        basis.set_indices(
            self._la_array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]).T)
        samples = self._la_array(np.random.uniform(-1, 1, (nvars, 101)))
        basis_mat = basis(samples)
        exact_basis_vals_1d = []
        exact_basis_derivs_1d = []
        for dd in range(nvars):
            x = samples[dd, :]
            exact_basis_vals_1d.append(
                self._la_stack([1+0.*x, x, 0.5*(3.*x**2-1)], axis=0).T)
            exact_basis_derivs_1d.append(
                self._la_stack([0.*x, 1.0+0.*x, 3.*x], axis=0).T)
            exact_basis_vals_1d[-1] /= self._la_sqrt(
                1./(2*self._la_arange(degree+1)+1))
            exact_basis_derivs_1d[-1] /= self._la_sqrt(
                1./(2*self._la_arange(degree+1)+1))

        exact_basis_mat = self._la_stack(
            [exact_basis_vals_1d[0][:, 0], exact_basis_vals_1d[0][:, 1],
             exact_basis_vals_1d[1][:, 1], exact_basis_vals_1d[0][:, 2],
             exact_basis_vals_1d[0][:, 1]*exact_basis_vals_1d[1][:, 1],
             exact_basis_vals_1d[1][:, 2]], axis=0).T

        assert self._la_allclose(basis_mat, exact_basis_mat)

    def test_legendre_poly_1d(self):
        degree = 3
        poly = LegendrePolynomial1D(backend=self.get_backend())
        poly.set_recursion_coefficients(degree+1)
        quad_x, quad_w = np.polynomial.legendre.leggauss(degree+1)
        # make weights have probablity weight function w=1/2
        quad_x = self._la_array(quad_x)
        quad_w = self._la_array(quad_w/2.0)
        vals = poly(quad_x, degree)
        # test orthogonality
        exact_moments = self._la_full((degree+1,), 0.)
        exact_moments[0] = 1.0
        assert np.allclose(vals.T @ quad_w, exact_moments)
        # test orthonormality
        assert np.allclose((vals.T*quad_w) @ vals, self._la_eye(degree+1))

        # check quadrature rule is computed correctly
        pquad_x, pquad_w = poly.gauss_quadrature_rule(degree+1)
        assert self._la_allclose(pquad_x, quad_x)
        assert self._la_allclose(pquad_w, quad_w/2)

    def _check_multiply_expansion(self, bexp1, bexp2, nqoi):
        coef1 = self._la_arange(bexp1.nterms()*nqoi).reshape(
            (bexp1.nterms(), nqoi))
        coef2 = self._la_arange(bexp2.nterms()*nqoi).reshape(
            (bexp2.nterms(), nqoi))
        bexp1.set_coefficients(coef1)
        bexp2.set_coefficients(coef2)

        bexp3 = bexp1*bexp2
        samples = self._la_array(
            np.random.uniform(-1, 1, (bexp1.nvars(), 101)))
        assert self._la_allclose(bexp3(samples), bexp1(samples)*bexp2(samples))

        for order in range(4):
            bexp = bexp1**order
            assert self._la_allclose(bexp(samples), bexp1(samples)**order)

    def _check_multiply_monomial_expansion(self, nvars, nterms_1d, nqoi):
        bkd = self.get_backend()
        basis1 = MonomialBasis(backend=bkd)
        indices1 = self._la_cartesian_product(
            [self._la_arange(nterms_1d, dtype=int)]*nvars)
        basis2 = MonomialBasis(backend=bkd)
        indices2 = self._la_cartesian_product(
            [self._la_arange(nterms_1d+1, dtype=int)]*nvars)
        basis1.set_indices(indices1)
        basis2.set_indices(indices2)
        bexp1 = MonomialExpansion(basis1, solver=None, nqoi=nqoi)
        bexp2 = MonomialExpansion(basis2, solver=None, nqoi=nqoi)
        self._check_multiply_expansion(bexp1, bexp2, nqoi)

    def test_multiply_monomial_expansion(self):
        test_cases = [[1, 3, 2], [2, 3, 2]]
        for test_case in test_cases:
            self._check_multiply_monomial_expansion(*test_case)

    def _check_multiply_pce(self, nvars, nterms_1d, nqoi):
        bkd = self.get_backend()
        polys_1d = [LegendrePolynomial1D(bkd)]*nvars
        basis1 = OrthonormalPolynomialBasis(polys_1d)
        indices1 = self._la_cartesian_product(
            [self._la_arange(nterms_1d, dtype=int)]*nvars)
        basis2 = OrthonormalPolynomialBasis(polys_1d)
        indices2 = self._la_cartesian_product(
            [self._la_arange(nterms_1d+1, dtype=int)]*nvars)
        basis1.set_indices(indices1)
        basis2.set_indices(indices2)
        bexp1 = PolynomialChaosExpansion(basis1, solver=None, nqoi=nqoi)
        bexp2 = PolynomialChaosExpansion(basis2, solver=None, nqoi=nqoi)
        self._check_multiply_expansion(bexp1, bexp2, nqoi)

    def test_multiply_pce(self):
        test_cases = [[1, 3, 2], [2, 3, 2]]
        for test_case in test_cases:
            self._check_multiply_pce(*test_case)


class TestNumpyMonomialBasis(
        unittest.TestCase, TestMonomialBasis, NumpyLinAlgMixin):
    def setUp(self):
        np.random.seed(1)

    def get_backend(self):
        return NumpyLinAlgMixin()


class TestTorchMonomialBasis(
        unittest.TestCase, TestMonomialBasis, TorchLinAlgMixin):
    def setUp(self):
        np.random.seed(1)

    def get_backend(self):
        return TorchLinAlgMixin()


if __name__ == '__main__':
    unittest.main()
