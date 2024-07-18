import unittest

import numpy as np

from pyapprox.surrogates.bases.numpybasis import (
    NumpyMonomialBasis, NumpyBasisExpansion, NumpyLstSqSolver, NumpyOMPSolver)
from pyapprox.util.utilities import cartesian_product


class TestMonomialBasis(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _check_basis(self, nvars, nterms_1d):
        basis = NumpyMonomialBasis()
        basis.set_indices(cartesian_product([np.arange(nterms_1d)]*nvars))
        samples = np.random.uniform(-1, 1, (nvars, 4))
        basis_mat = basis(samples)
        for ii, index in enumerate(basis._indices.T):
            assert np.allclose(
                basis_mat[:, ii], np.prod(samples.T**index, axis=1))
        
    def test_basis(self):
        test_cases = [[1, 4], [2, 4], [3, 4]]
        for test_case in test_cases:
            self._check_basis(*test_case)
        
    def _check_jacobian(self, nvars, nterms_1d):
        basis = NumpyMonomialBasis()
        indices = cartesian_product([np.arange(nterms_1d)]*nvars)
        basis.set_indices(indices)
        samples = np.random.uniform(-1, 1, (nvars, 4))
        # samples = np.array([[-1, 1], [1, 0.5], [0.5, 0.5], [0.5, 1]]).T
        basis_mat = basis(samples)
        jac = basis.jacobian(samples)
        derivs = np.stack([samples*0, samples*0+1, 2*samples])
        dims = np.arange(nvars)
        for ii in range(indices.shape[1]):
            for dd in range(nvars):
                index = indices[:, ii:ii+1].copy()
                # evaluate basis that has constant in direction of derivative
                index[dd] = 0
                basis.set_indices(index)
                deriv_dd = derivs[indices[dd, ii], dd, :]*np.prod(
                    basis(samples), axis=1)
                assert np.allclose(deriv_dd, jac[:, ii, dd])

    def test_jacobian(self):
        test_cases = [[1, 4], [2, 4], [3, 4]]
        for test_case in test_cases:
            self._check_basis(*test_case)

    def _check_basis_expansion(self, nvars, solver, nqoi):
        nterms_1d = 3
        basis = NumpyMonomialBasis()
        indices = cartesian_product([np.arange(nterms_1d)]*nvars)
        basis.set_indices(indices)
        basisexp = NumpyBasisExpansion(basis, solver=solver, nqoi=nqoi)
        ntrain_samples = 2*basis.nterms()
        train_samples = np.cos(
            np.random.uniform(0, np.pi, (nvars, ntrain_samples)))

        # Attempt to recover coefficients of additive function
        def fun(samples):
            values = np.sum(samples**2+samples, axis=0)[:, None]+1.0
            # Create 2 QoI
            return np.hstack([(ii+1)*values for ii in range(nqoi)])

        train_values = fun(train_samples)
        basisexp.fit(train_samples, train_values)
        coef = basisexp.get_coefficients()
        print(coef, solver)
        nonzero_indices = np.hstack(
            (np.where(np.count_nonzero(basis._indices, axis=0)==0)[0],
             np.where(np.count_nonzero(basis._indices, axis=0)==1)[0]))
        true_coef = np.full((basis.nterms(), basisexp.nqoi()), 0)
        print(true_coef.shape)
        true_coef[nonzero_indices, 0] = 1
        true_coef[nonzero_indices, 1] = 2
        assert np.allclose(coef, true_coef)
        samples = np.random.uniform(-1, 1, (nvars, 1000))
        assert np.allclose(basisexp(samples), fun(samples))

    def test_linear_expansion(self):
        test_cases = [[1, NumpyLstSqSolver(), 2],
                      [2, NumpyLstSqSolver(), 2],
                      [3, NumpyLstSqSolver(), 2],
                      [1, NumpyOMPSolver(max_nonzeros=3), 1],
                      [2, NumpyOMPSolver(max_nonzeros=6), 1],
                      [3, NumpyOMPSolver(max_nonzeros=9), 1]]
        for test_case in test_cases[1:]:
            self._check_basis_expansion(*test_case)
        

if __name__ == '__main__':
    unittest.main()
        
