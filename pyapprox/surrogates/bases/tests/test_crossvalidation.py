import unittest

import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
# from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.orthopoly import LegendrePolynomial1D
from pyapprox.surrogates.bases.basis import OrthonormalPolynomialBasis
from pyapprox.surrogates.bases.basisexp import PolynomialChaosExpansion
from pyapprox.surrogates.bases.linearsystemsolvers import (
    LstSqSolver,
    OMPSolver,
)
from pyapprox.surrogates.interp.indexing import compute_hyperbolic_indices
from pyapprox.surrogates.bases.crossvalidation import (
    KFoldCrossValidation,
    GridSearchStructureParameterIterator,
    CrossValidationStructureSearch,
    PolynomialDegreeIterator,
    OMPNTermsIterator,
)


class TestCrossValidation:
    def setUp(self):
        np.random.seed(1)

    def _check_pce_cv(self, nvars):
        bkd = self.get_backend()
        degree = 2
        polys_1d = [LegendrePolynomial1D(backend=bkd) for dd in range(nvars)]
        basis = OrthonormalPolynomialBasis(polys_1d)
        basis.set_indices(
            bkd.asarray(compute_hyperbolic_indices(basis.nvars(), degree))
        )
        pce = PolynomialChaosExpansion(
            basis, solver=LstSqSolver(backend=bkd), nqoi=1
        )

        def fun(samples):
            return bkd.sum(samples**2, axis=0)[:, None]

        ntrain_samples = basis.nterms() * 5
        train_samples = bkd.asarray(
            np.random.uniform(-1, 1, (nvars, ntrain_samples))
        )
        train_values = fun(train_samples)
        kcv = KFoldCrossValidation(train_samples, train_values, pce)
        score = kcv.run()
        assert score < 1e-13

    def test_pce_cv(self):
        for nvars in range(1, 3):
            self._check_pce_cv(nvars)

    def _check_pce_cv_search(self, nvars, ntrain_samples):
        bkd = self.get_backend()
        degree = 3
        polys_1d = [LegendrePolynomial1D(backend=bkd) for dd in range(nvars)]
        basis = OrthonormalPolynomialBasis(polys_1d)
        basis.set_hyperbolic_indices(degree, 1.)
        pce = PolynomialChaosExpansion(
            basis, solver=OMPSolver(backend=bkd, verbosity=2, rtol=0), nqoi=1
        )

        def fun(samples):
            return bkd.sum(samples**3, axis=0)[:, None]

        train_samples = bkd.asarray(
            np.random.uniform(-1, 1, (nvars, ntrain_samples))
        )
        train_values = fun(train_samples)

        kcv = KFoldCrossValidation(train_samples, train_values, pce)
        search1 = PolynomialDegreeIterator([1, 2, 3], [0.1, 1.0])
        search2 = OMPNTermsIterator([2, 3**nvars+1, np.inf])
        search = GridSearchStructureParameterIterator([search1, search2])
        cv_search = CrossValidationStructureSearch(kcv, search)
        best_structure_params, results, best_idx = cv_search.run()
        self.assertEqual(best_structure_params, ((3, 0.1), 3**nvars+1))
        assert results[best_idx][0] < 1e-15
        print(pce._solver._termination_flag)
        # if ntrain_samples is too small then algorithm may termite with
        # flag 2
        assert pce._solver._termination_flag == 1

    def test_pce_cv_search(self):
        test_cases = [[1, 20], [2, 100]]
        for test_case in test_cases[-1:]:
            self._check_pce_cv_search(*test_case)


class TestNumpyCrossValidation(TestCrossValidation, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# class TestTorchCrossValidation(TestCrossValidation, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
