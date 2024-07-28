import unittest

import numpy as np
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.bases.linearsystemsolvers import OMPSolver


class TestLinearSolvers:
    def setUp(self):
        np.random.seed(1)

    def test_omp_solver(self):
        # Test dense least squares solution is recovered
        # when OMP is allowed to add all columns
        bkd = self.get_backend()
        ncols = 10
        Amat = bkd._la_asarray(np.random.uniform(-1, 1, (101, ncols)))
        assert np.linalg.matrix_rank(Amat) == ncols
        coefs = bkd._la_asarray(np.random.normal(0, 1, (ncols, 1)))
        Bmat = Amat @ coefs
        solver = OMPSolver(max_nonzeros=np.inf, rtol=0, backend=bkd)
        omp_coefs = solver.solve(Amat, Bmat)
        assert np.allclose(omp_coefs, coefs)
        assert solver._termination_flag == 1

        # Test exit when columns are not independent
        solver = OMPSolver(max_nonzeros=np.inf, rtol=0, backend=bkd)
        Cmat = bkd._la_hstack((Amat, Amat[:, 2:3]))
        omp_coefs = solver.solve(Cmat, Bmat)
        assert np.allclose(omp_coefs[:-1], coefs)
        assert np.allclose(omp_coefs[-1, 0], 0)
        assert solver._termination_flag == 2

        # Test recovery of sparse coefficients
        solver = OMPSolver(max_nonzeros=np.inf, rtol=1e-14, backend=bkd)
        sparse_coefs = bkd._la_copy(coefs)
        ninactive_cols = 4
        inactive_indices = bkd._la_asarray(np.random.permutation(
            np.arange(ncols)), dtype=int)[:ninactive_cols]
        sparse_coefs[inactive_indices] = 0.
        Bmat = Amat @ sparse_coefs
        omp_coefs = solver.solve(Amat, Bmat)
        assert np.allclose(omp_coefs, sparse_coefs)
        assert solver._termination_flag == 0


class TestNumpyLinearSolvers(TestLinearSolvers, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin()


class TestTorchLinearSolvers(TestLinearSolvers, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin()


if __name__ == "__main__":
    unittest.main(verbosity=2)
