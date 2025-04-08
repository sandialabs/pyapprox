import unittest

import numpy as np
from pyapprox.util.backends.numpy import NumpyMixin
from pyapprox.util.backends.torch import TorchMixin
from pyapprox.surrogates.affine.linearsystemsolvers import OMPSolver


class TestLinearSolvers:
    def setUp(self):
        np.random.seed(1)

    def test_omp_solver(self):
        # Test dense least squares solution is recovered
        # when OMP is allowed to add all columns
        bkd = self.get_backend()
        ncols = 10
        Amat = bkd.asarray(np.random.uniform(-1, 1, (101, ncols)))
        assert np.linalg.matrix_rank(Amat) == ncols
        coefs = bkd.asarray(np.random.normal(0, 1, (ncols, 1)))
        Bmat = Amat @ coefs
        solver = OMPSolver(max_nonzeros=np.inf, rtol=0, backend=bkd)
        omp_coefs = solver.solve(Amat, Bmat)
        assert bkd.allclose(omp_coefs, coefs)
        assert solver._termination_flag == 1

        # Test exit when columns are not independent
        solver = OMPSolver(max_nonzeros=np.inf, rtol=0, backend=bkd)
        Cmat = bkd.hstack((Amat, Amat[:, 2:3]))
        omp_coefs = solver.solve(Cmat, Bmat)
        assert bkd.allclose(omp_coefs[:-1], coefs)
        assert bkd.allclose(omp_coefs[-1, 0], bkd.zeros(2,))
        assert solver._termination_flag == 2

        # Test recovery of sparse coefficients
        solver = OMPSolver(max_nonzeros=np.inf, rtol=1e-14, backend=bkd)
        sparse_coefs = bkd.copy(coefs)
        ninactive_cols = 4
        inactive_indices = bkd.asarray(np.random.permutation(
            np.arange(ncols)), dtype=int)[:ninactive_cols]
        sparse_coefs[inactive_indices] = 0.
        Bmat = Amat @ sparse_coefs
        omp_coefs = solver.solve(Amat, Bmat)
        assert bkd.allclose(omp_coefs, sparse_coefs)
        assert solver._termination_flag == 0


class TestNumpyLinearSolvers(TestLinearSolvers, unittest.TestCase):
    def get_backend(self):
        return NumpyMixin


class TestTorchLinearSolvers(TestLinearSolvers, unittest.TestCase):
    def get_backend(self):
        return TorchMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
