import unittest

import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin


class TestLinalg:
    def setUp(self):
        np.random.seed(1)
    
    def test_update_cholesky_decomposition(self):
        nvars = 5
        B = self.atleast2d(np.random.normal(0, 1, (nvars, nvars)))
        A = B.T @ B
        
        L = self.cholesky(A)
        A_11 = A[:nvars-2, :nvars-2]
        A_12 = A[:nvars-2, nvars-2:]
        A_22 = A[nvars-2:, nvars-2:]
        assert self.allclose(self.block([[A_11, A_12], [A_12.T, A_22]]), A)
        L_11 = self.cholesky(A_11)
        L_up = self.update_cholesky_factorization(L_11, A_12, A_22)[0]
        assert self.allclose(L, L_up)
          

class TestNumpyLinalg(unittest.TestCase, TestLinalg, NumpyLinAlgMixin):
    pass


class TestTorchLinalg(unittest.TestCase, TestLinalg, TorchLinAlgMixin):
    pass


if __name__ == '__main__':
    unittest.main()
