import unittest

import numpy as np

from pyapprox.expdesign.local import (
    DOptimalLstSqCriterion,
    DOptimalQuantileCriterion,
    COptimalLstSqCriterion,
)
from pyapprox.surrogates.bases.univariate import Monomial1D
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin


class TestLocalOED:
    def setUp(self):
        np.random.seed(1)

    def _check_criterion_gradients(self, crit_cls, hetro, *args):
        bkd = self.get_backend()
        poly_degree = 3
        ndesign_pts = 11
        design_samples = bkd.linspace(-1, 1, ndesign_pts)[None, :]
        monomial = Monomial1D(poly_degree+1, backend=bkd)
        design_factors = monomial(design_samples)
        if hetro:
            noise_mult = 1+design_samples[0, :]**2+1
        else:
            noise_mult = None
        crit = crit_cls(design_factors, *args, noise_mult, backend=bkd)
        iterate = bkd.array(
            np.random.uniform(0, 1, (ndesign_pts, 1))
        )
        print(crit(iterate))
        print(crit.jacobian(iterate))
        # print(bkd.jacobian(crit, iterate))
        print(crit.approx_jacobian(iterate))
        # assert False
        errors = crit.check_apply_jacobian(iterate, disp=True)
        assert errors.min()/errors.max() < 5e-7

        errors = crit.check_apply_hessian(iterate, disp=True)

    def test_criterion_gradients(self):
        bkd = self.get_backend()
        test_cases = [
            # [DOptimalLstSqCriterion, False],
            # [DOptimalLstSqCriterion, True],
            # [DOptimalQuantileCriterion, False],
            # [DOptimalQuantileCriterion, True],
            [COptimalLstSqCriterion, False, bkd.ones(4)],
            [COptimalLstSqCriterion, True, bkd.ones(4)]
        ]
        for test_case in test_cases:
            self._check_criterion_gradients(*test_case)



class TestNumpyLocalOED(TestLocalOED, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


# class TestTorchLocalOED(TestLocalOED, unittest.TestCase):
#     def get_backend(self):
#         return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
