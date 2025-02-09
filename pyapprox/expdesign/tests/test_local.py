import unittest

import numpy as np

from pyapprox.expdesign.local import (
    DOptimalLstSqCriterion,
    DOptimalQuantileCriterion,
)
from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.surrogates.bases.univariate import Monomial1D


class TestLocalOED:
    def setUp(self):
        np.random.seed(1)

    def _check_criterion_gradients(self, crit_cls, hetro):
        bkd = self.get_backend()
        poly_degree = 3
        ndesign_pts = 11
        design_samples = bkd.linspace(-1, 1, ndesign_pts)[None, :]
        # noise_multiplier = None
        monomial = Monomial1D(poly_degree+1, backend=bkd)
        design_factors = monomial(design_samples)
        if hetro:
            noise_mult = 1+design_samples[0, :]**2+1
        else:
            noise_mult = None
        crit = crit_cls(design_factors, noise_mult)
        iterate = bkd.array(
            np.random.uniform(0, 1, (ndesign_pts, 1))
        )
        errors = crit.check_apply_jacobian(iterate, disp=True)
        print(crit.jacobian(iterate))
        print(crit.approx_jacobian(iterate))
        assert errors.min()/errors.max() < 5e-7

    def test_criterion_gradients(self):
        test_cases = [
            [DOptimalLstSqCriterion, False],
            [DOptimalLstSqCriterion, True],
            [DOptimalQuantileCriterion, False],
            [DOptimalQuantileCriterion, True],
        ]
        for test_case in test_cases:
            self._check_criterion_gradients(*test_case)



class TestNumpyLocalOED(TestLocalOED, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin


if __name__ == "__main__":
    unittest.main(verbosity=2)
