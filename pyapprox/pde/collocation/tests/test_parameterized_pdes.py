import unittest

import numpy as np
import matplotlib.pyplot as plt

# from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.pde.collocation.parameterized_pdes import LotkaVolterraModel
from pyapprox.pde.collocation.timeintegration import (
    BackwardEulerResidual, CrankNicholsonResidual, ForwardEulerResidual,
    HeunResidual
)
from pyapprox.pde.collocation.newton import NewtonSolver
from pyapprox.pde.collocation.tests.test_timeintegration import (
    TransientSingleStateLinearFunctional
)


class TestParameterizedModels:
    def setUp(self):
        np.random.seed(1)

    def _check_lotka_volterra(self, time_residual_cls):
        bkd = self.get_backend()
        functional = TransientSingleStateLinearFunctional(3, 12, backend=bkd)
        # for some reason error in check apply jacobian depend on tolerance
        newton_solver = NewtonSolver(verbosity=0, rtol=1e-12, atol=1e-12)
        model = LotkaVolterraModel(
            0, 10, 1, time_residual_cls, functional,
            newton_solver=newton_solver
        )
        sample = bkd.array(np.random.uniform(0.3, 0.7, model.nvars()))[:, None]
        fd_eps = bkd.flip(bkd.logspace(-13, -1, 12))
        errors = model.check_apply_jacobian(sample, fd_eps, disp=True)
        print(errors.min()/errors.max())
        assert errors.min()/errors.max() < 1.2e-6

    def test_lotka_volterra(self):
        test_cases = [
            [BackwardEulerResidual],
            [CrankNicholsonResidual],
            [ForwardEulerResidual],
            [HeunResidual],
        ]
        for test_case in test_cases:
            self._check_lotka_volterra(*test_case)


class TestTorchParameterizedModels(TestParameterizedModels, unittest.TestCase):
    def get_backend(self):
        return TorchLinAlgMixin


if __name__ == "__main__":
    unittest.main()
