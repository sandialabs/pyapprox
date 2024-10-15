import unittest

import torch
torch.manual_seed(1)

from pyapprox.surrogates.bases.optimizers import ScipyLBFGSB
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin


class TestOptimizers(unittest.TestCase):

    def setUp(self):
        self._bkd = TorchLinAlgMixin
        self.pi = 3.1415926535897932

    def loss(self, x):
        xstar = self._bkd.asarray([4.2, 1.0, 10.4, self.pi])
        return ((x-xstar)**2).sum()

    def objective_fun(self, x, **kwargs):
        xtorch = self._bkd.asarray(x)
        xtorch.requires_grad = True
        nll = self.loss(xtorch)
        nll.backward()
        val = nll.item()
        nll_grad = xtorch.grad.detach().numpy().copy()
        xtorch.grad.zero_()
        xtorch.detach()
        return val, nll_grad

    def test_lbfgsb(self):
        optimizer = ScipyLBFGSB()
        optimizer.set_tolerance(1e-12)
        xopt = self._bkd.asarray([4.2, 1.0, 10.4, self.pi])
        optimizer.set_objective_function(self.objective_fun)
        optimizer.set_bounds(
            self._bkd.to_numpy(
                self._bkd.tile(self._bkd.asarray([-self._bkd.inf(),
                                              self._bkd.inf()]),
                                     (4, 1))
            )
        )
        x0 = self._bkd.zeros((4,))
        res = optimizer.optimize(self._bkd.to_numpy(x0))
        assert self._bkd.allclose(self._bkd.asarray(res.x), xopt)
        assert self._bkd.abs(self._bkd.asarray(res.fun)) < 1e-12

if __name__ == '__main__':
    optimizers_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestOptimizers))
    unittest.TextTestRunner(verbosity=2).run(optimizers_test_suite)
