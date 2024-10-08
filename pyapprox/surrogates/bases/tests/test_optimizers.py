import unittest

import numpy as np

from pyapprox.sciml.util._torch_wrappers import asarray
from pyapprox.sciml.optimizers import LBFGSB, Adam
from pyapprox.sciml.network import CERTANN
from pyapprox.sciml.integraloperators import FourierConvolutionOperator
from pyapprox.sciml.activations import IdentityActivation


class TestOptimizers(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def loss(self, x):
        xstar = asarray(np.asarray([4.2, 1.0, 10.4, np.pi]))
        return ((asarray(x)-xstar)**2).sum()

    def objective_fun(self, x, **kwargs):
        xtorch = asarray(x, requires_grad=True)
        nll = self.loss(xtorch)
        nll.backward()
        val = nll.item()
        nll_grad = xtorch.grad.detach().numpy().copy()
        return val, nll_grad

    def test_lbfgsb(self):
        optimizer = LBFGSB()
        optimizer.set_tolerance(1e-12)
        xopt = np.asarray([4.2, 1.0, 10.4, np.pi])
        optimizer.set_objective_function(self.objective_fun)
        optimizer.set_bounds(np.tile(np.asarray([-np.inf, np.inf]), (4, 1)))
        x0 = asarray(np.zeros((4,)), requires_grad=True)
        res = optimizer.optimize(x0)
        assert np.allclose(res.x, xopt)
        assert np.abs(res.fun) < 1e-12

        # Sanity check: Does default CERTANN objective function work with this
        # optimizer?
        nvars = 8
        ctn = CERTANN(nvars, [FourierConvolutionOperator(2)],
                      [IdentityActivation()], optimizer=LBFGSB())
        samples = asarray(np.random.uniform(-1, 1, (nvars, 1)))
        values = asarray(np.random.uniform(-1, 1, (nvars, 1)))
        ctn.fit(samples, values)

    def test_adam(self):
        optimizer = Adam(epochs=400, lr=1.0)
        xopt = np.asarray([4.2, 1.0, 10.4, np.pi])
        optimizer.set_objective_function(self.objective_fun)
        x0 = asarray(np.zeros((4,)), requires_grad=True)
        res = optimizer.optimize(x0)
        assert np.allclose(res.x, xopt)
        assert np.abs(res.fun) < 1e-12

        # Sanity check: Does default CERTANN objective function work with this
        # optimizer?
        nvars = 8
        ctn = CERTANN(nvars, [FourierConvolutionOperator(2)],
                      [IdentityActivation()], optimizer=Adam())
        samples = asarray(np.random.uniform(-1, 1, (nvars, 1)))
        values = asarray(np.random.uniform(-1, 1, (nvars, 1)))
        ctn.fit(samples, values)


if __name__ == '__main__':
    optimizers_test_suite = (
        unittest.TestLoader().loadTestsFromTestCase(TestOptimizers))
    unittest.TextTestRunner(verbosity=2).run(optimizers_test_suite)
