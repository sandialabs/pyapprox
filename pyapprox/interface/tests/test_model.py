import unittest

import numpy as np
import sympy as sp

from pyapprox.interface.model import ModelFromCallable, ScipyModelWrapper


class TestModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def _evaluate_sp_lambda(self, sp_lambda, sample):
        # sp_lambda returns a single function output
        assert sample.ndim == 2 and sample.shape[1] == 1
        vals = np.atleast_2d(sp_lambda(*sample[:, 0]))
        return vals

    def test_scalar_model_from_callable(self):
        symbs = sp.symbols(["x", "y", "z"])
        nvars = len(symbs)
        sp_fun = sum([s*(ii+1) for ii, s in enumerate(symbs)])**4
        sp_grad = [sp_fun.diff(x) for x in symbs]
        sp_hessian = [[sp_fun.diff(x).diff(y) for x in symbs] for y in symbs]
        model = ModelFromCallable(
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample),
            lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample) @ vec,
            lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian), sample) @ vec)
        sample = np.random.uniform(0, 1, (nvars, 1))
        model.check_apply_jacobian(sample, disp=True)
        # check full jacobian is computed correctly from apply_jacobian
        # when jacobian() is not provided
        assert np.allclose(
           model.jacobian(sample), self._evaluate_sp_lambda(
               sp.lambdify(symbs, sp_grad, "numpy"), sample))
        model.check_apply_hessian(sample, disp=True)
        # check full jacobian is computed correctly from apply_jacobian
        # when hessian() is not provided
        assert np.allclose(model.hessian(sample), self._evaluate_sp_lambda(
               sp.lambdify(symbs, sp_hessian, "numpy"), sample))

    def test_vector_model_from_callable(self):
        symbs = sp.symbols(["x", "y", "z"])
        nvars = len(symbs)
        sp_fun = [sum([s*(ii+1) for ii, s in enumerate(symbs)])**4,
                  sum([s*(ii+1) for ii, s in enumerate(symbs)])**5]
        sp_grad = [[fun.diff(x) for x in symbs] for fun in sp_fun]
        model = ModelFromCallable(
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample),
            lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample) @ vec)
        sample = np.random.uniform(0, 1, (nvars, 1))
        model.check_apply_jacobian(sample, disp=True)
        # check full jacobian is computed correctly from apply_jacobian
        # when jacobian() is not provided
        assert np.allclose(
           model.jacobian(sample), self._evaluate_sp_lambda(
               sp.lambdify(symbs, sp_grad, "numpy"), sample))

    def test_scipy_wrapper(self):
        symbs = sp.symbols(["x", "y", "z"])
        nvars = len(symbs)
        sp_fun = sum([s*(ii+1) for ii, s in enumerate(symbs)])**4
        sp_grad = [sp_fun.diff(x) for x in symbs]
        sp_hessian = [[sp_fun.diff(x).diff(y) for x in symbs] for y in symbs]
        model = ModelFromCallable(
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample),
            lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_grad, "numpy"), sample) @ vec,
            lambda sample, vec: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_hessian), sample) @ vec)
        scipy_model = ScipyModelWrapper(model)
        # check scipy model works with 1D sample array
        sample = np.random.uniform(0, 1, (nvars))
        assert np.allclose(
           scipy_model.jac(sample), self._evaluate_sp_lambda(
               sp.lambdify(symbs, sp_grad, "numpy"), sample[:, None]))
        assert np.allclose(scipy_model.hess(sample), self._evaluate_sp_lambda(
               sp.lambdify(symbs, sp_hessian, "numpy"), sample[:, None]))

        # test error is thrown if scipy model does not return a scalar output
        sp_fun = [sum([s*(ii+1) for ii, s in enumerate(symbs)])**4,
                  sum([s*(ii+1) for ii, s in enumerate(symbs)])**5]
        model = ModelFromCallable(
            lambda sample: self._evaluate_sp_lambda(
                sp.lambdify(symbs, sp_fun, "numpy"), sample))
        scipy_model = ScipyModelWrapper(model)
        self.assertRaises(ValueError, scipy_model, sample)


if __name__ == "__main__":
    model_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestModel)
    unittest.TextTestRunner(verbosity=2).run(model_test_suite)
