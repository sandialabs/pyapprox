import unittest
import numpy as np
import torch

from pyapprox.surrogates.autogp._torch_wrappers import log
from pyapprox.surrogates.autogp.kernels import (
    ConstantKernel, MaternKernel, PeriodicMaternKernel)


def approx_jacobian_3D(f, x0, epsilon=np.sqrt(np.finfo(float).eps)):
    fval = f(x0)
    jacobian = np.zeros((fval.shape[0], fval.shape[1], x0.shape[0]))
    for ii in range(len(x0)):
        dx = np.full((x0.shape[0]), 0.)
        dx[ii] = epsilon
        fval_perturbed = f(x0+dx)
        jacobian[..., ii] = (fval_perturbed - fval) / epsilon
    return jacobian


class TestKernels(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_kernels(self):
        kernel_inf = MaternKernel(np.inf, 1.0, [1e-1, 1], 2)
        values = torch.as_tensor([0.5, 0.5], dtype=torch.double)
        kernel_inf.hyp_list.set_active_opt_params(log(values))
        assert np.allclose(kernel_inf.hyp_list.get_values(), values)

        nsamples1, nsamples2 = 5, 3
        X = np.random.normal(0, 1, (2, nsamples1))
        Y = np.random.normal(0, 1, (2, nsamples2))
        assert np.allclose(kernel_inf.diag(X), np.diag(kernel_inf(X, X)))

        const0 = 2.0
        kernel_prod = kernel_inf*ConstantKernel(const0)
        assert np.allclose(kernel_prod.diag(X), const0*kernel_inf.diag(X))
        assert np.allclose(kernel_prod.diag(X), np.diag(kernel_prod(X, X)))
        assert np.allclose(kernel_prod(X, Y), const0*kernel_inf(X, Y))

        const1 = 3.0
        kernel_sum = kernel_prod+ConstantKernel(const1)
        assert np.allclose(
            kernel_sum.diag(X), const0*kernel_inf.diag(X)+const1)
        assert np.allclose(kernel_sum.diag(X), np.diag(kernel_sum(X, X)))
        assert np.allclose(kernel_sum(X, Y), const0*kernel_inf(X, Y)+const1)

        kernel_periodic = PeriodicMaternKernel(
            0.5, 1.0, [1e-1, 1], 1, [1e-1, 1])
        values = torch.as_tensor([0.5, 0.5], dtype=torch.double)
        kernel_periodic.hyp_list.set_active_opt_params(log(values))
        assert np.allclose(kernel_periodic.hyp_list.get_values(), values)
        assert np.allclose(
            kernel_periodic.diag(X), np.diag(kernel_periodic(X, X)))

    def check_kernel_jacobian(self, kernel, nsamples):
        X = np.random.uniform(-1, 1, (kernel.nvars(), nsamples))

        def fun(active_params_opt):
            if not isinstance(active_params_opt, np.ndarray):
                active_params_opt.requires_grad = True
            else:
                active_params_opt = torch.as_tensor(
                    active_params_opt, dtype=torch.double)
            kernel.hyp_list.set_active_opt_params(active_params_opt)
            return kernel(X)

        jacobian = torch.autograd.functional.jacobian(
            fun, kernel.hyp_list.get_active_opt_params())
        for hyp in kernel.hyp_list.hyper_params:
            hyp._values = hyp._values.clone().detach()
        assert np.allclose(
            jacobian.numpy(),
            approx_jacobian_3D(
                fun, kernel.hyp_list.get_active_opt_params().detach().numpy()))

    def test_kernel_jacobian(self):
        nvars, nsamples = 2, 3
        kernel = MaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
        self.check_kernel_jacobian(kernel, nsamples)


if __name__ == "__main__":
    kernels_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestKernels)
    unittest.TextTestRunner(verbosity=2).run(kernels_test_suite)
