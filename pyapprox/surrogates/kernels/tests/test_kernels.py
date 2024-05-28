import unittest
import numpy as np

from pyapprox.surrogates.kernels.numpykernels import (
    NumpyConstantKernel, NumpyMaternKernel, NumpyPeriodicMaternKernel,
    NumpyGaussianNoiseKernel)
from pyapprox.surrogates.kernels.torchkernels import (
    TorchMaternKernel, TorchPeriodicMaternKernel,
    TorchConstantKernel, TorchGaussianNoiseKernel)
from pyapprox.util.hyperparameter.numpyhyperparameter import (
    NumpyIdentityHyperParameterTransform, NumpyLogHyperParameterTransform)


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

    def _check_kernels(self, MaternKernel, ConstantKernel,
                       PeriodicMaternKernel):
        kernel_inf = MaternKernel(np.inf, 1.0, [1e-1, 1], 2)
        values = kernel_inf._la_atleast1d([0.5, 0.5])
        kernel_inf.hyp_list.set_active_opt_params(kernel_inf._la_log(values))
        assert np.allclose(kernel_inf.hyp_list.get_values(), values)

        nsamples1, nsamples2 = 5, 3
        X = np.random.normal(0, 1, (2, nsamples1))
        Y = np.random.normal(0, 1, (2, nsamples2))
        assert np.allclose(
            kernel_inf.diag(X), kernel_inf._la_get_diagonal(kernel_inf(X, X)))

        const0 = 2.0
        kernel_prod = kernel_inf*ConstantKernel(const0)
        assert np.allclose(kernel_prod.diag(X), const0*kernel_inf.diag(X))
        assert np.allclose(
            kernel_prod.diag(X),
            kernel_inf._la_get_diagonal(kernel_prod(X, X)))
        assert np.allclose(kernel_prod(X, Y), const0*kernel_inf(X, Y))

        const1 = 3.0
        kernel_sum = kernel_prod+ConstantKernel(const1)
        assert np.allclose(
            kernel_sum.diag(X), const0*kernel_inf.diag(X)+const1)
        assert np.allclose(
            kernel_sum.diag(X), kernel_prod._la_get_diagonal(kernel_sum(X, X)))
        assert np.allclose(kernel_sum(X, Y), const0*kernel_inf(X, Y)+const1)

        kernel_periodic = PeriodicMaternKernel(
            0.5, 1.0, [1e-1, 1], 1, [1e-1, 1])
        values = kernel_periodic._la_atleast1d([0.5, 0.5])
        kernel_periodic.hyp_list.set_active_opt_params(
            kernel_periodic._la_log(values))
        assert np.allclose(kernel_periodic.hyp_list.get_values(), values)
        assert np.allclose(
            kernel_periodic.diag(X), kernel_periodic._la_get_diagonal(
                kernel_periodic(X, X)))

    def test_kernels(self):
        test_cases = [
            [NumpyMaternKernel, NumpyConstantKernel,
             NumpyPeriodicMaternKernel],
            [TorchMaternKernel, TorchConstantKernel,
             TorchPeriodicMaternKernel]]
        for case in test_cases:
            self._check_kernels(*case)

    def check_kernel_jacobian(self, torch_kernel, np_kernel, nsamples):
        X = np.random.uniform(-1, 1, (torch_kernel.nvars(), nsamples))
        torch_jacobian = torch_kernel.jacobian(torch_kernel._la_atleast2d(X))
        for hyp in torch_kernel.hyp_list.hyper_params:
            hyp._values = hyp._values.clone().detach()

        def fun(active_params_opt):
            np_kernel.hyp_list.set_active_opt_params(active_params_opt)
            return np_kernel(X)
        assert np.allclose(
            torch_jacobian.numpy(),
            approx_jacobian_3D(
                fun, np_kernel.hyp_list.get_active_opt_params()))

    def test_kernel_jacobian(self):
        nvars, nsamples = 2, 3
        torch_kernel = TorchMaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
        np_kernel = NumpyMaternKernel(
            np.inf, 1.0, [1e-1, 1], nvars)
        self.check_kernel_jacobian(torch_kernel, np_kernel, nsamples)

        const = 1
        torch_kernel = (TorchConstantKernel(const) *
                        TorchMaternKernel(np.inf, 1.0, [1e-1, 1], nvars))
        np_kernel = (
            NumpyConstantKernel(const) *
            NumpyMaternKernel(np.inf, 1.0, [1e-1, 1], nvars))
        self.check_kernel_jacobian(torch_kernel, np_kernel, nsamples)

        const = 1
        torch_kernel = (
            TorchMaternKernel(np.inf, 1.0, [1e-1, 1], nvars) +
            TorchGaussianNoiseKernel(1, [1e-2, 10]))
        np_kernel = (
            NumpyMaternKernel(
                np.inf, 1.0, [1e-1, 1], nvars) +
            NumpyGaussianNoiseKernel(1, [1e-2, 10]))
        self.check_kernel_jacobian(torch_kernel, np_kernel, nsamples)


if __name__ == "__main__":
    kernels_test_suite = unittest.TestLoader().loadTestsFromTestCase(
        TestKernels)
    unittest.TextTestRunner(verbosity=2).run(kernels_test_suite)
