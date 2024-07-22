import unittest
import copy

import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixin
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.kernels.numpykernels import (
    NumpyConstantKernel, NumpyMaternKernel, NumpyPeriodicMaternKernel,
    NumpyGaussianNoiseKernel)
from pyapprox.surrogates.kernels.torchkernels import (
    TorchMaternKernel, TorchPeriodicMaternKernel,
    TorchConstantKernel, TorchGaussianNoiseKernel)


def approx_jacobian_3D(f, x0, epsilon=np.sqrt(np.finfo(float).eps),
                       backend=NumpyLinAlgMixin()):
    fval = f(x0)
    jacobian = backend._la_full(
        (fval.shape[0], fval.shape[1], x0.shape[0]), 0.)
    for ii in range(len(x0)):
        dx = backend._la_full((x0.shape[0],), 0.)
        dx[ii] = epsilon
        fval_perturbed = f(x0+dx)
        jacobian[..., ii] = (fval_perturbed - fval) / epsilon
    return jacobian


class TestKernels:
    def setUp(self):
        np.random.seed(1)

    def test_kernels(self):
        (MaternKernel, ConstantKernel, GaussianNoiseKernel,
         PeriodicMaternKernel) = self.get_kernels()
        kernel_inf = MaternKernel(np.inf, 1.0, [1e-1, 1], 2)
        values = kernel_inf._la_atleast1d([0.5, 0.5])
        kernel_inf.hyp_list.set_active_opt_params(kernel_inf._la_log(values))
        assert self._la_allclose(kernel_inf.hyp_list.get_values(), values)

        nsamples1, nsamples2 = 5, 3
        X = self._la_array(np.random.normal(0, 1, (2, nsamples1)))
        Y = self._la_array(np.random.normal(0, 1, (2, nsamples2)))
        assert self._la_allclose(
            kernel_inf.diag(X), kernel_inf._la_get_diagonal(kernel_inf(X, X)))

        const0 = 2.0
        kernel_prod = kernel_inf*ConstantKernel(const0)
        assert self._la_allclose(
            kernel_prod.diag(X), const0*kernel_inf.diag(X))
        assert self._la_allclose(
            kernel_prod.diag(X),
            kernel_inf._la_get_diagonal(kernel_prod(X, X)))
        assert self._la_allclose(kernel_prod(X, Y), const0*kernel_inf(X, Y))

        const1 = 3.0
        kernel_sum = kernel_prod+ConstantKernel(const1)
        assert self._la_allclose(
            kernel_sum.diag(X), const0*kernel_inf.diag(X)+const1)
        assert self._la_allclose(
            kernel_sum.diag(X), kernel_prod._la_get_diagonal(kernel_sum(X, X)))
        assert self._la_allclose(
            kernel_sum(X, Y), const0*kernel_inf(X, Y)+const1)

        kernel_periodic = PeriodicMaternKernel(
            0.5, 1.0, [1e-1, 1], 1, [1e-1, 1])
        values = kernel_periodic._la_atleast1d([0.5, 0.5])
        kernel_periodic.hyp_list.set_active_opt_params(
            kernel_periodic._la_log(values))
        assert self._la_allclose(kernel_periodic.hyp_list.get_values(), values)
        assert self._la_allclose(
            kernel_periodic.diag(X), kernel_periodic._la_get_diagonal(
                kernel_periodic(X, X)))

    def _check_kernel_jacobian(self, kernel, nsamples):
        kernel_copy = copy.deepcopy(kernel)
        X = self._la_array(
            np.random.uniform(-1, 1, (kernel.nvars(), nsamples)))
        jacobian = kernel.jacobian(X)
        for hyp in kernel.hyp_list.hyper_params:
            hyp._values = hyp._values.clone().detach()

        def fun(active_params_opt):
            kernel_copy.hyp_list.set_active_opt_params(active_params_opt)
            return kernel_copy(X)
        assert self._la_allclose(
            jacobian,
            approx_jacobian_3D(
                fun, kernel_copy.hyp_list.get_active_opt_params(),
                backend=self))

    def test_kernel_jacobian(self):
        (MaternKernel, ConstantKernel, GaussianNoiseKernel,
         PeriodicMaternKernel) = self.get_kernels()
        if not self.jacobian_implemented():
            return
        nvars, nsamples = 2, 3
        kernel = MaternKernel(np.inf, 1.0, [1e-1, 1], nvars)
        self._check_kernel_jacobian(kernel, nsamples)

        const = 1
        kernel = (ConstantKernel(const) *
                  MaternKernel(np.inf, 1.0, [1e-1, 1], nvars))
        self._check_kernel_jacobian(kernel, nsamples)
        const = 1
        kernel = (
            MaternKernel(np.inf, 1.0, [1e-1, 1], nvars) +
            GaussianNoiseKernel(1, [1e-2, 10]))
        self._check_kernel_jacobian(kernel, nsamples)


class TestNumpyKernels(
        unittest.TestCase, TestKernels, NumpyLinAlgMixin):
    def get_kernels(self):
        return (NumpyMaternKernel, NumpyConstantKernel,
                NumpyGaussianNoiseKernel, NumpyPeriodicMaternKernel)

    def jacobian_implemented(self):
        return False


class TestTorchKernels(
        unittest.TestCase, TestKernels, TorchLinAlgMixin):
    def get_kernels(self):
        return (TorchMaternKernel, TorchConstantKernel,
                TorchGaussianNoiseKernel, TorchPeriodicMaternKernel)

    def jacobian_implemented(self):
        return True


if __name__ == "__main__":
    unittest.main()
