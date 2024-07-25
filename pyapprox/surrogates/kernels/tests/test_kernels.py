import unittest
import copy

import numpy as np

from pyapprox.util.linearalgebra.numpylinalg import NumpyLinAlgMixinp
from pyapprox.util.linearalgebra.torchlinalg import TorchLinAlgMixin
from pyapprox.surrogates.kernels import (
    ConstantKernel,
    MaternKernel,
    PeriodicMaternKernel,
    GaussianNoiseKernel,
)
from pyapprox.util.utilities import approx_jacobian_3D
from pyapprox.util.sys_utilities import package_available

if package_available("jax"):
    from pyapprox.util.linearalgebra.jaxlinalg import JaxLinAlgMixin

import warnings
warnings.filterwarnings("error")


class TestKernels:
    def setUp(self):
        np.random.seed(1)

    def test_kernels(self):
        bkd = self.get_backend()
        kernel_inf = MaternKernel(np.inf, 1.0, [1e-1, 1], 2, backend=bkd)
        values = bkd._la_atleast1d([0.5, 0.5])
        kernel_inf.hyp_list.set_active_opt_params(bkd._la_log(values))
        assert bkd._la_allclose(kernel_inf.hyp_list.get_values(), values)

        nsamples1, nsamples2 = 5, 3
        X = bkd._la_array(np.random.normal(0, 1, (2, nsamples1)))
        Y = bkd._la_array(np.random.normal(0, 1, (2, nsamples2)))
        assert bkd._la_allclose(
            kernel_inf.diag(X), bkd._la_get_diagonal(kernel_inf(X, X))
        )

        const0 = 2.0
        kernel_prod = kernel_inf * ConstantKernel(const0, backend=bkd)
        assert bkd._la_allclose(kernel_prod.diag(X), const0 * kernel_inf.diag(X))
        assert bkd._la_allclose(
            kernel_prod.diag(X), bkd._la_get_diagonal(kernel_prod(X, X))
        )
        assert bkd._la_allclose(kernel_prod(X, Y), const0 * kernel_inf(X, Y))

        const1 = 3.0
        kernel_sum = kernel_prod + ConstantKernel(const1, backend=bkd)
        assert bkd._la_allclose(
            kernel_sum.diag(X), const0 * kernel_inf.diag(X) + const1
        )
        assert bkd._la_allclose(
            kernel_sum.diag(X), bkd._la_get_diagonal(kernel_sum(X, X))
        )
        assert bkd._la_allclose(kernel_sum(X, Y), const0 * kernel_inf(X, Y) + const1)

        kernel_periodic = PeriodicMaternKernel(
            0.5, 1.0, [1e-1, 1], 1, [1e-1, 1], backend=bkd
        )
        values = bkd._la_atleast1d([0.5, 0.5])
        kernel_periodic.hyp_list.set_active_opt_params(bkd._la_log(values))
        assert bkd._la_allclose(kernel_periodic.hyp_list.get_values(), values)
        assert bkd._la_allclose(
            kernel_periodic.diag(X), bkd._la_get_diagonal(kernel_periodic(X, X))
        )

    def _check_kernel_jacobian(self, kernel, nsamples):
        bkd = kernel._bkd
        kernel_copy = copy.deepcopy(kernel)
        X = bkd._la_array(np.random.uniform(-1, 1, (kernel.nvars(), nsamples)))
        jacobian = kernel.jacobian(X)
        # The following loop prevents torch from throwing error
        # RuntimeError: Only Tensors created explicitly by the user...
        for hyp in kernel.hyp_list.hyper_params:
            hyp._values = bkd._la_copy(bkd._la_detach(hyp._values))

        def fun(active_params_opt):
            kernel_copy.hyp_list.set_active_opt_params(active_params_opt)
            return kernel_copy(X)
        assert bkd._la_allclose(
            jacobian,
            approx_jacobian_3D(
                fun, kernel_copy.hyp_list.get_active_opt_params(), bkd=bkd
            ),
        )

    def test_kernel_jacobian(self):
        if not self.jacobian_implemented():
            self.skipTest("Jacobian not implemented")
        bkd = self.get_backend()
        nvars, nsamples = 2, 3
        kernel = MaternKernel(np.inf, 1.0, [1e-1, 1], nvars, backend=bkd)
        self._check_kernel_jacobian(kernel, nsamples)

        const = 1
        kernel = ConstantKernel(const, backend=bkd) * MaternKernel(
            np.inf, 1.0, [1e-1, 1], nvars, backend=bkd
        )
        self._check_kernel_jacobian(kernel, nsamples)
        const = 1
        kernel = MaternKernel(
            np.inf, 1.0, [1e-1, 1], nvars, backend=bkd
        ) + GaussianNoiseKernel(1, [1e-2, 10], backend=bkd)
        self._check_kernel_jacobian(kernel, nsamples)


class TestNumpyKernels(TestKernels, unittest.TestCase):
    def get_backend(self):
        return NumpyLinAlgMixin()

    def jacobian_implemented(self):
        return False


class TestTorchKernels(TestKernels, unittest.TestCase):
    def setUp(self):
        if not package_available("torch"):
            self.skipTest("torch not available")
        TestKernels.setUp(self)

    def get_backend(self):
        return TorchLinAlgMixin()

    def jacobian_implemented(self):
        return True


class TestJaxKernels(TestKernels, unittest.TestCase):
    def setUp(self):
        if not package_available("jax"):
            self.skipTest("jax not available")
        TestKernels.setUp(self)

    def get_backend(self):
        return JaxLinAlgMixin()

    def jacobian_implemented(self):
        return True


if __name__ == "__main__":
    unittest.main(verbosity=2)
