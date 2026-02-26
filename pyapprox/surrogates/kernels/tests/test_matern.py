import numpy as np
import pytest

from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.hessian import (
    FunctionWithJacobianAndHVPFromCallable,
    FunctionWithJacobianFromCallable,
)
from pyapprox.surrogates.kernels.matern import (
    ExponentialKernel,
    Matern32Kernel,
    Matern52Kernel,
    SquaredExponentialKernel,
)


def create_matern_kernel(nu, lenscale, lenscale_bounds, nvars, bkd, fixed=False):
    """Helper to create appropriate Matern kernel based on nu value."""
    if nu == np.inf:
        return SquaredExponentialKernel(lenscale, lenscale_bounds, nvars, bkd, fixed)
    elif nu == 2.5 or nu == 5 / 2:
        return Matern52Kernel(lenscale, lenscale_bounds, nvars, bkd, fixed)
    elif nu == 1.5 or nu == 3 / 2:
        return Matern32Kernel(lenscale, lenscale_bounds, nvars, bkd, fixed)
    elif nu == 0.5 or nu == 1 / 2:
        return ExponentialKernel(lenscale, lenscale_bounds, nvars, bkd, fixed)
    else:
        raise ValueError(f"Unsupported nu value: {nu}")


class TestMaternKernel:

    def _setup_data(self, bkd):
        np.random.seed(1)
        self.nu_values = [np.inf, 5 / 2, 3 / 2, 1 / 2]
        # Separate test points for nu=0.5 (exponential kernel) to avoid
        # coincident X1/X2 points where the gradient is undefined
        self.X1_exp = bkd.array([[0.1, 0.4, 0.9], [0.15, 0.35, 0.65]])
        self.X2_exp = bkd.array([[0.0, 0.5, 1.0], [0.0, 0.25, 0.75]])
        self.lenscale = bkd.array([1.0, 2.0])  # Vector-valued length scale
        self.lenscale_bounds = (0.1, 10.0)
        self.nvars = 2
        self.X1 = bkd.array([[0.0, 0.5, 1.0], [0.0, 0.25, 0.75]])
        self.X2 = bkd.array([[0.0, 0.5, 1.0], [0.0, 0.25, 0.75]])

    def test_kernel_matrix(self, bkd) -> None:
        """
        Test the computation of the kernel matrix for supported nu values.
        """
        self._setup_data(bkd)
        for nu in self.nu_values:
            kernel = create_matern_kernel(
                nu=nu,
                lenscale=self.lenscale,
                lenscale_bounds=self.lenscale_bounds,
                nvars=self.nvars,
                bkd=bkd,
            )
            kernel_matrix = kernel(self.X1, self.X2)
            assert kernel_matrix.shape == (self.X1.shape[1], self.X2.shape[1])
            assert bkd.all_bool(
                kernel_matrix >= 0
            )  # Kernel values must be non-negative

    def test_diagonal(self, bkd) -> None:
        """
        Test the computation of the diagonal of the kernel matrix.
        """
        self._setup_data(bkd)
        for nu in self.nu_values:
            kernel = create_matern_kernel(
                nu=nu,
                lenscale=self.lenscale,
                lenscale_bounds=self.lenscale_bounds,
                nvars=self.nvars,
                bkd=bkd,
            )
            diag = kernel.diag(self.X1)
            assert diag.shape == (self.X1.shape[1],)
            bkd.assert_allclose(diag, bkd.ones(self.X1.shape[1]))

    def test_jacobian(self, bkd) -> None:
        """
        Test the computation of the Jacobian of the kernel with respect to
        input data.
        """
        self._setup_data(bkd)
        for nu in self.nu_values:
            kernel = create_matern_kernel(
                nu=nu,
                lenscale=self.lenscale,
                lenscale_bounds=self.lenscale_bounds,
                nvars=self.nvars,
                bkd=bkd,
            )
            # Use non-coincident points for nu=0.5 (exponential kernel)
            # because the gradient is undefined at r=0 (kink)
            if nu == 0.5 or nu == 1 / 2:
                X1 = self.X1_exp
                X2 = self.X2_exp
            else:
                X1 = self.X1
                X2 = self.X2
            jacobian = kernel.jacobian(X1, X2)
            assert jacobian.shape == (X1.shape[1], X2.shape[1], self.nvars)
            assert bkd.all_bool(
                bkd.isfinite(jacobian)
            )  # Jacobian values must be finite

            # Wrap the function using FunctionWithJacobianFromCallable
            vec = bkd.ones((X2.shape[1], 1))
            function_object = FunctionWithJacobianFromCallable(
                nqoi=1,  # just computing jacobian for one sample
                nvars=self.nvars,
                fun=lambda x, _X2=X2: kernel(
                    bkd.reshape(x, (self.nvars, -1)), _X2
                )
                @ vec,
                jacobian=lambda x, _X2=X2: bkd.einsum(
                    "ijk,jl->ik",
                    kernel.jacobian(bkd.reshape(x, (self.nvars, -1)), _X2),
                    vec,
                ),
                bkd=bkd,
            )
            # Initialize DerivativeChecker
            checker = DerivativeChecker(function_object)
            # Check derivatives
            sample = bkd.flatten(X1[:, :1])[:, None]
            errors = checker.check_derivatives(sample)

            # Assert that the gradient errors are below a tolerance
            assert checker.error_ratio(errors[0]) <= 1e-6

    def test_param_jacobian(self, bkd) -> None:
        """
        Test the computation of the Jacobian of the kernel with respect to
        hyperparameters.
        """
        self._setup_data(bkd)
        for nu in self.nu_values:
            kernel = create_matern_kernel(
                nu=nu,
                lenscale=self.lenscale,
                lenscale_bounds=self.lenscale_bounds,
                nvars=self.nvars,
                bkd=bkd,
            )
            jacobian_params = kernel.jacobian_wrt_params(self.X1)
            assert jacobian_params.shape == (
                self.X1.shape[1],
                self.X1.shape[1],
                self.nvars,
            )
            assert bkd.all_bool(
                bkd.isfinite(jacobian_params)
            )  # Jacobian values must be finite

            # Wrap the function using FunctionWithJacobianFromCallable
            vec = bkd.ones((self.X1.shape[1], 1))

            def fun(p):
                kernel.hyp_list().hyperparameters()[0].set_active_values(p[:, 0])
                return kernel(self.X1, self.X1) @ vec

            def jac(p):
                kernel.hyp_list().hyperparameters()[0].set_active_values(p[:, 0])
                return bkd.einsum(
                    "ijk,jl->ik",
                    kernel.jacobian_wrt_params(self.X1),
                    vec,
                )

            function_object = FunctionWithJacobianFromCallable(
                nqoi=self.X1.shape[1],
                nvars=self.nvars,
                fun=fun,
                jacobian=jac,
                bkd=bkd,
            )
            # Initialize DerivativeChecker
            checker = DerivativeChecker(function_object)
            # Check derivatives
            sample = kernel.hyp_list().hyperparameters()[0].get_active_values()[:, None]
            errors = checker.check_derivatives(sample, verbosity=0)
            # Assert that the gradient errors are below a tolerance
            assert checker.error_ratio(errors[0]) < 5e-6

    def test_repr(self, bkd) -> None:
        """
        Test the string representation of the MaternKernel.
        """
        self._setup_data(bkd)
        kernel = create_matern_kernel(
            nu=3 / 2,
            lenscale=self.lenscale,
            lenscale_bounds=self.lenscale_bounds,
            nvars=self.nvars,
            bkd=bkd,
        )
        repr_str = repr(kernel)
        # Check for specific class name (Matern32Kernel for nu=3/2)
        assert "Matern32Kernel" in repr_str
        assert "lenscale" in repr_str

    def test_hvp_wrt_x1(self, bkd) -> None:
        """
        Test HVP (Hessian-vector product) using derivative checker.

        Tests that the HVP implementation matches finite difference
        approximations for all supported Matern smoothness parameters.
        """
        self._setup_data(bkd)
        for ii, nu in enumerate(self.nu_values):
            np.random.seed(42 + ii)
            kernel = create_matern_kernel(
                nu=nu,
                lenscale=self.lenscale,
                lenscale_bounds=self.lenscale_bounds,
                nvars=self.nvars,
                bkd=bkd,
            )

            # Use non-coincident points for nu=0.5 (exponential kernel)
            # because the gradient is undefined at r=0 (kink)
            if nu == 0.5 or nu == 1 / 2:
                X1 = self.X1_exp
                X2 = self.X2_exp
            else:
                X1 = self.X1
                X2 = self.X2

            # Test point and direction
            x_test = X1[:, :1]  # Shape (nvars, 1)
            direction = bkd.array(np.random.randn(self.nvars, 1))
            direction = direction / bkd.norm(direction)
            direction_flat = bkd.flatten(direction)  # Shape (nvars,)

            # Compute kernel value and derivatives for verification
            K = kernel(x_test, X2)  # Shape (1, n2)
            jac = kernel.jacobian(x_test, X2)  # Shape (1, n2, nvars)
            hvp = kernel.hvp_wrt_x1(
                x_test, X2, direction_flat
            )  # Shape (1, n2, nvars)

            # Verify shapes
            assert K.shape == (1, X2.shape[1])
            assert jac.shape == (1, X2.shape[1], self.nvars)
            assert hvp.shape == (1, X2.shape[1], self.nvars)

            # Verify HVP using derivative checker
            # We need a scalar function for the checker, so we contract with a
            # vector
            vec = bkd.ones((X2.shape[1], 1))

            def kernel_func(x_shaped, _X2=X2):
                """Kernel function contracted with vec: R^nvars -> R."""
                x_reshaped = bkd.reshape(x_shaped, (self.nvars, 1))
                K_val = kernel(x_reshaped, _X2)  # Shape (1, n2)
                return K_val @ vec  # Shape (1, 1)

            def jacobian_func(x_shaped, _X2=X2):
                """Jacobian contracted with vec: R^nvars -> R^nvars."""
                x_reshaped = bkd.reshape(x_shaped, (self.nvars, 1))
                jac_val = kernel.jacobian(x_reshaped, _X2)  # Shape (1, n2, nvars)
                # Contract: (1, n2, nvars) with (n2, 1) -> (1, nvars)
                result = bkd.einsum("ijk,jl->ik", jac_val, vec)
                return result  # Shape (1, nvars)

            def hvp_func(x_shaped, v_shaped, _X2=X2):
                """HVP contracted with vec: R^nvars x R^nvars -> R^nvars."""
                x_reshaped = bkd.reshape(x_shaped, (self.nvars, 1))
                v_flat = bkd.flatten(v_shaped)  # Shape (nvars,)
                hvp_val = kernel.hvp_wrt_x1(
                    x_reshaped, _X2, v_flat
                )  # Shape (1, n2, nvars)
                # Contract: (1, n2, nvars) with (n2, 1) -> (nvars, 1)
                # einsum 'ijk,jl->ki' gives (nvars, 1) not (1, nvars)
                result = bkd.einsum("ijk,jl->ki", hvp_val, vec)
                return result  # Shape (nvars, 1)

            # Create function object with HVP
            function_with_hvp = FunctionWithJacobianAndHVPFromCallable(
                nvars=self.nvars,
                fun=kernel_func,
                jacobian=jacobian_func,
                hvp=hvp_func,
                bkd=bkd,
            )

            # Create derivative checker
            checker = DerivativeChecker(function_with_hvp)

            # Check derivatives at test point
            sample = bkd.flatten(x_test)[:, None]  # Shape (nvars, 1)
            errors = checker.check_derivatives(
                sample, direction=direction, verbosity=0
            )

            # Verify Jacobian is correct
            jac_error = errors[0]
            assert bkd.all_bool(bkd.isfinite(jac_error))
            jac_ratio = float(checker.error_ratio(jac_error))
            assert (
                jac_ratio < 1e-6
            ), f"Jacobian error ratio for nu={nu}: {jac_ratio}"

            # Verify HVP is correct
            hvp_error = errors[1]
            assert bkd.all_bool(bkd.isfinite(hvp_error))
            hvp_ratio = float(checker.error_ratio(hvp_error))
            assert (
                hvp_ratio < 5e-5
            ), f"HVP error ratio for nu={nu}: {hvp_ratio}"
