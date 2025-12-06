import unittest
from typing import Generic, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pyapprox.typing.util.backends.protocols import Backend, Array
from pyapprox.typing.util.backends.numpy import NumpyBkd
from pyapprox.typing.util.backends.torch import TorchBkd
from pyapprox.typing.surrogates.kernels.matern import (
    Matern32Kernel,
    Matern52Kernel,
)
# from pyapprox.typing.surrogates.kernels.constant import ConstantKernel  # Not yet implemented
from pyapprox.typing.surrogates.kernels.multioutput.independent import (
    IndependentMultiOutputKernel
)
from pyapprox.typing.surrogates.kernels.multioutput.linear_coregionalization import (
    LinearCoregionalizationKernel
)


class TestIndependentMultiOutputKernel(Generic[Array], unittest.TestCase):
    """
    Base test class for IndependentMultiOutputKernel.

    Derived classes must implement the bkd() method to provide the backend.
    """

    def setUp(self) -> None:
        """
        Set up test environment for IndependentMultiOutputKernel.
        """
        np.random.seed(42)
        self.nvars = 2
        self.noutputs = 3
        self.nsamples = [10, 5, 8]  # Different number of samples per output

        # Create independent kernels for each output
        self.kernels = []
        for i in range(self.noutputs):
            kernel = Matern52Kernel(
                [1.0] * self.nvars,
                (0.1, 10.0),
                self.nvars,
                self.bkd()
            )
            self.kernels.append(kernel)

        self.mo_kernel = IndependentMultiOutputKernel(self.kernels)

        # Create sample data for each output
        self.X_list = [
            self.bkd().array(np.random.randn(self.nvars, self.nsamples[i]))
            for i in range(self.noutputs)
        ]
        self.X2_list = [
            self.bkd().array(np.random.randn(self.nvars, self.nsamples[i]))
            for i in range(self.noutputs)
        ]

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_initialization(self) -> None:
        """
        Test IndependentMultiOutputKernel initialization.
        """
        mo_kernel = IndependentMultiOutputKernel(self.kernels)
        self.assertEqual(mo_kernel.noutputs(), self.noutputs)
        self.assertEqual(mo_kernel.nvars(), self.nvars)
        self.assertIsNotNone(mo_kernel.bkd())

    def test_empty_kernels_error(self) -> None:
        """
        Test that empty kernels list raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            IndependentMultiOutputKernel([])
        self.assertIn("cannot be empty", str(context.exception))

    def test_backend_mismatch_error(self) -> None:
        """
        Test that mixed backends raise ValueError.
        """
        # Create kernel with different backend
        if isinstance(self.bkd(), NumpyBkd):
            other_bkd = TorchBkd()
        else:
            other_bkd = NumpyBkd()

        kernel_other = Matern52Kernel(
            [1.0] * self.nvars,
            (0.1, 10.0),
            self.nvars,
            other_bkd
        )

        with self.assertRaises(ValueError) as context:
            IndependentMultiOutputKernel([self.kernels[0], kernel_other])
        self.assertIn("same backend type", str(context.exception))

    def test_nvars_mismatch_error(self) -> None:
        """
        Test that kernels with different nvars raise ValueError.
        """
        kernel_different = Matern52Kernel(
            [1.0, 1.0, 1.0],  # 3 dimensions instead of 2
            (0.1, 10.0),
            3,  # Different nvars
            self.bkd()
        )

        with self.assertRaises(ValueError) as context:
            IndependentMultiOutputKernel([self.kernels[0], kernel_different])
        self.assertIn("same number of input variables", str(context.exception))

    def test_hyperparameter_list_combination(self) -> None:
        """
        Test that hyperparameter lists are combined correctly.
        """
        total_params = sum(k.hyp_list().nparams() for k in self.kernels)
        self.assertEqual(self.mo_kernel.hyp_list().nparams(), total_params)

    def test_stacked_kernel_matrix_shape(self) -> None:
        """
        Test that stacked kernel matrix has correct shape.
        """
        K = self.mo_kernel(self.X_list, block_format=False)

        n_total = sum(self.nsamples)
        self.assertEqual(K.shape, (n_total, n_total))

    def test_stacked_kernel_matrix_cross_shape(self) -> None:
        """
        Test that cross-covariance has correct shape.
        """
        K = self.mo_kernel(self.X_list, self.X2_list, block_format=False)

        n_total_1 = sum(self.nsamples)
        n_total_2 = sum(self.nsamples)
        self.assertEqual(K.shape, (n_total_1, n_total_2))

    def test_stacked_kernel_is_block_diagonal(self) -> None:
        """
        Test that stacked kernel matrix is block-diagonal.
        """
        K = self.mo_kernel(self.X_list, block_format=False)

        # Extract diagonal blocks and verify off-diagonal blocks are zero
        row_offset = 0
        for i in range(self.noutputs):
            n_i = self.nsamples[i]

            # Extract diagonal block
            K_ii = K[row_offset:row_offset+n_i, row_offset:row_offset+n_i]

            # Compute expected diagonal block
            K_expected = self.kernels[i](self.X_list[i], self.X_list[i])
            self.bkd().assert_allclose(K_ii, K_expected)

            # Check off-diagonal blocks are zero
            col_offset = 0
            for j in range(self.noutputs):
                n_j = self.nsamples[j]
                if i != j:
                    K_ij = K[row_offset:row_offset+n_i, col_offset:col_offset+n_j]

                    # Should be all zeros
                    expected_zero = self.bkd().zeros((n_i, n_j))
                    self.bkd().assert_allclose(K_ij, expected_zero)

                col_offset += n_j

            row_offset += n_i

    def test_block_format_structure(self) -> None:
        """
        Test that block format returns correct structure.
        """
        K_blocks = self.mo_kernel(self.X_list, block_format=True)

        # Should be list of lists
        self.assertIsInstance(K_blocks, list)
        self.assertEqual(len(K_blocks), self.noutputs)

        for i in range(self.noutputs):
            self.assertIsInstance(K_blocks[i], list)
            self.assertEqual(len(K_blocks[i]), self.noutputs)

    def test_block_format_diagonal_blocks(self) -> None:
        """
        Test that diagonal blocks in block format are correct.
        """
        K_blocks = self.mo_kernel(self.X_list, block_format=True)

        for i in range(self.noutputs):
            # Diagonal block should not be None
            self.assertIsNotNone(K_blocks[i][i])

            # Should match kernel evaluation
            K_expected = self.kernels[i](self.X_list[i], self.X_list[i])
            self.bkd().assert_allclose(K_blocks[i][i], K_expected)

    def test_block_format_off_diagonal_blocks(self) -> None:
        """
        Test that off-diagonal blocks in block format are None.
        """
        K_blocks = self.mo_kernel(self.X_list, block_format=True)

        for i in range(self.noutputs):
            for j in range(self.noutputs):
                if i != j:
                    # Off-diagonal blocks should be None
                    self.assertIsNone(K_blocks[i][j])

    def test_wrong_length_X_list_error(self) -> None:
        """
        Test that wrong length X_list raises ValueError.
        """
        wrong_list = self.X_list[:2]  # Too short

        with self.assertRaises(ValueError) as context:
            self.mo_kernel(wrong_list)
        self.assertIn("must have", str(context.exception))

    def test_wrong_length_X2_list_error(self) -> None:
        """
        Test that wrong length X2_list raises ValueError.
        """
        wrong_list = self.X2_list[:2]  # Too short

        with self.assertRaises(ValueError) as context:
            self.mo_kernel(self.X_list, wrong_list)
        self.assertIn("must have", str(context.exception))

    def test_jacobian_wrt_params_shape(self) -> None:
        """
        Test that parameter Jacobian has correct shape.
        """
        jac = self.mo_kernel.jacobian_wrt_params(self.X_list)

        n_total = sum(self.nsamples)
        nparams_total = self.mo_kernel.hyp_list().nparams()

        self.assertEqual(jac.shape, (n_total, n_total, nparams_total))

    def test_jacobian_wrt_params_is_block_diagonal(self) -> None:
        """
        Test that parameter Jacobian has block-diagonal structure.
        """
        jac = self.mo_kernel.jacobian_wrt_params(self.X_list)

        # Verify diagonal blocks match individual kernel Jacobians
        row_offset = 0
        param_offset = 0
        for i in range(self.noutputs):
            n_i = self.nsamples[i]
            jac_i_expected = self.kernels[i].jacobian_wrt_params(self.X_list[i])
            nparams_i = jac_i_expected.shape[2]

            # Extract diagonal block
            jac_ii = jac[
                row_offset:row_offset+n_i,
                row_offset:row_offset+n_i,
                param_offset:param_offset+nparams_i
            ]

            self.bkd().assert_allclose(jac_ii, jac_i_expected)

            row_offset += n_i
            param_offset += nparams_i

    def test_repr(self) -> None:
        """
        Test string representation.
        """
        repr_str = repr(self.mo_kernel)
        self.assertIn("IndependentMultiOutputKernel", repr_str)
        self.assertIn(f"noutputs={self.noutputs}", repr_str)

    def test_with_different_kernel_types(self) -> None:
        """
        Test with different kernel types for different outputs.
        """
        # Mix Matern and Constant kernels
        # kernels = [
        #     MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), self.nvars, self.bkd()),
        #     ConstantKernel(2.0, (0.1, 10.0), self.bkd()),  # Not yet implemented
        # ]

        # Note: ConstantKernel has nvars=0, so this should raise an error
        # Let's create a test with compatible kernels instead
        kernels_compat = [
            Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, self.bkd()),
            Matern32Kernel([0.5, 0.5], (0.1, 10.0), self.nvars, self.bkd()),
        ]

        mo_kernel = IndependentMultiOutputKernel(kernels_compat)
        X_list_2 = self.X_list[:2]
        K = mo_kernel(X_list_2)

        # Verify shape
        n_total = self.nsamples[0] + self.nsamples[1]
        self.assertEqual(K.shape, (n_total, n_total))

    def test_single_output(self) -> None:
        """
        Test with single output (edge case).
        """
        mo_kernel = IndependentMultiOutputKernel([self.kernels[0]])
        X_list = [self.X_list[0]]

        K = mo_kernel(X_list)
        K_expected = self.kernels[0](self.X_list[0], self.X_list[0])

        self.bkd().assert_allclose(K, K_expected)

    def test_jacobian_wrt_params_derivative_checker(self) -> None:
        """
        Test parameter Jacobian using finite differences (DerivativeChecker).

        This validates that the analytical Jacobian matches numerical
        finite difference approximation.
        """
        from pyapprox.typing.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker
        )

        # Create wrapper class for DerivativeChecker
        class KernelWrapper:
            """Wrapper to make kernel compatible with DerivativeChecker."""

            def __init__(self, kernel, X_list, bkd):
                self._kernel = kernel
                self._X_list = X_list
                self._bkd = bkd

                # Compute total size
                self._n_total = sum(X.shape[1] for X in X_list)

            def bkd(self):
                return self._bkd

            def nvars(self):
                """Number of parameters being optimized."""
                return self._kernel.hyp_list().nactive_params()

            def nqoi(self):
                """Number of outputs - n_total^2 for flattened kernel matrix."""
                return self._n_total * self._n_total

            def __call__(self, params: Array) -> Array:
                """
                Evaluate kernel matrix with given hyperparameters.

                Parameters
                ----------
                params : Array, shape (nparams,) or (nparams, 1)
                    Hyperparameters.

                Returns
                -------
                Array, shape (n_total*n_total, 1)
                    Flattened kernel matrix.
                """
                # Flatten params if needed (DerivativeChecker may pass (nparams, 1))
                params_flat = self._bkd.reshape(params, (-1,))

                # Update hyperparameters
                self._kernel.hyp_list().set_active_values(params_flat)

                # Compute kernel matrix
                K = self._kernel(self._X_list, self._X_list, block_format=False)

                # Return flattened and reshaped
                K_flat = self._bkd.reshape(K, (self._n_total * self._n_total, 1))
                return K_flat

            def jacobian(self, params: Array) -> Array:
                """
                Compute Jacobian of kernel matrix w.r.t. hyperparameters.

                Parameters
                ----------
                params : Array, shape (nparams,) or (nparams, 1)
                    Hyperparameters.

                Returns
                -------
                Array, shape (n_total*n_total, nparams)
                    Jacobian matrix.
                """
                # Flatten params if needed
                params_flat = self._bkd.reshape(params, (-1,))

                # Update hyperparameters
                self._kernel.hyp_list().set_active_values(params_flat)

                # Compute parameter Jacobian
                jac = self._kernel.jacobian_wrt_params(self._X_list)
                # jac shape: (n_total, n_total, nparams)

                # Reshape to (n_total*n_total, nparams)
                nparams = jac.shape[2]
                jac_reshaped = self._bkd.reshape(jac, (self._n_total * self._n_total, nparams))

                return jac_reshaped

            def jvp(self, params: Array, v: Array) -> Array:
                """
                Compute Jacobian-vector product.

                Parameters
                ----------
                params : Array, shape (nparams,) or (nparams, 1)
                    Hyperparameters.
                v : Array, shape (nparams,) or (nparams, 1)
                    Vector for JVP.

                Returns
                -------
                Array, shape (n_total*n_total, 1)
                    JVP result.
                """
                # Compute Jacobian
                J = self.jacobian(params)  # (n_total*n_total, nparams)

                # Flatten v if needed
                v_flat = self._bkd.reshape(v, (-1, 1))  # (nparams, 1)

                # JVP = J @ v
                return J @ v_flat

        # Create wrapper
        wrapper = KernelWrapper(self.mo_kernel, self.X_list, self.bkd())

        # Create derivative checker
        checker = DerivativeChecker(wrapper)

        # Get current hyperparameters
        params = self.mo_kernel.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes
        fd_eps = self.bkd().flip(self.bkd().logspace(-14, 0, 15))

        # Check gradient accuracy
        errors = checker.check_derivatives(
            params[:, None],  # Shape: (nactive, 1)
            fd_eps=fd_eps,
            relative=True,
            verbosity=0
        )

        # Get gradient error
        grad_error = errors[0]

        # Minimum error should be small
        min_error = float(self.bkd().min(grad_error))
        self.assertLess(min_error, 1e-6,
                       f"Minimum gradient relative error {min_error} exceeds threshold")


# NumPy implementation
class TestIndependentMultiOutputKernelNumpy(
    TestIndependentMultiOutputKernel[NDArray[Any]]
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch implementation
class TestIndependentMultiOutputKernelTorch(
    TestIndependentMultiOutputKernel[torch.Tensor]
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestLinearCoregionalizationKernel(Generic[Array], unittest.TestCase):
    """
    Base test class for LinearCoregionalizationKernel.

    Derived classes must implement the bkd() method to provide the backend.
    """

    def setUp(self) -> None:
        """
        Set up test environment for LinearCoregionalizationKernel.
        """
        np.random.seed(42)
        self.nvars = 2
        self.noutputs = 2
        self.nsamples = 10

        # Create base kernels (shared across outputs)
        self.kernels = [
            Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, self.bkd()),
            Matern32Kernel([0.5, 0.5], (0.1, 10.0), self.nvars, self.bkd()),
        ]

        # Create coregionalization matrices
        # B1: Positive correlation
        B1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        # B2: Mixed correlation
        B2 = np.array([[0.5, -0.2], [-0.2, 0.5]])
        self.coreg_matrices = [self.bkd().array(B1), self.bkd().array(B2)]

        self.lmc_kernel = LinearCoregionalizationKernel(
            self.kernels,
            self.coreg_matrices,
            self.noutputs
        )

        # Create sample data
        # For LMC, all outputs use same input locations
        X_base = self.bkd().array(np.random.randn(self.nvars, self.nsamples))
        self.X_list = [X_base] * self.noutputs

    def bkd(self) -> Backend[Array]:
        """
        Override this method in derived classes to provide the backend.
        """
        raise NotImplementedError(
            "Derived classes must implement this method."
        )

    def test_initialization(self) -> None:
        """
        Test LinearCoregionalizationKernel initialization.
        """
        lmc = LinearCoregionalizationKernel(
            self.kernels,
            self.coreg_matrices,
            self.noutputs
        )
        self.assertEqual(lmc.noutputs(), self.noutputs)
        self.assertEqual(lmc.nvars(), self.nvars)
        self.assertIsNotNone(lmc.bkd())

    def test_empty_kernels_error(self) -> None:
        """
        Test that empty kernels list raises ValueError.
        """
        with self.assertRaises(ValueError) as context:
            LinearCoregionalizationKernel([], self.coreg_matrices, self.noutputs)
        self.assertIn("cannot be empty", str(context.exception))

    def test_length_mismatch_error(self) -> None:
        """
        Test that mismatched lengths raise ValueError.
        """
        with self.assertRaises(ValueError) as context:
            LinearCoregionalizationKernel(
                self.kernels,
                [self.coreg_matrices[0]],  # Too short
                self.noutputs
            )
        self.assertIn("must equal", str(context.exception))

    def test_coreg_matrix_shape_error(self) -> None:
        """
        Test that wrong-shaped coregionalization matrix raises ValueError.
        """
        bad_matrix = self.bkd().array(np.random.randn(3, 3))  # Wrong shape
        with self.assertRaises(ValueError) as context:
            LinearCoregionalizationKernel(
                self.kernels,
                [bad_matrix, self.coreg_matrices[1]],
                self.noutputs
            )
        self.assertIn("expected", str(context.exception))

    def test_hyperparameter_list_combination(self) -> None:
        """
        Test that hyperparameter lists are combined correctly.
        """
        total_params = sum(k.hyp_list().nparams() for k in self.kernels)
        self.assertEqual(self.lmc_kernel.hyp_list().nparams(), total_params)

    def test_stacked_kernel_matrix_shape(self) -> None:
        """
        Test that stacked kernel matrix has correct shape.
        """
        K = self.lmc_kernel(self.X_list, block_format=False)

        n_total = self.nsamples * self.noutputs
        self.assertEqual(K.shape, (n_total, n_total))

    def test_block_format_structure(self) -> None:
        """
        Test that block format returns correct structure.
        """
        K_blocks = self.lmc_kernel(self.X_list, block_format=True)

        # Should be list of lists
        self.assertIsInstance(K_blocks, list)
        self.assertEqual(len(K_blocks), self.noutputs)

        for i in range(self.noutputs):
            self.assertIsInstance(K_blocks[i], list)
            self.assertEqual(len(K_blocks[i]), self.noutputs)

            # All blocks should be non-None (unlike independent kernel)
            for j in range(self.noutputs):
                self.assertIsNotNone(K_blocks[i][j])

    def test_block_format_shapes(self) -> None:
        """
        Test that blocks in block format have correct shapes.
        """
        K_blocks = self.lmc_kernel(self.X_list, block_format=True)

        for i in range(self.noutputs):
            for j in range(self.noutputs):
                self.assertEqual(
                    K_blocks[i][j].shape,
                    (self.nsamples, self.nsamples)
                )

    def test_lmc_formula_correctness(self) -> None:
        """
        Test that LMC kernel correctly implements K = sum_q B_q ⊗ k_q.
        """
        K_blocks = self.lmc_kernel(self.X_list, block_format=True)

        # Manually compute expected result
        expected_blocks = [[
            self.bkd().zeros((self.nsamples, self.nsamples))
            for _ in range(self.noutputs)
        ] for _ in range(self.noutputs)]

        for q in range(len(self.kernels)):
            K_q = self.kernels[q](self.X_list[0], self.X_list[0])
            B_q = self.coreg_matrices[q]

            for i in range(self.noutputs):
                for j in range(self.noutputs):
                    expected_blocks[i][j] = (
                        expected_blocks[i][j] + B_q[i, j] * K_q
                    )

        # Compare
        for i in range(self.noutputs):
            for j in range(self.noutputs):
                self.bkd().assert_allclose(
                    K_blocks[i][j],
                    expected_blocks[i][j]
                )

    def test_symmetry(self) -> None:
        """
        Test that kernel matrix is symmetric for self-covariance.
        """
        K = self.lmc_kernel(self.X_list, block_format=False)

        # Check symmetry
        K_T = self.bkd().moveaxis(K, 0, 1)  # Transpose
        self.bkd().assert_allclose(K, K_T, atol=1e-10)

    def test_repr(self) -> None:
        """
        Test string representation.
        """
        repr_str = repr(self.lmc_kernel)
        self.assertIn("LinearCoregionalizationKernel", repr_str)
        self.assertIn(f"noutputs={self.noutputs}", repr_str)


# NumPy implementations
class TestLinearCoregionalizationKernelNumpy(
    TestLinearCoregionalizationKernel[NDArray[Any]]
):
    def setUp(self) -> None:
        self._bkd = NumpyBkd()
        super().setUp()

    def bkd(self) -> NumpyBkd:
        return self._bkd


# PyTorch implementations
class TestLinearCoregionalizationKernelTorch(
    TestLinearCoregionalizationKernel[torch.Tensor]
):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestIndependentKernelIntegration(Generic[Array], unittest.TestCase):
    """
    Integration tests for IndependentMultiOutputKernel with same input data.

    Tests kernel behavior when all outputs use the same input locations,
    which is a common pattern in multi-output GPs.
    """

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        """Set up test environment."""
        np.random.seed(42)

        self.nvars = 2
        self.noutputs = 3
        self.n_train = 20

        # Create shared training data
        self.X_train_np = np.random.randn(self.nvars, self.n_train)
        self.X_train = self.bkd().array(self.X_train_np)

        # Create independent kernels for each output
        self.kernels = []
        for i in range(self.noutputs):
            kernel = Matern52Kernel(
                [1.0] * self.nvars,
                (0.1, 10.0),
                self.nvars,
                self.bkd()
            )
            self.kernels.append(kernel)

        self.mo_kernel = IndependentMultiOutputKernel(self.kernels)

    def test_same_input_locations_block_diagonal(self) -> None:
        """Test that kernel is block-diagonal when all outputs use same inputs."""
        # All outputs use same X
        X_list = [self.X_train] * self.noutputs

        K = self.mo_kernel(X_list, block_format=False)

        # Should be shape (n_train * noutputs, n_train * noutputs)
        expected_size = self.n_train * self.noutputs
        self.assertEqual(K.shape, (expected_size, expected_size))

        # Test block-diagonal structure
        for i in range(self.noutputs):
            row_start = i * self.n_train
            row_end = (i + 1) * self.n_train

            # Diagonal block should match single kernel
            K_ii = K[row_start:row_end, row_start:row_end]
            K_expected = self.kernels[i](self.X_train, self.X_train)
            self.bkd().assert_allclose(K_ii, K_expected, rtol=1e-10)

            # Off-diagonal blocks should be zero
            for j in range(self.noutputs):
                n_j = self.n_train
                if i != j:
                    col_start = j * self.n_train
                    col_end = (j + 1) * self.n_train
                    K_ij = K[row_start:row_end, col_start:col_end]

                    expected_zero = self.bkd().zeros((self.n_train, self.n_train))
                    self.bkd().assert_allclose(K_ij, expected_zero, atol=1e-10)

    def test_same_input_symmetric(self) -> None:
        """Test symmetry when all outputs use same inputs."""
        X_list = [self.X_train] * self.noutputs
        K = self.mo_kernel(X_list, block_format=False)

        # Test symmetry
        K_T = self.bkd().moveaxis(K, 0, 1)
        self.bkd().assert_allclose(K, K_T, rtol=1e-10)


class TestIndependentKernelIntegrationNumpy(
    TestIndependentKernelIntegration[NDArray[Any]]
):
    """NumPy backend tests for independent kernel integration."""

    def bkd(self) -> NumpyBkd:
        if not hasattr(self, '_bkd'):
            self._bkd = NumpyBkd()
        return self._bkd


class TestIndependentKernelIntegrationTorch(
    TestIndependentKernelIntegration[torch.Tensor]
):
    """PyTorch backend tests for independent kernel integration."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


class TestLMCKernelIntegration(Generic[Array], unittest.TestCase):
    """
    Integration tests for LinearCoregionalizationKernel.

    Tests LMC kernel properties with realistic coregionalization matrices.
    """

    def bkd(self) -> Backend[Array]:
        """Override in derived classes."""
        raise NotImplementedError("Derived classes must implement this method.")

    def setUp(self) -> None:
        """Set up test environment."""
        np.random.seed(42)

        self.nvars = 2
        self.noutputs = 2
        self.ncomponents = 2
        self.n_train = 15

        # Create training data
        X_train_np = np.random.randn(self.nvars, self.n_train)
        self.X_train = self.bkd().array(X_train_np)

        # Create base kernels
        self.base_kernels = []
        for q in range(self.ncomponents):
            kernel = Matern52Kernel(
                [1.0] * self.nvars,
                (0.1, 10.0),
                self.nvars,
                self.bkd()
            )
            self.base_kernels.append(kernel)

        # Create coregionalization matrices
        B1_np = np.array([[1.0, 0.7], [0.7, 1.0]])
        B2_np = np.array([[0.5, -0.2], [-0.2, 0.5]])

        self.coreg_matrices = [
            self.bkd().array(B1_np),
            self.bkd().array(B2_np),
        ]

        # Create LMC kernel
        self.lmc_kernel = LinearCoregionalizationKernel(
            self.base_kernels,
            self.coreg_matrices,
            self.noutputs
        )

    def test_lmc_symmetry_same_inputs(self) -> None:
        """Test that LMC kernel is symmetric when using same inputs."""
        # For LMC, all outputs use same X
        X_list = [self.X_train] * self.noutputs

        K = self.lmc_kernel(X_list, block_format=False)

        # Should be symmetric
        K_T = self.bkd().moveaxis(K, 0, 1)
        self.bkd().assert_allclose(K, K_T, rtol=1e-10)

    def test_lmc_output_correlation(self) -> None:
        """Test that LMC captures correlations between outputs."""
        X_list = [self.X_train] * self.noutputs

        K_blocks = self.lmc_kernel(X_list, block_format=True)

        # K[0][1] should have correlations (not all zero)
        K_01 = K_blocks[0][1]
        K_10 = K_blocks[1][0]

        # Should be symmetric
        K_01_T = self.bkd().moveaxis(K_01, 0, 1)
        self.bkd().assert_allclose(K_01, K_01_T, rtol=1e-10)
        self.bkd().assert_allclose(K_01, K_10, rtol=1e-10)

        # Some elements should be non-zero
        K_01_abs = self.bkd().abs(K_01)
        K_01_max_value = self.bkd().max(K_01_abs).item()
        self.assertGreater(K_01_max_value, 0.0)


class TestLMCKernelIntegrationNumpy(
    TestLMCKernelIntegration[NDArray[Any]]
):
    """NumPy backend tests for LMC kernel integration."""

    def bkd(self) -> NumpyBkd:
        if not hasattr(self, '_bkd'):
            self._bkd = NumpyBkd()
        return self._bkd


class TestLMCKernelIntegrationTorch(
    TestLMCKernelIntegration[torch.Tensor]
):
    """PyTorch backend tests for LMC kernel integration."""

    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        self._bkd = TorchBkd()
        super().setUp()

    def bkd(self) -> TorchBkd:
        return self._bkd


# Custom test loader
def load_tests(
    loader: unittest.TestLoader, tests, pattern: str
) -> unittest.TestSuite:
    """
    Custom test loader to exclude base classes.
    """
    test_suite = unittest.TestSuite()
    for test_class in [
        TestIndependentMultiOutputKernelNumpy,
        TestIndependentMultiOutputKernelTorch,
        TestLinearCoregionalizationKernelNumpy,
        TestLinearCoregionalizationKernelTorch,
        TestIndependentKernelIntegrationNumpy,
        TestIndependentKernelIntegrationTorch,
        TestLMCKernelIntegrationNumpy,
        TestLMCKernelIntegrationTorch,
    ]:
        test_suite.addTests(loader.loadTestsFromTestCase(test_class))
    return test_suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = load_tests(loader, [], None)
    runner = unittest.TextTestRunner()
    runner.run(suite)
