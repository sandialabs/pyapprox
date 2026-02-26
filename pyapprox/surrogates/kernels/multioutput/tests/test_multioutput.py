import numpy as np
import pytest

from pyapprox.surrogates.kernels.matern import (
    Matern32Kernel,
    Matern52Kernel,
)

# from pyapprox.surrogates.kernels.constant import ConstantKernel  # Not yet implemented
from pyapprox.surrogates.kernels.multioutput.independent import (
    IndependentMultiOutputKernel,
)
from pyapprox.surrogates.kernels.multioutput.linear_coregionalization import (
    LinearCoregionalizationKernel,
)
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.backends.torch import TorchBkd


class TestIndependentMultiOutputKernel:
    """
    Base test class for IndependentMultiOutputKernel.
    """

    def _setup_data(self, bkd):
        np.random.seed(42)
        self.nvars = 2
        self.noutputs = 3
        self.nsamples = [10, 5, 8]  # Different number of samples per output

        # Create independent kernels for each output
        self.kernels = []
        for i in range(self.noutputs):
            kernel = Matern52Kernel(
                [1.0] * self.nvars, (0.1, 10.0), self.nvars, bkd
            )
            self.kernels.append(kernel)

        self.mo_kernel = IndependentMultiOutputKernel(self.kernels)

        # Create sample data for each output
        self.X_list = [
            bkd.array(np.random.randn(self.nvars, self.nsamples[i]))
            for i in range(self.noutputs)
        ]
        self.X2_list = [
            bkd.array(np.random.randn(self.nvars, self.nsamples[i]))
            for i in range(self.noutputs)
        ]

    def test_initialization(self, bkd) -> None:
        """
        Test IndependentMultiOutputKernel initialization.
        """
        self._setup_data(bkd)
        mo_kernel = IndependentMultiOutputKernel(self.kernels)
        assert mo_kernel.noutputs() == self.noutputs
        assert mo_kernel.nvars() == self.nvars
        assert mo_kernel.bkd() is not None

    def test_empty_kernels_error(self, bkd) -> None:
        """
        Test that empty kernels list raises ValueError.
        """
        self._setup_data(bkd)
        with pytest.raises(ValueError) as context:
            IndependentMultiOutputKernel([])
        assert "cannot be empty" in str(context.value)

    def test_backend_mismatch_error(self, bkd) -> None:
        """
        Test that mixed backends raise ValueError.
        """
        self._setup_data(bkd)
        # Create kernel with different backend
        if isinstance(bkd, NumpyBkd):
            other_bkd = TorchBkd()
        else:
            other_bkd = NumpyBkd()

        kernel_other = Matern52Kernel(
            [1.0] * self.nvars, (0.1, 10.0), self.nvars, other_bkd
        )

        with pytest.raises(ValueError) as context:
            IndependentMultiOutputKernel([self.kernels[0], kernel_other])
        assert "same backend type" in str(context.value)

    def test_nvars_mismatch_error(self, bkd) -> None:
        """
        Test that kernels with different nvars raise ValueError.
        """
        self._setup_data(bkd)
        kernel_different = Matern52Kernel(
            [1.0, 1.0, 1.0],  # 3 dimensions instead of 2
            (0.1, 10.0),
            3,  # Different nvars
            bkd,
        )

        with pytest.raises(ValueError) as context:
            IndependentMultiOutputKernel([self.kernels[0], kernel_different])
        assert "same number of input variables" in str(context.value)

    def test_hyperparameter_list_combination(self, bkd) -> None:
        """
        Test that hyperparameter lists are combined correctly.
        """
        self._setup_data(bkd)
        total_params = sum(k.hyp_list().nparams() for k in self.kernels)
        assert self.mo_kernel.hyp_list().nparams() == total_params

    def test_stacked_kernel_matrix_shape(self, bkd) -> None:
        """
        Test that stacked kernel matrix has correct shape.
        """
        self._setup_data(bkd)
        K = self.mo_kernel(self.X_list, block_format=False)

        n_total = sum(self.nsamples)
        assert K.shape == (n_total, n_total)

    def test_stacked_kernel_matrix_cross_shape(self, bkd) -> None:
        """
        Test that cross-covariance has correct shape.
        """
        self._setup_data(bkd)
        K = self.mo_kernel(self.X_list, self.X2_list, block_format=False)

        n_total_1 = sum(self.nsamples)
        n_total_2 = sum(self.nsamples)
        assert K.shape == (n_total_1, n_total_2)

    def test_stacked_kernel_is_block_diagonal(self, bkd) -> None:
        """
        Test that stacked kernel matrix is block-diagonal.
        """
        self._setup_data(bkd)
        K = self.mo_kernel(self.X_list, block_format=False)

        # Extract diagonal blocks and verify off-diagonal blocks are zero
        row_offset = 0
        for i in range(self.noutputs):
            n_i = self.nsamples[i]

            # Extract diagonal block
            K_ii = K[row_offset : row_offset + n_i, row_offset : row_offset + n_i]

            # Compute expected diagonal block
            K_expected = self.kernels[i](self.X_list[i], self.X_list[i])
            bkd.assert_allclose(K_ii, K_expected)

            # Check off-diagonal blocks are zero
            col_offset = 0
            for j in range(self.noutputs):
                n_j = self.nsamples[j]
                if i != j:
                    K_ij = K[
                        row_offset : row_offset + n_i, col_offset : col_offset + n_j
                    ]

                    # Should be all zeros
                    expected_zero = bkd.zeros((n_i, n_j))
                    bkd.assert_allclose(K_ij, expected_zero)

                col_offset += n_j

            row_offset += n_i

    def test_block_format_structure(self, bkd) -> None:
        """
        Test that block format returns correct structure.
        """
        self._setup_data(bkd)
        K_blocks = self.mo_kernel(self.X_list, block_format=True)

        # Should be list of lists
        assert isinstance(K_blocks, list)
        assert len(K_blocks) == self.noutputs

        for i in range(self.noutputs):
            assert isinstance(K_blocks[i], list)
            assert len(K_blocks[i]) == self.noutputs

    def test_block_format_diagonal_blocks(self, bkd) -> None:
        """
        Test that diagonal blocks in block format are correct.
        """
        self._setup_data(bkd)
        K_blocks = self.mo_kernel(self.X_list, block_format=True)

        for i in range(self.noutputs):
            # Diagonal block should not be None
            assert K_blocks[i][i] is not None

            # Should match kernel evaluation
            K_expected = self.kernels[i](self.X_list[i], self.X_list[i])
            bkd.assert_allclose(K_blocks[i][i], K_expected)

    def test_block_format_off_diagonal_blocks(self, bkd) -> None:
        """
        Test that off-diagonal blocks in block format are None.
        """
        self._setup_data(bkd)
        K_blocks = self.mo_kernel(self.X_list, block_format=True)

        for i in range(self.noutputs):
            for j in range(self.noutputs):
                if i != j:
                    # Off-diagonal blocks should be None
                    assert K_blocks[i][j] is None

    def test_wrong_length_X_list_error(self, bkd) -> None:
        """
        Test that wrong length X_list raises ValueError.
        """
        self._setup_data(bkd)
        wrong_list = self.X_list[:2]  # Too short

        with pytest.raises(ValueError) as context:
            self.mo_kernel(wrong_list)
        assert "must have" in str(context.value)

    def test_wrong_length_X2_list_error(self, bkd) -> None:
        """
        Test that wrong length X2_list raises ValueError.
        """
        self._setup_data(bkd)
        wrong_list = self.X2_list[:2]  # Too short

        with pytest.raises(ValueError) as context:
            self.mo_kernel(self.X_list, wrong_list)
        assert "must have" in str(context.value)

    def test_jacobian_wrt_params_shape(self, bkd) -> None:
        """
        Test that parameter Jacobian has correct shape.
        """
        self._setup_data(bkd)
        jac = self.mo_kernel.jacobian_wrt_params(self.X_list)

        n_total = sum(self.nsamples)
        nparams_total = self.mo_kernel.hyp_list().nparams()

        assert jac.shape == (n_total, n_total, nparams_total)

    def test_jacobian_wrt_params_is_block_diagonal(self, bkd) -> None:
        """
        Test that parameter Jacobian has block-diagonal structure.
        """
        self._setup_data(bkd)
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
                row_offset : row_offset + n_i,
                row_offset : row_offset + n_i,
                param_offset : param_offset + nparams_i,
            ]

            bkd.assert_allclose(jac_ii, jac_i_expected)

            row_offset += n_i
            param_offset += nparams_i

    def test_repr(self, bkd) -> None:
        """
        Test string representation.
        """
        self._setup_data(bkd)
        repr_str = repr(self.mo_kernel)
        assert "IndependentMultiOutputKernel" in repr_str
        assert f"noutputs={self.noutputs}" in repr_str

    def test_with_different_kernel_types(self, bkd) -> None:
        """
        Test with different kernel types for different outputs.
        """
        self._setup_data(bkd)
        kernels_compat = [
            Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, bkd),
            Matern32Kernel([0.5, 0.5], (0.1, 10.0), self.nvars, bkd),
        ]

        mo_kernel = IndependentMultiOutputKernel(kernels_compat)
        X_list_2 = self.X_list[:2]
        K = mo_kernel(X_list_2)

        # Verify shape
        n_total = self.nsamples[0] + self.nsamples[1]
        assert K.shape == (n_total, n_total)

    def test_single_output(self, bkd) -> None:
        """
        Test with single output (edge case).
        """
        self._setup_data(bkd)
        mo_kernel = IndependentMultiOutputKernel([self.kernels[0]])
        X_list = [self.X_list[0]]

        K = mo_kernel(X_list)
        K_expected = self.kernels[0](self.X_list[0], self.X_list[0])

        bkd.assert_allclose(K, K_expected)

    def test_jacobian_wrt_params_derivative_checker(self, bkd) -> None:
        """
        Test parameter Jacobian using finite differences (DerivativeChecker).

        This validates that the analytical Jacobian matches numerical
        finite difference approximation.
        """
        from pyapprox.interface.functions.derivative_checks.derivative_checker import (
            DerivativeChecker,
        )

        self._setup_data(bkd)

        # Create wrapper class for DerivativeChecker
        class KernelWrapper:
            """Wrapper to make kernel compatible with DerivativeChecker."""

            def __init__(self, kernel, X_list, bkd_):
                self._kernel = kernel
                self._X_list = X_list
                self._bkd = bkd_

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

            def __call__(self, params):
                """
                Evaluate kernel matrix with given hyperparameters.
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

            def jacobian(self, params):
                """
                Compute Jacobian of kernel matrix w.r.t. hyperparameters.
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
                jac_reshaped = self._bkd.reshape(
                    jac, (self._n_total * self._n_total, nparams)
                )

                return jac_reshaped

            def jvp(self, params, v):
                """
                Compute Jacobian-vector product.
                """
                # Compute Jacobian
                J = self.jacobian(params)  # (n_total*n_total, nparams)

                # Flatten v if needed
                v_flat = self._bkd.reshape(v, (-1, 1))  # (nparams, 1)

                # JVP = J @ v
                return J @ v_flat

        # Create wrapper
        wrapper = KernelWrapper(self.mo_kernel, self.X_list, bkd)

        # Create derivative checker
        checker = DerivativeChecker(wrapper)

        # Get current hyperparameters
        params = self.mo_kernel.hyp_list().get_active_values()

        # Use logarithmically-spaced step sizes
        fd_eps = bkd.flip(bkd.logspace(-14, 0, 15))

        # Check gradient accuracy
        errors = checker.check_derivatives(
            params[:, None],  # Shape: (nactive, 1)
            fd_eps=fd_eps,
            relative=True,
            verbosity=0,
        )

        # Get gradient error
        grad_error = errors[0]

        # Minimum error should be small
        min_error = float(bkd.min(grad_error))
        assert (
            min_error < 1e-6
        ), f"Minimum gradient relative error {min_error} exceeds threshold"


class TestLinearCoregionalizationKernel:
    """
    Base test class for LinearCoregionalizationKernel.
    """

    def _setup_data(self, bkd):
        np.random.seed(42)
        self.nvars = 2
        self.noutputs = 2
        self.nsamples = 10

        # Create base kernels (shared across outputs)
        self.kernels = [
            Matern52Kernel([1.0, 1.0], (0.1, 10.0), self.nvars, bkd),
            Matern32Kernel([0.5, 0.5], (0.1, 10.0), self.nvars, bkd),
        ]

        # Create coregionalization matrices
        # B1: Positive correlation
        B1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        # B2: Mixed correlation
        B2 = np.array([[0.5, -0.2], [-0.2, 0.5]])
        self.coreg_matrices = [bkd.array(B1), bkd.array(B2)]

        self.lmc_kernel = LinearCoregionalizationKernel(
            self.kernels, self.coreg_matrices, self.noutputs
        )

        # Create sample data
        # For LMC, all outputs use same input locations
        X_base = bkd.array(np.random.randn(self.nvars, self.nsamples))
        self.X_list = [X_base] * self.noutputs

    def test_initialization(self, bkd) -> None:
        """
        Test LinearCoregionalizationKernel initialization.
        """
        self._setup_data(bkd)
        lmc = LinearCoregionalizationKernel(
            self.kernels, self.coreg_matrices, self.noutputs
        )
        assert lmc.noutputs() == self.noutputs
        assert lmc.nvars() == self.nvars
        assert lmc.bkd() is not None

    def test_empty_kernels_error(self, bkd) -> None:
        """
        Test that empty kernels list raises ValueError.
        """
        self._setup_data(bkd)
        with pytest.raises(ValueError) as context:
            LinearCoregionalizationKernel([], self.coreg_matrices, self.noutputs)
        assert "cannot be empty" in str(context.value)

    def test_length_mismatch_error(self, bkd) -> None:
        """
        Test that mismatched lengths raise ValueError.
        """
        self._setup_data(bkd)
        with pytest.raises(ValueError) as context:
            LinearCoregionalizationKernel(
                self.kernels,
                [self.coreg_matrices[0]],  # Too short
                self.noutputs,
            )
        assert "must equal" in str(context.value)

    def test_coreg_matrix_shape_error(self, bkd) -> None:
        """
        Test that wrong-shaped coregionalization matrix raises ValueError.
        """
        self._setup_data(bkd)
        bad_matrix = bkd.array(np.random.randn(3, 3))  # Wrong shape
        with pytest.raises(ValueError) as context:
            LinearCoregionalizationKernel(
                self.kernels, [bad_matrix, self.coreg_matrices[1]], self.noutputs
            )
        assert "expected" in str(context.value)

    def test_hyperparameter_list_combination(self, bkd) -> None:
        """
        Test that hyperparameter lists are combined correctly.
        """
        self._setup_data(bkd)
        total_params = sum(k.hyp_list().nparams() for k in self.kernels)
        assert self.lmc_kernel.hyp_list().nparams() == total_params

    def test_stacked_kernel_matrix_shape(self, bkd) -> None:
        """
        Test that stacked kernel matrix has correct shape.
        """
        self._setup_data(bkd)
        K = self.lmc_kernel(self.X_list, block_format=False)

        n_total = self.nsamples * self.noutputs
        assert K.shape == (n_total, n_total)

    def test_block_format_structure(self, bkd) -> None:
        """
        Test that block format returns correct structure.
        """
        self._setup_data(bkd)
        K_blocks = self.lmc_kernel(self.X_list, block_format=True)

        # Should be list of lists
        assert isinstance(K_blocks, list)
        assert len(K_blocks) == self.noutputs

        for i in range(self.noutputs):
            assert isinstance(K_blocks[i], list)
            assert len(K_blocks[i]) == self.noutputs

            # All blocks should be non-None (unlike independent kernel)
            for j in range(self.noutputs):
                assert K_blocks[i][j] is not None

    def test_block_format_shapes(self, bkd) -> None:
        """
        Test that blocks in block format have correct shapes.
        """
        self._setup_data(bkd)
        K_blocks = self.lmc_kernel(self.X_list, block_format=True)

        for i in range(self.noutputs):
            for j in range(self.noutputs):
                assert K_blocks[i][j].shape == (self.nsamples, self.nsamples)

    def test_lmc_formula_correctness(self, bkd) -> None:
        """
        Test that LMC kernel correctly implements K = sum_q B_q (x) k_q.
        """
        self._setup_data(bkd)
        K_blocks = self.lmc_kernel(self.X_list, block_format=True)

        # Manually compute expected result
        expected_blocks = [
            [
                bkd.zeros((self.nsamples, self.nsamples))
                for _ in range(self.noutputs)
            ]
            for _ in range(self.noutputs)
        ]

        for q in range(len(self.kernels)):
            K_q = self.kernels[q](self.X_list[0], self.X_list[0])
            B_q = self.coreg_matrices[q]

            for i in range(self.noutputs):
                for j in range(self.noutputs):
                    expected_blocks[i][j] = expected_blocks[i][j] + B_q[i, j] * K_q

        # Compare
        for i in range(self.noutputs):
            for j in range(self.noutputs):
                bkd.assert_allclose(K_blocks[i][j], expected_blocks[i][j])

    def test_symmetry(self, bkd) -> None:
        """
        Test that kernel matrix is symmetric for self-covariance.
        """
        self._setup_data(bkd)
        K = self.lmc_kernel(self.X_list, block_format=False)

        # Check symmetry
        K_T = bkd.moveaxis(K, 0, 1)  # Transpose
        bkd.assert_allclose(K, K_T, atol=1e-10)

    def test_repr(self, bkd) -> None:
        """
        Test string representation.
        """
        self._setup_data(bkd)
        repr_str = repr(self.lmc_kernel)
        assert "LinearCoregionalizationKernel" in repr_str
        assert f"noutputs={self.noutputs}" in repr_str


class TestIndependentKernelIntegration:
    """
    Integration tests for IndependentMultiOutputKernel with same input data.

    Tests kernel behavior when all outputs use the same input locations,
    which is a common pattern in multi-output GPs.
    """

    def _setup_data(self, bkd):
        np.random.seed(42)

        self.nvars = 2
        self.noutputs = 3
        self.n_train = 20

        # Create shared training data
        self.X_train_np = np.random.randn(self.nvars, self.n_train)
        self.X_train = bkd.array(self.X_train_np)

        # Create independent kernels for each output
        self.kernels = []
        for i in range(self.noutputs):
            kernel = Matern52Kernel(
                [1.0] * self.nvars, (0.1, 10.0), self.nvars, bkd
            )
            self.kernels.append(kernel)

        self.mo_kernel = IndependentMultiOutputKernel(self.kernels)

    def test_same_input_locations_block_diagonal(self, bkd) -> None:
        """Test that kernel is block-diagonal when all outputs use same inputs."""
        self._setup_data(bkd)
        # All outputs use same X
        X_list = [self.X_train] * self.noutputs

        K = self.mo_kernel(X_list, block_format=False)

        # Should be shape (n_train * noutputs, n_train * noutputs)
        expected_size = self.n_train * self.noutputs
        assert K.shape == (expected_size, expected_size)

        # Test block-diagonal structure
        for i in range(self.noutputs):
            row_start = i * self.n_train
            row_end = (i + 1) * self.n_train

            # Diagonal block should match single kernel
            K_ii = K[row_start:row_end, row_start:row_end]
            K_expected = self.kernels[i](self.X_train, self.X_train)
            bkd.assert_allclose(K_ii, K_expected, rtol=1e-10)

            # Off-diagonal blocks should be zero
            for j in range(self.noutputs):
                if i != j:
                    col_start = j * self.n_train
                    col_end = (j + 1) * self.n_train
                    K_ij = K[row_start:row_end, col_start:col_end]

                    expected_zero = bkd.zeros((self.n_train, self.n_train))
                    bkd.assert_allclose(K_ij, expected_zero, atol=1e-10)

    def test_same_input_symmetric(self, bkd) -> None:
        """Test symmetry when all outputs use same inputs."""
        self._setup_data(bkd)
        X_list = [self.X_train] * self.noutputs
        K = self.mo_kernel(X_list, block_format=False)

        # Test symmetry
        K_T = bkd.moveaxis(K, 0, 1)
        bkd.assert_allclose(K, K_T, rtol=1e-10)


class TestLMCKernelIntegration:
    """
    Integration tests for LinearCoregionalizationKernel.

    Tests LMC kernel properties with realistic coregionalization matrices.
    """

    def _setup_data(self, bkd):
        np.random.seed(42)

        self.nvars = 2
        self.noutputs = 2
        self.ncomponents = 2
        self.n_train = 15

        # Create training data
        X_train_np = np.random.randn(self.nvars, self.n_train)
        self.X_train = bkd.array(X_train_np)

        # Create base kernels
        self.base_kernels = []
        for q in range(self.ncomponents):
            kernel = Matern52Kernel(
                [1.0] * self.nvars, (0.1, 10.0), self.nvars, bkd
            )
            self.base_kernels.append(kernel)

        # Create coregionalization matrices
        B1_np = np.array([[1.0, 0.7], [0.7, 1.0]])
        B2_np = np.array([[0.5, -0.2], [-0.2, 0.5]])

        self.coreg_matrices = [
            bkd.array(B1_np),
            bkd.array(B2_np),
        ]

        # Create LMC kernel
        self.lmc_kernel = LinearCoregionalizationKernel(
            self.base_kernels, self.coreg_matrices, self.noutputs
        )

    def test_lmc_symmetry_same_inputs(self, bkd) -> None:
        """Test that LMC kernel is symmetric when using same inputs."""
        self._setup_data(bkd)
        # For LMC, all outputs use same X
        X_list = [self.X_train] * self.noutputs

        K = self.lmc_kernel(X_list, block_format=False)

        # Should be symmetric
        K_T = bkd.moveaxis(K, 0, 1)
        bkd.assert_allclose(K, K_T, rtol=1e-10)

    def test_lmc_output_correlation(self, bkd) -> None:
        """Test that LMC captures correlations between outputs."""
        self._setup_data(bkd)
        X_list = [self.X_train] * self.noutputs

        K_blocks = self.lmc_kernel(X_list, block_format=True)

        # K[0][1] should have correlations (not all zero)
        K_01 = K_blocks[0][1]
        K_10 = K_blocks[1][0]

        # Should be symmetric
        K_01_T = bkd.moveaxis(K_01, 0, 1)
        bkd.assert_allclose(K_01, K_01_T, rtol=1e-10)
        bkd.assert_allclose(K_01, K_10, rtol=1e-10)

        # Some elements should be non-zero
        K_01_abs = bkd.abs(K_01)
        K_01_max_value = bkd.max(K_01_abs).item()
        assert K_01_max_value > 0.0
