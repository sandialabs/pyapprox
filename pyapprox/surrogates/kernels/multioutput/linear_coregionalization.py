"""
Linear Model of Coregionalization (LMC) kernel for multi-output GPs.

This module provides the LinearCoregionalizationKernel, which models correlations
between outputs using the linear model of coregionalization formulation.
"""

from typing import Generic, List, Optional, Union

from pyapprox.surrogates.kernels.protocols import Kernel
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class LinearCoregionalizationKernel(Generic[Array]):
    """
    Linear Model of Coregionalization (LMC) kernel.

    The LMC kernel models multi-output covariance as a sum of Kronecker products:

        K = sum_q B_q ⊗ k_q(x, x')

    where:
    - k_q are base kernels (shared across outputs)
    - B_q are coregionalization matrices (model output correlations)
    - ⊗ denotes Kronecker product

    This formulation naturally shares hyperparameters across outputs through
    the base kernels k_q, while allowing output-specific correlations through
    the B_q matrices.

    Parameters
    ----------
    kernels : List[Kernel]
        List of base kernels k_q. All must use the same backend and nvars.
    coregionalization_matrices : List[Array]
        List of coregionalization matrices B_q, one per kernel.
        Each B_q has shape (noutputs, noutputs) and should be positive
        semi-definite for a valid covariance function.
    noutputs : int
        Number of outputs/quantities of interest.

    Attributes
    ----------
    _kernels : List[Kernel]
        Base kernels.
    _coreg_matrices : List[Array]
        Coregionalization matrices.
    _noutputs : int
        Number of outputs.
    _bkd : Backend
        Backend for numerical computations.

    Examples
    --------
    >>> from pyapprox.surrogates.kernels import MaternKernel
    >>> from pyapprox.surrogates.kernels.multioutput import
    LinearCoregionalizationKernel
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> # Create base kernels (shared across outputs)
    >>> k1 = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
    >>> k2 = MaternKernel(1.5, [0.5, 0.5], (0.1, 10.0), 2, bkd)
    >>> # Create coregionalization matrices (2 outputs)
    >>> B1 = bkd.array([[1.0, 0.5], [0.5, 1.0]])  # Positive correlation
    >>> B2 = bkd.array([[0.5, -0.2], [-0.2, 0.5]])  # Mixed correlation
    >>> # Create LMC kernel
    >>> lmc = LinearCoregionalizationKernel([k1, k2], [B1, B2], 2)
    >>> # Evaluate kernel
    >>> X1 = [bkd.array(np.random.randn(2, 10)), bkd.array(np.random.randn(2, 5))]
    >>> K = lmc(X1)  # Shape: (15, 15) - stacked
    """

    def __init__(
        self,
        kernels: List[Kernel],
        coregionalization_matrices: List[Array],
        noutputs: int,
    ):
        """
        Initialize the LinearCoregionalizationKernel.

        Parameters
        ----------
        kernels : List[Kernel]
            List of base kernels.
        coregionalization_matrices : List[Array]
            List of coregionalization matrices, one per kernel.
        noutputs : int
            Number of outputs.

        Raises
        ------
        ValueError
            If kernels or coregionalization_matrices lists are empty.
            If lengths don't match.
            If kernels have different backends or nvars.
            If coregionalization matrices have wrong shape.
        """
        if not kernels:
            raise ValueError("kernels list cannot be empty")

        if not coregionalization_matrices:
            raise ValueError("coregionalization_matrices list cannot be empty")

        if len(kernels) != len(coregionalization_matrices):
            raise ValueError(
                f"Number of kernels ({len(kernels)}) must equal number of "
                f"coregionalization matrices ({len(coregionalization_matrices)})"
            )

        # Validate all kernels have same backend and nvars
        bkd_class = kernels[0].bkd().__class__
        nvars = kernels[0].nvars()
        for i, kernel in enumerate(kernels[1:], 1):
            if kernel.bkd().__class__ != bkd_class:
                raise ValueError(
                    f"All kernels must have the same backend type. "
                    f"Kernel 0 has {bkd_class.__name__}, "
                    f"kernel {i} has {kernel.bkd().__class__.__name__}"
                )
            if kernel.nvars() != nvars:
                raise ValueError(
                    f"All kernels must have the same number of input variables. "
                    f"Kernel 0 has nvars={nvars}, kernel {i} has nvars={kernel.nvars()}"
                )

        self._bkd = kernels[0].bkd()

        # Validate coregionalization matrices
        for i, B in enumerate(coregionalization_matrices):
            if B.shape != (noutputs, noutputs):
                raise ValueError(
                    f"Coregionalization matrix {i} has shape {B.shape}, "
                    f"expected ({noutputs}, {noutputs})"
                )

        self._kernels = kernels
        self._coreg_matrices = coregionalization_matrices
        self._noutputs = noutputs
        self._nvars = nvars
        self._ncomponents = len(kernels)

        # Combine hyperparameter lists from all kernels
        self._hyp_list = sum(
            [k.hyp_list() for k in kernels], HyperParameterList([], bkd=self._bkd)
        )

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
            Backend for numerical computations.
        """
        return self._bkd

    def hyp_list(self) -> HyperParameterList:
        """
        Return the combined list of hyperparameters from all base kernels.

        Note: Coregionalization matrices are typically fixed or optimized
        separately, so they are not included in the hyperparameter list.

        Returns
        -------
        hyp_list : HyperParameterList
            Combined hyperparameter list from base kernels.
        """
        return self._hyp_list

    def noutputs(self) -> int:
        """
        Return the number of outputs modeled by this kernel.

        Returns
        -------
        noutputs : int
            Number of outputs.
        """
        return self._noutputs

    def nvars(self) -> int:
        """
        Return the number of input variables.

        Returns
        -------
        nvars : int
            Number of input dimensions.
        """
        return self._nvars

    def _kronecker_product_kernel_block(self, B: Array, K: Array) -> List[List[Array]]:
        """
        Compute Kronecker product B ⊗ K.

        For B of shape (noutputs, noutputs) and K of shape (n1, n2),
        returns blocks[i][j] = B[i, j] * K for all i, j.

        Parameters
        ----------
        B : Array
            Coregionalization matrix, shape (noutputs, noutputs).
        K : Array
            Kernel matrix, shape (n1, n2).

        Returns
        -------
        blocks : List[List[Array]]
            Kronecker product as list of blocks.
        """
        blocks: List[List[Array]] = []
        for i in range(self._noutputs):
            row: List[Array] = []
            for j in range(self._noutputs):
                # Kronecker product: B[i, j] * K
                block = B[i, j] * K
                row.append(block)
            blocks.append(row)
        return blocks

    def __call__(
        self,
        X1_list: List[Array],
        X2_list: Optional[List[Array]] = None,
        block_format: bool = False,
    ) -> Union[Array, List[List[Array]]]:
        """
        Compute the LMC kernel matrix.

        Parameters
        ----------
        X1_list : List[Array]
            List of input data arrays, one per output.
            Each array has shape (nvars, n_samples_output_i).
        X2_list : List[Array], optional
            List of input data arrays for cross-covariance.
            If None, computes self-covariance using X1_list.
        block_format : bool, optional
            If True, return kernel as list of lists of blocks.
            If False, return stacked kernel matrix (default).

        Returns
        -------
        K : Array or List[List[Array]]
            LMC kernel matrix.

            If block_format=False:
                Stacked matrix of shape (n_total_1, n_total_2)

            If block_format=True:
                List of lists where K[i][j] is the kernel block between
                outputs i and j, shape (n_samples_i, n_samples_j).

        Raises
        ------
        ValueError
            If length of X1_list doesn't match number of outputs.
            If X2_list is provided but length doesn't match.
        """
        if len(X1_list) != self._noutputs:
            raise ValueError(
                f"X1_list must have {self._noutputs} arrays (one per output), "
                f"got {len(X1_list)}"
            )

        if X2_list is None:
            X2_list = X1_list
        elif len(X2_list) != self._noutputs:
            raise ValueError(
                f"X2_list must have {self._noutputs} arrays (one per output), "
                f"got {len(X2_list)}"
            )

        # For LMC, all outputs use same input locations
        # We'll use the first array as representative
        # (in practice, X1_list[i] should all be the same for LMC)
        X1 = X1_list[0]
        X2 = X2_list[0]

        # Initialize result blocks
        if block_format:
            # Start with zero blocks
            result_blocks: List[List[Array]] = []
            for i in range(self._noutputs):
                row: List[Array] = []
                for j in range(self._noutputs):
                    n1 = X1.shape[1]
                    n2 = X2.shape[1]
                    row.append(self._bkd.zeros((n1, n2)))
                result_blocks.append(row)

            # Sum over components: K = sum_q B_q ⊗ k_q(x, x')
            for q in range(self._ncomponents):
                K_q = self._kernels[q](X1, X2)
                B_q = self._coreg_matrices[q]

                # Compute Kronecker product and add to result
                kron_blocks = self._kronecker_product_kernel_block(B_q, K_q)
                for i in range(self._noutputs):
                    for j in range(self._noutputs):
                        result_blocks[i][j] = result_blocks[i][j] + kron_blocks[i][j]

            return result_blocks

        else:
            # Compute stacked matrix
            n1 = X1.shape[1]
            n2 = X2.shape[1]
            n1_total = n1 * self._noutputs
            n2_total = n2 * self._noutputs

            result = self._bkd.zeros((n1_total, n2_total))

            # Sum over components
            for q in range(self._ncomponents):
                K_q = self._kernels[q](X1, X2)
                B_q = self._coreg_matrices[q]

                # Add Kronecker product contribution
                for i in range(self._noutputs):
                    for j in range(self._noutputs):
                        row_start = i * n1
                        row_end = (i + 1) * n1
                        col_start = j * n2
                        col_end = (j + 1) * n2

                        result[row_start:row_end, col_start:col_end] = (
                            result[row_start:row_end, col_start:col_end]
                            + B_q[i, j] * K_q
                        )

            return result

    def __repr__(self) -> str:
        """
        Return string representation of the kernel.

        Returns
        -------
        repr : str
            String representation.
        """
        kernel_reprs = [repr(k) for k in self._kernels]
        return (
            f"LinearCoregionalizationKernel(\n"
            f"  noutputs={self._noutputs},\n"
            f"  ncomponents={self._ncomponents},\n"
            f"  kernels={kernel_reprs}\n"
            f")"
        )
