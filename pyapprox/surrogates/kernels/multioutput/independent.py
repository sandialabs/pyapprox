"""
Independent multi-output kernel with block-diagonal structure.

This module provides the IndependentMultiOutputKernel, which models independent
outputs using separate kernels for each output. The resulting kernel matrix has
a block-diagonal structure.
"""

from typing import Generic, List, Optional, Union

from pyapprox.surrogates.kernels.protocols import Kernel
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


class IndependentMultiOutputKernel(Generic[Array]):
    """
    Independent multi-output kernel with block-diagonal structure.

    This kernel models multiple outputs as independent, using a separate
    kernel for each output. The resulting covariance matrix has a block-diagonal
    structure:

        K = [[K_0,    0,  ...,    0],
             [  0,  K_1,  ...,    0],
             [ ..., ..., ...,  ...],
             [  0,    0,  ..., K_M]]

    where K_i is the kernel matrix for output i.

    Parameters
    ----------
    kernels : List[Kernel]
        List of kernels, one for each output.
        All kernels must use the same backend.

    Attributes
    ----------
    _kernels : List[Kernel]
        List of kernels for each output.
    _bkd : Backend
        Backend for numerical computations.
    _noutputs : int
        Number of outputs.

    Examples
    --------
    >>> from pyapprox.surrogates.kernels import MaternKernel
    >>> from pyapprox.surrogates.kernels.multioutput import IndependentMultiOutputKernel
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> # Create separate kernels for 2 outputs
    >>> kernel1 = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
    >>> kernel2 = MaternKernel(1.5, [0.5, 0.5], (0.1, 10.0), 2, bkd)
    >>> mo_kernel = IndependentMultiOutputKernel([kernel1, kernel2])
    >>> # Create data for each output
    >>> X1 = [bkd.array(np.random.randn(2, 10)), bkd.array(np.random.randn(2, 5))]
    >>> # Compute kernel matrix
    >>> K = mo_kernel(X1)  # Shape: (15, 15) - stacked
    >>> K_blocks = mo_kernel(X1, block_format=True)  # List of blocks
    """

    def __init__(self, kernels: List[Kernel]):
        """
        Initialize the IndependentMultiOutputKernel.

        Parameters
        ----------
        kernels : List[Kernel]
            List of kernels, one for each output.

        Raises
        ------
        ValueError
            If kernels list is empty.
            If kernels have different backends.
            If kernels have different nvars.
        """
        if not kernels:
            raise ValueError("kernels list cannot be empty")

        # Validate all kernels have same backend
        bkd_class = kernels[0].bkd().__class__
        for i, kernel in enumerate(kernels[1:], 1):
            if kernel.bkd().__class__ != bkd_class:
                raise ValueError(
                    f"All kernels must have the same backend type. "
                    f"Kernel 0 has {bkd_class.__name__}, "
                    f"kernel {i} has {kernel.bkd().__class__.__name__}"
                )

        # Validate all kernels have same nvars
        nvars = kernels[0].nvars()
        for i, kernel in enumerate(kernels[1:], 1):
            if kernel.nvars() != nvars:
                raise ValueError(
                    f"All kernels must have the same number of input variables. "
                    f"Kernel 0 has nvars={nvars}, kernel {i} has nvars={kernel.nvars()}"
                )

        self._kernels = kernels
        self._bkd = kernels[0].bkd()
        self._noutputs = len(kernels)
        self._nvars = nvars

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
        Return the combined list of hyperparameters from all kernels.

        Returns
        -------
        hyp_list : HyperParameterList
            Combined hyperparameter list.
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

    def __call__(
        self,
        X1_list: List[Array],
        X2_list: Optional[List[Array]] = None,
        block_format: bool = False,
    ) -> Union[Array, List[List[Optional[Array]]]]:
        """
        Compute the independent multi-output kernel matrix.

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
        K : Array or List[List[Optional[Array]]]
            Independent multi-output kernel matrix.

            If block_format=False:
                Block-diagonal matrix of shape (n_total_1, n_total_2)

            If block_format=True:
                List of lists where K[i][j] is:
                - K_i(X1_i, X2_j) if i == j (diagonal block)
                - None if i != j (off-diagonal blocks are zero)

        Raises
        ------
        ValueError
            If length of X1_list doesn't match number of outputs.
            If X2_list is provided but length doesn't match.
        """
        # Validate input
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

        # Compute diagonal blocks
        if block_format:
            # Return list of lists with diagonal blocks and None for off-diagonal
            blocks: List[List[Optional[Array]]] = []
            for i in range(self._noutputs):
                row: List[Optional[Array]] = []
                for j in range(self._noutputs):
                    if i == j:
                        # Diagonal block: compute kernel
                        K_ij = self._kernels[i](X1_list[i], X2_list[j])
                        row.append(K_ij)
                    else:
                        # Off-diagonal block: None (zero covariance)
                        row.append(None)
                blocks.append(row)
            return blocks
        else:
            # Return stacked block-diagonal matrix
            diagonal_blocks = []
            for i in range(self._noutputs):
                K_i = self._kernels[i](X1_list[i], X2_list[i])
                diagonal_blocks.append(K_i)

            # Create block-diagonal matrix
            # Compute total size
            n1_sizes = [X1_list[i].shape[1] for i in range(self._noutputs)]
            n2_sizes = [X2_list[i].shape[1] for i in range(self._noutputs)]
            n1_total = sum(n1_sizes)
            n2_total = sum(n2_sizes)

            # Initialize full matrix with zeros
            K_full = self._bkd.zeros((n1_total, n2_total))

            # Fill in diagonal blocks
            row_offset = 0
            col_offset = 0
            for i in range(self._noutputs):
                n1 = n1_sizes[i]
                n2 = n2_sizes[i]
                K_full[row_offset : row_offset + n1, col_offset : col_offset + n2] = (
                    diagonal_blocks[i]
                )
                row_offset += n1
                col_offset += n2

            return K_full

    def jacobian_wrt_params(self, X_list: List[Array]) -> Array:
        """
        Compute Jacobian of kernel w.r.t. hyperparameters.

        For independent kernels, the Jacobian has a block-diagonal structure
        matching the kernel matrix.

        Parameters
        ----------
        X_list : List[Array]
            List of input data arrays, one per output.
            Each array has shape (nvars, n_samples_output_i).

        Returns
        -------
        jac : Array
            Jacobian of shape (n_total, n_total, nparams_total) where
            n_total = sum of samples across all outputs.

        Raises
        ------
        ValueError
            If length of X_list doesn't match number of outputs.
        NotImplementedError
            If any kernel doesn't support jacobian_wrt_params.
        """
        if len(X_list) != self._noutputs:
            raise ValueError(
                f"X_list must have {self._noutputs} arrays (one per output), "
                f"got {len(X_list)}"
            )

        # Check all kernels support parameter Jacobian
        for i, kernel in enumerate(self._kernels):
            if not hasattr(kernel, "jacobian_wrt_params"):
                raise NotImplementedError(
                    f"Kernel {i} does not support jacobian_wrt_params"
                )

        # Compute Jacobians for each kernel
        jacs = []
        for i in range(self._noutputs):
            jac_i = self._kernels[i].jacobian_wrt_params(X_list[i])
            jacs.append(jac_i)

        # Compute sizes
        n_sizes = [X_list[i].shape[1] for i in range(self._noutputs)]
        n_total = sum(n_sizes)
        nparams_total = self._hyp_list.nparams()

        # Initialize full Jacobian
        jac_full = self._bkd.zeros((n_total, n_total, nparams_total))

        # Fill in block-diagonal Jacobian
        row_offset = 0
        param_offset = 0
        for i in range(self._noutputs):
            n_i = n_sizes[i]
            nparams_i = jacs[i].shape[2]

            # Fill in block [row_offset:row_offset+n_i, row_offset:row_offset+n_i,
            # param_offset:param_offset+nparams_i]
            jac_full[
                row_offset : row_offset + n_i,
                row_offset : row_offset + n_i,
                param_offset : param_offset + nparams_i,
            ] = jacs[i]

            row_offset += n_i
            param_offset += nparams_i

        return jac_full

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
            f"IndependentMultiOutputKernel(\n"
            f"  noutputs={self._noutputs},\n"
            f"  kernels={kernel_reprs}\n"
            f")"
        )
