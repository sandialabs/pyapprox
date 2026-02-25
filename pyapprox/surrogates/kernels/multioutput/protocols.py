"""
Protocols for multi-output kernels.

This module defines the interface for multi-output kernels that can model
covariance across multiple outputs or quantities of interest.
"""

from typing import Generic, List, Optional, Protocol, Union, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import HyperParameterList


@runtime_checkable
class MultiOutputKernelProtocol(Protocol, Generic[Array]):
    """
    Protocol for multi-output kernel implementations.

    Multi-output kernels model the covariance structure across multiple
    outputs or quantities of interest. They can represent independent
    outputs (block-diagonal structure) or correlated outputs (dense structure).

    The protocol supports two output formats:
    - block_format=True: Returns list of lists representing kernel blocks
    - block_format=False: Returns single stacked kernel matrix
    """

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for numerical computations.

        Returns
        -------
        bkd : Backend[Array]
            Backend for numerical computations.
        """
        ...

    def hyp_list(self) -> HyperParameterList:
        """
        Return the list of hyperparameters associated with the kernel.

        Returns
        -------
        hyp_list : HyperParameterList
            List of hyperparameters.
        """
        ...

    def noutputs(self) -> int:
        """
        Return the number of outputs modeled by this kernel.

        Returns
        -------
        noutputs : int
            Number of outputs/quantities of interest.
        """
        ...

    def nvars(self) -> int:
        """
        Return the number of input variables.

        Returns
        -------
        nvars : int
            Number of input dimensions.
        """
        ...

    def __call__(
        self,
        X1_list: List[Array],
        X2_list: Optional[List[Array]] = None,
        block_format: bool = False,
    ) -> Union[Array, List[List[Optional[Array]]]]:
        """
        Compute the multi-output kernel matrix.

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
            Multi-output kernel matrix.

            If block_format=False (default):
                Single array of shape (n_total_1, n_total_2) where
                n_total_1 = sum(n_samples per output in X1_list)
                n_total_2 = sum(n_samples per output in X2_list)

            If block_format=True:
                List of lists K[i][j] representing kernel blocks between
                outputs i and j. Each block K[i][j] has shape
                (n_samples_output_i, n_samples_output_j).
                For independent kernels, off-diagonal blocks may be None
                or zero arrays.

        Examples
        --------
        For 2 outputs with 10 and 5 samples respectively:

        block_format=False:
            Returns (15, 15) matrix: [[K_00, K_01],
                                       [K_10, K_11]]

        block_format=True:
            Returns [[K_00, K_01],
                     [K_10, K_11]]
            where K_00 is (10, 10), K_01 is (10, 5),
                  K_10 is (5, 10),  K_11 is (5, 5)
        """
        ...
