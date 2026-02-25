"""
Training data management for Multi-output Gaussian Processes.

This module provides the MultiOutputGPTrainingData class which encapsulates
and validates training data for multi-output GP regression.
"""

from typing import Generic, List
from pyapprox.util.backends.protocols import Array, Backend


class MultiOutputGPTrainingData(Generic[Array]):
    """
    Encapsulates and validates training data for Multi-output GP regression.

    Stores data in user-friendly format (1, n_samples) per output, matching
    the standard shape convention, and provides stacked format for internal
    computations.

    Parameters
    ----------
    X_list : List[Array]
        Training input data for each output. Each array has shape (nvars, n_i).
    y_list : List[Array]
        Training output data for each output. Each array has shape (1, n_i).
    bkd : Backend[Array]
        Backend for numerical operations.

    Raises
    ------
    ValueError
        If data shapes are inconsistent or invalid.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> import numpy as np
    >>> bkd = NumpyBkd()
    >>> X1 = bkd.array(np.random.randn(2, 10))  # 2D input, 10 samples for output 1
    >>> X2 = bkd.array(np.random.randn(2, 10))  # 2D input, 10 samples for output 2
    >>> y1 = bkd.array(np.random.randn(1, 10))  # 1 output, 10 samples
    >>> y2 = bkd.array(np.random.randn(1, 10))  # 1 output, 10 samples
    >>> data = MultiOutputGPTrainingData([X1, X2], [y1, y2], bkd)
    >>> data.noutputs()
    2
    >>> data.n_samples_per_output()
    [10, 10]
    """

    def __init__(
        self,
        X_list: List[Array],
        y_list: List[Array],
        bkd: Backend[Array]
    ):
        self._bkd = bkd
        self._validate_and_store(X_list, y_list)

    def _validate_and_store(
        self, X_list: List[Array], y_list: List[Array]
    ) -> None:
        """
        Validate training data shapes and store.

        Parameters
        ----------
        X_list : List[Array]
            Training inputs for each output, shape (nvars, n_i) each.
        y_list : List[Array]
            Training outputs for each output, shape (1, n_i) each.

        Raises
        ------
        ValueError
            If shapes are invalid or inconsistent.
        """
        if len(X_list) != len(y_list):
            raise ValueError(
                f"X_list and y_list must have same length, "
                f"got {len(X_list)} and {len(y_list)}"
            )

        if len(X_list) == 0:
            raise ValueError("X_list and y_list must not be empty")

        # Validate first X to get nvars
        if X_list[0].ndim != 2:
            raise ValueError(
                f"X_list[0] must be 2D (nvars, n_samples), "
                f"got shape {X_list[0].shape}"
            )
        nvars = X_list[0].shape[0]

        n_samples_list = []

        for i, (X, y) in enumerate(zip(X_list, y_list)):
            # Validate X
            if X.ndim != 2:
                raise ValueError(
                    f"X_list[{i}] must be 2D (nvars, n_samples), "
                    f"got shape {X.shape}"
                )
            if X.shape[0] != nvars:
                raise ValueError(
                    f"X_list[{i}] has {X.shape[0]} variables, "
                    f"expected {nvars}"
                )

            n_samples = X.shape[1]
            if n_samples == 0:
                raise ValueError(f"X_list[{i}] must have at least one sample")

            # Validate y - shape should be (1, n_samples)
            if y.ndim != 2:
                raise ValueError(
                    f"y_list[{i}] must be 2D (1, n_samples), "
                    f"got shape {y.shape}"
                )
            if y.shape[0] != 1:
                raise ValueError(
                    f"y_list[{i}] must have shape (1, n_samples), "
                    f"got shape {y.shape}. Each output should have nqoi=1."
                )
            if y.shape[1] != n_samples:
                raise ValueError(
                    f"y_list[{i}] has {y.shape[1]} samples, "
                    f"expected {n_samples} to match X_list[{i}]"
                )

            n_samples_list.append(n_samples)

        # Store validated data
        self._X_list = X_list
        self._y_list = y_list
        self._nvars = nvars
        self._n_samples_list = n_samples_list
        self._noutputs = len(X_list)

        # Precompute stacked format for internal use
        self._y_stacked = self._compute_stacked_y()

    def _compute_stacked_y(self) -> Array:
        """
        Compute stacked y format for internal kernel computations.

        Returns
        -------
        Array
            Stacked outputs, shape (sum(n_i), 1).
        """
        # Flatten each y from (1, n_i) to (n_i,), then concatenate
        flattened = [self._bkd.reshape(y, (-1,)) for y in self._y_list]
        stacked = self._bkd.concatenate(flattened, axis=0)
        return self._bkd.reshape(stacked, (-1, 1))

    def X_list(self) -> List[Array]:
        """
        Return training input data for each output.

        Returns
        -------
        List[Array]
            Training inputs, each with shape (nvars, n_i).
        """
        return self._X_list

    def y_list(self) -> List[Array]:
        """
        Return training output data for each output.

        Returns
        -------
        List[Array]
            Training outputs, each with shape (1, n_i).
        """
        return self._y_list

    def y_stacked(self) -> Array:
        """
        Return stacked training outputs for internal computations.

        Returns
        -------
        Array
            Stacked outputs, shape (sum(n_i), 1).
        """
        return self._y_stacked

    def n_samples_per_output(self) -> List[int]:
        """
        Return number of training samples for each output.

        Returns
        -------
        List[int]
            Number of samples for each output.
        """
        return self._n_samples_list

    def n_total_samples(self) -> int:
        """
        Return total number of training samples across all outputs.

        Returns
        -------
        int
            Total number of samples.
        """
        return sum(self._n_samples_list)

    def nvars(self) -> int:
        """
        Return number of input variables.

        Returns
        -------
        int
            Number of input dimensions.
        """
        return self._nvars

    def noutputs(self) -> int:
        """
        Return number of outputs.

        Returns
        -------
        int
            Number of output quantities.
        """
        return self._noutputs

    def __repr__(self) -> str:
        """
        Return string representation of training data.

        Returns
        -------
        str
            Description of training data dimensions.
        """
        return (
            f"MultiOutputGPTrainingData(noutputs={self._noutputs}, "
            f"nvars={self._nvars}, n_samples_per_output={self._n_samples_list})"
        )
