"""
Training data management for Gaussian Processes.

This module provides the GPTrainingData class which encapsulates and
validates training data for GP regression.
"""

from typing import Generic, Optional
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.surrogates.gaussianprocess.output_transform import (
    OutputAffineTransformProtocol,
)


class GPTrainingData(Generic[Array]):
    """
    Encapsulates and validates training data for Gaussian Process regression.

    This class stores training inputs and outputs, validates their shapes,
    and provides convenient access methods.

    Parameters
    ----------
    X_train : Array
        Training input data, shape (nvars, n_train).
    y_train : Array
        Training output data, shape (nqoi, n_train).
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
    >>> X = bkd.array(np.random.randn(2, 10))  # 2D input, 10 samples
    >>> y = bkd.array(np.random.randn(1, 10))  # 1 output, 10 samples
    >>> data = GPTrainingData(X, y, bkd)
    >>> data.n_samples()
    10
    >>> data.nvars()
    2
    """

    def __init__(
        self,
        X_train: Array,
        y_train: Array,
        bkd: Backend[Array],
        output_transform: Optional[OutputAffineTransformProtocol[Array]] = None,
    ):
        self._bkd = bkd
        self._output_transform = output_transform
        self._validate_and_store(X_train, y_train)

    def _validate_and_store(self, X_train: Array, y_train: Array) -> None:
        """
        Validate training data shapes and store.

        Parameters
        ----------
        X_train : Array
            Training inputs, shape (nvars, n_train).
        y_train : Array
            Training outputs, shape (nqoi, n_train).

        Raises
        ------
        ValueError
            If shapes are invalid or inconsistent.
        """
        # Validate X_train
        if X_train.ndim != 2:
            raise ValueError(
                f"X_train must be 2D (nvars, n_train), got shape {X_train.shape}"
            )

        nvars, n_train = X_train.shape

        if n_train == 0:
            raise ValueError("X_train must have at least one sample")

        # Validate y_train - shape is (nqoi, n_train)
        if y_train.ndim != 2:
            raise ValueError(
                f"y_train must be 2D (nqoi, n_train), got shape {y_train.shape}"
            )

        nqoi, n_train_y = y_train.shape

        if n_train_y != n_train:
            raise ValueError(
                f"Number of samples mismatch: X_train has {n_train} samples, "
                f"y_train has {n_train_y} samples"
            )

        if nqoi == 0:
            raise ValueError("y_train must have at least one output dimension")

        # Store validated data
        self._X_train = X_train
        self._y_train = y_train
        self._nvars = nvars
        self._n_train = n_train
        self._nqoi = nqoi

    def X(self) -> Array:
        """
        Return training input data.

        Returns
        -------
        Array
            Training inputs, shape (nvars, n_train).
        """
        return self._X_train

    def y(self) -> Array:
        """
        Return training output data.

        Returns
        -------
        Array
            Training outputs, shape (nqoi, n_train).
        """
        return self._y_train

    def n_samples(self) -> int:
        """
        Return number of training samples.

        Returns
        -------
        int
            Number of training samples.
        """
        return self._n_train

    def nvars(self) -> int:
        """
        Return number of input variables.

        Returns
        -------
        int
            Number of input dimensions.
        """
        return self._nvars

    def nqoi(self) -> int:
        """
        Return number of output variables.

        Returns
        -------
        int
            Number of output dimensions (quantities of interest).
        """
        return self._nqoi

    def output_transform(
        self,
    ) -> Optional[OutputAffineTransformProtocol[Array]]:
        """
        Return the output transform, or None if not set.

        Returns
        -------
        Optional[OutputAffineTransformProtocol[Array]]
            The output affine transform used to scale training outputs.
        """
        return self._output_transform

    def __repr__(self) -> str:
        """
        Return string representation of training data.

        Returns
        -------
        str
            Description of training data dimensions.
        """
        return (
            f"GPTrainingData(n_samples={self._n_train}, "
            f"nvars={self._nvars}, nqoi={self._nqoi})"
        )
