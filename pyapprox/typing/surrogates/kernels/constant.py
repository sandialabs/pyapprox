"""
Constant kernel for scaling other kernels or modeling constant mean.

This module provides the ConstantKernel, which returns a constant value for
all input pairs. It's useful for scaling other kernels through multiplication
or adding a constant offset through addition.
"""

from typing import Tuple

from pyapprox.typing.util.hyperparameter import LogHyperParameter
from pyapprox.typing.util.hyperparameter import HyperParameterList
from pyapprox.typing.util.backends.protocols import Array, Backend
from pyapprox.typing.surrogates.kernels.protocols import Kernel


class ConstantKernel(Kernel):
    """
    Constant kernel that returns a constant value for all input pairs.

    K(X1, X2) = constant_value

    This kernel is useful for:
    - Scaling other kernels through multiplication
    - Adding a constant offset through addition
    - Modeling a constant mean function

    Parameters
    ----------
    constant_value : float
        The constant value returned by the kernel.
    constant_bounds : Tuple[float, float]
        Bounds for the constant value parameter.
    bkd : Backend
        Backend for numerical computations.
    fixed : bool, optional
        Whether the hyperparameter is fixed (default is False).

    Examples
    --------
    >>> from pyapprox.typing.surrogates.kernels import MaternKernel, ConstantKernel
    >>> from pyapprox.typing.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> matern = MaternKernel(2.5, [1.0, 1.0], (0.1, 10.0), 2, bkd)
    >>> constant = ConstantKernel(2.0, (0.1, 10.0), bkd)
    >>> scaled_kernel = matern * constant  # Scale Matern kernel by constant
    """

    def __init__(
        self,
        constant_value: float,
        constant_bounds: Tuple[float, float],
        bkd: Backend[Array],
        fixed: bool = False,
    ):
        """
        Initialize the ConstantKernel.

        Parameters
        ----------
        constant_value : float
            The constant value returned by the kernel.
        constant_bounds : Tuple[float, float]
            Bounds for the constant value parameter.
        bkd : Backend[Array]
            Backend for numerical computations.
        fixed : bool, optional
            Whether the hyperparameter is fixed (default is False).
        """
        super().__init__(bkd)

        # Use LogHyperParameter to ensure constant is positive
        self._log_constant = LogHyperParameter(
            "constant",
            1,  # Scalar parameter
            [constant_value],
            constant_bounds,
            bkd=self._bkd,
            fixed=fixed,
        )
        self._hyp_list = HyperParameterList([self._log_constant])

    def hyp_list(self) -> HyperParameterList:
        """
        Return the list of hyperparameters associated with the kernel.

        Returns
        -------
        hyp_list : HyperParameterList
            List of hyperparameters.
        """
        return self._hyp_list

    def nvars(self) -> int:
        """
        Return the number of input variables.

        For ConstantKernel, this is inferred from the input data shape
        and not stored as an attribute.

        Returns
        -------
        nvars : int
            Returns 0 as ConstantKernel doesn't depend on input dimensions.
        """
        # ConstantKernel doesn't have spatial dependence, so nvars is ambiguous.
        # We return 0 to indicate it works with any number of dimensions.
        return 0

    def diag(self, X1: Array) -> Array:
        """
        Return the diagonal of the kernel matrix.

        For ConstantKernel, the diagonal is a vector of constant values.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n).

        Returns
        -------
        diag : Array
            Diagonal of the kernel matrix, shape (n,).
        """
        n = X1.shape[1]
        constant_value = self._log_constant.exp_values()[0]
        return self._bkd.full((n,), constant_value)

    def __call__(self, X1: Array, X2: Array = None) -> Array:
        """
        Compute the constant kernel matrix.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n1).
        X2 : Array, optional
            Input data, shape (nvars, n2). If None, uses X1.

        Returns
        -------
        K : Array
            Constant kernel matrix, shape (n1, n2) or (n1, n1).
        """
        n1 = X1.shape[1]

        if X2 is None:
            n2 = n1
        else:
            n2 = X2.shape[1]

        constant_value = self._log_constant.exp_values()[0]
        return self._bkd.full((n1, n2), constant_value)

    def jacobian(self, X1: Array, X2: Array) -> Array:
        """
        Compute Jacobian of constant kernel w.r.t. inputs.

        Since the constant kernel has no spatial dependence,
        the Jacobian is zero everywhere.

        Parameters
        ----------
        X1 : Array
            Input data, shape (nvars, n1).
        X2 : Array
            Input data, shape (nvars, n2).

        Returns
        -------
        jac : Array
            Jacobian, shape (n1, n2, nvars). All zeros.
        """
        n1 = X1.shape[1]
        n2 = X2.shape[1]
        nvars = X1.shape[0]

        return self._bkd.zeros((n1, n2, nvars))

    def jacobian_wrt_params(self, samples: Array) -> Array:
        """
        Compute Jacobian of constant kernel w.r.t. hyperparameters.

        For ConstantKernel with log-parameterization:
        K = c
        log_c = log(c)
        dK/d(log_c) = dK/dc * dc/d(log_c) = 1 * c = c

        Parameters
        ----------
        samples : Array
            Input data, shape (nvars, n).

        Returns
        -------
        jac : Array
            Jacobian, shape (n, n, 1). All entries equal to constant value.
        """
        n = samples.shape[1]
        constant_value = self._log_constant.exp_values()[0]

        # dK/d(log_c) = c for all entries
        jac = self._bkd.full((n, n, 1), constant_value)
        return jac
