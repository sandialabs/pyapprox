"""
Protocols for design matrix computation.

Design matrices M0 and M1 are used in local OED criteria for linear regression.
For least squares regression with noise variance sigma^2:
    - M1 = sum_k w_k * phi_k * phi_k^T  (information matrix)
    - M0 = sum_k w_k * sigma_k^2 * phi_k * phi_k^T  (weighted information)

where phi_k are basis function values at design point k.
"""

from typing import Protocol, Generic, runtime_checkable

from pyapprox.util.backends.protocols import Array, Backend


@runtime_checkable
class DesignMatricesProtocol(Protocol, Generic[Array]):
    """
    Protocol for design matrix computation.

    Computes M0 and M1 matrices from design weights for use in OED criteria.
    These matrices aggregate information from individual design points.

    Methods
    -------
    bkd()
        Get the computational backend.
    ndesign_pts()
        Number of candidate design points.
    ndesign_vars()
        Number of design variables (basis functions).
    is_homoscedastic()
        Whether noise is constant across design points.
    M0(design_weights)
        Compute weighted information matrix M0.
    M1(design_weights)
        Compute information matrix M1.
    M1inv(design_weights)
        Compute inverse of M1.
    M0k()
        Get individual design matrices for M0.
    M1k()
        Get individual design matrices for M1.
    """

    def bkd(self) -> Backend[Array]:
        """Get the backend used for computations."""
        ...

    def ndesign_pts(self) -> int:
        """
        Number of candidate design points.

        Returns
        -------
        int
            Number of candidate design points.
        """
        ...

    def ndesign_vars(self) -> int:
        """
        Number of design variables (basis function dimension).

        Returns
        -------
        int
            Dimension of basis functions.
        """
        ...

    def is_homoscedastic(self) -> bool:
        """
        Whether noise is constant across design points.

        Returns
        -------
        bool
            True if noise variance is the same at all design points.
        """
        ...

    def M0(self, design_weights: Array) -> Array:
        """
        Compute weighted information matrix M0.

        M0 = sum_k w_k * M0_k

        For homoscedastic noise, M0 = M1.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (ndesign_pts, 1)

        Returns
        -------
        Array
            M0 matrix. Shape: (ndesign_vars, ndesign_vars)
        """
        ...

    def M1(self, design_weights: Array) -> Array:
        """
        Compute information matrix M1.

        M1 = sum_k w_k * M1_k

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (ndesign_pts, 1)

        Returns
        -------
        Array
            M1 matrix. Shape: (ndesign_vars, ndesign_vars)
        """
        ...

    def M1inv(self, design_weights: Array) -> Array:
        """
        Compute inverse of M1.

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (ndesign_pts, 1)

        Returns
        -------
        Array
            Inverse of M1. Shape: (ndesign_vars, ndesign_vars)
        """
        ...

    def M0k(self) -> Array:
        """
        Get individual design matrices for M0.

        Returns
        -------
        Array
            Individual M0 matrices. Shape: (ndesign_pts, ndesign_vars, ndesign_vars)
        """
        ...

    def M1k(self) -> Array:
        """
        Get individual design matrices for M1.

        Returns
        -------
        Array
            Individual M1 matrices. Shape: (ndesign_pts, ndesign_vars, ndesign_vars)
        """
        ...
