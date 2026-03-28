"""
Covariance hyperparameters using hyperspherical parameterization.

This module provides CovarianceHyperParameter, a special hyperparameter type
for learning covariance/correlation matrices in multi-output kernels using
hyperspherical coordinates. This ensures the resulting matrix is always
positive definite.
"""

import math
from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.hyperparameter import (
    HyperParameter,
    HyperParameterList,
)


class CovarianceHyperParameter(Generic[Array]):
    """
    Covariance matrix hyperparameter using hyperspherical parameterization.

    This class parameterizes a positive definite covariance matrix using
    hyperspherical coordinates via the Cholesky decomposition:
        Σ = L @ L^T

    where L is lower triangular. The elements of L are parameterized using:
    - Radii (diagonal elements): r_i > 0 for i = 0, ..., noutputs-1
    - Angles (off-diagonal): θ_ij ∈ (0, π) for i > j

    This ensures Σ is always positive definite during optimization.

    Mathematical Background
    -----------------------
    For a 2x2 matrix:
        L = [[r_0,    0  ],
             [r_1*cos(θ), r_1*sin(θ)]]

        Σ = [[r_0^2,              r_0*r_1*cos(θ)     ],
             [r_0*r_1*cos(θ),      r_1^2              ]]

    For larger matrices, the parameterization extends using n-sphere coordinates.

    Parameters
    ----------
    noutputs : int
        Dimensionality of the covariance matrix (noutputs x noutputs).
    radii_init : float, optional
        Initial value for all radii. Default: 1.0.
    radii_bounds : Tuple[float, float], optional
        Bounds for radii values. Default: (0.1, 10.0).
    angles_init : float, optional
        Initial value for all angles. Default: π/2.
    angles_bounds : Tuple[float, float], optional
        Bounds for angles. Default: (0.01, π - 0.01).
    bkd : Backend[Array]
        Backend for numerical computations.
    fixed : bool, optional
        If True, parameters are fixed. Default: False.

    Attributes
    ----------
    _noutputs : int
        Dimensionality of covariance matrix.
    _nradii : int
        Number of radii parameters (= noutputs).
    _nangles : int
        Number of angle parameters (= noutputs*(noutputs-1)/2).
    _hyp_list : HyperParameterList[Array]
        Combined hyperparameters for radii and angles.
    _covariance : Array or None
        Cached covariance matrix, updated when parameters change.

    Examples
    --------
    >>> from pyapprox.surrogates.kernels.multioutput.covariance_hyperparameter import
    CovarianceHyperParameter
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> # Create 3x3 covariance hyperparameter
    >>> cov_hyp = CovarianceHyperParameter(3, bkd=bkd)
    >>> # Get current covariance matrix
    >>> Sigma = cov_hyp.covariance()  # Shape: (3, 3)
    >>> # Access hyperparameters for optimization
    >>> hyp_list = cov_hyp.hyp_list()
    """

    def __init__(
        self,
        noutputs: int,
        radii_init: float = 1.0,
        radii_bounds: Tuple[float, float] = (0.1, 10.0),
        angles_init: float = math.pi / 2,
        angles_bounds: Tuple[float, float] = (0.01, math.pi - 0.01),
        bkd: Backend[Array] = None,
        fixed: bool = False,
    ):
        """
        Initialize CovarianceHyperParameter.

        Parameters
        ----------
        noutputs : int
            Dimensionality of covariance matrix.
        radii_init : float, optional
            Initial value for radii. Default: 1.0.
        radii_bounds : Tuple[float, float], optional
            Bounds for radii. Default: (0.1, 10.0).
        angles_init : float, optional
            Initial value for angles. Default: π/2.
        angles_bounds : Tuple[float, float], optional
            Bounds for angles. Default: (0.01, π - 0.01).
        bkd : Backend[Array]
            Backend for numerical computations.
        fixed : bool, optional
            If True, parameters are fixed. Default: False.
        """
        if bkd is None:
            raise ValueError("Backend must be provided")

        self._bkd = bkd
        self._noutputs = noutputs
        self._nradii = noutputs
        self._nangles = (noutputs * (noutputs - 1)) // 2

        # Create hyperparameters for radii
        hyperparams = []
        for i in range(self._nradii):
            hyp = HyperParameter(
                name=f"radius_{i}",
                nparams=1,
                values=radii_init,
                bounds=radii_bounds,
                bkd=bkd,
                fixed=fixed,
            )
            hyperparams.append(hyp)

        # Create hyperparameters for angles
        for i in range(self._nangles):
            hyp = HyperParameter(
                name=f"angle_{i}",
                nparams=1,
                values=angles_init,
                bounds=angles_bounds,
                bkd=bkd,
                fixed=fixed,
            )
            hyperparams.append(hyp)

        self._hyp_list = HyperParameterList(hyperparams, bkd=bkd)
        self._covariance = None  # Cached covariance matrix
        self._update_covariance()

    def bkd(self) -> Backend[Array]:
        """Return the backend."""
        return self._bkd

    def hyp_list(self) -> HyperParameterList[Array]:
        """Return the hyperparameter list."""
        return self._hyp_list

    def noutputs(self) -> int:
        """Return the dimensionality of the covariance matrix."""
        return self._noutputs

    def _get_radii(self) -> Array:
        """Get current radii values."""
        hyps = self._hyp_list.hyperparameters()
        radii = [hyps[i].get_values()[0] for i in range(self._nradii)]
        return self._bkd.array(radii)

    def _get_angles(self) -> Array:
        """Get current angle values."""
        hyps = self._hyp_list.hyperparameters()
        angles = [hyps[self._nradii + i].get_values()[0] for i in range(self._nangles)]
        return self._bkd.array(angles)

    def _hypersphere_to_cholesky(self) -> Array:
        """
        Convert hyperspherical coordinates to Cholesky factor.

        Returns
        -------
        L : Array
            Lower triangular Cholesky factor, shape (noutputs, noutputs).
        """
        radii = self._get_radii()
        angles = self._get_angles()

        L = self._bkd.zeros((self._noutputs, self._noutputs))

        # First row: just the radius
        L[0, 0] = radii[0]

        # Remaining rows use n-sphere parameterization
        angle_idx = 0
        for i in range(1, self._noutputs):
            # For row i, we need i+1 elements (including diagonal)
            # Use i angles to determine i elements, then radius determines last

            # Product of sines for normalization
            sin_product = radii[i]

            for j in range(i):
                theta = angles[angle_idx]
                L[i, j] = sin_product * self._bkd.cos(theta)
                sin_product = sin_product * self._bkd.sin(theta)
                angle_idx += 1

            # Diagonal element
            L[i, i] = sin_product

        return L

    def _update_covariance(self) -> None:
        """Update the cached covariance matrix."""
        L = self._hypersphere_to_cholesky()
        self._covariance = L @ L.T

    def covariance(self, update: bool = True) -> Array:
        """
        Get the covariance matrix.

        Parameters
        ----------
        update : bool, optional
            If True, recompute from current hyperparameters. Default: True.

        Returns
        -------
        Sigma : Array
            Covariance matrix, shape (noutputs, noutputs).
        """
        if update or self._covariance is None:
            self._update_covariance()
        return self._covariance

    def correlation(self, update: bool = True) -> Array:
        """
        Get the correlation matrix (normalized covariance).

        Parameters
        ----------
        update : bool, optional
            If True, recompute from current hyperparameters. Default: True.

        Returns
        -------
        R : Array
            Correlation matrix, shape (noutputs, noutputs).
        """
        Sigma = self.covariance(update=update)
        stds = self._bkd.sqrt(self._bkd.diag(Sigma))
        R = Sigma / (stds[:, None] @ stds[None, :])
        return R

    def cholesky_factor(self, update: bool = True) -> Array:
        """
        Get the Cholesky factor L where Σ = L @ L^T.

        Parameters
        ----------
        update : bool, optional
            If True, recompute from current hyperparameters. Default: True.

        Returns
        -------
        L : Array
            Lower triangular Cholesky factor, shape (noutputs, noutputs).
        """
        if update:
            self._update_covariance()
        return self._hypersphere_to_cholesky()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CovarianceHyperParameter(\n"
            f"  noutputs={self._noutputs},\n"
            f"  nradii={self._nradii},\n"
            f"  nangles={self._nangles}\n"
            f")"
        )
