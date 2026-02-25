"""
Base class for design matrix computation.

Design matrices M0 and M1 aggregate information from individual design points
for use in local OED criteria.
"""

from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class DesignMatricesBase(ABC, Generic[Array]):
    """
    Base class for design matrix computation.

    Computes M0 and M1 matrices from design weights. Includes caching
    to avoid recomputation when weights haven't changed.

    Parameters
    ----------
    design_factors : Array
        Basis function values at design points. Shape: (ndesign_pts, ndesign_vars)
    noise_mult : Array, optional
        Noise multipliers at each design point. Shape: (ndesign_pts,)
        If None, assumes homoscedastic noise.
    bkd : Backend[Array]
        Computational backend.
    """

    def __init__(
        self,
        design_factors: Array,
        bkd: Backend[Array],
        noise_mult: Optional[Array] = None,
    ) -> None:
        self._bkd = bkd
        self._validate_inputs(design_factors, noise_mult)
        self._design_factors = design_factors
        self._noise_mult = noise_mult
        self._cache: Dict[str, Tuple[Array, Array]] = {}

        # Compute individual design matrices
        self._M0k, self._M1k = self._compute_individual_design_matrices()

    def _validate_inputs(
        self, design_factors: Array, noise_mult: Optional[Array]
    ) -> None:
        """Validate input shapes."""
        if design_factors.ndim != 2:
            raise ValueError("design_factors must be a 2D array")
        ndesign_pts = design_factors.shape[0]
        if noise_mult is not None and (
            noise_mult.ndim != 1 or noise_mult.shape[0] != ndesign_pts
        ):
            raise ValueError(
                f"noise_mult must be 1D with shape ({ndesign_pts},), "
                f"got shape {noise_mult.shape}"
            )

    def bkd(self) -> Backend[Array]:
        """Get the backend."""
        return self._bkd

    def ndesign_pts(self) -> int:
        """Number of candidate design points."""
        return self._design_factors.shape[0]

    def ndesign_vars(self) -> int:
        """Number of design variables (basis function dimension)."""
        return self._design_factors.shape[1]

    def is_homoscedastic(self) -> bool:
        """Whether noise is constant across design points."""
        return self._noise_mult is None

    def design_factors(self) -> Array:
        """Get the design factors matrix."""
        return self._design_factors

    def noise_mult(self) -> Optional[Array]:
        """Get the noise multipliers."""
        return self._noise_mult

    @abstractmethod
    def _compute_individual_design_matrices(self) -> Tuple[Array, Array]:
        """
        Compute individual design matrices M0k and M1k.

        Returns
        -------
        M0k : Array
            Individual M0 matrices. Shape: (ndesign_pts, ndesign_vars, ndesign_vars)
        M1k : Array
            Individual M1 matrices. Shape: (ndesign_pts, ndesign_vars, ndesign_vars)
        """
        raise NotImplementedError

    def _cache_valid(self, name: str, design_weights: Array) -> bool:
        """Check if cached value is still valid."""
        if name not in self._cache:
            return False
        cached_weights, _ = self._cache[name]
        return bool(
            self._bkd.allclose(cached_weights, design_weights, atol=1e-15, rtol=1e-15)
        )

    def M0k(self) -> Array:
        """
        Get individual design matrices for M0.

        Returns
        -------
        Array
            Individual M0 matrices. Shape: (ndesign_pts, ndesign_vars, ndesign_vars)
        """
        return self._M0k

    def M1k(self) -> Array:
        """
        Get individual design matrices for M1.

        Returns
        -------
        Array
            Individual M1 matrices. Shape: (ndesign_pts, ndesign_vars, ndesign_vars)
        """
        return self._M1k

    def M0(self, design_weights: Array) -> Array:
        """
        Compute weighted information matrix M0.

        M0 = sum_k w_k * M0_k

        Parameters
        ----------
        design_weights : Array
            Design weights. Shape: (ndesign_pts, 1)

        Returns
        -------
        Array
            M0 matrix. Shape: (ndesign_vars, ndesign_vars)
        """
        if not self._cache_valid("M0", design_weights):
            M0 = self._bkd.einsum("i,ijk->jk", design_weights[:, 0], self._M0k)
            self._cache["M0"] = (self._bkd.copy(design_weights), M0)
        return self._cache["M0"][1]

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
        if not self._cache_valid("M1", design_weights):
            M1 = self._bkd.einsum("i,ijk->jk", design_weights[:, 0], self._M1k)
            self._cache["M1"] = (self._bkd.copy(design_weights), M1)
        return self._cache["M1"][1]

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
        if not self._cache_valid("M1inv", design_weights):
            M1 = self.M1(design_weights)
            M1inv = self._bkd.inv(M1)
            self._cache["M1inv"] = (self._bkd.copy(design_weights), M1inv)
        return self._cache["M1inv"][1]
