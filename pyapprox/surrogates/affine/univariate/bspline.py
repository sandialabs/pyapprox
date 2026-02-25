"""B-spline univariate basis functions.

This module provides B-spline bases for hierarchical sparse grids
and other approximation methods. B-splines offer smooth (C^{p-1})
approximation with local support.

Classes:
    BSpline1D: Standard B-spline basis on uniform knots
    HierarchicalBSpline1D: Hierarchical B-splines for adaptive refinement
"""

from typing import Generic, Optional, Tuple

import numpy as np
from scipy.interpolate import BSpline as ScipyBSpline

from pyapprox.util.backends.protocols import Array, Backend


def _evaluate_bspline_basis(
    bkd: Backend[Array],
    x: Array,
    i: int,
    degree: int,
    knots: Array,
    derivative: int = 0,
) -> Array:
    """Evaluate B-spline basis function using scipy.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    x : Array
        Evaluation points. Shape: (npoints,)
    i : int
        Basis function index.
    degree : int
        Degree of B-spline.
    knots : Array
        Knot vector. Shape: (nknots,)
    derivative : int
        Derivative order. Default: 0 (function value).

    Returns
    -------
    Array
        B-spline values. Shape: (npoints,)
    """
    x_np = bkd.to_numpy(x)
    knots_np = bkd.to_numpy(knots)
    nterms = len(knots_np) - degree - 1

    # Create coefficient vector with 1 at position i
    coeffs = np.zeros(nterms)
    coeffs[i] = 1.0

    # Create scipy BSpline and evaluate
    spline = ScipyBSpline(knots_np, coeffs, degree, extrapolate=False)

    if derivative == 0:
        vals = spline(x_np)
    else:
        vals = spline.derivative(derivative)(x_np)

    # Handle NaN at boundaries (scipy returns NaN outside domain)
    vals = np.nan_to_num(vals, nan=0.0)

    return bkd.asarray(vals)


class BSpline1D(Generic[Array]):
    """B-spline basis on uniform knots.

    Provides B-spline basis functions on the interval [0, 1] with
    uniform knot spacing. Supports evaluation and first/second derivatives.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    degree : int
        Polynomial degree. Default: 3 (cubic).
    nterms : int, optional
        Number of basis functions. If not provided, determined by knots.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> basis = BSpline1D(bkd, degree=3, nterms=5)
    >>> samples = bkd.linspace(0, 1, 10).reshape(1, -1)
    >>> values = basis(samples)  # Shape: (10, 5)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        degree: int = 3,
        nterms: Optional[int] = None,
    ):
        self._bkd = bkd
        self._degree = degree
        self._nterms = nterms if nterms is not None else degree + 1
        self._knots: Optional[Array] = None
        self._setup_knots()

    def _setup_knots(self) -> None:
        """Set up uniform knot vector with open knot repetition."""
        p = self._degree
        n = self._nterms

        # Open uniform knot vector
        # Number of interior knots
        n_interior = n - p - 1
        if n_interior < 0:
            n_interior = 0

        # Knots: p+1 zeros, interior, p+1 ones
        knots_list = [0.0] * (p + 1)
        if n_interior > 0:
            for i in range(1, n_interior + 1):
                knots_list.append(i / (n_interior + 1))
        knots_list.extend([1.0] * (p + 1))

        self._knots = self._bkd.asarray(knots_list)

    @property
    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def degree(self) -> int:
        """Return the polynomial degree."""
        return self._degree

    def nterms(self) -> int:
        """Return the number of basis functions."""
        return self._nterms

    def set_nterms(self, nterms: int) -> None:
        """Set the number of basis functions."""
        self._nterms = nterms
        self._setup_knots()

    def knots(self) -> Array:
        """Return the knot vector."""
        if self._knots is None:
            self._setup_knots()
        return self._knots

    def __call__(self, samples: Array) -> Array:
        """Evaluate B-spline basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples)

        Returns
        -------
        Array
            Basis values. Shape: (nsamples, nterms)
        """
        x = samples[0, :]
        nsamples = x.shape[0]
        result = self._bkd.zeros((nsamples, self._nterms))

        for i in range(self._nterms):
            vals = _evaluate_bspline_basis(
                self._bkd, x, i, self._degree, self._knots, derivative=0
            )
            result[:, i] = vals

        return result

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate first derivatives of B-spline basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (1, nsamples).
        """
        x = samples[0, :]
        nsamples = x.shape[0]
        result = self._bkd.zeros((nsamples, self._nterms))

        for i in range(self._nterms):
            vals = _evaluate_bspline_basis(
                self._bkd, x, i, self._degree, self._knots, derivative=1
            )
            result[:, i] = vals

        return result

    def hessian_batch(self, samples: Array) -> Array:
        """Evaluate second derivatives of B-spline basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            Second derivatives. Shape: (nsamples, nterms)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (1, nsamples).
        """
        x = samples[0, :]
        nsamples = x.shape[0]
        result = self._bkd.zeros((nsamples, self._nterms))

        for i in range(self._nterms):
            vals = _evaluate_bspline_basis(
                self._bkd, x, i, self._degree, self._knots, derivative=2
            )
            result[:, i] = vals

        return result

    def __repr__(self) -> str:
        return f"BSpline1D(degree={self._degree}, nterms={self._nterms})"


class HierarchicalBSpline1D(Generic[Array]):
    """Hierarchical B-spline basis for adaptive refinement.

    Provides a hierarchical structure of B-splines where each level
    adds basis functions at finer scales. This is suitable for locally
    adaptive sparse grids.

    At level 0: single basis spanning [0, 1]
    At level l: 2^l basis functions at scale 1/2^l

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    degree : int
        Polynomial degree. Default: 3 (cubic).
    max_level : int
        Maximum hierarchical level. Default: 5.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> basis = HierarchicalBSpline1D(bkd, degree=3, max_level=3)
    >>> # Evaluate specific hierarchical basis
    >>> samples = bkd.linspace(0, 1, 10).reshape(1, -1)
    >>> values = basis.evaluate_hierarchical(samples, level=1, index=0)
    """

    def __init__(
        self,
        bkd: Backend[Array],
        degree: int = 3,
        max_level: int = 5,
    ):
        self._bkd = bkd
        self._degree = degree
        self._max_level = max_level
        self._nterms = 1  # Default: just level 0

    @property
    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def degree(self) -> int:
        """Return the polynomial degree."""
        return self._degree

    def max_level(self) -> int:
        """Return the maximum hierarchical level."""
        return self._max_level

    def nterms(self) -> int:
        """Return current number of terms being evaluated."""
        return self._nterms

    def set_nterms(self, nterms: int) -> None:
        """Set number of terms to evaluate (for interface compatibility)."""
        self._nterms = nterms

    def nbasis_at_level(self, level: int) -> int:
        """Return number of basis functions at a given level.

        Level 0: 1 basis
        Level l > 0: 2^(l-1) bases (for l > 0)
        """
        if level == 0:
            return 1
        return 2 ** (level - 1)

    def total_basis_up_to_level(self, level: int) -> int:
        """Return total basis functions up to and including level."""
        if level == 0:
            return 1
        # 1 + 1 + 2 + 4 + ... + 2^(l-1) = 2^l
        return 2**level

    def level_index_to_flat(self, level: int, index: int) -> int:
        """Convert (level, index) to flat index.

        Parameters
        ----------
        level : int
            Hierarchical level (0-indexed).
        index : int
            Index within level (0-indexed).

        Returns
        -------
        int
            Flat index.
        """
        if level == 0:
            return 0
        return 2 ** (level - 1) + index

    def flat_to_level_index(self, flat_idx: int) -> Tuple[int, int]:
        """Convert flat index to (level, index).

        Parameters
        ----------
        flat_idx : int
            Flat index.

        Returns
        -------
        Tuple[int, int]
            (level, index) pair.
        """
        if flat_idx == 0:
            return 0, 0

        import math

        # Find level: flat_idx in [2^(l-1), 2^l)
        level = int(math.floor(math.log2(flat_idx))) + 1
        index = flat_idx - 2 ** (level - 1)
        return level, index

    def _get_knots_for_level(self, level: int) -> Array:
        """Get knot vector for a hierarchical level."""
        p = self._degree

        if level == 0:
            # Single basis spanning [0, 1]
            knots = [0.0] * (p + 1) + [1.0] * (p + 1)
        else:
            # Finer scale knots
            h = 1.0 / (2**level)
            n_internal = 2**level - 1

            knots = [0.0] * (p + 1)
            for i in range(1, n_internal + 1):
                knots.append(i * h)
            knots.extend([1.0] * (p + 1))

        return self._bkd.asarray(knots)

    def evaluate_hierarchical(self, samples: Array, level: int, index: int) -> Array:
        """Evaluate a single hierarchical basis function.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples)
        level : int
            Hierarchical level.
        index : int
            Index within level.

        Returns
        -------
        Array
            Basis values. Shape: (nsamples,)
        """
        x = samples[0, :]
        knots = self._get_knots_for_level(level)
        basis_idx = 0 if level == 0 else index
        return _evaluate_bspline_basis(
            self._bkd, x, basis_idx, self._degree, knots, derivative=0
        )

    def evaluate_hierarchical_derivative(
        self, samples: Array, level: int, index: int
    ) -> Array:
        """Evaluate first derivative of a hierarchical basis function.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples)
        level : int
            Hierarchical level.
        index : int
            Index within level.

        Returns
        -------
        Array
            Derivative values. Shape: (nsamples,)
        """
        x = samples[0, :]
        knots = self._get_knots_for_level(level)
        basis_idx = 0 if level == 0 else index
        return _evaluate_bspline_basis(
            self._bkd, x, basis_idx, self._degree, knots, derivative=1
        )

    def __call__(self, samples: Array) -> Array:
        """Evaluate all basis functions up to current nterms.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples)

        Returns
        -------
        Array
            Basis values. Shape: (nsamples, nterms)
        """
        nsamples = samples.shape[1]
        result = self._bkd.zeros((nsamples, self._nterms))

        for flat_idx in range(self._nterms):
            level, index = self.flat_to_level_index(flat_idx)
            vals = self.evaluate_hierarchical(samples, level, index)
            result[:, flat_idx] = vals

        return result

    def jacobian_batch(self, samples: Array) -> Array:
        """Evaluate first derivatives of all basis functions.

        Parameters
        ----------
        samples : Array
            Sample points. Shape: (1, nsamples). Must be 2D.

        Returns
        -------
        Array
            First derivatives. Shape: (nsamples, nterms)

        Raises
        ------
        ValueError
            If samples is not 2D with shape (1, nsamples).
        """
        nsamples = samples.shape[1]
        result = self._bkd.zeros((nsamples, self._nterms))

        for flat_idx in range(self._nterms):
            level, index = self.flat_to_level_index(flat_idx)
            vals = self.evaluate_hierarchical_derivative(samples, level, index)
            result[:, flat_idx] = vals

        return result

    def __repr__(self) -> str:
        return (
            f"HierarchicalBSpline1D(degree={self._degree}, "
            f"max_level={self._max_level}, nterms={self._nterms})"
        )
