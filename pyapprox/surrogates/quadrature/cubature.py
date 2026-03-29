"""Stroud cubature rules for hypercubes.

This module provides Stroud's cubature rules for integration over
hypercubes [-1, 1]^d. These rules are exact for polynomials up to
a specified degree.

Available rules:
- CdD2: Degree 2 rule (2*d points)
- CdD3: Degree 3 rule (2^d points)
- CdD5: Degree 5 rule (2*d^2 + 1 points)

Reference:
    Stroud, A.H. "Approximate Calculation of Multiple Integrals",
    Prentice-Hall, 1971.
"""

import math
from typing import Callable, Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class StroudCdD2(Generic[Array]):
    """Stroud's degree 2 cubature rule for hypercubes.

    Uses 2*d points, exact for polynomials up to degree 2.
    Points are at (+/- r, 0, ..., 0), (0, +/- r, 0, ...), etc.
    where r = sqrt(d/3).

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of variables.

    Examples
    --------
    >>> from pyapprox.util.backends.numpy import NumpyBkd
    >>> bkd = NumpyBkd()
    >>> rule = StroudCdD2(bkd, nvars=3)
    >>> samples, weights = rule()
    >>> print(f"nsamples = {samples.shape[1]}")
    nsamples = 6
    """

    def __init__(self, bkd: Backend[Array], nvars: int):
        self._bkd = bkd
        self._nvars = nvars
        self._samples, self._weights = self._build_rule()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nsamples(self) -> int:
        """Return the number of samples."""
        return 2 * self._nvars

    def _build_rule(self) -> Tuple[Array, Array]:
        """Build cubature points and weights."""
        d = self._nvars
        nsamples = 2 * d

        # r = sqrt(d/3) for [-1,1]^d
        r = math.sqrt(d / 3.0)

        # Build samples: +/- r along each axis
        samples = self._bkd.zeros((d, nsamples))
        for dim in range(d):
            samples[dim, 2 * dim] = r
            samples[dim, 2 * dim + 1] = -r

        # Equal weights summing to volume of [-1,1]^d = 2^d
        volume = 2.0**d
        weights = self._bkd.full((nsamples,), volume / nsamples)

        return samples, weights

    def __call__(self) -> Tuple[Array, Array]:
        """Return cubature samples and weights."""
        return self._bkd.copy(self._samples), self._bkd.copy(self._weights)

    def integrate(self, func: Callable[[Array], Array]) -> Array:
        """Integrate a function using this cubature rule."""
        values = func(self._samples)
        return self._bkd.sum(self._weights[:, None] * values, axis=0)

    def __repr__(self) -> str:
        return f"StroudCdD2(nvars={self._nvars}, nsamples={self.nsamples()})"


class StroudCdD3(Generic[Array]):
    """Stroud's degree 3 cubature rule for hypercubes.

    Uses 2^d points at the vertices of a scaled hypercube.
    Exact for polynomials up to degree 3.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of variables.
    """

    def __init__(self, bkd: Backend[Array], nvars: int):
        self._bkd = bkd
        self._nvars = nvars
        self._samples, self._weights = self._build_rule()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nsamples(self) -> int:
        """Return the number of samples."""
        return int(2**self._nvars)

    def _build_rule(self) -> Tuple[Array, Array]:
        """Build cubature points and weights."""
        d = self._nvars
        nsamples = 2**d

        # r = 1/sqrt(3) for [-1,1]^d
        r = 1.0 / math.sqrt(3.0)

        # Build samples at vertices of scaled hypercube
        samples = self._bkd.zeros((d, nsamples))
        for idx in range(nsamples):
            for dim in range(d):
                # Use bit representation to determine +/- r
                if (idx >> dim) & 1:
                    samples[dim, idx] = r
                else:
                    samples[dim, idx] = -r

        # Equal weights summing to volume of [-1,1]^d = 2^d
        volume = 2.0**d
        weights = self._bkd.full((nsamples,), volume / nsamples)

        return samples, weights

    def __call__(self) -> Tuple[Array, Array]:
        """Return cubature samples and weights."""
        return self._bkd.copy(self._samples), self._bkd.copy(self._weights)

    def integrate(self, func: Callable[[Array], Array]) -> Array:
        """Integrate a function using this cubature rule."""
        values = func(self._samples)
        return self._bkd.sum(self._weights[:, None] * values, axis=0)

    def __repr__(self) -> str:
        return f"StroudCdD3(nvars={self._nvars}, nsamples={self.nsamples()})"


class StroudCdD5(Generic[Array]):
    """Stroud's degree 5 cubature rule for hypercubes.

    Uses 2*d^2 + 1 points, exact for polynomials up to degree 5.
    Includes the origin plus points along coordinate axes and
    coordinate planes.

    Parameters
    ----------
    bkd : Backend[Array]
        Computational backend.
    nvars : int
        Number of variables.
    """

    def __init__(self, bkd: Backend[Array], nvars: int):
        self._bkd = bkd
        self._nvars = nvars
        self._samples, self._weights = self._build_rule()

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def nvars(self) -> int:
        """Return the number of variables."""
        return self._nvars

    def nsamples(self) -> int:
        """Return the number of samples."""
        return 2 * self._nvars**2 + 1

    def _build_rule(self) -> Tuple[Array, Array]:
        """Build cubature points and weights.

        Uses Hammer-Stroud formula with uniform weights (sum to 1).
        Multiply by 2^d to get weights summing to volume.
        """
        d = self._nvars
        nsamples = 2 * d * d + 1

        # Parameters for degree 5 rule (Hammer-Stroud)
        r = math.sqrt(3.0 / 5.0)

        # Weights (uniform, sum to 1)
        w0 = (25.0 * d * d - 115.0 * d + 162.0) / 162.0
        w1 = (70.0 - 25.0 * d) / 162.0
        w2 = 25.0 / 324.0

        samples = self._bkd.zeros((d, nsamples))
        weights = self._bkd.zeros((nsamples,))

        idx = 0

        # Center point
        weights[idx] = w0
        idx += 1

        # Points along axes: (+/- r, 0, ..., 0), etc.
        for dim in range(d):
            samples[dim, idx] = r
            weights[idx] = w1
            idx += 1

            samples[dim, idx] = -r
            weights[idx] = w1
            idx += 1

        # Points along diagonals: (+/- r, +/- r, 0, ..., 0), etc.
        for i in range(d):
            for j in range(i + 1, d):
                for ri in [-r, r]:
                    for rj in [-r, r]:
                        samples[i, idx] = ri
                        samples[j, idx] = rj
                        weights[idx] = w2
                        idx += 1

        # Scale weights by volume to integrate over [-1, 1]^d
        volume = 2.0**d
        weights = weights * volume

        return samples, weights

    def __call__(self) -> Tuple[Array, Array]:
        """Return cubature samples and weights."""
        return self._bkd.copy(self._samples), self._bkd.copy(self._weights)

    def integrate(self, func: Callable[[Array], Array]) -> Array:
        """Integrate a function using this cubature rule."""
        values = func(self._samples)
        return self._bkd.sum(self._weights[:, None] * values, axis=0)

    def __repr__(self) -> str:
        return f"StroudCdD5(nvars={self._nvars}, nsamples={self.nsamples()})"
