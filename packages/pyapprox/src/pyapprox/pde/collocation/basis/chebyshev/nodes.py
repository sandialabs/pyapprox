"""Chebyshev-Gauss-Lobatto collocation nodes.

Provides Chebyshev nodes for spectral collocation methods.
"""

from typing import Generic

import numpy as np

from pyapprox.util.backends.protocols import Array, Backend


class ChebyshevGaussLobattoNodes1D(Generic[Array]):
    """Generator for Chebyshev-Gauss-Lobatto nodes on [-1, 1].

    These are the extrema of the Chebyshev polynomial T_{n-1}(x),
    including the endpoints. They cluster near the boundaries,
    which helps avoid Runge's phenomenon in polynomial interpolation.

    Nodes: x_j = cos(j * pi / (n-1)), j = 0, ..., n-1

    Parameters
    ----------
    bkd : Backend
        Computational backend.
    """

    def __init__(self, bkd: Backend[Array]):
        self._bkd = bkd

    def bkd(self) -> Backend[Array]:
        """Return the computational backend."""
        return self._bkd

    def generate(self, npts: int) -> Array:
        """Generate Chebyshev-Gauss-Lobatto nodes.

        Parameters
        ----------
        npts : int
            Number of nodes to generate. Must be >= 1.

        Returns
        -------
        Array
            Nodes on [-1, 1]. Shape: (npts,)
            Ordered from x_0 = 1 to x_{n-1} = -1 (decreasing).
        """
        if npts < 1:
            raise ValueError(f"npts must be >= 1, got {npts}")

        if npts == 1:
            return self._bkd.zeros((1,))

        # x_j = cos(j * pi / (n-1)), j = 0, ..., n-1
        idx = self._bkd.arange(npts, dtype=float)
        nodes = self._bkd.cos(idx * np.pi / (npts - 1))
        return nodes
