from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class PiecewiseQuadratic(Generic[Array]):
    """
    Class for Piecewise quadratic basis functions and
    quadrature weights.
    """

    def __init__(self, nodes: Array, bkd: Backend[Array]):
        """
        Initialize the IrregularPiecewiseQuadratic object.

        Parameters
        ----------
        nodes : Array
            The nodes (abscissas) where the piecewise quadratic basis functions
        are defined.
        bkd : Backend
            Backend used for computations (e.g., NumPy or PyTorch).
        """
        assert nodes.ndim == 1, "Nodes must be a 1D array."
        if nodes.shape[0] % 2 != 1:
            raise ValueError("Nodes must contain an odd number of elements.")
        self._nodes = nodes
        self._bkd = bkd

    def __call__(self, xx: Array) -> Array:
        """
        Evaluate the piecewise quadratic basis functions at given points.

        Parameters
        ----------
        xx : Array
            Points where the basis functions are evaluated.

        Returns
        -------
        Array
            Values of the basis functions at the given points.
        """
        assert xx.ndim == 1, "Input points must be a 1D array."
        bkd = self._bkd
        nodes = self._nodes
        nnodes = nodes.shape[0]
        npts = xx.shape[0]

        if nnodes == 1:
            return bkd.ones((npts, nnodes))

        vals = bkd.zeros((npts, nnodes))

        # Elements are pairs: [0,1,2], [2,3,4], [4,5,6], ...
        # Each element has 3 nodes: left (even), middle (odd), right (even)
        n_elements = (nnodes - 1) // 2

        # Find element index using even-indexed nodes as element boundaries
        even_nodes = nodes[::2]  # nodes 0, 2, 4, ...
        elem_idx = bkd.searchsorted(even_nodes, xx, side="right") - 1
        elem_idx = bkd.clip(elem_idx, 0, n_elements - 1)

        # Get node indices for each element
        left_idx = 2 * elem_idx  # even: 0, 2, 4, ...
        mid_idx = left_idx + 1  # odd: 1, 3, 5, ...
        right_idx = left_idx + 2  # even: 2, 4, 6, ...

        # Get node values
        xl = nodes[left_idx]
        xm = nodes[mid_idx]
        xr = nodes[right_idx]

        # Compute Lagrange basis values for quadratic interpolation
        pt_indices = bkd.arange(npts)

        # Left basis (at even node): L_0(x) = (x-xm)(x-xr) / (xl-xm)(xl-xr)
        vals[pt_indices, left_idx] = (xx - xm) / (xl - xm) * (xx - xr) / (xl - xr)

        # Middle basis (at odd node): L_1(x) = (x-xl)(x-xr) / (xm-xl)(xm-xr)
        vals[pt_indices, mid_idx] = (xx - xl) / (xm - xl) * (xx - xr) / (xm - xr)

        # Right basis (at even node): L_2(x) = (x-xl)(x-xm) / (xr-xl)(xr-xm)
        vals[pt_indices, right_idx] = (xx - xl) / (xr - xl) * (xx - xm) / (xr - xm)

        return vals

    def _quadrature_weights(self) -> Array:
        """
        Compute quadrature weights for the given nodes.

        Returns
        -------
        Array
            Quadrature weights for the nodes.
        """
        nnodes = self._nodes.shape[0]
        if nnodes == 1:
            raise ValueError(
                "Cannot compute weights from a single point without bounds."
            )
        weights = [0.0 for ii in range(nnodes)]
        for ii in range(nnodes):
            if ii % 2 == 1:
                xl, xm, xr = self._nodes[ii - 1 : ii + 2]
                weights[ii] = (xl - xr) ** 3 / (6 * (xm - xl) * (xm - xr))
                continue
            if ii < nnodes - 2:
                xl, xm, xr = self._nodes[ii : ii + 3]
                weights[ii] += ((xr - xl) * (2 * xl - 3 * xm + xr)) / (6 * (xl - xm))
            if ii > 1:
                xl, xm, xr = self._nodes[ii - 2 : ii + 1]
                weights[ii] += ((xl - xr) * (xl - 3 * xm + 2 * xr)) / (6 * (xm - xr))
        return self._bkd.asarray(weights)

    def quadrature_rule(self) -> Tuple[Array, Array]:
        """
        Compute quadrature points and weights based on the nodes.

        Returns
        -------
        Tuple[Array, Array]
            Quadrature points and weights.
        """
        quadrature_weights = self._quadrature_weights()
        return self._nodes, quadrature_weights

    def nodes(self) -> Array:
        """
        Return the nodes.

        Returns
        -------
        Array
            The nodes (abscissas) where the piecewise linear basis functions
            are defined.
        """
        return self._nodes

    def bkd(self) -> Backend[Array]:
        """
        Return the backend used for computations.

        Returns
        -------
        Backend[Array]
            Backend used for computations (e.g., NumPy or PyTorch).
        """
        return self._bkd
