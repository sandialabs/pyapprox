from typing import Generic, Tuple

from pyapprox.typing.util.backends.protocols import Array, Backend


class PiecewiseLinear(Generic[Array]):
    """
    Class for computing piecewise linear basis functions and
    quadrature weights.

    Parameters
    ----------
    nodes : Array
        The nodes (abscissas) where the piecewise linear basis functions are
        defined.
    bkd : BackendMixin, optional
        Backend used for computations (e.g., NumPy or PyTorch)
    """

    def __init__(self, nodes: Array, bkd: Backend):
        """
        Initialize the PiecewiseLinear object.

        Parameters
        ----------
        nodes : Array
            The nodes (abscissas) where the piecewise linear basis functions
            are defined.
        bkd : Backend, optional
            Backend used for computations (e.g., NumPy or PyTorch).
        """
        assert nodes.ndim == 1, "Nodes must be a 1D array."
        self._nodes = nodes
        self._bkd = bkd

    def __call__(self, xx: Array) -> Array:
        """
        Evaluate the piecewise linear basis functions at given points.

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

        # Find interval indices for all points at once
        # idx[i] is the index of the rightmost node <= xx[i]
        idx = bkd.searchsorted(nodes, xx, side="right") - 1

        # Clamp to valid range [0, nnodes-2] for interior intervals
        idx = bkd.clip(idx, 0, nnodes - 2)

        # Get left and right nodes for each point's interval
        xl = nodes[idx]
        xr = nodes[idx + 1]

        # Compute normalized position within interval
        t = (xx - xl) / (xr - xl)

        # Create index arrays for scatter
        pt_indices = bkd.arange(npts)

        # Left basis function contribution (value = 1 - t at node idx)
        vals[pt_indices, idx] = 1 - t

        # Right basis function contribution (value = t at node idx + 1)
        vals[pt_indices, idx + 1] = t

        return vals

    def quadrature_rule(self) -> Tuple[Array, Array]:
        """
        Compute quadrature points and weights based on the nodes.

        Returns
        -------
        Tuple[Array, Array]
            Quadrature points and weights.
        """
        quadrature_points = (
            self._nodes
        )  # For non-constant basis, points are the nodes
        quadrature_weights = self._bkd.zeros(
            self._nodes.shape
        )  # Use zeros with shape
        for ii in range(self._nodes.shape[0]):
            if ii > 0:
                quadrature_weights[ii] += 0.5 * (
                    self._nodes[ii] - self._nodes[ii - 1]
                )
            if ii < self._nodes.shape[0] - 1:
                quadrature_weights[ii] += 0.5 * (
                    self._nodes[ii + 1] - self._nodes[ii]
                )
        return quadrature_points, quadrature_weights

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
        BackendMixin
            Backend used for computations (e.g., NumPy or PyTorch).
        """
        return self._bkd
