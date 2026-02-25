from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class PiecewiseConstantMidpoint(Generic[Array]):
    """
    Class for computing piecewise midpoint constant basis functions and
    quadrature rules.
    """

    def __init__(self, nodes: Array, bkd: Backend):
        """
        Initialize the PiecewiseConstantMidpoint object.

        Parameters
        ----------
        nodes : Array
            The nodes (abscissas) where the piecewise constant basis functions are
            defined.
        bkd : Backend
            Backend used for computations (e.g., NumPy or PyTorch).
        """
        assert nodes.ndim == 1, "Nodes must be a 1D array."
        self._nodes = nodes
        self._bkd = bkd

    def __call__(self, xx: Array) -> Array:
        """
        Evaluate the piecewise midpoint constant basis functions at given
        points.

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
        nnodes = self._nodes.shape[0]
        if nnodes == 1:
            return self._bkd.ones((xx.shape[0], nnodes))
        vals = self._bkd.zeros((xx.shape[0], nnodes - 1))
        for ii in range(nnodes - 1):
            xl = self._nodes[ii]
            xr = self._nodes[ii + 1]
            if ii < nnodes - 1:
                II = self._bkd.nonzero((xx >= xl) & (xx < xr))[0]
            else:
                II = self._bkd.nonzero((xx >= xl) & (xx <= xr))[0]
            vals[II, ii] = self._bkd.ones((II.shape[0],), dtype=float)
        return vals

    def quadrature_rule(self) -> Tuple[Array, Array]:
        """
        Compute quadrature points and weights based on the nodes.

        Returns
        -------
        Tuple[Array, Array]
            Quadrature points and weights.
        """
        quadrature_points, quadrature_weights = self._quadrature_rule_from_nodes(
            self._nodes
        )
        return quadrature_points, quadrature_weights

    def _quadrature_rule_from_nodes(self, nodes: Array) -> Tuple[Array, Array]:
        """
        Compute quadrature points and weights from the nodes.

        Parameters
        ----------
        nodes : Array
            The nodes (abscissas) where the quadrature rule is defined.

        Returns
        -------
        Tuple[Array, Array]
            Quadrature points and weights.
        """
        quadrature_points = (nodes[1:] + nodes[:-1]) / 2  # Midpoints of intervals
        quadrature_weights = self._bkd.diff(nodes)[
            :, None
        ]  # Differences between consecutive nodes
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
