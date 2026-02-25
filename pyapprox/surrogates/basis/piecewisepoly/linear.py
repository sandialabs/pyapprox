from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


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
        nodes = self._nodes
        nnodes = nodes.shape[0]
        if nnodes == 1:
            return self._bkd.ones((xx.shape[0], nnodes))
        vals = self._bkd.zeros((xx.shape[0], nnodes))
        for ii in range(nnodes):
            xm = nodes[ii]
            if ii > 0:
                xl = nodes[ii - 1]
                II = self._bkd.nonzero((xx >= xl) & (xx <= xm))[0]
                vals[II, ii] = (xx[II] - xl) / (xm - xl)
            if ii < nnodes - 1:
                xr = nodes[ii + 1]
                JJ = self._bkd.nonzero((xx >= xm) & (xx <= xr))[0]
                vals[JJ, ii] = (xr - xx[JJ]) / (xr - xm)
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
