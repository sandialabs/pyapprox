from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class PiecewiseQuadratic(Generic[Array]):
    """
    Class for Piecewise quadratic basis functions and
    quadrature weights.
    """

    def __init__(self, nodes: Array, bkd: Backend):
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
        nnodes = self._nodes.shape[0]
        if nnodes == 1:
            return self._bkd.ones((xx.shape[0], nnodes))
        vals = self._bkd.zeros((xx.shape[0], nnodes))
        for ii in range(nnodes):
            if ii % 2 == 1:
                xl, xm, xr = self._nodes[ii - 1 : ii + 2]
                II = self._bkd.nonzero((xx >= xl) & (xx <= xr))[0]
                vals[II, ii] = (
                    (xx[II] - xl) / (xm - xl) * (xx[II] - xr) / (xm - xr)
                )
                continue
            if ii < nnodes - 2:
                xl, xm, xr = self._nodes[ii : ii + 3]
                II = self._bkd.nonzero((xx >= xl) & (xx <= xr))[0]
                vals[II, ii] = (
                    (xx[II] - xm) / (xl - xm) * (xx[II] - xr) / (xl - xr)
                )
            if ii > 1:
                xl, xm, xr = self._nodes[ii - 2 : ii + 1]
                II = self._bkd.nonzero((xx >= xl) & (xx <= xr))[0]
                vals[II, ii] = (
                    (xx[II] - xl) / (xr - xl) * (xx[II] - xm) / (xr - xm)
                )
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
                weights[ii] += ((xr - xl) * (2 * xl - 3 * xm + xr)) / (
                    6 * (xl - xm)
                )
            if ii > 1:
                xl, xm, xr = self._nodes[ii - 2 : ii + 1]
                weights[ii] += ((xl - xr) * (xl - 3 * xm + 2 * xr)) / (
                    6 * (xm - xr)
                )
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
