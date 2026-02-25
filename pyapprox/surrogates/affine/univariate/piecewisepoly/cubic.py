from typing import Generic, Tuple

from pyapprox.util.backends.protocols import Array, Backend


class PiecewiseCubic(Generic[Array]):
    """
    Class for computing piecewise cubic basis functions and
    quadrature weights.
    """

    def __init__(self, nodes: Array, bkd: Backend):
        """
        Initialize the PiecewiseCubic object.

        Parameters
        ----------
        nodes : Array
            The nodes (abscissas) where the piecewise cubic basis functions are
            defined.
        bkd : Backend
            Backend used for computations (e.g., NumPy or PyTorch).
        """
        assert nodes.ndim == 1, "Nodes must be a 1D array."
        if nodes.shape[0] < 4 or (nodes.shape[0] - 4) % 3 != 0:
            raise ValueError(
                "Nodes must contain at least 4 elements and satisfy "
                "(nnodes - 4) % 3 == 0."
            )
        self._nodes = nodes
        self._bkd = bkd

    def __call__(self, xx: Array) -> Array:
        """
        Evaluate the piecewise cubic basis functions at given points.

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

        # Elements are groups of 4 nodes: [0,1,2,3], [3,4,5,6], [6,7,8,9], ...
        # Node indices: 0,1,2,3,4,5,6,... where multiples of 3 are boundaries
        # n_elements = (nnodes - 1) // 3
        n_elements = (nnodes - 1) // 3

        # Find element index using nodes at positions 0, 3, 6, ... as boundaries
        boundary_nodes = nodes[::3]  # nodes 0, 3, 6, ...
        elem_idx = bkd.searchsorted(boundary_nodes, xx, side="right") - 1
        elem_idx = bkd.clip(elem_idx, 0, n_elements - 1)

        # Get node indices for each element
        # Element k has nodes at indices 3k, 3k+1, 3k+2, 3k+3
        idx0 = 3 * elem_idx      # 0, 3, 6, ...
        idx1 = idx0 + 1          # 1, 4, 7, ...
        idx2 = idx0 + 2          # 2, 5, 8, ...
        idx3 = idx0 + 3          # 3, 6, 9, ...

        # Get node values
        x1 = nodes[idx0]
        x2 = nodes[idx1]
        x3 = nodes[idx2]
        x4 = nodes[idx3]

        # Compute Lagrange basis values for cubic interpolation
        pt_indices = bkd.arange(npts)

        # Basis at node 0 (idx0): L_0(x) = (x-x2)(x-x3)(x-x4) / (x1-x2)(x1-x3)(x1-x4)
        vals[pt_indices, idx0] = (
            (xx - x2) / (x1 - x2) *
            (xx - x3) / (x1 - x3) *
            (xx - x4) / (x1 - x4)
        )

        # Basis at node 1 (idx1): L_1(x) = (x-x1)(x-x3)(x-x4) / (x2-x1)(x2-x3)(x2-x4)
        vals[pt_indices, idx1] = (
            (xx - x1) / (x2 - x1) *
            (xx - x3) / (x2 - x3) *
            (xx - x4) / (x2 - x4)
        )

        # Basis at node 2 (idx2): L_2(x) = (x-x1)(x-x2)(x-x4) / (x3-x1)(x3-x2)(x3-x4)
        vals[pt_indices, idx2] = (
            (xx - x1) / (x3 - x1) *
            (xx - x2) / (x3 - x2) *
            (xx - x4) / (x3 - x4)
        )

        # Basis at node 3 (idx3): L_3(x) = (x-x1)(x-x2)(x-x3) / (x4-x1)(x4-x2)(x4-x3)
        vals[pt_indices, idx3] = (
            (xx - x1) / (x4 - x1) *
            (xx - x2) / (x4 - x2) *
            (xx - x3) / (x4 - x3)
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
        if nnodes < 4 or (nnodes - 4) % 3 != 0:
            raise ValueError(
                "Nodes must contain at least 4 elements"
                " and satisfy (nnodes - 4) % 3 == 0."
            )
        weights = [0.0 for ii in range(nnodes)]
        for ii in range(nnodes):
            if ii % 3 == 1:
                a, b, c, d = self._nodes[ii - 1 : ii + 3]
                weights[ii] = ((a - d) ** 3 * (a - 2 * c + d)) / (
                    12 * (-a + b) * (b - c) * (b - d)
                )
                continue
            if ii % 3 == 2:
                a, b, c, d = self._nodes[ii - 2 : ii + 2]
                weights[ii] = ((a - d) ** 3 * (a - 2 * b + d)) / (
                    12 * (-a + c) * (-b + c) * (c - d)
                )
                continue
            if ii % 3 == 0 and ii < nnodes - 3:
                a, b, c, d = self._nodes[ii : ii + 4]
                weights[ii] += (
                    (d - a)
                    * (
                        3 * a**2
                        + 6 * b * c
                        - 2 * (b + c) * d
                        + d**2
                        + 2 * a * (-2 * (b + c) + d)
                    )
                ) / (12 * (a - b) * (a - c))
            if ii % 3 == 0 and ii >= 3:
                a, b, c, d = self._nodes[ii - 3 : ii + 1]
                weights[ii] += (
                    (a - d)
                    * (
                        a**2
                        + 6 * b * c
                        - 2 * a * (b + c - d)
                        - 4 * b * d
                        - 4 * c * d
                        + 3 * d**2
                    )
                ) / (12 * (b - d) * (-c + d))
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
        BackendMixin
            Backend used for computations (e.g., NumPy or PyTorch).
        """
        return self._bkd
